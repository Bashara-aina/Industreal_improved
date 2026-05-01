import sys, os
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Training Script for Multi-Task IKEA Assembly Model
====================================================
Joint training of detection + pose + activity on 685K frames.

Key features:
  - Mixed precision (FP16) for RTX 3060 12GB
  - Gradient accumulation: batch 15 x accum 4 = effective 60
  - Kendall uncertainty weighting (auto-balances 3 tasks)
  - Class-balanced sampling for 2545:1 activity imbalance
  - Cosine annealing with linear warmup (5 epochs)
  - NaN/Inf skip guard (corrupt JPEG resilience)
  - Early stopping (patience=C.PATIENCE, currently 10; raise to 12 after epoch 50)
  - Checkpoint saving (best + periodic, with NaN guard)
  - JSONL logging for all metrics including Kendall weights
  - Ablation support via config flags (TRAIN_DET/POSE/ACT, USE_KENDALL)

Fix (2026-03-10): Combined validation metric now uses normalized weights when
  PCK is NaN (no visible keypoints).

Fix (2026-03-13a): BrokenPipeError -- set_start_method moved to module top-level.

Fix (2026-03-13b): Bus error / worker crash -- spawn + file_system sharing +
  persistent_workers=False.

Fix (2026-03-14): OOM (ENOMEM) at validation time.
  Pre-validation memory flush: clear COCO cache, zero grads, close SQLite
  connections, double gc.collect(), empty CUDA cache. See _flush_before_val().

Fix (2026-03-14b): IndentationError on line 415 (extra space before best_metric).

Fix (2026-03-15): Validation OOM (VM_FAULT_OOM in dmesg, VS Code crash).
  val_loader now uses C.VAL_BATCH_SIZE=4 and C.VAL_NUM_WORKERS=2
  instead of the training BATCH_SIZE=10 / NUM_WORKERS=4.
  Also reduced prefetch_factor to 1 for val loader to halve prefetch RAM.

Fix (2026-03-16): Resume block reset log_var_pose to 0.0 instead of -1.0.
  Corrected to fill_(-1.0) to preserve asymmetric Kendall initialization
  on early-epoch resumes (e.g. crash at epoch 1).

Usage:
  python train.py
  python train.py --resume runs/ikea_multitask/checkpoints/latest.pth

Author: Bashara
Date: February 2026 | Audited: March 2026
"""

import argparse
import importlib
import gc
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import psutil
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

import config as C
import ikea_dataset as _ikea_ds_module

C = cast(Any, C)
IKEAMultiTaskDataset = cast(Any, getattr(_ikea_ds_module, 'IKEAMultiTaskDataset'))
IKEAMultiTaskSequenceDataset = cast(Any, getattr(_ikea_ds_module, 'IKEAMultiTaskSequenceDataset'))
collate_fn = cast(Any, getattr(_ikea_ds_module, 'collate_fn'))
temporal_sequence_collate_fn = cast(Any, getattr(_ikea_ds_module, 'temporal_sequence_collate_fn'))

_model_module = importlib.import_module('model')
MultiTaskIKEA = cast(Any, getattr(_model_module, 'MultiTaskIKEA'))
count_parameters = cast(Any, getattr(_model_module, 'count_parameters'))

_losses_module = importlib.import_module('losses')
MultiTaskLoss = cast(Any, getattr(_losses_module, 'MultiTaskLoss'))

_evaluate_module = importlib.import_module('evaluate')
evaluate_all = cast(Any, getattr(_evaluate_module, 'evaluate_all'))

from training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)

_W_F1  = 0.40
_W_PCK = 0.35
_W_MAP = 0.25
assert abs(_W_F1 + _W_PCK + _W_MAP - 1.0) < 1e-9

_W_F1_NO_PCK  = _W_F1  / (_W_F1 + _W_MAP)
_W_MAP_NO_PCK = _W_MAP / (_W_F1 + _W_MAP)
assert abs(_W_F1_NO_PCK + _W_MAP_NO_PCK - 1.0) < 1e-9

CFG_TRAIN_DET = bool(getattr(C, 'TRAIN_DET', True))
CFG_TRAIN_POSE = bool(getattr(C, 'TRAIN_POSE', True))
CFG_TRAIN_ACT = bool(getattr(C, 'TRAIN_ACT', True))
CFG_USE_KENDALL = bool(getattr(C, 'USE_KENDALL', True))
CFG_USE_FILM = bool(getattr(C, 'USE_FILM', False))
CFG_USE_TEMPORAL = bool(getattr(C, 'USE_TEMPORAL', False))
CFG_USE_MULTIVIEW_ACTIVITY = bool(getattr(C, 'USE_MULTIVIEW_ACTIVITY', False))
CFG_VAL_NUM_WORKERS = int(getattr(C, 'VAL_NUM_WORKERS', C.NUM_WORKERS))
CFG_VAL_BATCH_SIZE = int(getattr(C, 'VAL_BATCH_SIZE', C.BATCH_SIZE))
CFG_EVAL_MAX_BATCHES = int(getattr(C, 'EVAL_MAX_BATCHES', 0))
CFG_EVAL_SAVE_DIR = Path(getattr(C, 'EVAL_SAVE_DIR', C.OUTPUT_ROOT / 'eval_outputs'))


def _refresh_runtime_cfg() -> None:
    global CFG_TRAIN_DET
    global CFG_TRAIN_POSE
    global CFG_TRAIN_ACT
    global CFG_USE_KENDALL
    global CFG_USE_FILM
    global CFG_USE_TEMPORAL
    global CFG_USE_MULTIVIEW_ACTIVITY
    global CFG_VAL_NUM_WORKERS
    global CFG_VAL_BATCH_SIZE
    global CFG_EVAL_MAX_BATCHES
    global CFG_EVAL_SAVE_DIR

    CFG_TRAIN_DET = bool(getattr(C, 'TRAIN_DET', True))
    CFG_TRAIN_POSE = bool(getattr(C, 'TRAIN_POSE', True))
    CFG_TRAIN_ACT = bool(getattr(C, 'TRAIN_ACT', True))
    CFG_USE_KENDALL = bool(getattr(C, 'USE_KENDALL', True))
    CFG_USE_FILM = bool(getattr(C, 'USE_FILM', False))
    CFG_USE_TEMPORAL = bool(getattr(C, 'USE_TEMPORAL', False))
    CFG_USE_MULTIVIEW_ACTIVITY = bool(getattr(C, 'USE_MULTIVIEW_ACTIVITY', False))
    CFG_VAL_NUM_WORKERS = int(getattr(C, 'VAL_NUM_WORKERS', C.NUM_WORKERS))
    CFG_VAL_BATCH_SIZE = int(getattr(C, 'VAL_BATCH_SIZE', C.BATCH_SIZE))
    CFG_EVAL_MAX_BATCHES = int(getattr(C, 'EVAL_MAX_BATCHES', 0))
    CFG_EVAL_SAVE_DIR = Path(getattr(C, 'EVAL_SAVE_DIR', C.OUTPUT_ROOT / 'eval_outputs'))


def seed_everything(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(getattr(C, 'CUDNN_DETERMINISTIC', False))
    torch.backends.cudnn.benchmark = bool(getattr(C, 'CUDNN_BENCHMARK', True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(C, 'ALLOW_TF32', True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(C, 'ALLOW_TF32', True))
    torch.set_float32_matmul_precision(str(getattr(C, 'MATMUL_PRECISION', 'high')))


def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
        mean = torch.tensor(C.IMAGENET_MEAN, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


def _build_loader(ds: Any, split: str, batch_size: int,
                  num_workers: int, prefetch: int = 1,
                  persistent: Optional[bool] = None) -> DataLoader:
    is_train = split == 'train'
    sampler  = ds.get_sampler() if is_train else None
    effective_prefetch = prefetch if num_workers > 0 else None
    # persistent_workers only for train loader (created once, reused).
    # Val loader is recreated each validation cycle -- persistent workers
    # would keep old worker processes alive until GC reaps them.
    if persistent is None:
        persistent = is_train and (num_workers > 0)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=C.PIN_MEMORY,
        drop_last=is_train,
        persistent_workers=bool(persistent),
        prefetch_factor=effective_prefetch,
    )


def _choose_num_workers(split: str, requested_workers: int, batch_size: int,
                        prefetch: int = 1) -> int:
    if requested_workers <= 0:
        return 0
    if not getattr(C, 'DATALOADER_AUTO_FALLBACK', True):
        return requested_workers

    shm_path = Path('/dev/shm')
    if not shm_path.exists():
        return requested_workers

    try:
        shm_usage = shutil.disk_usage(shm_path)
        shm_total_bytes = shm_usage.total
        shm_free_bytes = shm_usage.free
    except OSError:
        return requested_workers

    bytes_per_image = 3 * C.IMG_HEIGHT * C.IMG_WIDTH * 4
    est_inflight = (
        max(requested_workers, 1)
        * max(prefetch, 1)
        * batch_size
        * bytes_per_image
        * 2.0
    )

    safety_reserve = 512 * 1024 * 1024
    if shm_free_bytes < (est_inflight + safety_reserve):
        fallback_workers = 0 if shm_free_bytes < (1024 * 1024 * 1024) else 1
        logger.warning(
            f'[{split}] /dev/shm free space appears small '
            f'({shm_free_bytes / (1024**3):.2f} GB free / '
            f'{shm_total_bytes / (1024**3):.2f} GB total) for requested workers '
            f'({requested_workers}). Falling back to num_workers={fallback_workers} '
            f'to avoid DataLoader ENOMEM.'
        )
        return fallback_workers

    return requested_workers


def _flush_before_val(optimizer, val_ds: Any, epoch: int):
    """
    Aggressively free CPU RAM before starting validation.
    """
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e9

    coco_cache = getattr(_ikea_ds_module, '_PROC_COCO_CACHE', None)
    coco_lock = getattr(_ikea_ds_module, '_PROC_COCO_LOCK', None)
    if isinstance(coco_cache, dict):
        if coco_lock is None:
            coco_cache.clear()
        else:
            with coco_lock:
                coco_cache.clear()

    db_conn = getattr(val_ds, '_db_conn', None)
    if db_conn is not None:
        db_conn.close()
        setattr(val_ds, '_db_conn', None)

    optimizer.zero_grad(set_to_none=True)

    gc.collect()
    gc.collect()

    torch.cuda.empty_cache()

    rss_after = proc.memory_info().rss / 1e9
    freed_mb  = (rss_before - rss_after) * 1024
    logger.info(
        f'  [pre-val flush] RSS: {rss_before:.2f}GB -> {rss_after:.2f}GB '
        f'(freed ~{freed_mb:.0f} MB)'
    )


def train_one_epoch(
    model, criterion, loader, optimizer, scaler, device, epoch,
    accum_steps: int = C.GRAD_ACCUM_STEPS,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running = {
        'total': 0.0, 'det': 0.0, 'det_cls': 0.0,
        'det_reg': 0.0, 'pose': 0.0, 'activity': 0.0,
        'w_det': 0.0, 'w_pose': 0.0, 'w_act': 0.0,
        'log_var_det': 0.0, 'log_var_pose': 0.0, 'log_var_act': 0.0,
    }
    num_batches = 0
    nan_skips   = 0
    total_steps = 0
    t_start     = time.time()

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=True, dynamic_ncols=True)

    for step, (images, targets) in enumerate(pbar):
        total_steps = step + 1
        images = _prepare_images(images, device)

        for i in range(len(targets['detection'])):
            targets['detection'][i]['boxes']  = targets['detection'][i]['boxes'].to(device)
            targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to(device)
        targets['keypoints']      = targets['keypoints'].to(device)
        targets['visibility']     = targets['visibility'].to(device)
        targets['kpt_confidence'] = targets['kpt_confidence'].to(device)
        targets['activity']       = targets['activity'].to(device)

        with amp.autocast(enabled=C.MIXED_PRECISION):
            outputs = model(images)
        for _k in ('cls_preds', 'reg_preds', 'keypoints', 'act_logits'):
            if _k in outputs and isinstance(outputs[_k], torch.Tensor):
                outputs[_k] = outputs[_k].float()

        loss, loss_dict = criterion(outputs, targets)
        loss = loss / accum_steps

        if not torch.isfinite(loss):
            nan_skips += 1
            det_val = float(loss_dict.get('det', float('nan')))
            pose_val = float(loss_dict.get('pose', float('nan')))
            act_val = float(loss_dict.get('activity', float('nan')))
            if nan_skips <= 10:
                logger.warning(
                    f'  NaN/Inf loss at epoch {epoch} step {step + 1} '
                    f'(skip #{nan_skips}) -- '
                    f'det={det_val:.4f} pose={pose_val:.4f} act={act_val:.4f}; '
                    f'zeroing grads and continuing'
                )
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, loss_dict
            torch.cuda.empty_cache()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                C.GRAD_CLIP_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k]
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.3f}",
            'det':  f"{loss_dict['det']:.3f}",
            'pose': f"{loss_dict['pose']:.3f}",
            'act':  f"{loss_dict['activity']:.3f}",
        }, refresh=False)

    if nan_skips > 0:
        _total_steps  = total_steps
        _nan_fraction = nan_skips / max(_total_steps, 1)
        logger.warning(f'  Epoch {epoch}: skipped {nan_skips} NaN/Inf batches total')
        if _nan_fraction > 0.10:
            logger.error(
                f'  Epoch {epoch}: {nan_skips}/{_total_steps} NaN batches '
                f'({_nan_fraction:.1%}) exceeds 10% -- gradient signal unreliable'
            )

    avg = {k: v / max(num_batches, 1) for k, v in running.items()}
    avg['epoch_time'] = time.time() - t_start
    avg['nan_skips']  = nan_skips
    return avg


def train_one_epoch_temporal(
    model, criterion, loader, optimizer, scaler, device, epoch,
    accum_steps: int = C.GRAD_ACCUM_STEPS,
):
    """
    Temporal sequence training — processes T-frame video sequences.

    Uses forward_sequence() for temporal action localization + ordering.
    Loss = per-frame activity loss + temporal ordering loss.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running = {
        'total': 0.0, 'activity_seq': 0.0,
        'temporal_ordering': 0.0, 'tma_kl': 0.0,
    }
    num_batches = 0
    nan_skips = 0
    total_steps = 0
    t_start = time.time()

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Temporal]', leave=True, dynamic_ncols=True)

    for step, (images_seq, targets) in enumerate(pbar):
        total_steps = step + 1
        B, T = images_seq.shape[:2]
        images_seq = images_seq.to(device, non_blocking=True)

        if images_seq.dtype == torch.uint8:
            images_seq = images_seq.float().div_(255.0)
            mean = torch.tensor(C.IMAGENET_MEAN, device=device, dtype=images_seq.dtype).view(1, 1, 3, 1, 1)
            std = torch.tensor(C.IMAGENET_STD, device=device, dtype=images_seq.dtype).view(1, 1, 3, 1, 1)
            images_seq = (images_seq - mean) / std

        with amp.autocast(enabled=C.MIXED_PRECISION):
            outputs_seq = model.forward_sequence(images_seq)

        temporal_al = outputs_seq['temporal_al']
        act_logits_seq = outputs_seq['act_logits_seq']

        act_logits_flat = act_logits_seq.reshape(B * T, -1)
        activity_labels = targets['action_labels_seq'].to(device).reshape(B * T)
        activity_loss = F.cross_entropy(
            act_logits_flat, activity_labels,
            reduction='mean',
            label_smoothing=getattr(C, 'LABEL_SMOOTHING', 0.0),
        )

        ordering_loss = 0.0
        if hasattr(model, 'temporal_ordering_head') and model.training:
            ordering_result = outputs_seq.get('temporal_ordering')
            if ordering_result is not None:
                pair_scores = ordering_result['pair_scores']
                pair_labels = ordering_result['pair_labels']
                ordering_loss = F.binary_cross_entropy(
                    pair_scores, pair_labels, reduction='mean'
                )
            else:
                ordering_loss = torch.tensor(0.0, device=device)

        tma_kl_loss = 0.0
        if temporal_al.get('tma_logvar') is not None:
            tma_logvar = temporal_al['tma_logvar']
            tma_mean = temporal_al.get('temporal_features', torch.zeros_like(tma_logvar))
            tma_kl_loss = 0.5 * (-1 - tma_logvar + tma_mean.pow(2) + tma_logvar.exp()).mean()

        loss = activity_loss + 0.1 * ordering_loss + 0.01 * tma_kl_loss
        loss = loss / accum_steps

        if not torch.isfinite(loss):
            nan_skips += 1
            if nan_skips <= 10:
                logger.warning(
                    f'  [Temporal] NaN/Inf loss at epoch {epoch} step {step + 1} '
                    f'(skip #{nan_skips}) -- zeroing grads and continuing'
                )
            optimizer.zero_grad(set_to_none=True)
            del outputs_seq, loss, activity_loss
            torch.cuda.empty_cache()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                C.GRAD_CLIP_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running['total'] += loss.item() * accum_steps
        running['activity_seq'] += activity_loss.item()
        running['temporal_ordering'] += ordering_loss.item() if isinstance(ordering_loss, torch.Tensor) else ordering_loss
        running['tma_kl'] += tma_kl_loss.item() if isinstance(tma_kl_loss, torch.Tensor) else tma_kl_loss
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item() * accum_steps:.3f}",
            'act_seq': f"{activity_loss.item():.3f}",
            'tma_kl': f"{tma_kl_loss.item() if isinstance(tma_kl_loss, torch.Tensor) else tma_kl_loss:.3f}",
        }, refresh=False)

    if nan_skips > 0:
        logger.warning(f'  [Temporal] Epoch {epoch}: skipped {nan_skips} NaN/Inf batches')

    avg = {k: v / max(num_batches, 1) for k, v in running.items()}
    avg['epoch_time'] = time.time() - t_start
    avg['nan_skips'] = nan_skips
    return avg


def _has_nan(metrics: Dict) -> bool:
    for v in metrics.values():
        if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
            return True
        if isinstance(v, dict) and _has_nan(v):
            return True
        if isinstance(v, list):
            for item in v:
                if isinstance(item, (float, np.floating)) and (
                    math.isnan(item) or math.isinf(item)
                ):
                    return True
    return False


def _load_model_compat(model, state_dict):
    model_state = model.state_dict()
    compatible, skipped = {}, []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append((
                k,
                v.shape if hasattr(v, 'shape') else '?',
                model_state[k].shape if k in model_state else 'NOT IN MODEL',
            ))
    result = model.load_state_dict(compatible, strict=False)

    # Log loading summary for Bug 5 diagnostics
    logger.info(
        f'  Checkpoint load: {len(compatible)}/{len(model_state)} tensors loaded'
    )
    if result.missing_keys:
        # Categorize missing keys by component for clearer diagnostics
        missing_by_component = {}
        for key in result.missing_keys:
            component = key.split('.')[0]
            missing_by_component.setdefault(component, []).append(key)
        for comp, keys in sorted(missing_by_component.items()):
            logger.info(f'  MISSING ({comp}): {keys}')

    return result, skipped


def _check_ram(label: str = '', warn_gb: float = 50.0) -> float:
    proc     = psutil.Process()
    rss_gb   = proc.memory_info().rss / 1e9
    children = proc.children(recursive=True)
    total_gb = rss_gb + sum(c.memory_info().rss / 1e9 for c in children)
    if total_gb > warn_gb:
        logger.warning(
            f'[MEM] {label} RSS={rss_gb:.1f}GB, '
            f'total(+workers)={total_gb:.1f}GB -- approaching {warn_gb}GB limit!'
        )
    return total_gb


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    # std::bad_alloc is raised by the CUDA allocator on some PyTorch builds
    # instead of the usual "CUDA out of memory" message.
    if 'std::bad_alloc' in text or 'bad_alloc' in text:
        return True
    return 'out of memory' in text and ('cuda' in text or 'cublas' in text)


def _is_dataloader_shm_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    patterns = (
        'unable to allocate shared memory',
        'shared memory(shm)',
        'dataloader worker',
        'bus error',
    )
    return any(pattern in text for pattern in patterns)


def _compute_combined_metric(
    f1: float, pck: float, map50: float
) -> tuple[float, bool]:
    pck_nan = math.isnan(pck) or math.isinf(pck)
    if pck_nan:
        combined = f1 * _W_F1_NO_PCK + map50 * _W_MAP_NO_PCK
    else:
        combined = f1 * _W_F1 + pck * _W_PCK + map50 * _W_MAP
    return combined, pck_nan


def _apply_runtime_safety(device: torch.device) -> None:
    nice_value = int(getattr(C, 'TRAIN_NICE', 0))
    if nice_value > 0:
        try:
            os.nice(nice_value)
            logger.info(f'Process nice level increased by +{nice_value} for UI responsiveness.')
        except OSError as exc:
            logger.warning(f'Could not set process nice value (+{nice_value}): {exc}')

    thread_cap = int(getattr(C, 'TORCH_NUM_THREADS', 0))
    if thread_cap > 0:
        try:
            torch.set_num_threads(thread_cap)
            torch.set_num_interop_threads(max(1, min(4, thread_cap)))
            logger.info(
                f'Torch CPU threads capped: intraop={torch.get_num_threads()} '
                f'interop={torch.get_num_interop_threads()}'
            )
        except RuntimeError as exc:
            logger.warning(f'Could not update torch thread limits: {exc}')

    if device.type == 'cuda':
        mem_fraction = float(getattr(C, 'CUDA_MEMORY_FRACTION', 1.0))
        if 0.0 < mem_fraction < 1.0:
            try:
                device_index = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(mem_fraction, device=device_index)
                logger.info(
                    f'CUDA memory fraction cap enabled: {mem_fraction:.2f} '
                    f'(reserves ~{(1.0 - mem_fraction) * 100:.0f}% VRAM for desktop/apps).'
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                logger.warning(f'Could not set CUDA memory fraction cap: {exc}')


def main(args):
    seed_everything(C.SEED)

    log_dir  = C.LOG_DIR;  log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = C.CHECKPOINT_DIR; ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = CFG_EVAL_SAVE_DIR;  eval_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler(),
        ],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Config file: {Path(C.__file__).resolve()}')
    if torch.cuda.is_available():
        logger.info(f'GPU : {torch.cuda.get_device_name()}')
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f'VRAM: {vram:.1f} GB')

    _apply_runtime_safety(device)

    logger.info(
        f'Ablation: TRAIN_DET={CFG_TRAIN_DET}  TRAIN_POSE={CFG_TRAIN_POSE}  '
        f'TRAIN_ACT={CFG_TRAIN_ACT}  USE_KENDALL={CFG_USE_KENDALL}  USE_FILM={CFG_USE_FILM}'
    )

    logger.info('Building datasets ...')
    train_ds = IKEAMultiTaskDataset(
        split='train', img_size=C.IMG_SIZE, augment=True, seed=C.SEED
    )
    val_ds = IKEAMultiTaskDataset(
        split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED
    )

    train_prefetch = 2 if C.NUM_WORKERS > 0 else 1
    val_prefetch = int(getattr(C, 'VAL_PREFETCH_FACTOR', 1))
    train_workers = _choose_num_workers(
        'train', C.NUM_WORKERS, C.BATCH_SIZE, prefetch=train_prefetch
    )
    val_workers = _choose_num_workers(
        'val', CFG_VAL_NUM_WORKERS, CFG_VAL_BATCH_SIZE, prefetch=val_prefetch
    )

    train_batch_size = C.BATCH_SIZE
    train_accum_steps = C.GRAD_ACCUM_STEPS

    train_loader = _build_loader(
        train_ds, 'train', train_batch_size, train_workers, prefetch=train_prefetch
    )

    temporal_loader = None
    if CFG_USE_TEMPORAL:
        logger.info('Building temporal sequence dataset and loader ...')
        temporal_seq_ds = IKEAMultiTaskSequenceDataset(
            base_dataset=train_ds,
            sequence_len=getattr(C, 'TEMPORAL_SEQUENCE_LEN', 16),
            stride=getattr(C, 'TEMPORAL_STRIDE', 1),
            target_camera='dev3',
        )
        temporal_batch_size = getattr(C, 'TEMPORAL_BATCH_SIZE', 4)
        temporal_workers = min(train_workers, 2)
        temporal_loader = DataLoader(
            temporal_seq_ds,
            batch_size=temporal_batch_size,
            shuffle=True,
            num_workers=temporal_workers,
            collate_fn=temporal_sequence_collate_fn,
            pin_memory=C.PIN_MEMORY,
            drop_last=True,
            persistent_workers=False,
        )
        logger.info(
            f'Temporal loader: batch={temporal_batch_size} workers={temporal_workers} '
            f'sequences={len(temporal_seq_ds):,}'
        )

    class_counts = train_ds.class_counts
    logger.info(f'Training samples  : {len(train_ds):,}')
    logger.info(f'Validation samples: {len(val_ds):,}')
    logger.info(
        f'Train loader: batch={train_batch_size} workers={train_workers} prefetch={train_prefetch}'
    )
    logger.info(
        f'Val   loader: batch={CFG_VAL_BATCH_SIZE} workers={val_workers} prefetch={val_prefetch}'
    )
    _check_ram('after_datasets')

    logger.info('Building model ...')
    model  = MultiTaskIKEA(
        pretrained=True,
        use_film=CFG_USE_FILM,
        use_temporal=CFG_USE_TEMPORAL,
        use_multiview=CFG_USE_MULTIVIEW_ACTIVITY,
    ).to(device)
    params = count_parameters(model)
    logger.info(f'Total parameters  : {params["total_all"]:,}')
    logger.info(f'Trainable params  : {params["total_trainable"]:,}')
    for k, v in params.items():
        if not k.startswith('total'):
            logger.info(f'  {k:15s}: {v:>10,}')

    criterion = MultiTaskLoss(
        num_classes=C.NUM_ACT_CLASSES,
        train_det=CFG_TRAIN_DET,
        train_pose=CFG_TRAIN_POSE,
        train_act=CFG_TRAIN_ACT,
        use_kendall=CFG_USE_KENDALL,
    ).to(device)
    criterion.set_class_counts(class_counts)

    # ── Training Monitor (TensorBoard + matplotlib visualizations) ──────────
    monitor = TrainingMonitor(
        run_dir=ckpt_dir.parent,
        log_interval=10,
        save_viz_epochs=5,
        num_viz_samples=8,
        activity_class_names=C.ACT_CLASS_NAMES,
        enabled=True,
    )
    logger.info(f'[Monitor] Logging to {ckpt_dir.parent}')

    backbone_params, head_params = [], []
    loss_params = list(criterion.parameters())

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(ln in name for ln in
               ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': C.BASE_LR * 0.1},
        {'params': head_params,     'lr': C.BASE_LR},
    ]
    if loss_params:
        param_groups.append({'params': loss_params, 'lr': C.BASE_LR})
    optimizer = AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=C.WARMUP_EPOCHS)
    if C.USE_COSINE_ANNEALING:
        cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=C.T_0, T_mult=C.T_mult, eta_min=1e-6
        )
    else:
        cosine = CosineAnnealingLR(
            optimizer, T_max=C.EPOCHS - C.WARMUP_EPOCHS, eta_min=1e-6
        )
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                              milestones=[C.WARMUP_EPOCHS])

    scaler = amp.GradScaler(enabled=C.MIXED_PRECISION)

    start_epoch      = 0
    best_metric      = 0.0
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)

        load_result, skipped_keys = _load_model_compat(model, ckpt['model'])
        if skipped_keys:
            logger.warning(
                f'  Skipped {len(skipped_keys)} checkpoint key(s) (shape mismatch):'
            )
            for k, cs, ms in skipped_keys:
                logger.warning(
                    f'    {k}: ckpt={cs}  model={ms} -> re-initialized'
                )
        else:
            logger.info('  All checkpoint keys loaded (no shape mismatches).')
        if load_result.missing_keys:
            logger.info(
                f'  Missing keys (new in model, using init): '
                f'{load_result.missing_keys}'
            )

        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info('  Optimizer state restored.')
        except ValueError as e:
            logger.warning(
                f'  Could not restore optimizer state ({e}). '
                f'Re-initialized -- LR schedule continues from epoch '
                f'{ckpt["epoch"] + 1}.'
            )
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch      = ckpt['epoch'] + 1
        best_metric      = ckpt.get('best_metric', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        logger.info(f'Resumed from epoch {start_epoch}, best={best_metric:.4f}')

        # Reset Kendall log_var params only for early-epoch resumes.
        # After warmup, the learned log_var values are meaningful and
        # resetting them causes a loss spike and wasted epochs re-learning.
        if start_epoch < C.WARMUP_EPOCHS:
            with torch.no_grad():
                criterion.log_var_det.fill_(0.0)
                criterion.log_var_pose.fill_(-1.0)
                criterion.log_var_act.fill_(0.0)
            logger.info(
                '  Reset Kendall log_var params (early epoch resume): '
                'det=0.0  pose=-1.0  act=0.0'
            )
        else:
            logger.info(
                f'  Keeping learned Kendall log_var params '
                f'(epoch {start_epoch} >= warmup={C.WARMUP_EPOCHS}): '
                f'det={criterion.log_var_det.item():.3f}  '
                f'pose={criterion.log_var_pose.item():.3f}  '
                f'act={criterion.log_var_act.item():.3f}'
            )

        _ckpt_has_bn = any(
            'pose_head.deconv' in k and 'running_mean' in k
            for k in ckpt['model']
        )
        if _ckpt_has_bn:
            logger.warning(
                '  Old-arch checkpoint (BN in pose_head.deconv) -- '
                're-initializing pose head with Kaiming init'
            )
            best_metric      = 0.0
            patience_counter = 0
            logger.info('  best_metric and patience_counter reset (arch change).')
            optimizer.param_groups[0]['lr'] = C.BASE_LR * 0.1
            for pg in optimizer.param_groups[1:]:
                pg['lr'] = C.BASE_LR
            _remaining = max(C.EPOCHS - start_epoch, 1)
            scheduler  = CosineAnnealingLR(
                optimizer, T_max=_remaining, eta_min=1e-6
            )
            logger.info(f'  Scheduler reset: cosine over {_remaining} epochs.')
            model.pose_head._init_weights()
            logger.info('  Pose head re-initialized (BN->GN arch upgrade).')
        else:
            logger.info(
                '  Checkpoint uses new arch (GroupNorm) -- weights loaded as-is.'
            )

    log_file = open(log_dir / 'metrics.jsonl', 'a')

    logger.info('=' * 60)
    logger.info('Starting training')
    logger.info(f'  Epochs          : {C.EPOCHS}')
    logger.info(
        f'  Batch size      : {C.BATCH_SIZE} x '
        f'{C.GRAD_ACCUM_STEPS} = {C.EFFECTIVE_BATCH}'
    )
    logger.info(
        f'  Learning rate   : backbone={C.BASE_LR * 0.1:.1e}, '
        f'heads={C.BASE_LR:.1e}'
    )
    logger.info(f'  Grad clip norm  : {C.GRAD_CLIP_NORM}')
    logger.info(f'  Mixed precision : {C.MIXED_PRECISION}')
    logger.info(f'  Early stopping  : patience={C.PATIENCE}')
    logger.info(
        f'  Metric weights  : F1={_W_F1}  PCK={_W_PCK}  mAP={_W_MAP}  '
        f'(no-PCK: F1={_W_F1_NO_PCK:.4f}  mAP={_W_MAP_NO_PCK:.4f})'
    )
    logger.info('=' * 60)

    try:
        for epoch in range(start_epoch, C.EPOCHS):
            logger.info(f'\n--- Epoch {epoch}/{C.EPOCHS - 1} ---')
            criterion.set_epoch(epoch)

            train_attempt = 0
            while True:
                train_attempt += 1
                if train_attempt > 6:
                    raise RuntimeError(
                        'Exceeded maximum train retry attempts (6) for this epoch.'
                    )

                try:
                    train_metrics = train_one_epoch(
                        model,
                        criterion,
                        train_loader,
                        optimizer,
                        scaler,
                        device,
                        epoch,
                        accum_steps=train_accum_steps,
                    )
                    break
                except Exception as exc:
                    msg = str(exc)
                    is_loader_enomem = (
                        ('Cannot allocate memory' in msg or _is_dataloader_shm_error(exc))
                        and getattr(train_loader, 'num_workers', 0) > 0
                    )

                    if is_loader_enomem:
                        logger.exception(
                            'DataLoader worker ENOMEM/SHM error detected. '
                            'Rebuilding train loader with num_workers=0 and retrying epoch.'
                        )
                        train_workers = 0
                        train_prefetch = 1
                        train_loader = _build_loader(
                            train_ds,
                            'train',
                            train_batch_size,
                            train_workers,
                            prefetch=train_prefetch,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    if _is_cuda_oom(exc):
                        if train_batch_size <= 1:
                            logger.exception(
                                'CUDA OOM occurred even with train batch_size=1. '
                                'Cannot auto-reduce further.'
                            )
                            raise

                        new_batch_size = max(1, train_batch_size // 2)
                        new_accum_steps = max(
                            1,
                            int(math.ceil(C.EFFECTIVE_BATCH / new_batch_size)),
                        )

                        logger.exception(
                            'CUDA OOM during training. Retrying epoch with reduced '
                            f'batch size ({train_batch_size} -> {new_batch_size}) '
                            f'and adjusted grad accumulation '
                            f'({train_accum_steps} -> {new_accum_steps}).'
                        )

                        train_batch_size = new_batch_size
                        train_accum_steps = new_accum_steps
                        optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()

                        train_loader = _build_loader(
                            train_ds,
                            'train',
                            train_batch_size,
                            train_workers,
                            prefetch=(2 if train_workers > 0 else 1),
                        )
                        logger.info(
                            f'Rebuilt train loader: batch={train_batch_size} '
                            f'workers={train_workers} '
                            f'prefetch={(2 if train_workers > 0 else 1)}'
                        )
                        continue

                    raise
            scheduler.step()
            _check_ram(f'epoch_{epoch}_train')

            current_lr = optimizer.param_groups[1]['lr']
            logger.info(
                f'Train: loss={train_metrics["total"]:.4f}  '
                f'det={train_metrics["det"]:.4f}  '
                f'pose={train_metrics["pose"]:.4f}  '
                f'act={train_metrics["activity"]:.4f}  '
                f'lr={current_lr:.2e}  '
                f'time={train_metrics["epoch_time"]:.0f}s'
                + (
                    f'  nan_skips={train_metrics["nan_skips"]}'
                    if train_metrics['nan_skips'] > 0 else ''
                )
            )

            if CFG_USE_TEMPORAL and temporal_loader is not None:
                try:
                    temporal_metrics = train_one_epoch_temporal(
                        model,
                        criterion,
                        temporal_loader,
                        optimizer,
                        scaler,
                        device,
                        epoch,
                        accum_steps=max(1, train_accum_steps // 4),
                    )
                    logger.info(
                        f'Temporal: loss={temporal_metrics["total"]:.4f}  '
                        f'act_seq={temporal_metrics["activity_seq"]:.4f}  '
                        f'ordering={temporal_metrics["temporal_ordering"]:.4f}  '
                        f'tma_kl={temporal_metrics["tma_kl"]:.4f}'
                    )
                except Exception as exc:
                    logger.warning(f'Temporal training failed: {exc}')

            val_metrics = {}
            if (epoch + 1) % C.VAL_EVERY == 0:
                logger.info('Running validation ...')
                _flush_before_val(optimizer, val_ds, epoch)
                val_batch_size = CFG_VAL_BATCH_SIZE
                val_workers_rt = val_workers
                val_prefetch_rt = val_prefetch
                val_max_batches = CFG_EVAL_MAX_BATCHES

                val_attempt = 0
                while True:
                    val_attempt += 1
                    if val_attempt > 4:
                        raise RuntimeError(
                            'Exceeded maximum validation retry attempts (4).'
                        )

                    val_loader = _build_loader(
                        val_ds,
                        'val',
                        val_batch_size,
                        val_workers_rt,
                        prefetch=val_prefetch_rt,
                    )

                    val_temporal_loader = None
                    if CFG_USE_TEMPORAL:
                        val_seq_ds = IKEAMultiTaskSequenceDataset(
                            base_dataset=val_ds,
                            sequence_len=getattr(C, 'TEMPORAL_SEQUENCE_LEN', 16),
                            stride=1,
                            target_camera='dev3',
                        )
                        val_temporal_batch_size = max(2, val_batch_size // 2)
                        val_temporal_loader = DataLoader(
                            val_seq_ds,
                            batch_size=val_temporal_batch_size,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=temporal_sequence_collate_fn,
                            pin_memory=C.PIN_MEMORY,
                            drop_last=False,
                            persistent_workers=False,
                        )
                        logger.info(
                            f'[val] Temporal sequences: {len(val_seq_ds):,}'
                        )

                    try:
                        val_metrics = evaluate_all(
                            model,
                            criterion,
                            val_loader,
                            device,
                            max_batches=val_max_batches,
                            save_dir=(
                                str(eval_dir)
                                if getattr(C, 'SAVE_VAL_CONFUSION_MATRIX', False)
                                else None
                            ),
                            temporal_loader=val_temporal_loader,
                        )
                        break
                    except Exception as exc:
                        is_cpu_enomem = 'Cannot allocate memory' in str(exc)
                        is_cuda_oom = _is_cuda_oom(exc)
                        if not (is_cpu_enomem or is_cuda_oom):
                            raise

                        if is_cuda_oom:
                            logger.exception(
                                'Validation CUDA OOM detected. Reducing val load and retrying.'
                            )
                        else:
                            logger.exception(
                                'Validation ENOMEM detected. Reducing val load and retrying.'
                            )

                        val_batch_size = max(1, val_batch_size // 2)
                        val_workers_rt = 0
                        val_prefetch_rt = 1
                        val_max_batches = max(1, int(val_max_batches) // 2)
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info(
                            f'Validation retry settings: batch={val_batch_size} '
                            f'workers={val_workers_rt} prefetch={val_prefetch_rt} '
                            f'max_batches={val_max_batches}'
                        )
                        continue
                    finally:
                        del val_loader
                        gc.collect()
                        torch.cuda.empty_cache()

                logger.info(
                    f'Val: loss={val_metrics.get("loss", 0):.4f}  '
                    f'act_acc={val_metrics.get("act_accuracy", 0):.4f}  '
                    f'pck05={val_metrics.get("pck_at_005", 0):.4f}  '
                    f'pck10px={val_metrics.get("pck_at_10px", val_metrics.get("pck_at_01", 0)):.4f}  '
                    f'mAP50={val_metrics.get("det_mAP50", 0):.4f}  '
                    f'tmp_mAP50={val_metrics.get("temporal_mAP50", 0):.4f}  '
                    f'fps={val_metrics.get("fps", 0):.1f}  '
                    f'kTau={val_metrics.get("temporal_order_kendall_tau", 0):.4f}'
                )

                # ── Log to TrainingMonitor ─────────────────────────────────────
                temporal_results = {
                    'temporal_order_kendall_tau': val_metrics.get('temporal_order_kendall_tau', 0.0),
                    'temporal_mAP50':              val_metrics.get('temporal_mAP50', 0.0),
                }
                monitor.log_val_epoch(
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    temporal_results=temporal_results,
                    phase_results=None,
                    efficiency_results={
                        'fps':      val_metrics.get('fps', 0),
                        'gflops':   val_metrics.get('gflops', 0),
                        'params_M': val_metrics.get('params_M', 0),
                        'gpu_mem_gb': val_metrics.get('gpu_mem_gb', 0),
                    },
                    lr=current_lr,
                    confusion_matrix=val_metrics.get('act_confusion_matrix'),
                )

                # Save visualizations every save_viz_epochs (default 5)
                if (epoch + 1) % monitor.save_viz_epochs == 0:
                    # Grab a batch of images for annotated viz
                    try:
                        for batch_imgs, batch_targets in val_loader:
                            B = min(8, batch_imgs.shape[0])
                            flat_imgs = batch_imgs[:B].reshape(B, 3, 480, 640).to(device)
                            with torch.no_grad():
                                model.eval()
                                sample_outputs = model(flat_imgs)
                            monitor.cache_visualization_batch(
                                batch_imgs[:B], batch_targets,
                                {k: v[:B] if isinstance(v, torch.Tensor) else v
                                 for k, v in sample_outputs.items()}
                            )
                            break
                    except Exception:
                        pass  # non-fatal if viz batch fails
                    monitor.save_epoch_visualizations(epoch)

                _task_keys = ('act_macro_f1_present', 'det_mAP50')
                _task_nan  = any(
                    math.isnan(val_metrics.get(k, float('nan')))
                    or math.isinf(val_metrics.get(k, float('nan')))
                    for k in _task_keys
                )
                if _task_nan:
                    logger.warning(
                        '  Core task metrics contain NaN -- '
                        'skipping checkpoint and patience update'
                    )
                else:
                    # Use PCK@10px (fixed pixel) as primary pose metric -- matches IKEA ASM paper
                    _pck = val_metrics.get('pck_at_10px', val_metrics.get('pck_at_01', float('nan')))
                    _f1   = val_metrics.get('act_macro_f1_present', 0.0)
                    _mAP  = val_metrics.get('det_mAP50', 0.0)

                    combined, pck_nan = _compute_combined_metric(_f1, _pck, _mAP)

                    if pck_nan:
                        logger.warning(
                            f'  PCK is NaN -- excluded from combined metric.  '
                            f'Using renormalized weights: '
                            f'F1={_W_F1_NO_PCK:.4f}  mAP={_W_MAP_NO_PCK:.4f}'
                        )

                    logger.info(
                        f'  combined={combined:.4f}  '
                        f'(best={best_metric:.4f}  '
                        f'patience={patience_counter}/{C.PATIENCE})'
                    )

                    if combined > best_metric:
                        best_metric      = combined
                        patience_counter = 0
                        torch.save({
                            'epoch':            epoch,
                            'model':            model.state_dict(),
                            'optimizer':        optimizer.state_dict(),
                            'scheduler':        scheduler.state_dict(),
                            'scaler':           scaler.state_dict(),
                            'best_metric':      best_metric,
                            'patience_counter': patience_counter,
                            'val_metrics':      val_metrics,
                        }, ckpt_dir / 'best.pth')
                        logger.info(f'  ** New best model (combined={combined:.4f}) **')
                    else:
                        patience_counter += 1
                        logger.info(
                            f'  No improvement '
                            f'({patience_counter}/{C.PATIENCE})'
                        )

                    _patience = 12 if epoch >= 50 else C.PATIENCE
                    if patience_counter >= _patience:
                        logger.info(
                            f'Early stopping at epoch {epoch} '
                            f'(patience={_patience})'
                        )
                        break

            torch.save({
                'epoch':            epoch,
                'model':            model.state_dict(),
                'optimizer':        optimizer.state_dict(),
                'scheduler':        scheduler.state_dict(),
                'scaler':           scaler.state_dict(),
                'best_metric':      best_metric,
                'patience_counter': patience_counter,
            }, ckpt_dir / 'latest.pth')

            record = {'epoch': epoch, 'lr': current_lr, 'train': train_metrics}
            if val_metrics:
                record['val'] = val_metrics
                record['val_pose_summary'] = {
                    'pck_at_10px': val_metrics.get('pck_at_10px', None),
                    'pck_at_005': val_metrics.get('pck_at_005', None),
                    'pck_at_01': val_metrics.get('pck_at_01', None),
                    'pck_at_02': val_metrics.get('pck_at_02', None),
                }
            log_file.write(json.dumps(record, default=str) + '\n')
            log_file.flush()
    finally:
        log_file.close()
        monitor.close()

    logger.info('Training complete.')
    logger.info(f'Best combined metric: {best_metric:.4f}')
    logger.info(f'Checkpoints: {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Multi-Task IKEA Model (Unified Multi-Camera)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset (improved4 = manual_pseudo + no FiLM)
  python train.py --preset improved4
  
  # Custom configuration
  python train.py --dataset manual_only --film --detection all_cameras
  
  # Resume training
  python train.py --checkpoint path/to/checkpoint.pth --max-epochs 200
  
  # Debug mode (quick validation)
  python train.py --preset improved3 --debug --max-epochs 2
        """
    )
    
    # Preset shortcuts
    parser.add_argument(
        '--preset',
        type=str,
        choices=[
            'benchmark_full', 'benchmark_quick', 'benchmark_vit_temporal', 'benchmark_multiview',
            'improved3', 'improved4', 'improved3_film', 'improved4_film',
            'improved3_temporal', 'improved4_temporal', 'improved3_multiview',
        ],
        default=None,
        help='Use a preset configuration (overrides other options)'
    )
    
    # Individual config overrides
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['manual_only', 'manual_pseudo'],
        default=None,
        help='Dataset mode: manual_only (1%%) or manual_pseudo (1%% + 99%% pseudo-GT)'
    )
    parser.add_argument(
        '--film',
        action='store_true',
        help='Enable FiLM conditioning (overrides config)'
    )
    parser.add_argument(
        '--detection',
        type=str,
        choices=['all_cameras', 'dev3_only'],
        default=None,
        help='Detection mode: all_cameras (dev1+2+3) or dev3_only (fallback)'
    )
    
    # Training overrides
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Override C.EPOCHS'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override C.BATCH_SIZE'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (alias for --resume)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (small dataset, quick validation)'
    )
    
    args = parser.parse_args()
    
    # Apply preset if specified
    if args.preset:
        C.apply_preset(args.preset)
        logger.info(f'Applied preset: {args.preset}')
    
    # Override with individual config flags
    if args.dataset:
        C.DATASET_MODE = args.dataset
    if args.film:
        C.USE_FILM = True
    if args.detection:
        C.DETECTION_MODE = args.detection
    
    # Override training hyperparameters
    if args.max_epochs is not None:
        C.EPOCHS = args.max_epochs
    if args.batch_size is not None:
        C.BATCH_SIZE = args.batch_size
        C.EFFECTIVE_BATCH = C.BATCH_SIZE * C.GRAD_ACCUM_STEPS
    
    # Debug mode
    if args.debug:
        C.DEBUG_MODE = True
        C.DEBUG_MAX_VIDEOS = 5
        C.VAL_EVERY = 1
        if args.max_epochs is None:
            C.EPOCHS = min(C.EPOCHS, 3)
        logger.info('[train] Debug mode enabled: small dataset, fast validation')
    
    # Update dynamic paths after config changes
    C.update_dynamic_paths()
    
    # Handle checkpoint/resume aliases
    resume_path = args.checkpoint or args.resume
    if resume_path:
        args.resume = resume_path
    
    # Log final configuration
    logger.info(f'[train] Config: DATASET_MODE={C.DATASET_MODE}, '
                f'USE_FILM={C.USE_FILM}, DETECTION_MODE={C.DETECTION_MODE}')
    logger.info(f'[train] Output: {C.OUTPUT_ROOT}')
    
    # Validate and fallback
    C._validate_and_fallback()
    logger.info(f'[train] Detection mode (after validation): {C.DETECTION_MODE}')

    # Refresh cached runtime flags after all config mutations.
    _refresh_runtime_cfg()
    
    main(args)
