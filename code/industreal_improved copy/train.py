import sys, os

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Training Script for Multi-Task IndustReal Model
================================================
Joint training of ASD detection + head pose + activity + PSR on IndustReal dataset.

Key features:
  - Mixed precision (FP16) for RTX 3060 12GB
  - Gradient accumulation: batch 4 x accum 8 = effective 32
  - Kendall uncertainty weighting (auto-balances 4 tasks)
  - Class-balanced sampling for activity imbalance
  - Cosine annealing with linear warmup (5 epochs)
  - NaN/Inf skip guard (corrupt frame resilience)
  - Early stopping (patience=C.PATIENCE)
  - Checkpoint saving (best + periodic, with NaN guard)
  - JSONL logging for all metrics including Kendall weights
  - 4-task combined validation metric: mAP50 + macro-F1

Usage:
  python train.py
  python train.py --resume runs/industreal/checkpoints/latest.pth

Author: Bashara
Date: April 2026
"""

import argparse
import gc
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
import industreal_dataset as _ds_module
import model as _model_module
import model as _popw_model_module
import losses as _losses_module
import evaluate as _evaluate_module

IndustRealMultiTaskDataset = getattr(_ds_module, 'IndustRealMultiTaskDataset')
collate_fn = getattr(_ds_module, 'collate_fn')

MultiTaskIndustReal = getattr(_model_module, 'MultiTaskIndustReal', None)
count_parameters_old = getattr(_model_module, 'count_parameters', None)

POPWMultiTaskModel = getattr(_popw_model_module, 'POPWMultiTaskModel')
count_parameters = getattr(_popw_model_module, 'count_parameters')
set_backbone_stage_requires_grad = getattr(
    _popw_model_module, 'set_backbone_stage_requires_grad'
)

MultiTaskLoss = getattr(_losses_module, 'MultiTaskLoss')

evaluate_all = getattr(_evaluate_module, 'evaluate_all')

logger = logging.getLogger(__name__)

# Combined metric weights for 4-task IndustReal
_W_DET  = 0.30
_W_ACT  = 0.35
_W_POSE = 0.15
_W_PSR  = 0.20

# Config ablation flags (cached at startup)
CFG_TRAIN_DET       = bool(getattr(C, 'TRAIN_DET', True))
CFG_TRAIN_HEAD_POSE = bool(getattr(C, 'TRAIN_HEAD_POSE', True))
CFG_TRAIN_ACT       = bool(getattr(C, 'TRAIN_ACT', True))
CFG_TRAIN_PSR       = bool(getattr(C, 'TRAIN_PSR', True))
CFG_USE_KENDALL     = bool(getattr(C, 'USE_KENDALL', True))
CFG_VAL_NUM_WORKERS = int(getattr(C, 'VAL_NUM_WORKERS', C.NUM_WORKERS))
CFG_VAL_BATCH_SIZE  = int(getattr(C, 'VAL_BATCH_SIZE', C.BATCH_SIZE))
CFG_EVAL_MAX_BATCHES = int(getattr(C, 'EVAL_MAX_BATCHES', 0))


def _refresh_runtime_cfg() -> None:
    global CFG_TRAIN_DET, CFG_TRAIN_HEAD_POSE, CFG_TRAIN_ACT
    global CFG_TRAIN_PSR, CFG_USE_KENDALL
    global CFG_VAL_NUM_WORKERS, CFG_VAL_BATCH_SIZE, CFG_EVAL_MAX_BATCHES

    CFG_TRAIN_DET       = bool(getattr(C, 'TRAIN_DET', True))
    CFG_TRAIN_HEAD_POSE = bool(getattr(C, 'TRAIN_HEAD_POSE', True))
    CFG_TRAIN_ACT       = bool(getattr(C, 'TRAIN_ACT', True))
    CFG_TRAIN_PSR       = bool(getattr(C, 'TRAIN_PSR', True))
    CFG_USE_KENDALL     = bool(getattr(C, 'USE_KENDALL', True))
    CFG_VAL_NUM_WORKERS = int(getattr(C, 'VAL_NUM_WORKERS', C.NUM_WORKERS))
    CFG_VAL_BATCH_SIZE  = int(getattr(C, 'VAL_BATCH_SIZE', C.BATCH_SIZE))
    CFG_EVAL_MAX_BATCHES = int(getattr(C, 'EVAL_MAX_BATCHES', 0))


def seed_everything(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(getattr(C, 'CUDNN_DETERMINISTIC', False))
    torch.backends.cudnn.benchmark = bool(getattr(C, 'CUDNN_BENCHMARK', True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(C, 'ALLOW_TF32', True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(C, 'ALLOW_TF32', True))


def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)

        # Doc 02 D.1: RandAugment for backbone (photometric only)
        # Only on training images; photometric ops: brightness, contrast, posterize, etc.
        if bool(getattr(C, 'USE_RANDAUGMENT', False)):
            try:
                from torchvision.transforms.v2 import RandAugment
                rand_aug = RandAugment(num_ops=2, magnitude=9)
                images = rand_aug(images)
            except Exception:
                pass

        mean = torch.tensor(C.IMAGENET_MEAN, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


def _build_loader(
    ds: Any,
    split: str,
    batch_size: int,
    num_workers: int,
    prefetch: int = 1,
    persistent: Optional[bool] = None,
) -> DataLoader:
    is_train = split == 'train'
    sampler = ds.get_sampler() if is_train else None
    effective_prefetch = prefetch if num_workers > 0 else None
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


def _choose_num_workers(
    split: str,
    requested_workers: int,
    batch_size: int,
    prefetch: int = 1,
) -> int:
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
            f'({requested_workers}). Falling back to num_workers={fallback_workers}.'
        )
        return fallback_workers
    return requested_workers


def _flush_before_val(optimizer) -> None:
    """Aggressively free CPU RAM before starting validation."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e9

    # Clear COCO cache ( IndustReal uses COCO-format OD labels)
    if hasattr(_ds_module, '_PROC_COCO_CACHE'):
        cache = getattr(_ds_module, '_PROC_COCO_CACHE', None)
        if isinstance(cache, dict):
            cache.clear()

    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()

    rss_after = proc.memory_info().rss / 1e9
    freed_mb = (rss_before - rss_after) * 1024
    logger.info(
        f'  [pre-val flush] RSS: {rss_before:.2f}GB -> {rss_after:.2f}GB '
        f'(freed ~{freed_mb:.0f} MB)'
    )


def mixup_activity(
    outputs: Dict,
    targets: Dict,
    alpha: float = 0.4,
):
    """
    Apply mixup augmentation to activity task only.
    Mixup blends pixels globally. CutMix pastes rectangular patches.
    """
    if alpha <= 0:
        return outputs, targets

    B = outputs['act_logits'].shape[0]
    if B < 2:
        return outputs, targets

    lam = np.random.beta(alpha, alpha)

    activity_labels = targets['activity']
    same_label_mask = activity_labels.unsqueeze(0) == activity_labels.unsqueeze(1)
    same_label_mask = same_label_mask.float()

    if lam < 0.3 or lam > 0.7:
        return outputs, targets
    if same_label_mask.sum() > 0.5 * B * B:
        return outputs, targets

    indices = torch.randperm(B)

    act_logits = outputs['act_logits']
    mixed_act_logits = lam * act_logits + (1 - lam) * act_logits[indices]

    num_classes = act_logits.shape[1]
    activity_onehot = F.one_hot(activity_labels, num_classes).float()
    mixed_activity = lam * activity_onehot + (1 - lam) * activity_onehot[indices]

    mixed_outputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in outputs.items()}
    mixed_outputs['act_logits'] = mixed_act_logits

    mixed_targets = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in targets.items()}
    mixed_targets['activity'] = mixed_activity
    mixed_targets['_mixup_lambda'] = lam

    return mixed_outputs, mixed_targets


def cutmix_activity(
    outputs: Dict,
    targets: Dict,
    images: torch.Tensor,
    alpha: float = 1.0,
):
    """
    CutMix augmentation for activity (Doc 02 D.2).

    Pastes a rectangular patch from one video into another.
    Better than Mixup for fine-grained recognition — model has to identify
    which region drives the label.

    Alternates with Mixup on even/odd epochs.
    """
    if alpha <= 0:
        return outputs, targets

    B = outputs['act_logits'].shape[0]
    if B < 2:
        return outputs, targets

    lam = np.random.beta(alpha, alpha)
    _, _, H, W = images.shape

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    indices = torch.randperm(B, device=images.device)

    images_mixed = images.clone()
    images_mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    act_logits = outputs['act_logits']
    mixed_act_logits = lam * act_logits + (1 - lam) * act_logits[indices]

    activity_labels = targets['activity']
    num_classes = act_logits.shape[1]
    activity_onehot = F.one_hot(activity_labels, num_classes).float()
    mixed_activity = lam * activity_onehot + (1 - lam) * activity_onehot[indices]

    mixed_outputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in outputs.items()}
    mixed_outputs['act_logits'] = mixed_act_logits

    mixed_targets = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in targets.items()}
    mixed_targets['activity'] = mixed_activity
    mixed_targets['_mixup_lambda'] = lam

    return mixed_outputs, mixed_targets


def get_stage(epoch: int) -> int:
    """
    Doc 02 B.1: Three-stage training schedule.

    Stage 1 (epochs 1-5): Detection-only warmup
      - Active losses: L_det only
      - Backbone: layer1-3 frozen, layer4 + FPN + det head trainable

    Stage 2 (epochs 6-15): Add pose + head pose
      - Active losses: L_det + L_pose + L_head_pose
      - Activity and PSR heads exist but NOT in loss yet

    Stage 3 (epochs 16-100): Full multi-task with EMA
      - All losses active
      - EMA decay 0.999 starts here
    """
    stage1_end = int(getattr(C, 'STAGE1_EPOCHS', 5))
    stage2_end = stage1_end + int(getattr(C, 'STAGE2_EPOCHS', 10))

    if epoch <= stage1_end:
        return 1
    if epoch <= stage2_end:
        return 2
    return 3


def _set_stage_requires_grad(model: nn.Module, stage: int, backbone_type: str) -> None:
    """
    Freeze/unfreeze model parameters based on training stage.

    Doc 02 B.1 + 01_HONEST_AUDIT.md B.2: Explicit parameter freezing per stage.

    Stage 1 (epochs 1-5): layer1-3 frozen, + activity/PSR heads frozen
      → Detection backbone warms up without corrupted gradients from random heads

    Stage 2 (epochs 6-15): layer1-2 frozen, + activity/PSR heads frozen
      → Pose head added; backbone layer3 now trainable for mid-level features

    Stage 3 (epochs 16+): all trainable
      → Full multi-task training

    This is distinct from loss masking (which already exists). Parameter freezing
    prevents the AdamW optimizer from maintaining state for frozen params and
    ensures zero gradient flow to frozen layers.
    """
    # First, unfreeze everything (Stage 3 = all trainable)
    for p in model.parameters():
        p.requires_grad = True

    if stage == 1:
        # Freeze layer1-3 (ResNet) / stages[0-2] (ConvNeXt) — keep layer4 trainable
        if backbone_type == 'resnet50':
            for layer_idx in [1, 2, 3]:
                set_backbone_stage_requires_grad(
                    model, backbone_type, stage=layer_idx, requires_grad=False
                )
        elif backbone_type == 'convnext_tiny':
            for stage_idx in [0, 1, 2]:
                set_backbone_stage_requires_grad(
                    model, backbone_type, stage=stage_idx, requires_grad=False
                )
        # Freeze task heads
        for name, p in model.named_parameters():
            if 'activity_head' in name or 'psr_head' in name:
                p.requires_grad = False

    elif stage == 2:
        # Freeze layer1-2 (ResNet) / stages[0-1] (ConvNeXt) — keep layer3-4 trainable
        if backbone_type == 'resnet50':
            for layer_idx in [1, 2]:
                set_backbone_stage_requires_grad(
                    model, backbone_type, stage=layer_idx, requires_grad=False
                )
        elif backbone_type == 'convnext_tiny':
            for stage_idx in [0, 1]:
                set_backbone_stage_requires_grad(
                    model, backbone_type, stage=stage_idx, requires_grad=False
                )
        # Freeze activity/PSR heads (pose and head_pose remain trainable)
        for name, p in model.named_parameters():
            if 'activity_head' in name or 'psr_head' in name:
                p.requires_grad = False

    # stage == 3: all trainable (already set above)

    # Count frozen vs trainable
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Stage {stage}: frozen={frozen/1e6:.1f}M, trainable={trainable/1e6:.1f}M')


def train_one_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    accum_steps: int = C.GRAD_ACCUM_STEPS,
    ema=None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Doc 02 B.1: Staged training — determine current stage
    stage = get_stage(epoch)
    staged_training = bool(getattr(C, 'STAGED_TRAINING', True))

    # Doc 01 B.2 + Doc 02 B.1: Freeze/unfreeze backbone stages and heads per stage
    # Only applies when STAGED_TRAINING=True (default True in config)
    if staged_training:
        backbone_type = str(getattr(C, 'BACKBONE', 'resnet50'))
        _set_stage_requires_grad(model, stage, backbone_type)

    running = {
        'total': 0.0,
        'det': 0.0,
        'det_cls': 0.0,
        'det_reg': 0.0,
        'head_pose': 0.0,
        'activity': 0.0,
        'psr': 0.0,
        'w_det': 0.0,
        'w_pose': 0.0,
        'w_act': 0.0,
        'w_psr': 0.0,
        'log_var_det': 0.0,
        'log_var_head_pose': 0.0,
        'log_var_act': 0.0,
        'log_var_psr': 0.0,
    }
    num_batches = 0
    nan_skips = 0
    total_steps = 0
    t_start = time.time()

    pbar = tqdm(loader, desc=f'Epoch {epoch} [stage={stage}]', leave=True, dynamic_ncols=True)

    for step, (images, targets) in enumerate(pbar):
        total_steps = step + 1
        images = _prepare_images(images, device)

        # Move detection targets to device
        for i in range(len(targets['detection'])):
            targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].to(device)
            targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to(device)
        targets['head_pose'] = targets['head_pose'].to(device)
        targets['psr_labels'] = targets['psr_labels'].to(device)
        targets['activity'] = targets['activity'].to(device)
        hand_joints = targets.get('hand_joints', torch.zeros_like(
            images[:, :1, 0, 0]
        )).to(device, non_blocking=True)

        with amp.autocast(enabled=C.MIXED_PRECISION):
            clip_rgb = targets.get('clip_rgb')
            if clip_rgb is not None:
                clip_rgb = clip_rgb.to(device)
            outputs = model(images, clip_rgb=clip_rgb)
        for _k in ('cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits'):
            if _k in outputs and isinstance(outputs[_k], torch.Tensor):
                outputs[_k] = outputs[_k].float()

        # Doc 02 D.2: Alternate Mixup/CutMix each epoch
        if C.USE_MIXUP and epoch >= C.ACT_WARMUP_EPOCHS:
            use_cutmix = bool(getattr(C, 'CUTMIX_ALPHA', 0) > 0 and epoch % 2 == 1)
            if use_cutmix:
                outputs, targets = cutmix_activity(
                    outputs, targets, images, getattr(C, 'CUTMIX_ALPHA', 1.0),
                )
                outputs, targets = mixup_activity(outputs, targets, C.MIXUP_ALPHA)

        # Doc 02 B.1: Staged loss computation
        criterion.set_epoch(epoch)
        loss, loss_dict = criterion(outputs, targets)

        # Override losses based on stage
        if staged_training:
            if stage == 1:
                loss_dict['activity'] = 0.0
                loss_dict['psr'] = 0.0
                loss = loss_dict['det']
            elif stage == 2:
                loss_dict['activity'] = 0.0
                loss_dict['psr'] = 0.0
                loss = loss_dict['det'] + loss_dict['pose']

        loss = loss / accum_steps

        if not torch.isfinite(loss):
            nan_skips += 1
            det_val = float(loss_dict.get('det', float('nan')))
            pose_val = float(loss_dict.get('head_pose', float('nan')))
            act_val = float(loss_dict.get('activity', float('nan')))
            psr_val = float(loss_dict.get('psr', float('nan')))
            if nan_skips <= 10:
                logger.warning(
                    f'  NaN/Inf loss at epoch {epoch} step {step + 1} '
                    f'(skip #{nan_skips}) -- '
                    f'det={det_val:.4f} pose={pose_val:.4f} '
                    f'act={act_val:.4f} psr={psr_val:.4f}; '
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
            if ema is not None and stage >= 3:
                ema.update()
            optimizer.zero_grad(set_to_none=True)

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k]
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.3f}",
            'det':  f"{loss_dict['det']:.3f}",
            'pose': f"{loss_dict['head_pose']:.3f}",
            'act':  f"{loss_dict['activity']:.3f}",
            'psr':  f"{loss_dict['psr']:.3f}",
        }, refresh=False)

    if nan_skips > 0:
        logger.warning(f'  Epoch {epoch}: skipped {nan_skips} NaN/Inf batches total')
        if nan_skips / max(total_steps, 1) > 0.10:
            logger.error(
                f'  Epoch {epoch}: {nan_skips}/{total_steps} NaN batches '
                f'({nan_skips / max(total_steps, 1):.1%}) exceeds 10% -- '
                f'gradient signal unreliable'
            )

    avg = {k: v / max(num_batches, 1) for k, v in running.items()}
    avg['epoch_time'] = time.time() - t_start
    avg['nan_skips'] = nan_skips
    avg['stage'] = stage
    return avg


def _has_nan(metrics: Dict) -> bool:
    for v in metrics.values():
        if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
            return True
        if isinstance(v, dict) and _has_nan(v):
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
    logger.info(f'  Checkpoint load: {len(compatible)}/{len(model_state)} tensors loaded')
    if result.missing_keys:
        missing_by_component = {}
        for key in result.missing_keys:
            component = key.split('.')[0]
            missing_by_component.setdefault(component, []).append(key)
        for comp, keys in sorted(missing_by_component.items()):
            logger.info(f'  MISSING ({comp}): {keys}')
    return result, skipped


def _check_ram(label: str = '', warn_gb: float = 50.0) -> float:
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1e9
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
    map50: float,
    macro_f1_act: float,
    mae_head_pose: float,
    macro_f1_psr: float,
) -> float:
    """Combined validation metric for 4-task IndustReal."""
    head_pose_acc = 1.0 / (1.0 + mae_head_pose)
    combined = (
        _W_DET * map50
        + _W_ACT * macro_f1_act
        + _W_POSE * head_pose_acc
        + _W_PSR * macro_f1_psr
    )
    return combined


def _apply_runtime_safety(device: torch.device) -> None:
    nice_value = int(getattr(C, 'TRAIN_NICE', 0))
    if nice_value > 0:
        try:
            os.nice(nice_value)
            logger.info(f'Process nice level increased by +{nice_value}.')
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
                    f'CUDA memory fraction cap enabled: {mem_fraction:.2f}.'
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                logger.warning(f'Could not set CUDA memory fraction cap: {exc}')


def main(args):
    seed_everything(C.SEED)

    log_dir = C.LOG_DIR;        log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = C.CHECKPOINT_DIR; ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    if torch.cuda.is_available():
        logger.info(f'GPU : {torch.cuda.get_device_name()}')
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f'VRAM: {vram:.1f} GB')

    _apply_runtime_safety(device)

    logger.info(
        f'Ablation: TRAIN_DET={CFG_TRAIN_DET}  '
        f'TRAIN_HEAD_POSE={CFG_TRAIN_HEAD_POSE}  '
        f'TRAIN_ACT={CFG_TRAIN_ACT}  '
        f'TRAIN_PSR={CFG_TRAIN_PSR}  '
        f'USE_KENDALL={CFG_USE_KENDALL}'
    )

    # Debug mode overrides
    max_recordings = None
    if C.DEBUG_MODE:
        max_recordings = C.DEBUG_MAX_VIDEOS
        logger.info(
            f'[train] Debug mode: max_recordings={max_recordings}, '
            f'VAL_EVERY={C.VAL_EVERY}'
        )

    logger.info('Building datasets ...')
    train_ds = IndustRealMultiTaskDataset(
        split='train',
        img_size=C.IMG_SIZE,
        augment=True,
        seed=C.SEED,
        max_recordings=max_recordings,
    )
    val_ds = IndustRealMultiTaskDataset(
        split='val',
        img_size=C.IMG_SIZE,
        augment=False,
        seed=C.SEED,
        max_recordings=max_recordings,
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
        train_ds,
        'train',
        train_batch_size,
        train_workers,
        prefetch=train_prefetch,
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
    backbone_type = str(getattr(C, 'BACKBONE', 'resnet50'))
    use_headpose_film = bool(getattr(C, 'USE_HEADPOSE_FILM', False))
    use_videomae = bool(getattr(C, 'USE_VIDEOMAE', False))
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type=backbone_type,
        use_headpose_film=use_headpose_film,
        use_videomae=use_videomae,
    ).to(device)
    params = count_parameters(model)

    USE_EMA = bool(getattr(C, 'USE_EMA', True))
    EMA_DECAY = float(getattr(C, 'EMA_DECAY', 0.999))
    ema = None
    if USE_EMA:
        from model import EMA as EMAClass
        ema = EMAClass(model, decay=EMA_DECAY, device=device)
        logger.info(f'EMA enabled: decay={EMA_DECAY}')
    else:
        logger.info('EMA disabled')
    logger.info(f'Backbone type     : {backbone_type}')
    logger.info(f'HeadPoseFiLM      : {use_headpose_film}')
    logger.info(f'VideoMAE stream   : {use_videomae}')
    logger.info(f'Total parameters  : {params["total_all"]:,}')
    logger.info(f'Trainable params  : {params["total_trainable"]:,}')
    for k, v in params.items():
        if not k.startswith('total'):
            logger.info(f'  {k:15s}: {v:>10,}')

    criterion = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=CFG_TRAIN_DET,
        train_pose=CFG_TRAIN_HEAD_POSE,
        train_act=CFG_TRAIN_ACT,
        train_psr=CFG_TRAIN_PSR,
        use_kendall=CFG_USE_KENDALL,
    ).to(device)
    criterion.set_class_counts(class_counts)

    backbone_params, head_params = [], []
    loss_params = list(criterion.parameters())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(ln in name for ln in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']):
            backbone_params.append(param)
        else:
            head_params.append(param)

    use_lion = bool(getattr(C, 'USE_LION', False))

    if use_lion:
        try:
            from lion_pytorch import Lion
            param_groups = [
                {'params': backbone_params, 'lr': C.BASE_LR * 0.1 * 0.3},
                {'params': head_params,     'lr': C.BASE_LR * 0.3},
            ]
            if loss_params:
                param_groups.append({'params': loss_params, 'lr': C.BASE_LR * 0.3})
            optimizer = Lion(param_groups, weight_decay=C.WEIGHT_DECAY * 3)
            logger.info('Optimizer: Lion (LR ~3× smaller, WD ~3× larger than AdamW)')
        except ImportError:
            logger.warning('Lion not installed, falling back to AdamW')
            use_lion = False
            param_groups = [
                {'params': backbone_params, 'lr': C.BASE_LR * 0.1},
                {'params': head_params,     'lr': C.BASE_LR},
            ]
            if loss_params:
                param_groups.append({'params': loss_params, 'lr': C.BASE_LR})
            optimizer = AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)
    else:
        param_groups = [
            {'params': backbone_params, 'lr': C.BASE_LR * 0.1},
            {'params': head_params,     'lr': C.BASE_LR},
        ]
        if loss_params:
            param_groups.append({'params': loss_params, 'lr': C.BASE_LR})
        optimizer = AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)
        logger.info('Optimizer: AdamW with differential LR (backbone=0.1×, heads=1×)')

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=C.WARMUP_EPOCHS)
    if bool(getattr(C, 'ONE_CYCLE_LR', False)):
        # Doc 02 E.2: OneCycleLR with super-convergence
        # High peak LR (5e-4) + aggressive cosine decay
        # Doc 01 B.3 fix: make max_lr dynamic based on actual num param groups
        n_groups = len(param_groups)
        backbone_lr = C.BASE_LR * 0.1
        head_lr = C.BASE_LR
        max_lr = (
            [backbone_lr * 0.5]  # backbone: lower LR for transformer backbone
            + [head_lr * 0.5] * (n_groups - 1)  # all other groups share head-scale LR
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=C.EPOCHS,
            steps_per_epoch=len(train_loader) // train_accum_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        scheduler = SequentialLR(optimizer, [warmup, scheduler],
                               milestones=[C.WARMUP_EPOCHS])
        logger.info('Scheduler: OneCycleLR (pct_start=0.1, max_lr=[5e-5, 5e-4])')
    elif C.USE_COSINE_ANNEALING:
        cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=C.T_0, T_mult=C.T_mult, eta_min=1e-6
        )
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[C.WARMUP_EPOCHS])
        logger.info('Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)')
    else:
        cosine = CosineAnnealingLR(
            optimizer, T_max=C.EPOCHS - C.WARMUP_EPOCHS, eta_min=1e-6
        )
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                               milestones=[C.WARMUP_EPOCHS])

    scaler = amp.GradScaler(enabled=C.MIXED_PRECISION)

    start_epoch = 0
    best_metric = 0.0
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        load_result, skipped_keys = _load_model_compat(model, ckpt['model'])
        if skipped_keys:
            logger.warning(
                f'  Skipped {len(skipped_keys)} checkpoint key(s) (shape mismatch):'
            )
            for k, cs, ms in skipped_keys:
                logger.warning(f'    {k}: ckpt={cs}  model={ms} -> re-initialized')
        else:
            logger.info('  All checkpoint keys loaded (no shape mismatches).')
        if load_result.missing_keys:
            logger.info(
                f'  Missing keys (new in model, using init): '
                f'{load_result.missing_keys}'
            )

        # Restore EMA shadow if present in checkpoint
        if ema is not None and 'ema_shadow' in ckpt and ckpt['ema_shadow']:
            ema.shadow.update({
                k: v.to(ema.device) if ema.device else v
                for k, v in ckpt['ema_shadow'].items()
                if k in ema.shadow
            })
            logger.info('  EMA shadow weights restored from checkpoint.')

        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info('  Optimizer state restored.')
        except ValueError as e:
            logger.warning(
                f'  Could not restore optimizer state ({e}). '
                f'Re-initialized -- LR schedule continues.'
            )
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = float(ckpt.get('best_metric', 0.0))
        patience_counter = int(ckpt.get('patience_counter', 0))
        logger.info(f'Resumed from epoch {start_epoch}, best={best_metric:.4f}')

        # Reset Kendall log_var params only for early-epoch resumes.
        if start_epoch < C.WARMUP_EPOCHS:
            with torch.no_grad():
                criterion.log_var_det.fill_(0.0)
                criterion.log_var_head_pose.fill_(-1.0)
                criterion.log_var_act.fill_(0.0)
                criterion.log_var_psr.fill_(0.0)
            logger.info(
                '  Reset Kendall log_var params (early epoch resume): '
                'det=0.0  head_pose=-1.0  act=0.0  psr=0.0'
            )
        else:
            logger.info(
                f'  Keeping learned Kendall log_var params '
                f'(epoch {start_epoch} >= warmup={C.WARMUP_EPOCHS}): '
                f'det={criterion.log_var_det.item():.3f}  '
                f'head_pose={criterion.log_var_head_pose.item():.3f}  '
                f'act={criterion.log_var_act.item():.3f}  '
                f'psr={criterion.log_var_psr.item():.3f}'
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
        f'  Combined metric weights: '
        f'det={_W_DET}  act={_W_ACT}  pose={_W_POSE}  psr={_W_PSR}'
    )
    logger.info('=' * 60)

    _eff_cache: Dict[str, Any] = {'epoch': -1, 'metrics': None}

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
                        ema=ema,
                    )
                    break
                except Exception as exc:
                    msg = str(exc)
                    is_loader_enomem = (
                        ('Cannot allocate memory' in msg
                         or _is_dataloader_shm_error(exc))
                        and getattr(train_loader, 'num_workers', 0) > 0
                    )
                    if is_loader_enomem:
                        logger.exception(
                            'DataLoader worker ENOMEM/SHM error. '
                            'Rebuilding train loader with num_workers=0 and retrying.'
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
                            f'CUDA OOM during training. Retrying epoch with reduced '
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
                            f'workers={train_workers}'
                        )
                        continue
                    raise

            scheduler.step()
            _check_ram(f'epoch_{epoch}_train')

            current_lr = optimizer.param_groups[1]['lr']
            logger.info(
                f'Train: loss={train_metrics["total"]:.4f}  '
                f'det={train_metrics["det"]:.4f}  '
                f'pose={train_metrics["head_pose"]:.4f}  '
                f'act={train_metrics["activity"]:.4f}  '
                f'psr={train_metrics["psr"]:.4f}  '
                f'lr={current_lr:.2e}  '
                f'time={train_metrics["epoch_time"]:.0f}s'
                + (
                    f'  nan_skips={train_metrics["nan_skips"]}'
                    if train_metrics['nan_skips'] > 0 else ''
                )
            )

            if C.LOG_EFFICIENCY_EVERY > 0 and (
                epoch == 0 or (epoch + 1) % C.LOG_EFFICIENCY_EVERY == 0
            ):
                if _eff_cache['epoch'] != epoch:
                    from evaluate import compute_efficiency_metrics
                    _eff_cache['epoch'] = epoch
                    _eff_cache['metrics'] = compute_efficiency_metrics(
                        model, device,
                        img_size=(C.IMG_HEIGHT, C.IMG_WIDTH),
                        num_hand_coords=52,
                        warmup_runs=3,
                        timed_runs=20,
                        batch_size=1,
                    )
                eff = _eff_cache['metrics']
                logger.info(
                    f'Efficiency: params={eff["eff_params_m"]:.2f}M  '
                    f'gflops={eff["eff_gflops"]:.1f}G  '
                    f'fps={eff["eff_fps"]:.1f}  '
                    f'res={eff["eff_resolution"]}'
                )

            val_metrics = {}
            if (epoch + 1) % C.VAL_EVERY == 0:
                logger.info('Running validation ...')
                _flush_before_val(optimizer)

                # Use EMA weights for validation (if EMA is enabled)
                if ema is not None:
                    ema.get_ema()
                    logger.info('  [EMA] Using exponential-moving-average weights for val')

                val_batch_size_rt = CFG_VAL_BATCH_SIZE
                val_workers_rt = val_workers
                val_prefetch_rt = val_prefetch
                val_max_batches_rt = CFG_EVAL_MAX_BATCHES

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
                        val_batch_size_rt,
                        val_workers_rt,
                        prefetch=val_prefetch_rt,
                    )
                    try:
                        val_metrics = evaluate_all(
                            model,
                            criterion,
                            val_loader,
                            device,
                            max_batches=val_max_batches_rt,
                        )
                    except Exception as exc:
                        is_cpu_enomem = 'Cannot allocate memory' in str(exc)
                        is_cuda_oom_v = _is_cuda_oom(exc)
                        if not (is_cpu_enomem or is_cuda_oom_v):
                            raise
                        if is_cuda_oom_v:
                            logger.exception(
                                'Validation CUDA OOM detected. Reducing val load and retrying.'
                            )
                        else:
                            logger.exception(
                                'Validation ENOMEM detected. Reducing val load and retrying.'
                            )
                        val_batch_size_rt = max(1, val_batch_size_rt // 2)
                        val_workers_rt = 0
                        val_prefetch_rt = 1
                        val_max_batches_rt = max(1, int(val_max_batches_rt) // 2)
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info(
                            f'Validation retry settings: batch={val_batch_size_rt} '
                            f'workers={val_workers_rt} prefetch={val_prefetch_rt} '
                            f'max_batches={val_max_batches_rt}'
                        )
                        continue
                    finally:
                        del val_loader
                        gc.collect()
                        torch.cuda.empty_cache()

                if ema is not None:
                    ema.restore()
                    logger.info('  [EMA] Restored original weights after val')

                logger.info(
                    f'Val: loss={val_metrics.get("loss", 0):.4f}  '
                    f'mAP50={val_metrics.get("det_mAP50", 0):.4f}  '
                    f'act_acc={val_metrics.get("act_accuracy", 0):.4f}  '
                    f'act_macro_f1={val_metrics.get("act_macro_f1", 0):.4f}  '
                    f'head_pose_mae={val_metrics.get("head_pose_MAE", 0):.4f}  '
                    f'psr_macro_f1={val_metrics.get("psr_macro_f1", 0):.4f}'
                )

                _task_keys = ('det_mAP50', 'act_macro_f1', 'psr_macro_f1')
                _task_nan = any(
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
                    _map50 = val_metrics.get('det_mAP50', 0.0)
                    _f1_act = val_metrics.get('act_macro_f1', 0.0)
                    _mae_pose = val_metrics.get('head_pose_MAE', float('nan'))
                    _f1_psr = val_metrics.get('psr_macro_f1', 0.0)

                    combined = _compute_combined_metric(
                        _map50, _f1_act, _mae_pose, _f1_psr
                    )
                    logger.info(
                        f'  combined={combined:.4f}  '
                        f'(best={best_metric:.4f}  '
                        f'patience={patience_counter}/{C.PATIENCE})'
                    )

                    if combined > best_metric:
                        best_metric = combined
                        patience_counter = 0
                        # Save EMA weights for best checkpoint (more stable)
                        save_dict = {
                            'epoch':            epoch,
                            'optimizer':       optimizer.state_dict(),
                            'scheduler':       scheduler.state_dict(),
                            'scaler':          scaler.state_dict(),
                            'best_metric':     best_metric,
                            'patience_counter': patience_counter,
                            'val_metrics':     val_metrics,
                        }
                        if ema is not None:
                            # Apply EMA weights temporarily for saving
                            ema.get_ema()
                            save_dict['model'] = model.state_dict()
                            ema.restore()
                            save_dict['ema_shadow'] = {k: v.clone() for k, v in ema.shadow.items()}
                        else:
                            save_dict['model'] = model.state_dict()
                        torch.save(save_dict, ckpt_dir / 'best.pth')
                        logger.info(
                            f'  ** New best model (combined={combined:.4f}) **'
                        )
                    else:
                        patience_counter += 1
                        logger.info(
                            f'  No improvement ({patience_counter}/{C.PATIENCE})'
                        )

                    if patience_counter >= C.PATIENCE:
                        logger.info(
                            f'Early stopping at epoch {epoch} '
                            f'(patience={C.PATIENCE})'
                        )
                        break

            torch.save({
                'epoch':            epoch,
                'model':           model.state_dict(),
                'optimizer':       optimizer.state_dict(),
                'scheduler':       scheduler.state_dict(),
                'scaler':          scaler.state_dict(),
                'best_metric':     best_metric,
                'patience_counter': patience_counter,
                'ema_shadow':      {k: v.clone() for k, v in ema.shadow.items()} if ema is not None else {},
            }, ckpt_dir / 'latest.pth')

            record = {
                'epoch': epoch,
                'lr': current_lr,
                'train': train_metrics,
            }
            if val_metrics:
                record['val'] = val_metrics
            log_file.write(json.dumps(record, default=str) + '\n')
            log_file.flush()

    finally:
        log_file.close()

    # =========================================================================
    # Stochastic Weight Averaging — SWA (Doc 02 E.3)
    # =========================================================================
    if bool(getattr(C, 'USE_SWA', False)):
        try:
            from torch.optim.swa_utils import AveragedModel, SWALR
        except ImportError:
            logger.warning('torch.optim.swa_utils not available — skipping SWA')
        else:
            swa_epochs = int(getattr(C, 'SWA_EPOCHS', 10))
            swa_lr = float(getattr(C, 'SWA_LR', 1e-5))
            logger.info(f'Running SWA for {swa_epochs} epochs at LR={swa_lr:.2e}')

            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr)
            swa_start_epoch = epoch + 1

            for swa_epoch in range(swa_start_epoch, swa_start_epoch + swa_epochs):
                logger.info(f'\n--- SWA Epoch {swa_epoch} ---')
                swa_train_metrics = train_one_epoch(
                    model,
                    criterion,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    swa_epoch,
                    accum_steps=train_accum_steps,
                    ema=None,
                )
                swa_scheduler.step()
                logger.info(
                    f'SWA train: loss={swa_train_metrics["total"]:.4f}  '
                    f'lr={optimizer.param_groups[1]["lr"]:.2e}'
                )

            logger.info('Updating SWA BatchNorm statistics ...')
            if hasattr(model, 'module'):
                swa_model.module = model
            else:
                swa_model.module = model

            try:
                from torch.optim.swa_utils import update_bn
                update_bn(train_loader, swa_model, device=device)
            except Exception as e:
                logger.warning(f'SWA BN update failed: {e}')

            swa_state = {
                'epoch': swa_start_epoch + swa_epochs - 1,
                'model': swa_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'swa': True,
            }
            torch.save(swa_state, ckpt_dir / 'swa.pth')
            logger.info(f'SWA checkpoint saved to {ckpt_dir / "swa.pth"}')

            del swa_model
            gc.collect()
            torch.cuda.empty_cache()

    logger.info('Training complete.')
    logger.info(f'Best combined metric: {best_metric:.4f}')
    logger.info(f'Checkpoints: {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Multi-Task IndustReal Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        help='Preset (kept for backwards compat -- no-op on IndustReal)',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Override C.EPOCHS',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override C.BATCH_SIZE',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (small dataset, fast validation)',
    )
    parser.add_argument(
        '--seed', '-s',
        type=int, default=None,
        help='Random seed override (sets C.SEED before training). '
             'Doc 03 C: use 42, 123, 7 for multi-seed runs.',
    )

    args = parser.parse_args()

    if args.preset:
        try:
            import config as _cfg_mod
            if hasattr(_cfg_mod, 'apply_preset'):
                _cfg_mod.apply_preset(args.preset)
                _refresh_runtime_cfg()
                logger.info(f'[train] Applied preset: {args.preset}')
            else:
                logger.warning(f'[train] Config has no apply_preset — ignoring --preset')
        except Exception as exc:
            logger.warning(f'[train] Failed to apply preset {args.preset}: {exc}')

    if args.max_epochs is not None:
        C.EPOCHS = args.max_epochs
    if args.batch_size is not None:
        C.BATCH_SIZE = args.batch_size
        C.EFFECTIVE_BATCH = C.BATCH_SIZE * C.GRAD_ACCUM_STEPS

    if args.debug:
        C.DEBUG_MODE = True
        C.DEBUG_MAX_VIDEOS = 5
        C.VAL_EVERY = 1
        logger.info('[train] Debug mode enabled: small dataset, fast validation')

    if args.seed is not None:
        C.SEED = args.seed
        logger.info(f'[train] Seed overridden to {args.seed}')

    _refresh_runtime_cfg()

    main(args)
