import faulthandler
import signal
faulthandler.enable()
# Use correct Python 3.13 API (was register_signal_handler, now just register)
faulthandler.register(signal.SIGUSR1)  # faulthandler.dump traceback on SIGUSR1

import sys
import os
from pathlib import Path

# Resolve symlinks so /home/... and /media/... both resolve to the same real path
# Then add each src/ subdirectory to sys.path — same pattern as smoke_test.py
_SRC = Path(__file__).resolve().parent.parent  # src/
for _sub in ['models', 'training', 'evaluation', 'data', str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# CRITICAL FIX: add project root so `from src import config` resolves correctly
# in model.py. Without this, `from src import config as C` resolves to src/src/config.py
# (relative to the src/ entry) which doesn't exist → silent AttributeError on None.
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

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
from typing import Any, Callable, Dict, Optional

import numpy as np
import psutil
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# --- THREAD CONVOVOY FIX (Bashara 2026-05-07) ---
# Reduce OpenMP/numpy/PyTorch thread counts to eliminate lock convoy.
# Without this: 28 threads all contend on jemalloc + GIL futex → deadlock.
# With this: 4 threads max → no convoy, GPU fully utilized.
os.environ['OMP_NUM_THREADS']       = '4'
os.environ['MKL_NUM_THREADS']       = '4'
os.environ['OPENBLAS_NUM_THREADS']   = '4'
os.environ['NUMEXPR_NUM_THREADS']    = '4'
os.environ['MALLOC_ARENA_MAX']      = '4'
# --- REPRODUCIBILITY FIX (I-5 2026-05-19) ---
# Required for deterministic GPU ops and hash-seed stability
# Note: C not yet imported, using hardcoded seed; C.SEED used later after import
os.environ['PYTHONHASHSEED']        = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
os.environ['CUDA_LAUNCH_BLOCKING']  = '1'
# ------------------------------------------------------------
import model as _model_module
import model as _popw_model_module
import losses as _losses_module
import evaluate as _evaluate_module
import data as _ds_module
from src import config as C

IndustRealMultiTaskDataset = getattr(_ds_module, 'IndustRealMultiTaskDataset')
# Doc 01 §D.2: When USE_PSR_SEQUENCE_MODE=True, use collate_fn_sequence which
# groups frames by (recording_id, camera_view) and provides psr_labels_seq,
# sequence_lengths, and frame_indices for temporal PSR training.
_collate_fn_name = 'collate_fn_sequences' if C.USE_PSR_SEQUENCE_MODE else 'collate_fn'
collate_fn = getattr(_ds_module, _collate_fn_name)

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

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def _prepare_images(images: torch.Tensor, device: torch.device, training: bool = True) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)

        if bool(getattr(C, 'USE_RANDAUGMENT', False)) and training:
            try:
                from torchvision.transforms.v2 import RandAugment
                rand_aug = RandAugment(num_ops=2, magnitude=9)
                if images.dim() == 5:
                    BT, C_, H, W = images.shape
                    images = rand_aug(images.view(BT, C_, H, W)).view(BT, C_, H, W)
                else:
                    images = rand_aug(images)
            except Exception:
                pass

        mean = torch.tensor(C.IMAGENET_MEAN, device=device, dtype=images.dtype)
        std = torch.tensor(C.IMAGENET_STD, device=device, dtype=images.dtype)
        if images.dim() == 5:
            mean = mean.view(1, 1, 3, 1, 1)
            std = std.view(1, 1, 3, 1, 1)
        else:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        images = (images - mean) / std

    if images.dim() == 5:
        BT = images.shape[0] * images.shape[1]
        images = images.view(BT, images.shape[2], images.shape[3], images.shape[4])

    return images


def _worker_seed_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for deterministic augmentation (I-4 2026-05-19)."""
    worker_seed = C.SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


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
    # --- CONVOY FIX (Bashara 2026-05-07) ---
    # Thread limits (OMP_NUM_THREADS=4, etc.) prevent the fork convoy that
    # used to cause 16 threads to block on jemalloc arenas + log fd.
    # DO NOT use 'spawn' — it triggers Python 3.13 loky semaphore bugs
    # that kill the process at shutdown. Fork is safe here because the
    # thread caps eliminate the convoy.
    # Cap prefetch to 2 to reduce worker memory pressure.
    _eff_prefetch = min(effective_prefetch, 2) if effective_prefetch else None
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
        prefetch_factor=_eff_prefetch,
        worker_init_fn=_worker_seed_fn if num_workers > 0 else None,
        # multiprocessing_context=None  ← use default fork (safe with thread caps)
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
    CutMix augmentation for activity (Doc 2 D.2).

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
    Doc 2 B.1: Three-stage training schedule.

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

    Doc 2 B.1 + 01_HONEST_AUDIT.md B.2: Explicit parameter freezing per stage.

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
        # [FIX #7 LOW] Paper §Training: "stages[0-1] frozen" for ConvNeXt — only 2 stages, not 3
        if backbone_type == 'resnet50':
            for layer_idx in [1, 2, 3]:
                set_backbone_stage_requires_grad(
                    model, backbone_type, stage=layer_idx, requires_grad=False
                )
        elif backbone_type == 'convnext_tiny':
            for stage_idx in [0, 1]:  # Paper: stages[0-1] frozen (not 0,1,2)
                set_backbone_stage_requires_grad(
                    model, backbone_type, stage=stage_idx, requires_grad=False
                )
        # Freeze task heads
        for name, p in model.named_parameters():
            if 'activity_head' in name or 'psr_head' in name:
                p.requires_grad = False

    elif stage == 2:
        # Freeze layer1-2 (ResNet) / stages[0-1] (ConvNeXt) — keep layer3-4 trainable
        # [FIX #7 LOW] Paper §Training: "stages[0] frozen" for ConvNeXt — only 1 stage, not 2
        if backbone_type == 'resnet50':
            for layer_idx in [1, 2]:
                set_backbone_stage_requires_grad(
                    model, backbone_type, stage=layer_idx, requires_grad=False
                )
        elif backbone_type == 'convnext_tiny':
            for stage_idx in [0]:  # Paper: stage[0] frozen (not 0,1)
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
    ckpt_dir,
    accum_steps: int = C.GRAD_ACCUM_STEPS,
    ema=None,
    seq_loader=None,
    resume_batch: int = 0,   # FIX: skip N batches for mid-epoch resume
    best_metric: float = 0.0,  # FIX: pass best_metric explicitly to avoid closure scoping issue
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    seq_iter = None
    seq_every = int(getattr(C, 'PSR_SEQ_EVERY_N_BATCHES', 10))
    if seq_loader is not None:
        seq_iter = iter(seq_loader)

    # Doc 2 B.1: Staged training — determine current stage
    stage = get_stage(epoch)
    staged_training = bool(getattr(C, 'STAGED_TRAINING', True))

    # Doc 01 B.2 + Doc 2 B.1: Freeze/unfreeze backbone stages and heads per stage
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
        'log_var_pose': 0.0,
        'log_var_act': 0.0,
        'log_var_psr': 0.0,
    }
    _debug_logged_once = False  # [DEBUG] flag to suppress first-step logs
    num_batches = 0
    nan_skips = 0
    total_steps = 0
    seq_steps = 0
    t_start = time.time()

    _heartbeat_interval = 10   # log heartbeat every N batches

    _debug_interval = 10       # [DEBUG] per-batch loss debug every N batches

    # --- PROGRESS BAR (Bashara 2026-05-08: kept pbar for display, enumerate over loader directly) ---
    pbar = tqdm(loader, desc=f'Epoch {epoch} [stage={stage}]', leave=True, dynamic_ncols=True)

    # --- CRASH-SAFE CHECKPOINT SAVE (Bashara 2026-05-09) ---
    # Saves minimal recovery state to crash_recovery.pth. This is called at the
    # START of each epoch and every _checkpoint_interval batches so that if training
    # is killed or crashes, we can resume from the last safe point.
    def _checkpoint_has_nan(model) -> bool:
        """Guard: check model tensors for NaN/Inf before saving."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if not torch.isfinite(param).all():
                    logger.warning(
                        f'  [NaN_GUARD] Parameter {name} contains NaN/Inf -- '
                        f'skipping checkpoint save'
                    )
                    return True
        return False

    def _save_named_checkpoint(ckpt_dir: Path, tag: str) -> Optional[Path]:
        """Save a named periodic checkpoint with full state."""
        if _checkpoint_has_nan(model):
            return None
        try:
            save_dict = {
                'tag': tag,
                'epoch': epoch,
                'step': num_batches,
                'batch': num_batches,   # 1-indexed batch counter within epoch (for mid-epoch resume)
                'total_steps': total_steps,
                'seq_steps': seq_steps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'nan_skips': nan_skips,
                'running': running,
                'num_batches': num_batches,
                'timestamp': time.time(),
            }
            if ema is not None:
                save_dict['ema_shadow'] = {k: v.clone() for k, v in ema.shadow.items()}
            if criterion is not None:
                save_dict['criterion'] = {
                    'log_var_det': criterion.log_var_det.data.clone(),
                    'log_var_pose': criterion.log_var_pose.data.clone(),
                    'log_var_act': criterion.log_var_act.data.clone(),
                    'log_var_psr': criterion.log_var_psr.data.clone(),
                }
            path = ckpt_dir / f'{tag}.pth'
            torch.save(save_dict, path)
            logger.info(f'  [CHECKPOINT] Saved {tag} to {path.name}')
            torch.cuda.synchronize()
            return path
        except Exception as exc:
            logger.warning(f'  [CHECKPOINT] Failed to save {tag}: {exc}')
            return None

    def _save_crash_recovery(ckpt_dir: Path, tag: str = '') -> None:
        try:
            if _checkpoint_has_nan(model):
                logger.warning(
                    '  [CRASH_RECOVERY] Skipping save -- model has NaN/Inf params'
                )
                return
            recovery_path = ckpt_dir / 'crash_recovery.pth'
            save_dict = {
                'tag': tag,
                'epoch': epoch,
                'step': num_batches,
                'batch': 0,   # Always 0: force epoch-boundary resume (avoids DataLoader pin_memory race during islice fast-forward)
                'total_steps': total_steps,
                'seq_steps': seq_steps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'nan_skips': nan_skips,
                'running': running,
                'best_metric': best_metric,   # Closure variable from main(); default 0.0 if not yet set
                'timestamp': time.time(),
            }
            if ema is not None:
                save_dict['ema_shadow'] = {k: v.clone() for k, v in ema.shadow.items()}
            # FIX: Save criterion (Kendall log_vars) for full state recovery
            # FIX: Save best_metric so resume doesn't regress on model selection
            if criterion is not None:
                save_dict['criterion'] = {
                    'log_var_det': criterion.log_var_det.data.clone(),
                    'log_var_pose': criterion.log_var_pose.data.clone(),
                    'log_var_act': criterion.log_var_act.data.clone(),
                    'log_var_psr': criterion.log_var_psr.data.clone(),
                }
            torch.save(save_dict, recovery_path)
            logger.info(f'  [CRASH_RECOVERY] Saved {tag} crash checkpoint to {recovery_path}')
            # CRITICAL: force flush to disk so even SIGKILL won't lose this
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(f'  [CRASH_RECOVERY] Failed to save crash checkpoint: {exc}')

    # --- SIGNAL HANDLERS for C-level crashes (Bashara 2026-05-08) ---
    # CUDA assertions / segfaults arrive as signals. Catch them and log before exit.
    def _sig_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.error(f'  [FATAL SIGNAL] {sig_name} received at step={num_batches} epoch={epoch}')
        logger.error('  [FATAL SIGNAL] Dumping faulthandler traceback:')
        faulthandler.dump_traceback()
        _save_crash_recovery(ckpt_dir, f'fatal_signal_{sig_name}')
        sys.exit(99)
    for _sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        signal.signal(_sig, _sig_handler)
    # Also catch SIGTERM (timeout / external kill) and SIGINT (Ctrl+C)
    def _sig_term_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f'  [SIGNAL] {sig_name} received at step={num_batches} epoch={epoch} -- saving crash recovery and exiting gracefully')
        _save_crash_recovery(ckpt_dir, f'signal_{sig_name}')
        sys.exit(0)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _sig_term_handler)
    # -----------------------------------------------------------------

    # Save crash recovery checkpoint BEFORE first batch (epoch start)
    _save_crash_recovery(ckpt_dir, 'epoch_start')

    _checkpoint_interval = 50  # Save crash checkpoint every 50 batches

    # FIX: Mid-epoch resume — fast-forward DataLoader to resume_batch without any compute.
    # All model/optimizer/EMA/criterion state is already restored from checkpoint.
    # We just need to advance the iterator to the correct position.
    if resume_batch > 0:
        logger.info(f'  Fast-forwarding DataLoader to batch {resume_batch} (no compute)...')
        from itertools import islice
        # Consume resume_batch items without processing (no GPU compute, no optimizer step)
        _ = list(islice(pbar, resume_batch))
        logger.info(f'  DataLoader positioned at batch {resume_batch}. Starting training.')

    for step, (images, targets) in enumerate(pbar):
        total_steps = step + 1

        # --- GPU MEMORY SNAPSHOT every 10 batches (skip step 0) ---
        if step > 0 and step % 10 == 0:
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.info(
                f'  [GPU mem] step={step}  allocated={mem_alloc:.2f}GB  '
                f'reserved={mem_reserved:.2f}GB'
            )

        # --- HEARTBEAT LOG (Bashara 2026-05-07) ---
        if step > 0 and step % _heartbeat_interval == 0:
            elapsed = time.time() - t_start
            logger.info(
                f'  [Epoch {epoch} batch {step}/{len(loader)}] '
                f'elapsed={elapsed:.0f}s  speed={step/elapsed:.1f} batch/s'
            )
        # ----------------------------------------

        # --- DATA INTEGRITY CHECK (Bashara 2026-05-08) ---
        # Catch NaN/Inf in images BEFORE the expensive CUDA forward pass.
        # A single bad frame can crash the GPU kernel silently.
        if step > 0:  # skip step 0 (first batch can have weird init values)
            _finite, _idx = torch.max(torch.abs(images), dim=0)
            _finite = torch.isfinite(_finite).all()
            if not _finite:
                nan_skips += 1
                logger.warning(
                    f'  [BAD_SAMPLE] NaN/Inf in images at epoch {epoch} step {step} '
                    f'(skip #{nan_skips}) -- skipping batch, zeroing grads'
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                pbar.set_postfix_str(f'BAD_SAMPLE skip={nan_skips}', refresh=True)
                continue
        # ---------------------------------------------------------

        # Doc 01 §D.2: Alternate PSR sequence batch every seq_every steps
        is_seq_batch = (seq_iter is not None and step > 0 and step % seq_every == 0)
        if is_seq_batch:
            try:
                images_seq, targets_seq = next(seq_iter)
            except StopIteration:
                seq_iter = iter(seq_loader)
                images_seq, targets_seq = next(seq_iter)
            B_seq = images_seq.shape[0]
            T_seq = images_seq.shape[1]
            images_seq = _prepare_images(images_seq, device)
            targets_seq = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                           for k, v in targets_seq.items()}
            clip_rgb_seq = targets_seq.get('clip_rgb')
            if clip_rgb_seq is not None:
                clip_rgb_seq = clip_rgb_seq.to(device)
            with amp.autocast('cuda', enabled=C.MIXED_PRECISION):
                outputs_seq = model(images_seq, clip_rgb=clip_rgb_seq)
                for _k in ('cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits'):
                    if _k in outputs_seq and isinstance(outputs_seq[_k], torch.Tensor):
                        outputs_seq[_k] = outputs_seq[_k].float()

                psr_logits_seq = outputs_seq['psr_logits'].view(B_seq, T_seq, -1)

                criterion.set_epoch(epoch)

                # [FIX #1] Save criterion flags BEFORE PSR-only sequence batch —
                # they are mutated below and must be restored so subsequent normal
                # batches are not corrupted (criterion persists across all batches).
                _saved_train_det  = criterion.train_det
                _saved_train_pose = criterion.train_pose
                _saved_train_act  = criterion.train_act
                _saved_train_psr  = criterion.train_psr

                criterion.train_psr = True
                criterion.train_pose = False
                criterion.train_act = False
                criterion.train_det = False

                fake_outputs = {
                    'psr_logits': psr_logits_seq,
                    # Omit head_pose: model produces [BT,9] but fake is [B_seq,T_seq,9]
                    # → shape mismatch in MultiTaskLoss.forward head_pose MSELoss.
                    # head_pose MSELoss only fires when 'head_pose' in outputs AND not None.
                    # Omitting it → loss_head_pose=zero (line 690), correct for PSR-only branch.
                    # Also omit cls_preds/reg_preds/heatmaps/keypoints — train_det=False
                    # and train_pose=False in this branch, so these are never consumed.
                }
                fake_targets = {
                    'psr_labels': targets_seq['psr_labels'],
                    # Omit head_pose from fake_targets too — matched to fake_outputs omission.
                    # detection/activity/hand_joints also omitted: train_det=False,
                    # train_act=False, train_pose=False in this branch.
                }
                loss_seq, loss_dict_seq = criterion(fake_outputs, fake_targets)
                loss_dict_seq = {k: 0.0 for k in loss_dict_seq}
                loss_dict_seq['psr'] = loss_seq.item()
                loss_dict_seq['total'] = loss_seq.item()

                loss_seq = loss_seq / float(accum_steps)
            if not torch.isfinite(loss_seq):
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                del outputs_seq, loss_seq, loss_dict_seq, fake_outputs, fake_targets
                torch.cuda.empty_cache()
                continue
            scaler.scale(loss_seq).backward()
            for k in running:
                if k in loss_dict_seq:
                    running[k] += loss_dict_seq[k]
            num_batches += 1
            pbar.set_postfix_str(
                f"loss={loss_dict_seq.get('total', loss_seq):.3f} "
                f"det={loss_dict_seq['det']:.3f} "
                f"pose={loss_dict_seq['head_pose']:.3f} "
                f"act={loss_dict_seq['activity']:.3f} "
                f"psr={loss_dict_seq['psr']:.3f} seq=1",
                refresh=True
            )
            del images_seq, targets_seq, outputs_seq, loss_seq, loss_dict_seq
            del fake_outputs, fake_targets
            torch.cuda.empty_cache()
            seq_steps += 1
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
            # [FIX #1] Restore criterion flags AFTER PSR-only sequence batch
            criterion.train_det  = _saved_train_det
            criterion.train_pose = _saved_train_pose
            criterion.train_act  = _saved_train_act
            criterion.train_psr  = _saved_train_psr
            continue

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

        with amp.autocast('cuda', enabled=C.MIXED_PRECISION):
            clip_rgb = targets.get('clip_rgb')
            if clip_rgb is not None:
                clip_rgb = clip_rgb.to(device)
            outputs = model(images, clip_rgb=clip_rgb)
        for _k in ('cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits'):
            if _k in outputs and isinstance(outputs[_k], torch.Tensor):
                outputs[_k] = outputs[_k].float()

        # Doc 2 D.2: Alternate Mixup/CutMix each epoch
        if C.USE_MIXUP and epoch >= int(getattr(C, 'ACT_RAMP_EPOCHS', 5)):
            use_cutmix = bool(getattr(C, 'CUTMIX_ALPHA', 0) > 0 and epoch % 2 == 1)
            if use_cutmix:
                outputs, targets = cutmix_activity(
                    outputs, targets, images, getattr(C, 'CUTMIX_ALPHA', 1.0),
                )

        # Doc 2 B.1: Staged loss computation
        criterion.set_epoch(epoch)
        loss, loss_dict = criterion(outputs, targets)

        # Override losses based on stage — keep `loss` as 0D tensor for NaN/isfinite guard
        if staged_training:
            if stage == 1:
                loss_dict['activity'] = 0.0
                loss_dict['psr'] = 0.0
            elif stage == 2:
                loss_dict['activity'] = 0.0
                loss_dict['psr'] = 0.0

        # Apply accum_steps scaling — preserve tensor dtype for isfinite() check
        loss = loss / float(accum_steps)
        # Kendall branch already handles staged precision zeroing internally (Bug 2 fix).
        # Only apply the manual staged override when Kendall is DISABLED, since
        # in that case the Kendall branch computes unweighted losses and we need
        # to manually zero frozen-task contributions to preserve gradient flow.
        # When Kendall IS active, the Kendall total IS the correct staged loss.
        if staged_training and not criterion.use_kendall:
            if stage == 1:
                loss = torch.tensor(loss_dict['det'] / float(accum_steps),
                                     dtype=torch.float32, device=device)
                loss.requires_grad_(True)
            elif stage == 2:
                loss = torch.tensor((loss_dict['det'] + loss_dict['pose']) / float(accum_steps),
                                     dtype=torch.float32, device=device)
                loss.requires_grad_(True)

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
        # Guard: clamp per-component losses to prevent NaN from propagating into Kendall total
        # (only needed when Kendall is active; staged non-Kendall losses are already scalar tensors)
        if criterion.use_kendall:
            for key in ['det', 'head_pose', 'activity', 'psr']:
                v = loss_dict.get(key)
                if v is not None and not torch.isfinite(torch.tensor(v, dtype=torch.float32)):
                    loss_dict[key] = 0.0
            # [DEBUG] Per-batch loss + model output shape logging every _debug_interval batches
            if step > 0 and step % _debug_interval == 0:
                # Log all individual loss components
                logger.info(
                    f'  [DEBUG epoch={epoch} step={step}] '
                    f'total={loss_dict["total"]:.4f} '
                    f'det={loss_dict["det"]:.4f} '
                    f'det_cls={loss_dict["det_cls"]:.4f} '
                    f'det_reg={loss_dict["det_reg"]:.4f} '
                    f'pose={loss_dict["pose"]:.4f} '
                    f'head_pose={loss_dict["head_pose"]:.4f} '
                    f'act={loss_dict["activity"]:.4f} '
                    f'psr={loss_dict["psr"]:.4f}'
                )
            _debug_logged_once = True

        scaler.scale(loss).backward()

        # Doc 2 §B.1: Kendall gradient sentinel — log gradient norms of log_var params
        log_kendall_every = int(getattr(C, 'LOG_KENDALL_GRAD_EVERY', 100))
        if log_kendall_every > 0:
            _log_kendall_gradient_sentinel(criterion, step, log_kendall_every)

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

        # --- NaN/ZERO GUARD: zero per-component losses that are NaN before accumulating.
        # Prevents NaN from propagating into running averages (logged every epoch).
        for k in ('det', 'det_cls', 'det_reg', 'head_pose', 'activity', 'psr',
                  'w_det', 'w_pose', 'w_act', 'w_psr',
                  'log_var_det', 'log_var_pose', 'log_var_act', 'log_var_psr'):
            v = loss_dict.get(k)
            if v is not None and not (isinstance(v, float) and math.isfinite(v)):
                loss_dict[k] = 0.0
        for k in running:
            if k in loss_dict:
                v = loss_dict[k]
                if isinstance(v, float) and math.isfinite(v):
                    running[k] += v
                else:
                    running[k] += 0.0  # NaN/Inf contribution zeroed
        num_batches += 1

        # [2% FIX] Enforce TRAIN_MAX_STEPS at batch granularity — BEFORE logging
        if getattr(C, 'TRAIN_MAX_STEPS', 0) > 0:
            if not hasattr(C, '_global_step'):
                C._global_step = 0
            C._global_step += 1
            if C._global_step >= C.TRAIN_MAX_STEPS:
                logger.info(f'  [2pct] batch-level TRAIN_MAX_STEPS limit reached ({C._global_step}). Stopping.')
                break

        # Doc 2 §B.3: Loss component breakdown (logged every 50 steps)
        if (step + 1) % 50 == 0:
            loss_dict['total'] = loss_dict.get('total', loss)
            _log_loss_component_breakdown(loss_dict, stage, epoch)

        pbar.set_postfix_str(
            f"loss={loss_dict['total']:.3f} "
            f"det={loss_dict['det']:.3f}(c={loss_dict['det_cls']:.3f},g={loss_dict['det_reg']:.3f}) "
            f"pose={loss_dict['head_pose']:.3f} "
            f"act={loss_dict['activity']:.3f} "
            f"psr={loss_dict['psr']:.3f} "
            f"wd={loss_dict['w_det']:.2f}",
            refresh=True
        )

        # --- PERIODIC CHECKPOINT every _checkpoint_interval batches (Bashara 2026-05-09) ---
        # Save named checkpoint (epoch_N_batch_M.pth) so we can resume from any crash point.
        # Also save crash_recovery.pth as a always-overwritten safety net.
        if (step + 1) % _checkpoint_interval == 0:
            ckpt_tag = f'epoch_{epoch}_batch_{(step + 1)}'
            _save_named_checkpoint(ckpt_dir, ckpt_tag)
            _save_crash_recovery(ckpt_dir, f'batch_{(step + 1)}')
            torch.cuda.synchronize()  # force GPU flush before continuing

        # --- CPU RAM WATCHDOG every 50 batches (Bashara 2026-05-09) ---
        # Check host RAM before we risk OOM. Alert if < 2GB available.
        if (step + 1) % 50 == 0:
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = {}
                    for line in f:
                        if ':' in line:
                            k, v = line.strip().split(':')
                            meminfo[k.strip()] = v.strip()
                avail_kb = int(meminfo.get('MemAvailable', '0').split()[0])
                avail_gb = avail_kb / 1024 / 1024
                buffers_kb = int(meminfo.get('Buffers', '0').split()[0])
                cached_kb = int(meminfo.get('Cached', '0').split()[0])
                if avail_gb < 2.0:
                    logger.error(
                        f'  [CPU RAM WARNING] Available={avail_gb:.1f}GB -- '
                        f'OOM risk! Consider reducing VAL_NUM_WORKERS or NUM_WORKERS'
                    )
                else:
                    logger.info(
                        f'  [CPU RAM] step={step + 1}  avail={avail_gb:.1f}GB  '
                        f'buffers={buffers_kb/1024:.0f}GB  cached={cached_kb/1024:.0f}GB'
                    )
            except Exception as exc:
                logger.warning(f'  [CPU RAM] watchdog failed: {exc}')

        # --- DataLoader worker health check every 100 batches (Bashara 2026-05-09) ---
        # Catch DataLoader worker crashes (common cause of training death).
        # DataLoader does not expose a public is_alive() check, so we rely on
        # catching exceptions during iteration — if a worker dies, next() on the
        # iterator raises BrokenPipeError or FileNotFoundError. This is detected
        # by the outer try/except in main() which rebuilds the loader.
        # We additionally try a non-blocking iterator join to detect stale workers.
        if (step + 1) % 100 == 0 and loader.num_workers > 0:
            try:
                import multiprocessing.util as _mp_util
                # _worker_result_queue is a SimpleQueue that holds worker results.
                # If workers are alive it will be non-empty after a non-blocking get.
                # This is a best-effort check — it does NOT guarantee workers are healthy
                # but a positive detection means at least one worker exited.
                if hasattr(loader, '_worker_result_queue'):
                    import queue
                    try:
                        # Non-blocking check — if queue has items, workers may have produced results
                        # but also could mean results weren't collected (normal operation)
                        # The real health signal is in the exception path below.
                        loader._worker_result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    except Exception:
                        pass
                logger.info(f'  [DataLoader] step={step + 1}  health_check=done')
            except Exception:
                pass

    total_all_steps = total_steps + seq_steps
    if nan_skips > 0:
        logger.warning(f'  Epoch {epoch}: skipped {nan_skips} NaN/Inf batches total')
        if nan_skips / max(total_all_steps, 1) > 0.10:
            logger.error(
                f'  Epoch {epoch}: {nan_skips}/{total_all_steps} NaN batches '
                f'({nan_skips / max(total_all_steps, 1):.1%}) exceeds 10% -- '
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


# ===========================================================================
# Monitoring Hooks (Doc 2 §B)
# ===========================================================================

def _log_kendall_gradient_sentinel(criterion, step_idx: int, log_interval: int) -> None:
    """
    Doc 2 §B.1: Kendall gradient sentinels.
    Every N steps, log gradient norms of Kendall log_var params.
    All should be nonzero in stages where corresponding task is active.
    """
    if step_idx % log_interval != 0:
        return
    try:
        grad_det = criterion.log_var_det.grad.norm().item() if criterion.log_var_det.grad is not None else 0.0
        grad_pose = criterion.log_var_pose.grad.norm().item() if criterion.log_var_pose.grad is not None else 0.0
        grad_act = criterion.log_var_act.grad.norm().item() if criterion.log_var_act.grad is not None else 0.0
        grad_psr = criterion.log_var_psr.grad.norm().item() if criterion.log_var_psr.grad is not None else 0.0
        logger.debug(
            f'  [Kendall grad] det={grad_det:.6f} pose={grad_pose:.6f} '
            f'act={grad_act:.6f} psr={grad_psr:.6f}'
        )
    except Exception:
        pass


def _log_loss_component_breakdown(
    loss_dict: Dict,
    stage: int,
    epoch: int,
) -> None:
    """
    Doc 2 §B.3: Loss component breakdown.
    Log each Kendall-weighted component separately so we can detect dominance.
    """
    total = loss_dict.get('total', 0.0)
    det = loss_dict.get('det', 0.0)
    pose = loss_dict.get('pose', 0.0)
    head_pose = loss_dict.get('head_pose', 0.0)
    act = loss_dict.get('activity', 0.0)
    psr = loss_dict.get('psr', 0.0)

    lv_det = loss_dict.get('log_var_det', 0.0)
    lv_pose = loss_dict.get('log_var_pose', 0.0)
    lv_act = loss_dict.get('log_var_act', 0.0)
    lv_psr = loss_dict.get('log_var_psr', 0.0)

    w_det = loss_dict.get('w_det', 0.0)
    w_pose = loss_dict.get('w_pose', 0.0)
    w_act = loss_dict.get('w_act', 0.0)
    w_psr = loss_dict.get('w_psr', 0.0)

    logger.debug(
        f'  [Loss breakdown] stage={stage} '
        f'det={det:.4f}(w={w_det:.3f},lv={lv_det:.2f}) '
        f'hp={head_pose:.4f}(w={w_pose:.3f},lv={lv_pose:.2f}) '
        f'act={act:.4f}(w={w_act:.3f},lv={lv_act:.2f}) '
        f'psr={psr:.4f}(w={w_psr:.3f},lv={lv_psr:.2f}) '
        f'total={total:.4f}'
    )


def _check_stage_transition(model: nn.Module, criterion, stage: int, epoch: int, backbone_type: str) -> None:
    """
    Doc 2 §B.2: Stage transition assertion.
    At stage start, log trainable param counts to catch freezing bugs.
    Also log Kendall log_sigma values so we can verify they are initialized
    correctly at each stage entry.
    """
    if not bool(getattr(C, 'LOG_STAGE_TRANSITION', True)):
        return

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_trainable = 0
    head_pose_trainable = 0
    act_trainable = 0
    psr_trainable = 0
    other_trainable = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(ln in name for ln in ['layer1', 'layer2', 'layer3', 'layer4', 'backbone', 'convnext']):
            backbone_trainable += p.numel()
        elif 'head_pose_head' in name or 'pose_head' in name:
            head_pose_trainable += p.numel()
        elif 'activity_head' in name:
            act_trainable += p.numel()
        elif 'psr_head' in name:
            psr_trainable += p.numel()
        else:
            other_trainable += p.numel()

    logger.info(
        f'[Epoch {epoch}] stage={stage} transition: '
        f'backbone={backbone_trainable/1e6:.2f}M  '
        f'hp={head_pose_trainable/1e4:.1f}K  '
        f'act={act_trainable/1e4:.1f}K  '
        f'psr={psr_trainable/1e3:.1f}K  '
        f'other={other_trainable/1e4:.1f}K  '
        f'frozen={frozen/1e6:.2f}M'
    )

    if criterion is not None:
        logger.info(
            f'  [Kendall log_sigma] '
            f'det={criterion.log_var_det.item():.3f}  '
            f'hp={criterion.log_var_pose.item():.3f}  '
            f'act={criterion.log_var_act.item():.3f}  '
            f'psr={criterion.log_var_psr.item():.3f}  '
            f'(sigma_det={np.exp(criterion.log_var_det.item()):.3f}  '
            f'sigma_hp={np.exp(criterion.log_var_pose.item()):.3f}  '
            f'sigma_act={np.exp(criterion.log_var_act.item()):.3f}  '
            f'sigma_psr={np.exp(criterion.log_var_psr.item()):.3f})'
        )

    expected_hp = 0 if stage == 1 else -1
    expected_act = 0 if stage in (1, 2) else -1
    expected_psr = 0 if stage in (1, 2) else -1

    if stage == 1:
        if head_pose_trainable > 1000:
            logger.warning(f'  WARNING: head_pose has {head_pose_trainable} trainable params in stage 1!')
        if act_trainable > 1000:
            logger.warning(f'  WARNING: activity has {act_trainable} trainable params in stage 1!')
        if psr_trainable > 1000:
            logger.warning(f'  WARNING: PSR has {psr_trainable} trainable params in stage 1!')


def _check_per_class_activity_sanity(
    val_metrics: Dict,
    epoch: int,
    split: str = 'val',
) -> None:
    """
    Doc 2 §B.4: Per-class activity sanity.
    Every 10 epochs of Stage 3, log top-5 hardest/easiest classes by per-class F1.
    """
    if epoch % 10 != 0 or epoch < int(getattr(C, 'STAGE1_EPOCHS', 5)) + int(getattr(C, 'STAGE2_EPOCHS', 10)):
        return

    per_class_acc = val_metrics.get('act_per_class_acc', [])
    if not per_class_acc:
        return

    from evaluate import report_per_class_accuracy
    report_per_class_accuracy(per_class_acc, C.ACT_CLASS_NAMES, k=5)


def _get_ema_decay(epoch: int) -> float:
    """
    Doc 2 A.2: EMA decay schedule.
    Stage 3 epoch 1 (overall epoch ~16): 0.999 — slow catch-up, stable init
    Stage 3 epoch 2 (overall epoch ~17): 0.9995 — medium
    Stage 3 epoch 3+ (overall epoch ~18+): 0.9999 — standard final decay
    """
    if epoch == 16:
        return 0.999
    elif epoch == 17:
        return 0.9995
    return 0.9999


def _check_psr_prevalence_sanity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> None:
    """
    Doc 2 §B.5: PSR component prevalence sanity.
    Log predicted vs GT prevalence per component. Should match within ±5%.
    """
    log_interval = int(getattr(C, 'LOG_PSR_PREVALENCE_EVERY', 10))
    if log_interval == 0 or epoch % log_interval != 0:
        return

    try:
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, targets in tqdm(loader, desc='PSR prevalence check', leave=False):
                images = _prepare_images(images, device, training=False)
                outputs = model(images)
                preds = torch.sigmoid(outputs['psr_logits'])
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets['psr_labels'].numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        pred_prevalence = all_preds.mean(axis=0)
        gt_prevalence = all_labels.mean(axis=0)

        for i in range(min(11, len(pred_prevalence))):
            delta = abs(pred_prevalence[i] - gt_prevalence[i])
            status = 'OK' if delta < 0.05 else 'WARN'
            logger.info(
                f'  [PSR comp={i:2d}] pred={pred_prevalence[i]:.3f} '
                f'gt={gt_prevalence[i]:.3f} delta={delta:.3f} [{status}]'
            )

        model.train()
    except Exception as e:
        logger.warning(f'PSR prevalence check failed: {e}')
        try:
            model.train()
        except Exception:
            pass


def _compare_raw_vs_ema(
    model: nn.Module,
    criterion: nn.Module,
    val_ds,
    device: torch.device,
    val_metrics: Dict,
    epoch: int,
    ckpt_dir: Path,
) -> None:
    """
    Doc 2 §B.6: Raw vs EMA val metric comparison.
    Rebuilds val loader internally (original is deleted after EMA val).
    Runs raw-model validation and compares to EMA metrics.
    Only meaningful in Stage 3 where EMA differs from raw.
    """
    try:
        logger.info('  [Stage 3] Running raw-model validation for comparison ...')
        raw_loader = _build_loader(
            val_ds,
            'val',
            CFG_VAL_BATCH_SIZE,
            2,
            prefetch=2,
        )
        raw_metrics = evaluate_all(
            model,
            criterion,
            raw_loader,
            device,
            max_batches=int(getattr(C, 'CFG_EVAL_MAX_BATCHES', 50)),
        )
        del raw_loader
        gc.collect()
        torch.cuda.empty_cache()
        _ema_delta = {
            'det_mAP50': val_metrics.get('det_mAP50', 0) - raw_metrics.get('det_mAP50', 0),
            'act_macro_f1': val_metrics.get('act_macro_f1', 0) - raw_metrics.get('act_macro_f1', 0),
            'psr_macro_f1': val_metrics.get('psr_macro_f1', 0) - raw_metrics.get('psr_macro_f1', 0),
        }
        logger.info(
            f'  [Stage 3] EMA vs Raw delta — '
            f'mAP50={_ema_delta["det_mAP50"]:+.4f}  '
            f'act_f1={_ema_delta["act_macro_f1"]:+.4f}  '
            f'psr_f1={_ema_delta["psr_macro_f1"]:+.4f}'
        )
    except Exception as exc:
        logger.warning(f'  [Stage 3] Raw-vs-EMA comparison failed: {exc}')


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


def _log_config_hash() -> str:
    """
    Log a SHA256 hash of the full config to enable reproducibility verification.
    Any change to config.py will change the hash, making it easy to detect
    which configuration produced which results.
    """
    import hashlib
    import json

    # Collect all uppercase config vars that affect training
    cfg_keys = [
        'SEED', 'EPOCHS', 'BATCH_SIZE', 'GRAD_ACCUM_STEPS', 'BASE_LR',
        'WEIGHT_DECAY', 'WARMUP_EPOCHS', 'GRAD_CLIP_NORM', 'MIXED_PRECISION',
        'BACKBONE', 'USE_KENDALL', 'USE_EMA', 'EMA_DECAY',
        'USE_MIXUP', 'MIXUP_ALPHA', 'CUTMIX_ALPHA',
        'USE_HAND_FILM', 'USE_HEADPOSE_FILM', 'USE_VIDEOMAE',
        'USE_TMA_CELL', 'USE_TEMPORAL_BANK', 'FEATURE_BANK_WINDOW',
        'TRAIN_DET', 'TRAIN_HEAD_POSE', 'TRAIN_ACT', 'TRAIN_PSR',
        'CUDNN_DETERMINISTIC', 'CUDNN_BENCHMARK', 'ALLOW_TF32',
        'NUM_WORKERS', 'PIN_MEMORY',
    ]
    cfg_values = {}
    for k in cfg_keys:
        v = getattr(C, k, None)
        if v is not None:
            cfg_values[k] = v

    cfg_str = json.dumps(cfg_values, sort_keys=True, default=str)
    h = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]
    logger.info(f'Config hash: {h}  (seed={cfg_values.get("SEED", "?")})')
    logger.info(f'Config hash details: {cfg_str[:200]}')
    return h


def main(args):
    seed_everything(C.SEED)

    log_dir = C.LOG_DIR;        log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = C.CHECKPOINT_DIR; ckpt_dir.mkdir(parents=True, exist_ok=True)

    _config_hash = _log_config_hash()

    # --- FLUSHING FILE HANDLER FIX (Bashara 2026-05-07) ---
    # Standard FileHandler buffers up to 8KB before flushing.
    # With the thread convoy, buffered log lines were NEVER reaching the file.
    # This custom handler flushes after EVERY write() call.
    class _FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()   # force write to disk immediately
        def write(self, msg):
            super().write(msg)
            self.flush()   # force write on every chunk

    _train_log_path = log_dir / 'train.log'
    _fh = _FlushingFileHandler(_train_log_path, mode='a')
    _fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    _root_logger = logging.getLogger()
    _root_logger.setLevel(logging.INFO)
    _root_logger.handlers.clear()   # remove any default handlers
    _root_logger.addHandler(_fh)
    _root_logger.addHandler(logging.StreamHandler())   # stdout still goes to train_log_YYYYMMDD.log
    # --------------------------------------------------------

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
    max_recordings_train = None
    max_recordings_val = None
    if C.DEBUG_MODE:
        max_recordings_train = C.DEBUG_MAX_VIDEOS
        max_recordings_val = C.DEBUG_MAX_VIDEOS
        logger.info(
            f'[train] Debug mode: max_recordings={max_recordings_train}, '
            f'VAL_EVERY={C.VAL_EVERY}'
        )

    # Subset ratio: use fraction of recordings for faster smoke testing
    subset_ratio = args.subset_ratio
    if subset_ratio < 1.0 and not C.DEBUG_MODE:
        import pandas as pd
        train_csv_path = C.TRAIN_CSV
        val_csv_path = C.VAL_CSV
        train_df = pd.read_csv(train_csv_path, header=None,
                               names=['rec_id', 'frame', 'action', 'f0', 'f1'])
        val_df = pd.read_csv(val_csv_path, header=None,
                              names=['rec_id', 'frame', 'action', 'f0', 'f1'])
        n_train_recs = train_df['rec_id'].nunique()
        n_val_recs = val_df['rec_id'].nunique()
        max_recordings_train = max(4, int(n_train_recs * subset_ratio))
        max_recordings_val = max(4, int(n_val_recs * subset_ratio))
        logger.info(
            f'[train] Subset ratio={subset_ratio}: '
            f'train {n_train_recs}→{max_recordings_train} recs, '
            f'val {n_val_recs}→{max_recordings_val} recs'
        )

    logger.info('Building datasets ...')
    train_ds = IndustRealMultiTaskDataset(
        split='train',
        img_size=C.IMG_SIZE,
        augment=True,
        seed=C.SEED,
        max_recordings=max_recordings_train,
    )
    val_ds = IndustRealMultiTaskDataset(
        split='val',
        img_size=C.IMG_SIZE,
        augment=False,
        seed=C.SEED,
        max_recordings=max_recordings_val,
    )

    train_prefetch = int(getattr(C, 'TRAIN_PREFETCH_FACTOR', 2)) if C.NUM_WORKERS > 0 else 1
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

    seq_train_loader = None
    if C.USE_PSR_SEQUENCE_MODE:
        seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 32)
        train_seq_ds = IndustRealMultiTaskDataset(
            split='train',
            img_size=C.IMG_SIZE,
            augment=True,
            seed=C.SEED,
            max_recordings=max_recordings_train,
            sequence_mode=True,
            sequence_length=seq_len,
        )
        seq_train_loader = _build_loader(
            train_seq_ds,
            'train_seq',
            1,
            train_workers,
            prefetch=train_prefetch,
        )
        logger.info(
            f'[train] PSR sequence mode: len={len(train_seq_ds):,} '
            f'samples ({seq_len} frames/window, stride=1)'
        )

    class_counts = train_ds.class_counts[:C.NUM_CLASSES_ACT - 1]  # ActivityHead outputs 74, not 75

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
    use_hand_film = bool(getattr(C, 'USE_HAND_FILM', True))
    use_headpose_film = bool(getattr(C, 'USE_HEADPOSE_FILM', False))
    use_videomae = bool(getattr(C, 'USE_VIDEOMAE', False))
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type=backbone_type,
        use_hand_film=use_hand_film,
        use_headpose_film=use_headpose_film,
        use_videomae=use_videomae,
        train_pose=CFG_TRAIN_HEAD_POSE,
    ).to(device)
    # Note: channels_last on model-level caused RuntimeError: required rank 4 tensor
    # (VideoMAE's EncoderDecoder has non-4D params like biases/LayerNorm that can't use CL).
    # Keeping input-level channels_last in _prepare_images which is safe.
    model = model.to(device)
    # Tag model with PSR sequence length so forward knows how to reshape
    model._seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 4) if C.USE_PSR_SEQUENCE_MODE else 1
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
    logger.info(f'Hand-FiLM (PoseFiLM): {use_hand_film}')
    logger.info(f'VideoMAE stream   : {use_videomae}')
    logger.info(f'Total parameters  : {params["total_all"]:,}')
    logger.info(f'Trainable params  : {params["total_trainable"]:,}')
    for k, v in params.items():
        if not k.startswith('total'):
            logger.info(f'  {k:15s}: {v:>10,}')

    criterion = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT - 1,  # ActivityHead outputs 74 classes, not 75 (75=74 AR + 1 NA prepended at dataset level)
        num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=CFG_TRAIN_DET,
        train_pose=CFG_TRAIN_HEAD_POSE,
        train_act=CFG_TRAIN_ACT,
        train_psr=CFG_TRAIN_PSR,
        use_kendall=CFG_USE_KENDALL,
    ).to(device)
    criterion.set_class_counts(class_counts)

    if hasattr(train_ds, 'psr_prevalence'):
        psr_prev = torch.from_numpy(train_ds.psr_prevalence)
        criterion.set_psr_class_counts(psr_prev)
        logger.info(
            f'PSR per-component prevalence: '
            f'{psr_prev.numpy().round(3).tolist()}'
        )

    backbone_params, head_params, bias_params = [], [], []
    loss_params = list(criterion.parameters())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(ln in name for ln in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']):
            backbone_params.append(param)
        elif 'bias' in name:
            # Doc 03: bias params get 0.3× head LR to prevent collapse from locked EMA
            bias_params.append(param)
        else:
            head_params.append(param)

    use_lion = bool(getattr(C, 'USE_LION', False))

    # Bias LR factor — 0.3× head LR prevents EMA-locked bias from collapsing
    BIAS_LR_FACTOR = 0.3
    backbone_lr = C.BASE_LR * 0.1
    head_lr = C.BASE_LR
    bias_lr = head_lr * BIAS_LR_FACTOR

    try:
        from lion_pytorch import Lion
    except ImportError:
        if C.USE_LION:
            raise RuntimeError(
                "USE_LION=True but 'lion-pytorch' is not installed. "
                "Install with: pip install lion-pytorch"
            )
        use_lion = False
    if use_lion:
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr * 0.3},
            {'params': head_params,      'lr': head_lr},
            {'params': bias_params,       'lr': bias_lr},
        ]
        if loss_params:
            param_groups.append({'params': loss_params, 'lr': head_lr})
        optimizer = Lion(param_groups, weight_decay=C.WEIGHT_DECAY * 3)
        logger.info('Optimizer: Lion (backbone=0.1×, heads=1×, bias=0.3×)')
    else:
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params,      'lr': head_lr},
            {'params': bias_params,       'lr': bias_lr},
        ]
        if loss_params:
            param_groups.append({'params': loss_params, 'lr': head_lr})
        optimizer = AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)
        logger.info('Optimizer: AdamW with differential LR (backbone=0.1×, heads=1×, bias=0.3×)')

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=C.WARMUP_EPOCHS)
    if bool(getattr(C, 'ONE_CYCLE_LR', False)):
        # Doc 2 E.2: OneCycleLR with super-convergence
        # High peak LR (5e-4) + aggressive cosine decay
        # Doc 01 B.3 fix: make max_lr dynamic based on actual num param groups
        n_groups = len(param_groups)
        backbone_lr_local = C.BASE_LR * 0.1
        head_lr_local = C.BASE_LR
        bias_lr_local = head_lr_local * BIAS_LR_FACTOR
        max_lr = (
            [backbone_lr_local * 0.5]  # backbone: lower LR for transformer backbone
            + [head_lr_local * 0.5] * (n_groups - 2)  # head params groups
            + [bias_lr_local * 0.5]   # bias params group
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

    scaler = amp.GradScaler('cuda', enabled=C.MIXED_PRECISION)

    start_epoch = 0
    best_metric = 0.0
    patience_counter = 0

    videomae_warmup_state = {
        'active': False,
        'param_group_idx': -1,
        'unfreeze_lr': 0.0,
        'epochs_remaining': 0,
    }

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # FIX: Accept 'model' (named checkpoints), 'model_state' (crash_recovery), 'model_state_dict' (crash_recovery v2)
        model_state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt.get('model')))
        load_result, skipped_keys = _load_model_compat(model, model_state)
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
        # FIX: Accept both 'ema_shadow' (named checkpoints) and 'ema_state' (crash_recovery)
        ema_key = 'ema_state' if 'ema_state' in ckpt else 'ema_shadow'
        if ema is not None and ema_key in ckpt and ckpt[ema_key]:
            ema.shadow.update({
                k: v.to(ema.device) if ema.device else v
                for k, v in ckpt[ema_key].items()
                if k in ema.shadow
            })
            logger.info('  EMA shadow weights restored from checkpoint.')

        try:
            # FIX: Accept 'optimizer' (named checkpoints), 'optimizer_state' (crash_recovery), 'optimizer_state_dict' (crash_recovery v2)
            opt_state = ckpt.get('optimizer_state_dict', ckpt.get('optimizer_state', ckpt.get('optimizer')))
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
                logger.info('  Optimizer state restored.')
            else:
                logger.warning('  No optimizer state found in checkpoint — re-initialized.')
        except (ValueError, AttributeError, KeyError) as e:
            logger.warning(
                f'  Could not restore optimizer state ({e}). '
                f'Re-initialized -- LR schedule continues.'
            )
        try:
            # FIX: Accept 'scheduler' (named checkpoint), 'lr_scheduler_state' (crash_recovery), 'scheduler_state_dict'
            sched_state = ckpt.get('scheduler_state_dict', ckpt.get('scheduler', ckpt.get('lr_scheduler_state', {})))
            scaler_state = ckpt.get('scaler_state_dict', ckpt.get('scaler_state', ckpt.get('scaler', {})))
            if sched_state:
                scheduler.load_state_dict(sched_state)
            if scaler_state:
                scaler.load_state_dict(scaler_state)
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(
                f'  Could not restore scheduler/scaler state ({e}). '
                f'Re-initialized -- LR schedule continues.'
            )
        start_epoch = ckpt['epoch'] + 1
        best_metric = float(ckpt.get('best_metric', 0.0))
        patience_counter = int(ckpt.get('patience_counter', 0))
        logger.info(f'Resumed from epoch {start_epoch}, best={best_metric:.4f}')

        # FIX: Mid-epoch resume — skip batches already processed in the partially-completed epoch.
        # batch > 0 means we crashed mid-epoch (not at epoch boundary). We need to resume from
        # the saved batch position in train_one_epoch instead of restarting the epoch from batch 0.
        resume_batch = int(ckpt.get('batch', 0))
        if resume_batch > 0:
            # Mid-epoch crash: keep same epoch, skip ahead to resume_batch.
            # start_epoch stays at ckpt['epoch'] (NOT +1) so we continue the same epoch.
            start_epoch = ckpt['epoch']
            _resume_batch_info = [resume_batch]
            logger.info(
                f'  Mid-epoch resume: epoch {start_epoch}, will skip {resume_batch} batches '
                f'(recreating DataLoader iterator to position {resume_batch})'
            )
        else:
            # Epoch-boundary or no batch info: normal resume from next epoch.
            start_epoch = ckpt['epoch'] + 1
            _resume_batch_info = [0]

        # FIX: Restore criterion (Kendall log_vars) from checkpoint if present
        if 'criterion' in ckpt and ckpt['criterion']:
            try:
                criterion_state = ckpt['criterion']
                criterion.log_var_det.data.copy_(criterion_state['log_var_det'].to(device))
                criterion.log_var_pose.data.copy_(criterion_state['log_var_pose'].to(device))
                criterion.log_var_act.data.copy_(criterion_state['log_var_act'].to(device))
                criterion.log_var_psr.data.copy_(criterion_state['log_var_psr'].to(device))
                logger.info(
                    f'  Restored Kendall log_vars from checkpoint: '
                    f'det={criterion.log_var_det.item():.3f}  '
                    f'head_pose={criterion.log_var_pose.item():.3f}  '
                    f'act={criterion.log_var_act.item():.3f}  '
                    f'psr={criterion.log_var_psr.item():.3f}'
                )
            except Exception as exc:
                logger.warning(f'  Could not restore criterion state ({exc})')

        # Reset Kendall log_var params only for early-epoch resumes.
        # This is intentional — early checkpoints haven't learned meaningful values yet.
        if start_epoch < C.WARMUP_EPOCHS:
            with torch.no_grad():
                criterion.log_var_det.fill_(0.0)
                criterion.log_var_pose.fill_(-1.0)
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
                f'head_pose={criterion.log_var_pose.item():.3f}  '
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

    # FIX: If resuming mid-epoch, pass resume_batch to train_one_epoch so it can
    # fast-forward the DataLoader iterator without re-computing forward passes.
    # _resume_batch_info is set in the resume block above.
    _resume_batch = _resume_batch_info[0] if '_resume_batch_info' in dir() and _resume_batch_info[0] > 0 else 0

    try:
        _train_start_epoch = _override_start_epoch if _override_start_epoch is not None else start_epoch
        for epoch in range(_train_start_epoch, C.EPOCHS):
            logger.info(f'\n--- Epoch {epoch}/{C.EPOCHS - 1} ---')
            criterion.set_epoch(epoch)

            # Doc 2 §B.2: Stage transition validation — log trainable param counts
            current_stage = get_stage(epoch)
            prev_stage = get_stage(epoch - 1) if epoch > 0 else current_stage
            if current_stage != prev_stage:
                _check_stage_transition(model, criterion, current_stage, epoch, C.BACKBONE)

                # Doc 2 §C.1: Kendall log_var reset at Stage 3 entry.
                # During Stage 2, log_var_act drifts (prec_act=0 so only lv_act is trained).
                # Fresh precision values at Stage 3 start prevent suboptimal activity/PSR
                # precision from carrying over. (Bug #9 reincarnation prevention.)
                if current_stage == 3:
                    criterion.log_var_act.data.fill_(0.0)
                    criterion.log_var_psr.data.fill_(0.0)
                    logger.info(
                        '[Epoch %d] Stage 3 entry: reset log_var_act=0, log_var_psr=0 '
                        '(were drifted from Stage 2)' % epoch
                    )

                # Doc 2 §C.2: Fresh EMA at Stage 3 entry.
                # EMA decay=0.999 tracks frozen params for ~700 steps before catching up.
                # Starting fresh EMA at Stage 3 ensures activity/PSR heads (random init)
                # are tracked from epoch 1 of their training, not from epoch 0.
                if current_stage == 3 and ema is not None:
                    from model import EMA as EMAClass
                    ema = EMAClass(model, decay=EMA_DECAY, device=device)
                    logger.info(
                        '[Epoch %d] Stage 3: reinitialized EMA from current model state' % epoch
                    )

            # Doc 01 §B.1: Unfreeze VideoMAE stream at configured epoch to let the
            # temporal stream adapt to IndustReal kinematics after backbone is warmed up.
            # This is a one-time event — subsequent epochs skip the check.
            unfreeze_epoch = int(getattr(C, 'VIDEOMAE_UNFREEZE_EPOCH', -1))
            if unfreeze_epoch >= 0 and epoch == unfreeze_epoch and C.USE_VIDEOMAE:
                if hasattr(model, 'videomae_stream'):
                    videomae_lr = float(getattr(C, 'VIDEOMAE_UNFREEZE_LR', 1e-5))
                    opt_params = model.videomae_stream.unfreeze(lr=videomae_lr)
                    optimizer.add_param_group(opt_params[0])
                    videomae_warmup_epochs = int(getattr(C, 'VIDEOMAE_WARMUP_EPOCHS', 3))
                    videomae_warmup_state['active'] = True
                    videomae_warmup_state['param_group_idx'] = len(optimizer.param_groups) - 1
                    videomae_warmup_state['unfreeze_lr'] = videomae_lr
                    videomae_warmup_state['epochs_remaining'] = videomae_warmup_epochs
                    logger.info(
                        '[Epoch %d] VideoMAE stream unfrozen at lr=%.0e, warmup=%d epochs'
                        % (epoch, videomae_lr, videomae_warmup_epochs)
                    )
                else:
                    logger.warning(
                        '[Epoch %d] USE_VIDEOMAE=True but model has no videomae_stream attribute'
                        % epoch
                    )

            if ema is not None:
                ema.set_decay(_get_ema_decay(epoch))

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
                        ckpt_dir,
                        accum_steps=train_accum_steps,
                        ema=ema,
                        seq_loader=seq_train_loader,
                        resume_batch=_resume_batch,
                        best_metric=best_metric,
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
            if videomae_warmup_state['active'] and videomae_warmup_state['epochs_remaining'] > 0:
                vid_idx = videomae_warmup_state['param_group_idx']
                vid_base = videomae_warmup_state['unfreeze_lr']
                warmup_total = int(getattr(C, 'VIDEOMAE_WARMUP_EPOCHS', 3))
                completed = warmup_total - videomae_warmup_state['epochs_remaining']
                alpha = (completed + 1) / warmup_total
                target_vid_lr = vid_base * (0.1 + 0.9 * alpha)
                optimizer.param_groups[vid_idx]['lr'] = target_vid_lr
                videomae_warmup_state['epochs_remaining'] -= 1
                logger.debug(
                    '[Epoch %d] VideoMAE warmup lr=%.0e (%d/%d)'
                    % (epoch, target_vid_lr, warmup_total - videomae_warmup_state['epochs_remaining'] - 1, warmup_total)
                )
            _check_ram(f'epoch_{epoch}_train')

            # [2% AUDIT] TRAIN_MAX_STEPS: break epoch loop if step limit reached
            if getattr(C, 'TRAIN_MAX_STEPS', 0) > 0:
                _batch_count = train_metrics.get('num_batches', 0)
                if not hasattr(C, '_global_step'):
                    C._global_step = 0
                C._global_step += _batch_count
                logger.info(f'  [2pct] global_step={C._global_step}/{C.TRAIN_MAX_STEPS}')
                if C._global_step >= C.TRAIN_MAX_STEPS:
                    logger.info(f'  [2pct] TRAIN_MAX_STEPS limit reached ({C._global_step}). Stopping training.')
                    break

            current_lr = optimizer.param_groups[1]['lr']
            ema_decay_str = ''
            if ema is not None:
                ema_decay_str = f'  ema_decay={ema.decay:.4f}'
            logger.info(
                f'Train: loss={train_metrics["total"]:.4f}  '
                f'det={train_metrics["det"]:.4f}  '
                f'pose={train_metrics["head_pose"]:.4f}  '
                f'act={train_metrics["activity"]:.4f}  '
                f'psr={train_metrics["psr"]:.4f}  '
                f'lr={current_lr:.2e}  '
                f'kd_d={train_metrics["log_var_det"]:.3f}  '
                f'kd_p={train_metrics["log_var_pose"]:.3f}  '
                f'kd_a={train_metrics["log_var_act"]:.3f}  '
                f'kd_r={train_metrics["log_var_psr"]:.3f}'
                f'{ema_decay_str}  '
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
                        model,
                        img_size=(C.IMG_HEIGHT, C.IMG_WIDTH),
                        device=device,
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

                # [FIX] Extra aggressive GPU memory cleanup before validation
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # [FIX] Additional sync and cache clear with small sleep to allow GPU to fully release memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.set_device(0)  # Reset CUDA context state
                    torch.cuda.empty_cache()

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
                        persistent=False,
                    )
                    try:
                        val_metrics = evaluate_all(
                            model,
                            criterion,
                            val_loader,
                            device,
                            max_batches=val_max_batches_rt,
                        )
                        _check_per_class_activity_sanity(val_metrics, epoch)
                    except Exception as exc:
                        is_cpu_enomem = 'Cannot allocate memory' in str(exc)
                        is_cuda_oom_v = _is_cuda_oom(exc)
                        # --- BASHARA 2026-05-22: Catch ALL validation exceptions (not just OOM).
                        # The empty_guard_failed RuntimeError from evaluate.py is NOT an OOM error,
                        # but it may be recoverable with reduced max_batches + worker reset.
                        # Only re-raise immediately if it's a truly unrecoverable error.
                        if is_cpu_enomem or is_cuda_oom_v:
                            pass  # Fall through to retry logic below
                        else:
                            # Non-OOM exception: check if it has "empty batch" or "act_preds" in message
                            # These are often recoverable if we reduce eval scope
                            exc_str = str(exc)
                            if 'empty' in exc_str.lower() and ('act_preds' in exc_str or 'batch' in exc_str.lower()):
                                logger.warning(
                                    f'Validation non-OOM exception (possibly recoverable): {exc_str[:200]}'
                                )
                                # Reduce scope and retry, but only once — don't loop 4x
                                if val_attempt == 1:
                                    val_batch_size_rt = max(1, val_batch_size_rt // 2)
                                    val_workers_rt = 0
                                    val_prefetch_rt = 1
                                    val_max_batches_rt = max(1, int(val_max_batches_rt) // 2)
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    logger.info(
                                        f'Validation retry (non-OOM, reducing scope): batch={val_batch_size_rt} '
                                        f'workers={val_workers_rt} prefetch={val_prefetch_rt} '
                                        f'max_batches={val_max_batches_rt}'
                                    )
                                    del val_loader
                                    continue
                            # All other non-OOM exceptions: re-raise immediately (no point retrying)
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
                        del val_loader
                        continue
                    finally:
                        del val_loader
                        gc.collect()
                        torch.cuda.empty_cache()

                if ema is not None:
                    ema.restore()
                    logger.info('  [EMA] Restored original weights after val')

                def _s(v, alt=0.0):
                    """Safe numeric: replace NaN/Inf with alt."""
                    if isinstance(v, float) and math.isfinite(v):
                        return v
                    return alt

                logger.info(
                    f'Val: loss={_s(val_metrics.get("loss")):.4f}  '
                    f'mAP50={_s(val_metrics.get("det_mAP50")):.4f}  '
                    f'mAP50_all={_s(val_metrics.get("det_mAP50_all_frames")):.4f}  '
                    f'act_clip={_s(val_metrics.get("act_accuracy")):.4f}  '
                    f'act_frame={_s(val_metrics.get("act_frame_accuracy")):.4f}  '
                    f'act_macro_f1={_s(val_metrics.get("act_macro_f1")):.4f}  '
                    f'head_pose_mae={_s(val_metrics.get("head_pose_MAE")):.4f}  '
                    f'psr_f1={_s(val_metrics.get("psr_overall_f1")):.4f}  '
                    f'psr_f1_tol5={_s(val_metrics.get("psr_overall_f1_at5")):.4f}'
                )

                if ema is not None and current_stage == 3:
                    _compare_raw_vs_ema(
                        model, criterion, val_ds, device, val_metrics, epoch, ckpt_dir
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
                        # FIX: Save criterion (Kendall log_vars) in best checkpoint
                        save_dict['criterion'] = {
                            'log_var_det': criterion.log_var_det.data.clone(),
                            'log_var_pose': criterion.log_var_pose.data.clone(),
                            'log_var_act': criterion.log_var_act.data.clone(),
                            'log_var_psr': criterion.log_var_psr.data.clone(),
                        }
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
                # FIX: Save criterion (Kendall log_vars) in latest checkpoint
                'criterion': {
                    'log_var_det': criterion.log_var_det.data.clone(),
                    'log_var_pose': criterion.log_var_pose.data.clone(),
                    'log_var_act': criterion.log_var_act.data.clone(),
                    'log_var_psr': criterion.log_var_psr.data.clone(),
                } if criterion is not None else {},
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
    # Stochastic Weight Averaging — SWA (Doc 2 E.3)
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
                    ckpt_dir,
                    accum_steps=train_accum_steps,
                    ema=None,
                    seq_loader=seq_train_loader,
                    resume_batch=0,
                    best_metric=best_metric,
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
    parser.add_argument(
        '--subset-ratio',
        type=float,
        default=getattr(C, 'SUBSET_RATIO', 1.0),
        help='Fraction of recordings to use (0.0-1.0). Default: 1.0 (all data). '
             'Use 0.1 for 10%% dataset smoke test.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Override DataLoader num_workers. Use 0 to disable multiprocessing '
             '(avoids shared-memory crashes on some systems).',
    )
    parser.add_argument(
        '--no-staged-training',
        action='store_true',
        help='Disable 3-stage progressive training. Activates ALL 5 heads from epoch 0 '
             '(equivalent to being in stage 3). Use for quick smoke tests.',
    )
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=None,
        help='Override starting epoch (e.g., 16 to jump to stage-3 equivalent). '
             'Does NOT load checkpoint — model starts fresh. '
             'Use with --no-staged-training to bypass stage gates.',
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
                logger.warning('[train] Config has no apply_preset — ignoring --preset')
        except Exception as exc:
            logger.warning(f'[train] Failed to apply preset {args.preset}: {exc}')

    if args.max_epochs is not None:
        C.EPOCHS = args.max_epochs
    if args.batch_size is not None:
        C.BATCH_SIZE = args.batch_size
        C.EFFECTIVE_BATCH = C.BATCH_SIZE * C.GRAD_ACCUM_STEPS
    if args.num_workers is not None:
        C.NUM_WORKERS = args.num_workers
        logger.info(f'[train] num_workers overridden to {args.num_workers}')

    # TRAIN_MAX_STEPS: limit total optimizer steps for quick runs (env var)
    _env_max_steps = int(os.environ.get('TRAIN_MAX_STEPS', '0'))
    if _env_max_steps > 0:
        C.TRAIN_MAX_STEPS = _env_max_steps
        logger.info(f'[train] TRAIN_MAX_STEPS={_env_max_steps} — will early-stop at this step count')

    _override_start_epoch = None  # set by --start-epoch
    if args.no_staged_training:
        C.STAGED_TRAINING = False
        logger.info('[train] STAGED_TRAINING=False — all 5 heads active from epoch 0')
    if args.start_epoch is not None:
        _override_start_epoch = args.start_epoch
        logger.info(f'[train] start_epoch override: {_override_start_epoch} (fresh init, no checkpoint)')

    if args.debug:
        C.DEBUG_MODE = True
        C.DEBUG_MAX_VIDEOS = 5
        C.VAL_EVERY = 999
        logger.info('[train] Debug mode enabled: small dataset, fast validation')

    if args.seed is not None:
        C.SEED = args.seed
        logger.info(f'[train] Seed overridden to {args.seed}')

    _refresh_runtime_cfg()

    main(args)


# Alias for compatibility with config references (Item 28)
_train_epoch = train_one_epoch
