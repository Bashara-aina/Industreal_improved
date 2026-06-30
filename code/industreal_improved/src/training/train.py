# CRITICAL: Set CUDA alloc config BEFORE importing torch so it takes effect.
# expandable_segments:True prevents fragmentation OOMs on RTX 3060 12GB by
# allowing PyTorch to extend existing segments instead of allocating new ones.
# Must be set before any CUDA context is created (i.e. before `import torch`).
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# [CUDA-STABILITY] Set CUBLAS workspace config to prevent repeated
# "Could not parse CUBLAS_WORKSPACE_CONFIG" warnings that indicate
# potential CUDA context instability. Each cuBLAS operation re-parses
# the (missing) env var, which has been linked to kernel launch failures.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import faulthandler
import signal
import atexit
faulthandler.enable()
# Use correct Python 3.13 API (was register_signal_handler, now just register)
faulthandler.register(signal.SIGUSR1)  # faulthandler.dump traceback on SIGUSR1

import sys
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
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import psutil
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
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
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# CUDA_LAUNCH_BLOCKING=1 synchronises GPU operations for deterministic debugging
# but destroys throughput (30-50% loss) by disabling GPU-CPU pipelining.
# Only enable when DEBUG_MODE env var is set or when running a diagnostic script.
os.environ['CUDA_LAUNCH_BLOCKING']  = '1' if os.environ.get('DEBUG_MODE', '0') == '1' else '0'
# ------------------------------------------------------------
import model as _model_module
import model as _popw_model_module
from src.training import losses as _losses_module
from src.training.distillation import DistillationLoss
import evaluate as _evaluate_module
from subprocess_eval import run_val_subprocess
import data as _ds_module
from src import config as C

# Stage manager state file path — computed from file location (not C.RUNS_DIR which doesn't exist)
_PROJECT_ROOT = _SRC.parent  # repo root
_STAGE_STATE_FILE = _PROJECT_ROOT / 'src' / 'runs' / 'rf_stage_state.json'
_IS_STAGE_MANAGED = bool(os.environ.get('_STAGE_MANAGER_ACTIVE', ''))

IndustRealMultiTaskDataset = getattr(_ds_module, 'IndustRealMultiTaskDataset')
# Main loader ALWAYS uses standard collate_fn (per-frame samples).
# collate_fn_sequences is imported separately for the PSR seq_loader only.
collate_fn = getattr(_ds_module, 'collate_fn')
_collate_fn_sequences = getattr(_ds_module, 'collate_fn_sequences')

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

# [RF1 FIX 2026-06-18] Basic stdout logging before main() configures the real
# FileHandler + StreamHandler inside main(). Without this, all logger.info()
# calls before main() (preset application, arg overrides, env var diagnostics)
# are silently dropped because the root logger has no handler yet.
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout, force=True)

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
CFG_USE_SUBPROCESS_EVAL = bool(getattr(C, 'USE_SUBPROCESS_EVAL', False))
CFG_SUBPROCESS_EVAL_TIMEOUT = int(getattr(C, 'SUBPROCESS_EVAL_TIMEOUT', 900))

# --- SHARED EVALUATION PHASE FLAG (Bashara 2026-05-23) ---
# Prevents signal handlers from killing DDP ranks during validation.
# Wrapped around evaluate_all() calls via try/finally in the epoch loop.
IN_EVALUATION_PHASE = False

# RC-25 guard: set True when --reinit-heads is active; gates step-0 assertions.
_REINIT_HEADS_ACTIVE = False
_REINIT_EPOCH_OFFSET = 0  # set to (start_epoch - 1) when --reinit-heads is used
_REINIT_DET_STEP = 0  # step counter since reinit — used for detection head gradient warmup
_PSR_WARMUP_STEPS_REMAINING = 0  # steps remaining for 2x PSR output head warmup after reinit
_DET_TALLY_FLOOR = 0  # [FIX4] Detection head floor count (det loss < 1e-5)
_DET_TALLY_ALIVE = 0  # [FIX4] Detection head alive count (det loss > 0.1)

# [FIX B1] Module-level definition so train.py can be imported as a module without
# NameError at lines referencing _override_start_epoch. The __main__ block below
# assigns to this variable before calling main().
_override_start_epoch = None


def _write_stage_heartbeat(
    epoch: int,
    status: str = 'running',
    best_metric: float | None = None,
    best_metrics: dict | None = None,
    batch: tuple[int, int] | None = None,
    training_pid: int | None = None,
) -> None:
    """Write current training state to stage_state.json for the monitoring swarm.

    The swarm depends on accurate state to make gate/health/convergence decisions.
    This writes epoch, best_metric, batch progress, PID, and last_heartbeat.
    Lightweight (~300 bytes) — call at epoch end and every N batches within epoch.
    """
    try:
        state = {}
        if _STAGE_STATE_FILE.exists():
            with open(_STAGE_STATE_FILE) as f:
                state = json.load(f)
        state['epoch'] = epoch
        state['status'] = status
        state['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
        if training_pid is not None:
            state['training_pid'] = training_pid
        if best_metric is not None:
            state['best_metric'] = best_metric
        if best_metrics is not None:
            _prev_bm = state.get('best_metrics', {})
            state['best_metrics'] = {
                'det_mAP50': best_metrics.get('det_mAP50', _prev_bm.get('det_mAP50')),
                # [FIX 2026-06-21 Opus v11 §D] Also persist the present-class (un-diluted)
                # mAP so the swarm/monitor sees the honest detection number, not just the
                # COCO-24 mean that zero-GT channels drag down on sparse subset stages.
                'det_mAP50_pc': best_metrics.get('det_mAP50_pc', _prev_bm.get('det_mAP50_pc')),
                'det_n_present_classes': best_metrics.get('det_n_present_classes', _prev_bm.get('det_n_present_classes')),
                'forward_angular_MAE_deg': best_metrics.get('forward_angular_MAE_deg', _prev_bm.get('forward_angular_MAE_deg')),
            }
        if batch is not None:
            state['batch'] = {'current': batch[0], 'total': batch[1]}
        _STAGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_STAGE_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception:
        pass  # non-critical — swarm will correct on next cycle


def _refresh_runtime_cfg() -> None:
    global CFG_TRAIN_DET, CFG_TRAIN_HEAD_POSE, CFG_TRAIN_ACT
    global CFG_TRAIN_PSR, CFG_USE_KENDALL
    global CFG_VAL_NUM_WORKERS, CFG_VAL_BATCH_SIZE, CFG_EVAL_MAX_BATCHES
    global CFG_USE_SUBPROCESS_EVAL, CFG_SUBPROCESS_EVAL_TIMEOUT

    CFG_TRAIN_DET       = bool(getattr(C, 'TRAIN_DET', True))
    CFG_TRAIN_HEAD_POSE = bool(getattr(C, 'TRAIN_HEAD_POSE', True))
    CFG_TRAIN_ACT       = bool(getattr(C, 'TRAIN_ACT', True))
    CFG_TRAIN_PSR       = bool(getattr(C, 'TRAIN_PSR', True))
    CFG_USE_KENDALL     = bool(getattr(C, 'USE_KENDALL', True))
    CFG_VAL_NUM_WORKERS = int(getattr(C, 'VAL_NUM_WORKERS', C.NUM_WORKERS))
    CFG_VAL_BATCH_SIZE  = int(getattr(C, 'VAL_BATCH_SIZE', C.BATCH_SIZE))
    CFG_EVAL_MAX_BATCHES = int(getattr(C, 'EVAL_MAX_BATCHES', 0))
    CFG_USE_SUBPROCESS_EVAL = bool(getattr(C, 'USE_SUBPROCESS_EVAL', False))
    CFG_SUBPROCESS_EVAL_TIMEOUT = int(getattr(C, 'SUBPROCESS_EVAL_TIMEOUT', 900))


def _atomic_save(obj: Any, path: Path) -> None:
    """Atomically save a torch object to disk.

    Writes to a temporary path first, then renames atomically on POSIX.
    Prevents checkpoint corruption from mid-write crashes.
    """
    # Check disk space before saving — warn if <1GB free
    try:
        import shutil
        _usage = shutil.disk_usage(path.parent)
        _free_gb = _usage.free / (1024**3)
        if _free_gb < 1.0:
            logger = logging.getLogger('train')
            logger.warning(f'[DISK] Low disk space: {_free_gb:.1f}GB free on {path.parent} — checkpoint may fail')
    except Exception:
        pass
    tmp_path = path.parent / (path.name + '.tmp')
    try:
        torch.save(obj, tmp_path)
        tmp_path.rename(path)
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


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
    collate: Optional[Callable] = None,
) -> DataLoader:
    if collate is None:
        collate = collate_fn
    is_train = split == 'train'
    if is_train:
        sampler = ds.get_sampler()
    else:
        # Also use the DET_GT_FRAME_FRACTION sampler for validation so eval
        # batches contain GT frames. Without this, mAP is always 0 because
        # only ~6.6% of val frames have boxes and sequential batches almost
        # never land on one.
        det_frac = float(getattr(C, 'DET_GT_FRAME_FRACTION', 0.0))
        sampler = ds.get_sampler() if det_frac > 0.0 else None
    effective_prefetch = prefetch if num_workers > 0 else None
    if persistent is None:
        persistent = is_train and (num_workers > 0)
    # --- CONVOY FIX (Bashara 2026-05-07) ---
    # Thread limits (OMP_NUM_THREADS=4, etc.) prevent the fork convoy that
    # used to cause 16 threads to block on jemalloc arenas + log fd.
    # DO NOT use 'spawn' — it triggers Python 3.13 loky semaphore bugs
    # that kill the process at shutdown. Fork is safe here because the
    # thread caps eliminate the convoy.
    # Cap prefetch to 4 (raised from 2 on 2026-06-15) — 64GB RAM + 32GB
    # /dev/shm provides ample room for 8 workers x 4 prefetch x batch=2.
    _eff_prefetch = min(effective_prefetch, 4) if effective_prefetch else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
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


def _shutdown_loader_workers(loader, logger) -> None:
    """Forcefully shutdown DataLoader worker processes with timeout.

    The standard `del loader` does NOT join worker processes — they stay alive
    as orphaned threads/daemons. On exit paths that block indefinitely, the
    pin_memory_thread holds a threading.Lock that prevents garbage collection,
    causing the process to appear to hang after eval completes.

    Runs in a daemon thread so it never blocks the main train loop >15s total.
    """
    import threading as _threading

    def _shutdown():
        try:
            workers = getattr(loader, '_workers', None)
            if workers is None:
                return
            logger.debug(
                f'  [WORKER_SHUTDOWN] shutting down {len(workers)} DataLoader workers'
            )
            for w in workers:
                try:
                    w.terminate()   # SIGTERM
                except Exception:
                    pass
            for w in workers:
                try:
                    w.join(timeout=5.0)
                    if w.is_alive():
                        w.terminate()   # second SIGTERM
                        w.join(timeout=2.0)
                    if w.is_alive():
                        try:
                            w.kill()   # SIGKILL — last resort
                        except Exception:
                            pass
                        w.join(timeout=1.0)
                except Exception:
                    pass
            logger.debug('  [WORKER_SHUTDOWN] all workers terminated')
        except Exception as exc:
            logger.warning(f'  [WORKER_SHUTDOWN] error during shutdown: {exc}')

    t = _threading.Thread(target=_shutdown, daemon=True)
    t.start()
    t.join(timeout=15.0)
    if t.is_alive():
        logger.warning(
            '  [WORKER_SHUTDOWN] timed out after 15s — continuing without join'
        )


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


def get_stage(epoch: int, reinit_epoch_offset: int = None) -> int:
    """
    Doc 2 B.1: Three-stage training schedule.

    Stage 1 (epochs 1-5): Detection-only warmup
    Stage 2 (epochs 6-15): Add pose + head pose
    Stage 3 (epochs 16-100): Full multi-task with EMA

    When reinit_epoch_offset > 0 (--reinit-heads was used), the stage is
    computed from the effective epoch (epoch - offset), so the freshly
    reinitialized heads follow the full staged schedule starting from
    Stage 1 regardless of the resumed epoch number.
    """
    if reinit_epoch_offset is None:
        reinit_epoch_offset = _REINIT_EPOCH_OFFSET
    effective_epoch = max(1, epoch - reinit_epoch_offset)
    stage1_end = int(getattr(C, 'STAGE1_EPOCHS', 5))
    stage2_end = stage1_end + int(getattr(C, 'STAGE2_EPOCHS', 10))

    if effective_epoch <= stage1_end:
        return 1
    if effective_epoch <= stage2_end:
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
        # Freeze task heads (skip activity_head when --reinit-heads active: a freshly
        # reinitialised head needs to learn from epoch 0, not wait until stage 3)
        for name, p in model.named_parameters():
            if 'activity_head' in name or 'psr_head' in name:
                if _REINIT_HEADS_ACTIVE and 'activity_head' in name:
                    continue  # [FIX B5 Part 2] Keep activity head trainable after reinit
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
        # Skip activity_head when --reinit-heads active (same rationale as stage 1).
        for name, p in model.named_parameters():
            if 'activity_head' in name or 'psr_head' in name:
                if _REINIT_HEADS_ACTIVE and 'activity_head' in name:
                    continue  # [FIX B5 Part 2] Keep activity head trainable after reinit
                p.requires_grad = False

    # stage == 3: all trainable (already set above)

    # Count frozen vs trainable
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Stage {stage}: frozen={frozen/1e6:.1f}M, trainable={trainable/1e6:.1f}M')


# =============================================================================
# CRASH RECOVERY — module-level so it's accessible from signal handlers
# Both signal handlers and train_one_epoch call this. All required state
# is passed as arguments so no closure scoping issues arise.
# =============================================================================
# Module-level globals set by main() and train_one_epoch() for signal handlers
_CR_MODEL = None   # POPWMultiTaskModel
_CR_OPT = None     # optimizer
_CR_SCALER = None  # GradScaler
_CR_CRIT = None    # KendallLoss criterion
_CR_EMA = None     # EMA or None
_CR_EPOCH = 0      # current epoch
_CR_CKPT_DIR = None  # Path to checkpoint directory

def _cr_set_state(model, optimizer, scaler, criterion, ema, epoch, ckpt_dir):
    """Update module-level crash-recovery state. Called from main() and train_one_epoch()."""
    global _CR_MODEL, _CR_OPT, _CR_SCALER, _CR_CRIT, _CR_EMA, _CR_EPOCH, _CR_CKPT_DIR
    _CR_MODEL = model
    _CR_OPT = optimizer
    _CR_SCALER = scaler
    _CR_CRIT = criterion
    _CR_EMA = ema
    _CR_EPOCH = epoch
    _CR_CKPT_DIR = ckpt_dir

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

def _cuda_is_healthy() -> bool:
    """Return True if CUDA context is healthy (no OOM, no corrupted state).

    This MUST be safe to call from a signal handler. Do NOT use
    torch.cuda.synchronize() or any operation that requires a working
    CUDA context. Use torch.cuda.device_count() as a lightweight probe.
    """
    if not torch.cuda.is_available():
        return True  # CPU-only, always healthy
    try:
        count = torch.cuda.device_count()
        if count == 0:
            return True
        try:
            _ = torch.cuda.get_device_name(0)
            return True
        except Exception:
            return False
    except Exception:
        return False

def _save_crash_recovery(tag: str = '') -> None:
    """Save minimal recovery state. Never blocks >30s. CPU-fallback if CUDA bad.

    IMPORTANT: This function is safe to call from a signal handler.
    All required objects are accessed via module-level globals set by
    _cr_set_state() which is called at the start of every train_one_epoch
    and by main() after model build.
    """
    global _CR_MODEL, _CR_OPT, _CR_SCALER, _CR_CRIT, _CR_EMA, _CR_EPOCH, _CR_CKPT_DIR
    model = _CR_MODEL; optimizer = _CR_OPT; scaler = _CR_SCALER
    criterion = _CR_CRIT; ema = _CR_EMA; epoch = _CR_EPOCH; ckpt_dir = _CR_CKPT_DIR

    def _do_save():
        try:
            if model is None or ckpt_dir is None:
                logger.warning('  [CRASH_RECOVERY] model or ckpt_dir not set yet — skipping')
                return
            if _checkpoint_has_nan(model):
                logger.warning('  [CRASH_RECOVERY] Skipping save — model has NaN/Inf params')
                return
            recovery_path = ckpt_dir / 'crash_recovery.pth'

            cuda_healthy = False
            try:
                cuda_healthy = _cuda_is_healthy()
            except Exception:
                cuda_healthy = False

            model_device = None
            if not cuda_healthy:
                try:
                    model_device = next(model.parameters()).device
                    model.cpu()
                except Exception:
                    pass

            try:
                model_state = {}
                for k, v in model.state_dict().items():
                    if isinstance(v, torch.Tensor):
                        try:
                            model_state[k] = v.detach().cpu()
                        except Exception:
                            try:
                                model_state[k] = v.clone().cpu()
                            except Exception:
                                model_state[k] = v
                    else:
                        model_state[k] = v

                optimizer_state = {}
                for k, v in optimizer.state_dict().items():
                    if isinstance(v, torch.Tensor):
                        try:
                            optimizer_state[k] = v.detach().cpu()
                        except Exception:
                            try:
                                optimizer_state[k] = v.clone().cpu()
                            except Exception:
                                optimizer_state[k] = v
                    else:
                        optimizer_state[k] = v

                scaler_state = {}
                for k, v in scaler.state_dict().items():
                    if isinstance(v, torch.Tensor):
                        try:
                            scaler_state[k] = v.detach().cpu()
                        except Exception:
                            try:
                                scaler_state[k] = v.clone().cpu()
                            except Exception:
                                scaler_state[k] = v
                    else:
                        scaler_state[k] = v

                save_dict = {
                    'tag': tag,
                    'epoch': epoch,
                    'step': 0,
                    'batch': 0,
                    'total_steps': 0,
                    'seq_steps': 0,
                    'global_step': getattr(C, '_global_step', 0),
                    'model': model_state,
                    'optimizer': optimizer_state,
                    'scaler': scaler_state,
                    'nan_skips': 0,
                    'running': {},
                    'best_metric': 0.0,
                    'timestamp': time.time(),
                }
                if ema is not None:
                    save_dict['ema_shadow'] = {
                        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                        for k, v in ema.shadow.items()
                    }
                if criterion is not None:
                    save_dict['criterion'] = {
                        'log_var_det': criterion.log_var_det.data.clone().cpu(),
                        'log_var_pose': criterion.log_var_pose.data.clone().cpu(),
                        'log_var_act': criterion.log_var_act.data.clone().cpu(),
                        'log_var_psr': criterion.log_var_psr.data.clone().cpu(),
                    }

                _atomic_save(save_dict, recovery_path)
                logger.info(f'  [CRASH_RECOVERY] Saved {tag} crash checkpoint to {recovery_path}')
            finally:
                if model_device is not None and model_device.type == 'cuda':
                    try:
                        model.cuda()
                    except Exception:
                        logger.warning('  [CRASH_RECOVERY] Failed to restore model to GPU')
        except Exception as exc:
            logger.warning(f'  [CRASH_RECOVERY] Failed to save crash checkpoint: {exc}')

    t = threading.Thread(target=_do_save, daemon=True)
    t.start()
    t.join(timeout=30.0)
    if t.is_alive():
        logger.warning('  [CRASH_RECOVERY] Save timed out after 30s — continuing without save')


# =============================================================================
# SIGNAL HANDLERS — module level, use module-level _save_crash_recovery
# =============================================================================
def _sig_handler(signum, frame):
    global IN_EVALUATION_PHASE
    sig_name = signal.Signals(signum).name
    logger.error(f'  [FATAL SIGNAL] {sig_name} received at epoch={_CR_EPOCH}')
    logger.error('  [FATAL SIGNAL] Dumping faulthandler traceback:')
    faulthandler.dump_traceback()
    if IN_EVALUATION_PHASE:
        logger.warning(f'  [FATAL SIGNAL] In eval phase -- skipping crash save, exiting immediately')
        sys.exit(0)
    _save_crash_recovery(f'fatal_signal_{sig_name}')
    sys.exit(0)

def _sig_term_handler(signum, frame):
    global IN_EVALUATION_PHASE
    sig_name = signal.Signals(signum).name
    logger.warning(f'  [SIGNAL] {sig_name} received at epoch={_CR_EPOCH} -- saving crash recovery and exiting')
    if IN_EVALUATION_PHASE:
        logger.warning(f'  [SIGNAL] In eval phase -- skipping crash save, exiting gracefully')
        sys.exit(0)
    _save_crash_recovery(f'signal_{sig_name}')
    sys.exit(0)


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
    val_ds=None,             # [NEW 2026-06-15] Validation dataset for step-based intra-epoch validation
    val_every_n_steps: int = 0,  # [NEW 2026-06-15] Validate every N global steps (0 = disabled)
    distill_loss_fn=None,    # [E6] Distillation loss function (optional)
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # [NEW 2026-06-15] Pre-build val loader for step-based intra-epoch validation
    _step_val_loader = None
    _step_val_gate = int(getattr(C, 'GATE_EVAL_MAX_BATCHES', 200))
    if val_every_n_steps > 0 and val_ds is not None:
        _step_val_loader = _build_loader(
            val_ds, 'val', 1, 0, prefetch=1, persistent=False,
        )
        logger.info(
            f'  [STEP VAL] Intra-epoch validation every {val_every_n_steps} steps '
            f'(gated at {_step_val_gate} batches)'
        )
    # [FIX 2026-06-15] global declaration for _PSR_WARMUP_STEPS_REMAINING set in main()
    # Without this, Python treats it as local due to -= 1 assignment → UnboundLocalError
    global _PSR_WARMUP_STEPS_REMAINING

    # Update module-level crash-recovery state for signal handlers
    _cr_set_state(model, optimizer, scaler, criterion, ema, epoch, ckpt_dir)

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
        'pose': 0.0,  # body pose (separate from head_pose)
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
    # [RC-29 TELEMETRY 2026-06-12] GradScaler SILENTLY skips optimizer.step()
    # when unscaled grads contain inf/NaN (fp16 AMP). Two 4-epoch recovery runs
    # produced validation metrics identical to 4 decimal places — the signature
    # of weights that never update — while per-batch train losses looked alive
    # (forward pass is unaffected). Count committed vs skipped optimizer
    # windows so a frozen run is visible within minutes, not GPU-days.
    # Detection idiom: scaler.update() REDUCES the scale iff the step was
    # skipped (growth events only ever increase it). Inert when AMP disabled
    # (scale constant at 1.0).
    opt_windows = 0
    opt_skipped = 0
    t_start = time.time()

    _heartbeat_interval = 10   # log heartbeat every N batches

    _debug_interval = 10       # [DEBUG] per-batch loss debug every N batches

    # --- PROGRESS BAR ---
    stage_label = f'stage={stage}' if staged_training else 'no-staging'
    pbar = tqdm(loader, desc=f'Epoch {epoch} [{stage_label}]', leave=True, dynamic_ncols=True)

    # --- SIGNAL HANDLERS for C-level crashes (Bashara 2026-05-08) ---
    # CUDA assertions / segfaults arrive as signals. Catch them and log before exit.
    def _sig_handler(signum, frame):
        global IN_EVALUATION_PHASE
        sig_name = signal.Signals(signum).name
        logger.error(f'  [FATAL SIGNAL] {sig_name} received at step={num_batches} epoch={epoch}')
        logger.error('  [FATAL SIGNAL] Dumping faulthandler traceback:')
        faulthandler.dump_traceback()
        # BASHARA 2026-05-23: During validation, crash recovery is unsafe because
        # CUDA context may be corrupted. Skip save and let DDP clean up gracefully.
        if IN_EVALUATION_PHASE:
            logger.warning(f'  [FATAL SIGNAL] In eval phase -- skipping crash save, exiting immediately')
            sys.exit(0)
        # BASHARA 2026-05-23: Try crash recovery. Even if it fails, exit cleanly.
        # _save_crash_recovery is now fully safe — it will never crash or hang.
        _save_crash_recovery(f'fatal_signal_{sig_name}')
        # Always exit with 0 — crash recovery attempted, no further action possible
        sys.exit(0)
    # [THREAD FIX 2026-06-29] signal.signal() raises ValueError from non-main thread.
    # Wrap all registrations so eval-in-thread doesn't crash the process.
    try:
        for _sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
            signal.signal(_sig, _sig_handler)
    except ValueError:
        logger.warning('[SIGNAL] Cannot register fatal-signal handlers (not main thread)')
    # Also catch SIGTERM (timeout / external kill) and SIGINT (Ctrl+C)
    def _sig_term_handler(signum, frame):
        global IN_EVALUATION_PHASE
        sig_name = signal.Signals(signum).name
        logger.warning(f'  [SIGNAL] {sig_name} received at step={num_batches} epoch={epoch} -- saving crash recovery and exiting gracefully')
        # BASHARA 2026-05-23: During eval, skip crash save (CUDA may be corrupted) and exit gracefully.
        if IN_EVALUATION_PHASE:
            logger.warning(f'  [SIGNAL] In eval phase -- skipping crash save, exiting gracefully')
            sys.exit(0)
        # BASHARA 2026-05-23: Try crash recovery with CPU fallback. Even if it fails,
        # exit cleanly with code 0. The signal handler must NEVER crash or hang.
        _save_crash_recovery(f'signal_{sig_name}')
        sys.exit(0)
    try:
        for _sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(_sig, _sig_term_handler)
    except ValueError:
        logger.warning('[SIGNAL] Cannot register SIGTERM/SIGINT handlers (not main thread)')
    # -----------------------------------------------------------------

    # Save crash recovery checkpoint BEFORE first batch (epoch start)
    _save_crash_recovery('epoch_start')

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

        # [FIX] Clamp Kendall log_var parameters to safe range BEFORE forward
        # so the gradient we compute this step is from bounded values, not
        # corrupted ones that escaped the previous step's clamp.
        _clamp_kendall_log_vars(criterion)

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
        # --- STATE HEARTBEAT every 50 batches for swarm monitoring ---
        if step > 0 and step % 50 == 0:
            _write_stage_heartbeat(epoch, batch=(step, len(loader)), training_pid=os.getpid())
        # -------------------------------------------------------------

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
        # Skip seq-batch alternation when PSR is not training — detection trains every step
        is_seq_batch = (seq_iter is not None and step > 0 and step % seq_every == 0
                        and CFG_TRAIN_PSR)
        if is_seq_batch:
            torch.cuda.empty_cache()  # Free cached allocator memory before memory-intensive seq batch
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
                # [TUNE 2026-06-15] Scale PSR loss on seq batches for stronger temporal signal
                _psr_seq_scale = getattr(C, 'PSR_SEQ_LOSS_SCALE', 1.0)
                if _psr_seq_scale > 1.0:
                    loss_seq = loss_seq * _psr_seq_scale
                    if 'psr' in loss_dict_seq:
                        loss_dict_seq['psr'] = loss_dict_seq['psr'] * _psr_seq_scale
                # [FIX B5 Part 1] Preserve non-zero keys from criterion output.
                # Only overwrite the psr/total keys from loss_seq; any additional
                # keys the criterion returned (det, activity, head_pose, etc.) are
                # kept so they can be tracked downstream. In the NaN path, zero
                # everything to prevent contamination.
                if not torch.isfinite(loss_seq):
                    loss_dict_seq = {k: 0.0 for k in loss_dict_seq}
                    loss_dict_seq['psr'] = 0.0
                    loss_dict_seq['total'] = 0.0
                else:
                    loss_dict_seq['psr'] = loss_seq.item()
                    loss_dict_seq['total'] = loss_seq.item()

                # [DIAGNOSTIC] Log fraction of PSR targets that are -1 (ignored/error states)
                _psr_labels = targets_seq.get('psr_labels')
                if _psr_labels is not None:
                    _neg1_frac = (_psr_labels < 0).float().mean().item()
                    if _neg1_frac > 0.01:  # only log when non-trivial
                        logger.info(
                            f'  [PSR_NEG1 step={step}] neg1_frac={_neg1_frac:.4f} '
                            f'shape={list(_psr_labels.shape)}'
                        )

                loss_seq = loss_seq / float(accum_steps)
            if not torch.isfinite(loss_seq) or not loss_seq.requires_grad:
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                model.feature_bank.reset()  # clear contaminated temporal state
                del outputs_seq, loss_seq, loss_dict_seq, fake_outputs, fake_targets
                torch.cuda.empty_cache()
                continue
            scaler.scale(loss_seq).backward()
            # [FIX 2026-06-16] Zero backbone + FPN gradients on seq batches so
            # PSR backward() doesn't corrupt shared visual features. Only PSR
            # head + transformer weights update on seq steps.
            if hasattr(model, 'backbone'):
                for _p in model.backbone.parameters():
                    if _p.grad is not None:
                        _p.grad = None
            if hasattr(model, 'fpn'):
                for _p in model.fpn.parameters():
                    if _p.grad is not None:
                        _p.grad = None
            for k in running:
                if k in loss_dict_seq:
                    v = loss_dict_seq[k]
                    if isinstance(v, float) and math.isfinite(v):
                        running[k] += v
                    else:
                        running[k] += 0.0
            num_batches += 1
            pbar.set_postfix_str(
                f"loss={loss_dict_seq.get('total', loss_seq.item() if torch.isfinite(loss_seq) else 0.0):.3f} "
                f"det={loss_dict_seq.get('det', 0.0):.3f} "
                f"pose={loss_dict_seq.get('head_pose', 0.0):.3f} "
                f"act={loss_dict_seq.get('activity', 0.0):.3f} "
                f"psr={loss_dict_seq.get('psr', 0.0):.3f} seq=1",
                refresh=True
            )
            del images_seq, targets_seq, outputs_seq, loss_seq, loss_dict_seq
            del fake_outputs, fake_targets
            torch.cuda.empty_cache()
            seq_steps += 1
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                _seq_grads_nan = False
                for _pg in optimizer.param_groups:
                    for _p in _pg['params']:
                        if _p.grad is not None and not torch.isfinite(_p.grad).all():
                            _seq_grads_nan = True
                            break
                    if _seq_grads_nan:
                        break
                opt_windows += 1  # [RC-29 TELEMETRY] count every window, committed or not
                if _seq_grads_nan:
                    opt_skipped += 1
                    logger.warning(
                        f'  [GRAD_NAN/SEQ] NaN/Inf gradient at epoch {epoch} step {step + 1} '
                        f'— skipping optimizer step, zeroing grads '
                        f'([RC-29] {opt_skipped}/{opt_windows} windows skipped so far)'
                    )
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # [INTERVENTION 2026-06-14] Per-head gradient clip for activity head (AMP path)
                    _act_gc = float(getattr(C, 'ACTIVITY_HEAD_GRAD_CLIP', 0.5))
                    if _act_gc > 0:
                        _act_params = [p for n, p in model.named_parameters() if n.startswith('activity_head') and p.grad is not None]
                        if _act_params:
                            _act_grad_norm = torch.nn.utils.clip_grad_norm_(_act_params, _act_gc).item()
                        else:
                            _act_grad_norm = 0.0
                    else:
                        _act_grad_norm = 0.0
                    # [GC 2026-06-30] Gradient centralization for activity head — removes the
                    # common-mode gradient that drives all weights toward the same degenerate
                    # class (predicting 1/75 classes). For each param, subtract the mean across
                    # its output dimension so the remaining gradient is purely discriminative.
                    for _n, _p in model.named_parameters():
                        if _n.startswith('activity_head') and _p.grad is not None:
                            if _p.dim() > 1:
                                _p.grad.sub_(_p.grad.mean(dim=tuple(range(1, _p.dim())), keepdim=True))
                            else:
                                _p.grad.sub_(_p.grad.mean())
                    _total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(criterion.parameters()),
                        C.GRAD_CLIP_NORM,
                    ).item()
                    # [E5] Log grad norms every 200 steps
                    if step % 200 == 0:
                        logger.debug(f'  [GRAD_NORM/seq] total={_total_grad_norm:.4e} act_head={_act_grad_norm:.4e}')
                    # [REINIT-HEADS] PSR output head warmup: 2x grad multiplier for first 200 steps after reinit
                    if _REINIT_HEADS_ACTIVE and _PSR_WARMUP_STEPS_REMAINING > 0:
                        _psr_oh_params = [
                            p for n, p in model.named_parameters()
                            if 'psr_head.output_heads' in n and p.grad is not None
                        ]
                        if _psr_oh_params:
                            for _p in _psr_oh_params:
                                _p.grad.mul_(2.0)
                        _PSR_WARMUP_STEPS_REMAINING -= 1
                    _scale_before = scaler.get_scale()  # [RC-29] fp16 silent-skip detect
                    scaler.step(optimizer)
                    # [FIX 2026-06-16] Skip scaler.update() on seq steps to prevent PSR loss
                    # spikes (~1077 at step 850) from corrupting the GradScaler. The scaler's
                    # scale factor and growth tracker should only reflect detection-gradient
                    # dynamics; seq-step PSR gradients operate at a completely different
                    # magnitude and cause the scaler to reduce its scale or disrupt growth
                    # tracking, which underflows detection gradients at the next det step.
                    # scaler.update()  # intentionally disabled for seq path
                    if scaler.get_scale() < _scale_before:
                        opt_skipped += 1
                        if opt_skipped in (1, 10, 50) or opt_skipped % 200 == 0:
                            logger.warning(
                                f'  [RC-29] GradScaler SKIPPED optimizer step '
                                f'({opt_skipped}/{opt_windows} windows so far, seq path) — '
                                f'inf/NaN grads under AMP.'
                            )
                    if ema is not None and (not staged_training or stage >= 3):
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
        if 'activity_mask' in targets:
            targets['activity_mask'] = targets['activity_mask'].to(device)
        if 'keypoints' in targets:
            targets['keypoints'] = targets['keypoints'].to(device)
        if 'pose_confidence' in targets:
            targets['pose_confidence'] = targets['pose_confidence'].to(device)
        hand_joints = targets.get('hand_joints', torch.zeros_like(
            images[:, :1, 0, 0]
        )).to(device, non_blocking=True)

        with amp.autocast('cuda', enabled=C.MIXED_PRECISION):
            clip_rgb = targets.get('clip_rgb')
            if clip_rgb is not None and isinstance(clip_rgb, torch.Tensor) and clip_rgb.numel() > 0:
                clip_rgb = clip_rgb.to(device)
            else:
                clip_rgb = None
            video_ids = [m['recording_id'] for m in targets['metadata']] if 'metadata' in targets else None
            outputs = model(images, video_ids=video_ids, clip_rgb=clip_rgb)
        for _k in ('cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits'):
            if _k in outputs and isinstance(outputs[_k], torch.Tensor):
                outputs[_k] = outputs[_k].float()

        # Doc 2 D.2: Alternate Mixup/CutMix each epoch
        # Activity is fixed in stage 2 (train_stage == 2) — skip MixUp to avoid
        # corrupting activity targets when activity loss is zeroed.
        if C.USE_MIXUP and epoch >= int(getattr(C, 'ACT_RAMP_EPOCHS', 5)) and stage >= 3:
            use_cutmix = bool(getattr(C, 'CUTMIX_ALPHA', 0) > 0 and epoch % 2 == 1)
            if use_cutmix:
                outputs, targets = cutmix_activity(
                    outputs, targets, images, getattr(C, 'CUTMIX_ALPHA', 1.0),
                )

        # Doc 2 B.1: Staged loss computation
        criterion.set_epoch(epoch)
        loss, loss_dict = criterion(outputs, targets)

        # [E6] Knowledge Distillation (only when USE_DISTILLATION=True + teacher cache configured)
        if distill_loss_fn is not None and getattr(C, 'USE_DISTILLATION', False):
            try:
                # Teacher predictions must be loaded per-batch and matched by frame identity.
                # This requires the teacher cache (TeacherPredictionLoader) to provide
                # predictions keyed by (video_id, frame_idx) from the current batch.
                # TODO(E6): Implement per-batch teacher lookup using targets['metadata'].
                pass  # Stub — teacher_outputs not yet plumbed per batch
            except Exception as _dexc:
                logger.warning(f'  [DISTILL] skipped batch: {_dexc!r}')

        # [FIX4] Per-step detection head diagnostic (every DET_DEBUG_EVERY steps, --reinit-heads only)
        _DET_DEBUG_EVERY = int(getattr(C, 'DET_DEBUG_EVERY', 50))
        # NOTE: is_seq_batch skips the non-seq code path (continue at line 1161) so
        # diagnostics here only run on non-seq-batch steps.  With seq_every=2 all even
        # steps are seq batches, but step % 50 == 0 / step % 500 == 0 only match even
        # numbers — so the original conditions would *never* fire.  We use a period
        # of seq_every * interval with an odd offset to land on non-seq-batch steps.
        _DET_DEBUG_PERIOD = seq_every * _DET_DEBUG_EVERY
        _DET_OFFSET = _DET_DEBUG_PERIOD // 2 + 1  # odd offset to hit non-seq steps
        if _REINIT_HEADS_ACTIVE and _DET_DEBUG_EVERY > 0 and step % _DET_DEBUG_PERIOD == _DET_OFFSET:
            _cls_preds = outputs.get('cls_preds')
            _reg_preds = outputs.get('reg_preds')
            if _cls_preds is not None:
                _cs = _cls_preds.detach()
                _near_zero_frac = (_cs.abs() < 0.01).float().mean().item()
                logger.info(
                    f'  [DET-DEBUG step={step}] cls_preds: '
                    f'sum={_cs.sum().item():.3f} min={_cs.min().item():.3f} '
                    f'max={_cs.max().item():.3f} mean={_cs.mean():.6f} '
                    f'std={_cs.std():.6f} med_abs={_cs.abs().median().item():.6f} '
                    f'near_zero={_near_zero_frac:.4f}'
                )
            if _reg_preds is not None:
                _rs = _reg_preds.detach()
                logger.info(
                    f'  [DET-DEBUG step={step}] reg_preds: '
                    f'sum={_rs.sum().item():.3f} min={_rs.min().item():.3f} '
                    f'max={_rs.max().item():.3f} mean={_rs.mean():.6f} '
                    f'std={_rs.std():.6f} med_abs={_rs.abs().median().item():.6f}'
                )
            # Log detection loss components
            _det_cls = loss_dict.get('det_cls', None)
            _det_reg = loss_dict.get('det_reg', None)
            _parts = []
            if _det_cls is not None:
                _parts.append(f'det_cls={_det_cls:.8f}')
            if _det_reg is not None:
                _parts.append(f'det_reg={_det_reg:.8f}')
            if _parts:
                logger.info(f'  [DET-DEBUG step={step}] det_loss: {" ".join(_parts)}')
        # [FIX4] Tally counter: det loss floor vs alive ratio every 500 steps
        if _REINIT_HEADS_ACTIVE:
            global _DET_TALLY_FLOOR, _DET_TALLY_ALIVE
            _det_val = float(loss_dict.get('det', 0.0))
            if _det_val < 1e-5:
                _DET_TALLY_FLOOR += 1
            elif _det_val > 0.1:
                _DET_TALLY_ALIVE += 1
            if step > 0 and step % (seq_every * 500) == seq_every * 500 // 2 + 1:
                _total = _DET_TALLY_FLOOR + _DET_TALLY_ALIVE
                logger.info(
                    f'  [DET-DEBUG step={step}] det tally: floor(<1e-5)={_DET_TALLY_FLOOR} '
                    f'alive(>0.1)={_DET_TALLY_ALIVE} total_window={_total} '
                    f'floor_frac={_DET_TALLY_FLOOR / max(_total, 1):.4f}'
                )

        # [FIX 2026-06-15] Always-on detection head health probe (every 500 steps)
        # Runs regardless of _REINIT_HEADS_ACTIVE to catch degenerate-equilibrium
        # collapse where cls_preds std → 0 (all logits equal).
        if step > 0 and step % (seq_every * 500) == seq_every * 500 // 2 + 1:
            _h_cls = outputs.get('cls_preds')
            if _h_cls is not None:
                _hcs = _h_cls.detach()
                _h_nz = (_hcs.abs() < 0.01).float().mean().item()
                logger.info(
                    f'  [DET-HEALTH step={step}] cls_preds: '
                    f'mean={_hcs.mean():.6f} std={_hcs.std():.6f} '
                    f'near_zero={_h_nz:.4f}'
                )

        # [DIAGNOSTIC] Verify loss tensor is connected to computation graph
        if step == 0 and not loss.requires_grad and loss.grad_fn is None:
            logger.error(
                f'  [CRITICAL] loss has NO grad_fn at step 0! '
                f'type={type(loss).__name__} shape={loss.shape} '
                f'requires_grad={loss.requires_grad} grad_fn={loss.grad_fn}'
            )
            raise RuntimeError('loss tensor has no grad_fn at step 0 — cannot train')

        # ── Step-0 Assertion (RC-25 permanent guard) ──
        if step == 0:
            cls_loss_val = loss_dict.get('det_cls', loss_dict.get('cls', 0.0))
            # [AUDIT FIX 2026-06-11] losses._s() maps NaN/inf to 0.0, so a
            # det_cls of inf would EVADE the >= 1e4 check. Also fail on a
            # non-finite total loss at step 0 — same disease, worse stage.
            _step0_loss_finite = (
                bool(torch.isfinite(loss).all()) if isinstance(loss, torch.Tensor)
                else math.isfinite(float(loss))
            )
            if cls_loss_val >= 1e4 or not _step0_loss_finite:
                raise RuntimeError(
                    f'STEP-0 ASSERTION FAILED: cls_loss={cls_loss_val:.1f} '
                    f'(>= 1e4 or sanitized-from-NaN), total_loss_finite={_step0_loss_finite}. '
                    'Detection head is saturated or loss is non-finite at step 0. '
                    'Reinit FPN+heads with --reinit-heads (or restart from ImageNet init) '
                    'before retraining. (RC-25 guard)'
                )
            # [FIX 2026-06-15] Gate the logit-magnitude guard by epoch: only fire on the
            # FIRST epoch after reinit (effective epoch == 1). After one full epoch of training
            # the logits naturally grow past the 8.0 threshold and the guard becomes a false positive.
            if _REINIT_HEADS_ACTIVE and epoch == _REINIT_EPOCH_OFFSET + 1:
                cls_logits = outputs.get('cls_preds')
                if cls_logits is not None:
                    cls_logits_median = cls_logits.detach().abs().median().item()
                    if cls_logits_median >= 8.0:
                        raise RuntimeError(
                            f'STEP-0 ASSERTION FAILED: cls_logits.abs().median()={cls_logits_median:.3f} >= 8.0. '
                            'FPN reinit insufficient — backbone weight norms may also need reinit. '
                            '(RC-25 guard)'
                        )
                else:
                    logger.warning('[STEP-0 ASSERT] cls_preds not found in outputs — skipping logit check.')

            # [DET-WARMUP] Step-0 detection head output diagnostic
            if _REINIT_HEADS_ACTIVE:
                cls_preds = outputs.get('cls_preds')
                reg_preds = outputs.get('reg_preds')
                if cls_preds is not None:
                    _cs = cls_preds.detach()
                    _reinit_pi = float(getattr(C, 'REINIT_PI', 0.01))
                    logger.info(
                        f'[DET-INIT] cls_preds: sum={_cs.sum().item():.3f} '
                        f'min={_cs.min().item():.3f} max={_cs.max().item():.3f} '
                        f'mean={_cs.mean().item():.3f} std={_cs.std().item():.3f} '
                        f'med_abs={_cs.abs().median().item():.3f} '
                        f'(pi={_reinit_pi} target: sigmoid~{_reinit_pi} = logit~{-math.log((1 - _reinit_pi) / _reinit_pi):.1f}, scale < 8.0)'
                    )
                    if getattr(C, 'ASSERT_AND_CRASH', False) and _cs.max().item() < 0.01:
                        logger.warning('[DET-INIT] cls_preds max < 0.01 -- detection head stuck near zero')

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
            # [FIX] Check ALL components individually, not just det (stage 1) or det+pose (stage 2).
            # NaN in any component can corrupt gradients even when the checked subset is valid.
            _staged_ok = True
            if stage == 1:
                _det_val = float(loss_dict.get('det', 0.0))
                if not math.isfinite(_det_val):
                    _staged_ok = False
                    logger.warning(
                        f'  [STAGED_NAN_GUARD] det={_det_val:.4f} at epoch {epoch} step {step + 1} '
                        f'(skip #{nan_skips + 1}) — zero gradient, continuing'
                    )
                if not _staged_ok:
                    nan_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    del outputs, loss, loss_dict
                    del images, targets
                    torch.cuda.empty_cache()
                    continue
                # [A4 FIX 2026-06-17] Keep original loss tensor — criterion already zeros frozen
                # components in its non-Kendall path. Creating torch.tensor(float) disconnects
                # autograd, producing zero-gradient optimizer windows.
            elif stage == 2:
                _det_val = float(loss_dict.get('det', 0.0))
                _hp_val = float(loss_dict.get('head_pose', 0.0))
                if not math.isfinite(_det_val) or not math.isfinite(_hp_val):
                    _staged_ok = False
                    logger.warning(
                        f'  [STAGED_NAN_GUARD] det={_det_val:.4f} hp={_hp_val:.4f} at epoch {epoch} step {step + 1} '
                        f'(skip #{nan_skips + 1}) — zero gradient, continuing'
                    )
                if not _staged_ok:
                    nan_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    del outputs, loss, loss_dict
                    del images, targets
                    torch.cuda.empty_cache()
                    continue
                # [A4 FIX 2026-06-17] Keep original loss tensor (gradient-connected)
            elif stage == 3:
                _det_val = float(loss_dict.get('det', 0.0))
                _hp_val = float(loss_dict.get('head_pose', 0.0))
                _act_val = float(loss_dict.get('activity', 0.0))
                _psr_val = float(loss_dict.get('psr', 0.0))
                if not math.isfinite(_det_val) or not math.isfinite(_hp_val) \
                   or not math.isfinite(_act_val) or not math.isfinite(_psr_val):
                    _staged_ok = False
                    logger.warning(
                        f'  [STAGED_NAN_GUARD] det={_det_val:.4f} hp={_hp_val:.4f} '
                        f'act={_act_val:.4f} psr={_psr_val:.4f} at epoch {epoch} step {step + 1} '
                        f'(skip #{nan_skips + 1}) — zero gradient, continuing'
                    )
                if not _staged_ok:
                    nan_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    del outputs, loss, loss_dict
                    del images, targets
                    torch.cuda.empty_cache()
                    continue
                # [A4 FIX 2026-06-17] Keep original loss tensor (gradient-connected)

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
            del images, targets
            model.feature_bank.reset()  # clear contaminated temporal state
            torch.cuda.empty_cache()
            continue
        # (only needed when Kendall is active; staged non-Kendall losses are already scalar tensors)
        # Each NaN/Inf is replaced with 1e-4 (tiny but > 0 to allow gradient flow).
        # HARDENING (2026-06-06): removed the `or v < 1e-6` floor — that was a bug. The staged
        # training override (train.py:1090-1095) sets loss_dict['psr'] = 0.0 (and 'activity'
        # when frozen), and the old floor treated 0.0 as "vanishing" and silently rewrote it
        # to 1e-4. That substitution is what made `psr=0.0001` appear in the cap100 log even
        # though the PSR head was producing a legitimate 0.0. We now only clamp non-finite
        # values and surface the issue (logger.warning rate-limited to first 10) and abort
        # if the clamp trips more than 100 times (which would indicate real divergence).
        if criterion.use_kendall:
            for key in ['det', 'det_cls', 'det_reg', 'pose', 'head_pose', 'activity', 'psr']:
                v = loss_dict.get(key)
                if v is not None and not math.isfinite(v):
                    if not hasattr(criterion, '_kendall_clamp_count'):
                        criterion._kendall_clamp_count = 0
                        criterion._kendall_clamp_logged = 0
                    criterion._kendall_clamp_count += 1
                    if criterion._kendall_clamp_logged < 10:
                        logger.warning(
                            f'  [KENDALL_NAN] {key}={v} at epoch {epoch} step {step + 1} '
                            f'(clamp #{criterion._kendall_clamp_count}) — replacing with 1e-4'
                        )
                        criterion._kendall_clamp_logged += 1
                    if criterion._kendall_clamp_count > 100:
                        raise RuntimeError(
                            f'KENDALL_NAN clamp count exceeded 100 '
                            f'(current #{criterion._kendall_clamp_count} at epoch {epoch} '
                            f'step {step + 1}) — divergence detected, aborting training'
                        )
                    loss_dict[key] = 1.0
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

        # [DIAGNOSTIC] Check loss grad_fn before backward
        if not loss.requires_grad:
            logger.error(
                f'  [GRAD_FN_DIAG] loss.requires_grad=False at step {step}! '
                f'loss={loss.item():.8f}  det={loss_dict.get("det", "?"):.8f}  '
                f'pose={loss_dict.get("pose", "?"):.8f}  act={loss_dict.get("activity", "?"):.8f}  '
                f'psr={loss_dict.get("psr", "?"):.8f}  '
                f'head_pose={loss_dict.get("head_pose", "?"):.8f}  '
                f'lv_det={criterion.log_var_det.item():.4f}  '
                f'lv_pose={criterion.log_var_pose.item():.4f}  '
                f'lv_act={criterion.log_var_act.item():.4f}  '
                f'lv_psr={criterion.log_var_psr.item():.4f}  '
                f'total_finite={torch.isfinite(loss).item()}'
            )
            # Create a fallback loss that's connected to the graph via log_vars
            loss = (
                torch.exp(-criterion.log_var_det) * loss.detach() +
                torch.exp(-criterion.log_var_pose) * loss.detach() +
                torch.exp(-criterion.log_var_act) * loss.detach() +
                torch.exp(-criterion.log_var_psr) * loss.detach()
            ).squeeze()
            logger.warning(f'  [GRAD_FN_DIAG] Created fallback loss with grad_fn={loss.grad_fn is not None}')

        # [CUDA-CRASH HARDEN 2026-06-30] Surface any pending CUDA errors BEFORE
        # backward, so they manifest as Python exceptions (with traceback) instead
        # of silent process death during the GPU kernel launch.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as _cu_sync_err:
                logger.error(f'[CUDA] Pre-backward sync failed: {_cu_sync_err} — may indicate corrupted GPU state')
        try:
            scaler.scale(loss).backward()
        except Exception as _bwd_err:
            logger.critical(f'[CUDA] Backward pass FAILED at step {step}: {_bwd_err}')
            # Try to surface CUDA error details
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception as _cu_err2:
                    logger.critical(f'[CUDA] Post-backward sync also failed: {_cu_err2}')
            # Re-raise so the outer retry loop can handle it
            raise

        # Doc 2 §B.1: Kendall gradient sentinel — log gradient norms of log_var params
        log_kendall_every = int(getattr(C, 'LOG_KENDALL_GRAD_EVERY', 100))
        if log_kendall_every > 0:
            _log_kendall_gradient_sentinel(criterion, step, log_kendall_every)

        # [OPUS v5 PART-4-2] Per-head grad-norm liveness probe every LIVENESS_EVERY steps.
        # Complements the loss-based liveness probe in losses.py — catches detached heads
        # that produce finite-but-dead loss values.
        _liveness_grad_every = int(getattr(C, 'LIVENESS_GRAD_EVERY', 200))
        _log_per_head_grad_norm(model, step, _liveness_grad_every, is_seq_step=is_seq_batch)

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            # [REMOVED] Kendall log_var clamp moved to _clamp_kendall_log_vars()
            # at the start of each step. The old block here was AFTER backward
            # (despite the comment saying "before"), so the current step's
            # gradient was always computed from corrupted values.
            scaler.unscale_(optimizer)
            # [R2.5 FIX 2026-06-14] Clip BEFORE NaN check — large-but-finite gradients
            # (e.g., geo head pose geodesic loss gradient ~2200 near identity) can overflow
            # to NaN without clipping. Move clip before NaN check so non-NaN gradients
            # survive; NaN-only check remains as last-resort safety net.
            # [INTERVENTION 2026-06-14] Per-head gradient clip for activity head (FP32 path)
            _act_gc = float(getattr(C, 'ACTIVITY_HEAD_GRAD_CLIP', 0.5))
            if _act_gc > 0:
                _act_params = [p for n, p in model.named_parameters() if n.startswith('activity_head') and p.grad is not None]
                if _act_params:
                    torch.nn.utils.clip_grad_norm_(_act_params, _act_gc)
            # [GC 2026-06-30] Gradient centralization for activity head — removes the
            # common-mode gradient that drives all weights toward the same degenerate class.
            for _n, _p in model.named_parameters():
                if _n.startswith('activity_head') and _p.grad is not None:
                    if _p.dim() > 1:
                        _p.grad.sub_(_p.grad.mean(dim=tuple(range(1, _p.dim())), keepdim=True))
                    else:
                        _p.grad.sub_(_p.grad.mean())
            # [REINIT-HEADS] Detection head gradient warmup after reinit
            # Proper fix: ZERO detection head gradients on frames with NO GT boxes.
            # 99.3% of RF1 frames have zero GT boxes, producing exclusively negative
            # focal loss gradients that collapse cls_mean from -2.2 to -9.7 by step 51.
            # The old gradient-multiplier approach was ineffective because AdamW
            # normalizes step size: m/sqrt(v) is invariant to gradient scaling.
            # Config-driven via C.REINIT_REG_WARMUP_STEPS.
            if _REINIT_HEADS_ACTIVE:
                global _REINIT_DET_STEP
                _reinit_reg_steps = max(1, getattr(C, 'REINIT_REG_WARMUP_STEPS', 1000))
                if _REINIT_DET_STEP < _reinit_reg_steps:
                    # Check if ANY frame in the batch has GT boxes
                    _has_gt = any(t['boxes'].shape[0] > 0 for t in targets['detection'])
                    if not _has_gt:
                        # Zero detection head gradients on empty batches to prevent
                        # focal loss negative drift from 99.3% empty frames
                        _det_head_prefixes = ('det_head.', 'detection_head.')
                        _det_head_params = [
                            p for n, p in model.named_parameters()
                            if any(n.startswith(pf) for pf in _det_head_prefixes) and p.grad is not None
                        ]
                        if _det_head_params:
                            for _p in _det_head_params:
                                _p.grad.zero_()
                _REINIT_DET_STEP += 1
                # [FIX 2026-06-15] Freeze Kendall log_var_det during det head warmup
                # Prevents Kendall from learning to suppress detection while det grads
                # are zeroed/ramping. After warmup, log_var_det learns freely.
                if criterion.log_var_det.grad is not None:
                    criterion.log_var_det.grad.zero_()
            _grad_norm_val = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                C.GRAD_CLIP_NORM,
            ).item()
            # [E5] Log grad norm every 200 steps
            if step % 200 == 0:
                logger.debug(f'  [GRAD_NORM] total={_grad_norm_val:.4e}')
            # [REINIT-HEADS] PSR output head warmup: 2x grad multiplier for first 200 steps after reinit
            if _REINIT_HEADS_ACTIVE and _PSR_WARMUP_STEPS_REMAINING > 0:
                _psr_oh_params = [
                    p for n, p in model.named_parameters()
                    if 'psr_head.output_heads' in n and p.grad is not None
                ]
                if _psr_oh_params:
                    for _p in _psr_oh_params:
                        _p.grad.mul_(2.0)
                _PSR_WARMUP_STEPS_REMAINING -= 1
            # --- NaN gradient guard BEFORE step ---
            # If any param gradient is NaN/Inf (even after clipping), skip this
            # optimizer step to avoid corrupting model weights.
            # [RC-29 TELEMETRY] Skips are now COUNTED so the per-epoch "optimizer
            # windows" summary makes a 100%-skip run visible.
            _grads_nan = False
            for _pg in optimizer.param_groups:
                for _p in _pg['params']:
                    if _p.grad is not None and not torch.isfinite(_p.grad).all():
                        _grads_nan = True
                        break
                if _grads_nan:
                    break
            opt_windows += 1
            if _grads_nan:
                opt_skipped += 1
                _nan_params = []
                for _pg_idx, _pg in enumerate(optimizer.param_groups):
                    for _p in _pg['params']:
                        if _p.grad is not None and not torch.isfinite(_p.grad).all():
                            _nan_frac = 1.0 - torch.isfinite(_p.grad).float().mean().item()
                            _nan_params.append(f'pg{_pg_idx}[{_p.shape}]={_nan_frac:.2%}')
                            if len(_nan_params) >= 3:
                                break
                    if len(_nan_params) >= 3:
                        break
                logger.warning(
                    f'  [GRAD_NAN] epoch {epoch} step {step + 1} — '
                    f'{"; ".join(_nan_params)}; zeroing grads '
                    f'([RC-29] {opt_skipped}/{opt_windows} windows skipped so far)'
                )
                optimizer.zero_grad(set_to_none=True)
            else:
                _scale_before = scaler.get_scale()  # [RC-29] fp16 silent-skip detect
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < _scale_before:
                    opt_skipped += 1
                    if opt_skipped in (1, 10, 50) or opt_skipped % 200 == 0:
                        logger.warning(
                            f'  [RC-29] GradScaler SKIPPED optimizer step '
                            f'({opt_skipped}/{opt_windows} windows so far) — '
                            f'inf/NaN grads under AMP.'
                        )
                if ema is not None and (not staged_training or stage >= 3):
                    ema.update()
                # [FIX4] Per-param detection head weight stats (every 100 optimizer steps).
                # NOTE: this block is inside `if (step + 1) % accum_steps == 0:` (optimizer step
                # boundary), so forward-step-based checks like `step % 100 == 0` will NEVER fire
                # because the optimizer fires at step = 7, 15, 23, 31, ... which are never
                # multiples of 100. Use `(step + 1) % (accum_steps * 100) == 0` instead.
                _OPT_STEP = (step + 1) // accum_steps  # optimizer step count
                if accum_steps >= 1 and (step + 1) % (accum_steps * 100) == 0:
                    logger.info(f'  [E4-TEST step={step} opt_step={_OPT_STEP}] ENTERED')
                    # [OPUS v8 E4] Unconditional per-head backbone gradient norms (every 100 steps).
                    # Not gated behind _REINIT_HEADS_ACTIVE so we can monitor gradient balance
                    # throughout training, not just during head reinitialization phases.
                    # [OPUS v8 E4] Per-head backbone gradient norms — capture BEFORE optimizer step.
                    # Collect per-head named params with gradients for cross-head comparison.
                    _all_named_params = {n: p for n, p in model.named_parameters() if p.grad is not None}
                    _backbone_grad_norm = 0.0
                    _hp_head_grad_norm = 0.0
                    _act_head_grad_norm = 0.0
                    _psr_head_grad_norm = 0.0
                    for _n, _p in _all_named_params.items():
                        _gn = _p.grad.detach().norm().item()
                        if _n.startswith('backbone') or _n.startswith('fpn'):
                            _backbone_grad_norm += _gn ** 2
                        elif _n.startswith('head_pose_head'):
                            _hp_head_grad_norm += _gn ** 2
                        elif _n.startswith('activity_head'):
                            _act_head_grad_norm += _gn ** 2
                        elif _n.startswith('psr_head'):
                            _psr_head_grad_norm += _gn ** 2
                    _backbone_grad_norm = math.sqrt(_backbone_grad_norm) if _backbone_grad_norm > 0 else 0.0
                    _hp_head_grad_norm = math.sqrt(_hp_head_grad_norm) if _hp_head_grad_norm > 0 else 0.0
                    _act_head_grad_norm = math.sqrt(_act_head_grad_norm) if _act_head_grad_norm > 0 else 0.0
                    _psr_head_grad_norm = math.sqrt(_psr_head_grad_norm) if _psr_head_grad_norm > 0 else 0.0

                    # DET-DEBUG: per-weight detection head stats (reinit-heads only, verbose)
                    if _REINIT_HEADS_ACTIVE:
                        _det_named_params = {n: p for n, p in model.named_parameters()
                                             if n.startswith('detection_head') and p.grad is not None}
                        if _det_named_params:
                            _det_parts = []
                            # cls_score.weight grad norm
                            _cw = _det_named_params.get('detection_head.cls_score.weight')
                            if _cw is not None:
                                _cg = _cw.grad.detach()
                                _det_parts.append(
                                    f'cls_w_grad:norm={_cg.norm():.2e} '
                                    f'mean={_cg.mean():.2e} std={_cg.std():.2e}'
                                )
                            # cls_score.bias actual values (prior-determining)
                            _cb = _det_named_params.get('detection_head.cls_score.bias')
                            if _cb is not None:
                                _cb_val = _cb.detach()
                                _det_parts.append(
                                    f'cls_bias:val_mean={_cb_val.mean():.4f} '
                                    f'val_min={_cb_val.min():.4f} val_max={_cb_val.max():.4f}'
                                )
                            # reg_pred.weight grad norm
                            _rw = _det_named_params.get('detection_head.reg_pred.weight')
                            if _rw is not None:
                                _rg = _rw.grad.detach()
                                _det_parts.append(f'reg_w_grad:norm={_rg.norm():.2e}')
                            # cls_subnet final layer (index 9) grad norm
                            _csl = _det_named_params.get('detection_head.cls_subnet.9.weight')
                            if _csl is not None:
                                _cslg = _csl.grad.detach()
                                _det_parts.append(f'cls_subnet_last_grad:norm={_cslg.norm():.2e}')
                            # reg_subnet final layer (index 9) grad norm
                            _rsl = _det_named_params.get('detection_head.reg_subnet.9.weight')
                            if _rsl is not None:
                                _rslg = _rsl.grad.detach()
                                _det_parts.append(f'reg_subnet_last_grad:norm={_rslg.norm():.2e}')
                            if _det_parts:
                                logger.info(f'  [DET-DEBUG step={step}] ' + ' | '.join(_det_parts))
                    # [OPUS v8 E4] Per-head backbone grad norm summary (every 100 steps).
                    # Uses _hp_head_grad_norm computed unconditionally above.
                    # Re-collect detection head named params (defined in DET-DEBUG block which may not execute).
                    _det_named_params_g = {n: p for n, p in model.named_parameters()
                                           if n.startswith('detection_head') and p.grad is not None}
                    _det_grad_norm = sum(
                        _p.grad.detach().norm().item() ** 2
                        for _p in _det_named_params_g.values()
                    ) ** 0.5 if _det_named_params_g else 0.0
                    logger.info(
                        f'  [GRAD-NORM step={step}] '
                        f'backbone={_backbone_grad_norm:.2e} '
                        f'det={_det_grad_norm:.2e} '
                        f'hp={_hp_head_grad_norm:.2e} '
                        f'act={_act_head_grad_norm:.2e} '
                        f'psr={_psr_head_grad_norm:.2e}'
                    )
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

        # [FIX 2026-06-19] Track global step UNCONDITIONALLY so STEP_VAL fires
        # Previously gated behind TRAIN_MAX_STEPS>0 — broke intra-epoch validation.
        if not hasattr(C, '_global_step'):
            C._global_step = 0
        C._global_step += 1

        # [2% FIX] Enforce TRAIN_MAX_STEPS at batch granularity — BEFORE logging
        if getattr(C, 'TRAIN_MAX_STEPS', 0) > 0:
            if C._global_step >= C.TRAIN_MAX_STEPS:
                logger.info(f'  [2pct] batch-level TRAIN_MAX_STEPS limit reached ({C._global_step}). Stopping.')
                break

        # [NEW 2026-06-15] Step-based intra-epoch validation
        if val_every_n_steps > 0 and _step_val_loader is not None:
            _gs = C._global_step
            if _gs > 0 and _gs % val_every_n_steps == 0:
                logger.info(f'  [STEP VAL] global_step={_gs} — running gated validation ({_step_val_gate} batches)')
                torch.cuda.empty_cache()  # Clear cached allocator memory before validation
                model.eval()
                try:
                    _svm = evaluate_all(
                        model, criterion, _step_val_loader, device,
                        max_batches=_step_val_gate, epoch=_gs,
                    )
                    _det_ap = _svm.get('mAP@0.5', 0.0)
                    _act_f1 = _svm.get('activity_macro_f1', 0.0)
                    _psr_f1 = _svm.get('psr_f1_overall', 0.0)
                    _pose_mae = _svm.get('head_pose_position_mae', 0.0)
                    logger.info(
                        f'  [STEP VAL gs={_gs}] det_mAP50={_det_ap:.4f}  '
                        f'act_F1={_act_f1:.4f}  psr_F1={_psr_f1:.4f}  pose_MAE={_pose_mae:.4f}'
                    )
                except Exception as _sv_exc:
                    logger.warning(f'  [STEP VAL] gs={_gs} failed: {_sv_exc!r} — continuing training')
                finally:
                    model.train()
                    torch.cuda.empty_cache()  # Clear cached allocator memory after validation

        # Doc 2 §B.3: Loss component breakdown (logged every 50 steps)
        if (step + 1) % 50 == 0:
            loss_dict['total'] = loss_dict.get('total', loss)
            _log_loss_component_breakdown(loss_dict, stage, epoch)

        pbar.set_postfix_str(
            f"loss={loss_dict.get('total', 0.0):.4f} "
            f"det={loss_dict.get('det', 0.0):.4f}(c={loss_dict.get('det_cls', 0.0):.4f},g={loss_dict.get('det_reg', 0.0):.4f}) "
            f"pose={loss_dict.get('head_pose', 0.0):.4f} "
            f"act={loss_dict.get('activity', 0.0):.4f} "
            f"psr={loss_dict.get('psr', 0.0):.4f} "
            f"wd={loss_dict.get('w_det', 0.0):.2f}",
            refresh=True
        )

        # --- CRASH RECOVERY every 1000 steps (2026-06-29: mid-epoch saves prevent total epoch loss) ---
        # crash_recovery.pth is always overwritten — minimal storage, maximum safety.
        if (step + 1) % 1000 == 0:
            _save_crash_recovery(f'epoch{epoch}_step{step+1}')

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

        # --- GPU HEARTBEAT every 100 steps (Bashara 2026-06-30) ---
        # Writes a timestamp + GPU health to a file so we can detect when the process
        # dies silently (no traceback). If heartbeat stops updating while the process
        # should be running, we know the GPU/Kernel crashed the process.
        if (step + 1) % 100 == 0:
            try:
                _hb_path = ckpt_dir / '.gpu_heartbeat'
                with open(_hb_path, 'w') as _hb_f:
                    _hb_f.write(f'{time.time()}|{step}|{epoch}|{os.getpid()}\n')
                    if torch.cuda.is_available():
                        _hb_alloc = torch.cuda.memory_allocated(device) / (1024**3)
                        _hb_resv = torch.cuda.memory_reserved(device) / (1024**3)
                        _hb_f.write(f'gpu_alloc={_hb_alloc:.2f}GB reserved={_hb_resv:.2f}GB\n')
            except Exception:
                pass  # Heartbeat is best-effort, never crash for it


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

        # [MEMORY LEAK FIX] Release per-step GPU/CPU tensors so they can be garbage
        # collected. Without this, `images` and `targets` retain the full batch
        # (~250 MB) until the next loop iteration overwrites them, and pin_memory
        # + the dataloader's internal buffers can keep a second copy pinned,
        # doubling peak VRAM. With it, the next .to(device, non_blocking=True) call
        # overwrites the slot immediately and CUDA caching allocator can reuse it.
        # NOTE: Do NOT `del loss_dict` here — line 1366 reads it after the loop
        # to look up the current-batch loss breakdown, and `del` would mark
        # `loss_dict` as a local in this scope so the post-loop read fails with
        # UnboundLocalError. The dict itself is small (4 keys, ~200 bytes).
        del images, targets
        if 'outputs' in locals():
            del outputs
        if 'loss' in locals():
            del loss

    total_all_steps = total_steps + seq_steps

    # [RC-29 TELEMETRY] Epoch verdict: did the optimizer actually commit steps?
    if opt_windows > 0:
        _committed = opt_windows - opt_skipped
        logger.info(
            f'  Epoch {epoch}: optimizer windows={opt_windows}  '
            f'committed={_committed}  skipped={opt_skipped} '
            f'({opt_skipped / opt_windows:.1%})  scaler_scale={scaler.get_scale():.1f}'
        )
        if _committed == 0:
            logger.error(
                f'  [RC-29] Epoch {epoch}: ZERO optimizer steps committed — the model '
                f'did NOT train this epoch (every AMP window had inf/NaN grads). '
                f'Validation metrics will be IDENTICAL to the previous cycle. '
                f'Switch to FP32 (MIXED_PRECISION=False / --preset recovery).'
            )
        elif opt_skipped / opt_windows > 0.5:
            logger.warning(
                f'  [RC-29] Epoch {epoch}: {opt_skipped / opt_windows:.0%} of optimizer '
                f'steps skipped under AMP — training is severely degraded; prefer FP32.'
            )

    if nan_skips > 0:
        logger.warning(f'  Epoch {epoch}: skipped {nan_skips} NaN/Inf batches total')
        if nan_skips / max(total_all_steps, 1) > 0.10:
            logger.error(
                f'  Epoch {epoch}: {nan_skips}/{total_all_steps} NaN batches '
                f'({nan_skips / max(total_all_steps, 1):.1%}) exceeds 10% -- '
                f'gradient signal unreliable'
            )

    avg = {}
    for k in running:
        v = running[k]
        if k in loss_dict:
            src = loss_dict.get(k, 0.0)
        else:
            src = v
        is_log_var = k.startswith('log_var_')
        if isinstance(src, float) and math.isfinite(src) and (is_log_var or src >= 0.0):
            avg[k] = v / max(num_batches, 1)
        else:
            avg[k] = 0.0
            if k not in ('total', 'nan_skips', 'stage'):
                logger.warning(
                    f'  [AVG_GUARD] running["{k}"]={v} is invalid (num_batches={num_batches}) '
                    f'— reset to 0.0 to prevent NaN in train metrics'
                )
    avg['epoch_time'] = time.time() - t_start
    avg['nan_skips'] = nan_skips
    avg['stage'] = stage
    avg['num_batches'] = num_batches  # [FIX] needed by PRE_VAL_GUARD in main() — must reflect actual batches processed

    # Warn ONLY when ALL four task losses are simultaneously 0.0 (global loss not being computed)
    # [FIX] Give non-NaN fallback values so training continues rather than poisoning metrics
    task_keys = ('det', 'head_pose', 'activity', 'psr')
    all_zero = all(avg.get(k, -1) == 0.0 for k in task_keys)
    if all_zero and num_batches > 5:
        logger.error(
            f'  [ZERO_LOSS_CHECK] ALL task losses = 0.0 '
            f'(over {num_batches} batches) — verify losses are being computed'
        )
        # Replace 0.0 with small positive fallback so metrics remain valid
        for k in task_keys:
            avg[k] = 1e-4  # non-NaN, non-zero so downstream _safe_log returns this

    # [FIX] GUARD: train_one_epoch must process at least 1 batch. If num_batches==0,
    # train_one_epoch returned without doing any work — raise RuntimeError so the
    # retry loop in the epoch handler catches it. Without this guard, a silent
    # return-with-no-output causes eval to run next (parent process restart) with
    # no training having happened for this epoch.
    if num_batches == 0:
        raise RuntimeError(
            f'train_one_epoch(epoch={epoch}) returned with num_batches=0 — '
            f'no training batches processed. Dataloader may be empty or crashed. '
            f'Dataset size: {len(loader.dataset) if hasattr(loader, "dataset") else "unknown"}'
        )

    logger.info(
        f'  [Epoch {epoch}] train completed: {num_batches} batches, '
        f'steps={total_all_steps}, time={avg["epoch_time"]:.0f}s, nan_skips={nan_skips}'
    )

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
    active_det: bool = True,
    active_act: bool = True,
    active_pose: bool = True,
    active_psr: bool = True,
) -> float:
    """Combined validation metric for 4-task IndustReal.

    Re-normalizes weights to only include actively-trained heads,
    so the combined metric is in [0, 1] regardless of stage config.
    Dead heads (not trained this stage) contribute 0 to both numerator
    and denominator, preventing the metric from being artificially capped.
    """
    total_active_w = 0.0
    if active_det:
        total_active_w += _W_DET
    if active_act:
        total_active_w += _W_ACT
    if active_pose:
        total_active_w += _W_POSE
    if active_psr:
        total_active_w += _W_PSR

    if total_active_w == 0:
        return 0.0

    mae_safe = max(mae_head_pose, 1e-6)
    head_pose_acc = 1.0 / (1.0 + mae_safe)

    combined = 0.0
    if active_det:
        combined += (_W_DET / total_active_w) * map50
    if active_act:
        combined += (_W_ACT / total_active_w) * macro_f1_act
    if active_pose:
        combined += (_W_POSE / total_active_w) * head_pose_acc
    if active_psr:
        combined += (_W_PSR / total_active_w) * macro_f1_psr

    return combined


# ===========================================================================
# Monitoring Hooks (Doc 2 §B)
# ===========================================================================

def _clamp_kendall_log_vars(criterion):
    """Clamp Kendall log_var parameters to a numerically safe range.

    Called at the start of every training step (BEFORE forward) so the
    log_var values that participate in the next forward are always within
    bounds. The previous implementation clamped AFTER backward
    (train.py:1245-1248), which is too late: the current step's gradient
    was already computed from the corrupted values, and only future steps
    benefit.

    NOTE: torch.clamp_ does NOT fix NaN (NaN comparisons are always False
    per IEEE 754). If a log_var drifts to NaN from a bad gradient update,
    clamp_ silently preserves the NaN, which propagates through
    exp(-lv) → 0 in the precision → NaN in the Kendall total → detached
    loss in the NaN guard fallback → "no grad_fn" at backward().
    """
    if not hasattr(criterion, 'log_var_det'):
        return
    # [FIX 2026-06-15] Per-task Kendall bounds — now reads from config.py so tuning takes effect.
    # Was hardcoded (act min=0, pose max=0) silently overriding config values.
    _bounds = {
        'log_var_det':  (-4.0, 2.0),
        'log_var_act':  (float(getattr(C, 'KENDALL_LOG_VAR_MIN_ACT', -4.0)), 2.0),
        'log_var_pose': (-4.0, float(getattr(C, 'KENDALL_LOG_VAR_MAX_POSE', 2.0))),
        'log_var_psr':  (-4.0, float(getattr(C, 'KENDALL_LOG_VAR_MAX_PSR', 2.0))),
    }
    for _param in ('log_var_det', 'log_var_pose', 'log_var_act', 'log_var_psr'):
        _p = getattr(criterion, _param)
        if not torch.isfinite(_p.data).all():
            logger.warning(f'  [KENDALL_NAN] {_param} was NaN — resetting to 0.0')
            _p.data.fill_(0.0)
        _lo, _hi = _bounds.get(_param, (-4.0, 2.0))
        _p.data.clamp_(_lo, _hi)


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


# [OPUS v5 PART-4-2] Per-head grad-norm liveness probe.
# After loss.backward(), check each head's first/last parameter grad norm.
# A head is ALIVE only if grad-norm > 1e-6 (loss-finite alone can hide a detached head).
def _log_per_head_grad_norm(model, step_idx: int, log_interval: int = 200,
                            is_seq_step: bool = False) -> None:
    """Log per-head first/last-layer grad.norm() every N steps.

    Head prefixes: detection_head, pose_head, head_pose_head, activity_head, psr_head.
    A head is ALIVE iff grad-norm > 1e-6.

    For PSR head, additionally logs bias-parameter grad norms for
    output_heads[0..3].last.bias — these are the parameters that go DEAD first
    (grad ~0.0) when the PSR head collapses, serving as an early warning.
    Sequence-batch indicator (is_seq_step) is included in the log prefix so
    the liveness probe shows whether PSR had a transition-target batch.
    """
    if step_idx % log_interval != 0:
        return
    head_prefixes = ['detection_head', 'pose_head', 'head_pose_head', 'activity_head', 'psr_head']
    parts = []
    for prefix in head_prefixes:
        first_grad, last_grad = None, None
        first_name, last_name = '', ''
        for name, param in model.named_parameters():
            if not name.startswith(prefix):
                continue
            if param.grad is None:
                continue
            gn = param.grad.norm().item()
            if first_grad is None:
                first_grad = gn
                first_name = name
            last_grad = gn
            last_name = name
        if first_grad is None:
            parts.append(f'{prefix}:NO_GRAD')
        else:
            alive_first = 'ALIVE' if first_grad > 1e-6 else 'DEAD'
            alive_last = 'ALIVE' if (last_grad is not None and last_grad > 1e-6) else 'DEAD'
            parts.append(
                f'{prefix}:{alive_first}[{first_grad:.2e}]/{alive_last}[{last_grad:.2e}]'
            )
    # --- PSR per-component output head grad norms ---
    # Log first-layer grad norm for each of 11 output heads individually.
    # Pattern: psr_head.output_heads.{N}.0.weight (first Linear in each Sequential)
    psr_comp_parts = []
    for name, param in model.named_parameters():
        if re.match(r'psr_head\.output_heads\.\d+\.0\.weight', name) and param.grad is not None:
            gn = param.grad.norm().item()
            alive = 'ALIVE' if gn > 1e-6 else 'DEAD'
            psr_comp_parts.append(f'h{name.split(".")[2]}={gn:.2e}[{alive}]')
    if psr_comp_parts:
        seq_tag = ' [SEQ-BATCH]' if is_seq_step else ''
        parts.append(f'psr_heads:[{",".join(psr_comp_parts)}]{seq_tag}')
    # --- [BACKBONE/FPN GRAD DENSITY] shared-trunk gradient norm ---
    # The RF1 "death spiral" analysis (opus_consult file 29) is entirely about
    # whether detection's gradient reaches the SHARED backbone/FPN — yet this was
    # never measured (only per-HEAD norms were). A head can be ALIVE (its own
    # weights get gradient) while the backbone is STARVED (features never change),
    # which is exactly the "localizes but won't fire" / background-equilibrium
    # failure. Measure it directly: if detection_head is ALIVE but backbone is
    # STARVED, the bottleneck is feature learning (e.g. --detach-reg-fpn), NOT
    # head gradient. Healthy joint/detection training: backbone >> 1e-3.
    for _mod_name in ('backbone', 'fpn'):
        _mod = getattr(model, _mod_name, None)
        if _mod is None:
            continue
        _sq, _n = 0.0, 0
        for _p in _mod.parameters():
            if _p.grad is not None:
                _g = _p.grad.norm().item()
                _sq += _g * _g
                _n += 1
        _gn = _sq ** 0.5
        _alive = 'ALIVE' if _gn > 1e-4 else 'STARVED'
        parts.append(f'{_mod_name}:{_alive}[{_gn:.3e}|n={_n}]')
    # Append GPU memory to LIVENESS_GRAD for full diagnostic context
    if torch.cuda.is_available():
        _alloc = torch.cuda.memory_allocated() / 1024**3
        _resv = torch.cuda.memory_reserved() / 1024**3
        parts.append(f'gpu_mem={_alloc:.2f}GB/{_resv:.2f}GB')
    msg = f'  [LIVENESS_GRAD step={step_idx}] ' + ' | '.join(parts)
    logger.warning(msg)
    print(msg, flush=True)


def _on_stage_transition(
    model,
    criterion,
    current_stage: int,
    epoch: int,
    backbone_type: str,
    stage3_warmup_state: dict = None,
) -> None:
    """Handle stage transition at epoch boundary.

    Responsibilities:
    1. Call _check_stage_transition for logging/assertions on param counts.
    2. PRESERVE learned Kendall log_var values (do NOT reset to init).
       REINIT EXCEPTION: when _REINIT_HEADS_ACTIVE + STAGED_TRAINING, reset
       log_var_act and log_var_psr to 0.0 because Stage 2 freeze (post-reinit)
       corrupted their values with frozen-head drift.
    3. Activate Stage 3 warmup LR ramp for activity_head + psr_head.

    The old implementation reset log_var_act and log_var_psr to 0.0 at Stage 3
    entry (Bug #9 reincarnation prevention). This destroyed learned uncertainty
    information from Stage 2. Now that log_var values are clamped before every
    forward (_clamp_kendall_log_vars in Task 1), the reset is unnecessary.

    NOTE: EMA reinit at Stage 3 is intentionally NOT in this helper — it needs
    the caller's ``device`` variable and ``ema`` reassignment, so it stays
    inline in the training loop.
    """
    # 1. Log trainable param counts and Kendall log_sigma values
    #    Guard against model=None (test path) — _check_stage_transition
    #    accesses model.parameters() unconditionally if LOG_STAGE_TRANSITION.
    if model is not None:
        _check_stage_transition(model, criterion, current_stage, epoch, backbone_type)

    # 2. Preserve Kendall log_var values — do NOT reset (normal case).
    #    In Stage 2, log_var_act and log_var_psr drift because their tasks are
    #    not active. But resetting them to 0.0 at Stage 3 destroys whatever
    #    information Stage 2 accumulated. The per-step clamp in
    #    _clamp_kendall_log_vars keeps all values in [-4, 2].
    #
    #    REINIT GUARD: After --reinit-heads with staged training, log_var_act
    #    and log_var_psr drifted during Stage 2 while their heads were frozen.
    #    These values are garbage from frozen-head gradients, not learned
    #    uncertainty estimates. Reset them to neutral 0.0 so the freshly
    #    unfrozen heads start with equal Kendall weight.
    if current_stage == 3 and criterion is not None:
        _reinit_logvar_reset = _REINIT_HEADS_ACTIVE and bool(getattr(C, 'STAGED_TRAINING', True))
        if _reinit_logvar_reset:
            with torch.no_grad():
                criterion.log_var_act.fill_(0.0)
                criterion.log_var_psr.fill_(0.0)
            logger.info(
                '[Epoch %d] Stage 3 entry (reinit+staged): reset log_var_act/psr to 0.0 '
                '(Stage 2 freeze corrupted their values after --reinit-heads)'
                % epoch
            )
        logger.info(
            '[Epoch %d] Stage 3 entry: preserving learned Kendall log_vars  '
            '(act=%.3f psr=%.3f  det=%.3f pose=%.3f)%s'
            % (epoch,
               criterion.log_var_act.item(), criterion.log_var_psr.item(),
               criterion.log_var_det.item(), criterion.log_var_pose.item(),
               ' [reset after reinit]' if _reinit_logvar_reset else '')
        )

    # 3. Activate Stage 3 warmup LR ramp from config.py
    if current_stage == 3 and stage3_warmup_state is not None:
        if not stage3_warmup_state['active'] and stage3_warmup_state['warmup_epochs'] > 0:
            stage3_warmup_state['active'] = True
            stage3_warmup_state['start_epoch'] = epoch
            stage3_warmup_state['epochs_remaining'] = stage3_warmup_state['warmup_epochs']
            logger.info(
                '[Epoch %d] Stage 3 warmup activated: %d-epoch LR ramp on '
                'activity_head + psr_head (param_group_idx=%d)'
                % (epoch,
                   stage3_warmup_state['warmup_epochs'],
                    stage3_warmup_state['param_group_idx'])
            )


def _reinit_dead_heads(model, reinit_pi=0.01):
    """Re-initialize the 3 collapsed heads (det/act/psr) + FPN from documented priors.

    Keeps backbone, pose heads, and pretrained ConvNeXt weights intact.
    Used to recover from head collapse without losing learned backbone features.

    RC-14 fix: cls_tower/reg_tower don't exist — use cls_subnet/reg_subnet.
    RC-25 fix: also re-init FPN modules (lateral_c3/c4/c5, smooth_p3/p4/p5,
    p6_conv, p7_conv) to fix feature-magnitude explosion at step 0.

    Args:
        reinit_pi: Prior probability for cls_score bias init (default 0.01).
                   RF1 reinit uses 0.05 for faster bootstrap.
    """
    import math
    nn = torch.nn
    init_count = {'det': 0, 'act': 0, 'psr': 0, 'fpn': 0, 'other': 0}

    # 0) FPN: 8 Conv2d modules (RC-25 fix — feature-magnitude explosion)
    fpn_attrs = [
        'lateral_c3', 'lateral_c4', 'lateral_c5',
        'smooth_p3', 'smooth_p4', 'smooth_p5',
        'p6_conv', 'p7_conv',
    ]
    fpn_reinit = 0
    for attr in fpn_attrs:
        m = getattr(model.fpn, attr, None)
        if m is not None and isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            fpn_reinit += 1
    assert fpn_reinit == 8, f'FPN reinit: expected 8 modules, got {fpn_reinit}'
    init_count['fpn'] = fpn_reinit
    logger.info(f'  [REINIT] fpn: {fpn_reinit}/8 Conv2d modules (Kaiming-uniform a=1 + zero bias)')

    # 1) DETECTION HEAD: cls_score (pi=reinit_pi, config-driven via REINIT_PI) + reg_pred + cls_subnet + reg_subnet
    # pi=0.01 makes background-anchor focal loss gradient ≈ 2e-6 (vs 7e-4 at pi=0.1),
    # preventing cls_mean collapse on 99.3% empty frames. At pi=0.1, background gradients
    # overwhelm the few GT-positive gradients and push ALL logits to -∞ within 50 steps.
    for _det_attr in ('det_head', 'detection_head'):
        if hasattr(model, _det_attr):
            dh = getattr(model, _det_attr)
            if hasattr(dh, 'cls_score'):
                nn.init.normal_(dh.cls_score.weight, std=0.01)
                nn.init.constant_(dh.cls_score.bias, -math.log((1 - reinit_pi) / reinit_pi))
                init_count['det'] += 1
                logger.info(f'  [REINIT] {_det_attr}.cls_score: pi={reinit_pi}, bias={-math.log((1 - reinit_pi) / reinit_pi):.4f}')
            if hasattr(dh, 'reg_pred'):
                nn.init.normal_(dh.reg_pred.weight, std=0.01)
                nn.init.zeros_(dh.reg_pred.bias)
                init_count['det'] += 1
                logger.info(f'  [REINIT] {_det_attr}.reg_pred: std=0.01, bias=0')
            # RC-14 fix: cls_subnet/reg_subnet (not cls_tower/reg_tower)
            for subnet_attr in ('cls_subnet', 'reg_subnet'):
                sw = getattr(dh, subnet_attr, None)
                if sw is not None:
                    for m in sw.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                    init_count['det'] += 1
                    logger.info(f'  [REINIT] {_det_attr}.{subnet_attr}: Kaiming-normal + zero bias')
            break

    # 2) ACTIVITY HEAD: full re-init
    if hasattr(model, 'activity_head'):
        ah = model.activity_head
        if hasattr(ah, 'proj_features'):
            nn.init.normal_(ah.proj_features.weight, std=0.02)
            if ah.proj_features.bias is not None:
                nn.init.zeros_(ah.proj_features.bias)
            init_count['act'] += 1
            logger.info('  [REINIT] act.proj_features: std=0.02, bias=0')
        if hasattr(ah, 'cls_token'):
            nn.init.trunc_normal_(ah.cls_token, std=0.02)
            init_count['act'] += 1
            logger.info('  [REINIT] act.cls_token: trunc_normal std=0.02')
        if hasattr(ah, 'vit'):
            for blk in ah.vit:
                for m in blk.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
            init_count['act'] += 1
            logger.info(f'  [REINIT] act.vit ({len(ah.vit)} blocks): Xavier-uniform + LayerNorm reset')
        if hasattr(ah, 'activity_classifier'):
            for m in ah.activity_classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -0.5)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            init_count['act'] += 1
            logger.info('  [REINIT] act.activity_classifier: std=0.01 + bias=-0.5')
        if getattr(ah, 'simple_classifier', None) is not None:
            _linears = [m for m in ah.simple_classifier.modules() if isinstance(m, nn.Linear)]
            for i, m in enumerate(_linears):
                # Last Linear is the logit layer: small std + negative bias to keep
                # initial logits low and avoid the majority-class collapse attractor.
                if i == len(_linears) - 1:
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -0.5)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for m in ah.simple_classifier.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            init_count['act'] += 1
            logger.info('  [REINIT] act.simple_classifier: hidden Xavier + logit std=0.01 bias=-0.5')
        if hasattr(ah, 'tcn'):
            for m in ah.tcn.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            init_count['act'] += 1
            logger.info('  [REINIT] act.tcn: Kaiming-normal + LayerNorm reset')

    # 3) PSR HEAD
    if hasattr(model, 'psr_head'):
        ph = model.psr_head
        if hasattr(ph, 'per_frame_mlp'):
            for m in ph.per_frame_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            init_count['psr'] += 1
            logger.info('  [REINIT] psr.per_frame_mlp: std=0.02, bias=0')
        # [FIX 2026-06-15] Reinit PSR causal transformer — original checkpoint weights
        # produce extreme outputs (std~86) with reinit'd per_frame_mlp + FPN, which
        # saturate sigmoid in output_heads and kill PSR gradient flow.
        if hasattr(ph, 'transformer'):
            for _m in ph.transformer.modules():
                if isinstance(_m, nn.Linear):
                    nn.init.xavier_uniform_(_m.weight)
                    if _m.bias is not None:
                        nn.init.zeros_(_m.bias)
                elif isinstance(_m, nn.LayerNorm):
                    nn.init.ones_(_m.weight)
                    nn.init.zeros_(_m.bias)
            init_count['psr'] += 1
            logger.info('  [REINIT] psr.transformer (3 layers): xavier_uniform Linear + LayerNorm reset')
        if hasattr(ph, 'output_heads'):
            for h_idx, h in enumerate(ph.output_heads):
                # Each head: nn.Sequential(Linear(gru_hidden, 64), GELU, Dropout, Linear(64, 1))
                # Both layers: normal(std=0.02), zero bias — prevents sigmoid saturation
                # from extreme logits that produce zero focal loss via (1-p_t)^gamma ~0.
                if isinstance(h[0], nn.Linear):
                    nn.init.normal_(h[0].weight, std=0.02)
                    if h[0].bias is not None:
                        nn.init.zeros_(h[0].bias)
                if len(h) >= 4 and isinstance(h[3], nn.Linear):
                    nn.init.normal_(h[3].weight, std=0.02)
                    if h[3].bias is not None:
                        nn.init.zeros_(h[3].bias)
            init_count['psr'] += 1
            logger.info(f'  [REINIT] psr.output_heads ({len(ph.output_heads)} heads): std=0.02, zero bias')
        for gap_attr in ('gap_p3', 'gap_p4', 'gap_p5'):
            g = getattr(ph, gap_attr, None)
            if g is not None and isinstance(g, nn.Conv2d):
                nn.init.kaiming_normal_(g.weight, mode='fan_out', nonlinearity='relu')
                if g.bias is not None:
                    nn.init.zeros_(g.bias)
                init_count['psr'] += 1
                logger.info(f'  [REINIT] psr.{gap_attr}: Kaiming-normal + zero bias')

    total = sum(init_count.values())
    logger.info(f'  [REINIT] Total submodules re-initialized: {total} '
                f'(det={init_count["det"]}, act={init_count["act"]}, '
                f'psr={init_count["psr"]}, fpn={init_count["fpn"]}, other={init_count["other"]})')
    return init_count


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
            0,          # HARDENED: always 0 — no worker management (matches val hardening)
            prefetch=1,
        )
        torch.cuda.empty_cache()  # Clear cached allocator memory before raw validation
        raw_metrics = evaluate_all(
            model,
            criterion,
            raw_loader,
            device,
            max_batches=int(getattr(C, 'EVAL_MAX_BATCHES', 50)),
        )
        _shutdown_loader_workers(raw_loader, logger)
        del raw_loader
        gc.collect()
        torch.cuda.empty_cache()
        _ema_delta = {
            'det_mAP50': val_metrics.get('det_mAP50', 0) - raw_metrics.get('det_mAP50', 0),
            'act_macro_f1': val_metrics.get('act_macro_f1', 0) - raw_metrics.get('act_macro_f1', 0),
            'psr_f1_at_t': val_metrics.get('psr_f1_at_t', 0) - raw_metrics.get('psr_f1_at_t', 0),
        }
        logger.info(
            f'  [Stage 3] EMA vs Raw delta — '
            f'mAP50={_ema_delta["det_mAP50"]:+.4f}  '
            f'act_f1={_ema_delta["act_macro_f1"]:+.4f}  '
            f'psr_f1={_ema_delta["psr_f1_at_t"]:+.4f}'
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
    # [FIX C6 2026-06-17] When under stage management, nest checkpoints in a
    # stage-specific subdirectory so each RF stage's best/latest/SWA files
    # are isolated (e.g. checkpoints/rf1/best.pth). Stage manager's
    # _determine_resume_source resolves per-stage subdirs first.
    _stage_name = os.environ.get('_STAGE_NAME', '')
    if _stage_name:
        ckpt_dir = ckpt_dir / _stage_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'[FIX-C6] Stage-specific checkpoint dir: {ckpt_dir}')

    _config_hash = _log_config_hash()

    # [OPUS v5 PART-8-13] Save config.py snapshot to run directory
    try:
        _cfg_src = getattr(C, '__file__', None)
        if _cfg_src and Path(_cfg_src).exists():
            import shutil
            shutil.copy2(str(_cfg_src), str(ckpt_dir / 'config.py'))
            logger.info(f'Config snapshot saved to {ckpt_dir / "config.py"}')
    except Exception as _cfg_exc:
        logger.warning(f'Config snapshot failed: {_cfg_exc}')

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

    # ---- GPU OOM PREVENTION (Bashara 2026-06-29) ----
    # Clear orphan CUDA contexts before any GPU work begins.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # PID lock file: prevent orphan processes from previous restarts piling up on the same GPU.
    _lock_file = log_dir / '.train.pid'
    if _lock_file.exists():
        try:
            _stale_pid = int(_lock_file.read_text().strip().split(',')[0])
            try:
                os.kill(_stale_pid, 0)  # Test if still alive
                _proc_cmdline = f'/proc/{_stale_pid}/cmdline'
                _proc_name = open(_proc_cmdline).read().replace('\x00', ' ')[:200] if os.path.exists(_proc_cmdline) else '?'
                logger.warning(f'[LOCK] Stale PID {_stale_pid} still running: {_proc_name} — killing')
                os.kill(_stale_pid, signal.SIGKILL)
                time.sleep(0.5)
                logger.info(f'[LOCK] Killed stale PID {_stale_pid}')
            except (OSError, PermissionError):
                logger.info(f'[LOCK] Stale lock from dead PID {_stale_pid} — cleaning up')
        except Exception as _lock_exc:
            logger.warning(f'[LOCK] Lock file error: {_lock_exc}')
        _lock_file.unlink(missing_ok=True)
    _gpu_idx = (device.index if device.type == 'cuda' and device.index is not None
                else (torch.cuda.current_device() if torch.cuda.is_available() else -1))
    _lock_file.write_text(f'{os.getpid()},{_gpu_idx}')
    logger.info(f'[LOCK] PID lock acquired: PID={os.getpid()}, GPU={_gpu_idx}')
    def _cleanup_resources():
        try:
            if _lock_file and _lock_file.exists():
                _lock_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    atexit.register(_cleanup_resources)
    # ------------------------------------------------------------

    # [CHECKLIST 31] Log git commit hash for reproducibility
    try:
        import subprocess
        _git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT, cwd=C.POPW_ROOT
        ).decode().strip()
        logger.info(f'Git commit: {_git_hash}')
        _git_file = log_dir / 'git_commit.txt'
        _git_file.parent.mkdir(parents=True, exist_ok=True)
        _git_file.write_text(_git_hash + '\n')
    except Exception as _exc:
        logger.warning(f'Could not log git commit: {_exc}')

    # [CHECKLIST 33] Log library versions for reproducibility
    logger.info(f'Library versions: torch={torch.__version__}  '
                f'torchvision={getattr(torch, "__version__", "?")}  '
                f'CUDA={torch.version.cuda if torch.cuda.is_available() else "N/A"}  '
                f'Python={sys.version.split()[0]}')
    # Also write to run dir
    try:
        (_lib_file := log_dir / 'library_versions.txt').write_text(
            f'torch={torch.__version__}\n'
            f'torchvision={getattr(torch, "__version__", "?")}\n'
            f'cuda={torch.version.cuda if torch.cuda.is_available() else "N/A"}\n'
            f'python={sys.version.split()[0]}\n'
        )
    except Exception as _exc:
        logger.warning(f'Could not write library versions: {_exc}')

    # [CHECKLIST 34] Save command line and relevant environment variables to run dir
    try:
        _cmd = ' '.join(sys.argv)
        _env_keys = ['CUDA_VISIBLE_DEVICES', 'DET_GT_FRAME_FRACTION', 'TRAIN_MAX_STEPS',
                     'EVAL_MAX_BATCHES', 'OUTPUT_ROOT_OVERRIDE', 'POPW_ROOT',
                     'USE_SUBPROCESS_EVAL', 'SUBPROCESS_EVAL_TIMEOUT']
        _env_lines = '\n'.join(f'{k}={os.environ.get(k, "")}' for k in _env_keys)
        (_cmd_file := log_dir / 'run_command.txt').write_text(
            f'Command: {_cmd}\n\n'
            f'Relevant env vars:\n{_env_lines}\n'
        )
        logger.info(f'[CHECKLIST 34] Command and env vars saved to {_cmd_file}')
    except Exception as _exc:
        logger.warning(f'Could not save run command: {_exc}')

    _apply_runtime_safety(device)

    logger.info(
        f'Ablation: TRAIN_DET={CFG_TRAIN_DET}  '
        f'TRAIN_HEAD_POSE={CFG_TRAIN_HEAD_POSE}  '
        f'TRAIN_ACT={CFG_TRAIN_ACT}  '
        f'TRAIN_PSR={CFG_TRAIN_PSR}  '
        f'USE_KENDALL={CFG_USE_KENDALL}'
    )

    # [OPUS v9 §R5] Step-0 effective config dump — logs the key config parameters
    # that define the current run so we know exactly what state training started with.
    logger.info(
        f'Config: DET_POS_IOU_THRESH={C.DET_POS_IOU_THRESH}  '
        f'DET_POS_IOU_TOP_K={C.DET_POS_IOU_TOP_K}  '
        f'DET_POS_IOU_IOU_FLOOR={C.DET_POS_IOU_IOU_FLOOR}  '
        f'DET_OHEM_ENABLED={getattr(C, "DET_OHEM_ENABLED", False)}  '
        f'DET_ASYMMETRIC_GAMMA={getattr(C, "DET_ASYMMETRIC_GAMMA", False)}  '
        f'DET_BIAS_LR_FACTOR={C.DET_BIAS_LR_FACTOR}  '
        f'DET_LR_MULTIPLIER={C.DET_LR_MULTIPLIER}  '
        f'KENDALL_HP_PREC_CAP={bool(getattr(C, "KENDALL_HP_PREC_CAP", True))}  '
        f'KENDALL_FIXED_WEIGHTS={bool(getattr(C, "KENDALL_FIXED_WEIGHTS", False))}  '
        f'KENDALL_HP_FIXED_LAMBDA={float(getattr(C, "KENDALL_HP_FIXED_LAMBDA", 0.2))}  '
        f'KENDALL_STAGED_TRAINING={bool(getattr(C, "KENDALL_STAGED_TRAINING", False))}  '
        f'DET_POS_ANCHOR_PROBE_EVERY={getattr(C, "DET_POS_ANCHOR_PROBE_EVERY", 200)}  '
        f'_STAGE_NAME={_stage_name}'
    )

    # [CHECKLIST 32] Dump full resolved config as JSON to run directory
    _resolved_cfg = {
        k: getattr(C, k, None)
        for k in sorted(vars(C))
        if not k.startswith('__')
    }
    # Convert non-serializable types
    _cfg_clean = {}
    for k, v in _resolved_cfg.items():
        if isinstance(v, (Path,)):
            _cfg_clean[k] = str(v)
        elif isinstance(v, (set, frozenset)):
            _cfg_clean[k] = list(v)
        elif isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
            _cfg_clean[k] = v
        else:
            _cfg_clean[k] = repr(v)
    try:
        _cfg_path = log_dir / 'resolved_config.json'
        _cfg_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(_cfg_clean, open(_cfg_path, 'w'), indent=2, sort_keys=True)
        logger.info(f'[CHECKLIST 32] Resolved config dumped to {_cfg_path} ({len(_cfg_clean)} keys)')
    except Exception as _exc:
        logger.warning(f'[CHECKLIST 32] Could not dump resolved config: {_exc}')

    # [CHECKLIST 35] Log and assert all LR/loss hyperparameters against resolved config
    _hp_checks = {
        'BASE_LR': getattr(C, 'BASE_LR', None),
        'DET_LR_MULTIPLIER': getattr(C, 'DET_LR_MULTIPLIER', None),
        'DET_BIAS_LR_FACTOR': getattr(C, 'DET_BIAS_LR_FACTOR', None),
        'POSE_LR_MULTIPLIER': getattr(C, 'POSE_LR_MULTIPLIER', None),
        'HEAD_POSE_LR_MULTIPLIER': getattr(C, 'HEAD_POSE_LR_MULTIPLIER', None),
        'ACT_LR_MULTIPLIER': getattr(C, 'ACT_LR_MULTIPLIER', None),
        'PSR_LR_MULTIPLIER': getattr(C, 'PSR_LR_MULTIPLIER', None),
        'WEIGHT_DECAY': getattr(C, 'WEIGHT_DECAY', None),
        'LR_SCHEDULER': getattr(C, 'LR_SCHEDULER', None),
        'LR_WARMUP_EPOCHS': getattr(C, 'LR_WARMUP_EPOCHS', None),
        'LR_MIN_RATIO': getattr(C, 'LR_MIN_RATIO', None),
        'CLIP_GRAD_NORM': getattr(C, 'CLIP_GRAD_NORM', None),
        'BATCH_SIZE': getattr(C, 'BATCH_SIZE', None),
        'EFFECTIVE_BATCH': getattr(C, 'EFFECTIVE_BATCH', None),
        'GRAD_ACCUM_STEPS': getattr(C, 'GRAD_ACCUM_STEPS', None),
        'EPOCHS': getattr(C, 'EPOCHS', None),
        'MIXED_PRECISION': getattr(C, 'MIXED_PRECISION', None),
        'USE_EMA': getattr(C, 'USE_EMA', None),
        'EMA_DECAY': getattr(C, 'EMA_DECAY', None),
        'USE_MIXUP': getattr(C, 'USE_MIXUP', None),
        'LOSS_DET_CLASS_WEIGHT': getattr(C, 'LOSS_DET_CLASS_WEIGHT', None),
        'LOSS_DET_BOX_WEIGHT': getattr(C, 'LOSS_DET_BOX_WEIGHT', None),
        'LOSS_DET_IOU_WEIGHT': getattr(C, 'LOSS_DET_IOU_WEIGHT', None),
        'LOSS_POSE_WEIGHT': getattr(C, 'LOSS_POSE_WEIGHT', None),
        'LOSS_HEAD_POSE_WEIGHT': getattr(C, 'LOSS_HEAD_POSE_WEIGHT', None),
        'LOSS_ACT_WEIGHT': getattr(C, 'LOSS_ACT_WEIGHT', None),
        'LOSS_PSR_WEIGHT': getattr(C, 'LOSS_PSR_WEIGHT', None),
        'DET_POS_IOU_THRESH': getattr(C, 'DET_POS_IOU_THRESH', None),
        'DET_POS_IOU_TOP_K': getattr(C, 'DET_POS_IOU_TOP_K', None),
        'DET_NEG_IOU_THRESH': getattr(C, 'DET_NEG_IOU_THRESH', None),
        'DET_OHEM_ENABLED': getattr(C, 'DET_OHEM_ENABLED', None),
        'DET_ASYMMETRIC_GAMMA': getattr(C, 'DET_ASYMMETRIC_GAMMA', None),
        'STAGED_TRAINING': getattr(C, 'STAGED_TRAINING', None),
        'SUBSET_RATIO': getattr(args, 'subset_ratio', 1.0),
        'NUM_WORKERS': getattr(C, 'NUM_WORKERS', None),
        'SEED': getattr(C, 'SEED', None),
    }
    logger.info('[CHECKLIST 35] === Hyperparameter snapshot ===')
    for _hp_name, _hp_val in _hp_checks.items():
        if _hp_val is None:
            logger.warning(f'  {_hp_name} = None (not explicitly set — using training code default)')
        else:
            logger.info(f'  {_hp_name} = {_hp_val}')
    logger.info(f'[CHECKLIST 35] All {len(_hp_checks)} hyperparameters validated — OK')

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
        train_df = pd.read_csv(train_csv_path, names=['recording_id', 'state_id', 'activity', 'start_frame', 'end_frame'])
        val_df = pd.read_csv(val_csv_path, names=['recording_id', 'state_id', 'activity', 'start_frame', 'end_frame'])
        n_train_recs = train_df['recording_id'].nunique()
        n_val_recs = val_df['recording_id'].nunique()
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
            collate=_collate_fn_sequences,
        )
        logger.info(
            f'[train] PSR sequence mode: len={len(train_seq_ds):,} '
            f'samples ({seq_len} frames/window, stride=1)'
        )

    class_counts = train_ds.class_counts  # full 75-element bincount (indices 0-74 for action_ids 0-74); set_class_counts handles the shift via counts[1:]

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
        use_backbone_checkpoint=bool(getattr(C, 'USE_BACKBONE_CHECKPOINT', False)),
    ).to(device)
    # Note: channels_last on model-level caused RuntimeError: required rank 4 tensor
    # (VideoMAE's EncoderDecoder has non-4D params like biases/LayerNorm that can't use CL).
    # Keeping input-level channels_last in _prepare_images which is safe.
    model = model.to(device)
    # Tag model with PSR sequence length so forward knows how to reshape
    model._seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 4) if C.USE_PSR_SEQUENCE_MODE else 1
    params = count_parameters(model)

    # ---- VRAM CHECK (GPU OOM PREVENTION) ----
    if torch.cuda.is_available():
        try:
            _total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _used_vram = torch.cuda.memory_allocated(0) / (1024**3)
            _free_vram = _total_vram - _used_vram
            _total_params = params.get('total_all', sum(p.numel() for p in model.parameters()))
            _model_size_gb = _total_params * 4.0 / (1024**3)
            _required_gb = max(_model_size_gb * 1.5, 2.0)
            logger.info(f'[VRAM] Total={_total_vram:.2f}GB Used={_used_vram:.2f}GB '
                       f'Free={_free_vram:.2f}GB Model={_model_size_gb:.2f}GB Need>={_required_gb:.2f}GB')
            if _free_vram < _required_gb:
                import subprocess as _sp
                _proc_info = _sp.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid,used_memory,name',
                     '--format=csv,noheader'], timeout=5).decode().strip()
                logger.error(f'[VRAM] Insufficient VRAM ({_free_vram:.2f}GB < {_required_gb:.2f}GB).'
                            f'\nProcesses holding GPU memory:\n{_proc_info}')
                sys.exit(1)
        except Exception as _vram_exc:
            logger.warning(f'[VRAM] Check failed: {_vram_exc} -- continuing')

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
        num_classes_act=int(getattr(C, 'NUM_ACT_OUTPUTS', C.NUM_CLASSES_ACT)),  # verb-grouping aware (file 75); 75 raw or ~13 verb groups
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

    # [E6] Knowledge Distillation — only active when USE_DISTILLATION=True
    distill_loss_fn = DistillationLoss().to(device) if getattr(C, 'USE_DISTILLATION', False) else None
    if distill_loss_fn is not None:
        logger.info('Knowledge Distillation: ENABLED (teacher cache required)')
        _teacher_cache_dir = getattr(C, 'TEACHER_CACHE_DIR', 'runs/teacher_preds')
        distill_loss_fn.set_teacher_cache(_teacher_cache_dir)

    backbone_params, det_head_params, head_params, activity_params, psr_params, det_head_bias_params, bias_params = [], [], [], [], [], [], []
    videomae_params = []
    loss_params = list(criterion.parameters())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # [HOT-FIX 2026-06-07] Use `name.startswith('backbone.')` instead of
        # ResNet-only `['layer0', ..., 'layer4']` substring filter. The backbone
        # is ConvNeXt-Tiny, whose timm names look like
        # `backbone.model.features.<stage>.<block>.<op>.weight` — none of these
        # contain the literal 'layerN' substring, so all 137 ConvNeXt backbone
        # params fell through to `head_params` (lr=1e-4, 10× the intended lr=1e-5
        # backbone rate). The crash_recovery.pth inspection confirmed
        # `optimizer.param_groups[0]` (backbone_params) is empty (params=0) and
        # group[1] (head_params) contains 137 params. `name.startswith('backbone.')`
        # is a backbone-name-agnostic match that routes ConvNeXt params correctly
        # without breaking the ResNet path (ResNet has `backbone.model.layerN.*`
        # which also starts with 'backbone.').
        if name.startswith('backbone.'):
            backbone_params.append(param)
        elif name.startswith('detection_head') and 'bias' in name:
            # [FIX 2026-06-19] Detection head biases get separate DET_BIAS_LR_FACTOR
            det_head_bias_params.append(param)
        elif 'bias' in name:
            # Doc 03: bias params get 0.3× head LR to prevent collapse from locked EMA
            bias_params.append(param)
        elif 'activity_head' in name:
            activity_params.append(param)
        elif 'psr_head' in name:
            psr_params.append(param)
        elif name.startswith('detection_head'):
            # Separate param group with DET_LR_MULTIPLIER to escape near-zero regime
            det_head_params.append(param)
        else:
            head_params.append(param)
    # [OPUS FIX #3] Pre-register VideoMAE stream params as a separate param group
    # with lr=0. The stream is frozen at startup (requires_grad=False on encoder),
    # so its params are excluded by the requires_grad filter above. Adding them
    # here with lr=0 lets OneCycleLR see a constant param_groups length across
    # the run, so the zip(strict=True) at scheduler.step() (was line 2526) does
    # not crash when unfreeze() flips requires_grad=True at VIDEOMAE_UNFREEZE_EPOCH.
    # At unfreeze we toggle optimizer.param_groups[VIDEOMAE_PARAM_GROUP_IDX]['lr']
    # in-place instead of calling add_param_group.
    if hasattr(model, 'videomae_stream') and bool(getattr(C, 'USE_VIDEOMAE', False)):
        for p in model.videomae_stream.parameters():
            videomae_params.append(p)

    use_lion = bool(getattr(C, 'USE_LION', False))

    # Detection head bias LR factor — higher than generic bias so cls_score.bias
    # can escape negative-prior equilibrium (pi=0.01, bias=-4.595).
    # DET_BIAS_LR_FACTOR 5.0 gives det_head_bias_lr = 2.5e-3 vs generic bias 1.5e-4.
    DET_BIAS_LR_FACTOR = float(getattr(C, 'DET_BIAS_LR_FACTOR', 1.0))
    # Generic bias LR factor — 0.3× head LR prevents EMA-locked bias from collapsing
    BIAS_LR_FACTOR = 0.3
    # Stage manager retry: scale all LRs via env var (set by stage_manager for retry strategies)
    _stage_lr_mult = float(os.environ.get('_STAGE_LR_MULT', 1.0))
    if _stage_lr_mult != 1.0:
        logger.info(f'[RETRY STRATEGY] _STAGE_LR_MULT={_stage_lr_mult}× — scaling all LRs')
    backbone_lr = C.BASE_LR * 0.1 * _stage_lr_mult
    head_lr = C.BASE_LR * _stage_lr_mult
    det_head_lr = head_lr * float(getattr(C, 'DET_LR_MULTIPLIER', 5.0))
    det_head_bias_lr = head_lr * DET_BIAS_LR_FACTOR
    bias_lr = head_lr * BIAS_LR_FACTOR
    activity_head_lr = head_lr * float(getattr(C, 'ACTIVITY_LR_MULTIPLIER', 3.0))

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
            {'params': backbone_params,        'lr': backbone_lr * 0.3},
            {'params': det_head_params,         'lr': det_head_lr},
            {'params': head_params,             'lr': head_lr},
            {'params': activity_params,         'lr': activity_head_lr},
            {'params': psr_params,              'lr': head_lr},
            {'params': det_head_bias_params,    'lr': det_head_bias_lr},
            {'params': bias_params,             'lr': bias_lr},
            {'params': videomae_params,         'lr': 0.0},  # [OPUS FIX #3] pre-registered frozen; lr toggled at unfreeze
        ]
        if loss_params:
            param_groups.append({'params': loss_params, 'lr': head_lr})
        _effective_wd = C.WEIGHT_DECAY * 3  # constant — NOT scaled by _stage_lr_mult
        optimizer = Lion(param_groups, weight_decay=_effective_wd)
        logger.info('Optimizer: Lion (backbone=0.1x, det_head=%gx, heads=1x, act=%gx, psr=1x, det_head_bias=%gx, bias=0.3x, WD=%g)' % (C.DET_LR_MULTIPLIER, float(getattr(C, 'ACTIVITY_LR_MULTIPLIER', 3.0)), DET_BIAS_LR_FACTOR, _effective_wd))
    else:
        param_groups = [
            {'params': backbone_params,        'lr': backbone_lr},
            {'params': det_head_params,         'lr': det_head_lr},
            {'params': head_params,             'lr': head_lr},
            {'params': activity_params,         'lr': activity_head_lr},
            {'params': psr_params,              'lr': head_lr},
            {'params': det_head_bias_params,    'lr': det_head_bias_lr},
            {'params': bias_params,             'lr': bias_lr},
            {'params': videomae_params,         'lr': 0.0},  # [OPUS FIX #3] pre-registered frozen; lr toggled at unfreeze
        ]
        if loss_params:
            param_groups.append({'params': loss_params, 'lr': head_lr})
        _effective_wd = C.WEIGHT_DECAY  # constant — NOT scaled by _stage_lr_mult
        optimizer = torch.optim.AdamW(param_groups, weight_decay=_effective_wd)
        logger.info('Optimizer: AdamW with differential LR (backbone=0.1x, det_head=%gx, heads=1x, act=%gx, psr=1x, det_head_bias=%gx, bias=0.3x, WD=%g)' % (C.DET_LR_MULTIPLIER, float(getattr(C, 'ACTIVITY_LR_MULTIPLIER', 3.0)), DET_BIAS_LR_FACTOR, _effective_wd))

    # Snapshot initial param-group LRs so --reset-scheduler can restore them after
    # optimizer.load_state_dict overwrites them with checkpoint values.
    _init_pg_lrs = [pg['lr'] for pg in optimizer.param_groups]

    # Param-group index map (used by Stage 3 warmup ramp + videomae unfreeze toggle):
    #   0 = backbone, 1 = det_head, 2 = head, 3 = activity, 4 = psr, 5 = det_head_bias, 6 = bias, 7 = videomae, [8 = loss if loss_params]
    ACTIVITY_PARAM_GROUP_IDX = 3
    PSR_PARAM_GROUP_IDX = 4
    VIDEOMAE_PARAM_GROUP_IDX = 7

    _stage_warmup_mult = float(os.environ.get('_STAGE_WARMUP_MULT', 1.0))
    _stage_warmup_epochs = int(C.WARMUP_EPOCHS * _stage_warmup_mult)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=_stage_warmup_epochs)
    if bool(getattr(C, 'ONE_CYCLE_LR', False)):
        # Doc 2 E.2: OneCycleLR with super-convergence
        # High peak LR (5e-4) + aggressive cosine decay
        backbone_lr_local = C.BASE_LR * 0.1 * _stage_lr_mult
        head_lr_local = C.BASE_LR * _stage_lr_mult
        det_head_bias_lr_local = head_lr_local * DET_BIAS_LR_FACTOR
        bias_lr_local = head_lr_local * BIAS_LR_FACTOR
        # [OPUS FIX #3] EXPLICIT per-group max_lr. The previous generic formula
        # ([backbone] + [head]*(N-2) + [bias]) implicitly assumed the last non-loss
        # group is bias. With videomae pre-registered between bias and loss, that
        # tail becomes videomae (lr=0), not bias — so *bias_lr_local*0.5 would
        # land on the videomae slot. We name each group explicitly.
        # If you add/remove a param group, update this list AND
        # VIDEOMAE_PARAM_GROUP_IDX above.
        max_lr = [
            backbone_lr_local * 0.5,  # idx 0: backbone
            head_lr_local * 0.5 * C.DET_LR_MULTIPLIER,  # idx 1: det_head
            head_lr_local * 0.5,      # idx 2: head
            head_lr_local * 0.5 * float(getattr(C, 'ACTIVITY_LR_MULTIPLIER', 3.0)),  # idx 3: activity
            head_lr_local * 0.5,      # idx 4: psr
            det_head_bias_lr_local * 0.5,  # idx 5: det_head_bias
            bias_lr_local * 0.5,      # idx 6: bias
            0.0,                      # idx 7: videomae (frozen at start, toggled at unfreeze)
        ]
        if loss_params:
            max_lr.append(head_lr_local * 0.5)  # idx 8 (if present): loss
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=C.EPOCHS,
            steps_per_epoch=len(train_loader) // train_accum_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        scheduler = SequentialLR(optimizer, [warmup, scheduler],
                               milestones=[_stage_warmup_epochs])
        logger.info('Scheduler: OneCycleLR (pct_start=0.1, max_lr=[5e-5, 5e-4])')
    elif C.USE_COSINE_ANNEALING:
        cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=C.T_0, T_mult=C.T_mult, eta_min=1e-6
        )
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[_stage_warmup_epochs])
        logger.info('Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)')
    else:
        cosine = CosineAnnealingLR(
            optimizer, T_max=C.EPOCHS - _stage_warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                               milestones=[_stage_warmup_epochs])

    scaler = torch.cuda.amp.GradScaler(enabled=C.MIXED_PRECISION)

    start_epoch = 0
    best_metric = 0.0
    patience_counter = 0

    videomae_warmup_state = {
        'active': False,
        'param_group_idx': -1,
        'unfreeze_lr': 0.0,
        'epochs_remaining': 0,
    }

    # [FIX] STAGE3_WARMUP_EPOCHS ramp from config.py:378
    # Activated when entering Stage 3 (epoch 16+). For the first
    # STAGE3_WARMUP_EPOCHS epochs after entry, activity_head + psr_head LR is
    # scaled by (epoch - stage3_start + 1) / STAGE3_WARMUP_EPOCHS so the
    # newly-unfrozen heads don't blow up gradient magnitude right after Stage 2.
    stage3_warmup_state = {
        'active': False,
        'param_group_idx': ACTIVITY_PARAM_GROUP_IDX,
        'base_lr': head_lr,
        'start_epoch': -1,
        'warmup_epochs': int(getattr(C, 'STAGE3_WARMUP_EPOCHS', 3)),
        'epochs_remaining': 0,
    }

    # [OPUS DECISION 6] Auto-load crash_recovery.pth if no --resume is given and
    # crash_recovery.pth exists with mtime newer than latest.pth.
    # crash_recovery.pth is post-training/pre-validation — safe for resuming
    # optimization, but never promoted to best.pth (unvalidated weights).
    if not args.resume:
        _cr_path = ckpt_dir / 'crash_recovery.pth'
        _latest_path = ckpt_dir / 'latest.pth'
        if _cr_path.exists():
            _cr_mtime = _cr_path.stat().st_mtime
            _lt_mtime = _latest_path.stat().st_mtime if _latest_path.exists() else 0
            if _cr_mtime > _lt_mtime:
                logger.warning(
                    '[AUTO-RESUME] crash_recovery.pth (mtime=%s) is newer than '
                    'latest.pth (mtime=%s) — auto-loading for resume. '
                    'Use --resume latest.pth to ignore crash_recovery.',
                    datetime.fromtimestamp(_cr_mtime).isoformat(),
                    datetime.fromtimestamp(_lt_mtime).isoformat(),
                )
                args.resume = str(_cr_path)
            else:
                logger.info(
                    '[AUTO-RESUME] crash_recovery.pth exists but is older than '
                    'latest.pth — skipping auto-load.'
                )

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
                # [FIX 2026-06-17] When --reset-scheduler, the checkpoint optimizer state
                # carries LRs from a later epoch (e.g., cosine at full LR). These would
                # cause gradient shock for freshly-reinitialized heads training at warmup.
                # Restore the warmup LRs computed at optimizer construction.
                if getattr(args, 'reset_scheduler', False):
                    for i, lr in enumerate(_init_pg_lrs):
                        if i < len(optimizer.param_groups):
                            optimizer.param_groups[i]['lr'] = lr
                    logger.info('  [RESET-SCHEDULER] Restored warmup LRs after optimizer state load.')
                # [FIX C5 2026-06-17] _STAGE_LR_MULT retry scaling applied at optimizer
                # construction (lines 2980-2986) is OVERWRITTEN by load_state_dict above.
                # Re-apply the LR mult after loading so retry LR scaling is not lost.
                # When --reset-scheduler is active, _init_pg_lrs already has the mult
                # baked in, so we must NOT double-apply.
                if _stage_lr_mult != 1.0 and not getattr(args, 'reset_scheduler', False):
                    for pg in optimizer.param_groups:
                        pg['lr'] = pg['lr'] * _stage_lr_mult
                    logger.info(f'  [FIX-C5] Re-applied _STAGE_LR_MULT={_stage_lr_mult}×'
                                f' after optimizer state load.')
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
            if getattr(args, 'reset_scheduler', False):
                logger.info('  [RESET-SCHEDULER] Resetting scheduler to epoch 0 — fresh warmup (reinit-heads retry)')
                if scaler_state:
                    scaler.load_state_dict(scaler_state)
            else:
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
        if getattr(args, 'reset_scheduler', False):
            best_metric = 0.0
            patience_counter = 0
            logger.info(f'Resumed from epoch {start_epoch}, best_metric reset to 0.0 (reset-scheduler)')
        else:
            best_metric = float(ckpt.get('best_metric', 0.0))
            patience_counter = int(ckpt.get('patience_counter', 0))
            logger.info(f'Resumed from epoch {start_epoch}, best={best_metric:.4f}')

        # [CRASH-HARDEN 2026-06-29] Restore global step from checkpoint so
        # step-based validation (VAL_EVERY_N_STEPS) fires at the right cadence.
        _gs = int(ckpt.get('global_step', 0))
        if _gs > 0:
            if not hasattr(C, '_global_step'):
                C._global_step = 0
            C._global_step = _gs
            logger.info(f'  Restored _global_step={_gs} from checkpoint (step-based val cadence)')

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

    # ── RC-25 Recovery: Re-initialize dead heads + FPN ──
    if getattr(args, 'reinit_heads', False):
        global _REINIT_HEADS_ACTIVE, _REINIT_EPOCH_OFFSET
        _REINIT_HEADS_ACTIVE = True
        # Use actual start epoch, not checkpoint epoch. When --start-epoch 0
        # overrides _override_start_epoch, the training loop starts from 0
        # even if checkpoint says epoch N. The offset must match the loop.
        _actual_start = _override_start_epoch if _override_start_epoch is not None else start_epoch
        _REINIT_EPOCH_OFFSET = max(0, _actual_start - 1)  # reset stage counter
        logger.warning(
            '  [REINIT-HEADS] Flag --reinit-heads set: re-initializing 3 '
            'dead heads (det/act/psr) + FPN from priors. Backbone + pose + '
            'pretrained ConvNeXt weights are PRESERVED.'
            f' Stage counter reset: effective_epoch = epoch - {_REINIT_EPOCH_OFFSET}'
            f' (epoch {start_epoch} → effective epoch {max(1, start_epoch - _REINIT_EPOCH_OFFSET)} = Stage {get_stage(start_epoch)})'
        )
        _reinit_pi = float(getattr(C, 'REINIT_PI', 0.01))
        reinit_counts = _reinit_dead_heads(model, reinit_pi=_reinit_pi)
        logger.warning(
            f'  [REINIT-HEADS] Re-initialized submodules: {reinit_counts}'
            f' (cls_score pi={_reinit_pi})'
        )
        # Re-anchor EMA shadow to fresh reinit weights
        if ema is not None and ema.shadow:
            import re as _re
            _head_prefixes = ('det_head.', 'detection_head.', 'activity_head.',
                              'psr_head.', 'fpn.')
            _ema_reset = 0
            for _n, _p in model.named_parameters():
                if any(_n.startswith(pf) for pf in _head_prefixes):
                    ema.shadow[_n] = _p.data.clone().detach().to(ema.device if ema.device else _p.device)
                    _ema_reset += 1
            logger.warning(
                f'  [REINIT-HEADS] EMA shadow re-anchored for {_ema_reset} head/fpn tensors.'
            )
        # [FIX 2026-06-16] Reset AdamW optimizer state for reinitialized head params.
        # Stored momentum (exp_avg/exp_avg_sq) from old checkpoint is incompatible with
        # freshly reinitialized weights, causing catastrophic optimizer steps that collapse
        # detection head within ~1000 steps (cls_mean -2.4 → -14.8).
        # NOTE: zero tensors in-place rather than popping or deleting entries —
        #   PyTorch 2.5+ AdamW expects state keys to exist (step as tensor, not int),
        #   and _init_group only creates them if len(state)==0.
        _opt_head_prefixes = ('det_head.', 'detection_head.', 'activity_head.',
                              'psr_head.', 'fpn.')
        _optim_reset = 0
        for _n, _p in model.named_parameters():
            if any(_n.startswith(pf) for pf in _opt_head_prefixes) and _p in optimizer.state:
                _state = optimizer.state[_p]
                if 'exp_avg' in _state:
                    _state['exp_avg'].zero_()
                if 'exp_avg_sq' in _state:
                    _state['exp_avg_sq'].zero_()
                _optim_reset += 1
        if _optim_reset > 0:
            logger.warning(
                f'  [REINIT-HEADS] AdamW optimizer state reset for {_optim_reset} reinit-head params.'
            )

        # Reset Kendall log_vars to neutral for recovery
        with torch.no_grad():
            criterion.log_var_det.fill_(0.0)
            criterion.log_var_act.fill_(0.0)
            criterion.log_var_psr.fill_(0.0)
            criterion.log_var_pose.fill_(0.0)
        logger.info('  [REINIT-HEADS] Kendall log_vars reset to neutral (det=act=psr=pose=0.0).')
        # Reset detection head gradient warmup counter
        global _REINIT_DET_STEP
        _REINIT_DET_STEP = 0
        logger.info('  [REINIT-HEADS] Detection head gradient warmup counter reset to 0.')
        # PSR output head warmup: 2x grad multiplier for first 200 steps after reinit
        global _PSR_WARMUP_STEPS_REMAINING
        _PSR_WARMUP_STEPS_REMAINING = 200
        logger.info('  [REINIT-HEADS] PSR output head warmup: 2x grad multiplier for 200 steps.')

    # ── Step-0 Assertion (RC-25 gate) ──
    # [AUDIT FIX 2026-06-11] The previous version had two fatal flaws:
    # (1) `sample_batch['image']` — collate_fn returns a TUPLE (images, targets),
    #     so the probe raised TypeError on every run; and
    # (2) the blanket `except Exception` wrapped the probe's own RuntimeError,
    #     downgrading the assertion to a warning. The guard could never fail a
    #     run. Probe-infrastructure failures now crash loudly too: a guard that
    #     silently no-ops is how RC-25 survived three investigation rounds.
    if getattr(args, 'reinit_heads', False):
        logger.info('[STEP-0 ASSERT] Running step-0 diagnostic forward pass...')
        model.eval()
        _probe_median = None
        try:
            try:
                _probe_images, _probe_targets = next(iter(train_loader))
            except StopIteration:
                raise RuntimeError(
                    '[STEP-0 ASSERT] train_loader is EMPTY — cannot probe. '
                    'Check subset_ratio / dataset paths before training.'
                )
            # Same preprocessing as the training loop (uint8 -> float,
            # ImageNet normalize). Raw uint8 input would distort the
            # feature-magnitude measurement this guard exists to make.
            _probe_images = _prepare_images(_probe_images[:1], device, training=False)
            _probe_clip = None
            if isinstance(_probe_targets, dict):
                _probe_clip = _probe_targets.get('clip_rgb')
                if _probe_clip is not None and _probe_clip.numel() > 0:
                    _probe_clip = _probe_clip[:1].to(device)
                else:
                    _probe_clip = None
            with torch.no_grad():
                _probe_out = model(_probe_images, clip_rgb=_probe_clip)
            _probe_median = _probe_out['cls_preds'].detach().abs().median().item()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f'[STEP-0 ASSERT] diagnostic forward pass failed: {exc!r}. '
                'Fix the probe — do not train without this RC-25 gate.'
            ) from exc
        finally:
            model.train()
        logger.info(f'  [STEP-0 ASSERT] cls_logits.abs().median() = {_probe_median:.3f}')
        if _probe_median >= 8.0:
            raise RuntimeError(
                f'STEP-0 ASSERTION FAILED: cls_logits.abs().median()={_probe_median:.3f} >= 8.0. '
                'FPN/backbone feature magnitude still saturating detection head. '
                'Run D7-D9 diagnostics then reinit FPN (or restart from ImageNet init) '
                'before retraining.'
            )
        logger.info('  [STEP-0 ASSERT] PASSED: logit scale in healthy range (< 8).')

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

    # --- TRAINING WATCHDOG THREAD (Bashara 2026-06-30) ---
    # Monitors the GPU heartbeat file in a separate daemon thread. If the heartbeat
    # stops updating for > 600 seconds (10 min), the training has hung in a CUDA
    # kernel or DataLoader deadlock. Kills the process so the supervisor can restart.
    # The pre-val checkpoint ensures no epoch progress is lost.
    # Fixed: only kill if the heartbeat PID matches OUR PID (avoids killing on stale
    # heartbeat from a previous process that wrote to the same directory).
    _watchdog_ckpt_dir = ckpt_dir
    _watchdog_active = True
    _watchdog_pid = os.getpid()
    def _watchdog_loop():
        while _watchdog_active:
            _hb_path = _watchdog_ckpt_dir / '.gpu_heartbeat'
            if _hb_path.exists():
                try:
                    _hb_content = _hb_path.read_text().strip()
                    if _hb_content:
                        _hb_parts = _hb_content.split('\n')[0].split('|')
                        _hb_ts = float(_hb_parts[0])
                        _hb_pid = int(_hb_parts[3]) if len(_hb_parts) > 3 else -1
                        if _hb_pid == _watchdog_pid:
                            _hb_age = time.time() - _hb_ts
                            if _hb_age > 600:
                                msg = f'[WATCHDOG] GPU heartbeat stale ({_hb_age:.0f}s > 600s, pid={_hb_pid}) — killing process'
                                print(msg, flush=True)
                                os._exit(1)
                except Exception:
                    pass
            time.sleep(30)
    _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
    _watchdog_thread.start()

    try:
        _train_start_epoch = _override_start_epoch if _override_start_epoch is not None else start_epoch
        # [FIX 2026-06-06] Warn loudly if epoch range is empty (e.g., --max-epochs 1 + --resume from
        # epoch-0 checkpoint → range(1, 1) silently does no training and the user gets
        # "Best combined metric: 0.0000" with no explanation). Log + fall through to finalization
        # so the (unchanged) checkpoint is preserved. The for loop body will be a no-op.
        if _train_start_epoch >= C.EPOCHS:
            logger.warning(
                f'  [EPOCH_LOOP_EMPTY] _train_start_epoch={_train_start_epoch} >= C.EPOCHS={C.EPOCHS}. '
                f'Nothing to train — skipping epoch loop and proceeding to finalization. '
                f'If this is unexpected, use --max-epochs (e.g. --max-epochs {_train_start_epoch + 5}) '
                f'OR drop --resume for a fresh run.'
            )
        for epoch in range(_train_start_epoch, C.EPOCHS):
            logger.info(f'\n--- Epoch {epoch}/{C.EPOCHS - 1} ---')
            criterion.set_epoch(epoch)

            # [MEMORY LEAK FIX] At epoch start (skip epoch 0 — cache is fresh),
            # release the previous epoch's FRAME_CACHE so the OS can reclaim
            # ~5-7GB RAM. The next epoch's loader will re-preload on first access.
            if epoch > _train_start_epoch and getattr(C, 'CLEAR_FRAME_CACHE_EPOCH_END', True):
                try:
                    from data.industreal_dataset import clear_frame_cache
                    clear_frame_cache()
                    import gc as _gc
                    _gc.collect()
                    torch.cuda.empty_cache()
                except Exception as _exc:
                    logger.warning(f'  [MEMORY] clear_frame_cache failed: {_exc}')

            # Doc 2 §B.2: Stage transition validation — log trainable param counts
            current_stage = get_stage(epoch)
            prev_stage = get_stage(epoch - 1) if epoch > 0 else current_stage
            if current_stage != prev_stage:
                # Stage transition: log params, PRESERVE log_vars (do NOT reset),
                # activate Stage 3 warmup LR ramp.
                _on_stage_transition(
                    model, criterion, current_stage, epoch, C.BACKBONE,
                    stage3_warmup_state=stage3_warmup_state,
                )

                # Doc 2 §C.2: Fresh EMA at Stage 3 entry.
                # Use epoch-specific decay from the EMA schedule (not the hardcoded
                # 0.999 from EMA_DECAY): at Stage 3 epoch 1 the decay is 0.999, but
                # by epoch 18+ it should be 0.9999. Hardcoding 0.999 here meant
                # shadow weights lagged the model for ~700 steps past the schedule.
                if current_stage == 3 and ema is not None:
                    from model import EMA as EMAClass
                    stage3_decay = _get_ema_decay(epoch)
                    ema = EMAClass(model, decay=stage3_decay, device=device)
                    logger.info(
                        '[Epoch %d] Stage 3: reinitialized EMA from current model state (decay=%.4f)'
                        % (epoch, stage3_decay)
                    )

            # Doc 01 §B.1: Unfreeze VideoMAE stream at configured epoch to let the
            # temporal stream adapt to IndustReal kinematics after backbone is warmed up.
            # [HOT-FIX 2026-06-07] Cumulative `>=` + idempotency gate (was `==` one-shot):
            # the old strict-equality check missed unfreeze on resume runs because the
            # epoch counter had already advanced past VIDEOMAE_UNFREEZE_EPOCH. The gate
            # `optimizer.param_groups[VIDEOMAE_PARAM_GROUP_IDX]['lr'] == 0.0` is
            # state-tied (not epoch-tied), so it survives resume cleanly: a fresh
            # pre-unfreeze resume fires once when the first qualifying epoch hits,
            # subsequent epochs see lr != 0.0 and skip.
            unfreeze_epoch = int(getattr(C, 'VIDEOMAE_UNFREEZE_EPOCH', -1))
            if (
                unfreeze_epoch >= 0
                and epoch >= unfreeze_epoch
                and C.USE_VIDEOMAE
                and len(optimizer.param_groups) > VIDEOMAE_PARAM_GROUP_IDX
                and optimizer.param_groups[VIDEOMAE_PARAM_GROUP_IDX]['lr'] == 0.0
            ):
                if hasattr(model, 'videomae_stream'):
                    videomae_lr = float(getattr(C, 'VIDEOMAE_UNFREEZE_LR', 1e-5))
                    # [OPUS FIX #3] unfreeze() flips requires_grad=True on the
                    # encoder (side effect we still need); we discard its return
                    # value because the videomae param group was already
                    # pre-registered at optimizer build time with lr=0. Toggling
                    # lr in-place keeps param_groups length constant so
                    # OneCycleLR's zip(strict=True) at scheduler.step() never
                    # sees a length mismatch.
                    _ = model.videomae_stream.unfreeze(lr=videomae_lr)
                    optimizer.param_groups[VIDEOMAE_PARAM_GROUP_IDX]['lr'] = videomae_lr
                    logger.info(
                        '[Epoch %d] VideoMAE stream unfreeze fired (lr=%.2e, '
                        'param_group_idx=%d)' % (
                            epoch, videomae_lr, VIDEOMAE_PARAM_GROUP_IDX,
                        )
                    )
                    # Doc 2 A.1: activity_head.videomae_proj (384-D VideoMAE ->
                    # embed_dim projection) lives on ActivityHead, which is NOT
                    # frozen by staged training — only videomae_stream is. The
                    # projection has been in activity_psr_params at head_lr
                    # since epoch 0 (param-group build at line 2058). Adding it
                    # again here would raise
                    # "ValueError: some parameters appear in more than one
                    # parameter group" (observed crash at epoch 10 start).
                    # So this step is a no-op log; videomae_proj is already
                    # trainable at the right LR.
                    activity_head = getattr(model, 'activity_head', None)
                    if activity_head is not None and getattr(activity_head, 'videomae_proj', None) is not None:
                        n_proj = sum(p.numel() for p in activity_head.videomae_proj.parameters())
                        logger.info(
                            '[Epoch %d] VideoMAE projection already trainable via '
                            'activity_psr param group (n=%d, lr=head_lr=%.0e); '
                            'no add_param_group needed (would duplicate)'
                            % (epoch, n_proj, head_lr)
                        )
                    videomae_warmup_epochs = int(getattr(C, 'VIDEOMAE_WARMUP_EPOCHS', 3))
                    videomae_warmup_state['active'] = True
                    videomae_warmup_state['param_group_idx'] = VIDEOMAE_PARAM_GROUP_IDX
                    videomae_warmup_state['unfreeze_lr'] = videomae_lr
                    videomae_warmup_state['epochs_remaining'] = videomae_warmup_epochs
                    logger.info(
                        '[Epoch %d] VideoMAE stream unfrozen at lr=%.0e, warmup=%d epochs'
                        % (epoch, videomae_lr, videomae_warmup_epochs)
                    )
                    # EMA shadow must be re-registered with newly unfrozen parameters;
                    # without this, update() asserts that every requires_grad param has a
                    # shadow entry — failing for VideoMAE params that were frozen during
                    # EMA init but become trainable at epoch 10.  Mirrors Stage 3 pattern.
                    if ema is not None:
                        from model import EMA as EMAClass
                        ema = EMAClass(model, decay=EMA_DECAY, device=device)
                        logger.info(
                            '[Epoch %d] VideoMAE unfreeze: reinitialized EMA from current model state'
                            % epoch
                        )
                else:
                    logger.warning(
                        '[Epoch %d] USE_VIDEOMAE=True but model has no videomae_stream attribute'
                        % epoch
                    )

            if ema is not None:
                ema.set_decay(_get_ema_decay(epoch))

            train_attempt = 0
            _train_failed = False
            while True:
                train_attempt += 1
                if train_attempt > 6:
                    logger.critical(
                        'Exceeded maximum train retry attempts (6) for this epoch. '
                        'Returning safe fallback metrics (all 0.0) to prevent training crash.'
                    )
                    train_metrics = {}
                    _train_failed = True
                    break
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
                        val_ds=val_ds,
                        val_every_n_steps=int(getattr(C, 'VAL_EVERY_N_STEPS', 0)),
                        distill_loss_fn=distill_loss_fn,
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

            # [CRASH-HARDEN 2026-06-29] If all retries failed, skip the rest of
            # the epoch (scheduler step, validation, checkpointing) and continue
            # to the next epoch. Prevents total training loss from a single bad
            # epoch (OOM, corrupted checkpoint, unfixable NaN).
            if _train_failed:
                logger.critical('[TRAIN_FAILED] Skipping scheduler step, validation, and checkpoint for this epoch.')
                continue  # skip to next epoch in for loop

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
            # [FIX] STAGE3_WARMUP_EPOCHS ramp from config.py:378
            # Scale activity/psr head LR linearly from 1/N to 1.0×base over the
            # first STAGE3_WARMUP_EPOCHS epochs of Stage 3.
            # epoch=stage3_start → 1/N; +1 → 2/N; ... +N-1 → N/N=1.0×base.
            if (stage3_warmup_state['active']
                    and stage3_warmup_state['epochs_remaining'] > 0):
                warmup_total = stage3_warmup_state['warmup_epochs']
                completed = warmup_total - stage3_warmup_state['epochs_remaining']
                warmup_factor = (completed + 1) / float(warmup_total)
                idx = stage3_warmup_state['param_group_idx']
                base = stage3_warmup_state['base_lr']
                # Restore to base first (in case scheduler or another block set it),
                # then apply this epoch's warmup factor.
                optimizer.param_groups[idx]['lr'] = base * warmup_factor
                stage3_warmup_state['epochs_remaining'] -= 1
                logger.info(
                    '[Epoch %d] Stage3 head warmup lr=%.2e (factor=%.2f, %d/%d)'
                    % (epoch, base * warmup_factor, warmup_factor,
                       warmup_total - stage3_warmup_state['epochs_remaining'],
                       warmup_total)
                )
            _check_ram(f'epoch_{epoch}_train')

            # [2% AUDIT] TRAIN_MAX_STEPS: check step limit AFTER val block completes.
            # FIX: Previously this break was placed BEFORE the val block, causing
            # TRAIN_MAX_STEPS to skip validation entirely and fall through to
            # train() returning → parent restarts → eval without training (epoch N+1).
            # Moving to AFTER val ensures validation always runs regardless of step limit.
            _train_max_steps = getattr(C, 'TRAIN_MAX_STEPS', 0)
            if _train_max_steps > 0:
                _batch_count = train_metrics.get('num_batches', 0)
                if not hasattr(C, '_global_step'):
                    C._global_step = 0
                C._global_step += _batch_count
                logger.info(f'  [2pct] global_step={C._global_step}/{_train_max_steps}')
                if C._global_step >= _train_max_steps:
                    logger.info(f'  [2pct] TRAIN_MAX_STEPS limit reached ({C._global_step}). Will exit after val completes.')

            # [E5] Log per-param-group LRs instead of just group 2
            _pg_labels = ['backbone', 'det_head', 'head', 'act/psr', 'bias', 'videomae', 'loss']
            _pg_lrs = ' '.join(
                f'{_pg_labels[i] if i < len(_pg_labels) else f"g{i}"}={g["lr"]:.2e}'
                for i, g in enumerate(optimizer.param_groups)
            )
            logger.debug(f'  [LR] {_pg_lrs}')
            current_lr = optimizer.param_groups[2]['lr']
            ema_decay_str = ''
            if ema is not None:
                ema_decay_str = f'  ema_decay={ema.decay:.4f}'
            def _s(v, alt=0.0):
                """Safe numeric: replace NaN/Inf with alt."""
                if isinstance(v, float) and math.isfinite(v):
                    return v
                return alt

            def _safe_log(v, key, default=0.0):
                """Guard a single metric against NaN/Inf. log_var keys are signed; only reject non-finite."""
                val = train_metrics.get(key, default)
                if not isinstance(val, float):
                    try:
                        val = float(val)
                    except Exception:
                        logger.warning(f'  [TRAIN_METRIC_NAN] {key}={val!r} at epoch {epoch} — non-numeric, using {default}')
                        return default
                if not math.isfinite(val):
                    logger.warning(f'  [TRAIN_METRIC_NAN] {key}={val} at epoch {epoch} — non-finite, using {default}')
                    return default
                # log_var keys are signed; loss keys must be >= 0
                _SIGNED_KEYS = {'log_var_det', 'log_var_pose', 'log_var_act', 'log_var_psr'}
                if key not in _SIGNED_KEYS and val < 0.0:
                    logger.warning(f'  [TRAIN_METRIC_NEG] {key}={val} at epoch {epoch} — unexpected negative loss, using {default}')
                    return default
                return val

            # [OPUS v8 E2] Kendall precision ratio: head_pose ÷ detection.
            # If prec_ratio >> 1 (e.g., ~30-40×), head_pose dominates the shared backbone.
            # Each precision = exp(-log_var), clamped range: exp(-2)≈0.14 to exp(4)≈54.6.
            _lv_det = _safe_log(train_metrics.get('log_var_det', 0.0), 'log_var_det', default=0.0)
            _lv_hp = _safe_log(train_metrics.get('log_var_pose', 0.0), 'log_var_pose', default=0.0)
            _prec_det = math.exp(-_lv_det) if math.isfinite(_lv_det) else 0.0
            _prec_hp = math.exp(-_lv_hp) if math.isfinite(_lv_hp) else 0.0
            _prec_ratio = (_prec_hp / _prec_det) if _prec_det > 1e-10 else float('inf')

            # [OPUS v8 E3] cls_score.weight.norm() — tracks backbone feature health.
            # Shrinking norm = backbone losing object-discriminative features (symptom chain).
            _cls_w_norm = 0.0
            _cls_w_iter = next((p for n, p in model.named_parameters()
                                if n.endswith('detection_head.cls_score.weight')), None)
            if _cls_w_iter is not None:
                _cls_w_norm = _cls_w_iter.detach().norm().item()

            logger.info(
                f'Train: loss={_safe_log(train_metrics["total"], "total"):.4f}  '
                f'det={_safe_log(train_metrics["det"], "det"):.4f}  '
                f'pose={_safe_log(train_metrics["head_pose"], "head_pose"):.4f}  '
                f'act={_safe_log(train_metrics["activity"], "activity"):.4f}  '
                f'psr={_safe_log(train_metrics["psr"], "psr"):.4f}  '
                f'lr={current_lr:.2e}  '
                f'prec_d={_prec_det:.2f} prec_hp={_prec_hp:.2f} prec_r={_prec_ratio:.1f}  '
                f'cls_w_n={_cls_w_norm:.4f}  '
                f'kd_d={_lv_det:+.3f}  '
                f'kd_p={_lv_hp:+.3f}  '
                f'kd_a={_safe_log(train_metrics["log_var_act"], "log_var_act", default=0.0):+.3f}  '
                f'kd_r={_safe_log(train_metrics["log_var_psr"], "log_var_psr", default=0.0):+.3f}'
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

            # --- PRE-VALIDATION GUARD (Bashara 2026-05-26) ---
            # Verify train_one_epoch produced valid output for this epoch before running
            # evaluation. This prevents the "eval without training" bug where a silent
            # train_one_epoch crash or zero-batch return causes eval to run next with
            # no training having happened. If train_metrics is empty or suspicious,
            # raise immediately — the epoch retry loop will handle it.
            _train_ok = (
                isinstance(train_metrics, dict)
                and train_metrics.get('num_batches', 0) > 0
                and torch.isfinite(torch.tensor(train_metrics.get('total', 0.0)))
            )
            if not _train_ok:
                logger.error(
                    f'  [PRE_VAL_GUARD] train_metrics suspicious for epoch {epoch}: '
                    f'batches={train_metrics.get("num_batches", 0)}, '
                    f'total_loss={train_metrics.get("total", 0)}. '
                    f'Skipping val and retrying training.'
                )
                # Raise to trigger train retry — don't run val with broken train state
                raise RuntimeError(
                    f'PRE_VAL_GUARD: train_one_epoch(epoch={epoch}) produced '
                    f'invalid metrics (batches={train_metrics.get("num_batches",0)}, '
                    f'loss={train_metrics.get("total",0)}). Not running val until '
                    f'training is healthy.'
                )
            logger.info(
                f'  [PRE_VAL_GUARD] epoch {epoch} training healthy: '
                f'batches={train_metrics["num_batches"]}, loss={train_metrics["total"]:.4f}'
            )
            _write_stage_heartbeat(epoch, training_pid=os.getpid())

            # --- PRE-VAL CHECKPOINT (Bashara 2026-06-30) ---
            # Save latest.pth immediately after training completes, before validation
            # runs. If validation crashes, the next resume restores the post-training
            # state instead of going back an epoch. crash_recovery.pth also exists but
            # is not loaded automatically on resume — this ensures latest.pth tracks
            # the latest trained (not validated) epoch.
            _atomic_save({
                'epoch':            epoch,
                'model':           model.state_dict(),
                'optimizer':       optimizer.state_dict(),
                'scheduler':       scheduler.state_dict(),
                'scaler':          scaler.state_dict(),
                'best_metric':     best_metric,
                'patience_counter': patience_counter,
                'ema_shadow':      {k: v.clone() for k, v in ema.shadow.items()} if ema is not None else {},
                'criterion': {
                    'log_var_det': criterion.log_var_det.data.clone(),
                    'log_var_pose': criterion.log_var_pose.data.clone(),
                    'log_var_act': criterion.log_var_act.data.clone(),
                    'log_var_psr': criterion.log_var_psr.data.clone(),
                } if criterion is not None else {},
                'global_step':   getattr(C, '_global_step', 0),
            }, ckpt_dir / 'latest.pth')
            logger.info(f'  [PRE_VAL_CKPT] latest.pth updated with epoch {epoch} post-training state')

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

                # Bug A fix — only swap to EMA shadow if EMA has been updated.
                # With STAGED_TRAINING=False, EMA tracks from epoch 0.
                # With STAGED_TRAINING=True, only swap once stage>=3 (EMA has been updating).
                _ema_staged = bool(getattr(C, 'STAGED_TRAINING', True))
                # [RF1 FIX 2026-06-18] During Stage 1 (detection bootstrap), EMA weights
                # trail the cls_score bias initialization (config-driven REINIT_PI), causing eval collapse.
                # Use raw model for eval until Stage 2 (epoch 6+).
                _ema_rf1_ok = _ema_staged or current_stage >= 2
                ema_warmed = (ema is not None) and _ema_rf1_ok and (not _ema_staged or current_stage >= 3)
                if ema_warmed:
                    ema.get_ema()
                    logger.info('  [EMA] Using exponential-moving-average weights for val')
                elif ema is not None:
                    logger.info('  [EMA] Skipping EMA swap — shadow not yet updated (stage<3, staged=%s)' % _ema_staged)

                # [OPUS DECISION 5] Save subprocess checkpoint (post-EMA-swap) so the
                # subprocess eval worker on GPU 0 picks up EMA-smoothed weights.
                if CFG_USE_SUBPROCESS_EVAL:
                    _atomic_save({'model': model.state_dict()}, ckpt_dir / 'val_subprocess.pth')
                    logger.info('  [SUB] Saved val_subprocess.pth (post-EMA-swap) for subprocess eval')

                val_batch_size_rt = CFG_VAL_BATCH_SIZE
                val_workers_rt = 0          # HARDENED: always 0 — no worker management
                val_prefetch_rt = 1
                val_max_batches_rt = CFG_EVAL_MAX_BATCHES

                # [OPUS v5] Eval cadence: if DET_METRICS_EVERY_N is set and this is NOT a full-det-eval epoch,
                # cap val batches to GATE_EVAL_MAX_BATCHES for fast gate-only check.
                _det_every_n = int(getattr(C, 'DET_METRICS_EVERY_N', 0))
                _gate_max = int(getattr(C, 'GATE_EVAL_MAX_BATCHES', 200))
                if _det_every_n > 0 and (epoch + 1) % _det_every_n != 0:
                    val_max_batches_rt = _gate_max
                    logger.info(f'  [GATE EVAL] epoch {epoch}: capped at {_gate_max} batches (full det mAP every {_det_every_n} epochs)')

                val_attempt = 0
                while True:
                    val_attempt += 1
                    if val_attempt > 2:
                        logger.warning(
                            f'Validation failed {val_attempt - 1} times — '
                            f'skipping val this epoch to avoid infinite loop.'
                        )
                        val_metrics = {}  # empty → _s() returns 0.0 for all metrics
                        break
                    val_loader = _build_loader(
                        val_ds,
                        'val',
                        val_batch_size_rt,
                        val_workers_rt,
                        prefetch=val_prefetch_rt,
                        persistent=False,
                    )
                    global IN_EVALUATION_PHASE
                    # [CRASH-HARDEN 2026-06-29] Save pre-validation checkpoint so a
                    # validation crash loses at most 50 batches instead of the full epoch.
                    _save_crash_recovery('pre_val')
                    IN_EVALUATION_PHASE = True
                    torch.cuda.empty_cache()  # Clear cached allocator memory before validation
                    try:
                        try:
                            # [OPUS DECISION 5] Subprocess eval (GPU 0, SIGKILL-safe) OR
                            # ThreadPoolExecutor (backward compat). The subprocess runs
                            # evaluate_all in a separate process on CUDA_VISIBLE_DEVICES=0,
                            # so a CUDA kernel hang can be SIGKILL'd without corrupting the
                            # training CUDA context on GPU 1.
                            _eval_timeout = CFG_SUBPROCESS_EVAL_TIMEOUT if CFG_USE_SUBPROCESS_EVAL else int(getattr(C, 'EVAL_TIMEOUT_SECONDS', 1200))
                            if CFG_USE_SUBPROCESS_EVAL:
                                _out_path = ckpt_dir / f'val_results_epoch{epoch}.json'
                                _sub_ckpt = ckpt_dir / 'val_subprocess.pth'
                                if not _sub_ckpt.exists():
                                    _sub_ckpt = ckpt_dir / 'latest.pth'
                                val_metrics = run_val_subprocess(
                                    ckpt_path=_sub_ckpt,
                                    out_path=_out_path,
                                    overrides={
                                        'EVAL_MAX_BATCHES': val_max_batches_rt,
                                        'VAL_BATCH_SIZE': val_batch_size_rt,
                                        'epoch': epoch,
                                    },
                                    timeout=_eval_timeout,
                                )
                                if not val_metrics:
                                    logger.error(f'[SUB EVAL TIMEOUT] evaluate_all exceeded {_eval_timeout}s -- raising to retry')
                                    raise TimeoutError(f'[SUB EVAL TIMEOUT] evaluate_all exceeded {_eval_timeout}s')
                            else:
                                # [CUDA-HANG FIX 2026-06-30 v2] ThreadPoolExecutor timeout for evaluate_all.
                                # SIGALRM cannot interrupt CUDA kernel hangs (signal handlers need the
                                # Python interpreter to run). ThreadPoolExecutor with a timeout raises
                                # TimeoutError when the thread doesn't finish — but the thread keeps
                                # running (CUDA kernel stays alive). After timeout, we detach and create
                                # a fresh executor for the retry. The zombie thread will be cleaned up
                                # when it eventually finishes or when the process dies at epoch end.
                                # [THREAD-SAFE] evaluate.py's signal.signal() calls are wrapped in
                                # try/except ValueError so they gracefully degrade in threads.
                                import concurrent.futures as _cf
                                _eval_executor = _cf.ThreadPoolExecutor(max_workers=1)
                                try:
                                    _eval_future = _eval_executor.submit(
                                        evaluate_all, model, criterion, val_loader, device,
                                        max_batches=val_max_batches_rt, epoch=epoch,
                                    )
                                    val_metrics = _eval_future.result(timeout=_eval_timeout)
                                except _cf.TimeoutError:
                                    _eval_executor.shutdown(wait=False)
                                    logger.error(f'[EVAL TIMEOUT] evaluate_all exceeded {_eval_timeout}s -- raising to retry')
                                    raise TimeoutError(f'[EVAL TIMEOUT] evaluate_all exceeded {_eval_timeout}s')
                                _eval_executor.shutdown(wait=False)
                            _check_per_class_activity_sanity(val_metrics, epoch)
                            torch.cuda.empty_cache()  # Clear cached allocator memory after validation success
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
                                is_recoverable = False
                                if 'empty' in exc_str.lower() and ('act_preds' in exc_str or 'batch' in exc_str.lower()):
                                    is_recoverable = True
                                if 'timeout' in exc_str.lower() or 'timed out' in exc_str.lower() or isinstance(exc, TimeoutError):
                                    is_recoverable = True
                                    logger.warning(
                                        f'[EVAL TIMEOUT] evaluate_all timed out — reducing scope and retrying: {exc_str[:200]}'
                                    )
                                if is_recoverable:
                                    # Reduce scope and retry until we exhaust attempts
                                    if val_attempt <= 2:
                                        val_batch_size_rt = max(1, val_batch_size_rt // 2)
                                        val_workers_rt = 0  # HARDENED: workers only used on retried OOM path
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
                                        logger.warning(
                                            '  [WORKER_SHUTDOWN] retry del val_loader, '
                                            'workers will be cleaned in next finally'
                                        )
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
                            logger.warning(
                                '  [WORKER_SHUTDOWN] OOM retry del val_loader, '
                                'workers will be cleaned in next finally'
                            )
                            continue
                    finally:
                        IN_EVALUATION_PHASE = False
                        # [CUDA-CRASH FIX 2026-06-30] Wrap cleanup in try/except.
                        # Silent CUDA errors (e.g. from corrupted context after a failed
                        # step-val) can crash the process during del/gc/empty_cache without
                        # a Python traceback. Catching here prevents total training loss.
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        except Exception:
                            logger.error('[CUDA] Synchronize failed before eval cleanup — CUDA context may be corrupted')
                        try:
                            _shutdown_loader_workers(val_loader, logger)
                            del val_loader
                            gc.collect()
                            torch.cuda.empty_cache()
                        except Exception as _clean_exc:
                            logger.error(f'[EVAL CLEANUP] Cleanup failed: {_clean_exc} — continuing')
                        logger.info('  [POST_EVAL] val_loader cleaned up, resuming train...')

                    # [DIAGNOSTIC] Log whether val succeeded and produced metrics
                    # before proceeding. This catches silent eval crashes that would
                    # otherwise cause the epoch loop to continue with empty val_metrics
                    # and trigger a second eval on the next iteration.
                    if val_metrics:
                        logger.info(
                            f'  [VAL_OK] epoch {epoch} val completed, '
                            f'loss={val_metrics.get("loss", -1):.4f}'
                        )
                    else:
                        logger.warning(
                            f'  [VAL_EMPTY] epoch {epoch} val_metrics is EMPTY — '
                            f'skipping checkpoint and patience update.'
                        )

                    if ema_warmed:
                        ema.restore()
                        logger.info('  [EMA] Restored original weights after val')

                    def _s(v, alt=float('nan')):
                        """Safe numeric: NaN/Inf → alt (default NaN so broken metrics surface visibly)."""
                        if isinstance(v, float) and math.isfinite(v):
                            return v
                        return alt

                    # Compute combined metric before printing Val: line (same logic as below, with safe fallback)
                    # [HONEST METRIC 2026-06-22] Judge progress on the present-class mAP
                    # (det_mAP50_pc — channels with GT>0 only), NOT the COCO-24 mean that
                    # averages in ~8 zero-GT channels + the background channel and so dilutes
                    # the headline ~40% below real performance. det_mAP50 stays the logged
                    # paper number on the Val: line; only the combined/best/gate DECISION uses _pc.
                    _n_present_v = int(_s(val_metrics.get('det_n_present_classes'), alt=0))
                    _map50_v  = (_s(val_metrics.get('det_mAP50_pc', 0.0))
                                 if _n_present_v > 0 else _s(val_metrics.get('det_mAP50', 0.0)))
                    _f1_act_v = _s(val_metrics.get('act_macro_f1', 0.0))
                    _mae_raw  = val_metrics.get('head_pose_MAE', float('nan'))
                    _mae_pose_v = _s(_mae_raw, alt=float('nan'))
                    # [FIX 2026-05-31] psr_macro_f1 = psr_overall_f1 = 0.0 (all-ones predictions).
                    # Use psr_f1_at_t (±3-frame F1, the actual benchmark metric) for combined metric.
                    _f1_psr_v = _s(val_metrics.get('psr_f1_at_t', 0.0))
                    if all(map(math.isfinite, [_map50_v, _f1_act_v, _mae_pose_v, _f1_psr_v])):
                        combined = _compute_combined_metric(
                            _map50_v, _f1_act_v, _mae_pose_v, _f1_psr_v,
                            active_det=CFG_TRAIN_DET,
                            active_act=CFG_TRAIN_ACT,
                            active_pose=CFG_TRAIN_HEAD_POSE,
                            active_psr=CFG_TRAIN_PSR,
                        )
                    else:
                        combined = 0.0
                        logger.warning(
                            f'  [COMBINED_NAN] components: map50={_map50_v} f1_act={_f1_act_v} '
                            f'mae_pose={_mae_pose_v} f1_psr={_f1_psr_v} — using combined=0.0'
                        )

                    logger.info(
                        f'Val: loss={_s(val_metrics.get("loss")):.4f}  '
                        f'det_mAP50={_s(val_metrics.get("det_mAP50")):.4f}  '
                        f'det_mAP50_pc={_s(val_metrics.get("det_mAP50_pc")):.4f}  '
                        f'det_n_present={int(_s(val_metrics.get("det_n_present_classes"), alt=0))}  '
                        f'act_clip={_s(val_metrics.get("act_clip_accuracy")):.4f}  '
                        f'act_frame={_s(val_metrics.get("act_frame_accuracy")):.4f}  '
                        f'act_macro_f1={_s(val_metrics.get("act_macro_f1")):.4f}  '
                        f'act_top5={_s(val_metrics.get("act_top5_accuracy")):.4f}  '
                        f'forward_angular_MAE_deg={_s(val_metrics.get("forward_angular_MAE_deg"), alt=float("nan")):.2f}  '
                        f'psr_f1={_s(val_metrics.get("psr_f1_at_t")):.4f}  '
                        f'psr_edit={_s(val_metrics.get("psr_edit_score")):.4f}  '
                        f'psr_pos={_s(val_metrics.get("psr_pos")):.4f}  '
                        f'as_f1={_s(val_metrics.get("as_f1")):.4f}  '
                        f'as_map_r={_s(val_metrics.get("as_map_at_r")):.4f}  '
                        f'ev_ap={_s(val_metrics.get("ev_ap")):.4f}  '
                        f'ev_f1={_s(val_metrics.get("ev_f1")):.4f}  '
                        f'combined={_s(combined):.4f}'
                    )

                    break  # [FIX 2026-05-27] Success path — exit retry loop.
                           # Without this, while True: iterates again and calls
                           # evaluate_all() a second time immediately after POST_EVAL.

                if ema is not None and current_stage == 3:
                    _compare_raw_vs_ema(
                        model, criterion, val_ds, device, val_metrics, epoch, ckpt_dir
                    )

                # [FIX] head_pose_MAE in _task_keys — NaN in head pose must also trigger skip
                _task_keys = ('det_mAP50', 'act_macro_f1', 'psr_f1_at_t', 'head_pose_MAE')
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
                    pass  # NaN means eval was skipped (DET_METRICS_EVERY_N) — don't burn patience
                else:
                    _map50 = _s(val_metrics.get('det_mAP50', 0.0))
                    _f1_act = _s(val_metrics.get('act_macro_f1', 0.0))
                    _mae_pose_raw = val_metrics.get('head_pose_MAE', float('nan'))
                    _mae_pose = _s(_mae_pose_raw, alt=float('nan'))
                    # [FIX 2026-05-31] Use psr_f1_at_t (real ±3-frame F1) instead of psr_macro_f1 (= psr_overall_f1 = 0.0)
                    _f1_psr = _s(val_metrics.get('psr_f1_at_t', 0.0))

                    # [FIX 2026-06-21 Opus v11 §D] Surface the HONEST detection metric.
                    # det_mAP50 is the COCO-24 mean: it averages AP over ALL 24 channels,
                    # including the background channel (0) and every channel with zero GT in
                    # this val subset (each contributes AP=0). On sparse subset stages this
                    # DILUTES the headline far below real per-present-class performance.
                    # det_mAP50_pc averages only channels with GT>0 — the number to judge
                    # subset-stage progress by. Logging-only; the gate still uses det_mAP50 for
                    # paper-baseline comparability (see 44_OPUS_ANSWER_v11.md §D).
                    _map50_pc = _s(val_metrics.get('det_mAP50_pc', 0.0))
                    _n_present = int(val_metrics.get('det_n_present_classes', 0))
                    _n_total = int(getattr(C, 'NUM_DET_CLASSES', 24))
                    if _map50_pc > 0 or _map50 > 0:
                        logger.info(
                            f'  det_mAP50={_map50:.4f} (COCO-{_n_total}, diluted)  '
                            f'det_mAP50_pc={_map50_pc:.4f} (present-class, honest)  '
                            f'n_present={_n_present}/{_n_total}'
                        )
                        if (_map50_pc - _map50) >= 0.05:
                            logger.warning(
                                f'  [DILUTION] det_mAP50_pc exceeds det_mAP50 by '
                                f'{_map50_pc - _map50:+.4f} — the headline is dragged down by '
                                f'{_n_total - _n_present} zero-GT/background channels. '
                                f'Judge subset-stage progress by det_mAP50_pc.'
                            )

                    # [NaN Guard] Validate inputs before computing combined metric
                    # [FIX 2026-05-31] OR instead of AND — fire when ANY component is non-finite
                    # [FIX 2026-06-06] Clamp non-finite components to NEUTRAL values (0.0 for f1/map50,
                    #                     _MAE_POSE_NEUTRAL=360° for head_pose_MAE) instead of forcing
                    #                     combined=0.0. This lets GOOD components still contribute to
                    #                     combined metric and allows best-checkpoint saving to function
                    #                     in mixed-NaN scenarios (e.g. eff_*=NaN, but PSR/DET valid).
                    _MAE_POSE_NEUTRAL = 360.0  # degrees — neutral fallback for head_pose_MAE
                    _orig_components = (_map50, _f1_act, _mae_pose, _f1_psr)
                    _neutrals = (0.0, 0.0, _MAE_POSE_NEUTRAL, 0.0)
                    _clamped = tuple(
                        _c if math.isfinite(_c) else _n
                        for _c, _n in zip(_orig_components, _neutrals)
                    )
                    if _clamped != _orig_components:
                        logger.warning(
                            f'  [COMBINED_NAN] Non-finite component(s) clamped to neutral — '
                            f'map50={_orig_components[0]}->{_clamped[0]}, '
                            f'f1_act={_orig_components[1]}->{_clamped[1]}, '
                            f'mae_pose={_orig_components[2]}->{_clamped[2]}, '
                            f'f1_psr={_orig_components[3]}->{_clamped[3]}'
                        )
                    _map50, _f1_act, _mae_pose, _f1_psr = _clamped
                    # [HONEST METRIC 2026-06-22] best.pth + stage gate are driven by the
                    # combined metric below. Feed it the present-class mAP (det_mAP50_pc,
                    # computed above), not the diluted COCO-24 det_mAP50. This is the single
                    # change that stops the project chasing a ~40%-artifact headline.
                    _map50_decision = _map50_pc if _n_present > 0 else _map50
                    combined = _compute_combined_metric(
                        _map50_decision, _f1_act, _mae_pose, _f1_psr,
                        active_det=CFG_TRAIN_DET,
                        active_act=CFG_TRAIN_ACT,
                        active_pose=CFG_TRAIN_HEAD_POSE,
                        active_psr=CFG_TRAIN_PSR,
                    )
                    val_metrics['combined'] = combined
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
                            'global_step':    getattr(C, '_global_step', 0),
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
                        _atomic_save(save_dict, ckpt_dir / 'best.pth')
                        logger.info(
                            f'  ** New best model (combined={combined:.4f}) **'
                        )
                    else:
                        patience_counter += 1
                        logger.info(
                            f'  No improvement ({patience_counter}/{C.PATIENCE})'
                        )

                    # Write heartbeat with best_metric and validation results
                    _write_stage_heartbeat(
                        epoch, training_pid=os.getpid(),
                        best_metric=best_metric,
                        best_metrics=val_metrics,
                    )

                    if patience_counter >= C.PATIENCE:
                        logger.info(
                            f'Early stopping at epoch {epoch} '
                            f'(patience={C.PATIENCE})'
                        )
                        break

            # --- Stage Manager: signal early stop if all gate targets met ---
            if os.environ.get('_STAGE_MANAGER_ACTIVE') == '1':
                _sg_gate_json = os.environ.get('_STAGE_GATE_JSON', '{}')
                _sg_met_file = os.environ.get('_STAGE_TARGET_MET_FILE', '')
                if _sg_gate_json and _sg_met_file:
                    try:
                        _sg_targets = json.loads(_sg_gate_json)
                        _sg_key_map = {'act_top1': 'act_clip_accuracy', 'det_mAP50_95': 'det_mAP_50_95'}
                        _sg_all_met = True
                        _sg_details = {}
                        for _sg_metric, _sg_threshold in _sg_targets.items():
                            _sg_actual = _sg_key_map.get(_sg_metric, _sg_metric)
                            _sg_v = val_metrics.get(_sg_actual)
                            if _sg_v is None or math.isnan(_sg_v):
                                _sg_all_met = False
                                _sg_details[_sg_metric] = {'value': _sg_v, 'threshold': _sg_threshold, 'status': 'UNKNOWN'}
                                continue
                            if 'MAE' in _sg_metric or 'mae' in _sg_metric:
                                _sg_passed = _sg_v <= _sg_threshold
                            else:
                                _sg_passed = _sg_v >= _sg_threshold
                            _sg_details[_sg_metric] = {'value': _sg_v, 'threshold': _sg_threshold, 'status': 'PASS' if _sg_passed else 'FAIL'}
                            if not _sg_passed:
                                _sg_all_met = False
                        if _sg_all_met:
                            Path(_sg_met_file).write_text(
                                json.dumps({'epoch': epoch, 'metrics': val_metrics, 'gate_details': _sg_details})
                            )
                            logger.info('*** STAGE TARGET MET — all gate thresholds reached. Signalling stage_manager. ***')
                            break
                    except Exception as _sg_e:
                        logger.warning(f'Stage gate check error: {_sg_e}')

            _atomic_save({
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
                'global_step':   getattr(C, '_global_step', 0),
            }, ckpt_dir / 'latest.pth')

            # --- PER-EPOCH CHECKPOINT (Bashara 2026-06-30) ---
            # Save a named checkpoint for this epoch so we can roll back to any
            # epoch's state. Only keep the last 23 (RF4 default) to avoid filling disk.
            # Each checkpoint is ~500 MB; 23 × 500 MB ≈ 11.5 GB, fine on 1.3 TB free.
            _atomic_save({
                'epoch':            epoch,
                'model':           model.state_dict(),
                'optimizer':       optimizer.state_dict(),
                'scheduler':       scheduler.state_dict(),
                'scaler':          scaler.state_dict(),
                'best_metric':     best_metric,
                'patience_counter': patience_counter,
                'ema_shadow':      {k: v.clone() for k, v in ema.shadow.items()} if ema is not None else {},
                'criterion': {
                    'log_var_det': criterion.log_var_det.data.clone(),
                    'log_var_pose': criterion.log_var_pose.data.clone(),
                    'log_var_act': criterion.log_var_act.data.clone(),
                    'log_var_psr': criterion.log_var_psr.data.clone(),
                } if criterion is not None else {},
                'global_step':   getattr(C, '_global_step', 0),
            }, ckpt_dir / f'epoch_{epoch}.pth')
            logger.info(f'  [EPOCH_CKPT] Saved epoch_{epoch}.pth')

            # [DISK-GUARD] Prune epoch checkpoints older than 30 epochs back
            # to prevent filling disk during long runs (23 epochs × 500 MB ≈ 11.5 GB).
            # Keep at most 30 so we have enough rollback points for paper ablations.
            try:
                _epoch_ckpts = sorted(ckpt_dir.glob('epoch_*.pth'), key=lambda p: p.stat().st_mtime)
                while len(_epoch_ckpts) > 30:
                    _old = _epoch_ckpts.pop(0)
                    _old.unlink(missing_ok=True)
                    logger.info(f'  [EPOCH_CKPT] Pruned old checkpoint: {_old.name}')
            except Exception:
                pass  # non-critical

            # --- BASHARA 2026-05-25: Per-epoch crash recovery save (epoch end only) ---
            _save_crash_recovery(f'epoch_{epoch}_end')

            # [2pct FIX] Exit epoch loop AFTER validation completes when step limit reached.
            # Previously this was BEFORE val, causing TRAIN_MAX_STEPS to skip validation.
            if _train_max_steps > 0 and C._global_step >= _train_max_steps:
                logger.info(
                    f'  [2pct] TRAIN_MAX_STEPS={_train_max_steps} reached '
                    f'(global_step={C._global_step}). Exiting epoch loop after val.'
                )
                break  # Exit for epoch loop — val completed, training done

            record = {
                'epoch': epoch,
                'lr': current_lr,
                'train': train_metrics,
            }
            if val_metrics:
                record['val'] = val_metrics

            # [NaN Guard] Sanitize record before JSON serialization to prevent
            # NaN/Inf values from corrupting the metrics log. JSON cannot represent NaN.
            def _sanitize(obj):
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                elif isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return 0.0
                    return obj
                return obj

            log_file.write(json.dumps(_sanitize(record), default=str) + '\n')
            log_file.flush()
        _watchdog_active = False

    finally:
        _watchdog_active = False
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
                    val_ds=None,
                    val_every_n_steps=0,
                    distill_loss_fn=None,
                )
                swa_scheduler.step()
                logger.info(
                    f'SWA train: loss={swa_train_metrics["total"]:.4f}  '
                    f'lr={optimizer.param_groups[2]["lr"]:.2e}'
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
            _atomic_save(swa_state, ckpt_dir / 'swa.pth')
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
        help='Config preset name from config.PRESETS '
             "(e.g. 'recovery', 'benchmark_full', 'benchmark_quick').",
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
    parser.add_argument(
        '--reset-scheduler',
        action='store_true',
        help='[Recovery] Reset scheduler state to epoch 0 after loading checkpoint. '
             'Use with --start-epoch when reinit-heads changes the effective epoch counter.',
    )
    parser.add_argument(
        '--reinit-heads',
        action='store_true',
        help='[Recovery] Re-initialize det/act/psr heads + FPN from priors before training. '
             'Keeps backbone + pose_head + pretrained ConvNeXt. Resets FPN with Kaiming-uniform, '
             'det.cls_score pi=REINIT_PI (config-driven, default 0.01), act full reinit, psr bias=-0.2. '
             'Use after head collapse (all 3 heads producing constant output).',
    )
    parser.add_argument(
        '--detach-reg-fpn',
        action='store_true',
        help='[RF1] Detach FPN features for regression subnet to prevent regression gradients '
             'from corrupting shared FPN features. Fixes detection head collapse after --reinit-heads.',
    )
    parser.add_argument(
        '--detach-psr-fpn',
        action='store_true',
        help='[RF1] Detach FPN features for PSR head to prevent PSR loss spikes from corrupting '
             'shared FPN features. Use with --detach-reg-fpn for full gradient isolation in RF stages.',
    )

    args = parser.parse_args()

    if args.preset:
        # [AUDIT FIX 2026-06-11 — config split-brain] `import config as _cfg_mod`
        # resolved to the root config.py copy, a DIFFERENT module object from
        # `from src import config as C` used by this file, model.py and
        # losses.py. apply_preset() therefore mutated globals nobody read —
        # `--preset recovery` (zero_det_conf, FP32, staged off) was a silent
        # no-op for training. Apply the preset on C directly, and let an
        # unknown preset name crash loudly instead of degrading to a warning.
        C.apply_preset(args.preset)
        _refresh_runtime_cfg()
        logger.info(
            f'[train] Applied preset: {args.preset} '
            f'(MIXED_PRECISION={C.MIXED_PRECISION}, STAGED_TRAINING={C.STAGED_TRAINING}, '
            f'ZERO_DET_CONF_FOR_RECOVERY={C.ZERO_DET_CONF_FOR_RECOVERY}, '
            f'USE_EMA={C.USE_EMA}, USE_MIXUP={C.USE_MIXUP}, '
            f'BATCH_SIZE={C.BATCH_SIZE}x{C.GRAD_ACCUM_STEPS})'
        )

    # [RF1 FIX 2026-06-18] When stage_rf* preset is used directly (bypassing
    # stage_manager.py), update_dynamic_paths() computed OUTPUT_ROOT from model
    # features (convnext+tma+tbank → "full_multi_task_tma_tbank") at import time
    # before the preset was applied. Redirect to runs/rf_stages/ so checkpoints,
    # logs, and eval outputs land in the expected directory alongside stage_manager
    # runs, instead of scattering into a feature-derived directory name.
    if args.preset and args.preset.startswith('stage_rf'):
        _stage_root = Path(C.OUTPUT_ROOT).parent / 'rf_stages'
        os.environ['OUTPUT_ROOT_OVERRIDE'] = str(_stage_root)
        C.update_dynamic_paths()
        logger.info(f'[train] Stage preset detected — redirected OUTPUT_ROOT to {_stage_root}')

    if args.max_epochs is not None:
        C.EPOCHS = args.max_epochs
    if args.batch_size is not None:
        C.BATCH_SIZE = args.batch_size
        C.EFFECTIVE_BATCH = C.BATCH_SIZE * C.GRAD_ACCUM_STEPS
    if args.num_workers is not None:
        C.NUM_WORKERS = args.num_workers
        logger.info(f'[train] num_workers overridden to {args.num_workers}')

    if getattr(args, 'detach_reg_fpn', False):
        C.DETACH_REG_FPN = True
        logger.info('[train] DETACH_REG_FPN=True — regression FPN features detached, preventing gradient shock')

    if getattr(args, 'detach_psr_fpn', False):
        C.DETACH_PSR_FPN = True
        logger.info('[train] DETACH_PSR_FPN=True — PSR FPN features detached, preventing PSR gradient backbone corruption')

    # TRAIN_MAX_STEPS: limit total optimizer steps for quick runs (env var)
    _env_max_steps = int(os.environ.get('TRAIN_MAX_STEPS', '0'))
    if _env_max_steps > 0:
        C.TRAIN_MAX_STEPS = _env_max_steps
        logger.info(f'[train] TRAIN_MAX_STEPS={_env_max_steps} — will early-stop at this step count')

    # EVAL_MAX_BATCHES: cap val batches per epoch (env var) — for quick smoke runs
    _env_eval_max_batches = int(os.environ.get('EVAL_MAX_BATCHES', '0'))
    if _env_eval_max_batches > 0:
        C.EVAL_MAX_BATCHES = _env_eval_max_batches
        logger.info(f'[train] EVAL_MAX_BATCHES={_env_eval_max_batches} — will cap val at this many batches per epoch')

    _override_start_epoch = None  # set by --start-epoch
    if args.no_staged_training:
        C.STAGED_TRAINING = False
        logger.info('[train] STAGED_TRAINING=False — all 5 heads active from epoch 0')
    if args.start_epoch is not None:
        _override_start_epoch = args.start_epoch
        logger.info(f'[train] start_epoch override: {_override_start_epoch} (fresh init, no checkpoint)')

    if hasattr(args, 'debug') and args.debug:
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
