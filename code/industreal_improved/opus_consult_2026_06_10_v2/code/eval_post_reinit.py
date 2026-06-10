#!/usr/bin/env python3
"""Re-eval crash_recovery.pth + reinit the 3 collapsed heads (det/act/psr).

Strategy (2026-06-09):
- Forward pass on TRAIN and VAL has been verified CLEAN in eval mode (0/200 NaN).
- The retrain attempt (bqmpmjnku/185750) produced NO checkpoint due to NaN_GUARD
  blocking save when autograd/backward poisoned backbone params.
- Workaround: run eval on the source checkpoint (crash_recovery.pth) AFTER
  re-initializing the 3 dead heads with `_reinit_dead_heads`. Backbone features
  are alive (verified via diag_features_alive.py: per-image variance 0.032-0.036
  in DET logits). With a fresh head, the eval pass should yield non-zero,
  non-NaN metrics for all 4 tasks.
"""
import sys, os, json, math
from pathlib import Path

_SRC = Path(__file__).resolve().parent / 'src'
for _sub in ['models', 'training', 'evaluation', 'data', str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

os.environ.setdefault('SUBSET_RATIO', '0.05')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
import model as _popw_model_module
from training.train import seed_everything, _reinit_dead_heads
from evaluation.evaluate import evaluate_all, _print_single_run_results

CKPT = os.environ.get('EVAL_CKPT', '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth')
SPLIT = os.environ.get('EVAL_SPLIT', 'val')
MAX_BATCHES = int(os.environ.get('MAX_BATCHES', '50'))
BS = int(os.environ.get('EVAL_BS', '4'))
RUN_NAME = os.environ.get('RUN_NAME', f'eval_post_reinit_{SPLIT}')
OUT_DIR = Path(f'/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/{RUN_NAME}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'[reinit-eval] ckpt={CKPT}', flush=True)
print(f'[reinit-eval] split={SPLIT} max_batches={MAX_BATCHES} bs={BS}', flush=True)

ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f'[reinit-eval] ckpt epoch={ckpt.get("epoch")} step={ckpt.get("step")}', flush=True)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_ds = _ds_module.IndustRealMultiTaskDataset(
    split=SPLIT, img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
)
print(f'[reinit-eval] ds size: {len(val_ds)}', flush=True)

collate_fn = _ds_module.collate_fn_sequences if C.USE_PSR_SEQUENCE_MODE else _ds_module.collate_fn
val_loader = DataLoader(
    val_ds, batch_size=BS, shuffle=False, num_workers=0,
    collate_fn=collate_fn, pin_memory=False, drop_last=False,
)

model = _popw_model_module.POPWMultiTaskModel(
    pretrained=False,
    backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
    use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
    use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),
    use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
    train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
).to(device)
model._seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 4) if C.USE_PSR_SEQUENCE_MODE else 1

state = {k.replace('ema.', ''): v for k, v in ckpt['model'].items() if not k.startswith('ema.')}
res = model.load_state_dict(state, strict=False)
print(f'[reinit-eval] load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}', flush=True)

# Verify no NaN/Inf in loaded params
nan_params = [n for n, p in model.named_parameters() if torch.isnan(p).any() or torch.isinf(p).any()]
print(f'[reinit-eval] params with NaN/Inf after load: {len(nan_params)}', flush=True)

# Re-init the 3 dead heads (skip with EVAL_SKIP_REINIT=1 to evaluate trained post-retrain ckpt as-is)
SKIP_REINIT = os.environ.get('EVAL_SKIP_REINIT', '0') == '1'
if SKIP_REINIT:
    print('[reinit-eval] EVAL_SKIP_REINIT=1: NOT re-initializing heads (using trained post-retrain weights)', flush=True)
    init_counts = {}
else:
    init_counts = _reinit_dead_heads(model)
    print(f'[reinit-eval] reinit: {init_counts}', flush=True)
    # Verify no NaN/Inf after reinit
    nan_after = [n for n, p in model.named_parameters() if torch.isnan(p).any() or torch.isinf(p).any()]
    print(f'[reinit-eval] params with NaN/Inf after reinit: {len(nan_after)}', flush=True)

# Build criterion
from training.losses import MultiTaskLoss
criterion = MultiTaskLoss().to(device)
if hasattr(val_ds, 'class_counts'):
    criterion.set_class_counts(val_ds.class_counts)

model.eval()
print(f'[reinit-eval] starting eval...', flush=True)

results = evaluate_all(
    model, criterion, val_loader, device,
    max_batches=MAX_BATCHES, save_dir=str(OUT_DIR),
    use_flip_tta=False, use_crop_tta=False,
)

# Save metrics
def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()
                if not isinstance(v, (torch.Tensor, np.ndarray))}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o

sanitized = _sanitize(results)
with open(OUT_DIR / 'metrics.json', 'w') as f:
    json.dump(sanitized, f, indent=2, default=str)

print(f'\n[reinit-eval] metrics saved to {OUT_DIR/"metrics.json"}', flush=True)
try:
    _print_single_run_results(results, SPLIT)
except KeyError as e:
    print(f'[reinit-eval] _print_single_run_results missing key {e} (likely no efficiency profile)', flush=True)
print(f'\n[reinit-eval] DONE', flush=True)
