#!/usr/bin/env python3
"""Diagnose PSR NaN on TRAIN data with multiple batches.

Hypothesis: Specific train samples at certain indices cause NaN.
The training sampler is deterministic, so we can find the bad index.
"""
import sys, os
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
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
import model as _popw_model_module
from training.train import seed_everything, _reinit_dead_heads
from training.losses import MultiTaskLoss

CKPT = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth'

print(f'[psr-train] loading {CKPT}', flush=True)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f'[psr-train] ckpt epoch={ckpt.get("epoch")}', flush=True)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use TRAIN data (this is what training uses)
train_ds = _ds_module.IndustRealMultiTaskDataset(
    split='train', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
)
collate_fn = _ds_module.collate_fn_sequences if C.USE_PSR_SEQUENCE_MODE else _ds_module.collate_fn

# bs=4 (OOM recovery state) — but with no_grad to fit in 12GB
BATCH = 4
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0,
    collate_fn=collate_fn, pin_memory=False, drop_last=True)

model = _popw_model_module.POPWMultiTaskModel(
    pretrained=True, backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
    use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
    use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FIM', False)),
    use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
    train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
).to(device)
model._seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 4) if C.USE_PSR_SEQUENCE_MODE else 1
state = {k.replace('ema.', ''): v for k, v in ckpt['model'].items() if not k.startswith('ema.')}
res = model.load_state_dict(state, strict=False)
print(f'[psr-train] load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}', flush=True)
_reinit_dead_heads(model)
print(f'[psr-train] reinit done', flush=True)

# Build the criterion (this is what's used in training)
criterion = MultiTaskLoss().to(device)

model.eval()  # eval mode for memory efficiency (we only check loss, not backward)
nan_count = 0
checked = 0
for bi, (images, targets) in enumerate(train_loader):
    if bi >= 200: break  # check first 200 batches (covers step 0-199 of failed run)
    # Mimic train.py prep
    if images.ndim == 5:
        B, T, C_, H, W = images.shape
        images_in = images.view(B * T, C_, H, W).to(device).float() / 255.0
    else:
        images_in = images.to(device).float() / 255.0
        B = images_in.shape[0]
        T = 1
    targets_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

    with torch.no_grad():
        out = model(images_in)
        for k in ('cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits'):
            if k in out and isinstance(out[k], torch.Tensor):
                out[k] = out[k].float()

        try:
            loss, loss_dict = criterion(out, targets_dev)
            loss_finite = torch.isfinite(loss).item()
            nan_flag = loss_dict.get('__nan_detected__', False)
            psr_raw = loss_dict.get('psr', 'N/A')
            if not loss_finite or nan_flag:
                nan_count += 1
                print(f'[psr-train] batch {bi:4d} bs={BATCH}: LOSS NaN! finite={loss_finite} nan_flag={nan_flag} psr={psr_raw} '
                      f'logits_min/max/std={out["psr_logits"].min().item():.3f}/{out["psr_logits"].max().item():.3f}/{out["psr_logits"].std().item():.4f}', flush=True)
                if nan_count <= 3:
                    psr_lab = targets_dev['psr_labels']
                    print(f'  psr_labels: shape={tuple(psr_lab.shape)} dtype={psr_lab.dtype} '
                          f'unique={sorted(set(psr_lab.flatten().tolist()[:50]))} '
                          f'has_neg1={int((psr_lab < 0).sum())} has_nan={int(torch.isnan(psr_lab.float()).sum())}', flush=True)
        except Exception as e:
            print(f'[psr-train] batch {bi:4d}: EXCEPTION {e}', flush=True)
    checked += 1
    if bi % 50 == 0:
        print(f'[psr-train] progress: {bi}/200 checked, {nan_count} NaN', flush=True)

print(f'\n[psr-train] SUMMARY: {nan_count}/{checked} batches had NaN loss')
