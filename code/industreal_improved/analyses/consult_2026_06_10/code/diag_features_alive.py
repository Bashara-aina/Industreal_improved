#!/usr/bin/env python
"""Test: are backbone features alive? If re-initializing the dead heads makes the
model output non-constant predictions, the features are usable and we can recover.

This script:
1. Loads crash_recovery.pth
2. Re-initializes 3 dead heads (det cls, activity, PSR)
3. Runs 1 batch through model
4. Reports whether outputs vary across images
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
import math
import numpy as np
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
import model as _popw_model_module
from training.train import seed_everything

CKPT = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth'
print(f'[featalive] loading {CKPT}', flush=True)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f'[featalive] ckpt epoch={ckpt.get("epoch")}', flush=True)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_ds = _ds_module.IndustRealMultiTaskDataset(
    split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    max_recordings=4,
)
collate_fn = _ds_module.collate_fn_sequences if C.USE_PSR_SEQUENCE_MODE else _ds_module.collate_fn
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0,
    collate_fn=collate_fn, pin_memory=False, drop_last=False)

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
print(f'[featalive] load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}', flush=True)

# RE-INIT detection head cls_score
print()
print('=' * 60)
print('Re-initializing dead heads...')
print('=' * 60)
if hasattr(model, 'det_head'):
    dh = model.det_head
    if hasattr(dh, 'cls_score'):
        pi = 0.01
        nn.init.normal_(dh.cls_score.weight, std=0.01)
        nn.init.constant_(dh.cls_score.bias, -math.log((1 - pi) / pi))
        print(f'[featalive] det.cls_score reinit: pi={pi} bias={-math.log((1-pi)/pi):.4f}', flush=True)
    if hasattr(dh, 'reg_pred'):
        nn.init.normal_(dh.reg_pred.weight, std=0.01)
        nn.init.zeros_(dh.reg_pred.bias)
        print(f'[featalive] det.reg_pred reinit', flush=True)

# RE-INIT activity classifier
if hasattr(model, 'act_head'):
    ah = model.act_head
    if hasattr(ah, 'activity_classifier'):
        for m in ah.activity_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print(f'[featalive] act.activity_classifier reinit', flush=True)

# RE-INIT PSR output heads
if hasattr(model, 'psr_head'):
    ph = model.psr_head
    if hasattr(ph, 'output_heads'):
        for h in ph.output_heads:
            for m in h.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -1.0)
        print(f'[featalive] psr.output_heads reinit (bias=-1.0)', flush=True)
    if hasattr(ph, 'per_frame_mlp'):
        for m in ph.per_frame_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print(f'[featalive] psr.per_frame_mlp reinit', flush=True)

print()
print('=' * 60)
print('Test: 2 batches, verify outputs vary across images')
print('=' * 60)

model.train(False)
n_batches = 2
for bi, (images, targets) in enumerate(val_loader):
    if bi >= n_batches: break
    images = images.to(device).float() / 255.0
    if images.ndim == 5:
        B, T, C_, H, W = images.shape
        images = images.view(B * T, C_, H, W)
    with torch.no_grad():
        out = model(images)
    print(f'[featalive] batch {bi}: B={images.shape[0]}', flush=True)
    # Detection
    cls_logits = out['cls_preds']
    cls_sig = torch.sigmoid(cls_logits)
    max_scores = cls_sig.max(dim=-1).values  # [B, N]
    print(f'[featalive]   det max score per img: min={max_scores.min(dim=-1).values.min().item():.4f} max={max_scores.max(dim=-1).values.max().item():.4f} mean={max_scores.mean().item():.4f}', flush=True)
    # Per-image max score (should differ if features alive)
    per_img_max = max_scores.max(dim=-1).values.cpu().numpy()
    print(f'[featalive]   det per_img_max: {per_img_max.tolist()}', flush=True)
    # Activity
    al = out.get('act_logits', out.get('activity_logits'))
    if al is None:
        # search for key
        act_keys = [k for k in out.keys() if 'act' in k.lower() and 'videomae' not in k.lower() and 'act_logits' in k.lower()]
        al = out[act_keys[0]] if act_keys else None
        if al is not None:
            print(f'[featalive]   activity key found: {act_keys[0]}', flush=True)
    if al is not None:
        per_img_pred = al.argmax(dim=-1).cpu().numpy()
        per_img_max_logit = al.max(dim=-1).values.cpu().numpy()
        print(f'[featalive]   act logits stats: shape={tuple(al.shape)} min={al.min().item():.4f} max={al.max().item():.4f} std={al.std().item():.4f}', flush=True)
        print(f'[featalive]   act per_img_argmax: {per_img_pred.tolist()}', flush=True)
        print(f'[featalive]   act per_img_max_logit: {[f"{v:.4f}" for v in per_img_max_logit]}', flush=True)
        # Per-class max logit variance (signs of life)
        per_class_max = al.max(dim=0).values.cpu().numpy()
        per_class_min = al.min(dim=0).values.cpu().numpy()
        per_class_std = al.std(dim=0).cpu().numpy()
        print(f'[featalive]   act per_class_std: min={per_class_std.min():.4f} max={per_class_std.max():.4f} mean={per_class_std.mean():.4f}', flush=True)
        print(f'[featalive]   act per_class_max (top5): {sorted(per_class_max.tolist())[-5:]}', flush=True)
    # PSR
    if 'psr_logits' in out:
        pl = out['psr_logits']
        print(f'[featalive]   psr logit stats: min={pl.min().item():.4f} max={pl.max().item():.4f} std={pl.std().item():.4f}', flush=True)
        pl_arr = pl.cpu().numpy()
        for i in range(pl_arr.shape[0]):
            binary = (torch.sigmoid(pl[i]).cpu().numpy() > 0.5).astype(int).tolist()
            print(f'[featalive]     psr img{i}: {binary}', flush=True)
    import gc; gc.collect(); torch.cuda.empty_cache()

print()
print('=' * 60)
print('FEATALIVE COMPLETE')
print('=' * 60)
