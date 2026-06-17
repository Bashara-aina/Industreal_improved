#!/usr/bin/env python3
"""Diagnose WHERE the PSR NaN comes from in the OOM-recovery run.

Tests:
1. Load crash_recovery.pth (with reinit)
2. Run a few val batches (B=4 to mimic OOM recovery)
3. Check psr_logits for NaN/Inf
4. Check psr_labels for NaN/Inf
5. Manually compute the loss components
6. Report which step/element produces NaN
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
from training.train import seed_everything, _reinit_dead_heads

CKPT = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth'

print(f'[psr-diag] loading {CKPT}', flush=True)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f'[psr-diag] ckpt epoch={ckpt.get("epoch")} step={ckpt.get("step")}', flush=True)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_ds = _ds_module.IndustRealMultiTaskDataset(
    split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    max_recordings=4,
)
collate_fn = _ds_module.collate_fn_sequences if C.USE_PSR_SEQUENCE_MODE else _ds_module.collate_fn
# Test BOTH batch=8 and batch=4
for test_bs in [8, 4]:
    print(f'\n{"="*60}\nTest batch_size={test_bs}\n{"="*60}', flush=True)
    val_loader = DataLoader(val_ds, batch_size=test_bs, shuffle=False, num_workers=0,
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
    print(f'[psr-diag] load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}', flush=True)

    # Re-init heads (just like the failed reinit run did)
    _reinit_dead_heads(model)
    print(f'[psr-diag] reinit done', flush=True)

    model.train(False)
    n_batches = 3
    for bi, (images, targets) in enumerate(val_loader):
        if bi >= n_batches: break
        # Mimic the train.py prep
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

        # Check psr_logits
        pl = out['psr_logits']
        pl_nan = torch.isnan(pl).any().item()
        pl_inf = torch.isinf(pl).any().item()
        pl_finite = torch.isfinite(pl).all().item()
        # Check psr_labels
        psr_lab = targets_dev['psr_labels']
        lab_nan = torch.isnan(psr_lab.float()).any().item() if psr_lab.is_floating_point() else False
        lab_any_nonnumeric = False
        if not psr_lab.is_floating_point():
            # Check for -1 / invalid sentinel in int labels
            lab_any_nonnumeric = ((psr_lab < 0) | (psr_lab > 1)).any().item()
        print(f'[psr-diag] batch {bi} bs={test_bs}: '
              f'psr_logits shape={tuple(pl.shape)} '
              f'min={pl.min().item():.4f} max={pl.max().item():.4f} std={pl.std().item():.4f} '
              f'NaN={pl_nan} Inf={pl_inf} all_finite={pl_finite}', flush=True)
        print(f'[psr-diag]   psr_labels shape={tuple(psr_lab.shape)} dtype={psr_lab.dtype} '
              f'unique={sorted(set(psr_lab.flatten().tolist()[:50]))} '
              f'NaN={lab_nan} out_of_range={lab_any_nonnumeric}', flush=True)

        # Manually compute the loss for psr using reshape (handles non-contiguous)
        from training.losses import binary_focal_loss
        pl_flat = pl.reshape(-1).contiguous()
        lab_flat = psr_lab.reshape(-1).contiguous().float()
        try:
            loss_focal = binary_focal_loss(pl_flat, lab_flat, alpha=0.25, gamma=2.0)
            print(f'[psr-diag]   manual focal loss: {loss_focal.item():.4f} finite={torch.isfinite(loss_focal).item()}', flush=True)
        except Exception as e:
            print(f'[psr-diag]   manual focal loss FAILED: {e}', flush=True)
        try:
            loss_bce = nn.functional.binary_cross_entropy_with_logits(pl_flat, lab_flat, reduction='mean')
            print(f'[psr-diag]   manual bce loss: {loss_bce.item():.4f} finite={torch.isfinite(loss_bce).item()}', flush=True)
        except Exception as e:
            print(f'[psr-diag]   manual bce loss FAILED: {e}', flush=True)

        # Also try: detach+contiguous, like the real criterion
        pl_d = pl.detach().contiguous()
        lab_d = psr_lab.detach().contiguous().float()
        try:
            loss_focal2 = binary_focal_loss(pl_d, lab_d, alpha=0.25, gamma=2.0)
            print(f'[psr-diag]   manual focal loss (detach+contig): {loss_focal2.item():.4f} finite={torch.isfinite(loss_focal2).item()}', flush=True)
        except Exception as e:
            print(f'[psr-diag]   manual focal loss 2 FAILED: {e}', flush=True)

        del model, out, pl, psr_lab
        import gc; gc.collect(); torch.cuda.empty_cache()
        break  # one batch per bs is enough

print()
print('=' * 60)
print('PSR-NAAN-DIAG COMPLETE')
print('=' * 60)
