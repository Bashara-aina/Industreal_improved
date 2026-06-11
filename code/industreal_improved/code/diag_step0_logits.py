#!/usr/bin/env python3
"""D8 [opus RC-25] — Step-0 logit percentiles.

After FPN reinit (--reinit-heads), probes detection cls_logits on the first
training batch to verify the feature-magnitude fix. Computes percentiles of
cls_logits.abs() and checks median < 8.0 threshold.

Also prints:
  - % of logits with |z| > 8 (saturated sigmoid region)
  - % of logits with |z| > 20 (fully saturated)
  - per-class logit distribution

Usage:
  python code/diag_step0_logits.py
  CHECKPOINT=path/to/latest.pth REINIT_FPN=1 python code/diag_step0_logits.py
"""
import os, sys
from pathlib import Path
import numpy as np

PROJ = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '4')

import torch
import torch.nn as nn

import config as C
from models.model import POPWMultiTaskModel
from training.train import _reinit_dead_heads

CKPT = os.environ.get('CHECKPOINT', str(PROJ / 'src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BS = int(os.environ.get('EVAL_BS', '2'))
DO_REINIT = bool(int(os.environ.get('REINIT_FPN', '1')))


def load_and_reinit():
    """Load checkpoint, optionally reinit FPN+heads, return model.

    [AUDIT FIX 2026-06-11] The constructor was called with kwargs that don't
    exist (num_det_classes/...) — TypeError on every run; this script had
    never executed. Real signature: (pretrained, backbone_type,
    use_headpose_film, use_hand_film, use_videomae, train_pose).
    """
    C.ZERO_DET_CONF_FOR_RECOVERY = False
    model = POPWMultiTaskModel(
        pretrained=False,  # weights come from the checkpoint below
        backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
        use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', True)),
        use_videomae=False,  # det logits are independent of the VideoMAE stream
        train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
    ).to(DEVICE)

    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt.get('model')))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded checkpoint (missing={len(missing)} unexpected={len(unexpected)})')

    if DO_REINIT:
        print('Reinitializing FPN + heads via _reinit_dead_heads...')
        _reinit_dead_heads(model)

    return model


def main():
    print('D8: Step-0 Logit Percentiles')
    print(f'  Device: {DEVICE}  Batch size: {BS}  Reinit FPN: {DO_REINIT}')
    print(f'  Checkpoint: {CKPT}')

    model = load_and_reinit()
    model.eval()

    # Build a representative input.
    # [AUDIT FIX 2026-06-11] collate_fn returns (images, targets), not a dict,
    # and create_dataloaders doesn't exist. Images are uint8 and MUST be
    # ImageNet-normalized — feeding raw 0-255 floats would itself saturate the
    # head and produce a false RC-25 FAILED verdict.
    try:
        from torch.utils.data import DataLoader
        from data import IndustRealMultiTaskDataset, collate_fn
        ds = IndustRealMultiTaskDataset(
            split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
        )
        loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
        images, _targets = next(iter(loader))
        if images.dtype == torch.uint8:
            images = images.float().div(255.0)
            _mean = torch.tensor(C.IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
            _std = torch.tensor(C.IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
            images = (images - _mean) / _std
        images = images.to(DEVICE)
        print(f'Using real data: {images.shape}')
    except Exception as exc:
        print(f'Using random input (dataset unavailable: {exc!r})')
        images = torch.randn(BS, 3, C.IMG_HEIGHT, C.IMG_WIDTH).to(DEVICE)

    with torch.no_grad():
        out = model(images)
        cls_preds = out['cls_preds']  # [B, N_anchors, 24]

    cls_logits = cls_preds.float().flatten()
    abs_logits = cls_logits.abs()
    median_abs = abs_logits.median().item()
    p90 = abs_logits.kthvalue(int(abs_logits.numel() * 0.9)).values.item()
    p99 = abs_logits.kthvalue(int(abs_logits.numel() * 0.99)).values.item()
    p999 = abs_logits.kthvalue(int(abs_logits.numel() * 0.999)).values.item()
    mx = abs_logits.max().item()

    saturated_8 = (abs_logits > 8).float().mean().item() * 100
    saturated_20 = (abs_logits > 20).float().mean().item() * 100

    print(f'\n{"="*60}')
    print('  cls_logits distribution (abs values)')
    print(f'{"="*60}')
    print(f'  Total logits: {cls_logits.numel():,}')
    print(f'  median(|z|):  {median_abs:.3f}')
    print(f'  P90(|z|):     {p90:.3f}')
    print(f'  P99(|z|):     {p99:.3f}')
    print(f'  P99.9(|z|):   {p999:.3f}')
    print(f'  max(|z|):     {mx:.3f}')
    print(f'  |z| > 8:      {saturated_8:.1f}%  (sigmoid saturated region)')
    print(f'  |z| > 20:     {saturated_20:.1f}%  (fully saturated)')

    # RC-25 gate
    print(f'\n  RC-25 GATE: median(|z|) = {median_abs:.3f}')
    if median_abs < 8.0:
        print('  PASSED — logit scale in healthy range. Safe to train.')
    elif median_abs < 20.0:
        print('  MARGINAL — logit scale elevated. Expect high initial cls_loss.')
    else:
        print('  FAILED — logit scale saturated. Do NOT train without FPN reinit.')

    # Per-class stats
    cls_preds_reshaped = cls_preds.view(-1, 24)
    print(f'\n  Per-class |z| median:')
    for c in range(24):
        c_med = cls_preds_reshaped[:, c].abs().median().item()
        flag = ' !!' if c_med > 8 else ''
        print(f'    class {c:2d}: {c_med:.3f}{flag}')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
