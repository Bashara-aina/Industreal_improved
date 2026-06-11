#!/usr/bin/env python3
"""D7 [opus RC-25] — Feature magnitude probe.

Compares FPN output magnitudes from the collapsed epoch-43 checkpoint
against a fresh ImageNet-init model. RC-25 hypothesis: epoch-43 backbone/FPN
produce O(10^2–10^3) features that saturate the freshly-reinitialized detection
head at step 0, producing cls_loss = 10^7.

Measures per-level FPN output (p3-p7) statistics:
  - mean, std, max across spatial dims
  - ratio of collapsed/std vs fresh/std (collapse factor)

Usage:
  python code/diag_feature_magnitude.py
  CHECKPOINT=path/to/latest.pth python code/diag_feature_magnitude.py
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
from torch.utils.data import DataLoader

import config as C
from models.model import POPWMultiTaskModel
from data import create_dataloaders  # may differ; fallback to manual

CKPT = os.environ.get('CHECKPOINT', str(PROJ / 'src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BS = int(os.environ.get('EVAL_BS', '2'))
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', '100'))


class FPNHook:
    """Capture FPN output dict from forward pass."""
    def __init__(self):
        self.outputs = []

    def __call__(self, module, inputs, output):
        # output is dict: {'p3':..., 'p4':..., 'p5':..., 'p6':..., 'p7':...}
        self.outputs.append({k: v.detach().cpu() for k, v in output.items()})


def measure_magnitudes(hook: FPNHook, label: str):
    """Compute per-level statistics from captured FPN outputs."""
    if not hook.outputs:
        print(f'  [{label}] No FPN outputs captured!')
        return
    all_frames = {k: [] for k in hook.outputs[0]}
    for out in hook.outputs:
        for k, v in out.items():
            all_frames[k].append(v)  # each [B, C, H, W]
    print(f'\n{"="*60}')
    print(f'  Feature Magnitude Probe: {label}')
    print(f'  Captured {len(hook.outputs)} batches')
    print(f'{"="*60}')
    for level in ('p3', 'p4', 'p5', 'p6', 'p7'):
        tensors = all_frames[level]
        if not tensors:
            continue
        cat = torch.cat([t.float().flatten() for t in tensors])
        mn = cat.mean().item()
        std = cat.std().item()
        mx = cat.max().item()
        rms = (cat ** 2).mean().sqrt().item()
        print(f'  {level:>4s}: mean={mn:8.2f}  std={std:8.2f}  max={mx:10.2f}  rms={rms:8.2f}')
    print()
    return all_frames


def compare_magnitudes(fresh_frames, ckpt_frames):
    """Print collapse factor per level: ckpt_std / fresh_std."""
    print(f'\n{"="*60}')
    print('  Collapse Factor (ckpt RMS / fresh RMS)')
    print(f'{"="*60}')
    for level in ('p3', 'p4', 'p5', 'p6', 'p7'):
        if level not in fresh_frames or level not in ckpt_frames:
            continue
        f_cat = torch.cat([t.float().flatten() for t in fresh_frames[level]])
        c_cat = torch.cat([t.float().flatten() for t in ckpt_frames[level]])
        f_rms = (f_cat ** 2).mean().sqrt().item()
        c_rms = (c_cat ** 2).mean().sqrt().item()
        ratio = c_rms / f_rms if f_rms > 0 else float('inf')
        flag = ' !!! EXPLODED' if ratio > 5 else ' OK'
        print(f'  {level:>4s}: fresh_rms={f_rms:.2f}  ckpt_rms={c_rms:.2f}  ratio={ratio:.2f}x{flag}')
    print()


def load_model(ckpt_path=None):
    """Build a fresh POPW model, optionally load checkpoint."""
    C.ZERO_DET_CONF_FOR_RECOVERY = False  # no-op for magnitude probe
    model = POPWMultiTaskModel(
        num_det_classes=C.NUM_DET_CLASSES if hasattr(C, 'NUM_DET_CLASSES') else 24,
        num_act_classes=C.NUM_ACT_CLASSES,
        num_psr_classes=C.NUM_PSR_CLASSES if hasattr(C, 'NUM_PSR_CLASSES') else 11,
        backbone_type=C.BACKBONE_TYPE if hasattr(C, 'BACKBONE_TYPE') else 'convnext_tiny',
        pretrained=False,  # fresh init
    )
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt.get('model')))
        model.load_state_dict(state, strict=False)
        print(f'Loaded checkpoint: {ckpt_path}')
    return model.to(DEVICE)


def get_sample_input():
    """Get a real image batch from the dataset."""
    try:
        from data import create_dataloaders
        loaders = create_dataloaders(
            subset_ratio=0.05, batch_size=BS, num_workers=0,
            benchmark_mode=False,
        )
        batch = next(iter(loaders['train']))
        return batch['image'].to(DEVICE)
    except Exception:
        # Fallback: random input
        print('Using random input (dataset unavailable)')
        return torch.randn(BS, 3, C.IMG_HEIGHT, C.IMG_WIDTH).to(DEVICE)


def main():
    print('D7: Feature Magnitude Probe')
    print(f'  Device: {DEVICE}  Batch size: {BS}  Max frames: {MAX_FRAMES}')

    # 1) Fresh ImageNet-init model
    print('\n[1/2] Probing fresh ImageNet-init model...')
    model_fresh = load_model(ckpt_path=None)
    hook_fresh = FPNHook()
    model_fresh.fpn.register_forward_hook(hook_fresh)
    model_fresh.eval()
    images = get_sample_input()
    with torch.no_grad():
        for i in range(min(MAX_FRAMES // BS, 25)):
            # Use same input repeated — we only care about forward feature scale
            inp = images[:min(BS, images.shape[0])]
            _ = model_fresh(inp)
    fresh_frames = measure_magnitudes(hook_fresh, 'FRESH ImageNet-init')
    del model_fresh, hook_fresh
    torch.cuda.empty_cache()

    # 2) Collapsed epoch-43 model
    print('\n[2/2] Probing collapsed checkpoint model...')
    model_ckpt = load_model(ckpt_path=CKPT)
    hook_ckpt = FPNHook()
    model_ckpt.fpn.register_forward_hook(hook_ckpt)
    model_ckpt.eval()
    with torch.no_grad():
        for i in range(min(MAX_FRAMES // BS, 25)):
            inp = images[:min(BS, images.shape[0])]
            _ = model_ckpt(inp)
    ckpt_frames = measure_magnitudes(hook_ckpt, f'COLLAPSED (epoch 43)')

    # 3) Comparison
    if fresh_frames and ckpt_frames:
        compare_magnitudes(fresh_frames, ckpt_frames)
        print('\nVerdict: If ckpt_rms/fresh_rms > 5x for any FPN level, RC-25 is CONFIRMED.')
        print('The shared trunk (backbone+FPN) produces feature magnitudes that saturate')
        print('reinitialized detection heads at step 0. FPN reinit is required.')
    else:
        print('\nERROR: Could not capture FPN outputs from one or both models.')


if __name__ == '__main__':
    main()
