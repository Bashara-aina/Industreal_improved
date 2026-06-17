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
# [AUDIT FIX 2026-06-11] top-level `from data import create_dataloaders`
# crashed the whole script when that symbol doesn't exist (the data package
# exposes IndustRealMultiTaskDataset + collate_fn, not create_dataloaders).
# Dataset access now happens lazily inside get_sample_input() with a fallback.

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
    """Build a POPW model, optionally load checkpoint.

    [AUDIT FIX 2026-06-11] Two correctness bugs fixed:
    1. The constructor was called with kwargs that don't exist
       (num_det_classes/num_act_classes/num_psr_classes/BACKBONE_TYPE) —
       TypeError on every run; this script had never executed.
    2. The control model used pretrained=False (RANDOM torchvision init).
       The RC-25 baseline is the ImageNet-pretrained trunk — random-init
       feature magnitudes are a wrong denominator for the collapse factor.
       Control is now pretrained=True; FPN/heads are randomly initialized
       identically either way.
    use_videomae=False keeps the probe light (FPN magnitudes are independent
    of the VideoMAE stream).
    """
    C.ZERO_DET_CONF_FOR_RECOVERY = False  # no-op for magnitude probe
    model = POPWMultiTaskModel(
        pretrained=(ckpt_path is None),  # ImageNet control vs ckpt-overwritten
        backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
        use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', True)),
        use_videomae=False,
        train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
    )
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt.get('model')))
        # Filter incompatible keys (pre-GroupNorm checkpoint vs post-GroupNorm model)
        model_shape = {k: v.shape for k, v in model.state_dict().items()}
        compat = {k: v for k, v in state.items()
                  if k in model_shape and v.shape == model_shape[k]}
        skipped = len(state) - len(compat)
        model.load_state_dict(compat, strict=False)
        print(f'Loaded checkpoint: {ckpt_path} (loaded={len(compat)} skipped={skipped})')
    return model.to(DEVICE)


def _normalize_uint8(images: torch.Tensor) -> torch.Tensor:
    """uint8 [B,3,H,W] -> float ImageNet-normalized (mirrors train._prepare_images)."""
    if images.dtype == torch.uint8:
        images = images.float().div(255.0)
        mean = torch.tensor(C.IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


def get_sample_input():
    """Get a real, ImageNet-normalized image batch from the dataset."""
    try:
        from data import IndustRealMultiTaskDataset, collate_fn
        ds = IndustRealMultiTaskDataset(
            split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
        )
        loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
        # collate_fn returns (images, targets) — a tuple, not a dict.
        images, _targets = next(iter(loader))
        return _normalize_uint8(images).to(DEVICE)
    except Exception as exc:
        # Fallback: random input. NOTE: random N(0,1) input is acceptable for
        # the RELATIVE fresh-vs-ckpt comparison but absolute magnitudes will
        # differ from real frames — prefer real data when available.
        print(f'Using random input (dataset unavailable: {exc!r})')
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
