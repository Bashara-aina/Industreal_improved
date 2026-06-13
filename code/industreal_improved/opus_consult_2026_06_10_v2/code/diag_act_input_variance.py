#!/usr/bin/env python3
"""D4 [opus RC-19] — Activity-input variance diagnostic.

Forwards 200 val frames through the post-retrain checkpoint and captures the
`activity_proj` tensor (the input to the activity head's `proj_features` MLP).
`activity_proj = cat([det_conf, GAP(c5_mod), GAP(p4)])`.

We measure the across-frame std/mean ratio for the det_conf slice alone, and
for the full `activity_proj`:
  - with `det_conf` AS-IS (raw, unbounded logits — the bug)
  - with `det_conf` zeroed
  - with `det_conf = sigmoid(raw_logits)` (P7 fix)

Verdict:
  - std/mean of `det_conf` < 1% with raw, ≥10× with sigmoid → RC-19 confirmed.
  - If full `activity_proj` becomes meaningfully variable only after the
    sigmoid (or zeroing), the activity head is downstream of detection.
"""
import os, sys
from pathlib import Path
import numpy as np

PROJ = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
from models import model as _popw_model_module

CKPT = os.environ.get('CHECKPOINT', str(PROJ / 'src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth'))
MAX_BATCHES = int(os.environ.get('MAX_BATCHES', '50'))   # 50 × bs=4 = 200 frames
BS = int(os.environ.get('EVAL_BS', '4'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _Hook:
    """Capture `activity_proj` from the live forward pass."""
    def __init__(self):
        self.captured = []

    def __call__(self, module, inputs):
        # module is `self.activity_head.proj_features`; its input is activity_proj.
        # `inputs` is a tuple of args forwarded to the Linear.
        if isinstance(inputs, tuple) and len(inputs) > 0:
            self.captured.append(inputs[0].detach().cpu())


def _stats(name: str, t: torch.Tensor) -> dict:
    if t.numel() == 0:
        return {'name': name, 'shape': tuple(t.shape), 'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'snr_pct': 0.0}
    t = t.float()
    # Across-frame std/mean per dim, then averaged across dims.
    # `t` is [N_frames, D]; compute per-dim std and abs mean, then ratio.
    if t.dim() == 2 and t.shape[0] > 1:
        per_dim_std = t.std(dim=0)
        per_dim_mean_abs = t.mean(dim=0).abs().clamp(min=1e-9)
        per_dim_snr = per_dim_std / per_dim_mean_abs
        return {
            'name': name, 'shape': tuple(t.shape),
            'mean': float(t.mean()),
            'std':  float(t.std()),
            'min':  float(t.min()),
            'max':  float(t.max()),
            'snr_pct': float(per_dim_snr.mean()) * 100.0,    # avg across dims, in %
        }
    return {
        'name': name, 'shape': tuple(t.shape),
        'mean': float(t.mean()), 'std': float(t.std()),
        'min': float(t.min()), 'max': float(t.max()), 'snr_pct': 0.0,
    }


def main():
    print(f'[D4] loading {CKPT}')
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = _popw_model_module.POPWMultiTaskModel(
        pretrained=False,
        backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
        use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),
        use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
        train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
    ).to(DEVICE)
    state = {k.replace('ema.', ''): v for k, v in ckpt['model'].items() if not k.startswith('ema.')}
    res = model.load_state_dict(state, strict=False)
    print(f'[D4] load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()

    val_ds = _ds_module.IndustRealMultiTaskDataset(
        split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BS, shuffle=False, num_workers=0,
        collate_fn=_ds_module.collate_fn, pin_memory=False, drop_last=False,
    )

    hook = _Hook()
    # Forward-hook the activity_head.proj_features Linear so we capture its
    # `activity_proj` input without modifying the model.
    handle = model.activity_head.proj_features.register_forward_pre_hook(hook)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img = batch[0].to(DEVICE).float().div_(255.0)
            targets = batch[1]
            clip_rgb = targets.get('clip_rgb', None)
            if clip_rgb is not None:
                clip_rgb = clip_rgb.to(DEVICE)
            try:
                _ = model(img, video_ids=None, clip_rgb=clip_rgb)
            except Exception as e:
                print(f'[D4] forward failed at batch {i}: {e}')
                break
            if (i + 1) >= MAX_BATCHES:
                break

    handle.remove()

    if not hook.captured:
        print('[D4] no activity_proj captured. Aborting.')
        return

    # Concatenate across batches: [N_frames, D]
    captured = torch.cat(hook.captured, dim=0)
    N, D = captured.shape
    print(f'\n=== D4: activity_proj stats over {N} frames, dim={D} ===\n')

    # Slice 0:det_conf_len is 24 (= NUM_DET_CLASSES per model.py:1944-1945).
    det_dim = int(getattr(C, 'NUM_DET_CLASSES', 24))
    det_conf_raw = captured[:, :det_dim]
    p4_gap = captured[:, det_dim:det_dim + 256]    # c5_mod GAP
    p3_gap = captured[:, det_dim + 256:det_dim + 512]   # p4 GAP

    print(f'  Det slice (raw, raw logits)         : shape={tuple(det_conf_raw.shape)}')
    s_raw = _stats('det_raw', det_conf_raw)
    print(f'    mean={s_raw["mean"]:.4f}  std={s_raw["std"]:.4f}  '
          f'range=[{s_raw["min"]:.4f}, {s_raw["max"]:.4f}]  '
          f'avg per-dim std/|mean|={s_raw["snr_pct"]:.3f}%')

    det_conf_zero = torch.zeros_like(det_conf_raw)
    proj_zero = torch.cat([det_conf_zero, p4_gap, p3_gap], dim=1)
    s_zero = _stats('det_zero', det_conf_zero)
    print(f'\n  Det slice (ZEROED)                  : mean={s_zero["mean"]:.4f}  std={s_zero["std"]:.4f}  '
          f'avg per-dim std/|mean|=0.000%')
    s_full_zero = _stats('full_proj_with_zero_det', proj_zero)
    print(f'  Full activity_proj (with det=0)     : mean={s_full_zero["mean"]:.4f}  std={s_full_zero["std"]:.4f}  '
          f'avg per-dim std/|mean|={s_full_zero["snr_pct"]:.3f}%')

    det_conf_sig = det_conf_raw.sigmoid()
    proj_sig = torch.cat([det_conf_sig, p4_gap, p3_gap], dim=1)
    s_sig = _stats('det_sigmoid', det_conf_sig)
    print(f'\n  Det slice (SIGMOID)                 : mean={s_sig["mean"]:.4f}  std={s_sig["std"]:.4f}  '
          f'avg per-dim std/|mean|={s_sig["snr_pct"]:.3f}%')
    s_full_sig = _stats('full_proj_with_sigmoid_det', proj_sig)
    print(f'  Full activity_proj (with sigmoid)    : mean={s_full_sig["mean"]:.4f}  std={s_full_sig["std"]:.4f}  '
          f'avg per-dim std/|mean|={s_full_sig["snr_pct"]:.3f}%')

    print()
    if s_raw['std'] > 100:
        print('  ❌  RC-19 CONFIRMED.')
        print(f'     Det raw slice std={s_raw["std"]:.2f}, range=[{s_raw["min"]:.2f}, {s_raw["max"]:.2f}].')
        print('     This is a near-constant O(10²) input being concatenated with')
        print('     GAP features at O(1). The activity head sees the same giant')
        print('     number on every frame. Sigmoid bounding (or zeroing) restores')
        print('     normal-scale conditioning.')
    elif s_raw['snr_pct'] < 1.0:
        print(f'  ❌  RC-19 CONFIRMED (per-dim std/|mean|={s_raw["snr_pct"]:.3f}% < 1%).')
        print('     Det_conf is effectively constant across frames.')
    else:
        print('  ⚠️  RC-19 NOT strong here — det_conf already varies across frames.')
        print('     This usually means the retrain moved det off its collapsed peak.')


if __name__ == '__main__':
    main()
