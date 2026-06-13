#!/usr/bin/env python3
"""D6 [opus RC-16] — Attention-saturation diagnostic.

Forwards 50 val frames through the post-retrain checkpoint. Captures the
post-softmax attention weights from each `ViTTemporalBlock` in the activity
head (typically 2 blocks, each with `num_heads` heads). Reports, per block
and per head:
  - max attention weight (anywhere in the [T, T] matrix)
  - mean attention weight off the diagonal
  - entropy of the attention distribution averaged over rows

Verdict (opus P5 = model.py:1103 already multiplies by `scale = 1/sqrt(d)`):
  - max attn ≈ 1.0 with low off-diag mass  → attention saturated / one-hot
    (RC-16 is still biting — usually because RC-18 makes all tokens
    identical, so this test is only meaningful AFTER engaging the
    FeatureBank).
  - max attn ≈ 1/T and uniform across positions → all tokens identical
    (RC-18 / dead FeatureBank).
  - off-diag mass > 0.3 and entropy > 0  → attention is differentiated
    and healthy.
"""
import os, sys
from pathlib import Path
import statistics
import numpy as np

PROJ = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
from models import model as _popw_model_module

CKPT = os.environ.get('CHECKPOINT', str(PROJ / 'src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth'))
MAX_BATCHES = int(os.environ.get('MAX_BATCHES', '13'))  # 13 × bs=4 ≈ 50 frames
BS = int(os.environ.get('EVAL_BS', '4'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print(f'[D6] loading {CKPT}')
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
    model.load_state_dict(state, strict=False)
    model.eval()

    # Capture post-softmax attn weights from each ViT block via forward_hook on
    # `attn_dropout` (its input is the softmax output, shape [B, H, T, T]).
    attn_blocks = []   # list of (block_idx, attn_tensor)
    handles = []

    vit = getattr(model.activity_head, 'vit', None)
    if vit is None:
        # Older naming?  Try alternate attribute paths.
        for cand in ('vit', 'transformer', 'blocks'):
            vit = getattr(model.activity_head, cand, None)
            if vit is not None:
                break
    if vit is None:
        print('[D6] activity_head has no .vit/.transformer/.blocks attribute. Aborting.')
        return
    if not isinstance(vit, nn.ModuleList):
        # Wrap as list for uniform iteration
        vit = nn.ModuleList([vit])

    def make_hook(block_idx):
        def hook(module, inputs, output):
            attn = inputs[0]                       # [B, H, T, T] post-softmax
            attn_blocks.append((block_idx, attn.detach().cpu()))
        return hook

    for i, block in enumerate(vit):
        if hasattr(block, 'attn_dropout') and isinstance(block.attn_dropout, nn.Dropout):
            handles.append(block.attn_dropout.register_forward_hook(make_hook(i)))

    val_ds = _ds_module.IndustRealMultiTaskDataset(
        split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BS, shuffle=False, num_workers=0,
        collate_fn=_ds_module.collate_fn, pin_memory=False, drop_last=False,
    )

    n_frames = 0
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
                print(f'[D6] forward failed at batch {i}: {e}')
                break
            n_frames += img.shape[0]
            if n_frames >= 50:
                break

    for h in handles:
        h.remove()

    if not attn_blocks:
        print('[D6] no attention weights captured. Aborting.')
        return

    # Per-block: collect max, off-diag mass, entropy.
    by_block = {}
    for block_idx, attn in attn_blocks:
        B, H, T, _ = attn.shape
        if T < 2:
            continue
        flat = attn.view(-1, T, T).float()           # [N, T, T]
        # Per matrix: max, off-diag mass, row-entropy.
        N = flat.shape[0]
        eye = torch.eye(T).unsqueeze(0)
        diag_mass = (flat * eye).sum(dim=(1, 2))     # [N]
        total = flat.sum(dim=(1, 2)).clamp(min=1e-9)
        offdiag_frac = 1.0 - diag_mass / total
        # Entropy of each row's distribution (in nats): -sum(p*log(p))
        ent = -(flat.clamp(min=1e-9) * flat.clamp(min=1e-9).log()).sum(dim=-1)  # [N, T]
        max_attn = flat.view(N, -1).max(dim=1).values
        by_block.setdefault(block_idx, []).append((max_attn, offdiag_frac, ent))

    print(f'\n=== D6: Attention saturation over {n_frames} frames ({len(attn_blocks)} attn blocks captured) ===\n')
    print(f'  {"block":>6} {"#matrices":>10} {"max_attn":>12} {"offdiag_frac":>14} {"entropy_nats":>14}')
    print('  ' + '-' * 60)
    for block_idx in sorted(by_block):
        m, o, e = zip(*by_block[block_idx])
        m_cat = torch.cat([x for x in m])
        o_cat = torch.cat([x for x in o])
        e_cat = torch.cat([x for x in e])
        print(f'  {block_idx:>6d} {len(m):>10d} '
              f'{float(m_cat.mean()):>12.4f} {float(o_cat.mean()):>14.4f} {float(e_cat.mean()):>14.4f}')

    # Verdict using a single average across all blocks.
    all_max = torch.cat([torch.cat([x[0] for x in v]) for v in by_block.values()])
    all_off = torch.cat([torch.cat([x[1] for x in v]) for v in by_block.values()])
    all_ent = torch.cat([torch.cat([x[2] for x in v]) for v in by_block.values()])
    avg_max = float(all_max.mean())
    avg_off = float(all_off.mean())
    avg_ent = float(all_ent.mean())
    T = attn_blocks[0][1].shape[-1]
    uniform_max = 1.0 / T
    print()
    if avg_max > 0.9 and avg_off < 0.1:
        print(f'  ❌  ATTENTION SATURATED (avg max={avg_max:.3f} ≈ 1.0, off-diag mass={avg_off:.3f}).')
        print('     Each row of the attention matrix concentrates on a single token.')
        print('     After P5, this is most likely because ALL TOKENS ARE IDENTICAL')
        print('     (RC-18 — FeatureBank returns the current frame 16×).')
    elif avg_max < uniform_max * 1.5 and avg_off > 0.85:
        print(f'  ⚠️  UNIFORM ATTENTION (avg max={avg_max:.3f} ≈ 1/T={uniform_max:.3f}).')
        print('     All attention rows distribute weight evenly across positions.')
        print('     This is the RC-18 fingerprint — feature bank is dead and the')
        print('     TCN/ViT blocks are processing 17 near-identical tokens.')
    elif avg_off > 0.3:
        print(f'  ✅  ATTENTION HEALTHY (avg max={avg_max:.3f}, off-diag mass={avg_off:.3f}, '
              f'entropy={avg_ent:.3f} nats).')
        print('     Tokens differ enough to drive differentiated attention. P5 fix is intact.')
    else:
        print(f'  ⚠️  INCONCLUSIVE (max={avg_max:.3f}, off-diag={avg_off:.3f}, entropy={avg_ent:.3f}).')


if __name__ == '__main__':
    main()
