#!/usr/bin/env python3
"""D1 [opus RC-13] — Cosine-similarity check that `best.pth` is EMA-contaminated.

For each "head" tensor (det/act/psr) the diagnostic prints the cosine similarity
between the saved `best.pth` weight, the `latest.pth` weight (raw, end-of-epoch
from the SAME run), and the source `crash_recovery.pth` (the pre-reinit
"collapsed" checkpoint we resumed from).

Verdict:
    best ≈ crash (cos > 0.5) on head tensors  →  RC-13 confirmed.
    best ≈ latest (cos > 0.5) on head tensors →  best.pth is the raw trained
        model; EMA contamination is NOT the cause.
    Both low (< 0.5)                          →  either reinit moved the head
        far from both (expected) or there's a third cause.

Zero GPU. Just loads 3 checkpoints (~3 GB RAM peak) and prints a table.
"""
import sys, os
from pathlib import Path
import torch

CKPT_DIR = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints')

def _strip_ema(state: dict) -> dict:
    """Strip EMA prefix and the EMA shadow dict from a checkpoint state."""
    if 'model' in state and isinstance(state['model'], dict):
        return state['model']
    if 'ema' in state:
        return {k.replace('ema.', '', 1): v for k, v in state.items() if k.startswith('ema.')}
    return state

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float('nan')
    return float((a @ b) / (na * nb))

def main():
    paths = {
        'best':    CKPT_DIR / 'best.pth',
        'latest':  CKPT_DIR / 'latest.pth',
        'crash':   CKPT_DIR / 'crash_recovery.pth',
    }
    print('Loading 3 checkpoints (≈3 GB RAM peak)...', flush=True)
    states = {}
    for tag, p in paths.items():
        print(f'  {tag}: {p} ({p.stat().st_size/1e6:.0f} MB)', flush=True)
        states[tag] = torch.load(p, map_location='cpu', weights_only=False)

    models = {tag: _strip_ema(s) for tag, s in states.items()}

    # Head prefixes that should have been re-init'd between crash and latest
    head_prefixes = (
        'detection_head.', 'det_head.',
        'activity_head.',
        'psr_head.',
    )
    backbone_prefixes = (
        'backbone.', 'pose_film.', 'headpose_film.', 'hand_film.',
        'head_pose_head.', 'feature_bank.',
    )

    print('\n=== HEAD tensors (re-init should make `best` ≈ `latest`, NOT ≈ `crash`) ===\n')
    print(f'{"tensor":<55s} {"best↔latest":>12s} {"best↔crash":>12s} {"latest↔crash":>14s}')
    print('-' * 95)

    head_keys = set()
    for tag, m in models.items():
        for k in m.keys():
            if any(k.startswith(p) for p in head_prefixes):
                head_keys.add(k)

    head_cos_bl, head_cos_bc, head_cos_lc = [], [], []
    for k in sorted(head_keys):
        b = models['best'].get(k)
        l = models['latest'].get(k)
        c = models['crash'].get(k)
        if b is None or l is None or c is None:
            continue
        c_bl = _cos(b, l)
        c_bc = _cos(b, c)
        c_lc = _cos(l, c)
        head_cos_bl.append(c_bl)
        head_cos_bc.append(c_bc)
        head_cos_lc.append(c_lc)
        # Truncate tensor name
        name = k[:53] + '..' if len(k) > 55 else k
        print(f'{name:<55s} {c_bl:>12.3f} {c_bc:>12.3f} {c_lc:>14.3f}')

    print('\n=== BACKBONE tensors (should track crash more closely) ===\n')
    print(f'{"tensor":<55s} {"best↔latest":>12s} {"best↔crash":>12s} {"latest↔crash":>14s}')
    print('-' * 95)
    bb_keys = set()
    for tag, m in models.items():
        for k in m.keys():
            if any(k.startswith(p) for p in backbone_prefixes):
                bb_keys.add(k)
    bb_cos_bl, bb_cos_bc, bb_cos_lc = [], [], []
    for k in sorted(bb_keys)[:20]:  # show only first 20 backbone keys
        b = models['best'].get(k)
        l = models['latest'].get(k)
        c = models['crash'].get(k)
        if b is None or l is None or c is None:
            continue
        c_bl = _cos(b, l)
        c_bc = _cos(b, c)
        c_lc = _cos(l, c)
        bb_cos_bl.append(c_bl)
        bb_cos_bc.append(c_bc)
        bb_cos_lc.append(c_lc)
        name = k[:53] + '..' if len(k) > 55 else k
        print(f'{name:<55s} {c_bl:>12.3f} {c_bc:>12.3f} {c_lc:>14.3f}')

    print('\n=== Verdict ===\n')
    import statistics
    head_bc_med = statistics.median(head_cos_bc) if head_cos_bc else float('nan')
    head_bl_med = statistics.median(head_cos_bl) if head_cos_bl else float('nan')
    head_lc_med = statistics.median(head_cos_lc) if head_cos_lc else float('nan')
    bb_bc_med = statistics.median(bb_cos_bc) if bb_cos_bc else float('nan')
    bb_lc_med = statistics.median(bb_cos_lc) if bb_cos_lc else float('nan')

    print(f'  HEAD medians:     best↔latest={head_bl_med:.3f}   best↔crash={head_bc_med:.3f}   latest↔crash={head_lc_med:.3f}')
    print(f'  BACKBONE medians: best↔crash={bb_bc_med:.3f}     latest↔crash={bb_lc_med:.3f}')

    # When all three are identical (cos ≈ 1.0 for every pair), EMA was disabled
    # and the "best" checkpoint is just a copy.  RC-13 is definitively denied.
    if head_bl_med > 0.99 and head_bc_med > 0.99 and head_lc_med > 0.99:
        print('\n  ✅  RC-13 DENIED: all three checkpoints are identical (cos ≈ 1.0).')
        print('     EMA was disabled (P2 fix); best.pth is just a renamed copy of')
        print('     latest.pth, not an EMA blend.')
    elif head_bc_med > 0.5 and head_bc_med > head_bl_med + 0.1:
        print('\n  ❌  RC-13 CONFIRMED: head tensors in best.pth are more similar to')
        print('     the pre-reinit crash_recovery.pth than to the raw end-of-epoch')
        print('     latest.pth → best.pth is the EMA-contaminated blend, not the')
        print('     trained model. The post-retrain eval is measuring a corrupted')
        print('     checkpoint, not the result of the 1-epoch retrain.')
    elif head_bl_med > 0.5:
        print('\n  ✅  RC-13 DENIED: head tensors in best.pth track latest.pth, not')
        print('     crash_recovery.pth → best.pth is the raw trained model.')
    else:
        print('\n  ⚠️  INCONCLUSIVE: neither best↔latest nor best↔crash is clearly higher.')
        print('     This could happen if the reinit moved the head far from both, or')
        print('     if there is a third cause to investigate.')

if __name__ == '__main__':
    main()
