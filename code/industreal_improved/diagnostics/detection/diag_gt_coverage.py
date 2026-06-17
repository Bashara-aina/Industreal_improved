#!/usr/bin/env python3
"""diag_gt_coverage.py — Detection GT-frame coverage + sampler simulation.

WHY THIS EXISTS
---------------
The RF1 detection "death spiral" is a *data-sampling* problem, not an
optimisation problem. The whole diagnosis hinges on one number: what fraction
of the frames the detector actually sees per batch carry GT boxes?

The notes contain a contradiction that must be resolved empirically:
  * config.py:503  claims the activity-balanced sampler yields ~24% GT batches
  * the death-spiral writeup claims ~0.7% GT batches

This script answers it directly, with NO model and NO GPU. It:
  1. Applies a stage preset (default: stage_rf1) and resolves the same
     subset of recordings train.py would use for that --subset-ratio.
  2. Builds the dataset and reports, from ds.samples[*]['num_dets']:
       - total frames, frames-with-boxes (count + %),
       - per-recording OD coverage (how many recordings have ANY OD labels,
         and how many have ZERO — the recording-level sparsity check),
  3. Simulates the WeightedRandomSampler and reports the realised GT-frame
     fraction and expected GT frames per *effective* batch, both with
     DET_GT_FRAME_FRACTION OFF (legacy) and ON (the fix).

INTERPRETATION
--------------
  * If "frames with boxes" is a few % but per-batch GT fraction under the
    legacy sampler is <2%, the sampler is the bug -> the fix below raises it.
  * If MANY recordings have ZERO OD labels, the subset itself is starved:
    the greedy subset selection in _scan_and_index optimises ACTIVITY class
    coverage and ignores OD availability. No sampler can invent GT frames that
    aren't in the subset -> raise --subset-ratio or pick OD-bearing recordings.

USAGE
-----
  python diag_gt_coverage.py                         # stage_rf1, subset 0.2
  python diag_gt_coverage.py --preset stage_rf3 --subset-ratio 0.35
  python diag_gt_coverage.py --split train --draws 20000
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# --- Resolve project paths relative to THIS file (works on any machine) ------
PROJ = Path(__file__).resolve().parent
for p in (str(PROJ), str(PROJ / 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('OMP_NUM_THREADS', '4')

import config as C  # noqa: E402
from data.industreal_dataset import IndustRealMultiTaskDataset  # noqa: E402


def _count_train_recordings(csv_path) -> int:
    """Unique recording_ids in a split CSV (mirrors train.py's resolution)."""
    try:
        import pandas as pd
        return int(pd.read_csv(csv_path)['recording_id'].nunique())
    except Exception:
        ids = set()
        try:
            import csv as _csv
            with open(csv_path, encoding='utf-8') as f:
                for row in _csv.reader(f):
                    if row:
                        ids.add(row[0].strip())
        except Exception:
            return 0
        ids.discard('recording_id')  # header, if present
        return len(ids)


def _expected_gt_fraction(weights: np.ndarray, gt_mask: np.ndarray) -> float:
    """Expected fraction of draws that land on a GT frame (with replacement)."""
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float(weights[gt_mask].sum() / total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='stage_rf1')
    ap.add_argument('--subset-ratio', type=float, default=0.2)
    ap.add_argument('--split', default='train')
    ap.add_argument('--draws', type=int, default=20000,
                    help='Monte-Carlo samples to simulate the sampler.')
    args = ap.parse_args()

    print('=' * 72)
    print(f'  DET GT-FRAME COVERAGE  preset={args.preset}  '
          f'subset_ratio={args.subset_ratio}  split={args.split}')
    print('=' * 72)

    C.apply_preset(args.preset)

    # Resolve the same recording subset train.py would build for this ratio.
    max_recordings = None
    if args.subset_ratio < 1.0:
        n_recs = _count_train_recordings(C.TRAIN_CSV if args.split == 'train' else C.VAL_CSV)
        if n_recs > 0:
            max_recordings = max(4, int(n_recs * args.subset_ratio))
        print(f'  recordings in split: {n_recs} -> using {max_recordings}')

    ds = IndustRealMultiTaskDataset(
        split=args.split,
        img_size=C.IMG_SIZE,
        augment=False,
        seed=C.SEED,
        max_recordings=max_recordings,
    )

    samples = ds.samples
    n_total = len(samples)
    num_dets = np.array([s.get('num_dets', 0) for s in samples], dtype=np.int64)
    gt_mask = num_dets > 0
    n_gt = int(gt_mask.sum())

    print('\n--- Frame-level OD coverage ---')
    print(f'  total frames in index : {n_total:,}')
    print(f'  frames WITH boxes     : {n_gt:,}  ({100.0 * n_gt / max(n_total, 1):.3f}%)')
    print(f'  frames WITHOUT boxes  : {n_total - n_gt:,}')
    if n_gt:
        print(f'  boxes/frame (GT frames): mean={num_dets[gt_mask].mean():.2f}  '
              f'max={int(num_dets[gt_mask].max())}')

    # --- Per-recording OD coverage (recording-level sparsity check) ----------
    per_rec_total = Counter(s['recording_id'] for s in samples)
    per_rec_gt = Counter(s['recording_id'] for s in samples if s.get('num_dets', 0) > 0)
    recs = sorted(per_rec_total)
    recs_with_od = [r for r in recs if per_rec_gt.get(r, 0) > 0]
    recs_zero_od = [r for r in recs if per_rec_gt.get(r, 0) == 0]

    print('\n--- Recording-level OD coverage ---')
    print(f'  recordings in subset       : {len(recs)}')
    print(f'  recordings WITH any OD     : {len(recs_with_od)}')
    print(f'  recordings with ZERO OD    : {len(recs_zero_od)}')
    if recs_zero_od:
        print('  !! ZERO-OD recordings (detector gets NOTHING from these):')
        print('     ' + ', '.join(recs_zero_od[:12]) + (' ...' if len(recs_zero_od) > 12 else ''))
    print('  GT frames per recording (top 8):')
    for r in sorted(recs, key=lambda x: -per_rec_gt.get(x, 0))[:8]:
        print(f'     {r:32s} {per_rec_gt.get(r, 0):5d} GT / {per_rec_total[r]:6d} frames')

    # --- Sampler simulation: legacy (frac=0) vs fix (stage value) ------------
    eff_batch = int(getattr(C, 'BATCH_SIZE', 4)) * int(getattr(C, 'GRAD_ACCUM_STEPS', 8))
    print('\n--- Sampler simulation ---')
    print(f'  effective batch (BATCH_SIZE x GRAD_ACCUM) = {eff_batch}')

    saved = float(getattr(C, 'DET_GT_FRAME_FRACTION', 0.0))
    for label, frac in (('LEGACY (DET_GT_FRAME_FRACTION=0.0)', 0.0),
                        (f'FIX    (DET_GT_FRAME_FRACTION={saved})', saved)):
        C.DET_GT_FRAME_FRACTION = frac
        sampler = ds.get_sampler()
        weights = np.asarray(sampler.weights, dtype=np.float64)
        exp_frac = _expected_gt_fraction(weights, gt_mask)
        # Monte-Carlo realised fraction
        rng = np.random.default_rng(0)
        p = weights / weights.sum()
        drawn = rng.choice(n_total, size=min(args.draws, 200000), replace=True, p=p)
        realised = float(gt_mask[drawn].mean())
        print(f'\n  {label}')
        print(f'    expected GT-frame fraction : {exp_frac * 100:6.2f}%')
        print(f'    realised (MC) GT fraction  : {realised * 100:6.2f}%')
        print(f'    => GT frames per eff. batch: {realised * eff_batch:6.2f} / {eff_batch}')
        if realised * eff_batch < 1.0:
            print('    !! < 1 GT frame per step -> detector starves -> death spiral.')
    C.DET_GT_FRAME_FRACTION = saved

    print('\n' + '=' * 72)
    print('  VERDICT')
    if n_gt == 0:
        print('  No GT frames in subset. Detection cannot be learned here at all.')
        print('  -> raise --subset-ratio or select OD-bearing recordings.')
    elif len(recs_zero_od) > len(recs_with_od):
        print('  Most recordings have ZERO OD labels -> recording-level sparsity.')
        print('  The sampler fix helps but the subset is thin. Prefer OD-bearing')
        print('  recordings (make subset selection OD-aware) or raise subset_ratio.')
    else:
        print('  Subset has usable GT frames. The DET_GT_FRAME_FRACTION fix should')
        print('  lift per-batch GT frames well above 1 -> death spiral resolved.')
    print('=' * 72)


if __name__ == '__main__':
    main()
