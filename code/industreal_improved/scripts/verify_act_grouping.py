#!/usr/bin/env python3
"""Verify the active activity grouping against real class names and per-group
frame counts. Run from the repo root:

    python scripts/verify_act_grouping.py

Prints the current grouping mode, output count, group names, raw-id mapping,
and per-group TRAIN frame counts (if dataset loads).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

def main():
    names = list(C.ACT_CLASS_NAMES)
    print(f'Active grouping  : {C.ACT_CLASS_GROUPING}'
          f' (threshold={getattr(C, "ACT_HYBRID_THRESHOLD", "N/A")})')
    print(f'Raw classes      : {len(names)} (NUM_CLASSES_ACT={C.NUM_CLASSES_ACT})')
    print(f'Effective outputs: {C.NUM_ACT_OUTPUTS}')
    print(f'Output names     : {C.ACT_OUTPUT_NAMES}')
    print()

    # Raw-id mapping
    print('Raw id -> output:')
    for i, nm in enumerate(names):
        g = C.ACT_ID_TO_GROUP[i]
        oname = C.ACT_OUTPUT_NAMES[g] if g < C.NUM_ACT_OUTPUTS else '?'
        print(f'  {i:>2} {nm:<28} -> {g:>2} ({oname})')
    print()

    # Per-group TRAIN frame counts
    try:
        from src.data.industreal_dataset import IndustRealMultiTaskDataset
        from collections import Counter
        ds = IndustRealMultiTaskDataset(split='train')
        raw = [int(s['action_label']) for s in ds.samples]
        mapped = Counter(C.remap_activity_label(r) for r in raw if r >= 0)
        print(f'Total labeled frames: {sum(mapped.values())}')
        print('Per-group TRAIN frame counts:')
        for g in range(C.NUM_ACT_OUTPUTS):
            c = mapped.get(g, 0)
            flag = '' if c >= 100 else '  ** below ~100-frame learnable threshold'
            print(f'  {g:>2} {C.ACT_OUTPUT_NAMES[g]:<28} {c:>5}{flag}')
        thin = [C.ACT_OUTPUT_NAMES[g] for g in range(C.NUM_ACT_OUTPUTS) if mapped.get(g, 0) < 100]
        if thin:
            print()
            print(f'NOTE: {len(thin)} group(s) under ~100 frames: {thin}')
    except Exception as exc:
        print(f'[skip] per-group counts: {exc}')

if __name__ == '__main__':
    main()
