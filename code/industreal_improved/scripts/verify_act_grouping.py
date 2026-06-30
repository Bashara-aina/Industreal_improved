#!/usr/bin/env python3
"""
Verify the Route-A verb-grouping (file 75) against the REAL class names on disk,
BEFORE training. Run from the repo root:

    python scripts/verify_act_grouping.py

It prints, for ACT_CLASS_GROUPING='verb':
  - the number of verb groups (NUM_ACT_OUTPUTS)
  - the group names
  - the raw-id -> group mapping
  - (if the dataset loads) per-group TRAIN frame counts, so you can confirm every
    group clears the ~100-frame learnable threshold before you spend GPU time.

This does NOT modify config; it computes the 'verb' grouping directly so you can
inspect it whether or not ACT_CLASS_GROUPING is currently 'verb'.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402


def main():
    names = list(C.ACT_CLASS_NAMES)
    id_to_group, group_names, num_groups = C._build_act_grouping('verb')

    print(f'Raw activity classes : {len(names)} (NUM_CLASSES_ACT={C.NUM_CLASSES_ACT})')
    print(f'Verb groups          : {num_groups}')
    print(f'Group names          : {group_names}')
    print()
    print('Raw id -> group:')
    for i, nm in enumerate(names):
        print(f'  {i:>2} {nm:<28} -> {id_to_group[i]:>2} ({group_names[id_to_group[i]]})')

    # Optional: per-group TRAIN frame counts (requires the dataset to load).
    try:
        from src.data.industreal_dataset import IndustRealDataset  # noqa: E402
        # Honor whatever subset the training uses.
        ds = IndustRealDataset(split='train')
        from collections import Counter
        raw = [int(s['action_label']) for s in ds.samples]
        grp = Counter(C.remap_activity_label(r) for r in raw if r >= 0)
        print()
        print('Per-group TRAIN frame counts (grouped):')
        for g in range(num_groups):
            c = grp.get(g, 0)
            flag = '' if c >= 100 else '  <-- below ~100-frame learnable threshold'
            print(f'  {g:>2} {group_names[g]:<12} {c:>5}{flag}')
        thin = [group_names[g] for g in range(num_groups) if grp.get(g, 0) < 100]
        if thin:
            print()
            print(f'NOTE: {len(thin)} group(s) under ~100 frames: {thin}')
            print('Consider folding these into an "other" group or excluding from the')
            print('headline metric; report them honestly in the paper either way.')
    except Exception as exc:  # dataset not available in this environment
        print()
        print(f'[skip] per-group counts unavailable here ({type(exc).__name__}: {exc}).')
        print('Run this on the training machine to see real per-group frame counts.')


if __name__ == '__main__':
    main()
