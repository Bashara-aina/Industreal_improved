#!/usr/bin/env python3
"""Subject-overlap verification for train/val/test splits (fast version).

[OPUS 186 H-1/H-2/H-3, "non-negotiable"] Verify that no recording_id (subject)
appears in more than one of {train, val, test}. Subject overlap would
inflate metrics and invalidate the paper's MTL/ST comparison.

Fast version: reads recording IDs directly from the dataset's split
directories (no per-window iteration), so it completes in seconds.

Usage:
    python scripts/verify_subject_split.py
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Path setup
_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.NUM_ACT_OUTPUTS = 75
C.ACT_CLASS_GROUPING = "none"


def get_recording_ids_fast(split: str) -> set:
    """Get the set of recording_ids for a split (fast: just list dir)."""
    recordings_dir = C.RECORDINGS_ROOT / split
    if not recordings_dir.exists():
        return set()
    return {d.name for d in recordings_dir.iterdir() if d.is_dir()}


def main():
    parser = argparse.ArgumentParser(description="Verify no subject overlap across splits")
    parser.add_argument("--output", type=str, default="/tmp/subject_split_check.json")
    args = parser.parse_args()

    print(f"Verifying subject overlap via fast directory listing...")
    print(f"  RECORDINGS_ROOT = {C.RECORDINGS_ROOT}")

    splits = {}
    for split in ["train", "val", "test"]:
        ids = get_recording_ids_fast(split)
        splits[split] = ids
        print(f"  {split}: {len(ids)} recording_ids")

    # Check overlaps
    print(f"\nOverlap analysis:")
    overlaps = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    n_fail = 0
    for a, b in pairs:
        ov = splits[a] & splits[b]
        overlaps[f"{a}-{b}"] = {
            "n_overlap": len(ov),
            "ids": sorted(list(ov))[:20],
        }
        status = "❌ FAIL" if len(ov) > 0 else "✅ PASS"
        if len(ov) > 0:
            n_fail += 1
        print(f"  {a} ∩ {b}: {len(ov)} (status: {status})")
        if len(ov) > 0:
            print(f"    Sample overlapping IDs: {sorted(list(ov))[:5]}")

    # Save
    output = {
        "splits": {k: len(v) for k, v in splits.items()},
        "overlaps": overlaps,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {args.output}")
    print(f"Verdict: {output['verdict']}")
    if output["verdict"] == "FAIL":
        print("\n[OPUS 186 H-1/2/3] Subject overlap invalidates any SOTA comparison.")
        print("Re-split the dataset so each recording_id appears in only one split.")
        sys.exit(1)


if __name__ == "__main__":
    main()
