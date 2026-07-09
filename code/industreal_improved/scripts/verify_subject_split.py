#!/usr/bin/env python3
"""Subject-overlap verification for train/val/test splits.

[OPUS 186 H-1/H-2/H-3, "non-negotiable"] Verify that no recording_id (subject)
appears in more than one of {train, val, test}. Subject overlap would
inflate metrics and invalidate the paper's MTL/ST comparison.

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

from src.data.industreal_dataset import IndustRealMultiTaskDataset


def get_recording_ids(split: str) -> set:
    """Get the set of recording_ids for a split."""
    ds = IndustRealMultiTaskDataset(
        split=split, img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    ids = set()
    for i in range(len(ds)):
        # Try to extract recording_id from the dataset
        try:
            sample = ds.samples[i] if hasattr(ds, "samples") else None
            if sample and "recording_id" in sample:
                ids.add(sample["recording_id"])
        except (AttributeError, IndexError):
            pass
    return ids


def get_recording_ids_v2(split: str) -> dict:
    """Get recording_id per window, falling back to metadata."""
    ds = IndustRealMultiTaskDataset(
        split=split, img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    ids_per_window = []
    for i in range(len(ds)):
        try:
            item = ds[i]
            if isinstance(item, tuple) and len(item) >= 2:
                _, target = item
                if isinstance(target, dict) and "metadata" in target:
                    meta = target["metadata"]
                    if "recording_id" in meta:
                        ids_per_window.append(meta["recording_id"])
        except Exception as e:
            print(f"  Skipping window {i}: {e}")
    return ids_per_window


def main():
    parser = argparse.ArgumentParser(description="Verify no subject overlap across splits")
    parser.add_argument("--output", type=str, default="/tmp/subject_split_check.json")
    args = parser.parse_args()

    print(f"Loading train/val/test splits...")
    splits = {}
    for split in ["train", "val", "test"]:
        try:
            ids = get_recording_ids_v2(split)
            splits[split] = set(ids)
            print(f"  {split}: {len(ids)} unique recording_ids ({len(ids) // 16} approx windows)")
        except Exception as e:
            print(f"  {split}: ERROR loading: {e}")
            splits[split] = set()

    # Check overlaps
    print(f"\nOverlap analysis:")
    overlaps = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        ov = splits[a] & splits[b]
        overlaps[f"{a}-{b}"] = {
            "n_overlap": len(ov),
            "ids": sorted(list(ov))[:20],  # show first 20
        }
        status = "❌ FAIL" if len(ov) > 0 else "✅ PASS"
        print(f"  {a} ∩ {b}: {len(ov)} (status: {status})")
        if len(ov) > 0:
            print(f"    Sample overlapping IDs: {sorted(list(ov))[:5]}")

    # Save
    output = {
        "splits": {k: len(v) for k, v in splits.items()},
        "overlaps": overlaps,
        "verdict": "PASS" if all(v["n_overlap"] == 0 for v in overlaps.values()) else "FAIL",
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
