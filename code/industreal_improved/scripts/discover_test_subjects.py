#!/usr/bin/env python3
"""
Discovery script for IndustReal test-split subject IDs.

Scans the official recordings directory (defined in src/config.py as
POPW_ROOT / 'recordings'), enumerates all subject IDs across train/val/test
subdirectories, validates the 12/5/10 split, and prints the candidate IDs
for each split.

Usage:
    python scripts/discover_test_subjects.py

Output:
    - Train / Val / Test subject IDs
    - Recording counts per split
    - Validation: no overlap, expected counts
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import OrderedDict

# Add src to path so we can import config
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

# Known val IDs from bootstrap_ci.json (verified ground truth)
_KNOWN_VAL: set[str] = {"05", "14", "20", "24", "26"}


def extract_subject_ids(recording_names: list[str]) -> set[str]:
    """Extract unique 2-digit subject IDs from recording directory names.

    Recording names follow the pattern: ``{subject_id}_{assy|main}_{trial}``.
    """
    ids: set[str] = set()
    for name in recording_names:
        parts = name.split("_")
        if parts and parts[0].isdigit():
            ids.add(parts[0])
    return ids


def main() -> None:
    try:
        from config import RECORDINGS_ROOT
    except ImportError:
        print("ERROR: Could not import config. Make sure src/config.py exists.")
        sys.exit(1)

    recordings_root = RECORDINGS_ROOT
    if not recordings_root.exists():
        print(f"ERROR: Recordings root does not exist: {recordings_root}")
        sys.exit(1)

    print(f"Recordings root: {recordings_root}")
    print()

    splits_data: OrderedDict[str, dict] = OrderedDict()

    for split_name in ("train", "val", "test"):
        split_dir = recordings_root / split_name
        if not split_dir.is_dir():
            print(f"  WARNING: {split_dir} not found, skipping")
            continue

        recordings = sorted(
            d.name for d in split_dir.iterdir() if d.is_dir()
        )
        subject_ids = sorted(extract_subject_ids(recordings))

        splits_data[split_name] = {
            "recordings": recordings,
            "subject_ids": subject_ids,
            "n_recordings": len(recordings),
            "n_subjects": len(subject_ids),
        }

    # === Summary table ===
    print("=" * 72)
    print("INDUSTREAL SUBJECT SPLIT DISCOVERY REPORT")
    print("=" * 72)
    print()

    for split_name in ("train", "val", "test"):
        data = splits_data[split_name]
        print(f"[{split_name.upper()}]")
        print(f"  Subjects ({data['n_subjects']}): "
              f"{', '.join(data['subject_ids'])}")
        print(f"  Recordings ({data['n_recordings']}):")
        for rec in data["recordings"]:
            print(f"    {rec}")
        print()

    # === Validation ===
    print("-" * 72)
    print("VALIDATION")
    print("-" * 72)

    train_ids = set(splits_data["train"]["subject_ids"])
    val_ids = set(splits_data["val"]["subject_ids"])
    test_ids = set(splits_data["test"]["subject_ids"])

    all_ids = train_ids | val_ids | test_ids

    checks = [
        ("Train count == 12", len(train_ids) == 12),
        ("Val count == 5", len(val_ids) == 5),
        ("Test count == 10", len(test_ids) == 10),
        ("Total unique subjects == 27", len(all_ids) == 27),
        ("No train/val overlap", train_ids.isdisjoint(val_ids)),
        ("No train/test overlap", train_ids.isdisjoint(test_ids)),
        ("No val/test overlap", val_ids.isdisjoint(test_ids)),
        ("Val matches bootstrap_ci.json",
         val_ids == _KNOWN_VAL),
    ]

    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    # === Config snippet ===
    print()
    print("-" * 72)
    print("GENERATED CONFIG (copy into split_config.py)")
    print("-" * 72)
    print(f"""
TRAIN_SUBJECTS = {json.dumps(sorted(train_ids))}
VAL_SUBJECTS   = {json.dumps(sorted(val_ids))}
TEST_SUBJECTS  = {json.dumps(sorted(test_ids))}
""".strip())

    # === Manifest snippet ===
    print("-" * 72)
    print("GENERATED MANIFEST (copy into config/splits/industreal_split.json)")
    print("-" * 72)
    manifest = OrderedDict([
        ("_version", "1.0"),
        ("_date", "2026-07-08"),
        ("train", sorted(train_ids)),
        ("val", sorted(val_ids)),
        ("test", sorted(test_ids)),
        ("metadata", OrderedDict([
            ("n_train_subjects", len(train_ids)),
            ("n_val_subjects", len(val_ids)),
            ("n_test_subjects", len(test_ids)),
            ("n_train_recordings", splits_data["train"]["n_recordings"]),
            ("n_val_recordings", splits_data["val"]["n_recordings"]),
            ("n_test_recordings", splits_data["test"]["n_recordings"]),
            ("total_subjects", len(all_ids)),
        ])),
    ])
    print(json.dumps(manifest, indent=2))

    print()
    if all_pass:
        print("All checks passed. Split is consistent with the official release.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED. Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    import json  # noqa: F811 — needed in the module scope for the snippet print
    main()
