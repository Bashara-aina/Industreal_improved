#!/usr/bin/env python3
"""Check train/val/test subject disjointness.

Imports the canonical subject split from split_config (AAIML 7.1:
12 train / 5 val / 10 test) and asserts that no subject ID appears
in more than one split.

Usage:
    python scripts/check_train_val_subject_disjoint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for importing split_config
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from split_config import TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS


def main() -> None:
    train_set = set(TRAIN_SUBJECTS)
    val_set = set(VAL_SUBJECTS)
    test_set = set(TEST_SUBJECTS)

    all_ids = train_set | val_set | test_set
    expected_total = len(TRAIN_SUBJECTS) + len(VAL_SUBJECTS) + len(TEST_SUBJECTS)
    actual_total = len(all_ids)

    print("=" * 60)
    print("Train/Val/Test Subject Disjointness Check (AAIML 7.1)")
    print("=" * 60)
    print(f"  Train subjects ({len(TRAIN_SUBJECTS)}): {TRAIN_SUBJECTS}")
    print(f"  Val subjects   ({len(VAL_SUBJECTS)}): {VAL_SUBJECTS}")
    print(f"  Test subjects  ({len(TEST_SUBJECTS)}): {TEST_SUBJECTS}")
    print()

    # Check 1: Overlap detection
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    has_overlap = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    if train_val_overlap:
        print(f"  FAIL: Train/val overlap: {sorted(train_val_overlap)}")
    else:
        print("  OK:   No train/val overlap")

    if train_test_overlap:
        print(f"  FAIL: Train/test overlap: {sorted(train_test_overlap)}")
    else:
        print("  OK:   No train/test overlap")

    if val_test_overlap:
        print(f"  FAIL: Val/test overlap: {sorted(val_test_overlap)}")
    else:
        print("  OK:   No val/test overlap")

    # Check 2: Total unique count
    if actual_total < expected_total:
        overlap_count = expected_total - actual_total
        print(f"  FAIL: {overlap_count} subject(s) appear in multiple splits")
    elif actual_total == expected_total:
        print(f"  OK:   {actual_total} unique subjects across all splits")
    else:
        print(f"  WARN: {actual_total} unique subjects > expected {expected_total}")

    # Check 3: Verify expected split sizes
    assert (
        len(TRAIN_SUBJECTS) == 12
    ), f"Expected 12 train subjects, got {len(TRAIN_SUBJECTS)}"
    assert (
        len(VAL_SUBJECTS) == 5
    ), f"Expected 5 val subjects, got {len(VAL_SUBJECTS)}"
    assert (
        len(TEST_SUBJECTS) == 10
    ), f"Expected 10 test subjects, got {len(TEST_SUBJECTS)}"
    print("  OK:   Split sizes correct (12/5/10)")

    # Check 4: No subject ID is empty or malformed
    for split_name, subjects in [
        ("train", TRAIN_SUBJECTS),
        ("val", VAL_SUBJECTS),
        ("test", TEST_SUBJECTS),
    ]:
        for sid in subjects:
            if not sid.isdigit() or len(sid) != 2:
                print(f"  FAIL: {split_name} subject '{sid}' is not a 2-digit ID")
                has_overlap = True

    print()
    if has_overlap:
        print("RESULT: FAIL -- subject overlap detected (see above)")
        sys.exit(1)
    else:
        print("RESULT: PASS -- all splits are disjoint and correctly sized")
        print("  Note: This check is enforced at import time in split_config.py "
              "(assertions at module level).")


if __name__ == "__main__":
    main()
