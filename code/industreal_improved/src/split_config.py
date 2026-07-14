"""
Split configuration for IndustReal — subject-level train/val/test splits.

Per AAIML §7.1: 12 train / 5 val / 10 test subjects.
Val is for model selection only. Test is for every headline/SOTA number.

Usage:
    from split_config import TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS, get_split
    train_ids = get_split("train")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

# === Canonical subject IDs (frozen per §7.1) ===

TRAIN_SUBJECTS: List[str] = [
    "01",
    "02",
    "04",
    "06",
    "07",
    "11",
    "15",
    "16",
    "21",
    "22",
    "25",
    "27",
]

VAL_SUBJECTS: List[str] = [
    "05",
    "14",
    "20",
    "24",
    "26",
]

TEST_SUBJECTS: List[str] = [
    "03",
    "08",
    "09",
    "10",
    "12",
    "13",
    "17",
    "18",
    "19",
    "23",
]

ALL_SPLIT_IDS = {
    "train": TRAIN_SUBJECTS,
    "val": VAL_SUBJECTS,
    "test": TEST_SUBJECTS,
}

_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "splits" / "industreal_split.json"
)

# === Overlap assertion (enforced at import time) ===

_ALL_SET = set(TRAIN_SUBJECTS) | set(VAL_SUBJECTS) | set(TEST_SUBJECTS)
assert len(TRAIN_SUBJECTS) == 12, f"Expected 12 train subjects, got {len(TRAIN_SUBJECTS)}"
assert len(VAL_SUBJECTS) == 5, f"Expected 5 val subjects, got {len(VAL_SUBJECTS)}"
assert len(TEST_SUBJECTS) == 10, f"Expected 10 test subjects, got {len(TEST_SUBJECTS)}"
assert len(_ALL_SET) == 27, f"Expected 27 unique subjects, got {len(_ALL_SET)} (overlap detected)"
assert not (set(TRAIN_SUBJECTS) & set(VAL_SUBJECTS)), "Train/val overlap"
assert not (set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS)), "Train/test overlap"
assert not (set(VAL_SUBJECTS) & set(TEST_SUBJECTS)), "Val/test overlap"


def get_split(name: str) -> List[str]:
    """Return the list of subject IDs for a named split.

    Args:
        name: One of "train", "val", "test".

    Returns:
        Sorted list of subject ID strings (zero-padded 2-digit).

    Raises:
        KeyError: If name is not a valid split key.
    """
    if name not in ALL_SPLIT_IDS:
        valid = list(ALL_SPLIT_IDS.keys())
        raise KeyError(f"Unknown split '{name}'. Valid splits: {valid}")
    return list(ALL_SPLIT_IDS[name])


def load_manifest() -> dict:
    """Load and return the split manifest JSON as a dict."""
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {_MANIFEST_PATH}. Expected to exist after setup."
        )
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


def require_split(eval_split: str, allow_test_only: bool = False) -> None:
    """Guard utility: validate that an eval is targeting the correct split.

    Call this at the top of any evaluation script that produces headline or
    SOTA-table numbers. It raises an error if the eval split is 'val' when
    test-only results are expected.

    Args:
        eval_split: The split name being evaluated ("train", "val", "test").
        allow_test_only: If True, only 'test' is permitted (for SOTA claims).

    Raises:
        ValueError: If the split violates the protocol specified by §7.1.
    """
    if eval_split not in ALL_SPLIT_IDS:
        valid = list(ALL_SPLIT_IDS.keys())
        raise ValueError(f"Invalid eval_split='{eval_split}'. Must be one of {valid}.")

    if allow_test_only and eval_split != "test":
        raise ValueError(
            f"eval_split='{eval_split}' is not permitted for "
            f"SOTA-table writing. Only 'test' may be used for headline "
            f"numbers (AAIML §7.1). Set allow_test_only=False to run on "
            f"val for model selection."
        )
