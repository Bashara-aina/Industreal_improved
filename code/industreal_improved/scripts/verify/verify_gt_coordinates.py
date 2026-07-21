#!/usr/bin/env python3
"""
Verify actual GT box coordinate system from the real dataset.
Tests whether _extract_boxes_from_coco returns pixel coords or normalized [0,1].
"""

import sys
from pathlib import Path

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC))
for _sub in ["models", "training", "evaluation", "data"]:
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
from data.industreal_dataset import IndustRealMultiTaskDataset

print("=" * 60)
print("DATASET GT BOX COORDINATE VERIFICATION")
print("=" * 60)

ds = IndustRealMultiTaskDataset(
    split="train",
    img_size=C.IMG_SIZE,
    augment=False,
    seed=42,
    max_recordings=3,  # Only 3 recordings for quick test
)

print(f"Dataset: {len(ds)} samples")

all_box_ranges = []
for idx in range(min(len(ds), 10)):  # Check first 10 samples
    sample = ds[idx]
    det = sample.get("gt_boxes", {})
    if isinstance(det, dict):
        det = det.get("rgb", [])
    if not isinstance(det, list):
        det = [det] if det is not None else []

    for i, box_dict in enumerate(det[:5]):  # First 5 boxes per sample
        if box_dict is None:
            continue
        if hasattr(box_dict, "values"):
            boxes = list(box_dict.values())
        elif isinstance(box_dict, (list, tuple)) and len(box_dict) == 4:
            boxes = [box_dict]
        elif isinstance(box_dict, dict):
            boxes = list(box_dict.values()) if box_dict else []
        else:
            boxes = []

        if boxes and len(boxes) == 4:
            x1, y1, x2, y2 = boxes
            all_box_ranges.append((x1, y1, x2, y2))
            is_normalized = all(0 <= v <= 1 for v in [x1, y1, x2, y2])
            print(
                f"  Sample {idx}, box {i}: [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}] "
                f"{'NORMALIZED [0,1]' if is_normalized else 'PIXEL COORDS'}"
            )

print(f"\nTotal boxes checked: {len(all_box_ranges)}")
if all_box_ranges:
    x1s = [b[0] for b in all_box_ranges]
    y1s = [b[1] for b in all_box_ranges]
    x2s = [b[2] for b in all_box_ranges]
    y2s = [b[3] for b in all_box_ranges]
    print(f"X range: [{min(x1s):.2f}, {max(x2s):.2f}]")
    print(f"Y range: [{min(y1s):.2f}, {max(y2s):.2f}]")
    max_val = max(max(x1s), max(y1s), max(x2s), max(y2s))
    is_normalized = max_val <= 1.0
    print(
        f"\nCONCLUSION: Boxes are {'NORMALIZED [0,1]' if is_normalized else 'PIXEL COORDS (abs > 1)'}"
    )
    print(f"Max coordinate value: {max_val:.4f}")
else:
    print("No boxes found in samples!")

print("=" * 60)
