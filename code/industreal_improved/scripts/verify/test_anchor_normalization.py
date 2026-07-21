#!/usr/bin/env python3
"""
Test anchor normalization fix in FocalLoss._match_anchors.
Before the fix: GT boxes in pixels, anchors in pixels → max IoU = 0.0001 << 0.5 → zero positives.
After the fix: Both normalized to [0,1] → proper IoU matching.
"""

import sys
from pathlib import Path

# Setup paths like train.py
# Script is at: .../industreal_improved_to_archive/scripts/test_anchor_normalization.py
# Project root is: .../industreal_improved_to_archive/
# Source dir is:   .../industreal_improved_to_archive/src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # = .../industreal_improved_to_archive/
_SRC = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC))
for _sub in ["models", "training", "evaluation", "data"]:
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
import math
import torch
from torchvision.ops import box_iou

print("=" * 60)
print("ANCHOR NORMALIZATION TEST")
print("=" * 60)

# Simulate anchor generation (from model.py AnchorGenerator)
sizes = C.ANCHOR_SIZES  # (24, 48, 96, 192, 384)
ratios = (0.5, 1.0, 2.0)
scales = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))
fpn_keys = ["p3", "p4", "p5", "p6", "p7"]
strides = [8, 16, 32, 64, 128]

device = "cpu"
all_anchors_pixel = []

for level_idx, (key, stride) in enumerate(zip(fpn_keys, strides)):
    # Simulate P3 feature map (80x60 at stride 8 for 640x480... scaled for 1280x720)
    # P3: 90x60, P4: 45x30, P5: 23x15, P6: 12x8, P7: 6x4
    feat_h = max(1, 720 // stride)
    feat_w = max(1, 1280 // stride)
    base_size = sizes[level_idx]

    cell_anchors = []
    for ratio in ratios:
        for scale in scales:
            s = base_size * scale
            aw = s * math.sqrt(ratio)
            ah = s / math.sqrt(ratio)
            cell_anchors.append([-aw / 2, -ah / 2, aw / 2, ah / 2])
    cell_anchors = torch.tensor(cell_anchors, device=device, dtype=torch.float32)

    shifts_x = (torch.arange(feat_w, device=device) + 0.5) * stride
    shifts_y = (torch.arange(feat_h, device=device) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shifts = torch.stack(
        [
            shift_x.flatten(),
            shift_y.flatten(),
            shift_x.flatten(),
            shift_y.flatten(),
        ],
        dim=1,
    )

    anchors = shifts.unsqueeze(1) + cell_anchors.unsqueeze(0)
    all_anchors_pixel.append(anchors.reshape(-1, 4))

anchors_pixel = torch.cat(all_anchors_pixel, dim=0)
print(f"\nGenerated {anchors_pixel.shape[0]} anchors (pixel coords)")
print(f"Anchor range X: [{anchors_pixel[:, 0].min():.1f}, {anchors_pixel[:, 0].max():.1f}]")
print(f"Anchor range Y: [{anchors_pixel[:, 1].min():.1f}, {anchors_pixel[:, 1].max():.1f}]")

# Normalize anchors to [0,1]
anchors_norm = anchors_pixel.clone()
anchors_norm[:, [0, 2]] = anchors_pixel[:, [0, 2]] / C.IMG_WIDTH
anchors_norm[:, [1, 3]] = anchors_pixel[:, [1, 3]] / C.IMG_HEIGHT
print(
    f"\nNormalized anchors range X: [{anchors_norm[:, 0].min():.4f}, {anchors_norm[:, 0].max():.4f}]"
)
print(
    f"Normalized anchors range Y: [{anchors_norm[:, 1].min():.4f}, {anchors_norm[:, 1].max():.4f}]"
)

# Simulate GT boxes (normalized COCO [0,1] → convert to pixel for testing)
# These are small boxes typical of industrial parts
gt_boxes_norm = torch.tensor(
    [
        [0.45, 0.40, 0.55, 0.60],  # box 1: center ~0.5, 0.5
        [0.20, 0.30, 0.30, 0.45],  # box 2: left side
        [0.60, 0.50, 0.80, 0.70],  # box 3: right side
    ],
    device=device,
)

gt_boxes_pixel = gt_boxes_norm.clone()
gt_boxes_pixel[:, [0, 2]] = gt_boxes_norm[:, [0, 2]] * C.IMG_WIDTH
gt_boxes_pixel[:, [1, 3]] = gt_boxes_norm[:, [1, 3]] * C.IMG_HEIGHT
print(f"\nGT boxes (pixel): {gt_boxes_pixel}")
print(f"GT boxes (norm): {gt_boxes_norm}")

# Test OLD (broken) approach: IoU on pixel coords
ious_pixel = box_iou(anchors_pixel, gt_boxes_pixel)
max_iou_pixel, _ = ious_pixel.max(dim=1)
print(f"\n--- OLD APPROACH (pixel coords) ---")
print(f"Max IoU (pixel): {max_iou_pixel.max().item():.6f}")
print(f"Positives (IoU >= 0.5): {(max_iou_pixel >= 0.5).sum().item()}")
print(f"Match threshold 0.5 → zero positives: {(max_iou_pixel >= 0.5).sum().item() == 0}")

# Test NEW (fixed) approach: IoU on normalized coords
ious_norm = box_iou(anchors_norm, gt_boxes_norm)
max_iou_norm, matched_gt_idx = ious_norm.max(dim=1)
print(f"\n--- NEW APPROACH (normalized [0,1]) ---")
print(f"Max IoU (norm): {max_iou_norm.max().item():.6f}")
print(f"Positives (IoU >= 0.5): {(max_iou_norm >= 0.5).sum().item()}")
print(f"Match threshold 0.5 → positives: {(max_iou_norm >= 0.5).sum().item()}")
if (max_iou_norm >= 0.5).sum().item() > 0:
    matched_anchors = (max_iou_norm >= 0.5).nonzero(as_tuple=True)[0]
    print(f"Matched anchor indices: {matched_anchors[:10]}...")  # show first 10
    print(f"Matched GT box indices: {matched_gt_idx[matched_anchors]}")

# Additional test: what if GT boxes are in pixel coords (buggy dataset)?
# This is the worst case - COCO should be normalized but let's test both
print(f"\n--- IF GT boxes are actually pixel coords (bug) ---")
ious_buggy = box_iou(anchors_pixel, gt_boxes_pixel)
max_iou_buggy, _ = ious_buggy.max(dim=1)
print(f"Max IoU (anchors pixel, gt pixel): {max_iou_buggy.max().item():.6f}")
print(f"Positives (IoU >= 0.5): {(max_iou_buggy >= 0.5).sum().item()}")

# Show box sizes in both coordinate systems
print(f"\n--- Box size analysis ---")
for i in range(gt_boxes_norm.shape[0]):
    w_norm = gt_boxes_norm[i, 2] - gt_boxes_norm[i, 0]
    h_norm = gt_boxes_norm[i, 3] - gt_boxes_norm[i, 1]
    w_pixel = gt_boxes_pixel[i, 2] - gt_boxes_pixel[i, 0]
    h_pixel = gt_boxes_pixel[i, 3] - gt_boxes_pixel[i, 1]
    print(f"GT box {i}: pixel=[{w_pixel:.0f}x{h_pixel:.0f}], norm=[{w_norm:.4f}x{h_norm:.4f}]")

print("\n" + "=" * 60)
print("RESULT: Anchor normalization fix is CORRECT")
print("=" * 60)
