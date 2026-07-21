#!/usr/bin/env python3
"""
Smoke test: verify Mosaic + Copy-Paste augmentations on FullMultiModalDataset.

Tests:
  1. Dataset loads with mosaic/copy-paste enabled
  2. Single sample shape check [9, H, W]
  3. Boxes in normalized [0,1] range
  4. Direct test of mosaic() and copy_paste() functions with synthetic data
  5. 50-batch DataLoader smoke with no crashes
"""
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(0, str(_CODE_ROOT / "src"))

from src.augment.copy_paste import copy_paste
from src.augment.mosaic import mosaic
from torch.utils.data import DataLoader
from train_mtl_full_multimodal import FullMultiModalDataset, collate_real_targets

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

RECORDINGS_DIR = "/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train"
OUT_DIR = Path(_CODE_ROOT) / "research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 640, 360

def make_dummy_images():
    """Create dummy PIL images for all 5 modalities."""
    return {
        "rgb": Image.new("RGB", (W, H), color=(120, 80, 40)),
        "vl": Image.new("L", (W, H), color=100),
        "stl": Image.new("L", (W, H), color=110),
        "str": Image.new("L", (W, H), color=90),
        "dep": Image.new("RGB", (W, H), color=(60, 100, 140)),
    }

# ========================================================================
# Test 1: Dataset loads (quick single sample)
# ========================================================================
print("=" * 60)
print("TEST 1: Load FullMultiModalDataset (single sample check)")
print("=" * 60)

ds = FullMultiModalDataset(
    recordings_dir=RECORDINGS_DIR,
    img_size=(W, H),
    mosaic_prob=0.3,
    copy_paste_prob=0.2,
)
print(f"  Dataset size: {len(ds)} samples")
assert len(ds) >= 4, f"Dataset has {len(ds)} samples, need >=4 for mosaic"

# Test a single sample
x, targets = ds[0]
assert x.shape == (9, H, W), f"Shape mismatch: {x.shape}"
assert x.min() >= 0.0 and x.max() <= 1.0
boxes = targets['boxes']
if boxes.numel() > 0:
    assert boxes.min() >= 0.0 and boxes.max() <= 1.0
    assert boxes.shape[1] == 4
print(f"  Sample 0: shape={x.shape}, boxes={boxes.shape[0]}")
print(f"  Dataset parameters: mosaic_prob={ds.mosaic_prob}, copy_paste_prob={ds.copy_paste_prob}")

# ========================================================================
# Test 2: Direct mosaic() function test with synthetic data
# ========================================================================
print("\n" + "=" * 60)
print("TEST 2: Direct mosaic() function — synthetic 9-channel test")
print("=" * 60)

# Create 4 dummy image sets with known boxes
imgs_list = [make_dummy_images() for _ in range(4)]

# Boxes: one box per image at known positions
# Box in top-left quadrant: cx=0.25, cy=0.25, w=0.1, h=0.1  (normalized)
# Box in top-right: cx=0.75, cy=0.25, ...
# Box in bottom-left: cx=0.25, cy=0.75, ...
# Box in bottom-right: cx=0.75, cy=0.75, ...
b1 = torch.tensor([[0.25, 0.25, 0.1, 0.1]], dtype=torch.float32)
b2 = torch.tensor([[0.75, 0.25, 0.15, 0.1]], dtype=torch.float32)
b3 = torch.tensor([[0.25, 0.75, 0.1, 0.12]], dtype=torch.float32)
b4 = torch.tensor([[0.75, 0.75, 0.12, 0.15]], dtype=torch.float32)

boxes_list = [b1, b2, b3, b4]
classes_list = [torch.tensor([1], dtype=torch.long),
                torch.tensor([2], dtype=torch.long),
                torch.tensor([3], dtype=torch.long),
                torch.tensor([4], dtype=torch.long)]

# Apply mosaic with prob=1.0
mos_imgs, mos_boxes, mos_classes = mosaic(
    imgs_list, boxes_list, classes_list,
    img_size=(W, H), prob=1.0,
)

# Check output
assert len(mos_imgs) == 5
for mod in ["rgb", "vl", "stl", "str", "dep"]:
    assert mod in mos_imgs, f"Missing modality: {mod}"
    assert mos_imgs[mod].size == (W, H), f"Wrong size for {mod}: {mos_imgs[mod].size}"

assert mos_boxes.shape[1] == 4, f"Box shape wrong: {mos_boxes.shape}"
assert mos_boxes.min() >= 0.0 and mos_boxes.max() <= 1.0, f"Box range: [{mos_boxes.min()}, {mos_boxes.max()}]"
assert mos_boxes.shape[0] == mos_classes.shape[0], "Box/class count mismatch"
print(f"  Mosaic output: {len(mos_imgs)} modalities, {mos_boxes.shape[0]} boxes, {mos_classes.shape[0]} classes")
print(f"  Box range: [{mos_boxes.min():.4f}, {mos_boxes.max():.4f}]")
for mod in ["rgb", "vl", "stl", "str", "dep"]:
    print(f"    {mod}: mode={mos_imgs[mod].mode}, size={mos_imgs[mod].size}")

# Check that boxes are within valid bounds (allow slight overlap with crop edge)
assert mos_boxes[:, 2].max() <= 1.0, f"Box w exceeds 1.0: {mos_boxes[:, 2].max()}"
assert mos_boxes[:, 3].max() <= 1.0, f"Box h exceeds 1.0: {mos_boxes[:, 3].max()}"
print("  Mosaic augmentation: OK")

# ========================================================================
# Test 3: Direct copy_paste() function test
# ========================================================================
print("\n" + "=" * 60)
print("TEST 3: Direct copy_paste() function — synthetic test")
print("=" * 60)

tgt_imgs = make_dummy_images()
src_imgs = make_dummy_images()

# Target: one box
tgt_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32)
tgt_classes = torch.tensor([5], dtype=torch.long)

# Source: two boxes (one will be pasted)
src_boxes = torch.tensor([
    [0.3, 0.3, 0.1, 0.1],
    [0.7, 0.7, 0.1, 0.1],
], dtype=torch.float32)
src_classes = torch.tensor([6, 7], dtype=torch.long)

cp_imgs, cp_boxes, cp_classes = copy_paste(
    tgt_imgs, tgt_boxes, tgt_classes,
    src_imgs, src_boxes, src_classes,
    img_size=(W, H), prob=1.0, iou_thresh=0.3, max_paste=8,
)

assert len(cp_imgs) == 5
for mod in ["rgb", "vl", "stl", "str", "dep"]:
    assert cp_imgs[mod].size == (W, H), f"Wrong size for {mod}: {cp_imgs[mod].size}"
assert cp_boxes.shape[1] == 4
assert cp_boxes.min() >= 0.0 and cp_boxes.max() <= 1.0
assert cp_boxes.shape[0] >= tgt_boxes.shape[0], "Should have at least original target boxes"
print(f"  Copy-Paste output: {cp_boxes.shape[0]} boxes, {cp_classes.shape[0]} classes")
print(f"  Original: {tgt_boxes.shape[0]} target + {src_boxes.shape[0]} source")
print(f"  Pasting added {cp_boxes.shape[0] - tgt_boxes.shape[0]} boxes")
for mod in ["rgb", "vl", "stl", "str", "dep"]:
    print(f"    {mod}: mode={cp_imgs[mod].mode}, size={cp_imgs[mod].size}")
print("  Copy-Paste augmentation: OK")

# ========================================================================
# Test 4: Mosaic skip path (prob=0)
# ========================================================================
print("\n" + "=" * 60)
print("TEST 4: Mosaic skip path (prob=0)")
print("=" * 60)

mos_imgs_skip, mos_boxes_skip, mos_classes_skip = mosaic(
    imgs_list, boxes_list, classes_list,
    img_size=(W, H), prob=0.0,
)
assert mos_boxes_skip.shape[0] == 4, f"Should have all 4 boxes when skipping: {mos_boxes_skip.shape[0]}"
print(f"  Skip path: {mos_boxes_skip.shape[0]} boxes (all original), OK")

# ========================================================================
# Test 5: Copy-Paste skip path (prob=0)
# ========================================================================
print("\n" + "=" * 60)
print("TEST 5: Copy-Paste skip path (prob=0)")
print("=" * 60)

cp_imgs_skip, cp_boxes_skip, cp_classes_skip = copy_paste(
    tgt_imgs, tgt_boxes, tgt_classes,
    src_imgs, src_boxes, src_classes,
    img_size=(W, H), prob=0.0,
)
assert cp_boxes_skip.shape[0] == 1, f"Should have 1 original box: {cp_boxes_skip.shape[0]}"
print(f"  Skip path: {cp_boxes_skip.shape[0]} boxes (original only), OK")

# ========================================================================
# Test 6: DataLoader smoke (50 batches)
# ========================================================================
print("\n" + "=" * 60)
print("TEST 6: DataLoader smoke test (50 batches)")
print("=" * 60)

ds_smoke = FullMultiModalDataset(
    recordings_dir=RECORDINGS_DIR,
    img_size=(W, H),
    mosaic_prob=0.3,
    copy_paste_prob=0.2,
)
loader = DataLoader(
    ds_smoke, batch_size=2, shuffle=True,
    collate_fn=collate_real_targets, num_workers=0, pin_memory=False,
)

t0 = time.time()
for i, (images, targets) in enumerate(loader):
    if i >= 50:
        break
    assert images.shape == (2, 9, H, W), f"Batch shape wrong: {images.shape}"
    assert images.dtype == torch.float32
    assert images.min() >= 0.0 and images.max() <= 1.0
    for j in range(2):
        boxes = targets['boxes'][j]
        if boxes.numel() > 0:
            assert boxes.min() >= 0.0 and boxes.max() <= 1.0
            assert boxes.shape[1] == 4
elapsed = time.time() - t0
print(f"  50 batches loaded in {elapsed:.1f}s ({50/elapsed:.1f} batches/s)")
print(f"  All batch shapes verified: [2, 9, {H}, {W}]")

# ========================================================================
# Results summary
# ========================================================================
print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
print(f"  Dataset: FullMultiModalDataset ({len(ds)} samples, {len(ds_smoke)} samples)")
print(f"  Input shape: [9, {H}, {W}] (RGB+VL+StereoL+StereoR+Depth)")
print("  Mosaic: prob=0.3, 4-image 2x2 grid, box coords shifted by tile+crop offset")
print("  Copy-Paste: prob=0.2, IoU>0.3 rejection, up to 8 objects per paste")
print("  CI: all outputs valid — shapes, ranges, types, skip paths")
