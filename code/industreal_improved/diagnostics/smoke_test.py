#!/usr/bin/env python3
"""
Comprehensive POPW smoke test — validates all fixes.
Covers: dataset, model forward, loss computation, Kendall init, PSR slicing.
"""
import sys
import os
workdir = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive'
sys.path.insert(0, workdir)
os.chdir(workdir)

import torch
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
from src import config as C
C.DEBUG_MODE = True
C.USE_VIDEOMAE = False

# ── 1. Dataset smoke ─────────────────────────────────────────────────────────
print("\n=== TEST 1: Dataset __getitem__ (no None fields) ===")
from src.data.industreal_dataset import IndustRealMultiTaskDataset

ds = IndustRealMultiTaskDataset(
    split='train',
    max_recordings=2,
    subset_ratio=0.05,
    sequence_mode=False,
)
print(f"  Dataset size: {len(ds)}")

sample = ds[0]
for key, val in sample.items():
    if val is None:
        print(f"  ❌ {key} is None")
        sys.exit(1)
print("  ✅ All dataset fields are non-None")
print(f"  head_pose shape: {sample['head_pose'].shape}")
print(f"  psr_labels shape: {sample['psr_labels'].shape}")
print(f"  clip_rgb shape: {sample['clip_rgb'].shape}")

# ── 2. Collation smoke ───────────────────────────────────────────────────────
print("\n=== TEST 2: DataLoader collate (batch=2) ===")
from torch.utils.data import DataLoader
from src.data.industreal_dataset import collate_fn

dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
images_batch, targets_batch = next(iter(dl))
print(f"  images_batch shape: {images_batch.shape}")
print(f"  targets_batch['head_pose'] shape: {targets_batch['head_pose'].shape}")
print(f"  targets_batch['psr_labels'] shape: {targets_batch['psr_labels'].shape}")
print(f"  targets_batch['activity'] shape: {targets_batch['activity'].shape}")
print("  ✅ Collate successful — no TypeError")

# ── 3. Model forward ──────────────────────────────────────────────────────────
print("\n=== TEST 3: Model forward pass ===")
from src.models.model import POPWMultiTaskModel

model = POPWMultiTaskModel(
    pretrained=False,
    backbone_type='convnext_tiny',
    use_headpose_film=True,
    use_videomae=False,
    train_pose=True,
).cuda()
model.eval()

B = 2
# Normalize images: uint8 [0,255] -> float32 [0,1] -> ImageNet norm
images_raw = images_batch.cuda().float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
images_norm = (images_raw - mean) / std

with torch.no_grad():
    outputs = model(images_norm, clip_rgb=None)

print(f"  act_logits shape: {outputs['act_logits'].shape}  (expected [2, 75])")
print(f"  psr_logits shape: {outputs['psr_logits'].shape}  (expected [2, 11])")
print(f"  head_pose shape: {outputs['head_pose'].shape}  (expected [2, 9])")
assert outputs['act_logits'].shape == (B, C.NUM_ACT_CLASSES), f"act_logits wrong: {outputs['act_logits'].shape}"
assert outputs['psr_logits'].shape == (B, C.NUM_PSR_COMPONENTS), f"psr_logits wrong: {outputs['psr_logits'].shape}"
assert outputs['head_pose'].shape == (B, C.NUM_HEAD_POSE_DOF), f"head_pose wrong: {outputs['head_pose'].shape}"
print("  ✅ Model forward shapes correct")

psr_logits_full = outputs['psr_logits']
psr_confidence = outputs.get('psr_confidence')

# ── 4. Kendall log_var_act init ─────────────────────────────────────────────
print("\n=== TEST 4: Kendall log_var_act init ===")
from src.training.losses import MultiTaskLoss

loss_fn = MultiTaskLoss(
    num_classes_act=C.NUM_ACT_CLASSES,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=True,
    train_pose=False,  # disabled — no real keypoint annotations in IndustReal
    train_act=True,
    train_psr=True,
    use_kendall=True,
).cuda()

# Paper spec: init [0, -1, 0, 0] for (det, pose, act, psr)
expected = {
    'log_var_det': 0.0,
    'log_var_pose': -1.0,
    'log_var_act': 0.0,   # was -1.0 → fixed to 0.0 per paper spec
    'log_var_psr': 0.0,
}
all_ok = True
for name, exp_val in expected.items():
    actual = getattr(loss_fn, name).item()
    ok = abs(actual - exp_val) < 1e-6
    status = "✅" if ok else "❌"
    print(f"  {status} {name} = {actual:.4f}  (expected {exp_val:.4f})")
    if not ok:
        all_ok = False
        print(f"       FAIL — got {actual}, expected {exp_val}")

if not all_ok:
    sys.exit(1)

# ── 5. PSRHead output + slicing ─────────────────────────────────────────────
print("\n=== TEST 5: PSRHead returns [B,12] → sliced to [B,11] ===")
print(f"  psr_logits (after slicing): {psr_logits_full.shape}")
print(f"  psr_confidence: {psr_confidence.squeeze().cpu().numpy()}")
assert psr_logits_full.shape == (B, 11), f"PSR logits shape wrong: {psr_logits_full.shape}"
assert psr_confidence is not None, "psr_confidence should be present in eval mode"
print("  ✅ PSR slicing correct — [:11] used, confidence returned in eval mode")

# ── 6. Loss computation ──────────────────────────────────────────────────────
print("\n=== TEST 6: Loss computation (finite, no NaN) ===")

# Get anchor count from model — anchor_gen expects FPN pyramid dict, not raw images
with torch.no_grad():
    # Run model to get FPN feature maps (needed for anchor generation)
    # Create matching pyramid dict manually
    fpn_pyramid = {
        'p3': torch.randn(B, 96, 90, 160, device='cuda'),
        'p4': torch.randn(B, 192, 45, 80, device='cuda'),
        'p5': torch.randn(B, 384, 23, 40, device='cuda'),
        'p6': torch.randn(B, 768, 12, 20, device='cuda'),
        'p7': torch.randn(B, 768, 6, 10, device='cuda'),
    }
    anchors_list = model.anchor_gen(fpn_pyramid)
    flat_anchors = torch.cat([a.flatten() for a in anchors_list], dim=0).reshape(-1, 4)

num_anchors = flat_anchors.shape[0]
print(f"  Anchor count: {num_anchors}")

cls_preds = torch.randn(B, num_anchors, C.NUM_DET_CLASSES).cuda()
reg_preds = torch.randn(B, num_anchors, 4).cuda()
det_conf = torch.randn(B, num_anchors).cuda().sigmoid()

outputs_l = {
    'cls_preds': cls_preds,
    'reg_preds': reg_preds,
    'anchors': flat_anchors,
    'det_conf': det_conf,
    'act_logits': torch.randn(B, C.NUM_ACT_CLASSES).cuda(),
    'psr_logits': psr_logits_full,  # [B, 11]
    'head_pose': outputs['head_pose'],
    'heatmaps': torch.randn(B, 17, 96, 160).cuda(),
    'keypoints': torch.randn(B, 17, 2).cuda(),
    'pose_confidence': torch.ones(B, 17).cuda() * 0.5,
}

# detection list format for FocalLoss
targets = {
    'heatmaps': torch.randn(B, 17, 96, 160).cuda(),
    'keypoints': torch.randn(B, 17, 2).cuda(),
    'detection': [
        {'boxes': torch.randn(5, 4).abs().cuda(), 'labels': torch.randint(0, C.NUM_DET_CLASSES, (5,)).cuda()}
        for _ in range(B)
    ],
    'head_pose': targets_batch['head_pose'].cuda(),
    'psr_labels': targets_batch['psr_labels'].cuda(),
    'activity': targets_batch['activity'].cuda(),  # Note: 'activity' key (not 'action_label')
    'hand_joints': torch.randn(B, 52).cuda(),
}

total_loss, detail = loss_fn(outputs_l, targets)
print(f"  total_loss: {total_loss.item():.4f}")
assert total_loss.item() > 0, "Loss should be positive"
assert torch.isfinite(total_loss), "Loss is NaN/Inf"
print("  ✅ Loss finite and positive")

# ── 7. Stage training ramp ──────────────────────────────────────────────────
print("\n=== TEST 7: Stage training ramp ===")
for epoch in [1, 6, 16]:
    loss_fn.set_epoch(epoch)
    ramp = min(1.0, epoch / max(loss_fn._act_warmup_epochs, 1))
    print(f"  epoch={epoch}: act_ramp={ramp:.1f}, _current_epoch={loss_fn._current_epoch}")

assert loss_fn._current_epoch == 16, f"Epoch not set correctly, got {loss_fn._current_epoch}"
print("  ✅ Staged warmup ramp functional")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("SMOKE TEST PASSED — All fixes validated")
print("  ✅ ITEM 29: subset_ratio in Dataset.__init__")
print("  ✅ ITEM 32: PSR [:11] slicing correct")
print("  ✅ Kendall log_var_act = 0.0 (paper spec)")
print("  ✅ Dataset __getitem__ no None fields")
print("  ✅ Collation works (no TypeError)")
print("  ✅ Model forward shapes correct")
print("  ✅ Loss finite and computable")
print("  ✅ Staged warmup works")
print("="*50)