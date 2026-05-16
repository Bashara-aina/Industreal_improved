#!/usr/bin/env python3
"""
Single-batch gradient sanity check for IndustReal multi-task training.
Tests that: (1) model outputs are finite, (2) loss is finite,
(3) gradients flow to all task heads, (4) Kendall log_vars receive gradients.
"""
import sys
from pathlib import Path

# Fix sys.path exactly like train.py does
_SRC = Path(__file__).resolve().parent / 'src'
for _sub in ['models', 'training', 'evaluation', 'data']:
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
import torch

# ---- Data ----
from data.industreal_dataset import IndustRealMultiTaskDataset
from torch.utils.data import DataLoader

def collate_fn(batch):
    """Custom collate to handle detection dict boxes."""
    images = torch.stack([b['images']['rgb'] for b in batch])
    gt_boxes = {'rgb': [b['gt_boxes']['rgb'] for b in batch]}
    gt_classes = {'rgb': [b['gt_classes']['rgb'] for b in batch]}
    head_pose = torch.stack([b['head_pose'] for b in batch])
    psr_labels = torch.stack([b['psr_labels'] for b in batch])
    action_label = torch.stack([b['action_label'] for b in batch])
    hand_joints = torch.stack([b['hand_joints'] for b in batch])
    clip_rgb = torch.stack([b['clip_rgb'] for b in batch]) if batch[0].get('clip_rgb') is not None else None
    metadata = [b['metadata'] for b in batch]
    return (
        {'rgb': images, 'clip_rgb': clip_rgb},
        {
            'detection': [{'boxes': b['gt_boxes']['rgb'], 'labels': b['gt_classes']['rgb']} for b in batch],
            'head_pose': head_pose,
            'psr_labels': psr_labels,
            'activity': action_label,
            'hand_joints': hand_joints,
            'box_mask': None,
        }
    )

# Build dataset (2 recordings, train split)
ds = IndustRealMultiTaskDataset(
    split='train',
    img_size=C.IMG_SIZE,
    augment=False,
    seed=42,
    max_recordings=2,
)
print(f"Dataset: {len(ds)} samples")

loader = DataLoader(
    ds,
    batch_size=1,  # single batch
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=False,
    drop_last=True,
)

# ---- Model ----
from models.model import POPWMultiTaskModel
model = POPWMultiTaskModel(
    pretrained=True,
    backbone_type=getattr(C, 'BACKBONE_TYPE', 'convnext_tiny'),
    use_headpose_film=True,
    use_videomae=False,
    train_pose=False,  # TRAIN_HEAD_POSE=False
).to('cuda')
model.train()

# ---- Loss ----
from training.losses import MultiTaskLoss
criterion = MultiTaskLoss(
    num_classes_act=C.NUM_ACT_CLASSES,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=C.TRAIN_DET,
    train_pose=False,  # TRAIN_HEAD_POSE=False
    train_act=C.TRAIN_ACT,
    train_psr=C.TRAIN_PSR,
    use_kendall=C.USE_KENDALL,
).to('cuda')
criterion.train()

# ---- Optimizer ----
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(criterion.parameters()),
    lr=1e-5,
    weight_decay=0.01,
)

# ---- Single Batch ----
images, targets = next(iter(loader))
images = images['rgb'].to('cuda', non_blocking=True)

# [FIX] sanity_check was missing uint8→float conversion that train.py has.
# ConvNeXt backbone expects float input, not uint8.
if images.dtype == torch.uint8:
    images = images.float().div_(255.0)
    # Normalize with ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    images = (images - mean) / std

clip_rgb = images  # same for now

for i in range(len(targets['detection'])):
    targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].to('cuda')
    targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to('cuda')
targets['head_pose'] = targets['head_pose'].to('cuda', non_blocking=True)
targets['psr_labels'] = targets['psr_labels'].to('cuda', non_blocking=True)
targets['activity'] = targets['activity'].to('cuda', non_blocking=True)
targets['hand_joints'] = targets['hand_joints'].to('cuda', non_blocking=True)
targets['box_mask'] = None

print(f"\nImages shape: {images.shape}")
print(f"Head pose shape: {targets['head_pose'].shape}")
print(f"PSR labels shape: {targets['psr_labels'].shape}")
print(f"Activity shape: {targets['activity'].shape}")
print(f"Detection boxes: {len(targets['detection'])} items")
if targets['detection']:
    print(f"  boxes[0] shape: {targets['detection'][0]['boxes'].shape}")
    print(f"  labels[0] shape: {targets['detection'][0]['labels'].shape}")

# ---- Forward ----

with torch.amp.autocast('cuda', enabled=C.MIXED_PRECISION):
    outputs = model(images, clip_rgb=clip_rgb)

print("\n=== OUTPUT SHAPES ===")
for k, v in outputs.items():
    if isinstance(v, torch.Tensor):
        finite = torch.isfinite(v).all().item()
        print(f"  {k}: {v.shape}, finite={finite}")
    elif isinstance(v, list):
        print(f"  {k}: list[{len(v)}]")
        for i, item in enumerate(v[:2]):
            if isinstance(item, torch.Tensor):
                print(f"    [{i}]: {item.shape}")
    else:
        print(f"  {k}: {type(v)}")

# ---- Loss ----
criterion.set_epoch(1)  # Stage 1
loss, loss_dict = criterion(outputs, targets)

print("\n=== LOSS DICT ===")
for k, v in loss_dict.items():
    print(f"  {k}: {v:.6f}")

print("\n=== KENDALL LOG_VARS ===")
print(f"  log_var_det: {criterion.log_var_det.data.item():.4f}")
print(f"  log_var_pose: {criterion.log_var_pose.data.item():.4f}")
print(f"  log_var_act: {criterion.log_var_act.data.item():.4f}")
print(f"  log_var_psr: {criterion.log_var_psr.data.item():.4f}")

# ---- Backward ----
print("\n=== LOSS VALUE ===")
print(f"  loss.item(): {loss.item():.6f}")
print(f"  torch.isfinite(loss): {torch.isfinite(loss).item()}")

loss.backward()

print("\n=== GRADIENT NORMS (top 15 model params) ===")
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None and param.grad.norm() > 0:
        grad_norms[name] = param.grad.norm().item()
sorted_norms = sorted(grad_norms.items(), key=lambda x: -x[1])[:15]
for name, norm in sorted_norms:
    print(f"  {name}: {norm:.6f}")

print("\n=== GRADIENT NORMS (criterion params) ===")
for name, param in criterion.named_parameters():
    if param.grad is not None and param.grad.norm() > 0:
        print(f"  criterion.{name}: {param.grad.norm().item():.6f}")

print("\n=== BACKBONE GRADIENT STATUS ===")
backbone_grads = [(n, p.grad.norm().item()) for n, p in model.named_parameters()
                   if 'backbone' in n and p.grad is not None and p.grad.norm() > 0]
if backbone_grads:
    for n, g in sorted(backbone_grads, key=lambda x: -x[1])[:5]:
        print(f"  {n}: {g:.6f}")
else:
    print("  NO BACKBONE GRADIENTS FOUND!")

print("\n=== DETECTION HEAD GRADIENT STATUS ===")
det_grads = [(n, p.grad.norm().item()) for n, p in model.named_parameters()
             if ('detection' in n or 'fpn' in n or 'cls' in n or 'reg' in n) and p.grad is not None and p.grad.norm() > 0]
if det_grads:
    for n, g in sorted(det_grads, key=lambda x: -x[1])[:5]:
        print(f"  {n}: {g:.6f}")
else:
    print("  NO DETECTION HEAD GRADIENTS FOUND!")

print("\n=== ACTIVITY HEAD GRADIENT STATUS ===")
act_grads = [(n, p.grad.norm().item()) for n, p in model.named_parameters()
             if ('activity' in n or 'act' in n or 'tcn' in n or 'vit' in n) and p.grad is not None and p.grad.norm() > 0]
if act_grads:
    for n, g in sorted(act_grads, key=lambda x: -x[1])[:5]:
        print(f"  {n}: {g:.6f}")
else:
    print("  NO ACTIVITY HEAD GRADIENTS FOUND!")

print("\nDone. Check gradients above - all should be > 0 for active tasks.")