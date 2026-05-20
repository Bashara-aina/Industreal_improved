#!/usr/bin/env python3
"""
Gradient flow diagnosis: ActivityHead vs DetectionHead.
Tests: (1) isolated activity head backward, (2) full model gradient trace at epoch 16.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / 'src'
for _sub in ['models', 'training', 'evaluation', 'data']:
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import POPWMultiTaskModel, ActivityHead
from data.industreal_dataset import IndustRealMultiTaskDataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    images = torch.stack([b['images']['rgb'] for b in batch])
    gt_boxes = {'rgb': [b['gt_boxes']['rgb'] for b in batch]}
    gt_classes = {'rgb': [b['gt_classes']['rgb'] for b in batch]}
    head_pose = torch.stack([b['head_pose'] for b in batch])
    psr_labels = torch.stack([b['psr_labels'] for b in batch])
    action_label = torch.stack([b['action_label'] for b in batch])
    hand_joints = torch.stack([b['hand_joints'] for b in batch])
    clip_rgb = torch.stack([b['clip_rgb'] for b in batch]) if batch[0].get('clip_rgb') is not None else None
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


# ---- Data ----
ds = IndustRealMultiTaskDataset(split='train', img_size=C.IMG_SIZE, augment=False, seed=42, max_recordings=2)
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False, drop_last=True)

# ---- Model ----
from models.model import POPWMultiTaskModel
model = POPWMultiTaskModel(
    pretrained=True,
    backbone_type=getattr(C, 'BACKBONE', 'convnext_tiny'),
    use_headpose_film=True,
    use_videomae=False,
    train_pose=False,
).to('cuda')
model.train()

# ---- Data batch ----
images, targets = next(iter(loader))
images = images['rgb'].to('cuda', non_blocking=True)
if images.dtype == torch.uint8:
    images = images.float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    images = (images - mean) / std

for i in range(len(targets['detection'])):
    targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].to('cuda')
    targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to('cuda')
targets['head_pose'] = targets['head_pose'].to('cuda', non_blocking=True)
targets['psr_labels'] = targets['psr_labels'].to('cuda', non_blocking=True)
targets['activity'] = targets['activity'].to('cuda', non_blocking=True)
targets['hand_joints'] = targets['hand_joints'].to('cuda', non_blocking=True)
targets['box_mask'] = None
targets['video_ids'] = [f'test_{i}' for i in range(2)]

# ============================================================
# TEST 1: Isolated ActivityHead - direct CE backward
# ============================================================
print("\n" + "="*60)
print("TEST 1: Isolated ActivityHead backward (no Kendall)")
print("="*60)

act_head = ActivityHead(
    c5_channels=768,  # ConvNeXt-Tiny C5
    p4_channels=256,
    det_conf_size=24,
    embed_dim=512,
    num_classes=C.NUM_ACT_CLASSES,
    dropout=0.1,
    window_size=16,
    use_vit=True,
    vit_drop_path=0.1,
    use_videomae=False,
).to('cuda')
act_head.train()

# Fake inputs that match real feature dimensions
proj_feat = torch.randn(2, 512, device='cuda', requires_grad=True)
det_conf = torch.randn(2, 24, device='cuda')
c5_fake = torch.randn(2, 768, 22, 40, device='cuda')
p4_fake = torch.randn(2, 256, 45, 80, device='cuda')

# Build activity_proj exactly like model.py does
activity_proj = torch.cat([
    det_conf,
    F.adaptive_avg_pool2d(c5_fake, 1).flatten(1),
    F.adaptive_avg_pool2d(p4_fake, 1).flatten(1),
], dim=1)
proj_feat_direct = act_head.proj_features(activity_proj)

# Temporal bank: simple repeat (no detach)
bank_output = proj_feat_direct.unsqueeze(1).expand(-1, 16, -1)  # [B, T=16, 512]

# Forward
act_logits = act_head(proj_feat=proj_feat_direct, temporal_bank=bank_output, videomae_feat=None)

# Direct CE loss
fake_target = torch.zeros(2, dtype=torch.long, device='cuda')
loss_act = F.cross_entropy(act_logits, fake_target)

print(f"  act_logits: {act_logits.shape}, loss_act: {loss_act.item():.4f}")
print(f"  loss_act.requires_grad: {loss_act.requires_grad}, grad_fn: {loss_act.grad_fn}")

loss_act.backward()

ah_grads = {n: p.grad.norm().item() for n, p in act_head.named_parameters() if p.grad is not None}
print(f"\n  ActivityHead params WITH grads: {len(ah_grads)}")
for n, g in sorted(ah_grads.items(), key=lambda x: -x[1]):
    print(f"    {n}: {g:.6f}")

no_grad = [n for n, p in act_head.named_parameters() if p.grad is None]
print(f"  ActivityHead params NO grads: {no_grad}")

# ============================================================
# TEST 2: Full model at epoch=16 (stage 3) - all heads active
# ============================================================
print("\n" + "="*60)
print("TEST 2: Full model backward at epoch=16 (stage 3)")
print("="*60)

from training.losses import MultiTaskLoss
criterion = MultiTaskLoss(
    num_classes_act=C.NUM_ACT_CLASSES,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=C.TRAIN_DET,
    train_pose=False,
    train_act=C.TRAIN_ACT,
    train_psr=C.TRAIN_PSR,
    use_kendall=C.USE_KENDALL,
).to('cuda')
criterion.train()
criterion.set_epoch(16)  # Stage 3

model.zero_grad()
criterion.zero_grad()

with torch.amp.autocast('cuda', enabled=C.MIXED_PRECISION):
    outputs = model(images, clip_rgb=None)

loss, loss_dict = criterion(outputs, targets)

print(f"\n  Loss dict at epoch 16:")
for k, v in loss_dict.items():
    print(f"    {k}: {v:.6f}")

print(f"\n  Kendall log_vars at epoch 16:")
print(f"    log_var_det: {criterion.log_var_det.data.item():.4f}")
print(f"    log_var_act: {criterion.log_var_act.data.item():.4f}")

# Check: what is prec_act at epoch 16?
lv_act = criterion.log_var_act.clamp(-4.0, 2.0).item()
prec_act = torch.exp(torch.tensor(-lv_act)).item()
print(f"    prec_act (exp(-log_var_act)): {prec_act:.6f}")
print(f"    act_ramp: {min(1.0, 16 / max(criterion._act_warmup_epochs, 1)):.4f}")

loss.backward()

# Activity head grads
print(f"\n  Activity head gradient check:")
ah_model_grads = {}
for name, param in model.named_parameters():
    if 'activity_head' in name and param.grad is not None:
        ah_model_grads[name] = param.grad.norm().item()

if ah_model_grads:
    print(f"  Activity head params WITH gradients ({len(ah_model_grads)}):")
    for n, g in sorted(ah_model_grads.items(), key=lambda x: -x[1])[:10]:
        print(f"    {n}: {g:.6f}")
else:
    print(f"  Activity head: NO gradients on any parameter!")

# Detection head grads for comparison
print(f"\n  Detection head gradient check:")
det_model_grads = {}
for name, param in model.named_parameters():
    if 'detection_head' in name and param.grad is not None:
        det_model_grads[name] = param.grad.norm().item()

if det_model_grads:
    print(f"  Detection head params WITH gradients ({len(det_model_grads)}):")
    for n, g in sorted(det_model_grads.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n}: {g:.6f}")
else:
    print(f"  Detection head: NO gradients!")

# Criterion log_var grads
print(f"\n  Criterion log_var gradients:")
for name, param in criterion.named_parameters():
    if param.grad is not None:
        print(f"    {name}: grad={param.grad.item():.6f}")

# Backbone grads
print(f"\n  Backbone gradient sample:")
bb_grads = [(n, p.grad.norm().item()) for n, p in model.named_parameters()
           if 'backbone' in n and p.grad is not None and p.grad.norm() > 0]
for n, g in sorted(bb_grads, key=lambda x: -x[1])[:5]:
    print(f"    {n}: {g:.6f}")

# ============================================================
# TEST 3: Activity head ONLY - with same Kendall structure
# ============================================================
print("\n" + "="*60)
print("TEST 3: Activity head + Kendall staged at epoch 16")
print("="*60)

criterion2 = MultiTaskLoss(
    num_classes_act=C.NUM_ACT_CLASSES,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=False,  # only activity
    train_pose=False,
    train_act=True,
    train_psr=False,
    use_kendall=True,
).to('cuda')
criterion2.train()
criterion2.set_epoch(16)

# Re-run forward pass
model.zero_grad()
criterion2.zero_grad()

# Simulate just activity head forward
with torch.amp.autocast('cuda', enabled=C.MIXED_PRECISION):
    outputs2 = model(images, clip_rgb=None)

loss2, loss_dict2 = criterion2(outputs2, targets)

print(f"  Loss dict (act only):")
for k, v in loss_dict2.items():
    print(f"    {k}: {v:.6f}")

print(f"  log_var_act: {criterion2.log_var_act.data.item():.4f}")
print(f"  prec_act: {torch.exp(-criterion2.log_var_act.clamp(-4,2)).item():.6f}")

loss2.backward()

ah_grads2 = {n: p.grad.norm().item() for n, p in model.named_parameters()
             if 'activity_head' in name and p.grad is not None}
if ah_grads2:
    print(f"\n  Activity head grads (act-only loss): {len(ah_grads2)} params")
    for n, g in sorted(ah_grads2.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n}: {g:.6f}")
else:
    print(f"\n  Activity head: STILL NO GRADIENTS even with act-only loss!")

# ============================================================
# TEST 4: Direct CE on act_logits from full model (no Kendall)
# ============================================================
print("\n" + "="*60)
print("TEST 4: Direct CE on act_logits (no Kendall) from full model")
print("="*60)

criterion3 = MultiTaskLoss(
    num_classes_act=C.NUM_ACT_CLASSES,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=False,
    train_pose=False,
    train_act=True,
    train_psr=False,
    use_kendall=False,  # no Kendall to simplify
).to('cuda')
criterion3.train()
criterion3.set_epoch(16)

model.zero_grad()
criterion3.zero_grad()

with torch.amp.autocast('cuda', enabled=C.MIXED_PRECISION):
    outputs3 = model(images, clip_rgb=None)

# Direct activity loss without Kendall
act_logits3 = outputs3['act_logits']
fake_target = torch.zeros(2, dtype=torch.long, device='cuda')
loss_act3 = F.cross_entropy(act_logits3, fake_target)

print(f"\n  act_logits requires_grad: {act_logits3.requires_grad}")
print(f"  act_logits.grad_fn: {act_logits3.grad_fn}")
print(f"  loss_act3: {loss_act3.item():.4f}")

# Check act_logits before backward
print(f"\n  Running backward on DIRECT activity CE loss...")
loss_act3.backward()

# Check after backward
ah_grads3 = {}
for name, param in model.named_parameters():
    if 'activity_head' in name and param.grad is not None:
        ah_grads3[name] = param.grad.norm().item()

if ah_grads3:
    print(f"  Activity head grads (after direct backward): {len(ah_grads3)}")
    for n, g in sorted(ah_grads3.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n}: {g:.6f}")
else:
    print(f"  Activity head: NO GRADIENTS even after direct CE backward!")

# Check backbone params near activity path
print(f"\n  Backbone params feeding activity head:")
for name, param in model.named_parameters():
    if 'backbone' in name and param.grad is not None and param.grad.norm() > 0:
        print(f"    {name}: {param.grad.norm().item():.6f}")

# ============================================================
# TEST 5: Trace activity_proj gradient flow
# ============================================================
print("\n" + "="*60)
print("TEST 5: Trace activity_proj gradient through model")
print("="*60)

model.zero_grad()
criterion.zero_grad()

# Intercept: hook into activity_proj output
activity_proj = None

def hook_fn(module, input, output):
    global activity_proj
    activity_proj = output
    print(f"  [FORWARD HOOK] proj_features output: {output.shape}, requires_grad: {output.requires_grad}")
    print(f"    output.grad_fn: {output.grad_fn}")

handle = model.activity_head.proj_features.register_forward_hook(hook_fn)

with torch.amp.autocast('cuda', enabled=C.MIXED_PRECISION):
    outputs4 = model(images, clip_rgb=None)

handle.remove()

# Now check if activity_proj has grad_fn after full backward
# Run backward through criterion
loss4, _ = criterion(outputs4, targets)
print(f"\n  Running backward through criterion...")
loss4.backward()

if activity_proj is not None:
    print(f"\n  activity_proj.requires_grad: {activity_proj.requires_grad}")
    print(f"  activity_proj.grad_fn: {activity_proj.grad_fn}")

# Check proj_features grad
for name, param in model.activity_head.named_parameters():
    if 'proj_features' in name:
        print(f"  {name}: grad={param.grad.norm().item() if param.grad is not None else 'None'}")

print("\n" + "="*60)
print("DONE")
print("="*60)