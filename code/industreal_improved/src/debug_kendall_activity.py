#!/usr/bin/env python3
"""
Narrowed gradient flow diagnosis: ActivityHead receives gradients
when loss is direct CE, but NOT when going through MultiTaskLoss.
This tells us the bug is in how MultiTaskLoss computes activity loss contribution.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
for _sub in ["models", "training", "evaluation", "data"]:
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
import torch
import torch.nn.functional as F

from models.model import POPWMultiTaskModel
from data.industreal_dataset import IndustRealMultiTaskDataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    import torch

    images = torch.stack([b["images"]["rgb"] for b in batch])
    gt_boxes = {"rgb": [b["gt_boxes"]["rgb"] for b in batch]}
    gt_classes = {"rgb": [b["gt_classes"]["rgb"] for b in batch]}
    head_pose = torch.stack([b["head_pose"] for b in batch])
    psr_labels = torch.stack([b["psr_labels"] for b in batch])
    action_label = torch.stack([b["action_label"] for b in batch])
    hand_joints = torch.stack([b["hand_joints"] for b in batch])
    clip_rgb = (
        torch.stack([b["clip_rgb"] for b in batch])
        if batch[0].get("clip_rgb") is not None
        else None
    )
    return (
        {"rgb": images, "clip_rgb": clip_rgb},
        {
            "detection": [
                {"boxes": b["gt_boxes"]["rgb"], "labels": b["gt_classes"]["rgb"]} for b in batch
            ],
            "head_pose": head_pose,
            "psr_labels": psr_labels,
            "activity": action_label,
            "hand_joints": hand_joints,
            "box_mask": None,
        },
    )


def get_ah_grads(model):
    return {
        n: p.grad.norm().item()
        for n, p in model.named_parameters()
        if "activity_head" in (n or "") and p.grad is not None and p.grad.norm() > 0
    }


def get_det_grads(model):
    return {
        n: p.grad.norm().item()
        for n, p in model.named_parameters()
        if "detection_head" in (n or "") and p.grad is not None and p.grad.norm() > 0
    }


# ---- Data ----
ds = IndustRealMultiTaskDataset(
    split="train", img_size=C.IMG_SIZE, augment=False, seed=42, max_recordings=2
)
loader = DataLoader(
    ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=False,
    drop_last=True,
)

# ---- Model ----
model = POPWMultiTaskModel(
    pretrained=True,
    backbone_type=getattr(C, "BACKBONE", "convnext_tiny"),
    use_headpose_film=True,
    use_videomae=False,
    train_pose=False,
).to("cuda")
model.train()

images, targets = next(iter(loader))
images = images["rgb"].to("cuda", non_blocking=True)
if images.dtype == torch.uint8:
    images = images.float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    images = (images - mean) / std

for i in range(len(targets["detection"])):
    targets["detection"][i]["boxes"] = targets["detection"][i]["boxes"].to("cuda")
    targets["detection"][i]["labels"] = targets["detection"][i]["labels"].to("cuda")
targets["head_pose"] = targets["head_pose"].to("cuda", non_blocking=True)
targets["psr_labels"] = targets["psr_labels"].to("cuda", non_blocking=True)
targets["activity"] = targets["activity"].to("cuda", non_blocking=True)
targets["hand_joints"] = targets["hand_joints"].to("cuda", non_blocking=True)
targets["box_mask"] = None
targets["video_ids"] = [f"test_{i}" for i in range(2)]

from training.losses import MultiTaskLoss

# ============================================================
# TEST: Activity + Detection Kendall (2 tasks)
# ============================================================
print("\n" + "=" * 60)
print("TEST: Activity + Detection Kendall (2 tasks), epoch=16")
print("=" * 60)

criterion = MultiTaskLoss(
    num_classes_act=C.NUM_ACT_CLASSES,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=True,
    train_pose=False,
    train_act=True,
    train_psr=False,
    use_kendall=True,
).to("cuda")
criterion.train()
criterion.set_epoch(16)

model.zero_grad()
criterion.zero_grad()

with torch.amp.autocast("cuda", enabled=C.MIXED_PRECISION):
    outputs = model(images, clip_rgb=None)

loss, loss_dict = criterion(outputs, targets)

print(f"\n  Losses:")
for k, v in loss_dict.items():
    if not k.startswith("w_") and not k.startswith("log_var") and k != "act_ramp":
        print(f"    {k}: {v:.6f}")

print(
    f"\n  Kendall log_vars: det={criterion.log_var_det.data.item():.4f}, act={criterion.log_var_act.data.item():.4f}"
)
print(
    f"  prec_act = exp(-log_var_act) = {torch.exp(-criterion.log_var_act.clamp(-4, 2)).item():.6f}"
)

print(f"\n  Running backward on total={loss.item():.4f} ...")
loss.backward()

ah = get_ah_grads(model)
det = get_det_grads(model)

print(f"\n  Activity head grads: {len(ah)}")
if ah:
    for n, g in sorted(ah.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n}: {g:.6f}")
else:
    print(f"  !! Activity head: NO GRADIENTS !!")

print(f"\n  Detection head grads: {len(det)}")
if det:
    for n, g in sorted(det.items(), key=lambda x: -x[1])[:3]:
        print(f"    {n}: {g:.6f}")

# ============================================================
# TEST: Direct CE backward (no Kendall)
# ============================================================
print("\n" + "=" * 60)
print("TEST: Direct CE backward on act_logits (no Kendall)")
print("=" * 60)

model.zero_grad()
criterion.zero_grad()

with torch.amp.autocast("cuda", enabled=C.MIXED_PRECISION):
    outputs2 = model(images, clip_rgb=None)

act_logits = outputs2["act_logits"]
loss_ce = F.cross_entropy(act_logits, targets["activity"])
print(f"  Direct CE loss: {loss_ce.item():.4f}")

loss_ce.backward()

ah2 = get_ah_grads(model)
det2 = get_det_grads(model)

print(f"\n  Activity head grads (direct CE): {len(ah2)}")
if ah2:
    for n, g in sorted(ah2.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n}: {g:.6f}")

print(f"\n  Detection head grads (after CE only): {len(det2)}")

# ============================================================
# KEY INSIGHT: Check if the Kendall total's grad_fn is actually
# connected to activity_head
# ============================================================
print("\n" + "=" * 60)
print("KEY TEST: Is Kendall total.backward() actually calling")
print("          the activity head's backward?")
print("=" * 60)

model.zero_grad()
criterion.zero_grad()

with torch.amp.autocast("cuda", enabled=C.MIXED_PRECISION):
    outputs3 = model(images, clip_rgb=None)

loss3, _ = criterion(outputs3, targets)

print(f"  loss3.requires_grad: {loss3.requires_grad}")
print(f"  loss3.grad_fn: {loss3.grad_fn}")


# Check if the graph from loss3 includes activity_head
# Run backward with hooks to see where it goes
class TraceHook:
    def __init__(self, name):
        self.name = name
        self.count = 0

    def __call__(self, module, grad_input, grad_output):
        self.count += 1
        if self.count <= 3:
            print(
                f"  [Hook] {self.name}: grad_out={grad_output[0].norm().item() if grad_output[0] is not None else None}"
            )


ah = model.activity_head
handles = []
handles.append(ah.proj_features.register_full_backward_hook(TraceHook("proj_features")))
handles.append(ah.tcn.register_full_backward_hook(TraceHook("tcn")))
handles.append(ah.vit[0].register_full_backward_hook(TraceHook("vit[0]")))
handles.append(ah.activity_classifier.register_full_backward_hook(TraceHook("activity_classifier")))

det_h = model.detection_head
handles.append(det_h.cls_score.register_full_backward_hook(TraceHook("det_cls_score")))
handles.append(det_h.reg_pred.register_full_backward_hook(TraceHook("det_reg_pred")))

loss3.backward()

for h in handles:
    h.remove()

# ============================================================
# CHECK: The Kendall total computation uses prec_act * loss_act
# Let's manually check if prec_act has a gradient that could
# block the backward chain
# ============================================================
print("\n" + "=" * 60)
print("CHECK: Does prec_act carry a gradient that blocks?")
print("=" * 60)

model.zero_grad()
criterion.zero_grad()

with torch.amp.autocast("cuda", enabled=C.MIXED_PRECISION):
    outputs4 = model(images, clip_rgb=None)

loss4, _ = criterion(outputs4, targets)

# Manually compute the gradient of loss4 w.r.t. act_logits
act_logits4 = outputs4["act_logits"]
act_logits4.retain_grad()
loss4.backward()

print(
    f"  act_logits4.grad: {act_logits4.grad.norm().item() if act_logits4.grad is not None else None}"
)
print(f"  act_logits4.grad_fn: {act_logits4.grad_fn}")

# If act_logits has grad but activity_head params don't,
# the problem is in the forward pass (detach somewhere)

# ============================================================
# FINAL CHECK: Test with no grad on backbone (isolate head)
# ============================================================
print("\n" + "=" * 60)
print("FINAL: Freeze backbone, only train activity_head")
print("=" * 60)

for param in model.parameters():
    if "activity_head" not in (param.name or ""):
        param.requires_grad = False

model.zero_grad()
criterion.zero_grad()
for param in model.parameters():
    if "activity_head" not in (param.name or ""):
        param.requires_grad = False

with torch.amp.autocast("cuda", enabled=C.MIXED_PRECISION):
    outputs5 = model(images, clip_rgb=None)

loss5, _ = criterion(outputs5, targets)
print(f"  total={loss5.item():.4f}")
loss5.backward()

ah5 = get_ah_grads(model)
if ah5:
    print(f"  Activity head grads (frozen backbone): {len(ah5)}")
    for n, g in sorted(ah5.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n}: {g:.6f}")
else:
    print(f"  !! Activity head: STILL NO GRADIENTS !!")

# Re-enable gradients
for param in model.parameters():
    param.requires_grad = True

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
