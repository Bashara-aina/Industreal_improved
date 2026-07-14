#!/usr/bin/env python3
import faulthandler
import signal

faulthandler.enable()
faulthandler.register(signal.SIGUSR1)

import sys
from pathlib import Path

# Resolve symlinks same as train.py
_SRC = Path(__file__).resolve().parent
for _sub in ["models", "training", "evaluation", "data"]:
    _p = _SRC / _sub
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

import config as C
from models import model as model_module
from training import losses as losses_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

B = 2

print("=== LOSS FUNCTION DEEP VERIFICATION ===\n")

# Test 1: Model forward pass
print("Test 1: Model forward pass + FocalLoss")
model = (
    model_module.POPWMultiTaskModel(backbone_type=C.BACKBONE, pretrained=False, use_videomae=False)
    .to(device)
    .eval()
)

images = torch.randn(B, 3, C.IMG_HEIGHT, C.IMG_WIDTH).to(device)
with torch.no_grad():
    outputs = model(images)

cls_preds = outputs["cls_preds"]
reg_preds = outputs["reg_preds"]
print(f"  cls_preds shape: {cls_preds.shape}")
print(f"  reg_preds shape: {reg_preds.shape}")

focal = losses_module.FocalLoss(alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA)
targets = [
    {
        "boxes": torch.tensor([[100, 100, 300, 300]], dtype=torch.float32, device=device),
        "labels": torch.tensor([2], dtype=torch.long, device=device),
    }
    for _ in range(B)
]
N = reg_preds.shape[1]
anchors = torch.zeros(N, 4, device=device)
anchors[:, 2:] = torch.rand(N, 2, device=device) * 200 + 10
anchors[:, :2] = torch.rand(N, 2, device=device) * 500

f_loss, f_debug = focal(cls_preds, reg_preds, anchors, targets)
print(f"  FocalLoss: {f_loss.item():.4f}")
print(f"  Finite: {torch.isfinite(f_loss).item()}")
print(
    f"  DEBUG type: {type(f_debug).__name__}, shape: {f_debug.shape if hasattr(f_debug, 'shape') else 'N/A'}"
)

# Test 2: WingLoss
print("\nTest 2: WingLoss")
wing = losses_module.WingLoss(omega=C.WING_OMEGA, epsilon=C.WING_EPSILON)
pred_kp = torch.randn(B, 17, 2, device=device) * 200
target_kp = torch.randn(B, 17, 2, device=device) * 200
w_loss = wing(pred_kp, target_kp)
print(f"  WingLoss: {w_loss.item():.4f}")
print(f"  Finite: {torch.isfinite(w_loss).item()}")

# Test 3: GIoU
print("\nTest 3: GIoU Loss")
pred_boxes = torch.tensor([[100, 100, 200, 200]] * B, device=device).float()
target_boxes = torch.tensor([[100, 100, 200, 200]] * B, device=device).float()
giou_loss = losses_module.generalized_box_iou_loss(pred_boxes, target_boxes)
print(f"  GIoU (perfect match): {giou_loss.mean().item():.4f}")

# Test 4: LDAMLoss
print("\nTest 4: LDAMLoss")
ldam = losses_module.LDAMLoss(num_classes=C.NUM_CLASSES_ACT, max_m=0.5, s=30)
act_logits = torch.randn(B, C.NUM_CLASSES_ACT, device=device) * 0.1
act_target = torch.randint(0, C.NUM_CLASSES_ACT, (B,), device=device)
ldam_loss = ldam(act_logits, act_target)
print(f"  LDAMLoss: {ldam_loss.item():.4f}")
print(f"  Finite: {torch.isfinite(ldam_loss).item()}")

# Test 5: BinaryFocalLoss (PSR)
print("\nTest 5: BinaryFocalLoss (PSR)")
psr_logits = torch.randn(B, C.NUM_PSR_COMPONENTS, device=device)
psr_target = torch.randint(0, 2, (B, C.NUM_PSR_COMPONENTS), device=device).float()
bf_loss = losses_module.binary_focal_loss(psr_logits, psr_target, alpha=0.25, gamma=2.0)
print(f"  BinaryFocalLoss: {bf_loss.item():.4f}")
print(f"  Finite: {torch.isfinite(bf_loss).item()}")

# Test 6: MultiTaskLoss staged
print("\nTest 6: MultiTaskLoss staged (epoch 1, 10, 20)")
criterion = losses_module.MultiTaskLoss(
    num_classes_act=C.NUM_CLASSES_ACT,
    num_psr_components=C.NUM_PSR_COMPONENTS,
).to(device)

targets_full = {
    "detection": targets,
    "keypoints": torch.randn(B, 17, 2, device=device),
    "pose_confidence": torch.rand(B, 17, device=device),
    "head_pose": torch.randn(B, 9, device=device),
    "activity": act_target,
    "psr_labels": psr_target,
}

for epoch in [1, 10, 20]:
    criterion.set_epoch(epoch)
    loss, ld = criterion(outputs, targets_full)
    print(
        f"  Epoch {epoch}: total={loss.item():.4f}, det={ld['det']:.3f}, hp={ld['head_pose']:.4f}, act={ld['activity']:.3f}, psr={ld['psr']:.4f}"
    )
    print(
        f"    w_det={ld.get('w_det', 0):.3f}, w_pose={ld.get('w_pose', 0):.3f}, w_act={ld.get('w_act', 0):.3f}, w_psr={ld.get('w_psr', 0):.3f}"
    )
    print(
        f"    Kendall: log_var_det={criterion.log_var_det.item():.3f}, log_var_pose={criterion.log_var_pose.item():.3f}"
    )

# Test 7: Backward pass (train mode, no no_grad)
print("\nTest 7: Full backward pass")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion.set_epoch(20)

# Fresh forward pass WITHOUT no_grad
outputs_train = model(images)

optimizer.zero_grad()
loss, ld = criterion(outputs_train, targets_full)
print(f"  Loss before backward: {loss.item():.4f}")

if torch.isfinite(loss):
    loss.backward()
    grads_count = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_grads = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradients: {grads_count} params have non-zero gradients")
    print(f"  Total gradient magnitude: {total_grads:.6f}")
    if grads_count == 0:
        print("  WARNING: No gradients flowing!")
    else:
        print(f"  Backward: OK")
else:
    print(f"  Backward: SKIPPED (loss not finite)")

print("\n=== VERIFICATION COMPLETE ===")
