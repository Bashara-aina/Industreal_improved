#!/usr/bin/env python3
"""
Minimal real-data smoke test: load real dataset batch, run one training step, report.
Uses same import path pattern as scripts/smoke_test.py and scripts/test_e2e_training.py.
"""
import sys, os, torch, time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, os.pardir))
SRC_DIR = os.path.join(WORK_DIR, 'src')
sys.path.insert(0, WORK_DIR)
sys.path.insert(1, os.path.join(SRC_DIR, 'models'))
sys.path.insert(2, os.path.join(SRC_DIR, 'training'))
sys.path.insert(3, os.path.join(SRC_DIR, 'evaluation'))
sys.path.insert(4, SRC_DIR)

import config as C
import torch
import torch.optim as optim

# Override config for speed
C.DEBUG_MODE = True
C.DEBUG_MAX_VIDEOS = 3
C.NUM_EPOCHS = 1
C.VAL_EVERY = 999
C.BATCH_SIZE = 2
C.GRAD_ACCUM_STEPS = 1
C.USE_KENDALL = True
C.TRAIN_AR = True
C.TRAIN_ASD = True
C.TRAIN_PSR = True
C.TRAIN_DET = True
C.TRAIN_HEAD_POSE = False

from models import model as model_module
from training import losses as losses_module
from data.industreal_dataset import IndustRealMultiTaskDataset as dataset_module

print("="*60)
print("MINIMAL REAL-DATA SMOKE TEST")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Build model
print("\n[1] Building model...")
model = model_module.POPWMultiTaskModel(
    backbone_type=C.BACKBONE,
    pretrained=False,
    use_videomae=False,
).to(device)
model.train()
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Model loaded: {total_params:.2f}M params")

# Build optimizer
print("\n[2] Building optimizer...")
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# Build loss — use exact signature from test_e2e_training.py
print("\n[3] Building loss...")
loss_fn = losses_module.MultiTaskLoss(
    num_classes_act=C.NUM_CLASSES_ACT,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=C.TRAIN_DET,
    train_pose=C.TRAIN_HEAD_POSE,
    train_act=C.TRAIN_AR,
    train_psr=C.TRAIN_PSR,
    use_kendall=C.USE_KENDALL,
)

# Build EMA
print("\n[4] Building EMA...")
ema = model_module.EMA(model, decay=C.EMA_DECAY)

# Load dataset
print("\n[5] Loading dataset...")
ds = dataset_module(
    split='train',
    augment=False,
    max_recordings=C.DEBUG_MAX_VIDEOS,
    subset_ratio=C.SUBSET_RATIO,
)
print(f"  Dataset: {len(ds)} samples")

loader = torch.utils.data.DataLoader(
    ds, batch_size=C.BATCH_SIZE,
    shuffle=False, num_workers=0,
    pin_memory=False,
)

print("\n[6] Running one training step...")
batch = next(iter(loader))
if batch is None:
    print("  ERROR: No batch returned")
    sys.exit(1)

# Unpack batch (no custom collate_fn → returns dict with 'images' dict)
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
frames = batch['images']['rgb'].to(device).float().div(255.0)  # [B, 3, H, W] uint8→float32 [0,1]

# Build targets dict matching MultiTaskLoss.forward expected format
targets = {
    'detection': [{'boxes': batch['gt_boxes']['rgb'][i], 'labels': batch['gt_classes']['rgb'][i]}
                  for i in range(frames.size(0))],
    'keypoints': batch['hand_joints'].reshape(frames.size(0), 26, 2)[:, :17, :],  # [B, 17, 2] from [B, 52]
    'pose_confidence': torch.ones(frames.size(0), 17, device=device),  # dense hand joints, no confidence
    'head_pose': batch['head_pose'],  # already on device
    'activity': batch['action_label'],  # already on device
    'psr_labels': batch['psr_labels'],  # already on device
}

t0 = time.time()
optimizer.zero_grad()

# Forward pass
outputs = model(frames)

# Ensure float
for key in ['cls_preds', 'reg_preds', 'heatmaps', 'keypoints',
            'pose_confidence', 'head_pose', 'act_logits', 'psr_logits']:
    if key in outputs and isinstance(outputs[key], torch.Tensor):
        outputs[key] = outputs[key].float()

# Compute loss
total_loss, loss_dict = loss_fn(outputs, targets)

print(f"  Total loss: {total_loss.item():.4f}")
for k, v in loss_dict.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {v.item():.4f}")

# Backward
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
optimizer.step()
ema.update()

dt = time.time() - t0

# Check for NaN
has_nan = False
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"  WARNING: NaN in grad({name})")
        has_nan = True

print(f"\n  Step time: {dt:.2f}s")
print(f"  Loss finite: {torch.isfinite(total_loss).item()}")

if has_nan:
    print("\n❌ SMOKE TEST FAILED: NaN detected in gradients")
    sys.exit(1)
elif not torch.isfinite(total_loss):
    print("\n❌ SMOKE TEST FAILED: Loss is not finite")
    sys.exit(1)
else:
    print("\n✅ SMOKE TEST PASSED — real data, one step, all finite")
    sys.exit(0)