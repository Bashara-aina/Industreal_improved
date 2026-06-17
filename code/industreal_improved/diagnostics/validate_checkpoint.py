#!/usr/bin/env python3
"""Run validation on saved checkpoint with reduced val batch size."""
import sys
import os
from pathlib import Path

# Proper path setup matching train.py — must resolve 'src' imports correctly
_SRC = Path(__file__).resolve().parent / 'src'  # .../industreal_improved_to_archive/src
for _sub in ['models', 'training', 'evaluation', 'data', str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# CRITICAL: add project root so `from src import config` resolves in model.py
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

import torch
import gc
import config as C
from models.model import POPWMultiTaskModel
from training.losses import MultiTaskLoss
from evaluation.evaluate import evaluate_all
from data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from torch.utils.data import DataLoader

# Paths
CKPT_PATH = _SRC / 'runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth'
RUN_DIR = _SRC / 'runs/full_multi_task_tma_tbank_benchmark'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")

# Build model
model = POPWMultiTaskModel(
    pretrained=False,
    backbone_type=C.BACKBONE,
    use_headpose_film=C.USE_HEADPOSE_FILM,
    use_hand_film=C.USE_HAND_FILM,
    use_videomae=C.USE_VIDEOMAE,
    train_pose=C.TRAIN_HEAD_POSE,
).to(DEVICE)

# Load EMA weights (stored under 'ema_shadow' key)
ema_state = ckpt.get('ema_shadow', {})
if ema_state:
    model.load_state_dict(ema_state, strict=False)
    print("Loaded EMA weights")
else:
    model.load_state_dict(ckpt.get('model', {}), strict=False)
    print("Loaded model weights from 'model' key")

model.eval()

# Build val dataset with 2% subset
val_ds = IndustRealMultiTaskDataset(
    split='val',
    img_size=C.IMG_SIZE,
    subset_ratio=0.02,
)
print(f"Val dataset: {len(val_ds)} sequences")

# Collate function
val_loader = DataLoader(
    val_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True,
)
print(f"Val loader: {len(val_loader)} batches x batch_size=2")

# Build criterion
criterion = MultiTaskLoss(
    num_classes_act=C.NUM_CLASSES_ACT,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=True,
    train_pose=True,
    train_act=True,
    train_psr=True,
    use_kendall=C.USE_KENDALL,
).to(DEVICE)
criterion.set_epoch(0)
criterion.eval()

# Run evaluation
print("\nRunning evaluation...")
try:
    val_metrics = evaluate_all(
        model,
        criterion,
        val_loader,
        DEVICE,
        max_batches=15,
    )
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, torch.Tensor):
            return f"{v.item():.4f}" if v.numel() == 1 else str(v.shape)
        if isinstance(v, dict):
            return f"dict(len={len(v)})"
        if isinstance(v, list):
            return f"list(len={len(v)})"
        return str(v)

    for k, v in sorted(val_metrics.items()):
        if not k.startswith('_'):
            print(f"  {k:40s}: {_fmt(v)}")
    print("="*60)
except Exception as e:
    print(f"Evaluation error: {e}")
    import traceback; traceback.print_exc()