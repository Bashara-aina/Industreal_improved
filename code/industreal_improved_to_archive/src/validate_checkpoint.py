#!/usr/bin/env python3
"""Standalone validation: loads checkpoint, runs eval, prints metrics."""
import sys, os, gc

# Fix sys.path the same way train.py does
_SRCDIR = os.path.dirname(os.path.abspath(__file__))  # .../src
_SRCPARENT = os.path.dirname(_SRCDIR)  # .../industreal_improved_to_archive

for _p in [
    _SRCDIR,                          # src/ (IMPORTANT: must come first)
    os.path.join(_SRCDIR, 'models'),
    os.path.join(_SRCDIR, 'training'),
    os.path.join(_SRCDIR, 'evaluation'),
    os.path.join(_SRCDIR, 'data'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Root dir last so src/config.py takes precedence
if _SRCPARENT not in sys.path:
    sys.path.append(_SRCPARENT)

import torch
from src import config as C
from models.model import POPWMultiTaskModel
from training.losses import MultiTaskLoss
from evaluation.evaluate import evaluate_all
from data.dataset import get_train_loader
from torch.utils.data import DataLoader

# Constants
VAL_BATCH_SIZE = 2
CKPT_PATH = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Loading checkpoint from {CKPT_PATH}...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
print(f"Checkpoint tag={ckpt.get('tag')}, epoch={ckpt.get('epoch')}")

print("Building model...")
model = POPWMultiTaskModel(
    pretrained=True,
    backbone_type=C.BACKBONE,
    use_hand_film=C.USE_HAND_FILM,
    use_headpose_film=C.USE_HEADPOSE_FILM,
    use_videomae=C.USE_VIDEOMAE,
    train_pose=False,  # Just for evaluation, not training
).to(DEVICE)

# Load weights from checkpoint
ema_state = ckpt.get('ema_shadow', {})
if ema_state:
    model.load_state_dict(ema_state, strict=False)
    print(f"Loaded EMA shadow weights ({len(ema_state)} tensors)")
else:
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    print("Loaded model weights")

model.eval()
for p in model.parameters():
    p.requires_grad = False

# Load criterion state from checkpoint
criterion = MultiTaskLoss(
    num_classes_act=C.NUM_CLASSES_ACT,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=True,
    train_pose=True,
    train_act=True,
    train_psr=True,
    use_kendall=True,
).to(DEVICE)
if 'criterion' in ckpt:
    criterion.load_state_dict(ckpt['criterion'], strict=False)
    print("Loaded criterion state")
criterion.set_epoch(0)
criterion.eval()
for p in criterion.parameters():
    p.requires_grad = False

print("Building val dataset (2% subset)...")
val_loader = get_train_loader(
    max_recordings=None,     # full val split
    batch_size=VAL_BATCH_SIZE,
    num_workers=0,          # stability on RTX 3060
    augment=False,
    sequence_mode=False,
)
print(f"Val loader: {len(val_loader)} batches x batch_size={VAL_BATCH_SIZE}")

print("\nClearing GPU memory...")
torch.cuda.empty_cache()
gc.collect()

print("Running evaluation (max 15 batches)...")
try:
    val_metrics = evaluate_all(model, criterion, val_loader, DEVICE, max_batches=15)
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    for k, v in val_metrics.items():
        print(f"  {k:30s}: {v}")
    print("="*60)
except Exception as e:
    print(f"Evaluation error: {e}")
    import traceback; traceback.print_exc()