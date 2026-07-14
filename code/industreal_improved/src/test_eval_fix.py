#!/usr/bin/env python3
"""
Minimal eval test — runs evaluate_all() with a small subset
to verify the defensive fixes work before launching full training.
"""

import sys, os, torch, logging
from pathlib import Path

# Match the same path setup that train.py uses
_SRC = Path("/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src")
_PARENT = _SRC.parent  # .../industreal_improved_to_archive/
for _sub in ["models", "training", "evaluation", "data", str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
os.chdir(_PARENT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"

logger.info(f"Device: {DEVICE}")
logger.info(f"Loading checkpoint: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
logger.info(f"  epoch={ckpt.get('epoch')} step={ckpt.get('step')} ")

# --- Build model ---
import config as C
from models.model import POPWMultiTaskModel

model = POPWMultiTaskModel(
    pretrained=False,  # checkpoint has trained weights
    backbone_type=str(getattr(C, "BACKBONE", "convnext_tiny")),
    use_hand_film=bool(getattr(C, "USE_HAND_FILM", True)),
    use_headpose_film=bool(getattr(C, "USE_HEADPOSE_FILM", False)),
    use_videomae=bool(getattr(C, "USE_VIDEOMAE", False)),
    train_pose=bool(getattr(C, "TRAIN_HEAD_POSE", False)),
)
model.to(DEVICE)
model.eval()

# Load weights
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
if missing:
    logger.warning(f"  Missing keys: {missing[:5]}...")
if unexpected:
    logger.warning(f"  Unexpected keys: {unexpected[:5]}...")
logger.info("  Model loaded OK")

# --- Build criterion ---
from training.losses import MultiTaskLoss

criterion = MultiTaskLoss(
    num_classes_act=C.NUM_CLASSES_ACT,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=True,
    train_pose=bool(getattr(C, "TRAIN_HEAD_POSE", False)),
    train_act=True,
    train_psr=True,
    use_kendall=bool(getattr(C, "USE_KENDALL", True)),
)
criterion.to(DEVICE)

# --- Build val loader (small subset for test) ---
import config as C
from data.industreal_dataset import IndustRealMultiTaskDataset
from torch.utils.data import DataLoader

val_ds = IndustRealMultiTaskDataset(
    split="val",
    img_size=C.IMG_SIZE,
    augment=False,
    seed=C.SEED,
)
val_loader = DataLoader(
    val_ds,
    batch_size=2,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
    collate_fn=getattr(
        __import__("data.industreal_dataset", fromlist=["collate_fn"]), "collate_fn"
    ),
)
logger.info(f"Val dataset: {len(val_ds)} samples, {len(val_loader)} batches")

# --- Run eval with small max_batches to test the fix ---
from evaluation.evaluate import evaluate_all

logger.info("=" * 60)
logger.info("Running eval test (first 50 batches)...")
logger.info("=" * 60)

try:
    metrics = evaluate_all(
        model,
        criterion,
        val_loader,
        DEVICE,
        max_batches=50,  # TEST: only 50 batches
    )
    logger.info("=" * 60)
    logger.info("EVAL TEST PASSED ✓")
    logger.info(f"  loss={metrics.get('loss', -1):.4f}")
    logger.info(f"  det_mAP50={metrics.get('det_mAP50', -1):.4f}")
    logger.info(f"  act_clip_accuracy={metrics.get('act_clip_accuracy', -1):.4f}")
    logger.info(f"  psr_overall_f1={metrics.get('psr_overall_f1', -1):.4f}")
    logger.info(f"  head_pose_MAE={metrics.get('head_pose_MAE', -1):.4f}")
    logger.info(f"  assembly_state_f1={metrics.get('assembly_state_f1', -1):.4f}")
    logger.info("=" * 60)
except Exception as e:
    logger.error("=" * 60)
    logger.error(f"EVAL TEST FAILED: {type(e).__name__}: {e}")
    import traceback

    logger.error(traceback.format_exc())
    logger.error("=" * 60)
    sys.exit(1)
