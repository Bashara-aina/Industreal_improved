#!/usr/bin/env python3
"""
Single-task activity-only ConvNeXt-Tiny training (Opus 141 ACT-MLP-10).

Patches src.config to enable ONLY the activity head (TRAIN_ACT=True,
TRAIN_DET=False, TRAIN_HEAD_POSE=False, TRAIN_PSR=False), then delegates
all training logic to train.py. All train.py CLI arguments are forwarded.

Purpose: isolates whether the multi-task setup causes the activity failure
(41/69 classes at zero accuracy, class collapse at 0.0236) or the backbone
is truly at ceiling (linear probe = 0.2169 ~= 0.2217 baseline).

Usage:
    python src/training/train_singletask_activity.py [train.py args...]

Examples:
    # Fresh start from COCO-pretrained backbone:
    python src/training/train_singletask_activity.py --batch-size 2

    # Resume from crash_recovery checkpoint:
    python src/training/train_singletask_activity.py --batch-size 2 \\
        --resume src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth
"""

import os
import sys
import argparse
from pathlib import Path

# ---- Phase 1: Ensure project root is on sys.path ----
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- Phase 2: Patch config BEFORE any training module imports ----
import src.config as C

C.TRAIN_ACT = True
C.TRAIN_DET = False
C.TRAIN_HEAD_POSE = False
C.TRAIN_PSR = False
C.STAGED_TRAINING = False
# Enable bf16 mixed precision. No PSR head means no seq-loss spikes that
# corrupted the FP16 GradScaler — bf16 is safe and gives ~1.5-2x throughput.
C.MIXED_PRECISION = True
C.AMP_DTYPE = "bf16"

# ---- Phase 3: Now import train (reads patched config at module level) ----
from src.training import train

# Patch the module-level CFG_TRAIN_* globals too so _refresh_runtime_cfg and
# any code that references CFG_TRAIN_ACT etc. reads the correct values.
train.CFG_TRAIN_ACT = True
train.CFG_TRAIN_DET = False
train.CFG_TRAIN_HEAD_POSE = False
train.CFG_TRAIN_PSR = False

# ---- Phase 4: Parse CLI args (mirrors train.py __main__) ----
parser = argparse.ArgumentParser(
    description="Single-task ConvNeXt-Tiny activity training (Opus 141 ACT-MLP-10)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--preset",
    type=str,
    default=None,
    help="Config preset name from config.PRESETS "
    "(e.g. 'recovery', 'benchmark_full', 'benchmark_quick').",
)
parser.add_argument("--max-epochs", type=int, default=None, help="Override C.EPOCHS")
parser.add_argument("--batch-size", type=int, default=None, help="Override C.BATCH_SIZE")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument(
    "--debug", action="store_true", help="Run in debug mode (small dataset, fast validation)"
)
parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed override")
parser.add_argument(
    "--subset-ratio",
    type=float,
    default=getattr(C, "SUBSET_RATIO", 1.0),
    help="Fraction of recordings to use (0.0-1.0)",
)
parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers")
parser.add_argument(
    "--no-staged-training", action="store_true", help="Disable 3-stage progressive training"
)
parser.add_argument("--start-epoch", type=int, default=None, help="Override starting epoch")
parser.add_argument(
    "--reset-scheduler", action="store_true", help="Reset scheduler state after loading checkpoint"
)
parser.add_argument(
    "--reinit-heads", action="store_true", help="Re-initialize det/act/psr heads + FPN from priors"
)

args = parser.parse_args()

# ---- Phase 5: Apply arg overrides (mirrors train.py __main__) ----
if args.preset:
    C.apply_preset(args.preset)
    train._refresh_runtime_cfg()
    train.logger.info(f"[train] Applied preset: {args.preset}")

if args.preset and args.preset.startswith("stage_rf"):
    _stage_root = Path(C.OUTPUT_ROOT).parent / "rf_stages"
    os.environ["OUTPUT_ROOT_OVERRIDE"] = str(_stage_root)
    C.update_dynamic_paths()
    train.logger.info(f"[train] Stage preset detected — redirected OUTPUT_ROOT to {_stage_root}")

if args.max_epochs is not None:
    C.EPOCHS = args.max_epochs
if args.batch_size is not None:
    C.BATCH_SIZE = args.batch_size
    C.EFFECTIVE_BATCH = C.BATCH_SIZE * C.GRAD_ACCUM_STEPS
if args.num_workers is not None:
    C.NUM_WORKERS = args.num_workers

# TRAIN_MAX_STEPS from env
_env_max_steps = int(os.environ.get("TRAIN_MAX_STEPS", "0"))
if _env_max_steps > 0:
    C.TRAIN_MAX_STEPS = _env_max_steps

# EVAL_MAX_BATCHES from env
_env_eval_max_batches = int(os.environ.get("EVAL_MAX_BATCHES", "0"))
if _env_eval_max_batches > 0:
    C.EVAL_MAX_BATCHES = _env_eval_max_batches

if args.no_staged_training:
    C.STAGED_TRAINING = False

if hasattr(args, "debug") and args.debug:
    C.DEBUG_MODE = True
    C.DEBUG_MAX_VIDEOS = 5
    C.VAL_EVERY = 999

if args.seed is not None:
    C.SEED = args.seed

train._refresh_runtime_cfg()

# ---- Phase 6: Run training ----
train.main(args)
