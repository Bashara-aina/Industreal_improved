#!/bin/bash
# ============================================================
# POPW Multi-Task Training — BACKBONE FINE-TUNING MODE
#
# Launches training with FREEZE_BACKBONE=False so the
# ConvNeXt-Tiny backbone is fine-tuned at a reduced LR
# (BASE_LR * BACKBONE_LR_MULT = 5e-4 * 0.01 = 5e-6).
#
# Heads train at BASE_LR (5e-4) with per-head multipliers.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1   # optional
#   bash scripts/train_finetune_backbone.sh
# ============================================================
set -e

WDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WDIR"

# ── Config ──────────────────────────────────────────────────
export FREEZE_BACKBONE=False
export BACKBONE_LR_MULT=0.01

# Inherit all default config values from config.py, only
# overriding the backbone fine-tuning flags via environment
# (config.py checks os.environ.get for each flag).
CFG_ARGS="--seed 42"

# ── Logging ─────────────────────────────────────────────────
RUN_DIR="$WDIR/src/runs/finetune_backbone"
LOG_DIR="$RUN_DIR/logs"
CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

LOG="$LOG_DIR/train_finetune_backbone.log"

echo "============================================================"
echo "POPW Multi-Task Training — Backbone Fine-Tuning"
echo "  FREEZE_BACKBONE=False"
echo "  BACKBONE_LR_MULT=$BACKBONE_LR_MULT"
echo "  Log: $LOG"
echo "  Checkpoints: $CKPT_DIR"
echo "============================================================"

# ── Run training ─────────────────────────────────────────────
python src/training/train.py $CFG_ARGS \
    --output-root "$RUN_DIR" \
    2>&1 | tee "$LOG"

TRAIN_EXIT=$?
echo "Training exit code: $TRAIN_EXIT"

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Training completed successfully."
    echo "Log: $LOG"
    echo "Checkpoints: $CKPT_DIR"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Training FAILED (exit code $TRAIN_EXIT)."
    echo "Check log: $LOG"
    echo "============================================================"
fi
exit $TRAIN_EXIT
