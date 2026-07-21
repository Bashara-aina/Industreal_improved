#!/usr/bin/env bash
# MViTv2-S fine-tuning training (Opus 144)
#
# Frozen probe = 0.3810 (above 0.30 threshold), need fine-tuning to close gap
# to MViTv2-S SOTA on Kinetics-400 (0.622 Top-1).
#
# PARALLEL EXECUTION (revised 2026-07-07):
#   GPU 0: V5 multi-task (stage_rf4 + 9 fixes, running)
#   GPU 1: MViTv2-S fine-tune (THIS SCRIPT, after single-task det finishes)
#
# Memory budget: ~3.5 GB with gradient checkpointing + FP16 (batch=2, T=16).
# Fits comfortably on RTX 5060 Ti 16 GB.
#
# Usage:
#   bash scripts/train_mvit_finetune.sh
# ============================================================
set -uo pipefail

# ── CUDA Configuration ──────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ───────────────────────────────────────────────────
WDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WDIR"

# ── Logging ─────────────────────────────────────────────────
SAVE_DIR="$WDIR/src/runs/mvit_finetune"
LOG_DIR="$SAVE_DIR/logs"
CKPT_DIR="$SAVE_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

LOG="$LOG_DIR/train_mvit_finetune.log"

echo "============================================================"
echo "MViTv2-S Fine-Tuning (Opus 144)"
echo "  Backbone: mvit_v2_s (Kinetics-400 pretrained)"
echo "  Frozen probe: 0.3810 (threshold 0.30)"
echo "  Log: $LOG"
echo "  Checkpoints: $CKPT_DIR"
echo "============================================================"

# ── Run training ────────────────────────────────────────────
python3 -u -m src.training.train_video_finetune \
    --backbone mvit_v2_s \
    --batch-size 2 \
    --epochs 20 \
    --lr 5e-5 \
    --backbone-lr 1e-5 \
    --warmup-epochs 3 \
    --clip-len 16 \
    --stride 8 \
    --weight-decay 1e-4 \
    --save-dir "$SAVE_DIR" \
    --num-workers 4 \
    --seed 42 \
    >> "$LOG" 2>&1

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "MViTv2-S fine-tuning completed successfully."
    echo "Checkpoints: $CKPT_DIR"
    echo "Results: $SAVE_DIR/results.json"
    echo "Log: $LOG"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "MViTv2-S fine-tuning FAILED (exit code $TRAIN_EXIT)."
    echo "Check log: $LOG"
    echo "============================================================"
fi

exit $TRAIN_EXIT
