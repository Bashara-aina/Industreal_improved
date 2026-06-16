#!/bin/bash
# =============================================================================
# POPW Training Restart v4 — 20% subset, STAGED_TRAINING=False (all heads from epoch 0)
# Fresh start, NO --resume. Abandons v3 (25% staged, stage 2 at epoch 10).
#
# Expected behavior: progress bar shows [no-staging] from epoch 0. All 5 heads
# (DET, POSE, ACT, PSR, HEAD_POSE) active immediately. PSR head no longer gated.
# This is the v1/v2 "broken" regime, deliberately retried for comparison.
# =============================================================================

set -e

PROJ_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/src"
RUN_NAME="full_multi_task_tma_tbank_v4_20pct_unstaged"

LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "POPW Training Restart v4 — 20% subset, STAGED_TRAINING=False"
echo "Run name: $RUN_NAME"
echo "Log dir: $LOG_DIR"
echo "Ckpt dir: $CKPT_DIR"
echo "Expected: [no-staging] from epoch 0, all 5 heads active"
echo "============================================"
echo ""

cd "$PROJ_DIR"

python training/train.py \
    --subset-ratio 0.20 \
    --max-epochs 31 \
    --seed 42 \
    --num-workers 0 \
    --no-staged-training \
    2>&1 | tee "$LOG_DIR/restart_20pct_unstaged_v4.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS — training completed"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $LOG_DIR/restart_20pct_unstaged_v4.log"
echo "============================================"
exit $EXIT_CODE
