#!/bin/bash
# =============================================================================
# POPW Training Restart — 25% subset, train to C.EPOCHS=31 total
# v3: STAGED_TRAINING=True (config.py:375 flipped from False to True).
# Resumes from crash_recovery.pth (epoch=3 from clean SIGTERM).
#
# No --no-staged-training flag — staged training is the new default.
# Expected behavior: progress bar shows [stage=1] in epoch 1, [stage=2] at
# epoch 6, [stage=3] at epoch 16. PSR head gated until stage 3, then warmup
# ramp 0.33→0.67→1.0×base over STAGE3_WARMUP_EPOCHS=3.
# =============================================================================

set -e

PROJ_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/src"
CKPT_PATH="/media/newadmin/master/POPW/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
RUN_NAME="full_multi_task_tma_tbank_benchmark"

LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "POPW Training Restart v3 — 25% subset, STAGED_TRAINING=True"
echo "Checkpoint: $CKPT_PATH"
echo "Log dir: $LOG_DIR"
echo "Expected: [stage=1] in epoch 1, [stage=2] at epoch 6, [stage=3] at epoch 16"
echo "============================================"
echo ""

cd "$PROJ_DIR"

python training/train.py \
    --resume "$CKPT_PATH" \
    --subset-ratio 0.25 \
    --max-epochs 31 \
    --seed 42 \
    --num-workers 0 \
    2>&1 | tee "$LOG_DIR/restart_25pct_15ep_v3.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS — training completed"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $LOG_DIR/restart_25pct_15ep_v3.log"
echo "============================================"
exit $EXIT_CODE
