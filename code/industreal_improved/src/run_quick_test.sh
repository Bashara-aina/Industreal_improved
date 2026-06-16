#!/bin/bash
# =============================================================================
# Quick 5% Dataset + 1 Epoch End-to-End Test
# Tests: train loop + validation, multi-stage DISABLED
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
RUN_NAME="quick_5pct_test"
LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "Quick 5% dataset + 1 epoch end-to-end test"
echo "============================================"
echo "Log dir: $LOG_DIR"
echo ""

cd "$PROJ_DIR"

python -u training/train.py \
    --no-staged-training \
    --max-epochs 1 \
    --subset-ratio 0.05 \
    --num-workers 0 \
    2>&1 | tee "$LOG_DIR/train.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS — train+val completed without error"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $LOG_DIR/train.log"
echo "============================================"
exit $EXIT_CODE