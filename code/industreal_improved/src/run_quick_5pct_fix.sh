#!/bin/bash
# =============================================================================
# Quick 5% subset training run with --no-staged-training
# All 5 heads active from epoch 0, expanded 16-metric Val: line
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
RUN_NAME="quick_5pct_3e_fix"
LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "Starting quick training run"
echo "Subset: 5%  Epochs: 3  Mode: no-staged-training"
echo "Log dir: $LOG_DIR"
echo "============================================"
echo ""

cd "$PROJ_DIR"

# Use script trick for truly unbuffered output with tee
# PYTHONUNBUFFERED=1 + -u for unbuffered python
PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u training/train.py \
    --max-epochs 3 \
    --subset-ratio 0.05 \
    --no-staged-training \
    --num-workers 0 \
    2>&1 | tee "$LOG_DIR/train.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $LOG_DIR/train.log"
echo "============================================"
exit $EXIT_CODE