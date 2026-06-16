#!/bin/bash
# =============================================================================
# Full 100% dataset, 100 epochs, no staged training, eval every epoch
# Bug A (EMA) and Bug B (log_var) fixes applied from prior session
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
RUN_NAME="popw_full_100e"
LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "Starting full 100% dataset training run"
echo "Epochs: 100  Mode: no-staged-training"
echo "Subset: 100%  Eval: every epoch"
echo "Log dir: $LOG_DIR"
echo "============================================"
echo ""

cd "$PROJ_DIR"

PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u training/train.py \
    --max-epochs 100 \
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