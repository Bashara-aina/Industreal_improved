#!/bin/bash
# =============================================================================
# Re-evaluate a checkpoint to get all metrics printed in the Val: line
# Uses --no-staged-training to match train.py's ema staged logic
# and loads a specific checkpoint for evaluation only.
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
CKPT_PATH="/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth"
RUN_NAME="reeval_from_ckpt"
LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "Re-evaluation from checkpoint"
echo "Checkpoint: $CKPT_PATH"
echo "Log dir: $LOG_DIR"
echo "============================================"
echo ""

cd "$PROJ_DIR"

# Run evaluation only (--eval-only or pass a checkpoint to eval)
# The train.py accepts --checkpoint path. If that fails,
# use the standalone evaluation script
python -u training/train.py \
    --no-staged-training \
    --max-epochs 1 \
    --subset-ratio 0.05 \
    --num-workers 0 \
    --checkpoint "$CKPT_PATH" \
    2>&1 | tee "$LOG_DIR/eval.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS — eval completed without error"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $LOG_DIR/eval.log"
echo "============================================"
exit $EXIT_CODE
