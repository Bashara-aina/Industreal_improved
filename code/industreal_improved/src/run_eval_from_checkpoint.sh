#!/bin/bash
# =============================================================================
# Evaluate a checkpoint using evaluate.py standalone entry point
# Prints ALL paper metrics in the same format as evaluate.py's output.
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
CKPT_PATH="${1:-/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth}"
SPLIT="${2:-val}"
MAX_BATCHES="${3:-9999}"
SAVE_DIR="$PROJ_DIR/runs/eval_from_ckpt"

mkdir -p "$SAVE_DIR"

echo "============================================"
echo "Evaluating checkpoint"
echo "Checkpoint: $CKPT_PATH"
echo "Split: $SPLIT"
echo "Max batches: $MAX_BATCHES"
echo "Save dir: $SAVE_DIR"
echo "============================================"
echo ""

cd "$PROJ_DIR"

python -u evaluation/evaluate.py \
    --checkpoint "$CKPT_PATH" \
    --split "$SPLIT" \
    --max-batches "$MAX_BATCHES" \
    --save-dir "$SAVE_DIR" \
    2>&1 | tee "$SAVE_DIR/eval.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $SAVE_DIR/eval.log"
echo "============================================"
exit $EXIT_CODE
