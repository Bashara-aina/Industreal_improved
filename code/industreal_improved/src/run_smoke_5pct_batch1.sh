#!/bin/bash
# =============================================================================
# Smoke test: 5% subset, 1 epoch, batch_size=1, all 5 heads from start
# Target: /media/newadmin/master/ -- the runtime copy
# Reason: Previous smoke_5pct_perhead OOM'd at ConvNeXt stage2 with batch=2
# Fix:   BATCH_SIZE=1 in config.py + --batch-size 1 override
# =============================================================================

set -e

PROJ_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/src"
RUN_NAME="smoke_5pct_perhead_batch1"
LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "Starting smoke test — BATCH_SIZE=1"
echo "Subset: 5%  Epochs: 1  Mode: no-staged-training"
echo "Log dir: $LOG_DIR"
echo "============================================"
echo ""

cd "$PROJ_DIR"

# Explicit --batch-size 1 ensures the OOM fix is honored
PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u training/train.py \
    --max-epochs 1 \
    --subset-ratio 0.05 \
    --no-staged-training \
    --num-workers 0 \
    --batch-size 1 \
    2>&1 | tee "$LOG_DIR/train.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS — smoke test completed without OOM"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $LOG_DIR/train.log"
echo "============================================"
exit $EXIT_CODE
