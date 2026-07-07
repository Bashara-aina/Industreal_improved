#!/usr/bin/env bash
# PSR Head Repair Training V3 — WITH GRADIENT FLOW (DETACH_PSR_FPN=False)
# Fix: stage_rf4 default has DETACH_PSR_FPN=True which blocks gradient flow
# This is the critical fix for the PSR gradient DEAD issue
set -uo pipefail

TRAIN_DIR=/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
CKPT="$TRAIN_DIR/src/runs/rf_stages/checkpoints/crash_recovery.pth"
LOG=/tmp/train_psr_repair_v3.log

echo "=============================================================="
echo " PSR Head Repair V3 — WITH GRADIENT FLOW"
echo "=============================================================="
echo " Checkpoint: $CKPT"
echo " Preset:     stage_rf4 (multi-task)"
echo " FIX:        DETACH_PSR_FPN=False (was True)"
echo " Preserves:  LeakyReLU + small-normal init + zero bias repair"
echo " GPU:        1 (RTX 3060)"
echo " Log:        $LOG"
echo ""

cd "$TRAIN_DIR"
echo "Starting at $(date)..."

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DETACH_PSR_FPN=False \
KENDALL_FIXED_WEIGHTS=1 \
python3 -u -m src.training.train_psr_repair_wrapper \
    --preset stage_rf4 \
    --batch-size 2 \
    --resume "$CKPT" \
    --epochs 5 \
    --log "$LOG"

EXIT_CODE=$?
echo "Training exited with code ${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
