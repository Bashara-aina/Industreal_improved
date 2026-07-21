#!/usr/bin/env bash
set -uo pipefail

TRAIN_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
CKPT="$TRAIN_DIR/src/runs/rf_stages/checkpoints/crash_recovery.pth"

echo "=============================================================="
echo " PSR V4 Repair - DENSE PSR LOSS (USE_PSR_TRANSITION=False)"
echo "=============================================================="
echo " Checkpoint: $CKPT"
echo " Preset:     ablation_psr_only (PSR single-task)"
echo " FIX:        USE_PSR_TRANSITION=False (dense per-frame focal loss)"
echo "             ablation_psr_only preset (PSR only, no interference)"
echo " LeakyReLU:  active (model.py repair)"
echo " GPU:        1 (RTX 3060)"
echo " Log:        /tmp/train_psr_v4.log"
echo ""

cd "$TRAIN_DIR"

export AMP_DTYPE=bf16
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1
export DETACH_PSR_FPN=False
export KENDALL_FIXED_WEIGHTS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_PSR_TRANSITION=False
export PSR_SEQ_EVERY_N_BATCHES=1

nohup python3 -u "$TRAIN_DIR/scripts/train_psr_repair_wrapper.py" \
    --preset ablation_psr_only \
    --batch-size 2 \
    --resume "$CKPT" \
    >> /tmp/train_psr_v4.log 2>&1 &

PID=$!
echo "  PID=$PID"
echo "  Log: /tmp/train_psr_v4.log"
echo "  USE_PSR_TRANSITION=False - PSR loss on EVERY batch (dense per-frame)"
echo "  PSR_SEQ_EVERY_N_BATCHES=1 - every batch is a sequence batch"
echo "  PSR single-task (no detection/activity/pose interference)"
echo "=============================================================="
