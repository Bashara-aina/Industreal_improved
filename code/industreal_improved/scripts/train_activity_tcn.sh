#!/usr/bin/env bash
# Train ActivityTCN (Opus 141 ACT-ARCH-4 Phase 1).
# Frozen ConvNeXt features -> TCN with dilations [1,2,4] -> 69-class classifier
set -uo pipefail

TRAIN_DIR=/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$TRAIN_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ActivityTCN training..."
echo "  GPU:        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (RTX 5060 Ti)"
echo "  Clip len:   16 frames"
echo "  Stride:     8 frames"
echo "  Epochs:     30"
echo "  Log:        /tmp/train_activity_tcn.log"

python3 -u -m src.training.train_activity_tcn \
    --clip-len 16 \
    --stride 8 \
    --batch-size 64 \
    --epochs 30 \
    --lr 1e-3 \
    --hidden 256 \
    --levels 3 \
    --save-dir src/runs/rf_stages/checkpoints/activity_tcn \
    >> /tmp/train_activity_tcn.log 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ActivityTCN training exited with code ${EXIT_CODE}"
exit ${EXIT_CODE}