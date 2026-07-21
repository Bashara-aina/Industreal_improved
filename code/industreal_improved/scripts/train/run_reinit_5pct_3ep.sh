#!/bin/bash
# =============================================================================
# REINIT HEADS RETRAIN — 5% subset, 3 epochs
# Resumes from crash_recovery.pth and re-initializes 3 dead heads (det/act/psr)
# while preserving the alive backbone + pose head.
#
# Hypothesis: backbone features are alive (verified via head re-init diagnostic
# showing per-image variance 0.032-0.036 in DET logits). Re-initializing the
# 3 heads from priors should let them re-learn from the alive backbone.
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="reinit_5pct_3ep_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ/src/runs/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

SRC_CKPT="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
LOG="$LOG_DIR/train.log"

# Kill any lingering GPU processes from prior runs
pkill -f "training/train.py" 2>/dev/null || true
sleep 2
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 3

# PYTHONPATH
export PYTHONPATH="$PROJ:$PROJ/src:$PROJ/src/models:$PROJ/src/training:$PROJ/src/evaluation:$PROJ/src/data:$PROJ/src/utils"
export SUBSET_RATIO=0.05
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "REINIT-HEADS 5% retrain"
echo "  Source ckpt: $SRC_CKPT"
echo "  Log: $LOG"
echo "  RUN_NAME: $RUN_NAME"
echo "============================================================"

cd "$PROJ/src"

# Run training: resume from crash_recovery (epoch 33) + --reinit-heads + continue 3 more epochs.
# --max-epochs=36 (not 3) because crash_recovery.pth has epoch=33 → start_epoch=33,
# and the epoch loop checks start_epoch < C.EPOCHS. If max-epochs=33, the loop is
# skipped entirely (EPOCH_LOOP_EMPTY). max-epochs=36 gives epochs 33,34,35.
python -u training/train.py \
    --resume "$SRC_CKPT" \
    --reinit-heads \
    --no-staged-training \
    --max-epochs 36 \
    --subset-ratio 0.05 \
    --num-workers 0 \
    --batch-size 8 \
    --seed 42 \
    2>&1 | tee "$LOG"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: SUCCESS — reinit retrain completed"
else
    echo "RESULT: FAILED with exit code $EXIT"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
