#!/bin/bash
# =============================================================================
# REINIT HEADS RETRAIN — 5% subset, 2-epoch, batch=2 (FIX 2026-06-10)
# Batch=4 OOMs in epoch 43 at the seq-mode PSR batch (T=4 window, 16 effective
# frames). Batch=2 halves the activation memory and is proven stable by the
# 60-step smoke test. Resumes from the epoch-42 checkpoint.
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="reinit_5pct_fp32_bs2_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ/src/runs/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Use the latest best.pth (epoch 42 success, 18:56 timestamp)
SRC_CKPT="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/best.pth"
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
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Full run: no TRAIN_MAX_STEPS limit; resume epoch 42, finish at epoch 44
export TRAIN_MAX_STEPS=0
export EVAL_MAX_BATCHES=20

echo "============================================================"
echo "REINIT-HEADS 5% retrain — batch=2 (smoke-test-proven config)"
echo "  Source ckpt: $SRC_CKPT"
echo "  Log: $LOG"
echo "  RUN_NAME: $RUN_NAME"
echo "============================================================"

cd "$PROJ/src"

# batch=2 in FP32 fits the seq-mode PSR batch (T=4) within 11.4 GB
python -u training/train.py \
    --resume "$SRC_CKPT" \
    --reinit-heads \
    --no-staged-training \
    --no-amp \
    --max-epochs 44 \
    --subset-ratio 0.05 \
    --num-workers 0 \
    --batch-size 2 \
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
