#!/bin/bash
# =============================================================================
# REINIT HEADS RETRAIN — 5% subset, 3 epochs — BF16 (FIX 2026-06-09)
# Diagnosed: AMP fp16 overflows the first backbone conv (720x1280 input),
# producing 83 Inf gradients. bf16 has the same speed on Ampere but no overflow.
# bf16 keeps all 386 grads finite; loss decreases 120->65 in 2 steps.
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="reinit_5pct_bf16_$(date +%Y%m%d_%H%M%S)"
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
export CUDA_LAUNCH_BLOCKING=0  # Disable sync for speed (debug off)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Limit steps for quick smoke — full run would do --max-epochs 36
export TRAIN_MAX_STEPS=200
export EVAL_MAX_BATCHES=20

echo "============================================================"
echo "REINIT-HEADS 5% retrain with BF16 (AMP fix)"
echo "  Source ckpt: $SRC_CKPT"
echo "  Log: $LOG"
echo "  RUN_NAME: $RUN_NAME"
echo "============================================================"

cd "$PROJ/src"

# Same args as run_reinit_5pct_3ep.sh + --bf16
python -u training/train.py \
    --resume "$SRC_CKPT" \
    --reinit-heads \
    --no-staged-training \
    --bf16 \
    --max-epochs 39 \
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
