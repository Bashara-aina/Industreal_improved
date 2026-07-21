#!/bin/bash
# =============================================================================
# SMOKE TEST 100 steps — verify no NaN cascade past step 32
# Previous 20-step smoke was inconclusive. Need to cross step 30 where the
# seq-mode batch first appeared in earlier runs.
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="smoke_fp32_100_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ/src/runs/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

SRC_CKPT="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
LOG="$LOG_DIR/train.log"

pkill -f "training/train.py" 2>/dev/null || true
sleep 2
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 3

export PYTHONPATH="$PROJ:$PROJ/src:$PROJ/src/models:$PROJ/src/training:$PROJ/src/evaluation:$PROJ/src/data:$PROJ/src/utils"
export SUBSET_RATIO=0.05
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1  # SYNC for diagnostic clarity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MAX_STEPS=100
export EVAL_MAX_BATCHES=10

echo "============================================================"
echo "SMOKE TEST 100 STEPS — FP32 + improved reinit biases"
echo "  Source ckpt: $SRC_CKPT"
echo "  Log: $LOG"
echo "  RUN_NAME: $RUN_NAME"
echo "============================================================"

cd "$PROJ/src"
python -u training/train.py \
    --resume "$SRC_CKPT" \
    --reinit-heads \
    --no-staged-training \
    --no-amp \
    --max-epochs 49 \
    --subset-ratio 0.05 \
    --num-workers 0 \
    --batch-size 4 \
    --seed 42 \
    2>&1 | tee "$LOG"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: SUCCESS — 100-step FP32 smoke completed"
else
    echo "RESULT: FAILED with exit code $EXIT"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
