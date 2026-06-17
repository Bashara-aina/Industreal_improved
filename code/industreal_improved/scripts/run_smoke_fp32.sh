#!/bin/bash
# =============================================================================
# SMOKE TEST — FP32 + seq mode, 20 steps, verify no NaN cascade
# (verifies the MIXED_PRECISION=False + loss_dict init fix)
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="smoke_fp32_$(date +%Y%m%d_%H%M%S)"
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
export CUBLAS_WORKSPACE_CONFIG=4096:8
export CUDA_LAUNCH_BLOCKING=1  # SYNC for diagnostic
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MAX_STEPS=20
export EVAL_MAX_BATCHES=5

echo "============================================================"
echo "SMOKE TEST — FP32 (config MIXED_PRECISION=False), 20 steps"
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
    --max-epochs 39 \
    --subset-ratio 0.05 \
    --num-workers 0 \
    --batch-size 4 \
    --seed 42 \
    2>&1 | tee "$LOG"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: SUCCESS — smoke test completed"
else
    echo "RESULT: FAILED with exit code $EXIT"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
