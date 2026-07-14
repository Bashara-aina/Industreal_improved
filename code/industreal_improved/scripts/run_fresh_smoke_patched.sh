#!/bin/bash
# =============================================================================
# FRESH SMOKE — 8-patches applied, quick validity test
# Target: ~60 train steps + 15 val batches; prove all 4 heads produce
# non-degenerate metrics. ~25-40 min wall time on RTX 3060.
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="fresh_smoke_8patch_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ/src/runs/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

SRC_CKPT="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
LOG="$LOG_DIR/train.log"

# Kill any lingering GPU processes from prior runs
pkill -f "training/train.py" 2>/dev/null || true
pkill -f "eval_post_reinit" 2>/dev/null || true
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
export CUDA_LAUNCH_BLOCKING=0  # off for speed; we are stable in FP32 per diag_amp_nan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Smoke limits: 60 train steps, 15 val batches
export TRAIN_MAX_STEPS=60
export EVAL_MAX_BATCHES=15
export RUN_NAME="$RUN_NAME"

echo "============================================================"
echo "FRESH SMOKE — 8 patches applied, --reinit-heads, --no-amp"
echo "  Source ckpt: $SRC_CKPT"
echo "  Run dir    : $PROJ/src/runs/$RUN_NAME"
echo "  TRAIN_MAX_STEPS=$TRAIN_MAX_STEPS, EVAL_MAX_BATCHES=$EVAL_MAX_BATCHES"
echo "============================================================"

cd "$PROJ/src"
python -u training/train.py \
    --resume "$SRC_CKPT" \
    --reinit-heads \
    --no-staged-training \
    --no-amp \
    --max-epochs 42 \
    --subset-ratio 0.05 \
    --num-workers 0 \
    --batch-size 2 \
    --seed 42 \
    2>&1 | tee "$LOG"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: SUCCESS — fresh smoke training completed"
    echo "CKPT: $CKPT_DIR"
    ls -la "$CKPT_DIR"
else
    echo "RESULT: FAILED with exit code $EXIT"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
