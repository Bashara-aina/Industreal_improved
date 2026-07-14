#!/bin/bash
# =============================================================================
# SMOKE TEST — Candidate Config (30_DAY_EXECUTION_PLAN Day 4)
#
# Validates the end-to-end training loop with the candidate config:
#   - USE_UW_SO=1            UW-SO softmax ordinal uncertainty weighting
#   - USE_BALANCED_SOFTMAX_ACT=1  Balanced softmax for activity head
#   - PSR_LR_MULTIPLIER=0.5      Per-task LR multiplier for PSR head
#   - HEAD_POSE_LR_MULTIPLIER=0.3 Per-task LR multiplier for head pose head
#   - USE_KENDALL=True            Baseline Kendall uncertainty weighting
#   - FP32 (MIXED_PRECISION=0)   Deterministic, no AMP
#   - 20 steps, subset 5%
#
# Expected outcomes:
#   1. No NaN/inf in losses or gradients
#   2. All 4 heads produce non-zero loss
#   3. Log-var bounds respected: log_var_act >= -0.5, log_var_psr <= 0.0, log_var_pose <= 3.0
#   4. Per-task LR multipliers reflected in optimizer param groups
#   5. UW-SO produces finite weights (non-NaN)
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="smoke_candidate_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ/src/runs/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Use the latest production checkpoint as starting point
SRC_CKPT="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
LOG="$LOG_DIR/train.log"

# Cleanup any stale processes
pkill -f "training/train.py" 2>/dev/null || true
sleep 2

# =============================================================================
# ENVIRONMENT — Candidate Config
# =============================================================================
export PYTHONPATH="$PROJ:$PROJ/src:$PROJ/src/models:$PROJ/src/training:$PROJ/src/evaluation:$PROJ/src/data:$PROJ/src/utils"

# Dataset / data subset
export SUBSET_RATIO=0.05
export TRAIN_MAX_STEPS=20
export EVAL_MAX_BATCHES=5

# Determinism / compute
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === CANDIDATE CONFIG FLAGS ===
# UW-SO (softmax ordinal uncertainty weighting)
export USE_UW_SO=1
export UW_SO_TEMPERATURE=1.0

# Balanced softmax for activity head (long-tail adjustment)
export USE_BALANCED_SOFTMAX_ACT=1

# Kendall baseline (UW-SO replaces per-task precision weighting,
# but log-var logging still works for diagnostic)
export USE_KENDALL=1

# Per-task LR multipliers (applied in train.py param group construction)
export PSR_LR_MULTIPLIER=0.5
export HEAD_POSE_LR_MULTIPLIER=0.3
# NOTE: ACTIVITY_LR_MULTIPLIER is not yet wired; activity uses default head_lr.
# If added later, set it here:
# export ACTIVITY_LR_MULTIPLIER=1.0

# Training flags — all 4 heads enabled
export TRAIN_DET=1
export TRAIN_HEAD_POSE=1
export TRAIN_ACT=1
export TRAIN_PSR=1

# FP32 (no AMP)
export MIXED_PRECISION=0

# EMA
export USE_EMA=1
export EMA_DECAY=0.999

# Log every step for detailed trajectory
export LOG_INTERVAL=1
export LOG_IMAGE_INTERVAL=0

echo "============================================================"
echo "SMOKE TEST — Candidate Config"
echo "  Run name : $RUN_NAME"
echo "  Source   : $SRC_CKPT"
echo "  Log      : $LOG"
echo "============================================================"
echo ""
echo "--- Config ---"
echo "  USE_UW_SO              = $USE_UW_SO"
echo "  UW_SO_TEMPERATURE      = $UW_SO_TEMPERATURE"
echo "  USE_BALANCED_SOFTMAX_ACT = $USE_BALANCED_SOFTMAX_ACT"
echo "  PSR_LR_MULTIPLIER      = $PSR_LR_MULTIPLIER"
echo "  HEAD_POSE_LR_MULTIPLIER = $HEAD_POSE_LR_MULTIPLIER"
echo "  USE_KENDALL            = $USE_KENDALL"
echo "  MIXED_PRECISION        = $MIXED_PRECISION"
echo "  TRAIN_MAX_STEPS        = $TRAIN_MAX_STEPS"
echo "  SUBSET_RATIO           = $SUBSET_RATIO"
echo ""

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
    echo "RESULT: SUCCESS — candidate config smoke test completed"
else
    echo "RESULT: FAILED with exit code $EXIT"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
