#!/usr/bin/env bash
# PSR Head Repair Training — real LeakyReLU + small-normal init + zero bias.
# Warm-start from crash_recovery.pth, multi-task (stage_rf4), bf16 mixed precision.
set -e

TRAIN_DIR=/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
CKPT="$TRAIN_DIR/src/runs/rf_stages/checkpoints/crash_recovery.pth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG=/tmp/train_psr_repair.log
PYTHON=/home/newadmin/.local/bin/python
WRAPPER="$TRAIN_DIR/scripts/train_psr_repair_wrapper.py"

echo "=============================================================="
echo " PSR Head Repair Training Launch — $TIMESTAMP"
echo "=============================================================="
echo " Checkpoint: $CKPT"
echo " Preset:     stage_rf4 (multi-task, all heads)"
echo " Batch:      2 (CUDA timeout mitigation)"
echo " Precision:  bf16 mixed (AMP_DTYPE=bf16)"
echo " GPU:        CUDA_VISIBLE_DEVICES=1"
echo " Log:        $LOG"
echo ""

# ── Step 1: Verify checkpoint ──
echo "[Step 1/3] Verifying checkpoint..."
if [ ! -f "$CKPT" ]; then
    echo "  ERROR: Checkpoint not found at $CKPT"
    exit 1
fi
CKPT_SIZE=$(stat -c%s "$CKPT" 2>/dev/null || echo "unknown")
echo "  OK: $CKPT ($((CKPT_SIZE / 1024 / 1024)) MB)"

# ── Step 2: Verify model.py has LeakyReLU repair ──
echo "[Step 2/3] Verifying model.py PSR head repair..."
if grep -q "LeakyReLU" "$TRAIN_DIR/src/models/model.py"; then
    echo "  OK: LeakyReLU found in model.py"
else
    echo "  WARNING: LeakyReLU NOT found — continuing anyway"
fi

# ── Step 3: Launch training ──
echo "[Step 3/3] Launching training..."
echo ""

cd "$TRAIN_DIR"

export AMP_DTYPE=bf16
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

nohup $PYTHON -u "$WRAPPER" \
    --preset stage_rf4 \
    --batch-size 2 \
    --resume "$CKPT" \
    >> "$LOG" 2>&1 &

PID=$!
echo "  PID=$PID"
echo "  Log: $LOG"
echo ""

echo "=============================================================="
echo " PSR Head Repair Training launched. PID=$PID"
echo " Monitor:  tail -f $LOG"
echo " Kill:     kill $PID"
echo "=============================================================="
