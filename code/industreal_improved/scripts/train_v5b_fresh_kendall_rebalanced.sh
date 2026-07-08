#!/usr/bin/env bash
V5b fresh multi-task with KENDALL_FIXED_WEIGHTS=0, fixes pose-overweight and classification-head collapse
# Resumes from V3 epoch-30 checkpoint (crash_recovery.pth).
# This is the load-bearing run for the paper — the only run that produces
# post-fix multi-task headline numbers for all 4 heads.
set -uo pipefail

TRAIN_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
CKPT="$TRAIN_DIR/src/runs/rf_stages/checkpoints/crash_recovery.pth"

echo "=============================================================="
echo " V5 MULTI-TASK REPAIR — ALL 9 FIXES (stage_rf4 + F-1 Fix 1+2)"
echo "=============================================================="
echo " Checkpoint: $CKPT (V3 epoch 30 state)"
echo " Preset:     stage_rf4 (full multi-task: det + head_pose + act + psr)"
echo ""
echo " FIXES (all in tree, verified by V4 LIVENESS probe):"
echo "   1. LeakyReLU + small-normal init + zero bias       (e618d929a)"
echo "   2. Sequential init index fix                       (6defe1f5f)"
echo "   3. Pose up-vector [6:9]                            (bff38b790)"
echo "   4. GT-balanced detection sampler                   (8cef56fc2)"
echo "   5. DET_GAMMA_NEG 1.5 -> 2.0                        (cd901f655)"
echo "   6. DETACH_PSR_FPN=False env-read                   (59f84c3d4)"
echo "   7. F-1 Fix 1: psr_head bypass under KENDALL_FIXED  (21ab3c3fd)"
echo "   8. F-1 Fix 2: Kendall staging guard                (08c55ae71)"
echo "   9. MIXED_PRECISION=True (bf16) + AMP_DTYPE=bf16    (wrapper)"
echo ""
echo " RUNTIME FLAGS (per V5 design):"
echo "   KENDALL_FIXED_WEIGHTS=0  # Let Kendall rebalance (NOT 1) -- fix the collapse on classification heads       (let Kendall rebalance -- classification heads were collapsing with fixed weights)"
echo "   USE_PSR_TRANSITION=False      (dense per-frame PSR loss on seq batches)"
echo "   DETACH_PSR_FPN=False          (PSR gradient flows to backbone)"
echo "   PSR_SEQ_EVERY_N_BATCHES=4     (every 4th batch is seq; 3/4 train det/pose/act)"
echo "   STAGED_TRAINING=False         (per stage_rf4 default)"
echo "   MAX_EPOCHS=50                 (cap from 99; early-stop patience=10 also active)"
echo ""
echo " GPU:        0 (RTX 3060) — freed from V4 at this moment"
echo " Log:        /tmp/train_v5.log"
echo "=============================================================="

cd "$TRAIN_DIR"

export AMP_DTYPE=bf16
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export DETACH_PSR_FPN=False
export KENDALL_FIXED_WEIGHTS=0  # Let Kendall rebalance (NOT 1) -- fix the collapse on classification heads
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_PSR_TRANSITION=False
export PSR_SEQ_EVERY_N_BATCHES=4

nohup python3 -u "$TRAIN_DIR/scripts/train_psr_repair_wrapper.py" \
    --preset stage_rf4 \
    --batch-size 2 \
    --max-epochs 50 \
   
    >> /tmp/train_v5.log 2>&1 &

PID=$!
echo "  PID=$PID"
echo "  Log: /tmp/train_v5.log"
echo ""
echo "  EXPECTED: all 4 heads ALIVE on first LIVENESS probe (step 500)"
echo "            det/act/head_pose/psr — multi-task gradient path open"
echo "  FAIL SIGNAL: any head reports DEAD[RMS=0.00] at step 500"
echo "=============================================================="
