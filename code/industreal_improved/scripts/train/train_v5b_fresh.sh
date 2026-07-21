#!/usr/bin/env bash
# V5b: FRESH MULTI-TASK from epoch 1 with STAGED_TRAINING=True
# Runs on GPU 1 in parallel with V5 (which is on GPU 0).
# Purpose: Test F-1 Fix 2 (Kendall staging guard under KENDALL_FIXED_WEIGHTS=1) properly.
# V5 has STAGED_TRAINING=False, so the staging code at losses.py:1756-1775 is never entered.
# V5b has STAGED_TRAINING=True, so the staging guard fires in stages 1-2 and is
# bypassed for PSR/head_pose when KENDALL_FIXED_WEIGHTS=1.
# This is the missing test for F-1 Fix 2.
set -uo pipefail

TRAIN_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"

echo "=============================================================="
echo " V5b FRESH MULTI-TASK — STAGED_TRAINING=True (test F-1 Fix 2)"
echo "=============================================================="
echo " Checkpoint: NONE (fresh from epoch 1)"
echo " Preset:     stage_rf4 (full multi-task: det + head_pose + act + psr)"
echo " GPU:        1 (RTX 5060 Ti) — runs in parallel with V5 on GPU 0"
echo ""
echo " DIFF FROM V5:"
echo "   - Fresh start (no resume from crash_recovery)"
echo "   - STAGED_TRAINING=True (test F-1 Fix 2: losses.py:1756-1775 guard)"
echo "   - STAGE1_EPOCHS=5, STAGE2_EPOCHS=10 (epoch 1-5 det-only, 6-15 det+head_pose+act, 16+ all)"
echo "   - MAX_EPOCHS=50 (same as V5)"
echo ""
echo " ALL OTHER FIXES SAME AS V5:"
echo "   - LeakyReLU + small-normal init + zero bias"
echo "   - GT-balanced sampler, DET_GAMMA_NEG=2.0"
echo "   - F-1 Fix 1: psr_head freeze bypass (KENDALL_FIXED_WEIGHTS=1)"
echo "   - DETACH_PSR_FPN=False, USE_PSR_TRANSITION=False"
echo "   - KENDALL_FIXED_WEIGHTS=1 (bypasses staging zero-out for PSR)"
echo "   - MIXED_PRECISION=True (bf16)"
echo ""
echo " Log: /tmp/train_v5b.log"
echo "=============================================================="

cd "$TRAIN_DIR"

export AMP_DTYPE=bf16
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1
export DETACH_PSR_FPN=False
export KENDALL_FIXED_WEIGHTS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_PSR_TRANSITION=False
export PSR_SEQ_EVERY_N_BATCHES=4
export STAGED_TRAINING=True

nohup python3 -u "$TRAIN_DIR/scripts/train_psr_repair_wrapper.py" \
    --preset stage_rf4 \
    --batch-size 2 \
    --max-epochs 50 \
    >> /tmp/train_v5b.log 2>&1 &

PID=$!
echo "  PID=$PID"
echo "  Log: /tmp/train_v5b.log"
echo ""
echo "  EXPECTED: Fresh from epoch 1, STAGED_TRAINING=True fires"
echo "            F-1 Fix 2 (Kendall staging guard) at losses.py:1756-1775"
echo "            Stages 1-2 (epochs 1-15) bypass PSR freeze when"
echo "            KENDALL_FIXED_WEIGHTS=1"
echo "  FAIL SIGNAL: 'psr=DEAD' on seq batches at any LIVENESS probe"
echo "  PASS SIGNAL: LIVENESS shows det/head_pose/act/psr all ALIVE"
echo "=============================================================="
