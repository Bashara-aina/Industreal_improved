#!/usr/bin/env bash
# V5c: FRESH MULTI-TASK with STAGED_TRAINING=True (test F-1 Fix 2 properly)
# Runs on GPU 1 in parallel with V5b (which is on GPU 0).
# V5b is V5-continued (resumed from epoch 33) — does NOT test F-1 Fix 2
# because the env var didn't apply. V5c is the proper test.
# Wrapper fix (commit pending) now applies STAGED_TRAINING after preset,
# so the staging code at losses.py:1756-1775 will fire in stages 1-2.
set -uo pipefail

TRAIN_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"

echo "=============================================================="
echo " V5c FRESH MULTI-TASK — F-1 FIX 2 PROPER TEST (GPU 1)"
echo "=============================================================="
echo " Checkpoint: NONE (fresh from epoch 1)"
echo " Preset:     stage_rf4 (full multi-task: det + head_pose + act + psr)"
echo " GPU:        1 (RTX 5060 Ti) — runs in parallel with V5b on GPU 0"
echo ""
echo " DIFF FROM V5b (which is V5-continued):"
echo "   - Fresh start from epoch 1 (no --resume)"
echo "   - STAGED_TRAINING=True (wrapper now applies this after preset)"
echo "   - MAX_EPOCHS=30 (shorter; F-1 Fix 2 effect visible by epoch ~15-20)"
echo ""
echo " PURPOSE: Test F-1 Fix 2 (Kendall staging guard) properly."
echo " V5b can't test Fix 2 because V5b resumed from V5's checkpoint"
echo " (which was trained with STAGED_TRAINING=False)."
echo " V5c starts fresh with STAGED_TRAINING=True, so the staging code at"
echo " losses.py:1756-1775 will fire in stages 1-2, and the _kendall_fixed"
echo " guard will skip the PSR zero-out (proving F-1 Fix 2 works)."
echo ""
echo " Log: /tmp/train_v5c.log"
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
    --max-epochs 30 \
    >> /tmp/train_v5c.log 2>&1 &

PID=$!
echo "  PID=$PID"
echo "  Log: /tmp/train_v5c.log"
echo ""
echo "  EXPECTED: Fresh from epoch 1, STAGED_TRAINING=True (wrapper"
echo "            applies), stages 1-2 exercise F-1 Fix 2 guard."
echo "  PASS SIGNAL: LIVENESS shows PSR alive in stages 1-2 AND"
echo "               stage 3 (already verified by V4 LIVENESS probe)"
echo "  FAIL SIGNAL: PSR dead in stage 1 OR stage 2"
echo "=============================================================="
