#!/bin/bash
# =============================================================================
# POPW Training Restart v2 — 25% subset, train to C.EPOCHS=31 total
# Resumes from crash_recovery.pth (currently epoch=3) → epoch 31 = 28 more epochs.
# NOTE: --max-epochs sets TOTAL epoch count (train.py:3184 C.EPOCHS=...), not
# an add-on. Previous comment said "15 epochs" which was wrong.
#
# *** DIFFERS FROM v1 ***: dropped --no-staged-training.
#
# Why: The v1 script listed "PSR stage-3 warmstart ramp (losses.py:1101)" as a
# structural fix but then ran with --no-staged-training. That contradiction
# was a real bug, not a documentation issue. With --no-staged-training set,
# train.py:3201 logs "STAGED_TRAINING=False — all 5 heads active from epoch 0",
# which means the PSR head was thrown in at full LR from epoch 0 with no
# warmup protection. Result: classic all-ones collapse on the imbalanced
# PSR task. v1 log preserved at restart_25pct_15ep.log for forensics.
#
# v2 uses default STAGED_TRAINING=True, so heads are gated and the
# STAGE3_WARMUP_EPOCHS=3 ramp (0.33x → 0.67x → 1.0x base LR on
# activity_head + psr_head) protects the PSR head from collapse when it
# first activates.
#
# All structural fixes applied:
#   1. Kendall clamp removed (losses.py:1264)
#   2. DET cls bias pi=0.10 (model.py:526)
#   3. DET pos_iou_thresh=0.3 via C.DET_POS_IOU_THRESH (losses.py:862)
#   4. CB_BETA=0.99 (config.py:343)
#   5. 50x weight ratio cap in ClassBalancedFocalLoss (losses.py after forward)
#   6. PSR stage-3 warmstart ramp (losses.py:1101)  ← NOW ACTIVE (v2 only)
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
CKPT_PATH="/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
RUN_NAME="full_multi_task_tma_tbank_benchmark"

LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "POPW Training Restart v2 — 25% subset, 28 epochs"
echo "Fix: dropped --no-staged-training so stage-3 warmup ramp is active"
echo "Checkpoint: $CKPT_PATH"
echo "Log dir: $LOG_DIR"
echo "============================================"
echo ""

cd "$PROJ_DIR"

python training/train.py \
    --resume "$CKPT_PATH" \
    --subset-ratio 0.25 \
    --max-epochs 31 \
    --seed 42 \
    --num-workers 0 \
    2>&1 | tee "$LOG_DIR/restart_25pct_15ep_v2.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: SUCCESS — training completed"
else
    echo "RESULT: FAILED with exit code $EXIT_CODE"
fi
echo "Log: $LOG_DIR/restart_25pct_15ep_v2.log"
echo "============================================"
exit $EXIT_CODE
