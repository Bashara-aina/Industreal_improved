#!/bin/bash
# =============================================================================
# POPW 1-Epoch Smoke Test (V2) — verify 3 patches (VMAE unfreeze, headpose_film position, ConvNeXt LR routing)
# =============================================================================
# V2: FIXED EPOCH_LOOP_EMPTY bug. Resume from crash_recovery.pth (saved epoch 21).
#   _train_start_epoch = saved_epoch + 1 = 22
#   --max-epochs 23 → train epoch 22 only (1 epoch). --max-epochs 22 would skip.
#
# Backgrounding pattern: nohup + </dev/null + disown (NOT `tee`).
#   - `tee` causes 8KB Python stdout buffering (per the comment in v3 log)
#   - `nohup` + `</dev/null` blocks SIGHUP from shell timeout
#   - `disown` detaches from job table so shell timeout doesn't propagate
#
# Smoke test acceptance criteria (verified via checkpoint diff + val metrics):
#   1. videomae_stream param rel > 0  (Patch #1: VMAE unfreeze)
#   2. headpose_film param rel > 0    (Patch #2: headpose_film before activity_proj)
#   3. psr_f1 > 0                     (was 0.0000 — collapse fix)
#   4. activity_macro_f1 > 0          (was 0.0002 — collapse fix)
#   5. val loss < 22.5056             (was 22.5056 — needs to decrease)
#
# Patch #3 (ConvNeXt LR routing) IS active in this smoke test: optimizer
# state does NOT load (param-group count mismatch → re-initialized), so the
# new name.startswith('backbone.') filter is applied on first param-group build.
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src"
RUN_NAME="full_multi_task_tma_tbank_benchmark"
CKPT="$PROJ_DIR/runs/$RUN_NAME/checkpoints/crash_recovery.pth"
LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/smoke_test_3patches_${TS}.log"
PID_FILE="$CKPT_DIR/smoke_test.pid"

echo "============================================"
echo "POPW 1-Epoch Smoke Test"
echo "  Patches:  #1 VMAE unfreeze, #2 headpose_film position, #3 ConvNeXt LR"
echo "  Resume :  $CKPT"
echo "  Log    :  $LOG"
echo "  PID    :  $PID_FILE"
echo "  Target :  --max-epochs 23 → runs epoch 22 (1 epoch from saved 21)"
echo "  Subset :  1.0 (full dataset)"
echo "  Staging:  OFF (--no-staged-training, matches parent)"
echo "============================================"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Launch in background with proper detachment
cd "$PROJ_DIR"
PYTHONUNBUFFERED=1 nohup python -u training/train.py \
    --resume "$CKPT" \
    --subset-ratio 1.0 \
    --max-epochs 23 \
    --seed 42 \
    --no-staged-training \
    --num-workers 0 \
    < /dev/null \
    > "$LOG" 2>&1 &

TRAIN_PID=$!
disown
echo "$TRAIN_PID" > "$PID_FILE"

echo "Training started: PID=$TRAIN_PID"
echo "Monitor with: tail -f $LOG"
echo "Stop with    : kill -TERM $TRAIN_PID"
echo "PID file     : $PID_FILE"
echo "============================================"
