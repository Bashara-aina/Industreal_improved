#!/usr/bin/env bash
# Launch R2.5 training with 2026-06-15 deep fixes:
# 1. evaluate.py: fix act_clip_ids → activity_mask filter (IndexError)
# 2. train.py: fix --reinit-heads log_var_pose reset
# 3. config.py: per-task Kendall bounds, activity weight, PSR warmup
# 4. losses.py: per-task log_var clamp, ACTIVITY_LOSS_WEIGHT, PSR step warmup
set -e
cd /media/newadmin/master/POPW/working/code/industreal_improved

LOG=runs/paper_run_r25_fix_20260615.log
CKPT=src/runs/full_multi_task_tma_tbank/checkpoints/best.pth

echo "[LAUNCH] $(date) — Starting R2.5 fix run" | tee -a "$LOG"
echo "[LAUNCH] Config: paper_run preset + --reinit-heads + per-task Kendall bounds" | tee -a "$LOG"
echo "[LAUNCH] Key fixes:" | tee -a "$LOG"
echo "  - ACTIVITY_HEAD_GRAD_CLIP=0.1 (was 0.5)" | tee -a "$LOG"
echo "  - ACTIVITY_LOSS_WEIGHT=0.3 (activity downweighted 70%)" | tee -a "$LOG"
echo "  - PSR log_var max=0 (can't be suppressed below precision 1.0)" | tee -a "$LOG"
echo "  - Pose log_var max=0 (can't be suppressed below precision 1.0)" | tee -a "$LOG"
echo "  - Activity log_var min=0 (can't precision-boost above prec 1.0)" | tee -a "$LOG"
echo "  - PSR step warmup: 3.0→1.0 over 3000 steps" | tee -a "$LOG"
echo "  - PSR_WEIGHT=30 (was 20)" | tee -a "$LOG"
echo "  - eval act_clip_ids filtered by activity_mask (fix IndexError)" | tee -a "$LOG"
echo "  - --reinit-heads now resets log_var_pose (was missing)" | tee -a "$LOG"

nohup python -u src/training/train.py \
  --preset paper_run \
  --resume "$CKPT" \
  --reinit-heads \
  >> "$LOG" 2>&1 &

PID=$!
echo "[LAUNCH] PID=$PID" | tee -a "$LOG"
echo "[LAUNCH] Log: $LOG" | tee -a "$LOG"
echo $PID > /tmp/r25_fix_pid.txt
