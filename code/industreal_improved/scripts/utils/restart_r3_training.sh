#!/usr/bin/env bash
# Restart R2.5 training with all fixes applied.
# Idempotent: kills existing training, backs up the run, launches fresh.
set -e

cd /media/newadmin/master/POPW/working/code/industreal_improved

# ── Timestamp ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=src/runs
LOG="$LOG_DIR/paper_run_r25_${TIMESTAMP}.log"

# ── Config ──
PRESET=paper_run
CKPT=src/runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth
MONITOR_SCRIPT=/home/newadmin/swarm-bot/scripts/monitor_r25_training.sh

echo "=============================================================="
echo " R2.5 Training Restart — $TIMESTAMP"
echo "=============================================================="
echo " Preset:       $PRESET"
echo " Checkpoint:   $CKPT"
echo " --reinit-heads: YES"
echo " Log:          $LOG"
echo " Monitor:      $MONITOR_SCRIPT"

# ── Step 1: Kill existing training ──
echo ""
echo "[R2.5] Step 1/4: Killing any existing training process..."
bash scripts/kill_training.sh
sleep 2

# ── Step 2: Create logs directory if needed ──
mkdir -p "$LOG_DIR"

# ── Step 3: Backup previous run ──
echo ""
echo "[R2.5] Step 2/4: Backing up previous run data..."
BACKUP_DIR="backups/paper_run_r25"
mkdir -p "$BACKUP_DIR"
for d in src/runs/paper_run_r25*; do
    if [ -d "$d" ] && [ "$d" != "src/runs/paper_run_r25_${TIMESTAMP}" ]; then
        echo "  Backing up $d → $BACKUP_DIR/"
        cp -a "$d" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done
echo "  Backup complete."

# ── Step 4: Launch new training ──
echo ""
echo "[R2.5] Step 3/4: Launching training..."
echo "  Command: nohup python -u src/training/train.py"
echo "    --preset $PRESET"
echo "    --resume $CKPT"
echo "    --reinit-heads"
echo ""

nohup python -u src/training/train.py \
    --preset "$PRESET" \
    --resume "$CKPT" \
    --reinit-heads \
    >> "$LOG" 2>&1 &

PID=$!
echo "  PID=$PID"
echo "  Log: $LOG"
echo $PID > /tmp/r25_training_pid.txt

# ── Step 5: Launch monitor (tmux if available, otherwise nohup) ──
echo ""
echo "[R2.5] Step 4/4: Starting monitor..."
if [ -f "$MONITOR_SCRIPT" ]; then
    if command -v tmux &>/dev/null; then
        # tmux available — use persistent session
        if tmux has-session -t r25_monitor 2>/dev/null; then
            echo "  Monitor tmux session 'r25_monitor' already exists — killing old session."
            tmux kill-session -t r25_monitor 2>/dev/null || true
            sleep 1
        fi
        tmux new-session -d -s r25_monitor
        tmux send-keys -t r25_monitor "while true; do clear && bash $MONITOR_SCRIPT && sleep 30; done" Enter
        echo "  Monitor launched in tmux session: r25_monitor"
        echo "  Attach with: tmux attach -t r25_monitor"
        echo "  Refreshes every 30 seconds."
    else
        # tmux not available — fall back to background nohup
        echo "  tmux not available — launching monitor as background process instead."
        nohup bash -c "while true; do clear && bash \"$MONITOR_SCRIPT\" && sleep 30; done" > /dev/null 2>&1 &
        MONITOR_PID=$!
        echo "  Monitor launched as background process (PID=$MONITOR_PID)"
        echo "  Refreshes every 30 seconds."
        echo "$MONITOR_PID" > /tmp/r25_monitor_pid.txt
    fi
else
    echo "  WARNING: Monitor script not found at $MONITOR_SCRIPT — skipping."
fi

echo ""
echo "=============================================================="
echo " R2.5 Training launched successfully."
echo " PID:     $PID"
echo " Log:     $LOG"
echo " Kill:    bash scripts/kill_training.sh"
echo "=============================================================="
