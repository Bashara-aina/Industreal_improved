#!/usr/bin/env bash
# Kill any running IndustReal training process gracefully.
# Uses SIGTERM first, then SIGKILL after timeout.
set -e

TRAIN_DIR=/media/newadmin/master/POPW/working/code/industreal_improved

# Auto-detect the latest training log
LOG=$(find "$TRAIN_DIR/src/runs" -name 'train.log' -path '*/logs/*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

echo "[KILL] $(date) — Looking for training processes..."

# Find all training PIDs (exclude grep, exclude this script)
PIDS=$(ps aux | grep 'python.*train.py' | grep -v grep | awk '{print $2}' | grep -v "^${$}$" || true)

if [ -z "$PIDS" ]; then
    echo "[KILL] No training processes found."
    exit 0
fi

echo "[KILL] Found PIDs: $(echo $PIDS | tr '\n' ' ')"
echo "[KILL] Sending SIGTERM to all..."

for PID in $PIDS; do
    kill -TERM "$PID" 2>/dev/null || true
done

# Wait up to 30 seconds for graceful shutdown
echo "[KILL] Waiting 30 seconds for graceful shutdown..."
for i in $(seq 1 30); do
    PIDS_REMAINING=$(ps aux | grep 'python.*train.py' | grep -v grep | awk '{print $2}' | grep -v "^${$}$" | tr '\n' ' ' || true)
    if [ -z "$PIDS_REMAINING" ]; then
        echo "[KILL] All processes terminated gracefully after ${i}s."
        break
    fi
    sleep 1
done

# Force kill any remaining
PIDS_REMAINING=$(ps aux | grep 'python.*train.py' | grep -v grep | awk '{print $2}' | grep -v "^${$}$" || true)
if [ -n "$PIDS_REMAINING" ]; then
    echo "[KILL] Sending SIGKILL to remaining PIDs: $(echo $PIDS_REMAINING | tr '\n' ' ')"
    for PID in $PIDS_REMAINING; do
        kill -KILL "$PID" 2>/dev/null || true
    done
    echo "[KILL] Force killed remaining processes."
fi

# Report last 20 lines of training log
if [ -n "$LOG" ]; then
    echo ""
    echo "[KILL] Last 20 lines of training log: $LOG"
    echo "========================================"
    tail -20 "$LOG"
    echo "========================================"
else
    echo "[KILL] No training log found."
fi

echo "[KILL] $(date) — Done."
