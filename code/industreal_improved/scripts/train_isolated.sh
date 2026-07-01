#!/usr/bin/env bash
# ==============================================================================
# train_isolated.sh — Crash-Hardened Training Launcher
# ==============================================================================
# Wraps the training script in a tmux session so it survives VS Code terminal
# crashes (Electron bug in Code 1.109.5, observed SIGILL/SIGTRAP crashes).
# When VS Code dies, SIGHUP is sent to its terminal children — but tmux
# intercepts SIGHUP and keeps the session alive.
#
# Usage:
#   ./scripts/train_isolated.sh [--attach] [extra train.py args...]
#
# Options:
#   --attach    Attach to the tmux session immediately
#   --tail      Follow the training log in this terminal (don't attach to tmux)
#
# The training's PID is written to /tmp/train_pid for external monitoring.
# Logs go to the normal training log location under src/runs/.
#
# Kill the training:
#   kill "$(cat /tmp/train_pid)"          # graceful SIGTERM
#   ./scripts/kill_training.sh             # kill all training processes
#   tmux kill-session -t industreal-train  # kill the tmux session entirely
#
# Reattach to monitor:
#   tmux attach -t industreal-train
# ==============================================================================
set -euo pipefail

SESSION_NAME="industreal-train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/src/training/train.py"

# Parse flags
ATTACH=0
TAIL_MODE=0
TRAIN_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --attach) ATTACH=1 ;;
        --tail) TAIL_MODE=1 ;;
        *) TRAIN_ARGS+=("$arg") ;;
    esac
done

# Check if training is already running in tmux
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[launch] Training session '$SESSION_NAME' already exists."
    echo "  Attach:  tmux attach -t $SESSION_NAME"
    echo "  Kill:    tmux kill-session -t $SESSION_NAME"
    if [ "$ATTACH" = 1 ]; then
        tmux attach -t "$SESSION_NAME"
    fi
    exit 0
fi

# Check that the training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "[launch] ERROR: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

# Build the command
CMD="cd '$SCRIPT_DIR' && python3 src/training/train.py ${TRAIN_ARGS[*]}"
PIDFILE="/tmp/train_pid"

# Create the tmux session
echo "[launch] Creating tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" -n train bash -c "
    # Write PID for external monitoring
    echo \$\$ > '$PIDFILE'
    echo \"[launch] Training PID: \$\$\"
    echo \"[launch] Started: \$(date)\"
    echo \"[launch] Command: python3 src/training/train.py ${TRAIN_ARGS[*]}\"
    echo \"[launch] SIGHUP-protected by tmux. To kill: tmux kill-session -t $SESSION_NAME\"
    echo \"\"
    $CMD
    EXIT_CODE=\$?
    echo \"[launch] Training exited with code \$EXIT_CODE at \$(date)\"
    echo \"[launch] PID file: $PIDFILE (remove if restarting)\"
    rm -f '$PIDFILE'
    sleep 5
"

# Wait a moment and check if the session is alive
sleep 1
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[launch] Session '$SESSION_NAME' created successfully."
    echo "  PID: $(cat "$PIDFILE" 2>/dev/null || echo 'unknown')"
    echo "  Log:  tail -f $SCRIPT_DIR/src/runs/rf_stages/logs/train.log"
    echo "  Reattach: tmux attach -t $SESSION_NAME"
    if [ "$TAIL_MODE" = 1 ]; then
        echo "[launch] Following training log (Ctrl+C to stop following)..."
        tail -f "$SCRIPT_DIR/src/runs/rf_stages/logs/"/train.log 2>/dev/null || \
        echo "[launch] Log file not yet created. Waiting 10s..." && \
        sleep 10 && \
        tail -f "$(find "$SCRIPT_DIR/src/runs" -name 'train.log' -path '*/logs/*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
    fi
    if [ "$ATTACH" = 1 ]; then
        tmux attach -t "$SESSION_NAME"
    fi
else
    echo "[launch] ERROR: tmux session failed to start."
    echo "  Is tmux installed? (apt install tmux)"
    exit 1
fi
