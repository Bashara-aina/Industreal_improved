#!/usr/bin/env bash
# Restart RF2 (Refinement Stage 2) training.
# Kills existing training, resumes from crash_recovery.pth with stage_rf2 preset.
# Idempotent: safe to run multiple times.
set -e

TRAIN_DIR=/media/newadmin/master/POPW/working/code/industreal_improved
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Use system python that has torch — NOT swarm .venv (which lacks torch)
PYTHON=/home/newadmin/.local/bin/python

LOG="$TRAIN_DIR/src/runs/rf_stages/logs/train.log"
STATE="$TRAIN_DIR/src/runs/rf_stage_state.json"
CKPT="$TRAIN_DIR/src/runs/rf_stages/checkpoints/crash_recovery.pth"

echo "=============================================================="
echo " RF2 Training Restart — $TIMESTAMP"
echo "=============================================================="
echo " Checkpoint: $CKPT"
echo " Preset:     stage_rf2"
echo " Log:        $LOG"
echo ""

# ── Step 1: Kill existing training ──
echo "[RF2] Step 1/4: Killing any existing training process..."
bash "$TRAIN_DIR/scripts/kill_training.sh" 2>/dev/null || true
sleep 2

# ── Step 2: Verify checkpoint ──
echo ""
echo "[RF2] Step 2/4: Verifying checkpoint..."
if [ ! -f "$CKPT" ]; then
    echo "  ERROR: Checkpoint not found at $CKPT"
    echo "  Falling back to latest.pth..."
    CKPT="$TRAIN_DIR/src/runs/rf_stages/checkpoints/latest.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  ERROR: No checkpoint found. Cannot restart."
        exit 1
    fi
fi
CKPT_SIZE=$(stat -c%s "$CKPT" 2>/dev/null || echo "unknown")
echo "  Using: $CKPT ($((CKPT_SIZE / 1024 / 1024)) MB)"

# ── Step 3: Reset state.json ──
echo ""
echo "[RF2] Step 3/4: Resetting state.json..."
$PYTHON -c "
from datetime import datetime
import json

now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S+00:00')
hb = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

state = {
    'current_stage': 'rf2',
    'stage_index': 1,
    'status': 'training',
    'training_pid': None,
    'epoch': 11,
    'best_metric': 0.181,
    'best_metrics': {'det_mAP50': 0.181, 'forward_angular_MAE_deg': None},
    'gate_passed': False,
    'checklist_results': {
        'gate': {'passed': False, 'details': {}},
        'health': {'passed': False, 'details': {}},
        'convergence': {'passed': False, 'details': {}},
        'validation': {'passed': False, 'details': {}},
        'stability': {'passed': False, 'details': {}}
    },
    'metric_history': [
        {'epoch': 7, 'det_mAP50': 0.007},
        {'epoch': 8, 'det_mAP50': 0.184},
        {'epoch': 9, 'det_mAP50': 0.181},
        {'epoch': 10, 'det_mAP50': 0.159}
    ],
    'retry_count': 0,
    'current_strategy': 'default',
    'strategies_tried': [],
    'det_health_history': [],
    'stage_history': [{'stage': 'rf1', 'status': 'completed', 'best_det_mAP50': 0.45}],
    'issues_log': [],
    'last_check_time': now,
    'run_start_time': now,
    'log_cursor': 0,
    'cross_stage_memory': {},
    'max_epochs': 36,
    'last_heartbeat': hb
}
with open('$STATE', 'w') as f:
    json.dump(state, f, indent=2)
print(f'  Reset state.json')
"

# ── Step 4: Launch training ──
echo ""
echo "[RF2] Step 4/4: Launching training..."
echo "  Command: nohup python -u src/training/train.py"
echo "    --preset stage_rf2"
echo "    --resume $CKPT"
echo "    --subset-ratio 0.35"
echo ""

cd "$TRAIN_DIR"

nohup $PYTHON -u src/training/train.py \
    --preset stage_rf2 \
    --resume "$CKPT" \
    --subset-ratio 0.35 \
    >> "$LOG" 2>&1 &

PID=$!
echo "  PID=$PID"
echo "  Log: $LOG"

# Write PID to state.json and .training_pid
$PYTHON -c "
import json
with open('$STATE', 'r') as f:
    s = json.load(f)
s['training_pid'] = $PID
with open('$STATE', 'w') as f:
    json.dump(s, f, indent=2)
"
echo "$PID" > "$TRAIN_DIR/src/runs/rf_stages/.training_pid"

echo ""
echo "=============================================================="
echo " RF2 Training launched successfully."
echo " PID:     $PID"
echo " Watch:   tail -f $LOG"
echo " Swarm:   python -m rf2_swarm --oneshot"
echo " Kill:    bash scripts/kill_training.sh"
echo "=============================================================="
