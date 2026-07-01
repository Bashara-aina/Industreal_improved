#!/bin/bash
BASE="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs"
STAGE_FILE="$BASE/rf_stage_state.json"
LOG_FILE="$BASE/rf3_monitor.log"

echo "[$(date)] rf3 monitor check" >> "$LOG_FILE"
if [ ! -f "$STAGE_FILE" ]; then
    echo "NO RUNNING — no stage state file found"
    exit 0
fi

STATE=$(cat "$STAGE_FILE")
STAGE=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('current_stage','unknown'))" 2>/dev/null)
STATUS=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
PID=$(echo "$STATE" | python3 -c "import sys,json; p=json.load(sys.stdin).get('training_pid'); print(p if p else 'null')" 2>/dev/null)
RETRY=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('retry_count',0))" 2>/dev/null)
EPOCH=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('epoch',0))" 2>/dev/null)
HB=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('last_heartbeat','?'))" 2>/dev/null)
BEST=$(echo "$STATE" | python3 -c "import sys,json; b=json.load(sys.stdin).get('best_metrics',{}); print(f'det_mAP50={b.get(\"det_mAP50\",\"?\")}')" 2>/dev/null)

if [ "$PID" != "null" ] && [ -n "$PID" ]; then
    if kill -0 "$PID" 2>/dev/null; then
        echo "RUNNING stage=$STAGE epoch=$EPOCH pid=$PID $BEST"
    else
        echo "STALE stage=$STAGE epoch=$EPOCH pid=$PID DEAD since=$HB $BEST"
    fi
elif [ "$STATUS" = "crashed" ] || [ "$RETRY" -ge 5 ]; then
    echo "CRASHED stage=$STAGE retry=$RETRY — needs intervention"
else
    echo "IDLE stage=$STAGE last_run=$HB $BEST"
fi
