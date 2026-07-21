#!/bin/bash
# Monitor MTL + ST pose training and report health
LOG_DIR="runs"
TASKS=("aa_main_mtl" "aa_st_pose")
while true; do
  clear
  echo "=== Training Status @ $(date) ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv | tail -n +2
  echo ""
  for t in "${TASKS[@]}"; do
    if [ -f "$LOG_DIR/$t/train.log" ]; then
      echo "--- $t ---"
      tail -2 "$LOG_DIR/$t/train.log" | grep -oE "Epoch [0-9]+ batch [0-9]+/[0-9]+|elapsed=[0-9]+s.*speed=[0-9.]+" | tail -2
      echo ""
    fi
  done
  sleep 30
done
