#!/usr/bin/env bash
# launch_uncapped_kendall.sh — Relax Kendall log_var bounds + disable prec cap.
#
# Relaxes Kendall learned-weight bounds so the optimizer can explore a wider
# range of log_var values, and disables KENDALL_HP_PREC_CAP so pose log_var is
# never artificially clamped to detection log_var.
#
# Usage:
#   bash scripts/launch_uncapped_kendall.sh
#
# Environment overrides (set here):
#   KENDALL_LOG_VAR_MIN_ACT   -4.0  (default -0.5)
#   KENDALL_LOG_VAR_MAX_PSR    2.0  (default  0.0)
#   KENDALL_LOG_VAR_MAX_POSE   4.0  (default  3.0)
#   KENDALL_HP_PREC_CAP     False  (default True)
#
# Output: logs + checkpoints in src/runs/uncapped_kendall/

set -euo pipefail

PROJECT_ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
OUTPUT_DIR="${PROJECT_ROOT}/src/runs/uncapped_kendall"
LOG_FILE="${OUTPUT_DIR}/train.log"
PID_FILE="${OUTPUT_DIR}/pid.txt"

# --- Relaxed bounds ---
export KENDALL_LOG_VAR_MIN_ACT="-4.0"
export KENDALL_LOG_VAR_MAX_PSR="2.0"
export KENDALL_LOG_VAR_MAX_POSE="4.0"
export KENDALL_HP_PREC_CAP="False"

# --- Training config ---
export CUDA_VISIBLE_DEVICES="0"
export AMP_DTYPE="bf16"
export OMP_NUM_THREADS="4"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting uncapped Kendall training..." | tee -a "${LOG_FILE}"
echo "  KENDALL_LOG_VAR_MIN_ACT=${KENDALL_LOG_VAR_MIN_ACT}" | tee -a "${LOG_FILE}"
echo "  KENDALL_LOG_VAR_MAX_PSR=${KENDALL_LOG_VAR_MAX_PSR}" | tee -a "${LOG_FILE}"
echo "  KENDALL_LOG_VAR_MAX_POSE=${KENDALL_LOG_VAR_MAX_POSE}" | tee -a "${LOG_FILE}"
echo "  KENDALL_HP_PREC_CAP=${KENDALL_HP_PREC_CAP}" | tee -a "${LOG_FILE}"
echo "  Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "  PID: $$" | tee -a "${LOG_FILE}"
echo "$$" > "${PID_FILE}"

python src/training/train.py \
    --max-epochs 100 \
    --batch-size 2 \
    >> "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "  Training PID: ${TRAIN_PID}" | tee -a "${LOG_FILE}"
echo "${TRAIN_PID}" > "${PID_FILE}"

# Tail the log briefly so the user sees startup
sleep 3
echo ""
echo "Training launched (PID ${TRAIN_PID}). Monitor with:"
echo "  tail -f ${LOG_FILE}"
echo "Kendall log vars logged every LOG_KENDALL_GRAD_EVERY steps at INFO level."
echo "Look for 'log_var_act', 'log_var_pose', 'log_var_psr' in the log."
