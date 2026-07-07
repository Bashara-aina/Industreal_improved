#!/usr/bin/env bash
# =============================================================================
# Single-task Activity MLP Training (Opus 141 ACT-MLP-10)
#
# Trains ONLY the activity MLP head — all other heads (detection, pose, PSR)
# are disabled. Single MLP on ConvNeXt-Tiny backbone for 69 activity classes.
#
# Purpose: isolate whether the multi-task setup causes the activity failure
# (41/69 classes at zero accuracy, class collapse at 0.0236) or the backbone
# is truly at ceiling (linear probe = 0.2169 ~= 0.2217 baseline).
#
# Starts from COCO-pretrained ConvNeXt-Tiny backbone (no checkpoint resume).
# Activity head is randomly initialized.
#
# Estimated runtime: 2-3 GPU-days on RTX 5060 Ti (100 epochs x batch=2).
#
# Usage:
#   bash scripts/train_singletask_activity.sh
#
# Monitor:
#   tail -f /tmp/train_singletask_act.log
#   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
#
# Kill:
#   kill "$(cat /tmp/train_singletask_act_pid 2>/dev/null)"
# =============================================================================
# NOTE: no 'set -e' — the retry loop below needs to catch training crashes.
# With errexit, the script would exit before the retry logic runs.
set -uo pipefail

# Auto-restart settings (max 5 crashes, 30s between retries)
MAX_RETRIES=5
RETRY_DELAY=30

# ── Hardware ─────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0          # RTX 5060 Ti (CUDA index 0)

# ── Thread / OOM mitigation (Opus 130 TI-1) ─────────────────────────────────
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=4

# ── RAM cache ────────────────────────────────────────────────────────────────
# Disable JPEG RAM cache entirely. systemd-oomd kills the training process
# during RAM cache fill due to memory pressure spikes (~2.7GB for 8000 images).
export RAM_CACHE_MAX_IMAGES=0

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
LOG_FILE="/tmp/train_singletask_act.log"
PID_FILE="/tmp/train_singletask_act_pid"

# ── Pre-cleanup ──────────────────────────────────────────────────────────────
for _old_pid in $(cat "${PID_FILE}" 2>/dev/null); do
    kill "${_old_pid}" 2>/dev/null || true
done
sleep 2
rm -f "${PID_FILE}"

# ── Retry loop ───────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

RETRY_COUNT=0
while true; do
    # Write PID for monitoring
    echo $$ > "${PID_FILE}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Single-task Activity MLP Training"
    if [ "${RETRY_COUNT}" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Retry #${RETRY_COUNT}/${MAX_RETRIES}"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================"
    echo ""
    echo "  Backbone:        ConvNeXt-Tiny"
    echo "  Tasks:           activity ONLY (detection/pose/psr disabled)"
    echo "  Activity heads:  Single MLP (69 classes)"
    echo "  Init:            COCO-pretrained backbone (no checkpoint resume)"
    echo "  Batch size:      2 (RTX 5060 Ti OOM mitigation)"
    echo "  GPU:             ${CUDA_VISIBLE_DEVICES} (RTX 5060 Ti)"
    echo "  OMP threads:     ${OMP_NUM_THREADS}"
    echo "  Precision:       bf16 mixed precision"
    echo "  Log file:        ${LOG_FILE}"
    echo "  PID file:        ${PID_FILE}"
    echo ""

    python src/training/train_singletask_activity.py \
        --batch-size 2 \
        --no-staged-training \
        >> "${LOG_FILE}" 2>&1

    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited with code ${EXIT_CODE}"

    # Check exit code — if clean exit (0 or SIGTERM 143), don't retry
    if [ ${EXIT_CODE} -eq 0 ] || [ ${EXIT_CODE} -eq 143 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Clean exit — no retry."
        rm -f "${PID_FILE}"
        exit ${EXIT_CODE}
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ ${RETRY_COUNT} -ge ${MAX_RETRIES} ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max retries (${MAX_RETRIES}) exhausted — giving up."
        rm -f "${PID_FILE}"
        exit ${EXIT_CODE}
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in ${RETRY_DELAY}s (retry #${RETRY_COUNT}/${MAX_RETRIES})..."
    rm -f "${PID_FILE}"
    sleep "${RETRY_DELAY}"
done
