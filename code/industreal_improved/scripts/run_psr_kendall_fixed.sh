#!/usr/bin/env bash
# =============================================================================
# Run PSR retraining with PSR_HEAD_REPAIR=1 + KENDALL_FIXED_WEIGHTS=1.
#
# Two env-driven fixes:
#   1. PSR_HEAD_REPAIR=1 — repaired transition heads:
#      LeakyReLU(0.01) instead of ReLU, bias=0.0 instead of -1.0,
#      Xavier init instead of normal(0, 0.01). See config.py PSR_HEAD_REPAIR.
#   2. KENDALL_FIXED_WEIGHTS=1 — fixed-Kendall ablation (Opus v8):
#      Fixed weights instead of learned Kendall prevents head_pose
#      from dominating the shared backbone.
#
# Both are env-driven — no code edits needed.
#
# Usage:
#   ./scripts/run_psr_kendall_fixed.sh
# =============================================================================
set -euo pipefail

# ── Hardware ─────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0          # RTX 5060 Ti

# ── Ablation env vars ────────────────────────────────────────────────────────
export KENDALL_FIXED_WEIGHTS=1         # Fixed weights instead of learned Kendall
export PSR_HEAD_REPAIR=1               # Repaired PSR transition heads
# NOTE: KENDALL_HP_FIXED_LAMBDA defaults to 0.2 (config.py)
# NOTE: KENDALL_HP_PREC_CAP defaults to True  (config.py)
# Both are the correct values for this ablation — no override needed.

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
CHECKPOINT="${PROJECT_ROOT}/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
LOG_FILE="/tmp/train_kendall_fixed.log"

# ── Launch ───────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching PSR_HEAD_REPAIR=1 + KENDALL_FIXED_WEIGHTS=1 training..."
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Batch size: 2 (avoids CUDA timeout on RTX 5060 Ti)"
echo "  Log file:   ${LOG_FILE}"
echo ""

python src/training/train.py \
    --batch-size 2 \
    --resume "${CHECKPOINT}" \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited with code ${EXIT_CODE}"
exit ${EXIT_CODE}
