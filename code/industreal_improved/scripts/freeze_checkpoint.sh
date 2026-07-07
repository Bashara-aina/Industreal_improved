#!/usr/bin/env bash
# =============================================================================
# freeze_checkpoint.sh — Freeze Protocol for Jul 20 Results Lock
#
# Copies the reporting checkpoint to a frozen archive, records SHA256 hashes
# of the checkpoint and all eval result files, captures git HEAD, and emits
# results_frozen.json as the single source of truth for paper tables.
#
# Usage:
#   ./scripts/freeze_checkpoint.sh                    # live freeze
#   ./scripts/freeze_checkpoint.sh --dry-run           # dry-run, no copies
#   ./scripts/freeze_checkpoint.sh --checkpoint <path> # custom checkpoint
#
# Output:
#   src/runs/rf_stages/checkpoints/reporting_checkpoint_frozen.pth
#   src/runs/rf_stages/checkpoints/results_frozen.json
# =============================================================================
set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
CHECKPOINTS="${PROJECT_ROOT}/src/runs/rf_stages/checkpoints"
BEST_PTH="${CHECKPOINTS}/best.pth"
FROZEN_PTH="${CHECKPOINTS}/reporting_checkpoint_frozen.pth"
RESULTS_JSON="${CHECKPOINTS}/results_frozen.json"

# Default eval result files to hash (relative to CHECKPOINTS, expanded below)
EVAL_RESULT_FILES=(
    "pose_kalman_eval/pose_kalman_results.json"
    "full_eval_ep18_stream/metrics.json"
    "full_eval_ep18_v2/metrics.json"
    "psr_optimal_thr/optimal_thresholds.json"
    "psr_optimal_thr_v2/optimal_thresholds.json"
    "psr_loo_cv/loo_cv_results.json"
    "psr_null_delta_table.md"
    "null_model_pos/null_model_pos.json"
    "activity_linear_probe.json"
    "activity_confusion_matrix.md"
    "activity_clip_ep18/activity_clip.json"
    "d1_yolov8m_v3/metrics.json"
    "d1_yolov8m_v2/metrics.json"
    "d1_yolov8m/metrics.json"
    "d3_full_eval/metrics.json"
    "d4_yolov8m_psr/metrics.json"
    "d4_retuned/sweep_results.json"
    "d4_retuned/verdict.json"
    "t3_full_eval.json"
    "t3_mecanno_eval.json"
    "psr_per_vs_trans/per_frame_vs_transition.json"
    "tta_results/tta_metrics.json"
    "forward_mae_per_recording.json"
    "psr_threshold_sweep.json"
    "up_vector_v3/up_vector_per_recording.json"
    "up_vector_v2/up_vector_per_recording.json"
)

# ── Parse args ─────────────────────────────────────────────────────────────────
DRY_RUN=false
CUSTOM_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --checkpoint) CUSTOM_CHECKPOINT="$2"; shift 2 ;;
        --help|-h)    head -30 "$0"; exit 0 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -n "$CUSTOM_CHECKPOINT" ]]; then
    BEST_PTH="$CUSTOM_CHECKPOINT"
    echo "[freeze] Using custom checkpoint: ${BEST_PTH}"
fi

# ── Validate ───────────────────────────────────────────────────────────────────
if [[ ! -f "$BEST_PTH" ]]; then
    echo "[freeze] FATAL: checkpoint not found at ${BEST_PTH}"
    exit 1
fi

if ! command -v sha256sum &>/dev/null; then
    echo "[freeze] FATAL: sha256sum not available"
    exit 1
fi

cd "${PROJECT_ROOT}"

# ── Gather SHAs ────────────────────────────────────────────────────────────────
echo "[freeze] Recording SHA256 of source checkpoint..."
BEST_SHA=$(sha256sum "${BEST_PTH}" | cut -d' ' -f1)
echo "         ${BEST_SHA}  ${BEST_PTH}"

COMMIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "[freeze] Git HEAD: ${COMMIT_SHA}"

# ── Copy checkpoint ────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[freeze] DRY-RUN: would cp ${BEST_PTH} -> ${FROZEN_PTH}"
else
    echo "[freeze] Copying checkpoint..."
    cp "${BEST_PTH}" "${FROZEN_PTH}"
    echo "[freeze] Wrote ${FROZEN_PTH}"
fi

# ── Record frozen SHA ──────────────────────────────────────────────────────────
if [[ -f "$FROZEN_PTH" ]]; then
    FROZEN_SHA=$(sha256sum "${FROZEN_PTH}" | cut -d' ' -f1)
else
    FROZEN_SHA="<dry-run-not-copied>"
fi

echo "[freeze] Frozen checkpoint SHA256: ${FROZEN_SHA}"
if [[ "$BEST_SHA" != "$FROZEN_SHA" && "$DRY_RUN" == "false" ]]; then
    echo "[freeze] WARNING: source and frozen SHA do not match!"
    echo "         Source: ${BEST_SHA}"
    echo "         Frozen: ${FROZEN_SHA}"
fi

# ── Hash eval result files ─────────────────────────────────────────────────────
declare -A FILE_HASHES
MISSING_FILES=()

for REL_PATH in "${EVAL_RESULT_FILES[@]}"; do
    FULL_PATH="${CHECKPOINTS}/${REL_PATH}"
    if [[ -f "$FULL_PATH" ]]; then
        FILE_HASHES["$REL_PATH"]=$(sha256sum "${FULL_PATH}" | cut -d' ' -f1)
    else
        FILE_HASHES["$REL_PATH"]="<file-not-found>"
        MISSING_FILES+=("$REL_PATH")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo "[freeze] Warning: ${#MISSING_FILES[@]} eval result file(s) not found:"
    for MF in "${MISSING_FILES[@]}"; do
        echo "         [MISSING] ${MF}"
    done
fi

# ── Build JSON ─────────────────────────────────────────────────────────────────
TIMESTAMP=$(date --iso-8601=seconds 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z")

# Build eval_files JSON array
EVAL_FILES_JSON=""
for REL_PATH in "${EVAL_RESULT_FILES[@]}"; do
    HASH="${FILE_HASHES[$REL_PATH]}"
    if [[ -n "$EVAL_FILES_JSON" ]]; then
        EVAL_FILES_JSON+=","
    fi
    EVAL_FILES_JSON+="    {\"path\": \"${REL_PATH}\", \"sha256\": \"${HASH}\"}"
done

RESULTS_JSON_CONTENT=$(
cat <<EOF
{
  "freeze_metadata": {
    "freeze_date": "2026-07-20",
    "timestamp": "${TIMESTAMP}",
    "protocol": "Opus 140 §5 Day 4-7 freeze protocol",
    "source_document": "140_OPUS_ANSWERS_V2.md §4, §5"
  },
  "checkpoint": {
    "source_path": "${BEST_PTH}",
    "source_sha256": "${BEST_SHA}",
    "frozen_path": "${FROZEN_PTH}",
    "frozen_sha256": "${FROZEN_SHA}",
    "matches_source": $(if [[ "$BEST_SHA" == "$FROZEN_SHA" ]]; then echo "true"; else echo "false"; fi)
  },
  "git": {
    "commit_sha": "${COMMIT_SHA}"
  },
  "eval_files": [
${EVAL_FILES_JSON}
  ],
  "\$schema": "src/runs/rf_stages/checkpoints/results_frozen_template.json"
}
EOF
)

# ── Write results_frozen.json ──────────────────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[freeze] DRY-RUN — would write results_frozen.json:"
    echo "${RESULTS_JSON_CONTENT}" | python3 -m json.tool 2>/dev/null || echo "${RESULTS_JSON_CONTENT}"
    echo ""
    echo "[freeze] DRY-RUN complete. No files were modified."
else
    echo "${RESULTS_JSON_CONTENT}" > "${RESULTS_JSON}"
    echo "[freeze] Wrote ${RESULTS_JSON}"
    echo "[freeze] Freeze complete."
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FREEZE CHECKPOINT SUMMARY                                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Source:   ${BEST_SHA:0:16}...  $(basename ${BEST_PTH})"
echo "║  Frozen:   ${FROZEN_SHA:0:16}...  $(basename ${FROZEN_PTH})"
echo "║  Git:      ${COMMIT_SHA:0:16}..."
echo "║  Eval files: ${#EVAL_RESULT_FILES[@]} total (${#MISSING_FILES[@]} missing)"
echo "║  Timestamp: ${TIMESTAMP}"
echo "╚══════════════════════════════════════════════════════════════╝"
