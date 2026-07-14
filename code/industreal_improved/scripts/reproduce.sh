#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Reproducibility orchestrator for POPW / IndustReal
# =============================================================================
# Run from anywhere (paths computed relative to this script):
#
#   bash scripts/reproduce.sh                    # full reproduction
#   bash scripts/reproduce.sh --dry-run          # preview only
#   bash scripts/reproduce.sh --data /path/to/datasets/industreal
#   bash scripts/reproduce.sh --cuda 0           # single GPU
#   bash scripts/reproduce.sh --cuda 0,1         # two GPUs
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES   GPU selection (default: 0)
#   POPW_ROOT              Override POPW_ROOT (dataset root, default: internal path)
#   ABLATION_EPOCHS        Ablation epochs (default: 25)
#   REPRODUCE_SKIP_{ST,MTL,ABL,EVAL}   Set to 1 to skip a step
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (everything relative to this script's location)
# ---------------------------------------------------------------------------
_HERE="$(cd "$(dirname "$0")" && pwd)"
_PROJECT_ROOT="$(cd "$_HERE/.." && pwd)"
_SRC="$_PROJECT_ROOT/src"
_ST_SCRIPTS="$_SRC/training"

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------
DATA_OVERRIDE=""
DRY_RUN=0
CUDA="${CUDA_VISIBLE_DEVICES:-0}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-25}"

# Seeds
ST_SEEDS=(103 104 105 106 107)
MTL_SEEDS=(42 123 7)

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --data) DATA_OVERRIDE="$2"; shift 2 ;;
        --cuda) CUDA="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--dry-run] [--data <path>] [--cuda <gpu_ids>]"
            echo ""
            echo "  --dry-run          Print commands without executing"
            echo "  --data <path>      Override POPW_ROOT (dataset location)"
            echo "  --cuda <gpus>      CUDA_VISIBLE_DEVICES (default: 0)"
            echo ""
            echo "  Environment:"
            echo "    CUDA_VISIBLE_DEVICES   Alternative GPU selection"
            echo "    POPW_ROOT              Same as --data"
            echo "    ABLATION_EPOCHS        Ablation epochs (default: 25)"
            echo "    REPRODUCE_SKIP_ST=1    Skip single-task baselines"
            echo "    REPRODUCE_SKIP_MTL=1   Skip multi-task training"
            echo "    REPRODUCE_SKIP_ABL=1   Skip ablation suite"
            echo "    REPRODUCE_SKIP_EVAL=1  Skip evaluation"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$CUDA"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
info()  { echo "[reproduce] $(date '+%H:%M:%S') $*"; }
die()   { echo "[reproduce] FATAL: $*" >&2; exit 1; }
run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] $*"
    else
        info "RUNNING: $*"
        eval "$@"
    fi
}

# ---------------------------------------------------------------------------
# Step 0 — Environment verification
# ---------------------------------------------------------------------------
info "=== Step 0: Environment check ==="

# CUDA
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -5 || true
else
    die "nvidia-smi not found — CUDA required."
fi

# Python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    die "python3 not found. Set PYTHON=python if using a different interpreter."
fi

$PYTHON -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1 || \
    die "torch not installed or broken."

$PYTHON -c "import cv2; print(f'opencv {cv2.__version__}')" 2>&1 || \
    die "opencv-python not installed."

$PYTHON -c "import timm; print(f'timm {timm.__version__}')" 2>&1 || \
    die "timm not installed."

# Dataset check
_POPW_ROOT="${DATA_OVERRIDE:-${POPW_ROOT:-}}"
if [[ -z "$_POPW_ROOT" ]]; then
    _POPW_ROOT="$($PYTHON -c "import src.config as C; print(C.POPW_ROOT)" 2>/dev/null || true)"
fi
if [[ -z "$_POPW_ROOT" || ! -d "$_POPW_ROOT/recordings" ]]; then
    die "
POPW_ROOT not found at '$_POPW_ROOT'.
Provide one of:
  --data /path/to/industreal
  export POPW_ROOT=/path/to/industreal
  or edit POPW_ROOT in src/config.py

Expected structure:
  \$POPW_ROOT/
    recordings/{train,val,test}/<rec_id>/{rgb/,AR_labels.csv,OD_labels.json,PSR_labels_raw.csv,pose.csv}
    train.csv
    val.csv
    test.csv
"
fi
info "POPW_ROOT = $_POPW_ROOT"
export POPW_ROOT="$_POPW_ROOT"

info "CUDA_VISIBLE_DEVICES = $CUDA"
info "Python = $(which $PYTHON)"
info "Project root = $_PROJECT_ROOT"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Single-Task Baselines (4 heads x 5 seeds)
# ---------------------------------------------------------------------------
if [[ "${REPRODUCE_SKIP_ST:-0}" != "1" ]]; then
    info "=== Step 1: Single-task baselines (4 heads x 5 seeds) ==="
    info "Seeds: ${ST_SEEDS[*]}"
    echo ""

    cd "$_PROJECT_ROOT"

    for TASK_SCRIPT in \
        "$_ST_SCRIPTS/train_singletask_pose.py" \
        "$_ST_SCRIPTS/train_singletask_detection.py" \
        "$_ST_SCRIPTS/train_singletask_activity.py" \
        "$_ST_SCRIPTS/train_singletask_psr.py"; do

        TASK_NAME="$(basename "$TASK_SCRIPT" .py | sed 's/train_singletask_//')"
        info "--- Single-task: $TASK_NAME ---"

        for SEED in "${ST_SEEDS[@]}"; do
            RUN_NAME="st_${TASK_NAME}_seed${SEED}"
            ST_OUTDIR="$_PROJECT_ROOT/src/runs/$RUN_NAME"
            run "$PYTHON" "$TASK_SCRIPT" \
                --seed "$SEED" \
                --max-epochs 50 \
                --batch-size 2 \
                ${DATA_OVERRIDE:+--preset benchmark_full}
        done
        echo ""
    done
else
    info "=== Step 1: SKIPPED (REPRODUCE_SKIP_ST=1) ==="
fi

# ---------------------------------------------------------------------------
# Step 2 — Multi-Task Learning
# ---------------------------------------------------------------------------
if [[ "${REPRODUCE_SKIP_MTL:-0}" != "1" ]]; then
    info "=== Step 2: Multi-task learning (3 seeds) ==="
    info "Seeds: ${MTL_SEEDS[*]}"
    echo ""

    cd "$_PROJECT_ROOT"

    for SEED in "${MTL_SEEDS[@]}"; do
        RUN_NAME="mtl_seed${SEED}"
        MTL_OUTDIR="$_PROJECT_ROOT/src/runs/$RUN_NAME"
        info "--- MTL seed=$SEED -> $MTL_OUTDIR ---"
        run env OUTPUT_ROOT_OVERRIDE="$MTL_OUTDIR" \
            $PYTHON "$_SRC/training/train.py" \
            --preset benchmark_full \
            --seed "$SEED" \
            --max-epochs 50 \
            --batch-size 2
    done
    echo ""
else
    info "=== Step 2: SKIPPED (REPRODUCE_SKIP_MTL=1) ==="
fi

# ---------------------------------------------------------------------------
# Step 3 — Ablation Suite
# ---------------------------------------------------------------------------
if [[ "${REPRODUCE_SKIP_ABL:-0}" != "1" ]]; then
    info "=== Step 3: Ablation suite ($ABLATION_EPOCHS epochs each) ==="
    echo ""

    ABLATIONS=(det act psr pose kendall-fixed grouping-none)
    for ABL in "${ABLATIONS[@]}"; do
        info "--- Ablation: $ABL ---"
        run env ABLATION_EPOCHS="$ABLATION_EPOCHS" \
            CUDA_VISIBLE_DEVICES="$CUDA" \
            bash "$_HERE/run_ablation_suite.sh" "$ABL"
        echo ""
    done
else
    info "=== Step 3: SKIPPED (REPRODUCE_SKIP_ABL=1) ==="
fi

# ---------------------------------------------------------------------------
# Step 4 — Evaluation
# ---------------------------------------------------------------------------
if [[ "${REPRODUCE_SKIP_EVAL:-0}" != "1" ]]; then
    info "=== Step 4: Test-split evaluation ==="

    cd "$_PROJECT_ROOT"

    # Evaluate single-task baselines
    if [[ "${REPRODUCE_SKIP_ST:-0}" != "1" ]]; then
        for TASK_SCRIPT in \
            "$_ST_SCRIPTS/train_singletask_pose.py" \
            "$_ST_SCRIPTS/train_singletask_detection.py" \
            "$_ST_SCRIPTS/train_singletask_activity.py" \
            "$_ST_SCRIPTS/train_singletask_psr.py"; do
            TASK_NAME="$(basename "$TASK_SCRIPT" .py | sed 's/train_singletask_//')"
            for SEED in "${ST_SEEDS[@]}"; do
                CKPT="$_PROJECT_ROOT/src/runs/st_${TASK_NAME}_seed${SEED}/checkpoints/best.pth"
                if [[ -f "$CKPT" ]]; then
                    run $PYTHON "$_SRC/evaluation/evaluate.py" \
                        --checkpoint "$CKPT" \
                        --split test \
                        --save-dir "$_PROJECT_ROOT/src/runs/st_${TASK_NAME}_seed${SEED}/eval_outputs"
                else
                    info "No checkpoint at $CKPT — skipping st_${TASK_NAME}_seed${SEED}"
                fi
            done
        done
    fi

    # Evaluate MTL
    if [[ "${REPRODUCE_SKIP_MTL:-0}" != "1" ]]; then
        for SEED in "${MTL_SEEDS[@]}"; do
            CKPT="$_PROJECT_ROOT/src/runs/mtl_seed${SEED}/checkpoints/best.pth"
            if [[ -f "$CKPT" ]]; then
                run $PYTHON "$_SRC/evaluation/evaluate.py" \
                    --checkpoint "$CKPT" \
                    --split test \
                    --save-dir "$_PROJECT_ROOT/src/runs/mtl_seed${SEED}/eval_outputs"
            else
                info "No checkpoint at $CKPT — skipping mtl_seed${SEED}"
            fi
        done
    fi

    # Evaluate ablations
    if [[ "${REPRODUCE_SKIP_ABL:-0}" != "1" ]]; then
        for ABL in "${ABLATIONS[@]}"; do
            CKPT="$_PROJECT_ROOT/src/runs/ablation_${ABL}/checkpoints/best.pth"
            if [[ -f "$CKPT" ]]; then
                run $PYTHON "$_SRC/evaluation/evaluate.py" \
                    --checkpoint "$CKPT" \
                    --split test \
                    --save-dir "$_PROJECT_ROOT/src/runs/ablation_${ABL}/eval_outputs"
            else
                info "No checkpoint at $CKPT — skipping ablation_${ABL}"
            fi
        done
    fi

    echo ""
fi

# ---------------------------------------------------------------------------
# Step 5 — Metrics aggregation (placeholder)
# ---------------------------------------------------------------------------
info "=== Step 5: Metrics aggregation ==="
echo ""
info "Results are in src/runs/*/eval_outputs/ for each run."
info ""
info "To aggregate across seeds for a given experiment, run:"
info "  $PYTHON $_SRC/../scripts/generate_paper_table.py --input-dir src/runs/mtl_seed*"
info ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "========================================"
info "Reproduction complete!"
info "========================================"
echo ""
info "Output structure:"
info "  src/runs/st_{pose,detection,activity,psr}_seed{103..107}/  (ST baselines)"
info "  src/runs/mtl_seed{42,123,7}/                               (MTL)"
info "  src/runs/ablation_{det,act,psr,pose,kendall-fixed,grouping-none}/  (ablations)"
info ""
info "To regenerate paper tables:"
info "  python scripts/generate_paper_table.py --input-dir src/runs"
