#!/bin/bash
# =============================================================================
# launch_st_baselines.sh — Unified 4-head x 5-seed ST baseline launcher
#
# Implements Tier 1.3 from FINAL_RANKED_RECOMMENDATIONS.md (ULTIMATE V2).
#
# Trains single-task (ST) versions of all 4 heads using the dedicated
# train_singletask_*.py scripts. Each head is trained from COCO-pretrained
# ConvNeXt-Tiny with BF16 mixed precision, 50 epochs, repeated over 5 seeds
# for statistical rigor (bootstrap CIs).
#
# Usage:
#   ./scripts/launch_st_baselines.sh                      # full run
#   ./scripts/launch_st_baselines.sh --dry-run            # print commands only
#   SKIP=pose,det ./scripts/launch_st_baselines.sh        # skip specific heads
#   DEVICES=0 ./scripts/launch_st_baselines.sh            # override CUDA device
#
# Expected outputs per head + seed:
#   runs/st_baselines/<head>/seed_<n>/
#   ├── checkpoints/          best.pth, epoch_*.pth
#   ├── logs/                 tensorboard events, train.log
#   ├── eval_outputs/         per-epoch validation metrics
#   └── metrics.json          final metrics with bootstrap CIs
#
# GPU-hours (RTX 3060, 50 epochs x 5 seeds):
#   Head        GPU-hours  Wall-clock (sequential)
#   pose        17.5       ~17.5 h
#   detection   35.0       ~35.0 h
#   activity    25.0       ~25.0 h
#   psr         25.0       ~25.0 h
#   ─────────────────────────────────
#   Total      102.5       ~102.5 h (~4.3 days)
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
# Project root (where src/, scripts/, etc. live inside the nested checkout)
PROJECT_ROOT="$(cd "$(dirname "$0")/../code/industreal_improved" && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/src/training"

SEEDS=(103 104 105 106 107)
EPOCHS=50
BATCH_SIZE=2
LR=5e-4
NUM_WORKERS=4

# Parse CLI
DRY_RUN=false
SKIP=""
if [ $# -gt 0 ]; then
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --skip) SKIP="$2"; shift ;;
        *) echo "Usage: $0 [--dry-run] [--skip pose,det,act,psr]" >&2; exit 1 ;;
    esac
fi

# GPU selection
CUDA_DEVICE="${DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MIXED_PRECISION=1

# ---- Head definitions -------------------------------------------------------
declare -A SCRIPTS
SCRIPTS[pose]="$SCRIPT_DIR/train_singletask_pose.py"
SCRIPTS[detection]="$SCRIPT_DIR/train_singletask_detection.py"
SCRIPTS[activity]="$SCRIPT_DIR/train_singletask_activity.py"
SCRIPTS[psr]="$SCRIPT_DIR/train_singletask_psr.py"

# PSR-specific defaults (ablation_psr_only preset uses batch_size=6, but we
# enforce a common batch_size=2 across all heads for consistency).
declare -A EXTRA_ARGS
EXTRA_ARGS[psr]="--sequence-mode"

declare -A GPU_HOURS
GPU_HOURS[pose]=17.5
GPU_HOURS[detection]=35.0
GPU_HOURS[activity]=25.0
GPU_HOURS[psr]=25.0

declare -A HEAD_NAMES
HEAD_NAMES[pose]="Head Pose"
HEAD_NAMES[detection]="Detection"
HEAD_NAMES[activity]="Activity"
HEAD_NAMES[psr]="PSR"

# ---- Run order: fastest first (pose -> detection -> activity -> psr) ---------
ORDER=(pose detection activity psr)

# ---- Summary header ---------------------------------------------------------
echo "=============================================================================="
echo "  ST Baseline Launcher — T1.3"
echo "  Date: $(date)"
echo "  GPU:  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
echo "  Epochs: $EPOCHS | Batch size: $BATCH_SIZE | LR: $LR"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output root: runs/st_baselines/<head>/seed_<n>/"
echo "=============================================================================="
echo ""

TOTAL_GPU_HOURS=0
for head in "${ORDER[@]}"; do
    TOTAL_GPU_HOURS=$(echo "$TOTAL_GPU_HOURS + ${GPU_HOURS[$head]}" | bc 2>/dev/null || \
        awk "BEGIN {print $TOTAL_GPU_HOURS + ${GPU_HOURS[$head]}}")
done
echo "Total estimated GPU-hours (all heads x 5 seeds): $TOTAL_GPU_HOURS"
echo "Estimated wall-clock (sequential, RTX 3060):      ~${TOTAL_GPU_HOURS}h (~$(echo "scale=1; $TOTAL_GPU_HOURS / 24" | bc)d)"
echo ""

# ---- Launch loop ------------------------------------------------------------
for head in "${ORDER[@]}"; do
    # Skip if requested
    if [ -n "$SKIP" ]; then
        IFS=',' read -ra SKIP_LIST <<< "$SKIP"
        skip_this=false
        for s in "${SKIP_LIST[@]}"; do
            if [ "$s" = "$head" ]; then
                skip_this=true
                break
            fi
        done
        $skip_this && echo "--- Skipping $head (per SKIP=$SKIP) ---" && continue
    fi

    script="${SCRIPTS[$head]}"
    extra="${EXTRA_ARGS[$head]-}"
    head_name="${HEAD_NAMES[$head]}"
    gpu_hours="${GPU_HOURS[$head]}"

    echo ""
    echo "=============================================================================="
    echo "  HEAD: $head_name ($head)"
    echo "  Script: $script"
    echo "  Estimated GPU-hours (5 seeds): $gpu_hours"
    echo "  Estimated wall-clock:          ~${gpu_hours}h"
    echo "=============================================================================="

    for seed in "${SEEDS[@]}"; do
        # Output directory: runs/st_baselines/<head>/seed_<n>/
        output_root="$PROJECT_ROOT/src/runs/st_baselines/$head/seed_$seed"
        mkdir -p "$output_root"

        cmd="OUTPUT_ROOT_OVERRIDE=$output_root python \"$script\" \
            --max-epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --lr $LR \
            --seed $seed \
            --num-workers $NUM_WORKERS \
            $extra"

        echo ""
        echo "--- [$head] seed=$seed starting at $(date) ---"

        if $DRY_RUN; then
            echo "[DRY-RUN] $cmd"
        else
            # shellcheck disable=SC2086
            OUTPUT_ROOT_OVERRIDE="$output_root" python "$script" \
                --max-epochs $EPOCHS \
                --batch-size $BATCH_SIZE \
                --lr $LR \
                --seed $seed \
                --num-workers $NUM_WORKERS \
                $extra
            echo "--- [$head] seed=$seed completed at $(date) ---"
        fi
    done

    echo "--- $head_name ($head) ALL 5 SEEDS COMPLETE ---"
done

# ---- Final summary ----------------------------------------------------------
echo ""
echo "=============================================================================="
echo "  ALL ST BASELINES COMPLETE"
echo "  Completed at: $(date)"
echo ""
echo "  Output structure:"
echo "  runs/st_baselines/"
for head in "${ORDER[@]}"; do
    echo "    $head/"
    for seed in "${SEEDS[@]}"; do
        echo "      seed_$seed/"
        echo "        checkpoints/   best.pth (chosen by val metric)"
        echo "        logs/          train.log + tensorboard"
        echo "        eval_outputs/  epoch_{N}_metrics.json"
        echo "        metrics.json   final with bootstrap CIs"
    done
done
echo "=============================================================================="
