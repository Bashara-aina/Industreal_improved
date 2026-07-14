#!/bin/bash
# =============================================================================
# launch_full_training_pipeline.sh — MTL baseline + 4 ST baselines + 3 ablations
#
# Orchestrates the full training plan per 30_DAY_EXECUTION_PLAN.md.
#
# GPU 0: 4 single-task baselines (pose -> act -> psr -> det), 3 seeds each
# GPU 1: MTL baseline (100ep), then 3 priority ablations (50ep each)
#
# GPU-h budgets (3 seeds, scaled from 5-seed):
#   GPU 0 — pose=10.5, act=15.0, psr=15.0, det=21.0       total=61.5
#   GPU 1 — MTL=50.0, UW-SO=25.0, uncapped-kendall=25.0, bifpn=25.0  total=125.0
#
# Usage:
#   ./scripts/launch_full_training_pipeline.sh              # full pipeline
#   ./scripts/launch_full_training_pipeline.sh --dry-run    # print commands only
#   ./scripts/launch_full_training_pipeline.sh --st-only    # GPU 0 ST baselines only
#   ./scripts/launch_full_training_pipeline.sh --mtl-only   # GPU 1 MTL only
#   ./scripts/launch_full_training_pipeline.sh --ablation-only  # GPU 1 ablations only
#
# Log: runs/training_pipeline_<timestamp>.log
# =============================================================================
set -euo pipefail

# ---- Paths ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/src/runs"
LOG_FILE="$LOG_DIR/training_pipeline_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# ---- Config ---------------------------------------------------------------
SEEDS=(42 123 7)
ST_EPOCHS=50
MTL_EPOCHS=100
ABLATION_EPOCHS=50
BATCH_SIZE=2
LR=5e-4
NUM_WORKERS=4

# GPU 0: ST heads order (per 30_DAY_EXECUTION_PLAN priority)
ST_ORDER=(pose activity psr detection)

# GPU 1: MTL then ablations
ABLATION_ORDER=(uwso uncapped_kendall bifpn)

# GPU-h budgets (3 seeds, from COMPUTE_SCHEDULE.md)
declare -A GPU_HOURS
GPU_HOURS[pose]=10.5
GPU_HOURS[activity]=15.0
GPU_HOURS[psr]=15.0
GPU_HOURS[detection]=21.0
GPU_HOURS[mtl]=50.0
GPU_HOURS[uwso]=25.0
GPU_HOURS[uncapped_kendall]=25.0
GPU_HOURS[bifpn]=25.0

declare -A HEAD_NAMES
HEAD_NAMES[pose]="Head Pose"
HEAD_NAMES[activity]="Activity"
HEAD_NAMES[psr]="PSR"
HEAD_NAMES[detection]="Detection"

# ---- Parse CLI ------------------------------------------------------------
DRY_RUN=false
RUN_MODE="all"  # all | st-only | mtl-only | ablation-only

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --st-only) RUN_MODE="st-only"; shift ;;
        --mtl-only) RUN_MODE="mtl-only"; shift ;;
        --ablation-only) RUN_MODE="ablation-only"; shift ;;
        *) echo "Usage: $0 [--dry-run] [--st-only|--mtl-only|--ablation-only]" >&2; exit 1 ;;
    esac
done

# ---- Helpers --------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

run_cmd() {
    # run_cmd <description> <gpu_id> <pid_var_name> -- <command...>
    # or run_cmd <description> <gpu_id> "foreground" -- <command...>
    local desc="$1"; shift
    local gpu="$1"; shift
    local flow="${1:-background}"; shift
    [[ "$1" == "--" ]] && shift

    log "=== [GPU $gpu] LAUNCH: $desc ==="
    log "  CMD: $*"

    if $DRY_RUN; then
        log "  [DRY-RUN] Would launch (GPU $gpu): $*"
        return 0
    fi

    local log_tag
    log_tag=$(echo "$desc" | tr ' /' '__')

    if [[ "$flow" == "foreground" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" bash -c "$*" 2>&1 | tee -a "$LOG_FILE"
        local rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            log "=== [GPU $gpu] FAILED (exit $rc): $desc ==="
            return $rc
        fi
        log "=== [GPU $gpu] COMPLETED: $desc ==="
        return 0
    else
        # background — store PID in variable name passed as $3
        local pvar="$flow"
        CUDA_VISIBLE_DEVICES="$gpu" bash -c "$*" >> "$LOG_FILE" 2>&1 &
        local pid=$!
        eval "$pvar=$pid"
        log "  PID=$pid"
        return 0
    fi
}

health_check() {
    local pid="$1" desc="$2" gpu="$3"
    if ! kill -0 "$pid" 2>/dev/null; then
        log "=== [GPU $gpu] HEALTH CHECK FAILED: $desc (PID $pid is dead) ==="
        log "=== ABORTING PIPELINE ==="
        exit 1
    fi
    log "  Health check OK: $desc (PID $pid)"
}

wait_for_pid() {
    local pid="$1" desc="$2" gpu="$3"
    log "=== [GPU $gpu] WAITING for $desc (PID $pid) ==="
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    # Check exit status via wait (non-blocking now)
    wait "$pid" 2>/dev/null
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "=== [GPU $gpu] $desc EXITED with code $rc ==="
        log "=== ABORTING PIPELINE ==="
        exit $rc
    fi
    log "=== [GPU $gpu] $desc COMPLETED ==="
}

print_header() {
    echo ""
    echo "=============================================================================="
    echo "  FULL TRAINING PIPELINE — $(date)"
    echo "  Log: $LOG_FILE"
    echo ""
    echo "  GPU 0 (ST baselines): ${ST_ORDER[*]}"
    echo "         Seeds: ${SEEDS[*]} | Epochs: $ST_EPOCHS"
    echo ""
    echo "  GPU 1 (MTL + ablations): MTL baseline (${MTL_EPOCHS}ep)"
    echo "         Then: ${ABLATION_ORDER[*]} (${ABLATION_EPOCHS}ep each)"
    echo ""
    echo "  Total GPU-h:"
    echo "    GPU 0 ST:"
    local total_gpu0=0 total_gpu1=0
    for h in "${ST_ORDER[@]}"; do
        printf "      %-12s %5.1f\n" "$h" "${GPU_HOURS[$h]}"
        total_gpu0=$(echo "$total_gpu0 + ${GPU_HOURS[$h]}" | bc)
    done
    total_gpu0=$(echo "$total_gpu0" | bc)
    echo "      ─────────────────────"
    printf "      %-12s %5.1f\n" "GPU 0 total" "$total_gpu0"
    echo ""
    echo "    GPU 1:"
    printf "      %-20s %5.1f\n" "mtl" "${GPU_HOURS[mtl]}"
    total_gpu1=${GPU_HOURS[mtl]}
    for h in "${ABLATION_ORDER[@]}"; do
        printf "      %-20s %5.1f\n" "$h" "${GPU_HOURS[$h]}"
        total_gpu1=$(echo "$total_gpu1 + ${GPU_HOURS[$h]}" | bc)
    done
    echo "      ─────────────────────────"
    printf "      %-20s %5.1f\n" "GPU 1 total" "$total_gpu1"
    local grand_total
    grand_total=$(echo "$total_gpu0 + $total_gpu1" | bc)
    printf "      GRAND TOTAL:          %5.1f GPU-h\n" "$grand_total"
    echo "=============================================================================="
    echo ""
}

print_header | tee -a "$LOG_FILE"

# ===========================================================================
# PHASE 1: GPU 0 — Single-task baselines (sequential per head, 3 seeds)
# ===========================================================================
run_st_baselines() {
    log ""
    log "======================================================================"
    log "PHASE 1 — GPU 0: Single-task baselines"
    log "Order: ${ST_ORDER[*]}"
    log "======================================================================"

    for head in "${ST_ORDER[@]}"; do
        local script="$PROJECT_ROOT/src/training/train_singletask_${head}.py"
        local head_name="${HEAD_NAMES[$head]}"
        local gpu_hours="${GPU_HOURS[$head]}"
        local extra_args=""

        [[ "$head" == "psr" ]] && extra_args="--sequence-mode"

        log ""
        log "--- HEAD: $head_name ($head) — estimated ${gpu_hours}h ---"

        for seed in "${SEEDS[@]}"; do
            local output_root="$PROJECT_ROOT/src/runs/st_baselines/${head}/seed_${seed}"
            mkdir -p "$output_root"

            local cmd
            cmd="OUTPUT_ROOT_OVERRIDE=$output_root python \"$script\" \
                --max-epochs $ST_EPOCHS \
                --batch-size $BATCH_SIZE \
                --lr $LR \
                --seed $seed \
                --num-workers $NUM_WORKERS \
                $extra_args"

            if $DRY_RUN; then
                log "  [DRY-RUN] GPU 0 — seed=$seed: $cmd"
            else
                log "  [GPU 0] $head seed=$seed starting..."
                CUDA_VISIBLE_DEVICES=0 bash -c "$cmd" 2>&1 | tee -a "$LOG_FILE"
                local rc=${PIPESTATUS[0]}
                if [ $rc -ne 0 ]; then
                    log "=== [GPU 0] $head seed=$seed FAILED (exit $rc) ==="
                    log "=== ABORTING PIPELINE ==="
                    exit $rc
                fi
                log "  [GPU 0] $head seed=$seed completed"
            fi
        done
        log "--- $head_name ($head) ALL 3 SEEDS COMPLETE ---"
    done
}

# ===========================================================================
# PHASE 2: GPU 1 — MTL baseline (background)
# ===========================================================================
run_mtl_baseline() {
    log ""
    log "======================================================================"
    log "PHASE 2 — GPU 1: MTL baseline (${MTL_EPOCHS} epochs)"
    log "======================================================================"

    local output_root="$PROJECT_ROOT/src/runs/mtl_baseline"
    mkdir -p "$output_root"

    local cmd
    cmd="OUTPUT_ROOT_OVERRIDE=$output_root python \"$PROJECT_ROOT/src/training/train.py\" \
        --max-epochs $MTL_EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS"

    if $DRY_RUN; then
        log "  [DRY-RUN] GPU 1 (background): $cmd"
        MTL_PID=99999
    else
        log "  [GPU 1] Launching MTL baseline in background..."
        CUDA_VISIBLE_DEVICES=1 bash -c "$cmd" >> "$LOG_FILE" 2>&1 &
        MTL_PID=$!
        log "  MTL PID=$MTL_PID"
        # Give it a moment to start, then health check
        sleep 10
        health_check "$MTL_PID" "MTL baseline" 1
    fi
}

# ===========================================================================
# PHASE 3: GPU 1 — Priority ablations (sequential, after MTL completes)
# ===========================================================================
run_ablations() {
    log ""
    log "======================================================================"
    log "PHASE 3 — GPU 1: Priority ablations (${ABLATION_EPOCHS} epochs each)"
    log "======================================================================"

    # ----- Ablation #1: UW-SO + per-task LR --------------------------------
    log ""
    log "--- ABLATION 1/3: UW-SO + per-task LR multipliers ---"

    local uwso_root="$PROJECT_ROOT/src/runs/ablation_uwso"
    mkdir -p "$uwso_root"

    local uwso_cmd
    uwso_cmd="USE_UW_SO=1 \
        UW_SO_TEMPERATURE=1.0 \
        USE_BALANCED_SOFTMAX_ACT=1 \
        USE_KENDALL=1 \
        PSR_LR_MULTIPLIER=0.5 \
        HEAD_POSE_LR_MULTIPLIER=0.3 \
        OUTPUT_ROOT_OVERRIDE=$uwso_root \
        python \"$PROJECT_ROOT/src/training/train.py\" \
        --max-epochs $ABLATION_EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS"

    if $DRY_RUN; then
        log "  [DRY-RUN] GPU 1: $uwso_cmd"
    else
        log "  [GPU 1] UW-SO ablation starting..."
        CUDA_VISIBLE_DEVICES=1 bash -c "$uwso_cmd" 2>&1 | tee -a "$LOG_FILE"
        local rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            log "=== [GPU 1] UW-SO ablation FAILED (exit $rc) ==="
            log "=== ABORTING PIPELINE ==="
            exit $rc
        fi
        log "  [GPU 1] UW-SO ablation completed"
    fi

    # ----- Ablation #2: Uncapped Kendall -----------------------------------
    log ""
    log "--- ABLATION 2/3: Uncapped Kendall (relaxed log-var bounds) ---"

    local kendall_root="$PROJECT_ROOT/src/runs/ablation_uncapped_kendall"
    mkdir -p "$kendall_root"

    local kendall_cmd
    kendall_cmd="KENDALL_LOG_VAR_MIN_ACT=-4.0 \
        KENDALL_LOG_VAR_MAX_PSR=2.0 \
        KENDALL_LOG_VAR_MAX_POSE=4.0 \
        KENDALL_HP_PREC_CAP=False \
        OUTPUT_ROOT_OVERRIDE=$kendall_root \
        python \"$PROJECT_ROOT/src/training/train.py\" \
        --max-epochs $ABLATION_EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS"

    if $DRY_RUN; then
        log "  [DRY-RUN] GPU 1: $kendall_cmd"
    else
        log "  [GPU 1] Uncapped Kendall ablation starting..."
        CUDA_VISIBLE_DEVICES=1 bash -c "$kendall_cmd" 2>&1 | tee -a "$LOG_FILE"
        local rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            log "=== [GPU 1] Uncapped Kendall FAILED (exit $rc) ==="
            log "=== ABORTING PIPELINE ==="
            exit $rc
        fi
        log "  [GPU 1] Uncapped Kendall completed"
    fi

    # ----- Ablation #3: BiFPN / TSBN --------------------------------------
    log ""
    log "--- ABLATION 3/3: BiFPN / TSBN gated ---"

    local bifpn_root="$PROJECT_ROOT/src/runs/ablation_bifpn"
    mkdir -p "$bifpn_root"

    local bifpn_cmd
    bifpn_cmd="USE_BIFPN=1 \
        OUTPUT_ROOT_OVERRIDE=$bifpn_root \
        python \"$PROJECT_ROOT/src/training/train.py\" \
        --max-epochs $ABLATION_EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS"

    if $DRY_RUN; then
        log "  [DRY-RUN] GPU 1: $bifpn_cmd"
    else
        log "  [GPU 1] BiFPN ablation starting..."
        CUDA_VISIBLE_DEVICES=1 bash -c "$bifpn_cmd" 2>&1 | tee -a "$LOG_FILE"
        local rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            log "=== [GPU 1] BiFPN ablation FAILED (exit $rc) ==="
            log "=== ABORTING PIPELINE ==="
            exit $rc
        fi
        log "  [GPU 1] BiFPN ablation completed"
    fi
}

# ===========================================================================
# EXECUTION
# ===========================================================================

# No backgrounded jobs at script exit — we manage PIDs explicitly
trap 'log "Pipeline interrupted — see $LOG_FILE for partial results"; exit 130' INT TERM

log "Pipeline started at $(date)"

# GPU 0: ST baselines (sequential, foreground, in background on GPU 0)
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "st-only" ]]; then
    log "Launching GPU 0 ST baselines in background..."
    run_st_baselines &
    ST_PID=$!
    log "ST baselines PID=$ST_PID (GPU 0)"
    sleep 5
    health_check "$ST_PID" "ST baselines" 0
else
    log "Skipping GPU 0 ST baselines (mode=$RUN_MODE)"
fi

# GPU 1: MTL baseline
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "mtl-only" ]]; then
    run_mtl_baseline
else
    log "Skipping GPU 1 MTL baseline (mode=$RUN_MODE)"
fi

# GPU 1: Wait for MTL to finish, then run ablations
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "ablation-only" ]]; then
    if [[ "$RUN_MODE" == "all" && $MTL_PID -gt 0 ]]; then
        wait_for_pid "$MTL_PID" "MTL baseline" 1
    fi
    run_ablations
else
    log "Skipping GPU 1 ablations (mode=$RUN_MODE)"
fi

# Wait for GPU 0 ST baselines to finish (if launched)
if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "st-only" ]]; then
    wait_for_pid "$ST_PID" "ST baselines" 0
fi

# ===========================================================================
# SUMMARY
# ===========================================================================
echo ""
echo "=============================================================================="
echo "  PIPELINE COMPLETE"
echo "  Finished at: $(date)"
echo "  Log: $LOG_FILE"
echo ""
echo "  Output structure:"
echo "  src/runs/"
echo "    mtl_baseline/                     MTL baseline (GPU 1)"
echo "    st_baselines/"
for h in "${ST_ORDER[@]}"; do
    echo "      $h/"
    for s in "${SEEDS[@]}"; do
        echo "        seed_$s/"
    done
done
echo "    ablation_uwso/                    UW-SO + per-task LR (GPU 1)"
echo "    ablation_uncapped_kendall/         Uncapped Kendall (GPU 1)"
echo "    ablation_bifpn/                    BiFPN/TSBN (GPU 1)"
echo "=============================================================================="
