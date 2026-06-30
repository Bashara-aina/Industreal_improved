#!/bin/bash
# =============================================================================
# POPW Diagnostics — RTX 3060 (12GB) — runs in parallel with 5060 Ti
# =============================================================================
# Usage:
#   ./run_3060_diagnostics.sh --phase 0            # Per-class diagnostic
#   ./run_3060_diagnostics.sh --phase 1            # Efficiency measurement
#   ./run_3060_diagnostics.sh --phase 2            # PSR go/no-go
#   ./run_3060_diagnostics.sh --phase 4            # Ablation A (single-task detection)
# =============================================================================

PROJ_DIR="/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src"
RUN_DIR="/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs"

# Parse args
PHASE=""
CKPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --ckpt)  CKPT="$2"; shift 2 ;;
        *) echo "Usage: $0 --phase [0|1|2|4] [--ckpt <path>]"; exit 1 ;;
    esac
done

# Use RTX 3060 (CUDA GPU 1)
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

cd "$PROJ_DIR"

case "$PHASE" in
    0)
        echo "Phase 0: Per-class detection diagnostic"
        python diag_per_class_truth.py --run "$RUN_DIR"
        ;;

    1)
        echo "Phase 1: Efficiency measurement on crash_recovery.pth"
        if [ -z "$CKPT" ]; then
            CKPT="$RUN_DIR/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth"
        fi
        python evaluation/evaluate.py \
            --ckpt "$CKPT" --profile-efficiency-only
        ;;

    2)
        LOG_DIR="$RUN_DIR/psr_gonogo_3060/logs"
        mkdir -p "$LOG_DIR"
        echo "Phase 2: PSR go/no-go"
        if [ -z "$CKPT" ]; then
            echo "ERROR: --ckpt required for PSR go/no-go"
            exit 1
        fi
        nohup python -u training/train.py \
            --preset stage_rf2 --train_psr True \
            --resume "$CKPT" \
            --epochs 2 --subset 0.35 \
            > "$LOG_DIR/train.log" 2>&1 &
        PID=$!
        disown
        echo "PSR go/no-go launched with PID: $PID"
        echo "Monitor: tail -f $LOG_DIR/train.log"
        ;;

    4)
        LOG_DIR="$RUN_DIR/ablation_A_3060/logs"
        CKPT_DIR="$RUN_DIR/ablation_A_3060/checkpoints"
        mkdir -p "$LOG_DIR" "$CKPT_DIR"
        echo "Phase 4: Ablation A — Single-task detection baseline"
        nohup python -u training/train.py \
            --preset recovery_det_only \
            --seed 123 --num-workers 4 \
            > "$LOG_DIR/train.log" 2>&1 &
        PID=$!
        disown
        echo "Ablation A launched with PID: $PID"
        echo "Monitor: tail -f $LOG_DIR/train.log"
        ;;

    *)
        echo "Usage: $0 --phase [0|1|2|4] [--ckpt <path>]"
        exit 1
        ;;
esac
