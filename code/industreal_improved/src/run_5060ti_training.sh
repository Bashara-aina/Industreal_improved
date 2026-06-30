#!/bin/bash
# =============================================================================
# POPW Training — RTX 5060 Ti (16GB)
# Phase B: stage_rf2 — Detection + Body/Head Pose (resume from checkpoint)
# Phase C: stage_rf3 — Add Activity (resume from Phase B checkpoint)
# =============================================================================
# Usage:
#   ./run_5060ti_training.sh --phase B --ckpt <path>   # RF2 from crash_recovery
#   ./run_5060ti_training.sh --phase C --ckpt <path>   # RF3 from RF2 best.pth
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
        *) echo "Usage: $0 --phase [B|C] --ckpt <path>"; exit 1 ;;
    esac
done

# Use RTX 5060 Ti (CUDA GPU 0)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export _STAGE_MANAGER_ACTIVE=1
export _STAGE_NAME=rf2

cd "$PROJ_DIR"

case "$PHASE" in
    B|b)
        LOG_DIR="$RUN_DIR/phase_B_5060ti/logs"
        CKPT_DIR="$RUN_DIR/phase_B_5060ti/checkpoints"
        mkdir -p "$LOG_DIR" "$CKPT_DIR"
        echo "============================================"
        echo "Phase B: stage_rf2 on RTX 5060 Ti"
        echo "Checkpoint: $CKPT"
        echo "Log: $LOG_DIR/train.log"
        echo "============================================"
        nohup python -u training/train.py \
            --preset stage_rf2 \
            --resume "$CKPT" \
            --seed 42 --num-workers 4 \
            > "$LOG_DIR/train.log" 2>&1 &
        PID=$!
        disown
        echo "Launched with PID: $PID"
        echo "Monitor: tail -f $LOG_DIR/train.log"
        ;;

    C|c)
        LOG_DIR="$RUN_DIR/phase_C_5060ti/logs"
        CKPT_DIR="$RUN_DIR/phase_C_5060ti/checkpoints"
        mkdir -p "$LOG_DIR" "$CKPT_DIR"
        echo "============================================"
        echo "Phase C: stage_rf3 on RTX 5060 Ti"
        echo "Checkpoint: $CKPT"
        echo "Log: $LOG_DIR/train.log"
        echo "============================================"
        nohup python -u training/train.py \
            --preset stage_rf3 \
            --resume "$CKPT" \
            --seed 42 --num-workers 4 \
            > "$LOG_DIR/train.log" 2>&1 &
        PID=$!
        disown
        echo "Launched with PID: $PID"
        echo "Monitor: tail -f $LOG_DIR/train.log"
        ;;

    *)
        echo "Usage: $0 --phase [B|C] --ckpt <path>"
        exit 1
        ;;
esac
