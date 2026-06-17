#!/bin/bash
# =============================================================================
# POST-RETRAIN RE-EVAL — validates FP32 retrain recovered 4-head metrics
# Loads the latest checkpoint from the FP32 retrain run, runs full 4-head eval,
# reports pass/fail on each metric (non-NaN, non-zero, >threshold).
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
# Auto-discover the most recent FP32 reinit run (post-patch retrain)
RETRAIN_RUN="$(ls -1d $PROJ/src/runs/reinit_5pct_fp32_* 2>/dev/null | sort | tail -1 | xargs -I{} basename {})"
if [ -z "$RETRAIN_RUN" ]; then
    echo "ERROR: no reinit_5pct_fp32_* run found under $PROJ/src/runs/"
    exit 1
fi
RETRAIN_CKPT="$PROJ/src/runs/$RETRAIN_RUN/checkpoints/best.pth"
FALLBACK_CKPT="$PROJ/src/runs/$RETRAIN_RUN/checkpoints/latest.pth"
FALLBACK2_CKPT="$PROJ/src/runs/$RETRAIN_RUN/checkpoints/crash_recovery.pth"
# train.py writes to OUTPUT_ROOT/checkpoints (default = full_multi_task_tma_tbank_benchmark),
# NOT the retrain's run dir. Add source-dir fallbacks so we pick up the actual retrain output.
SOURCE_CKPT="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/best.pth"
SOURCE_LATEST="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth"
SOURCE_CR="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"

# Pick the most informative checkpoint: best if it exists, else latest, else crash_recovery
if [ -f "$RETRAIN_CKPT" ]; then
    CKPT="$RETRAIN_CKPT"
    echo "[eval] using best.pth (retrain run dir)"
elif [ -f "$SOURCE_CKPT" ] && [ "$SOURCE_CKPT" -nt "$RETRAIN_CKPT" -o ! -f "$RETRAIN_CKPT" ]; then
    CKPT="$SOURCE_CKPT"
    echo "[eval] using source best.pth (train.py writes to OUTPUT_ROOT, not run dir)"
elif [ -f "$FALLBACK_CKPT" ]; then
    CKPT="$FALLBACK_CKPT"
    echo "[eval] using latest.pth (retrain run dir)"
elif [ -f "$SOURCE_LATEST" ]; then
    CKPT="$SOURCE_LATEST"
    echo "[eval] using source latest.pth"
elif [ -f "$FALLBACK2_CKPT" ]; then
    CKPT="$FALLBACK2_CKPT"
    echo "[eval] using crash_recovery.pth (training may still be in progress)"
elif [ -f "$SOURCE_CR" ]; then
    CKPT="$SOURCE_CR"
    echo "[eval] using source crash_recovery.pth"
else
    echo "ERROR: no checkpoint found under $PROJ/src/runs/$RETRAIN_RUN/checkpoints/ or source dir"
    exit 1
fi

RUN_NAME="eval_post_retrain_fp32_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval.log"

export PYTHONPATH="$PROJ:$PROJ/src:$PROJ/src/models:$PROJ/src/training:$PROJ/src/evaluation:$PROJ/src/data:$PROJ/src/utils"
export SUBSET_RATIO=0.05
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=4096:8
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_SPLIT=val
export EVAL_BS=4
export MAX_BATCHES=50
export EVAL_CKPT="$CKPT"
export EVAL_SKIP_REINIT=1
export RUN_NAME="$RUN_NAME"

echo "============================================================"
echo "POST-RETRAIN RE-EVAL — verifying FP32 retrain recovered metrics"
echo "  Source ckpt: $CKPT"
echo "  Output dir : $LOG_DIR"
echo "============================================================"

cd "$PROJ"
python -u eval_post_reinit.py 2>&1 | tee "$LOG"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: SUCCESS — re-eval completed"
    echo "  Metrics: $LOG_DIR/metrics.json"
    echo "  CSV    : $LOG_DIR/eval_results.csv"
    if [ -f "$LOG_DIR/metrics.json" ]; then
        echo ""
        echo "=== Key metrics (sanitized) ==="
        python3 -c "
import json
with open('$LOG_DIR/metrics.json') as f:
    m = json.load(f)
keys = ['loss', 'act_accuracy', 'act_top5_accuracy', 'act_macro_f1', 'act_clip_accuracy',
        'psr_overall_f1', 'psr_f1_at_t', 'psr_edit_score', 'psr_pos',
        'position_MAE_mm', 'head_pose_angular_MAE_deg', 'forward_x_MAE', 'forward_y_MAE', 'forward_z_MAE',
        'det_mAP50', 'det_mAP_50_95', 'det_mAP50_pc', 'det_mAP50_all_frames',
        'det_precision', 'det_recall', 'det_n_present_classes',
        'forward_angular_MAE_deg', 'up_angular_MAE_deg',
        'act_frame_accuracy', 'act_macro_recall', 'act_macro_f1_present', 'act_mean_per_class_acc',
        'n_samples']
for k in keys:
    v = m.get(k, 'MISSING')
    if isinstance(v, float):
        if v != v:  # NaN
            v = 'NaN'
        elif v == 0.0:
            v = '0.0 (Z!)'
    print(f'  {k:30s} = {v}')
"
    fi
else
    echo "RESULT: FAILED with exit code $EXIT"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
