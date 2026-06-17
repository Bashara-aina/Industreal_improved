#!/bin/bash
# =============================================================================
# D7 [opus OQ-8, RC-23] — zero-cost re-eval of latest.pth (RAW end-of-epoch
# weights) at MAX_BATCHES=200. This is the FIRST experiment opus recommends
# before any retrain: the EMA-contamination of best.pth (RC-13) and the
# 200-frame slice skew (RC-23) mean neither the post-retrain eval nor the
# standalone metric is a measurement of what the 1-epoch retrain actually
# learned. The "raw" end-of-epoch weights are already on disk in
# `latest.pth`, so this costs ZERO GPU-hours.
#
# Key knobs (all from the opus patches):
#   EVAL_SKIP_REINIT=1  → use latest.pth weights AS-IS, no head reinit
#                         (P3 made the heads live enough; we want to
#                          measure what the retrain produced)
#   P6 fix in eval_post_reinit.py:68 → uses collate_fn (with clip_rgb),
#                         not collate_fn_sequences (which dropped clip_rgb
#                         and forced the VideoMAE half to zero at eval)
#   P8 fix in eval_post_reinit.py:111-114 → MultiTaskLoss with the right
#                         num_classes_act / num_psr_components
#   MAX_BATCHES=200     → 4× the 50-batch slice that gave the misleading
#                         1-of-11 PSR pattern (RC-23)
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
SOURCE_DIR="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints"
LATEST="$SOURCE_DIR/latest.pth"
BEST="$SOURCE_DIR/best.pth"
CRASH="$SOURCE_DIR/crash_recovery.pth"

# Pick latest.pth if present (raw, end-of-epoch weights). Fall back to
# best.pth (raw, post-EMA; RC-13 contaminated), then to crash_recovery.pth
# (the pre-reinit collapsed source — only used if nothing else exists).
if [ -f "$LATEST" ]; then
    CKPT="$LATEST"
    KIND="latest"
elif [ -f "$BEST" ]; then
    CKPT="$BEST"
    KIND="best"
elif [ -f "$CRASH" ]; then
    CKPT="$CRASH"
    KIND="crash"
else
    echo "ERROR: no checkpoint under $SOURCE_DIR"
    exit 1
fi

RUN_NAME="eval_latest_p200_${KIND}_$(date +%Y%m%d_%H%M%S)"
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
export MAX_BATCHES=200
export EVAL_CKPT="$CKPT"
export EVAL_SKIP_REINIT=1
export RUN_NAME="$RUN_NAME"

echo "============================================================"
echo "D7: latest.pth re-eval (zero-cost) — opus OQ-8 / RC-23"
echo "  Source ckpt : $CKPT"
echo "  Kind        : $KIND  (raw end-of-epoch weights expected)"
echo "  MAX_BATCHES : 200    (4× the 50-batch slice that gave RC-23 skew)"
echo "  Patches     : P6 (collate_fn)  P8 (MultiTaskLoss args)  P7 (sigmoid det_conf)"
echo "  Output dir  : $LOG_DIR"
echo "============================================================"

cd "$PROJ"
python -u eval_post_reinit.py 2>&1 | tee "$LOG"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: SUCCESS — eval completed"
    echo "  Metrics: $LOG_DIR/metrics.json"
    echo ""
    echo "=== Pose A/B verdict (Q3 / OQ-8) ==="
    python3 -c "
import json
with open('$LOG_DIR/metrics.json') as f:
    m = json.load(f)
keys = [
    'position_MAE_mm', 'head_pose_angular_MAE_deg',
    'forward_angular_MAE_deg', 'up_angular_MAE_deg',
    'act_accuracy', 'act_top5_accuracy', 'act_macro_f1',
    'psr_overall_f1', 'psr_edit_score', 'psr_pos',
    'det_mAP50', 'det_mAP_50_95', 'det_n_present_classes',
    'n_samples',
]
for k in keys:
    v = m.get(k, 'MISSING')
    if isinstance(v, float):
        if v != v: v = 'NaN'
    print(f'  {k:30s} = {v}')
print()
# Quick pose verdict
ang = m.get('head_pose_angular_MAE_deg', None)
fwd = m.get('forward_angular_MAE_deg', None)
if isinstance(ang, (int, float)) and 10 <= ang <= 60:
    print(f'  ✅  head_pose_angular_MAE_deg={ang:.2f}°  — pose regression hypothesis (H3.2)')
    print('     was the dominant cause; raw weights are in the expected ~10-25° range')
    print('     (or near the 60° regression floor if backbone drift was the real driver).')
elif isinstance(fwd, (int, float)) and fwd > 50:
    print(f'  ⚠️  forward_angular_MAE_deg={fwd:.2f}°  — pose still regressed on raw weights.')
    print('     This shifts blame from RC-13 (EMA blend) toward H3.2 (backbone drift) or H3.3 (noise).')
else:
    print(f'  ⚠️  pose numbers unavailable or anomalous. Inspect metrics.json manually.')
"
else
    echo "RESULT: FAILED with exit code $EXIT"
    echo "  Check tail of $LOG for crash trace"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
