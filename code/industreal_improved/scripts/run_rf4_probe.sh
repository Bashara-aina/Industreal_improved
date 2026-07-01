#!/bin/bash
# ============================================================
# POPW RF4 50-Step Probe — Verify all 4 tasks before RF4 launch
#   bash scripts/run_rf4_probe.sh
# ============================================================
set -e

WDIR="/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved"
cd "$WDIR"

# Kill any lingering GPU processes
pkill -f "train.py" 2>/dev/null || true
sleep 2

# Export PYTHONPATH
export PYTHONPATH="$WDIR:$WDIR/src:$WDIR/src/models:$WDIR/src/training:$WDIR/src/evaluation:$WDIR/src/data:$WDIR/src/utils"

# 50-step probe: 2 epochs at 2% data, all 4 tasks from epoch 0
export TRAIN_MAX_STEPS=50
export TRAINER_DEBUG=1

LOG="$WDIR/src/runs/rf4_probe_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

echo "============================================================"
echo "POPW RF4 50-Step Probe"
echo "  Preset: stage_rf4 (all 4 tasks from epoch 0)"
echo "  Max steps: 50"
echo "  Data: 2% with greedy coverage stratification"
echo "  Log: $LOG"
echo "============================================================"

python src/training/train.py \
    --preset stage_rf4 \
    --max-epochs 2 \
    --subset-ratio 0.02 \
    --no-staged-training \
    2>&1 | tee "$LOG"

PROBE_EXIT=$?

echo ""
echo "=== PROBE EXIT CODE: $PROBE_EXIT ==="
echo ""
echo "=== CHECK THESE AT THE END OF THE LOG ==="
echo "  1. [DET-HEALTH] cls_preds mean: between -3 to -1"
echo "  2. [DET-HEALTH] det_gt_fraction: ~0.35-0.45"
echo "  3. [get_sampler] max/min ratio: <10x"
echo "  4. [DIVERSITY] pred_distinct: >= 10 groups"
echo "  5. [GRAD-NORM] all 4 heads: > 0"
echo "  6. [Kendall log_sigma] lv_det/lv_pose/lv_act/lv_psr: in [-1, +1]"
echo ""

if [ $PROBE_EXIT -ne 0 ] && [ $PROBE_EXIT -ne 143 ]; then
    echo "⚠️  PROBE FAILED (exit code $PROBE_EXIT) — fix before RF4 launch"
    exit 1
fi

echo "✅ PROBE PASSED — ready for RF4 full run"
echo "Launch: python src/training/train.py --preset stage_rf4 --subset-ratio 0.02 --no-staged-training 2>&1 | tee src/runs/rf4_$(date +%Y%m%d_%H%M%S).log"
