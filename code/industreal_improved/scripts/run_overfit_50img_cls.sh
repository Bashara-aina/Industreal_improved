#!/bin/bash
# =============================================================================
# 50-IMAGE CLS-ONLY OVERFIT EXPERIMENT (Opus v9 §R4)
# Tests whether the detection classifier CAN overfit a tiny sample.
#
# Usage:
#   ./scripts/run_overfit_50img_cls.sh [--lr 1e-4] [--epochs 200] [--n-images 50]
#
# Expectation: cls loss → 0 and pos_score → 1.0 within 200-500 steps.
# If NOT, the cls head has an architectural or label-noise problem.
# =============================================================================
set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="overfit_50img_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$PROJ:$PROJ/src:$PROJ/src/models:$PROJ/src/training:$PROJ/src/evaluation:$PROJ/src/data:$PROJ/src/utils"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "============================================================"
echo "50-IMAGE CLS-ONLY OVERFIT EXPERIMENT"
echo "  Run name: $RUN_NAME"
echo "  Log dir:  $LOG_DIR"
echo "  Args:     $@"
echo "============================================================"

cd "$PROJ"

python -u scripts/overfit_50img_cls.py "$@" 2>&1 | tee "$LOG_DIR/train.log"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: PASS — cls head CAN overfit"
else
    echo "RESULT: FAIL — cls head could not overfit (see log)"
fi
echo "Log: $LOG_DIR/train.log"
echo "============================================================"
exit $EXIT
