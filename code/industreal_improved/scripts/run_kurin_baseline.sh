#!/bin/bash
# Run Kurin Regularized Scalarization Baseline
# Kurin et al. "In Defense of the Unitary Scalarization" (NeurIPS 2022)
#
# Protocol: equal-weight sum of task losses + standard regularization
#   - Equal weights (no FAMO, no UW-SO)
#   - Weight decay 1e-3 (Kurin lambda for heterogeneous tasks)
#   - Dropout 0.5 (Kurin p=0.5 for encoder)
#   - No PCGrad (Kurin does not use gradient surgery)
#   - No Focal PSR (standard BCE per Kurin's "standard regularization")
#   - Early stopping on val loss (Kurin validation-based model selection)
#   - Cosine annealing schedule (same as main MTL)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
SCRIPT="$ROOT/scripts/train_mtl_mvit.py"
RUN_NAME="baseline_kurin_equal_weights_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$ROOT/src/runs/$RUN_NAME"

mkdir -p "$OUTPUT_DIR"

echo "=== $(date): Starting Kurin Regularized Scalarization baseline ==="
echo "Output dir: $OUTPUT_DIR"

python "$SCRIPT" \
    --equal-weights \
    --no-pcgrad \
    --weight-decay 1e-3 \
    --epochs 100 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --lr-backbone 1e-4 \
    --lr-head 1e-3 \
    --max-batches-per-epoch 8000 \
    --eval-every 5 \
    --num-workers 0 \
    --no-psr-focal \
    --det-aug \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "=== $(date): Kurin baseline complete ==="
