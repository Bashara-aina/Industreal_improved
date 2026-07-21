#!/bin/bash
# Run overfit probes for all 4 heads on GPU 0 (RTX 3060), then write doc 208.
# Each probe: freeze backbone, overfit target head on 200 clips × 2000 steps.
# If loss → 0 but eval metric stays 0 → eval harness bug (fix before training).
set -euo pipefail

ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
PROBE="$ROOT/scripts/overfit_probe.py"

export CUDA_VISIBLE_DEVICES=1  # RTX 3060
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

for HEAD in det act psr pose; do
    echo ""
    echo "============================================================"
    echo "=== $(date): OVERFIT PROBE --head $HEAD ==="
    echo "============================================================"
    python3 "$PROBE" \
        --head "$HEAD" \
        --n-clips 200 \
        --steps 2000 \
        --lr 1e-3 \
        --log-every 100 \
        --output "$ROOT/src/runs/overfit_${HEAD}_v2_results.json"
    echo "=== $(date): $HEAD probe complete ==="
done

echo ""
echo "=== $(date): All 4 overfit probes complete ==="
echo "Results: $ROOT/src/runs/overfit_*_v2_results.json"
