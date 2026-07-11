#!/bin/bash
# Train 3 missing ST baselines sequentially on GPU 1 (RTX 5060 Ti 16GB).
# After each completes, symlinks best.pt → st_checkpoints/ for MTL warm-start.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
ST_DIR="$ROOT/src/runs/st_checkpoints"
SCRIPT="$ROOT/scripts/train_st.py"

for TASK in det act psr; do
    echo "=== $(date): Starting ST-$TASK baseline ==="
    python "$SCRIPT" \
        --task "$TASK" \
        --epochs 50 \
        --batch-size 2 \
        --lr 3e-4 \
        --eval-every 5 \
        --output-dir "$ROOT/src/runs/st_${TASK}" \
        --max-batches-per-epoch 8000 \
        --num-workers 0

    # Symlink best checkpoint for MTL warm-start
    SRC="$ROOT/src/runs/st_${TASK}/best.pt"
    DST="$ST_DIR/st_${TASK}_best.pt"
    if [ -f "$SRC" ]; then
        ln -sf "$SRC" "$DST"
        echo "=== $(date): ST-$TASK complete. Symlinked $DST ==="
    else
        echo "=== $(date): WARNING: ST-$TASK best.pt not found at $SRC ==="
    fi
done

echo "=== $(date): All 3 ST baselines complete ==="
