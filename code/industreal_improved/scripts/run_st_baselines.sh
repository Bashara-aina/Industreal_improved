#!/bin/bash
# Run ST baselines per Doc 207 §9.3 — corrected launch order.
# ST-pose first (fastest baseline + paper headline), then act, det, psr.
# After each, symlink best.pt to st_checkpoints/ for MTL warm-start.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
ST_DIR="$ROOT/src/runs/st_checkpoints"
SCRIPT="$ROOT/scripts/train_st.py"

mkdir -p "$ST_DIR"

# [Doc 207 §9.3] Launch order: pose -> act -> det -> psr
# pose is first: fastest baseline and the paper's headline positive-transfer candidate.
for TASK in pose act det psr; do
    echo "=== $(date): Starting ST-${TASK} baseline (50 epochs, 8k batch/ep) ==="

    EXTRA=""
    if [ "$TASK" = "pose" ]; then
        # Use 6D + geodesic rotation loss for the headline pose number
        EXTRA="--pose-geodesic"
    fi
    if [ "$TASK" = "det" ]; then
        # Detection augmentation ON (default, explicit for clarity)
        EXTRA="--det-aug"
    fi

    python "$SCRIPT" \
        --task "$TASK" \
        --epochs 50 \
        --batch-size 2 \
        --lr 3e-4 \
        --eval-every 5 \
        --output-dir "$ROOT/src/runs/st_${TASK}" \
        --max-batches-per-epoch 8000 \
        --num-workers 0 \
        $EXTRA

    # Symlink best checkpoint for MTL warm-start
    SRC="$ROOT/src/runs/st_${TASK}/best.pt"
    DST="$ST_DIR/st_${TASK}_best.pt"
    if [ -f "$SRC" ]; then
        ln -sf "$SRC" "$DST"
        echo "=== $(date): ST-${TASK} complete. Symlinked $DST ==="
    else
        echo "=== $(date): WARNING: ST-${TASK} best.pt not found at $SRC ==="
    fi
done

echo "=== $(date): All 4 ST baselines complete (pose -> act -> det -> psr) ==="
