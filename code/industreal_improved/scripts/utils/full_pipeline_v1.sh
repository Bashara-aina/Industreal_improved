#!/bin/bash
# full_pipeline_v1.sh — Sequenced ST baselines → MTL with all 6 levers active.
# GPU 1: ST baselines (det → act → psr), each capped at 8000 batches/ep × 50ep.
# GPU 0: MTL launched AFTER all 4 ST checkpoints exist → full warm-start + distillation.
# All 6 levers from the 207 extra-lever document active at MTL launch time.
set -euo pipefail

ROOT="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
ST_DIR="$ROOT/src/runs/st_checkpoints"
ST_SCRIPT="$ROOT/scripts/train_st.py"
MTL_SCRIPT="$ROOT/scripts/train_mtl_mvit.py"
BUILD_SOUP="$ROOT/scripts/build_soup.py"

# ── Phase 1: ST baselines on GPU 1 (det, act, psr) ─────────────────────────
# Pose already exists at $ST_DIR/st_pose_best.pt

for TASK in det act psr; do
    echo ""
    echo "============================================================"
    echo "=== $(date): PHASE 1 — ST-${TASK} baseline ==="
    echo "============================================================"

    export CUDA_VISIBLE_DEVICES=0
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=4
    python3 "$ST_SCRIPT" \
        --task "$TASK" \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-4 \
        --eval-every 5 \
        --output-dir "$ROOT/src/runs/st_${TASK}" \
        --max-batches-per-epoch 8000 \
        --num-workers 0

    # Symlink best checkpoint for MTL warm-start + distillation
    SRC="$ROOT/src/runs/st_${TASK}/best.pt"
    DST="$ST_DIR/st_${TASK}_best.pt"
    if [ -f "$SRC" ]; then
        rm -f "$DST"
        ln -sf "$SRC" "$DST"
        echo "=== $(date): ST-${TASK} complete. Symlinked $DST ==="
    else
        echo "=== $(date): FATAL: ST-${TASK} best.pt not found at $SRC ==="
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "=== $(date): All 3 ST baselines complete ==="
echo "============================================================"

# Verify all 4 ST checkpoints exist
for TASK in det act psr pose; do
    CKPT="$ST_DIR/st_${TASK}_best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "FATAL: missing $CKPT"
        ls -la "$ST_DIR/"
        exit 1
    fi
done
echo "All 4 ST checkpoints confirmed."

# ── Phase 2: Build model soup backbone from ST specialists ────────────────
echo ""
echo "============================================================"
echo "=== $(date): PHASE 2 — Build model soup backbone ==="
echo "============================================================"
CLEANUP_OPTIONS="--opt-level 0"
SOUP_PATH="$ST_DIR/soup_backbone.pt"
python3 "$BUILD_SOUP" \
    --det "$ST_DIR/st_det_best.pt" \
    --act "$ST_DIR/st_act_best.pt" \
    --psr "$ST_DIR/st_psr_best.pt" \
    --pose "$ST_DIR/st_pose_best.pt" \
    --output "$SOUP_PATH"

# Copy soup to MTL output dir for auto-load
MTL_OUT="$ROOT/src/runs/mtl_all6_v1"
mkdir -p "$MTL_OUT"
cp "$SOUP_PATH" "$MTL_OUT/soup_backbone.pt"
echo "Soup backbone saved: $MTL_OUT/soup_backbone.pt"

# ── Phase 3: MTL training with ALL 6 levers on GPU 0 ──────────────────────
echo ""
echo "============================================================"
echo "=== $(date): PHASE 3 — MTL with all 6 levers ==="
echo "===  Levers active: budget, SWA, warm-start, distillation ==="
echo "===                 monotonicity (eval), threshold (config) ==="
echo "============================================================"

export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
python3 "$MTL_SCRIPT" \
    --epochs 50 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --max-batches-per-epoch 0 \
    --eval-every 5 \
    --num-workers 0 \
    --warm-start-dir "$ST_DIR" \
    --distill-teacher-dir "$ST_DIR" \
    --distill-alpha 0.1 \
    --distill-temperature 4.0 \
    --swa-checkpoints 5 \
    --output-dir "$MTL_OUT"

echo ""
echo "============================================================"
echo "=== $(date): PIPELINE COMPLETE ==="
echo "===  Output: $MTL_OUT ==="
echo "============================================================"
