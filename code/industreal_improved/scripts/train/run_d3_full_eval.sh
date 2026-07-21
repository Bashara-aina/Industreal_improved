#!/bin/bash
# D3: Full evaluation with per-frame prediction persistence.
set -e
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
CKPT="src/runs/rf_stages/checkpoints/epoch_11.pth"
OUT_DIR="src/runs/rf_stages/checkpoints/d3_full_eval"
mkdir -p "$OUT_DIR"

echo "[D3] Starting full eval on epoch 11 checkpoint at $(date)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 10 \
  python3 -u src/evaluation/subprocess_eval.py \
    --ckpt "$CKPT" \
    --out_path "$OUT_DIR/metrics.json" \
    --predictions_path "$OUT_DIR/per_frame_predictions.json" \
    --EVAL_MAX_BATCHES 0 \
    --persist_predictions 2>&1 | tee "$OUT_DIR/run.log"
echo "[D3] Complete at $(date)"
ls -la "$OUT_DIR/"
