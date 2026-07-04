#!/bin/bash
# D1: YOLOv8m eval on IndustReal validation set.
set -e
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
OUT_DIR="src/runs/rf_stages/checkpoints/d1_yolov8m"
mkdir -p "$OUT_DIR"

# Verify weights exist
WEIGHTS=""
if [ -f "weights/yolov8m_industreal.pt" ]; then
  WEIGHTS="weights/yolov8m_industreal.pt"
  echo "[D1] Using IndustReal-trained YOLOv8m"
elif [ -f "weights/yolov8m.pt" ]; then
  WEIGHTS="weights/yolov8m.pt"
  echo "[D1] Using COCO-pretrained yolov8m.pt (FALLBACK)"
else
  echo "[ERROR] No YOLOv8m weights found. Run scripts/download_yolov8m_industreal.sh first."
  exit 1
fi

# Run 10-frame class-mapping sanity check
echo "[D1] Running 10-frame class-mapping sanity check..."
CUDA_VISIBLE_DEVICES=0 python3 -u src/evaluation/yolov8m_sanity_check.py \
  --weights "$WEIGHTS" \
  --num_samples 10 \
  --out_path "$OUT_DIR/sanity_check.json"

# Full eval
echo "[D1] Starting full eval at $(date)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 10 \
  python3 -u src/evaluation/eval_yolov8m_full.py \
    --weights "$WEIGHTS" \
    --out_path "$OUT_DIR/metrics.json" \
    --num_classes 24 2>&1 | tee "$OUT_DIR/run.log"
echo "[D1] Complete at $(date)"
ls -la "$OUT_DIR/"
