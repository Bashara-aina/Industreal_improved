#!/bin/bash
# D4: YOLOv8m detections -> MonotonicDecoder -> PSR metrics.
set -e
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
OUT_DIR="src/runs/rf_stages/checkpoints/d4_yolov8m_psr"
mkdir -p "$OUT_DIR"

echo "[D4] Starting YOLOv8m -> PSR pipeline at $(date)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 10 \
  python3 -u src/evaluation/eval_yolov8m_psr.py \
    --yolo_weights "weights/yolov8m.pt" \
    --psr_decoder "src/runs/rf_stages/checkpoints/epoch_11.pth" \
    --out_path "$OUT_DIR/metrics.json" \
    --predictions_path "$OUT_DIR/per_frame_predictions.json" 2>&1 | tee "$OUT_DIR/run.log"
echo "[D4] Complete at $(date)"
ls -la "$OUT_DIR/"
