#!/bin/bash
# Download YOLOv8m weights for IndustReal.
# Try official repo first, fall back to Ultralytics COCO weights.
set -e
DEST_DIR="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/weights"
mkdir -p "$DEST_DIR"

# Try official IndustReal weights
INDUSTREAL_URL="https://github.com/microsoft/IndustReal/releases/download/v1.0/yolov8m_industreal.pt"
if curl -fL --connect-timeout 10 "$INDUSTREAL_URL" -o "$DEST_DIR/yolov8m_industreal.pt" 2>/dev/null; then
  echo "[OK] Downloaded IndustReal YOLOv8m weights from official source"
  ls -la "$DEST_DIR/yolov8m_industreal.pt"
else
  echo "[FALLBACK] IndustReal weights not available, downloading Ultralytics yolov8m.pt (COCO pretrained)"
  pip install ultralytics --quiet
  python3 -c "from ultralytics import YOLO; m = YOLO('yolov8m.pt'); print('[OK] yolov8m.pt downloaded')"
  cp yolov8m.pt "$DEST_DIR/" || echo "[WARN] Copy failed, but yolov8m.pt is in working dir"
fi
