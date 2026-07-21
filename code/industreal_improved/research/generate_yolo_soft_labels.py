#!/usr/bin/env python3
"""Generate soft pseudo-labels from YOLOv8 teacher for distillation.

Loads the fine-tuned YOLOv8 model, runs inference on real train + synthetic data,
saves YOLOv8 predictions (boxes + class + confidence) in our MTL training format.
"""
import argparse, json, sys, logging
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yolo_soft_labels")


def generate_for_dir(teacher, img_dir, out_label_dir, conf_thresh=0.15):
    """Run YOLOv8 on all images in img_dir, save predictions to out_label_dir."""
    img_dir = Path(img_dir)
    out_label_dir = Path(out_label_dir)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    logger.info(f"Processing {len(images)} images from {img_dir}")

    n_total = 0
    n_high_conf = 0
    for batch_start in range(0, len(images), 32):
        batch = images[batch_start: batch_start + 32]
        results = teacher.predict([str(p) for p in batch], conf=conf_thresh, imgsz=640,
                                  verbose=False, device='0', batch=32)
        for img_path, r in zip(batch, results):
            n_total += 1
            label_path = out_label_dir / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                img_w, img_h = 1280, 720
                for box in r.boxes:
                    cls = int(box.cls)
                    # Map synth class id (1-22) to our model class id (1-22, +1 = synth)
                    # YOLOv8 model: 0=background, 1-22=states, 23=error
                    # For real data: GT is cat_id 1-23, we map to 0-22
                    # YOLOv8 output: 0=background, 1-22=states (no error_state)
                    # So direct match works
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    score = float(box.conf)
                    # Format: class_id cx cy w h score (one extra column for score)
                    f.write(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.4f}\n')
                    if score > 0.3:
                        n_high_conf += 1
    logger.info(f"  -> {n_high_conf} high-conf detections from {n_total} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str,
                        default='runs/yolov8_finetune/real_only_from_synth_pretrained/weights/best.pt',
                        help='Path to YOLOv8 teacher weights')
    parser.add_argument('--synth-img-dir', type=str,
                        default='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/synth_yolo/images')
    parser.add_argument('--synth-out-dir', type=str,
                        default='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/synth_yolo/soft_labels')
    parser.add_argument('--real-img-dir', type=str,
                        default='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train_yolo/images')
    parser.add_argument('--real-out-dir', type=str,
                        default='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train_yolo/soft_labels')
    parser.add_argument('--conf', type=float, default=0.15)
    args = parser.parse_args()

    if not Path(args.teacher).exists():
        # Use the pretrained synthetic+real YOLOv8 if fine-tuned not ready
        args.teacher = 'asd_pretrained/asd_best_IndustRealandSynthetic.pt'
        logger.info(f"Using pretrained teacher: {args.teacher}")

    logger.info(f"Loading teacher: {args.teacher}")
    teacher = YOLO(args.teacher)

    if Path(args.synth_img_dir).exists():
        logger.info("Generating soft labels for SYNTHETIC data...")
        generate_for_dir(teacher, args.synth_img_dir, args.synth_out_dir, args.conf)

    if Path(args.real_img_dir).exists():
        logger.info("Generating soft labels for REAL train data...")
        generate_for_dir(teacher, args.real_img_dir, args.real_out_dir, args.conf)

    logger.info("Done! Soft labels generated.")


if __name__ == '__main__':
    main()
