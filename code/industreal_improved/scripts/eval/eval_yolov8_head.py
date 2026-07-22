#!/usr/bin/env python3
"""Eval script for v3.19 with YOLOv8 DFL head.

Uses the model's built-in YOLOv8 decode + simple NMS for mAP calculation.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

from train_mtl_full_multimodal import (
    FullMultiModalDataset,
    expand_conv_proj_to_9ch,
    WrappedMTL,
    collate_real_targets,
)
from src.models.mvit_mtl_model import MTLMViTModel
from src.models.yolov8_det_head import init_from_yolov8_weights
from src.evaluation.evaluate import compute_ap_per_class
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('eval_yolov8')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_for_eval(batch):
    """Collate for eval: images + (boxes in xyxy normalized, classes)."""
    images = torch.stack([b[0] for b in batch])  # [B, 9, H, W]
    targets = [b[1] for b in batch]
    # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) normalized
    gt_xyxy = []
    gt_classes = []
    for t in targets:
        boxes = t['boxes']
        classes = t['classes']
        if boxes.numel() > 0:
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            x1 = (cx - w / 2) * C.IMG_WIDTH
            y1 = (cy - h / 2) * C.IMG_HEIGHT
            x2 = (cx + w / 2) * C.IMG_WIDTH
            y2 = (cy + h / 2) * C.IMG_HEIGHT
            gt_xyxy.append(torch.stack([x1, y1, x2, y2], dim=1))
            gt_classes.append(classes)
        else:
            gt_xyxy.append(torch.zeros(0, 4))
            gt_classes.append(torch.zeros(0, dtype=torch.long))
    return images, gt_xyxy, gt_classes


def nms_pytorch(boxes, scores, iou_thresh=0.5):
    """Simple NMS."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-anchors', type=int, default=16)
    parser.add_argument('--max-frames', type=int, default=1000)
    parser.add_argument('--score-thresh', type=float, default=0.05)
    parser.add_argument('--nms-iou', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='/tmp/yolov8_det_eval.json')
    args = parser.parse_args()

    # Build model
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = MTLMViTModel(
        num_act_classes=75, num_det_classes=24, num_psr_components=11,
        num_anchors=args.num_anchors,
        use_yolov8_head=True,
    )
    expand_conv_proj_to_9ch(model)
    sd = ckpt['model_state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f'Loaded: missing={len(missing)}, unexpected={len(unexpected)}')
    model = WrappedMTL(model).to(DEVICE)
    model.eval()

    # Load val dataset
    val_ds = FullMultiModalDataset(
        recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/val',
        img_size=(640, 360),
        mosaic_prob=0.0,
        copy_paste_prob=0.0,
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            collate_fn=collate_real_targets, num_workers=0)

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    n_frames = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if n_frames >= args.max_frames:
                break

            # Get GT in xyxy format
            gt_xyxy = []
            gt_classes = []
            # targets is dict {boxes: [T1, T2], classes: [C1, C2], ...}
            if isinstance(targets, dict) and 'boxes' in targets:
                target_boxes_list = targets['boxes']
                target_classes_list = targets['classes']
            else:
                target_boxes_list = [t['boxes'] for t in targets]
                target_classes_list = [t['classes'] for t in targets]

            for boxes, classes in zip(target_boxes_list, target_classes_list):
                if boxes.numel() > 0:
                    cx = boxes[:, 0]
                    cy = boxes[:, 1]
                    w = boxes[:, 2]
                    h = boxes[:, 3]
                    x1 = (cx - w / 2) * C.IMG_WIDTH
                    y1 = (cy - h / 2) * C.IMG_HEIGHT
                    x2 = (cx + w / 2) * C.IMG_WIDTH
                    y2 = (cy + h / 2) * C.IMG_HEIGHT
                    gt_xyxy.append(torch.stack([x1, y1, x2, y2], dim=1).to(DEVICE))
                    gt_classes.append(classes.to(DEVICE))
                else:
                    gt_xyxy.append(torch.zeros(0, 4, device=DEVICE))
                    gt_classes.append(torch.zeros(0, dtype=torch.long, device=DEVICE))

            # Forward
            x = images.to(DEVICE).float().unsqueeze(2)
            mean = torch.tensor([0.45] * 9).view(1, 9, 1, 1, 1).to(DEVICE)
            std = torch.tensor([0.225] * 9).view(1, 9, 1, 1, 1).to(DEVICE)
            x = (x - mean) / std

            out = model(x)
            det_out = out['detection']

            # Decode YOLOv8 outputs
            level_boxes, level_scores = model.m.det_head.decode(det_out)

            B = x.shape[0]
            for b in range(B):
                img_boxes = []
                img_scores = []
                img_labels = []

                for level_idx, stride in enumerate([8.0, 16.0, 32.0]):
                    boxes = level_boxes[level_idx][b]  # [H*W, 4]
                    scores = level_scores[level_idx][b]  # [H*W, 24]

                    # Flatten and threshold
                    boxes_flat = boxes  # already [H*W, 4]
                    scores_flat = scores  # already [H*W, 24]
                    max_scores, max_classes = scores_flat.max(dim=1)
                    keep = max_scores > args.score_thresh
                    boxes_kept = boxes_flat[keep]
                    scores_kept = max_scores[keep]
                    classes_kept = max_classes[keep]

                    img_boxes.append(boxes_kept)
                    img_scores.append(scores_kept)
                    img_labels.append(classes_kept)

                if img_boxes:
                    img_boxes = torch.cat(img_boxes, dim=0)
                    img_scores = torch.cat(img_scores, dim=0)
                    img_labels = torch.cat(img_labels, dim=0)

                    # Per-class NMS
                    final_boxes = []
                    final_scores = []
                    final_labels = []
                    for c in range(24):
                        cm = img_labels == c
                        if cm.sum() == 0:
                            continue
                        keep = nms_pytorch(img_boxes[cm], img_scores[cm], args.nms_iou)
                        final_boxes.append(img_boxes[cm][keep])
                        final_scores.append(img_scores[cm][keep])
                        final_labels.append(torch.full((len(keep),), c, dtype=torch.long, device=DEVICE))

                    if final_boxes:
                        all_pred_boxes.append(torch.cat(final_boxes).cpu().numpy())
                        all_pred_scores.append(torch.cat(final_scores).cpu().numpy())
                        all_pred_labels.append(torch.cat(final_labels).cpu().numpy())
                    else:
                        all_pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
                        all_pred_scores.append(np.zeros(0, dtype=np.float32))
                        all_pred_labels.append(np.zeros(0, dtype=np.int64))
                else:
                    all_pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
                    all_pred_scores.append(np.zeros(0, dtype=np.float32))
                    all_pred_labels.append(np.zeros(0, dtype=np.int64))

                all_gt_boxes.append(gt_xyxy[b].cpu().numpy())
                all_gt_labels.append(gt_classes[b].cpu().numpy())
                n_frames += 1

            if n_frames % 100 == 0:
                logger.info(f'  {n_frames} frames in {time.time()-t0:.0f}s')

    logger.info(f'Total frames: {n_frames}, total preds: {sum(len(b) for b in all_pred_boxes)}')

    # mAP@0.5
    af_indices = [i for i in range(len(all_gt_boxes)) if len(all_gt_boxes[i]) > 0]
    af_boxes = [all_pred_boxes[i] for i in af_indices]
    af_scores = [all_pred_scores[i] for i in af_indices]
    af_labels = [all_pred_labels[i] for i in af_indices]
    af_gtb = [all_gt_boxes[i] for i in af_indices]
    af_gtl = [all_gt_labels[i] for i in af_indices]

    result = compute_ap_per_class(af_boxes, af_scores, af_labels, af_gtb, af_gtl, iou_thresh=0.5)
    mAP = float(result['mAP'])

    output = {
        'checkpoint': args.checkpoint,
        'n_frames': n_frames,
        'n_preds': sum(len(b) for b in all_pred_boxes),
        'mAP50': mAP,
        'per_class_AP': {str(k): float(v) for k, v in result['per_class_ap'].items()},
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f'mAP@0.5: {mAP:.4f}')
    logger.info(f'Per-class: {output["per_class_AP"]}')


if __name__ == '__main__':
    main()