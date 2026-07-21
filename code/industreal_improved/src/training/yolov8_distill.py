"""YOLOv8 distillation wrapper for MTL detection head.

Uses YOLOv8 as a teacher to provide soft labels for the MTL detection head.
The distillation loss is added to the standard detection loss, while activity,
pose, and PSR heads train with hard labels as before.

Design:
- YOLOv8 runs on each batch's RGB image (3-channel, 640x640)
- Produces: boxes (N, 4) in xyxy, scores (N,), classes (N,) in [0, 24)
- For each MTL anchor location, find which YOLOv8 box covers it
- Use YOLOv8's class prediction as soft label
- Distillation loss = weighted KL divergence between MTL predictions and YOLOv8 soft labels

This module is imported by train_mtl_v3_distill.py.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class YOLOv8Distiller:
    """Wraps YOLOv8 to produce soft labels for MTL detection distillation."""

    def __init__(
        self,
        weights_path: str = '/home/newadmin/swarm-bot/master/POPW/datasets/industreal/assembly_state_detection_model_weights/asd_best_IndustRealandSynthetic.pt',
        device: str = 'cuda',
        conf_threshold: float = 0.05,
    ):
        from ultralytics import YOLO
        logger.info(f'Loading YOLOv8 from {weights_path}...')
        self.yolo = YOLO(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        # Move to device
        self.yolo.to(device)
        self.yolo.model.eval()
        # Disable autograd
        for p in self.yolo.model.parameters():
            p.requires_grad = False
        logger.info(f'YOLOv8 loaded: {len(self.yolo.names)} classes')

    @torch.no_grad()
    def get_soft_labels(
        self,
        images_rgb: torch.Tensor,
        img_w: int = 640,
        img_h: int = 360,
    ) -> List[Dict]:
        """Run YOLOv8 on a batch of RGB images, return per-frame soft labels.

        Args:
            images_rgb: [B, 3, H, W] RGB images normalized for MTL model
            img_w, img_h: MTL model image dimensions

        Returns:
            List of B dicts, each with:
              'boxes': [N, 4] in (cx, cy, w, h) normalized [0,1]
              'classes': [N] class indices in [0, 24)
              'scores': [N] confidence scores in [0, 1]
        """
        from PIL import Image
        B = images_rgb.shape[0]
        results_list = []

        # Convert MTL-normalized images back to RGB uint8 for YOLOv8
        # MTL normalization: (img/255 - 0.45) / 0.225
        # Inverse: img_norm * 0.225 + 0.45, then * 255
        images_rgb_f = images_rgb * 0.225 + 0.45
        images_rgb_u8 = (images_rgb_f.clamp(0, 1) * 255).byte().cpu().numpy()
        # Shape: [B, 3, H, W] -> [B, H, W, 3] for YOLOv8
        images_rgb_hwc = images_rgb_u8.transpose(0, 2, 3, 1)

        for b in range(B):
            img = images_rgb_hwc[b]  # [H, W, 3] uint8
            # YOLOv8 requires PIL Image input (numpy arrays don't preprocess correctly)
            try:
                pil_img = Image.fromarray(img)
                result = self.yolo(pil_img, verbose=False, conf=self.conf_threshold)
            except Exception as e:
                logger.warning(f'YOLOv8 inference failed: {e}')
                results_list.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'classes': np.zeros((0,), dtype=np.int64),
                    'scores': np.zeros((0,), dtype=np.float32),
                })
                continue

            if result[0].boxes is None or len(result[0].boxes) == 0:
                results_list.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'classes': np.zeros((0,), dtype=np.int64),
                    'scores': np.zeros((0,), dtype=np.float32),
                })
                continue

            boxes_xyxy = result[0].boxes.xyxy.cpu().numpy()  # [N, 4] in pixel coords
            classes = result[0].boxes.cls.cpu().numpy().astype(np.int64)  # [N]
            scores = result[0].boxes.conf.cpu().numpy().astype(np.float32)  # [N]

            # Convert to (cx, cy, w, h) normalized
            x1, y1, x2, y2 = boxes_xyxy.T
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            boxes_cxcywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

            results_list.append({
                'boxes': boxes_cxcywh,
                'classes': classes,
                'scores': scores,
            })

        return results_list


def distill_loss(
    cls_logits: torch.Tensor,  # [B, 24, H, W]
    reg_preds: torch.Tensor,    # [B, 4*A, H, W]
    anchors: torch.Tensor,      # [H, W, A, 4] in (cx, cy, w, h)
    soft_labels: List[Dict],
    img_w: int = 640,
    img_h: int = 360,
    distill_weight: float = 1.0,
    score_thresh: float = 0.3,
) -> torch.Tensor:
    """Compute distillation loss from YOLOv8 soft labels.

    For each MTL anchor location, find which YOLOv8 box covers it.
    Apply soft cross-entropy loss on cls_logits based on YOLOv8's class.

    Args:
        cls_logits: [B, 24, H, W] classification logits
        reg_preds: [B, 4*A, H, W] regression predictions (not used for distill)
        anchors: [H, W, A, 4] in (cx, cy, w, h) normalized
        soft_labels: List of B dicts from YOLOv8Distiller.get_soft_labels()
        img_w, img_h: Image dimensions
        distill_weight: Weight for distillation loss term
        score_thresh: Only use YOLOv8 predictions above this score

    Returns:
        Scalar distillation loss
    """
    B = cls_logits.shape[0]
    H, W = cls_logits.shape[2], cls_logits.shape[3]
    A = anchors.shape[2]
    device = cls_logits.device

    total_loss = torch.tensor(0.0, device=device)
    n_total = 0

    for b in range(B):
        sl = soft_labels[b]
        if len(sl['boxes']) == 0:
            continue

        boxes = torch.from_numpy(sl['boxes']).to(device)  # [N, 4]
        classes = torch.from_numpy(sl['classes']).to(device)  # [N]
        scores = torch.from_numpy(sl['scores']).to(device)  # [N]

        # Filter by score
        keep = scores > score_thresh
        if keep.sum() == 0:
            continue
        boxes = boxes[keep]
        classes = classes[keep]
        scores = scores[keep]

        # For each YOLOv8 box, find which anchor locations overlap
        # boxes: [N, 4] in (cx, cy, w, h)
        # anchors: [H, W, A, 4] in (cx, cy, w, h)
        anchors_flat = anchors.reshape(-1, 4)  # [H*W*A, 4]

        # IoU between YOLOv8 boxes and anchors
        box_xyxy = torch.stack([
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ], dim=1)  # [N, 4]

        anc_xyxy = torch.stack([
            anchors_flat[:, 0] - anchors_flat[:, 2] / 2,
            anchors_flat[:, 1] - anchors_flat[:, 3] / 2,
            anchors_flat[:, 0] + anchors_flat[:, 2] / 2,
            anchors_flat[:, 1] + anchors_flat[:, 3] / 2,
        ], dim=1)  # [H*W*A, 4]

        # Compute IoU matrix [N, H*W*A]
        inter_x1 = torch.max(box_xyxy[:, None, 0], anc_xyxy[None, :, 0])
        inter_y1 = torch.max(box_xyxy[:, None, 1], anc_xyxy[None, :, 1])
        inter_x2 = torch.min(box_xyxy[:, None, 2], anc_xyxy[None, :, 2])
        inter_y2 = torch.min(box_xyxy[:, None, 3], anc_xyxy[None, :, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        area_box = boxes[:, 2] * boxes[:, 3]
        area_anc = anchors_flat[:, 2] * anchors_flat[:, 3]
        union = area_box[:, None] + area_anc[None, :] - inter
        iou = inter / union.clamp(min=1e-6)  # [N, H*W*A]

        # For each YOLOv8 box, find best matching anchor
        best_anc = iou.argmax(dim=1)  # [N]
        best_iou = iou.max(dim=1).values  # [N]

        # Filter: only use anchors with IoU > 0.3 with the YOLOv8 box
        valid = best_iou > 0.3
        if valid.sum() == 0:
            continue

        valid_anc_idx = best_anc[valid]  # [M]
        valid_classes = classes[valid]  # [M]
        valid_scores = scores[valid]  # [M]

        # For each matched anchor, apply soft cross-entropy loss
        # cls_logits shape: [B, 24, H, W]
        # Reshape to [B, H*W, 24]
        cls_flat = cls_logits[b].permute(1, 2, 0).reshape(-1, 24)  # [H*W, 24]
        # Replicate per anchor: [H*W*A, 24]
        cls_per_anc = cls_flat.repeat_interleave(A, dim=0)  # [H*W*A, 24]

        # Get logits for matched anchors
        matched_logits = cls_per_anc[valid_anc_idx]  # [M, 24]

        # Soft target: one-hot with score as confidence
        # target = score * one_hot(class) + (1 - score) / 24 * ones
        M = matched_logits.shape[0]
        num_classes = matched_logits.shape[1]

        # Soft cross-entropy
        # Use temperature T=1 for simplicity
        # Each anchor's target = score*one_hot(class) + (1-score)/(num_classes-1)*ones
        # Vectorized to preserve gradient
        target_dist = torch.full(
            (M, num_classes),
            1.0 / num_classes,
            device=device,
        )
        # For matched anchors: target = score * one_hot + (1-score) * uniform
        # = score * (one_hot - uniform) + uniform
        target_dist = target_dist + valid_scores.unsqueeze(1) * (
            F.one_hot(valid_classes, num_classes).float() - target_dist
        )
        log_probs = F.log_softmax(matched_logits, dim=1)
        # KL(target || pred) but in soft cross-entropy form
        loss_per_anchor = -(target_dist * log_probs).sum(dim=1)  # [M]
        total_loss = total_loss + loss_per_anchor.sum()
        n_total += M

    if n_total == 0:
        # No YOLOv8 detections matched - return a zero loss connected to cls_logits
        # so backward() still works.
        return (cls_logits.sum() * 0.0) * distill_weight

    return (total_loss / n_total) * distill_weight