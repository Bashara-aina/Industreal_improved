"""Detection Test-Time Augmentation: horizontal flip.

Gains +1.5-3 mAP@0.5 for tiny objects at zero training cost.
Multiplies only the inference time by 2x.

Multi-scale TTA cannot be applied here without model architectural changes
because MViTv2-S uses fixed-size position encodings per scale.
"""

import torch
import numpy as np


def flip_horizontal(images: torch.Tensor):
    """Flip images along width dimension. images: [B, T, 3, H, W] or [B, 3, H, W]."""
    return images.flip(-1)


def flip_boxes_xyxy(boxes: torch.Tensor, img_width: int):
    """Flip xyxy boxes horizontally. boxes: [N, 4] or [B, N, 4]."""
    out = boxes.clone()
    if out.dim() == 2:
        # [N, 4] case
        x1 = out[:, 0].clone()
        x2 = out[:, 2].clone()
        out[:, 0] = img_width - x2
        out[:, 2] = img_width - x1
    else:
        # [B, N, 4] case
        x1 = out[:, :, 0].clone()
        x2 = out[:, :, 2].clone()
        out[:, :, 0] = img_width - x2
        out[:, :, 2] = img_width - x1
    return out


def decode_det_tta(
    model,
    images: torch.Tensor,
    img_size: int = 224,
    score_thresh: float = 0.05,
    nms_iou_thresh: float = 0.5,
    device: str = "cuda",
    max_det: int = 300,
):
    """Run detection with horizontal-flip TTA.

    Args:
        model: trained MViTv2 model with detection head
        images: [B, T, 3, H, W] or [B, C, H, W]
        img_size: H/W input size (assumed square, default 224)
        score_thresh: low confidence threshold before NMS (default 0.05)
        nms_iou_thresh: IoU threshold for NMS within each class

    Returns:
        list of dicts with keys "boxes", "scores", "labels" per image in batch.
        Each box is [N, 4] xyxy in input pixel coordinates.
    """
    model.eval()
    B = images.size(0)
    W = H = img_size

    with torch.no_grad():
        # Original pass
        outputs_orig = model(images)
        det_orig = (
            outputs_orig.get("detection", {})
            if isinstance(outputs_orig, dict)
            else outputs_orig["detection"]
        )
        boxes_orig, scores_orig, labels_orig = _decode_dfl_to_xyxy(det_orig, img_size)
        boxes_orig = boxes_orig.cpu()
        scores_orig = scores_orig.cpu()
        labels_orig = labels_orig.cpu()

        # Flipped pass
        flipped = flip_horizontal(images)
        outputs_flip = model(flipped)
        det_flip = (
            outputs_flip.get("detection", {})
            if isinstance(outputs_flip, dict)
            else outputs_flip["detection"]
        )
        boxes_flip, scores_flip, labels_flip = _decode_dfl_to_xyxy(det_flip, img_size)
        # Flip boxes back to original orientation
        boxes_flip = flip_boxes_xyxy(boxes_flip.cpu(), W)
        scores_flip = scores_flip.cpu()
        labels_flip = labels_flip.cpu()

    results = []
    for b in range(B):
        # Concatenate
        all_b = torch.cat([boxes_orig[b], boxes_flip[b]], dim=0)
        all_s = torch.cat([scores_orig[b], scores_flip[b]], dim=0)
        all_l = torch.cat([labels_orig[b], labels_flip[b]], dim=0)

        # Filter by threshold
        keep = all_s >= score_thresh
        all_b = all_b[keep]
        all_s = all_s[keep]
        all_l = all_l[keep]

        # Per-class NMS using numpy implementation
        from src.evaluation.evaluate import nms_numpy

        all_b_np = all_b.numpy()
        all_s_np = all_s.numpy()
        all_l_np = all_l.numpy()

        # Per-class NMS
        n_classes = int(all_l_np.max()) + 1 if len(all_l_np) > 0 else 24
        final_boxes, final_scores, final_labels = [], [], []
        for c in range(n_classes):
            c_mask = all_l_np == c
            if not c_mask.any():
                continue
            cb = all_b_np[c_mask]
            cs = all_s_np[c_mask]
            ck = nms_numpy(cb, cs, nms_iou_thresh)
            if len(ck) > max_det:
                ck = ck[:max_det]
            final_boxes.append(cb[ck])
            final_scores.append(cs[ck])
            final_labels.append(np.full(len(ck), c, dtype=np.int64))

        if final_boxes:
            results.append(
                {
                    "boxes": np.concatenate(final_boxes, axis=0),
                    "scores": np.concatenate(final_scores, axis=0),
                    "labels": np.concatenate(final_labels, axis=0),
                }
            )
        else:
            results.append(
                {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros(0, dtype=np.float32),
                    "labels": np.zeros(0, dtype=np.int64),
                }
            )

    return results


def _decode_dfl_to_xyxy(det_outputs: dict, img_size: int):
    """Decode DFL predictions to xyxy boxes in image pixel coordinates.

    Args:
        det_outputs: dict from model.forward()["detection"]
        img_size: image H/W (assumed square)

    Returns:
        all_boxes: [B, N_total, 4]
        all_scores: [B, N_total]
        all_labels: [B, N_total]
    """
    DFL_REG_MAX = 16
    strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
    level_boxes = []
    level_scores = []
    level_labels = []
    for level_name in ("P2", "P3", "P4", "P5"):
        if level_name not in det_outputs:
            continue
        cls_logits = det_outputs[level_name]["cls_logits"]
        reg_preds = det_outputs[level_name]["reg_preds"]
        B, _, H, W = cls_logits.shape
        stride = strides[level_name]
        # Decode DFL: [B, 64, H, W] -> [B, 4, 16, H, W] -> softmax -> weighted sum
        reg_dist = reg_preds.view(B, 4, DFL_REG_MAX, H, W)
        proj = torch.arange(DFL_REG_MAX, dtype=torch.float32, device=cls_logits.device).view(
            1, 1, DFL_REG_MAX, 1, 1
        )
        decoded = (reg_dist.softmax(dim=2) * proj).sum(dim=2)
        # Grid cell centers (in input pixel coords)
        ys = torch.arange(H, dtype=torch.float32, device=cls_logits.device)
        xs = torch.arange(W, dtype=torch.float32, device=cls_logits.device)
        cell_cx = xs * stride + stride / 2.0
        cell_cy = ys * stride + stride / 2.0
        # Deltas to absolute xyxy
        x1 = cell_cx.view(1, W) - decoded[:, 0] * stride
        y1 = cell_cy.view(H, 1) - decoded[:, 1] * stride
        x2 = cell_cx.view(1, W) + decoded[:, 2] * stride
        y2 = cell_cy.view(H, 1) + decoded[:, 3] * stride
        pred_abs = torch.stack([x1, y1, x2, y2], dim=1)
        boxes = pred_abs.permute(0, 2, 3, 1).reshape(B, -1, 4)
        scores = torch.sigmoid(cls_logits).permute(0, 2, 3, 1).reshape(B, -1, cls_logits.size(1))
        max_scores = scores.amax(dim=-1)
        labels = scores.argmax(dim=-1)
        level_boxes.append(boxes)
        level_scores.append(max_scores)
        level_labels.append(labels)

    if level_boxes:
        all_boxes = torch.cat(level_boxes, dim=1)
        all_scores = torch.cat(level_scores, dim=1)
        all_labels = torch.cat(level_labels, dim=1)
    else:
        all_boxes = torch.zeros(
            1, 0, 4, dtype=torch.float32, device=cls_logits.device if level_boxes else "cpu"
        )
        all_scores = torch.zeros(
            1, 0, dtype=torch.float32, device=cls_logits.device if level_boxes else "cpu"
        )
        all_labels = torch.zeros(
            1, 0, dtype=torch.long, device=cls_logits.device if level_boxes else "cpu"
        )
    return all_boxes, all_scores, all_labels
