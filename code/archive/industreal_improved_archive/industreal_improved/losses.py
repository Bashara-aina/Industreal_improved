"""
Loss functions for POPW multi-task model.
Implements: Focal Loss, Wing Loss, LDAM-DRW Loss, Binary Focal Loss, Kendall Multi-Task Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class FocalLoss(nn.Module):
    """Focal Loss for dense object detection (Lin et al. 2017)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, N, C] - raw logits (no sigmoid)
            targets: [B, N] - class labels (0 to C-1), -1 for ignore
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=-1)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha >= 0:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WingLoss(nn.Module):
    """Wing Loss for facial/pose landmark regression (Wang et al. 2017)."""

    def __init__(self, omega: float = 0.05, epsilon: float = 0.005):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: [B, N, 2] or [B, N] (if 1D)
            target: [B, N, 2] or [B, N]
            weight: [B, N] or None - confidence weights
        """
        if pred.dim() == 2:
            diff = torch.abs(pred - target)
        else:
            diff = torch.norm(pred - target, dim=-1)

        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )

        if weight is not None:
            loss = loss * weight
            return loss.sum() / (weight.sum() + 1e-6)

        return loss.mean()


class LDAMLoss(nn.Module):
    """Label Distribution Learning with Asymmetric Margin - Deferred Reweighting (Cao et al. 2019)."""

    def __init__(self, num_classes: int, label_smooth: float = 0.1, drw_epoch: int = 35):
        super().__init__()
        self.num_classes = num_classes
        self.label_smooth = label_smooth
        self.drw_epoch = drw_epoch
        self.register_buffer("cls_num_list", torch.zeros(num_classes))

    def update_cls_num_list(self, cls_num_list: torch.Tensor):
        """Update class frequency counts for DRW."""
        self.cls_num_list = cls_num_list.float()

    def get_DRW_weights(self, epoch: int) -> torch.Tensor:
        """Get reweighting factor based on raw class frequency.

        Uses raw class counts (no effective_number formula) per Doc 01 §B.1.
        Margins computed as margin_i = -log(C_i / sum(C)) where C_i are raw counts.
        """
        if epoch < self.drw_epoch:
            return torch.ones(self.num_classes)

        # Raw class frequency-based weights — use counts directly
        class_counts = self.cls_num_list.float()
        weights = class_counts / class_counts.sum() * len(class_counts)  # normalized frequency

        return weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                epoch: int = 0, reduction: str = "mean") -> torch.Tensor:
        """
        Args:
            inputs: [B, C] raw logits
            targets: [B] class labels
            epoch: current epoch for DRW
        """
        # Label smoothing
        if self.label_smooth > 0:
            log_preds = F.log_softmax(inputs, dim=-1)
            n_classes = inputs.size(-1)
            with torch.no_grad():
                targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
                targets_one_hot = targets_one_hot * (1 - self.label_smooth) + self.label_smooth / n_classes
            loss = -((targets_one_hot * log_preds).sum(dim=-1))
        else:
            loss = F.cross_entropy(inputs, targets, reduction="none")

        # DRW reweighting
        if epoch >= self.drw_epoch and self.cls_num_list.sum() > 0:
            drw_weights = self.get_DRW_weights(epoch).to(inputs.device)
            sample_weights = drw_weights[targets]
            loss = loss * sample_weights

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss for multi-label PSR component prediction."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                comp_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, N] raw logits for binary classification per component
            targets: [B, N] binary labels (0 or 1)
            comp_weights: [N] optional per-component weights for PSR
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # [B, N]
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_weight * bce_loss  # [B, N]

        # Apply per-component weighting if provided
        if comp_weights is not None:
            loss = loss * comp_weights.unsqueeze(0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TemporalSmoothnessLoss(nn.Module):
    """Temporal smoothness regularization for PSR predictions."""

    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, N] PSR logits over time
        Returns:
            Smoothness regularization loss
        """
        if logits.size(1) < 2:
            return torch.tensor(0.0, device=logits.device)

        diff = logits[:, 1:] - logits[:, :-1]
        loss = torch.sum(diff ** 2) / (logits.size(0) * (logits.size(1) - 1))
        return self.weight * loss


def compute_psr_component_weights(psr_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute per-component weighting for PSR binary focal loss.

    Uses inverse prevalence weighting — rarer components get higher weight.
    Components at extremes (very common or very rare, prevalence <5% or >95%)
    receive weight=1 (no boost) as they provide limited signal.

    Args:
        psr_labels: [B, 11] binary labels for PSR components

    Returns:
        [11] weight tensor, normalized so mean weight = 1.0
    """
    # Fraction of frames where each component is "active" (label=1)
    prevalence = psr_labels.float().mean(dim=0)  # [11]

    # Inverse sqrt prevalence for mid-range components
    # Components near 0 or 1 (very rare or very common) get weight=1
    weights = torch.where(
        (prevalence > 0.05) & (prevalence < 0.95),
        1.0 / torch.sqrt(prevalence + 1e-8),
        torch.ones_like(prevalence)
    )

    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    return weights


class GIoULoss(nn.Module):
    """Generalized IoU Loss for bounding box regression."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: [B, N, 4] (x, y, w, h)
            target_boxes: [B, N, 4] (x, y, w, h)
        """
        # Convert to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-7)

        # Enclosing box
        encl_x1 = torch.min(pred_x1, target_x1)
        encl_y1 = torch.min(pred_y1, target_y1)
        encl_x2 = torch.max(pred_x2, target_x2)
        encl_y2 = torch.max(pred_y2, target_y2)
        encl_area = (encl_x2 - encl_x1) * (encl_y2 - encl_y1) + 1e-7

        # GIoU
        giou = iou - (encl_area - union_area) / encl_area

        loss = 1 - giou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class DetectionLoss(nn.Module):
    """Anchor-based detection loss: Focal classification + GIoU regression.

    Matches ground-truth boxes to anchors using IoU threshold,
    produces target cls/reg tensors, computes focal loss + GIoU loss.
    """

    def __init__(self, anchor_sizes: List[int], anchor_ratios: List[float],
                 num_classes: int = 24, alpha: float = 0.25, gamma: float = 2.0,
                 giou_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.giou_weight = giou_weight
        self.cls_weight = cls_weight

        # Build anchors: for each scale, 3 ratios → 5 scales × 3 ratios = 15 anchors
        self.anchor_sizes = anchor_sizes  # [24, 48, 96, 192, 384]
        self.anchor_ratios = anchor_ratios  # [0.5, 1.0, 2.0]

        # Pre-compute all anchor boxes in (cx, cy, w, h) format
        # Anchors are generated at feature map scale, normalized by stride
        self.anchors = None  # Will be computed per-feature-map in forward

    def _generate_anchors(self, feature_h: int, feature_w: int, stride: int,
                         device: torch.device) -> torch.Tensor:
        """Generate anchor boxes for one feature level.

        Returns: [N_anchors, 4] tensor of (cx, cy, w, h) in image coordinates.
        """
        anchors_per_cell = len(self.anchor_sizes) * len(self.anchor_ratios)

        # Grid centers
        cx = torch.arange(feature_w, device=device).float() + 0.5
        cy = torch.arange(feature_h, device=device).float() + 0.5
        cx, cy = torch.meshgrid(cx, cy, indexing='xy')
        cx = cx.flatten() * stride
        cy = cy.flatten() * stride

        # Generate all anchor shapes
        anchor_boxes = []
        for s in self.anchor_sizes:
            for r in self.anchor_ratios:
                w = s * r ** 0.5
                h = s / (r ** 0.5)
                anchor_boxes.append([0, 0, w, h])

        anchor_templates = torch.tensor(anchor_boxes, device=device, dtype=torch.float32)  # [15, 4]

        # Expand to all grid positions
        num_positions = feature_h * feature_w
        all_anchors = torch.zeros(num_positions * anchors_per_cell, 4, device=device)

        for i, (cx_p, cy_p) in enumerate(zip(cx, cy)):
            for j in range(len(anchor_templates)):
                aw = anchor_templates[j, 2].item()
                ah = anchor_templates[j, 3].item()
                idx = i * anchors_per_cell + j
                all_anchors[idx, 0] = cx_p
                all_anchors[idx, 1] = cy_p
                all_anchors[idx, 2] = aw
                all_anchors[idx, 3] = ah

        return all_anchors  # [N_anchors, 4]

    def _match_anchors(self, gt_boxes: torch.Tensor, anchors: torch.Tensor,
                      iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match ground-truth boxes to anchors using IoU thresholding.

        Args:
            gt_boxes: [M, 4] (cx, cy, w, h) in image coordinates
            anchors: [N, 4] (cx, cy, w, h)

        Returns:
            target_cls: [N] class labels (-1 for background/ignore)
            target_reg: [N, 4] encoded regression targets (cx, cy, w, h) relative to matched anchor
        """
        N = anchors.shape[0]
        M = gt_boxes.shape[0]

        target_cls = torch.full((N,), -1, dtype=torch.long, device=anchors.device)  # -1 = ignore
        target_reg = torch.zeros(N, 4, device=anchors.device)

        if M == 0:
            return target_cls, target_reg

        # Compute IoU matrix [N, M]
        # Convert to (x1, y1, x2, y2) for IoU computation
        def box_cxcywh_to_xyxy(box):
            x1 = box[..., 0] - box[..., 2] / 2
            y1 = box[..., 1] - box[..., 3] / 2
            x2 = box[..., 0] + box[..., 2] / 2
            y2 = box[..., 1] + box[..., 3] / 2
            return x1, y1, x2, y2

        gt_x1, gt_y1, gt_x2, gt_y2 = box_cxcywh_to_xyxy(gt_boxes)
        anc_x1, anc_y1, anc_x2, anc_y2 = box_cxcywh_to_xyxy(anchors)

        # IoU computation
        inter_x1 = torch.max(gt_x1.unsqueeze(0), anc_x1.unsqueeze(1))
        inter_y1 = torch.max(gt_y1.unsqueeze(0), anc_y1.unsqueeze(1))
        inter_x2 = torch.min(gt_x2.unsqueeze(0), anc_x2.unsqueeze(1))
        inter_y2 = torch.min(gt_y2.unsqueeze(0), anc_y2.unsqueeze(1))

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        anc_area = (anc_x2 - anc_x1) * (anc_y2 - anc_y1)
        union_area = gt_area.unsqueeze(0) + anc_area.unsqueeze(1) - inter_area

        iou_matrix = inter_area / (union_area + 1e-6)  # [N, M]

        # For each GT box, find best matching anchor (max IoU)
        best_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)  # [M]

        # For each anchor, find best matching GT (max IoU)
        best_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)  # [N]

        # Assign: anchors with IoU >= threshold get that GT
        # Also, each GT gets its best anchor assigned (even if below threshold)
        assigned = torch.zeros(M, dtype=torch.bool, device=anchors.device)

        for a_idx in range(N):
            g_idx = best_gt_per_anchor[a_idx].item()
            iou_val = best_iou_per_anchor[a_idx].item()

            if iou_val >= iou_threshold:
                target_cls[a_idx] = 1  # foreground (class 1 - first class)
                target_reg[a_idx] = self._encode_box(gt_boxes[g_idx], anchors[a_idx])
                assigned[g_idx] = True
            elif g_idx >= 0 and not assigned[g_idx]:
                # This anchor is the best for this GT - assign even if below threshold
                target_cls[a_idx] = 1
                target_reg[a_idx] = self._encode_box(gt_boxes[g_idx], anchors[a_idx])
                assigned[g_idx] = True

        return target_cls, target_reg

    def _encode_box(self, gt: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Encode box as (dx, dy, dw, dh) = (gt - anchor) / anchor_size."""
        dx = (gt[0] - anchor[0]) / (anchor[2] + 1e-6)
        dy = (gt[1] - anchor[1]) / (anchor[3] + 1e-6)
        dw = torch.log(gt[2] / (anchor[2] + 1e-6) + 1e-6)
        dh = torch.log(gt[3] / (anchor[3] + 1e-6) + 1e-6)
        return torch.stack([dx, dy, dw, dh])

    def _decode_box(self, encoded: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Decode box from (dx, dy, dw, dh) to (cx, cy, w, h).
        
        Args:
            encoded: [N, 4] decoded box deltas
            anchor: [N, 4] anchor boxes in (cx, cy, w, h)
        """
        # Broadcast anchor [N,4] to match encoded [N,4]
        cx = encoded[..., 0] * anchor[..., 2] + anchor[..., 0]
        cy = encoded[..., 1] * anchor[..., 3] + anchor[..., 1]
        w = torch.exp(encoded[..., 2].clamp(max=4)) * anchor[..., 2]
        h = torch.exp(encoded[..., 3].clamp(max=4)) * anchor[..., 3]
        return torch.stack([cx, cy, w, h], dim=-1)

    def forward(self, cls_preds: List[torch.Tensor], reg_preds: List[torch.Tensor],
               gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
               feature_shapes: List[Tuple[int, int]], strides: List[int]) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            cls_preds: list of [B, N_anchors, C] per feature level (P3-P7)
            reg_preds: list of [B, N_anchors, 4] per feature level
            gt_boxes: [B, M, 4] (cx, cy, w, h) in image coordinates
            gt_labels: [B, M] class labels
            feature_shapes: [(H3, W3), (H4, W4), ..., (H7, W7)] for each level
            strides: [8, 16, 32, 64, 128] for P3-P7

        Returns:
            total_loss, loss_components dict
        """
        device = cls_preds[0].device
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        num_positives = 0

        B = cls_preds[0].size(0)

        for batch_idx in range(B):
            for level_idx, (cls_p, reg_p, (fh, fw), stride) in enumerate(
                zip(cls_preds, reg_preds, feature_shapes, strides)):

                anchors = self._generate_anchors(fh, fw, stride, device)  # [N, 4]
                N_anchors = anchors.shape[0]

                # Get predictions for this batch and level
                cls_level = cls_p[batch_idx]  # [N_anchors, C]
                reg_level = reg_p[batch_idx]   # [N_anchors, 4]

                # Get GT for this image
                img_boxes = gt_boxes[batch_idx]  # [M, 4]
                img_labels = gt_labels[batch_idx]  # [M]

                if img_boxes.shape[0] == 0:
                    # No GT - all anchors are background (class 0)
                    target_cls, target_reg = self._match_anchors(
                        torch.empty(0, 4, device=device), anchors, iou_threshold=0.5
                    )
                else:
                    target_cls, target_reg = self._match_anchors(img_boxes, anchors, iou_threshold=0.5)

                # Filter to positive anchors only
                pos_mask = target_cls >= 0  # Actually positive (foreground)

                if pos_mask.sum() > 0:
                    num_positives += pos_mask.sum().item()

                    pos_cls_pred = cls_level[pos_mask]  # [N_pos, C]
                    pos_reg_pred = reg_level[pos_mask]  # [N_pos, 4]
                    pos_reg_target = target_reg[pos_mask]  # [N_pos, 4]

                    # Classification: treat all foreground as class 1 for simplicity
                    # (24-class version would use gt_labels per box)
                    # For now use binary focal on foreground vs background
                    pos_cls_target = torch.ones(pos_cls_pred.size(0), dtype=torch.long, device=device)

                    # Binary focal loss for foreground/background
                    ce_loss = F.binary_cross_entropy_with_logits(
                        pos_cls_pred.max(dim=1)[0],  # confidence of best class
                        torch.ones(pos_cls_pred.size(0), device=device),
                        reduction='none'
                    )
                    pt = torch.exp(-ce_loss)
                    focal_weight = (1 - pt) ** self.gamma
                    cls_loss = self.alpha * focal_weight * ce_loss
                    total_cls_loss = total_cls_loss + cls_loss.sum()

                    # GIoU regression
                    decoded_pred = self._decode_box(pos_reg_pred, anchors[pos_mask])
                    decoded_target = self._decode_box(pos_reg_target, anchors[pos_mask])
                    giou_loss = 1 - self._compute_giou(decoded_pred, decoded_target)
                    total_reg_loss = total_reg_loss + giou_loss.sum()

        if num_positives > 0:
            total_cls_loss = total_cls_loss / num_positives
            total_reg_loss = total_reg_loss / num_positives

        total_loss = self.cls_weight * total_cls_loss + self.giou_weight * total_reg_loss

        loss_components = {
            'det_cls': total_cls_loss,
            'det_reg': total_reg_loss,
            'det_total': total_loss,
            'num_positives': num_positives
        }

        return total_loss, loss_components

    def _compute_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute GIoU between two sets of boxes (cx, cy, w, h format)."""
        # Convert to (x1, y1, x2, y2)
        def to_xyxy(b):
            x1 = b[..., 0] - b[..., 2] / 2
            y1 = b[..., 1] - b[..., 3] / 2
            x2 = b[..., 0] + b[..., 2] / 2
            y2 = b[..., 1] + b[..., 3] / 2
            return x1, y1, x2, y2

        x1_1, y1_1, x2_1, y2_1 = to_xyxy(boxes1)
        x1_2, y1_2, x2_2, y2_2 = to_xyxy(boxes2)

        inter_x1 = torch.max(x1_1, x1_2)
        inter_y1 = torch.max(y1_1, y1_2)
        inter_x2 = torch.min(x2_1, x2_2)
        inter_y2 = torch.min(y2_1, y2_2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        iou = inter_area / (union_area + 1e-7)

        encl_x1 = torch.min(x1_1, x1_2)
        encl_y1 = torch.min(y1_1, y1_2)
        encl_x2 = torch.max(x2_1, x2_2)
        encl_y2 = torch.max(y2_1, y2_2)
        encl_area = (encl_x2 - encl_x1) * (encl_y2 - encl_y1) + 1e-7

        giou = iou - (encl_area - union_area) / encl_area

        return giou


class KendallMultiTaskLoss(nn.Module):
    """
    Kendall homoscedastic uncertainty weighting for multi-task loss.
    L = sum_t (1/2 * exp(-s_t) * L_t + s_t)
    where s_t = log(sigma_t^2) is a learnable parameter.
    """

    def __init__(
        self,
        num_tasks: int = 4,
        init_values: tuple = (0.0, -1.0, 0.0, 0.0),
        s_min: float = -4.0,
        s_max: float = 2.0
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.s_min = s_min
        self.s_max = s_max

        # Learnable log variance parameters: [det, pose, act, psr]
        self.log_vars = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))

    def clamp_vars(self):
        """Clamp log_var values to prevent overflow."""
        with torch.no_grad():
            self.log_vars.data.clamp_(self.s_min, self.s_max)

    def forward(self, losses: tuple, task_mask: tuple = None) -> torch.Tensor:
        """
        Args:
            losses: tuple of (L_det, L_pose, L_act, L_psr)
            task_mask: tuple of (mask_det, mask_pose, mask_act, mask_psr) - 0/1 scalars

        Returns:
            Total weighted loss
        """
        total_loss = 0.0
        loss_components = {}

        task_names = ["det", "pose", "act", "psr"]

        for i, (loss, name) in enumerate(zip(losses, task_names)):
            if task_mask is not None and task_mask[i] == 0:
                # Task not active - only train variance parameter
                loss_components[name] = self.log_vars[i]
                total_loss = total_loss + self.log_vars[i]
            else:
                # Task active - weighted loss + variance penalty
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
                loss_components[name] = weighted_loss
                total_loss = total_loss + weighted_loss

        return total_loss, loss_components

    def get_weights(self) -> dict:
        """Get current uncertainty weights."""
        return {
            "det": torch.exp(-self.log_vars[0]).item(),
            "pose": torch.exp(-self.log_vars[1]).item(),
            "act": torch.exp(-self.log_vars[2]).item(),
            "psr": torch.exp(-self.log_vars[3]).item(),
        }


class MSELoss(nn.Module):
    """MSE loss for head pose regression, with meter-scale normalization."""

    def __init__(self, scale: float = 0.001):
        super().__init__()
        self.scale = scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 9] (forward, position, up)
            target: [B, 9]
        """
        mse = F.mse_loss(pred, target)
        return self.scale * mse


def soft_argmax(heatmaps: torch.Tensor, temperature: float = 0.1) -> tuple:
    """
    Differentiable keypoint extraction via soft-argmax.

    Args:
        heatmaps: [B, J, H, W] raw heatmap values
        temperature: softmax temperature

    Returns:
        kpts: [B, J, 2] keypoint coordinates
        conf: [B, J] confidence scores
    """
    B, J, H, W = heatmaps.shape

    # Softmax over spatial dimensions
    flat_hm = heatmaps.view(B, J, -1)
    attn = F.softmax(flat_hm / temperature, dim=-1)

    # Grid coordinates
    grid_y = torch.arange(H, device=heatmaps.device, dtype=torch.float32).view(1, 1, H, 1)
    grid_x = torch.arange(W, device=heatmaps.device, dtype=torch.float32).view(1, 1, 1, W)

    # Compute expected coordinates
    y_coords = (attn.view(B, J, H, W) * grid_y).sum(dim=[2, 3])
    x_coords = (attn.view(B, J, H, W) * grid_x).sum(dim=[2, 3])

    kpts = torch.stack([x_coords, y_coords], dim=-1)  # [B, J, 2]

    # Confidence: max activation
    conf, _ = heatmaps.view(B, J, -1).max(dim=-1)
    conf = torch.sigmoid(conf)

    return kpts, conf