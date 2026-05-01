import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Multi-Task Loss Functions — IndustReal POPW Adaptation
=========================================================
ASD Detection  : Focal Loss (same as IKEA)
Head Pose      : L1 regression loss for 9-DoF head pose
Activity Rec.  : Class-Balanced Focal Loss (74 classes, same as IKEA)
PSR            : Multi-label BCE loss for 11 components
Multi-task     : Kendall homoscedastic uncertainty weighting (4 tasks)

Author: Bashara
Date: February 2026
"""

import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

import config as C


# ===========================================================================
# Detection — Focal Loss (unchanged from IKEA)
# ===========================================================================

class FocalLoss(nn.Module):
    """Focal Loss for ASD detection (Lin et al., 2017)."""

    def __init__(self, alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA,
                 pos_iou_thresh=0.5, neg_iou_thresh=0.4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def _match_anchors(self, anchors, gt_boxes, gt_labels):
        """Assign anchors to GTs. Returns matched labels (>=0 pos, -1 ignore, -2 bg) and boxes."""
        N = anchors.shape[0]
        device = anchors.device

        if gt_boxes.shape[0] == 0:
            return (torch.full((N,), -2, dtype=torch.long, device=device),
                    torch.zeros((N, 4), device=device))

        ious = box_iou(anchors, gt_boxes)
        max_iou, max_idx = ious.max(dim=1)

        labels = torch.full((N,), -1, dtype=torch.long, device=device)
        matched_boxes = gt_boxes[max_idx]
        labels[max_iou < self.neg_iou_thresh] = -2
        pos_mask = max_iou >= self.pos_iou_thresh
        labels[pos_mask] = gt_labels[max_idx[pos_mask]]

        for gi in range(gt_boxes.shape[0]):
            labels[ious[:, gi].argmax()] = gt_labels[gi]

        return labels, matched_boxes

    def _encode_boxes(self, anchors, gt_boxes):
        a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        a_w = anchors[:, 2] - anchors[:, 0]
        a_h = anchors[:, 3] - anchors[:, 1]
        g_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        g_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        g_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        g_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        # Clamp widths/heights to >0 before log to avoid NaN from log(negative)
        a_w = a_w.abs().clamp(min=1e-6)
        a_h = a_h.abs().clamp(min=1e-6)
        g_w = g_w.abs().clamp(min=1e-6)
        g_h = g_h.abs().clamp(min=1e-6)
        return torch.stack([
            (g_cx - a_cx) / (a_w + 1e-6),
            (g_cy - a_cy) / (a_h + 1e-6),
            torch.log(g_w / a_w),
            torch.log(g_h / a_h),
        ], dim=1)

    def forward(self, cls_preds, reg_preds, anchors, targets):
        """Returns cls_loss, reg_loss (both scalars)."""
        B = cls_preds.shape[0]
        device = cls_preds.device
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)

        for i in range(B):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)
            matched_labels, matched_boxes = self._match_anchors(anchors, gt_boxes, gt_labels)

            pos_mask = matched_labels >= 0
            neg_mask = matched_labels == -2
            valid_mask = pos_mask | neg_mask
            num_pos = max(pos_mask.sum().item(), 1)

            cls_pred = cls_preds[i][valid_mask]
            cls_target = torch.zeros_like(cls_pred)
            if pos_mask.sum() > 0:
                pos_in_valid = pos_mask[valid_mask]
                cls_target[pos_in_valid, matched_labels[valid_mask][pos_in_valid]] = 1.0

            p = torch.sigmoid(cls_pred)
            ce = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='none')
            p_t = p * cls_target + (1 - p) * (1 - cls_target)
            alpha_t = self.alpha * cls_target + (1 - self.alpha) * (1 - cls_target)
            total_cls = total_cls + (alpha_t * (1 - p_t) ** self.gamma * ce).sum() / num_pos

            if pos_mask.sum() > 0:
                deltas = self._encode_boxes(anchors[pos_mask], matched_boxes[pos_mask])
                total_reg = total_reg + F.smooth_l1_loss(
                    reg_preds[i][pos_mask], deltas, beta=0.11, reduction='sum'
                ) / num_pos

        return total_cls / B, total_reg / B


# ===========================================================================
# Head Pose — L1 Regression Loss (NEW)
# ===========================================================================

class HeadPoseLoss(nn.Module):
    """L1 regression loss for 9-DoF head pose.

    The 9-DoF head pose is: forward_vector[3] + position[3] + up_vector[3].
    Uses SmoothL1 (Huber) loss for outlier robustness.
    """

    def __init__(self, use_smooth: bool = True):
        super().__init__()
        self.use_smooth = use_smooth

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:  [B, 9] predicted head pose
            gt:    [B, 9] ground-truth head pose
            weight: [B] optional per-sample weight

        Returns:
            Scalar loss
        """
        if self.use_smooth:
            loss = F.smooth_l1_loss(pred, gt, reduction='none').sum(dim=1)
        else:
            loss = F.l1_loss(pred, gt, reduction='none').sum(dim=1)

        if weight is not None:
            loss = loss * weight
            denom = weight.sum().clamp(min=1.0)
        else:
            denom = pred.shape[0]

        return loss.sum() / denom


# ===========================================================================
# Procedure Step Recognition — Multi-label BCE (NEW)
# ===========================================================================

class PSRLoss(nn.Module):
    """Multi-label binary cross entropy for 11 PSR components.

    Ground truth uses -1 to indicate an error/unknown state that should
    not contribute to the loss (treated as 0 but tracked separately).
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        psr_logits: torch.Tensor,
        psr_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            psr_logits: [B, 11] sigmoid outputs
            psr_labels: [B, 11] binary targets (0/1), -1 for error/unknown

        Returns:
            loss: scalar
            info: dict with error_count
        """
        valid_mask = (psr_labels != -1)  # [B, 11]
        error_count = (~valid_mask).sum().item()

        labels_safe = psr_labels.clone()
        labels_safe[~valid_mask] = 0

        loss = F.binary_cross_entropy_with_logits(
            psr_logits, labels_safe, reduction='none'
        )  # [B, 11]
        loss = loss * valid_mask.float()

        if self.reduction == 'sum':
            loss = loss.sum()
        else:
            num_valid = valid_mask.sum().clamp(min=1.0)
            loss = loss.sum() / num_valid

        return loss, {'error_count': error_count}


# ===========================================================================
# Activity Recognition — Class-Balanced Focal Loss (unchanged from IKEA)
# ===========================================================================

class ClassBalancedFocalLoss(nn.Module):
    """CB Focal Loss (Cui et al., 2019) for 74-class activity imbalance."""

    def __init__(self, num_classes=C.NUM_CLASSES_ACT, beta=C.CB_BETA, gamma=C.CB_GAMMA):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.register_buffer('class_weights', torch.ones(num_classes))

    def set_class_counts(self, counts):
        counts = np.array(counts, dtype=np.float64)
        effective = np.where(
            counts > 0,
            (1.0 - np.power(self.beta, counts)) / (1.0 - self.beta),
            1.0,
        )
        weights = 1.0 / np.maximum(effective, 1e-8)
        weights = weights / weights.sum() * self.num_classes
        self.class_weights.data.copy_(
            torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        device = logits.device
        w = self.class_weights.to(device)[targets]
        ce = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce)
        return (w * (1 - p_t) ** self.gamma * ce).mean()


# ===========================================================================
# Multi-Task Loss — Kendall Uncertainty Weighting (4 tasks)
# ===========================================================================

class MultiTaskLoss(nn.Module):
    """
    Kendall homoscedastic uncertainty weighting for 4 tasks:
      detection + head_pose + activity + psr

    L = sum_t [ exp(-s_t) * L_t + s_t ] where s_t = log(sigma_t^2)

    Ablation flags:
      - TRAIN_DET       : ASD detection loss
      - TRAIN_HEAD_POSE : 9-DoF head pose regression loss
      - TRAIN_ACT       : Activity recognition loss (with warmup)
      - TRAIN_PSR       : Procedure step recognition loss
      - USE_KENDALL     : learned uncertainty vs equal weights

    log_var initialization (Kendall et al., 2018):
      - log_var_det       = 0.0  (precision=1.0, neutral)
      - log_var_head_pose = -1.0 (precision~2.7x higher at init)
      - log_var_act       = 0.0  (precision=1.0, neutral)
      - log_var_psr       = 0.0  (precision=1.0, neutral)

    Kendall clamp range [-4, 2]:
      - exp(-(-4)) = exp(4) ~ 55   max precision
      - exp(-(2))  = exp(-2) ~ 0.14 min precision
    """

    def __init__(
        self,
        num_classes_act: int = C.NUM_CLASSES_ACT,
        num_psr_components: int = C.NUM_PSR_COMPONENTS,
        train_det: bool = C.TRAIN_DET,
        train_head_pose: bool = C.TRAIN_HEAD_POSE,
        train_act: bool = C.TRAIN_ACT,
        train_psr: bool = C.TRAIN_PSR,
        use_kendall: bool = C.USE_KENDALL,
    ):
        super().__init__()
        self.train_det = train_det
        self.train_head_pose = train_head_pose
        self.train_act = train_act
        self.train_psr = train_psr
        self.use_kendall = use_kendall

        # Kendall log variances (log(sigma^2))
        self.log_var_det = nn.Parameter(torch.zeros(1))
        self.log_var_head_pose = nn.Parameter(torch.tensor([-1.0]))
        self.log_var_act = nn.Parameter(torch.zeros(1))
        self.log_var_psr = nn.Parameter(torch.zeros(1))

        # Sub-losses
        self.det_loss = FocalLoss()
        self.pose_loss = HeadPoseLoss(use_smooth=True)
        self.act_loss = ClassBalancedFocalLoss(num_classes=num_classes_act)
        self.psr_loss = PSRLoss(reduction='mean')

        # Activity warmup: ramp activity loss weight from 0→1 over first
        # ACT_WARMUP_EPOCHS epochs. Gives head pose time to stabilize before
        # FiLM conditioning (via pose features) drives activity gradients.
        self._act_warmup_epochs = int(getattr(C, 'ACT_WARMUP_EPOCHS', 5))
        self._current_epoch = 0

    def set_class_counts(self, counts):
        """Set class frequencies for CB focal loss."""
        self.act_loss.set_class_counts(counts)

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to update activity warmup ramp."""
        self._current_epoch = epoch

    def forward(self, outputs: Dict, targets: Dict):
        device = outputs['cls_preds'].device
        zero = torch.tensor(0.0, device=device)

        # --- Detection ---
        if self.train_det:
            cls_loss, reg_loss = self.det_loss(
                outputs['cls_preds'], outputs['reg_preds'],
                outputs['anchors'], targets['detection'],
            )
            loss_det = cls_loss + reg_loss
        else:
            cls_loss = reg_loss = loss_det = zero

        # --- Head Pose ---
        if self.train_head_pose:
            loss_pose = self.pose_loss(
                outputs['head_pose'],   # [B, 9]
                targets['head_pose'],   # [B, 9]
            )
        else:
            loss_pose = zero

        # --- Activity Recognition (with warmup ramp) ---
        if self.train_act:
            loss_act = self.act_loss(outputs['act_logits'], targets['activity'])
        else:
            loss_act = zero

        act_ramp = min(1.0, self._current_epoch / max(self._act_warmup_epochs, 1))
        loss_act = loss_act * act_ramp

        # --- PSR ---
        if self.train_psr:
            loss_psr, psr_info = self.psr_loss(
                outputs['psr_logits'],  # [B, 11]
                targets['psr_labels'],   # [B, 11]
            )
        else:
            loss_psr = zero
            psr_info = {'error_count': 0}

        # --- Combine with Kendall weighting ---
        if self.use_kendall:
            lv_det  = self.log_var_det.clamp(-4.0, 2.0)
            lv_pose = self.log_var_head_pose.clamp(-4.0, 2.0)
            lv_act  = self.log_var_act.clamp(-4.0, 2.0)
            lv_psr  = self.log_var_psr.clamp(-4.0, 2.0)

            prec_det  = torch.exp(-lv_det)
            prec_pose = torch.exp(-lv_pose)
            prec_act  = torch.exp(-lv_act)
            prec_psr  = torch.exp(-lv_psr)

            total = torch.tensor(0.0, device=device)
            if self.train_det:
                total = total + prec_det * loss_det + lv_det
            if self.train_head_pose:
                total = total + prec_pose * loss_pose + lv_pose
            if self.train_act:
                total = total + prec_act * loss_act + lv_act
            if self.train_psr:
                total = total + prec_psr * loss_psr + lv_psr
            total = total.squeeze()
        else:
            prec_det = prec_pose = prec_act = prec_psr = torch.tensor(1.0, device=device)
            total = loss_det + loss_pose + loss_act + loss_psr

        # Compute normalized weights for logging
        with torch.no_grad():
            wd   = prec_det.item()
            wp   = prec_pose.item()
            wa   = prec_act.item()
            wps  = prec_psr.item()
            ws   = wd + wp + wa + wps + 1e-8

        loss_dict = {
            'total': total.item(),
            'det_cls': cls_loss.item(),
            'det_reg': reg_loss.item(),
            'det': loss_det.item(),
            'head_pose': loss_pose.item(),
            'activity': loss_act.item(),
            'psr': loss_psr.item(),
            'psr_error_count': psr_info['error_count'],
            'w_det': wd / ws,
            'w_pose': wp / ws,
            'w_act': wa / ws,
            'w_psr': wps / ws,
            'log_var_det': self.log_var_det.item(),
            'log_var_head_pose': self.log_var_head_pose.item(),
            'log_var_act': self.log_var_act.item(),
            'log_var_psr': self.log_var_psr.item(),
        }
        return total, loss_dict