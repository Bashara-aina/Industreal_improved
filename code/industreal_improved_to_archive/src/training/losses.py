import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logger = logging.getLogger(__name__)

"""
POPW Loss Functions — Matches the exact diagram architecture
============================================================
L_det  = Focal Loss (α=0.25, γ=2) + GIoU for bounding boxes (Doc 2 C.1)
L_pose = Wing Loss (ω=0.05, ε=0.005) for keypoint regression
L_act  = LDAM-DRW (Doc 2 C.2) or CB-Focal Loss (β=0.999, γ=2.0, 74 classes)
L_psr  = Binary Focal Loss (α=0.25, γ=2.0) (Doc 2 C.3) — replaces BCE
L_total = Kendall(s_det, s_pose, s_act) with act_ramp = min(1, epoch/5)
init: s_det=0, s_pose=-1, s_act=0

Author: Bashara
Date: April 2026
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss

from src import config as C


def _get_kendall_stage(epoch: int) -> int:
    """Mirror of train.get_stage() — must stay in sync.
    
    Stage 1 (epochs 1-5):   detection only
    Stage 2 (epochs 6-15):  detection + head_pose  
    Stage 3 (epochs 16+):   all tasks
    """
    stage1_end = int(getattr(C, 'STAGE1_EPOCHS', 5))
    stage2_end = stage1_end + int(getattr(C, 'STAGE2_EPOCHS', 10))
    if epoch <= stage1_end:
        return 1
    if epoch <= stage2_end:
        return 2
    return 3


# ===========================================================================
# Detection Losses
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for ASD detection (Lin et al., 2017).
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    With α=0.25, γ=2 for class imbalance handling.

    Doc 2 C.1: GIoU loss replaces SmoothL1 for box regression.
    GIoU directly optimizes the IoU metric evaluated at mAP@0.5.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_iou_thresh: float = 0.5, neg_iou_thresh: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def _match_anchors(self, anchors: torch.Tensor, gt_boxes: torch.Tensor,
                      gt_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign anchors to GT boxes. Returns labels and matched boxes."""
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

    def _encode_boxes(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Encode GT boxes as deltas relative to anchors."""
        a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        a_w = (anchors[:, 2] - anchors[:, 0]).abs().clamp(min=1e-6)
        a_h = (anchors[:, 3] - anchors[:, 1]).abs().clamp(min=1e-6)

        g_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        g_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        g_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).abs().clamp(min=1e-6)
        g_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).abs().clamp(min=1e-6)

        return torch.stack([
            (g_cx - a_cx) / a_w,
            (g_cy - a_cy) / a_h,
            torch.log(g_w / a_w),
            torch.log(g_h / a_h),
        ], dim=1)

    def _decode_boxes(self, anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Decode anchor deltas to bounding boxes (xyxy format)."""
        a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        a_w = anchors[:, 2] - anchors[:, 0]
        a_h = anchors[:, 3] - anchors[:, 1]

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2].clamp(-4, 4)
        dh = deltas[:, 3].clamp(-4, 4)

        pred_w = torch.exp(dw) * a_w
        pred_h = torch.exp(dh) * a_h
        pred_cx = dx * a_w + a_cx
        pred_cy = dy * a_h + a_cy

        return torch.stack([
            pred_cx - pred_w / 2,
            pred_cy - pred_h / 2,
            pred_cx + pred_w / 2,
            pred_cy + pred_h / 2,
        ], dim=1)

    def forward(self, cls_preds: torch.Tensor, reg_preds: torch.Tensor = None,
                anchors: torch.Tensor = None, targets: List[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cls_loss, reg_loss (both scalars).

        Two call modes:
        - Detection (4 args): forward(cls_preds, reg_preds, anchors, targets)
          Returns (cls_loss, reg_loss) for full detection pipeline.
        - Standalone (2 args): forward(logits, targets) for simple classification.
          Returns (loss, zero_reg) where loss is a simple multiclass focal loss.
        """
        # --- C-3 standalone focal loss (2-arg mode) ---
        # 2-arg mode: forward(logits, targets) where targets is a 1D/2D tensor (not List[Dict])
        # 4-arg detection mode: targets is List[Dict]
        if anchors is None and (not isinstance(targets, list)):
            # Simple 2-arg call: forward(logits, targets)
            logits = cls_preds
            tgt = reg_preds if isinstance(reg_preds, torch.Tensor) else targets
            if tgt.dtype != torch.long:
                tgt = tgt.long()
            # Multiclass focal: FL = -alpha_t * (1-p_t)^gamma * log(p_t)
            probs = F.softmax(logits, dim=1)
            p_t = probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
            ce = F.cross_entropy(logits, tgt, reduction='none')
            alpha_t = self.alpha * torch.ones_like(p_t)
            loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
            return loss.mean(), torch.tensor(0.0, device=logits.device)

        # --- Full detection mode (4 args) ---
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

            # --- FIX #1: Clamp sigmoid inputs to prevent NaN in focal loss ---
            # p_t near 0 causes (1-p_t)^gamma → inf and log(0) → NaN
            p = torch.sigmoid(cls_pred).clamp(1e-7, 1.0 - 1e-7)
            ce = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='none')
            p_t = p * cls_target + (1 - p) * (1 - cls_target)
            alpha_t = self.alpha * cls_target + (1 - self.alpha) * (1 - cls_target)
            total_cls = total_cls + (alpha_t * (1 - p_t) ** self.gamma * ce).sum() / num_pos

            # C.1: GIoU loss replaces SmoothL1 — directly optimizes IoU metric
            # --- FIX #2: Guard GIoU against degenerate zero-area boxes ---
            if pos_mask.sum() > 0:
                pos_anchors = anchors[pos_mask]
                pos_deltas = reg_preds[i][pos_mask]
                pred_boxes = self._decode_boxes(pos_anchors, pos_deltas)
                gt_boxes_pos = matched_boxes[pos_mask]

                # Clip boxes to be valid before GIoU (zero-area boxes cause NaN)
                # [FIX] Build fully-modified boxes out-of-place via torch.stack, then clone
                pred_x1 = torch.clamp(pred_boxes[:, 0], 0, C.IMG_WIDTH)
                pred_y1 = torch.clamp(pred_boxes[:, 1], 0, C.IMG_HEIGHT)
                pred_x2 = torch.clamp(pred_boxes[:, 2], 0, C.IMG_WIDTH)
                pred_y2 = torch.clamp(pred_boxes[:, 3], 0, C.IMG_HEIGHT)
                # Ensure min width/height > 0
                pred_x2 = torch.maximum(pred_x2, pred_x1 + 1.0)
                pred_y2 = torch.maximum(pred_y2, pred_y1 + 1.0)
                pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1).clone()

                gt_x1 = gt_boxes_pos[:, 0]
                gt_y1 = gt_boxes_pos[:, 1]
                gt_x2 = torch.maximum(gt_boxes_pos[:, 2], gt_boxes_pos[:, 0] + 1.0)
                gt_y2 = torch.maximum(gt_boxes_pos[:, 3], gt_boxes_pos[:, 1] + 1.0)
                gt_boxes_pos = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1).clone()

                giou_loss = generalized_box_iou_loss(
                    pred_boxes, gt_boxes_pos, reduction='sum'
                )
                # Guard NaN GIoU (happens when boxes don't overlap at all)
                giou_loss = torch.where(
                    torch.isfinite(giou_loss),
                    giou_loss,
                    torch.tensor(0.0, device=device),
                )
                total_reg = total_reg + giou_loss / num_pos

        return total_cls / B, total_reg / B


# ===========================================================================
# GIoU Loss (standalone wrapper for C-5)
# ===========================================================================

class GIoULoss(nn.Module):
    """Standalone GIoU loss wrapper around torchvision's generalized_box_iou_loss.
    Directly optimizes IoU metric evaluated at mAP@0.5 (Doc 2 C.1)."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        return generalized_box_iou_loss(pred_boxes, target_boxes, reduction='mean')


# ===========================================================================
# Pose Loss — Wing Loss
# ===========================================================================

class WingLoss(nn.Module):
    """
    Wing Loss for robust keypoint regression.
    L = ω * ln(1 + |x|/ε)  for |x| < ω
       |x| - C              for |x| >= ω
    where C = ω - ω*ln(1+ω/ε)
    Parameters from diagram: ω=0.05, ε=0.005
    """
    def __init__(self, omega: float = 0.05, epsilon: float = 0.005):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * math.log(1 + omega / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C,
        )
        if weight is not None:
            loss = loss * weight.unsqueeze(-1)
        return loss.mean()


class PoseLoss(nn.Module):
    """
    Combined pose loss: Wing Loss for keypoints + L2 confidence regularization.
    """
    def __init__(self, wing_omega: float = 0.05, wing_epsilon: float = 0.005):
        super().__init__()
        self.wing_loss = WingLoss(omega=wing_omega, epsilon=wing_epsilon)

    def forward(self, pred_keypoints: torch.Tensor, pred_confidence: torch.Tensor,
                target_keypoints: torch.Tensor, target_confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_keypoints: [B, 17, 2]
            pred_confidence: [B, 17]
            target_keypoints: [B, 17, 2]
            target_confidence: [B, 17]
        Returns:
            loss: scalar
        """
        # Wing loss on keypoints — weighted by per-joint confidence (Doc 2 C.4)
        # Invalid joints (confidence=0) contribute zero loss
        loss_kp = self.wing_loss(pred_keypoints, target_keypoints, weight=target_confidence)

        # Confidence regularization: encourage high confidence when target is valid
        conf_loss = F.mse_loss(pred_confidence, target_confidence.clamp(0.0, 1.0))

        return loss_kp + 0.1 * conf_loss


# ===========================================================================
# Activity Loss — LDAM-DRW + CB-Focal (Doc 2 C.2)
# ===========================================================================

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss + Deferred Re-Weighting (DRW).

    Paper: "LDAM: Label-Distribution-Aware Margin Loss" (NeurIPS 2020)
    Used for long-tail classification on small datasets like IndustReal.

    DRW: Apply CB weights only AFTER epoch LDAM_DRW_EPOCH (when features are stable).
    Before that: standard cross-entropy with margin.

    The margin for class c: m_c = 1 / sqrt(sqrt(n_c))
    where n_c is the number of samples for class c.
    """
    def __init__(self, num_classes: int = 74, max_m: float = 0.5, s: float = 30,
                 cb_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.max_m = max_m
        self.s = s
        self.cb_weights = cb_weights
        self.register_buffer('class_weights', torch.ones(num_classes))
        self._raw_counts: Optional[np.ndarray] = None
        self._margins: Optional[torch.Tensor] = None

    def _compute_margins(self, cls_num_list: np.ndarray) -> torch.Tensor:
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(cls_num_list, 1e-8)))
        m_list = m_list * (self.max_m / m_list.max())
        return torch.tensor(m_list, dtype=torch.float32)

    def _ensure_margins(self, device: torch.device) -> torch.Tensor:
        """Lazily compute and cache margins."""
        if self._margins is None:
            counts = self._raw_counts if self._raw_counts is not None else np.ones(self.num_classes)
            self._margins = self._compute_margins(counts)
        return self._margins.to(device)

    @property
    def margin_cumsum(self) -> torch.Tensor:
        """Expose margins for inspection (CHECKLIST ITEM 31)."""
        if self._margins is None:
            return self._compute_margins(np.ones(self.num_classes))
        return self._margins

    def set_class_counts(self, counts):
        self._raw_counts = np.array(counts, dtype=np.float64)
        counts = np.array(counts, dtype=np.float64)
        effective = np.where(
            counts > 0,
            (1.0 - np.power(0.999, counts)) / (1.0 - 0.999),
            1.0,
        )
        weights = 1.0 / np.maximum(effective, 1e-8)
        weights = weights / weights.sum() * self.num_classes
        self.class_weights.data.copy_(torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch: int = 0, drw_epoch: int = 60) -> torch.Tensor:
        device = logits.device
        B, C = logits.shape

        is_soft_labels = (targets.dim() == 2 and targets.shape[1] == C)
        if is_soft_labels:
            hard_targets = targets.argmax(dim=1)
        else:
            hard_targets = targets

        m_list = self._compute_margins(
            self._raw_counts if self._raw_counts is not None else np.ones(self.num_classes),
        ).to(device)

        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, hard_targets.view(-1, 1), True)
        batch_m = m_list[hard_targets].view(-1, 1)
        x_m = logits - batch_m * index.float()
        # Clamp x_m to prevent overflow when s=30 and logits are large
        x_m = x_m.clamp(-10.0, 10.0)

        if epoch >= drw_epoch and self.cb_weights is not None:
            w = self.cb_weights.to(device)[hard_targets]
        else:
            w = torch.ones(B, device=device)

        # [FIX #C] Paper §2.2.4: label_smooth=0.1 for LDAM-DRW
        # [FIX #D] Clamp s*x_m to prevent softmax overflow (inf → NaN cascade)
        # softmax(±50) is numerically stable; beyond ±100 causes inf → NaN
        logits_safe = (self.s * x_m).clamp(-50.0, 50.0)
        return (w * F.cross_entropy(
            logits_safe, hard_targets, reduction='none',
            label_smoothing=0.1
        )).mean()


class ClassBalancedFocalLoss(nn.Module):
    """
    CB Focal Loss (Cui et al., 2019) for 74-class activity recognition.
    Effective number of samples: E(n) = (1 - β^n) / (1 - β)
    Weights: w_c = 1 / E(n_c)
    FL = (1 - p_t)^γ * CE_with_weights
    With optional label smoothing for better generalization.

    Doc 2 C.2: LDAM-DRW is preferred for long-tail IndustReal classes.
    """
    # Alias used by checklist scripts
    CBFocalLoss = None  # resolved after class definition below

    def __init__(self, num_classes: int = 74, beta: float = 0.999, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('class_weights', torch.ones(num_classes))

    def compute_beta_weights(self, frequencies: torch.Tensor) -> None:
        """
        Compute class-balanced weights from per-class sample frequencies.
        E(n) = (1 - β^n) / (1 - β)
        w_c = 1 / E(n_c)
        """
        freq = frequencies.float()
        effective = torch.where(
            freq > 0,
            (1.0 - torch.pow(self.beta, freq)) / (1.0 - self.beta),
            torch.ones_like(freq),
        )
        weights = 1.0 / effective.clamp(min=1e-8)
        weights = weights / weights.sum() * self.num_classes
        self.class_weights.data.copy_(weights)

    def set_class_counts(self, counts):
        counts = np.array(counts, dtype=np.float64)
        effective = np.where(
            counts > 0,
            (1.0 - np.power(self.beta, counts)) / (1.0 - self.beta),
            1.0,
        )
        weights = 1.0 / np.maximum(effective, 1e-8)
        weights = weights / weights.sum() * self.num_classes
        self.class_weights.data.copy_(torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        B, C = logits.shape

        probs = F.softmax(logits, dim=1)

        is_soft_labels = (targets.dim() == 2 and targets.shape[1] == C)

        if is_soft_labels:
            soft_targets = targets
            p_t = (soft_targets * probs).sum(dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            ce = -(soft_targets * log_probs).sum(dim=1)
        elif self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / self.num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

            p_t = (smooth_targets * probs).sum(dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            ce = -(smooth_targets * log_probs).sum(dim=1)
        else:
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            ce = F.cross_entropy(logits, targets, reduction='none')

        focal_weight = (1 - p_t) ** self.gamma

        if is_soft_labels:
            w = (self.class_weights.to(device).unsqueeze(0) * soft_targets).sum(dim=1)
        else:
            w = self.class_weights.to(device)[targets]

        loss = w * focal_weight * ce
        return loss.mean()


# Alias used by checklist scripts referencing CBFocalLoss
CBFocalLoss = ClassBalancedFocalLoss


class PSRFocalLoss(nn.Module):
    """
    Focal Loss for PSR — supports both multi-class (activity step classification)
    and multi-label (PSR component detection) modes.

    Multi-class mode (CHECKLIST ITEM 32): logits [B, 36], targets [B] integer class indices
    Multi-label mode (Doc 2 C.3): logits [B, 11], targets [B, 11] binary

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Detect mode: if targets are 1D integer indices → multi-class
        # if targets are 2D binary → multi-label (PSR components)
        if targets.dim() == 1 and targets.dtype == torch.long:
            # Multi-class mode: convert to one-hot, apply binary focal per class
            return self._multiclass_focal(logits, targets)
        else:
            # Multi-label mode: use binary focal loss (targets must be float)
            targets_float = targets.float()
            return binary_focal_loss(logits, targets_float, alpha=self.alpha, gamma=self.gamma)

    def _multiclass_focal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for multi-class classification."""
        ce = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.gather(F.softmax(logits, dim=1), dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * torch.ones_like(p_t)
        loss = alpha_t * focal_weight * ce
        return loss.mean()


def binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0,
                       per_component_alpha: torch.Tensor = None) -> torch.Tensor:
    """
    Binary Focal Loss for PSR (Doc 2 C.3).

    PSR has heavy class imbalance per component (component appears in <30% of frames).
    BCE WithLogitsLoss struggles with this; focal loss down-weights easy negatives.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    where p_t = p if y=1 else 1-p

    Args:
        logits: [B, C] or [B] tensor of sigmoid logits
        targets: [B, C] or [B] binary targets
        alpha: scalar fallback alpha (used when per_component_alpha is None)
        gamma: focal gamma
        per_component_alpha: [C] tensor of per-component alphas; if provided,
            overrides the scalar alpha. Computed as alpha_c = 2 * (1 - prevalence_c)
            so rare components get higher alpha (Doc 01 §D.4).
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)

    # --- FIX: Clamp p_t to prevent NaN in focal loss ---
    # p_t near 0 → (1-0)^gamma = 1, but -log(0) = inf → NaN
    # p_t near 1 → (1-1)^gamma = 0, log(1) = 0, but clamp is safe
    p_t = p_t.clamp(min=1e-7, max=1.0 - 1e-7)

    if per_component_alpha is not None:
        alpha_c = per_component_alpha.to(logits.device).unsqueeze(0)
        alpha_t = alpha_c * targets + (1 - alpha_c) * (1 - targets)
    else:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


# ===========================================================================
# Multi-Task Loss — Kendall Uncertainty Weighting
# ===========================================================================

class MultiTaskLoss(nn.Module):
    """
    Kendall homoscedastic uncertainty weighting for 4 tasks:
      detection + pose + activity + PSR

    L = sum_t [ exp(-s_t) * L_t + s_t ]

    Initialization per diagram:
      s_det = 0   (precision=1.0, neutral)
      s_pose = -1 (precision~2.7x higher at init)
      s_act = 0   (precision=1.0, neutral)
      s_psr = 0   (precision=1.0, neutral)

    Activity warmup: ramp from 0→1 over first 5 epochs.
    act_ramp = min(1, epoch/5)

    Kendall clamp range [-4, 2]:
      exp(-(-4)) = exp(4) ~ 54.6  max precision
      exp(-(2))  = exp(-2) ~ 0.135  min precision

    Doc 2 improvements:
      C.1: GIoU replaces SmoothL1 in FocalLoss
      C.2: LDAM-DRW replaces CB-Focal (when USE_LDAM_DRW=True)
      C.3: Binary focal loss replaces BCE for PSR
    """
    def __init__(
        self,
        num_classes_act: int = 74,
        num_psr_components: int = 11,
        train_det: bool = True,
        train_pose: bool = True,
        train_act: bool = True,
        train_psr: bool = True,
        use_kendall: bool = True,
    ):
        super().__init__()
        self.train_det = train_det
        self.train_pose = train_pose
        self.train_act = train_act
        self.train_psr = train_psr
        self.use_kendall = use_kendall

        # Kendall log variances (log σ²) — initialized on CPU; forward() moves
        # them to the correct device before use. This avoids GPU OOM at init
        # while keeping training fast (one-time per-forward transfer is negligible).
        #
        # DESIGN CHOICE (Warning C9): log_var_pose is shared for BOTH body pose AND
        # head pose. This is NOT a bug — it matches the paper §Multi-Task Loss spec:
        #   t ∈ {det, pose+head_pose, act, psr}
        # Body pose (17-keypoint Wing Loss) and head pose (9-DoF MSE) share a single
        # s_pose=-1 because both are pose tasks of similar magnitude (both ×0.001).
        # The paper never intended independent weighting for these two. If head pose
        # converges faster than body pose (or vice versa), the shared log_var means
        # they must be reweighted together — this is an intentional architectural choice
        # per the Kendall grouping, not a deficiency.
        self.log_var_det = nn.Parameter(torch.zeros(1))
        self.log_var_pose = nn.Parameter(torch.tensor([-1.0]))
        self.log_var_act = nn.Parameter(torch.zeros(1))  # Paper §Multi-Task Loss: init [0,-1,0,0] → use zeros(1) per spec
        self.log_var_psr = nn.Parameter(torch.zeros(1))

        # Sub-losses
        self.det_loss_fn = FocalLoss(alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA)
        self.pose_loss_fn = PoseLoss(wing_omega=C.WING_OMEGA, wing_epsilon=C.WING_EPSILON)

        # C.2: LDAM-DRW or CB-Focal for activity
        use_ldam = bool(getattr(C, 'USE_LDAM_DRW', False))
        if use_ldam:
            self.act_loss_fn = LDAMLoss(
                num_classes=num_classes_act,
                max_m=float(getattr(C, 'LDAM_MAX_M', 0.5)),
                s=float(getattr(C, 'LDAM_S', 30)),
            )
        else:
            self.act_loss_fn = ClassBalancedFocalLoss(
                num_classes=num_classes_act,
                beta=C.CB_BETA,
                gamma=C.CB_GAMMA,
                label_smoothing=getattr(C, 'CB_LABEL_SMOOTHING', 0.1),
            )
        self.use_ldam = use_ldam

        # C.3: Binary focal loss for PSR (instead of BCE)
        self.psr_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.use_psr_focal = bool(getattr(C, 'PSR_FOCAL_GAMMA', 0) > 0)
        self.psr_focal_alpha = float(getattr(C, 'PSR_FOCAL_ALPHA', 0.25))
        self.psr_focal_gamma = float(getattr(C, 'PSR_FOCAL_GAMMA', 2.0))
        self._psr_per_component_alpha: torch.Tensor = None
        self._psr_num_components = num_psr_components

        self.head_pose_loss_fn = nn.MSELoss(reduction='mean')

        self._psr_temporal_smooth_weight = float(getattr(C, 'PSR_TEMPORAL_SMOOTH_WEIGHT', 0.05))
        self._psr_temporal_history: Dict[str, List[torch.Tensor]] = {}

        self._act_warmup_epochs = int(getattr(C, 'ACT_RAMP_EPOCHS', 5))
        self._current_epoch = 0

    def set_class_counts(self, counts):
        self.act_loss_fn.set_class_counts(counts)

    def set_psr_class_counts(self, prevalence_per_component: torch.Tensor):
        """
        Doc 01 §D.4: Per-component PSR focal loss weighting.

        Each of the 11 PSR components has different prevalence (component 0 appears
        in ~95% of frames; component 10 in <30%). Rare components need higher
        focal alpha to avoid the model collapsing to predicting 0 everywhere.

        alpha_c = 2 * (1 - prevalence_c)
        - prevalence_c near 1.0 → alpha_c near 0 (common component, well-balanced)
        - prevalence_c near 0.0 → alpha_c near 2.0 (rare component, needs heavy upweighting)

        Call this once at training start after computing prevalence from dataset.

        Args:
            prevalence_per_component: [11] tensor of per-component "appears as 1" prevalence
                                      values in [0, 1]. Computed as:
                                      cache.psr_per_frame.mean(axis=0)
        """
        prev = prevalence_per_component.float().clamp(0.01, 0.99)
        alpha_c = 2.0 * (1.0 - prev)
        self._psr_per_component_alpha = alpha_c
        logger.debug(
            f'PSR per-component alpha: {alpha_c.numpy().round(3).tolist()} '
            f'(from prevalence {prev.numpy().round(3).tolist()})'
        )

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        # Find the first available tensor to determine device and dtype (PSR-only branch
        # omits cls_preds, so fall back to psr_logits or any other tensor).
        device = None
        output_dtype = None
        for key in ('cls_preds', 'psr_logits', 'act_logits', 'head_pose', 'heatmaps'):
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                device = outputs[key].device
                output_dtype = outputs[key].dtype
                break
        if device is None:
            raise RuntimeError(
                f'MultiTaskLoss.forward received an empty outputs dict: {list(outputs.keys())}'
            )

        # Ensure Kendall log_vars are on the same device as the model output
        # (they are CPU-initialized; autograd does NOT auto-move registered params)
        if self.log_var_det.device != device:
            self.log_var_det.data = self.log_var_det.data.to(device)
            self.log_var_pose.data = self.log_var_pose.data.to(device)
            self.log_var_act.data = self.log_var_act.data.to(device)
            self.log_var_psr.data = self.log_var_psr.data.to(device)

        zero = torch.tensor(0.0, device=device, dtype=output_dtype)

        # === Detection ===
        if self.train_det:
            cls_loss, reg_loss = self.det_loss_fn(
                outputs['cls_preds'], outputs['reg_preds'],
                outputs['anchors'], targets['detection'],
            )
            giou_weight = float(getattr(C, 'GIOU_WEIGHT', 2.0))
            loss_det = cls_loss + giou_weight * reg_loss
            # --- FIX: Floor loss_det at zero to prevent GIoU negative values
            # causing Kendall divergence. reg_loss can be negative (GIoU ∈ [-1,1]),
            # and with Kendall prec = exp(-lv_det) up to ~54.6, a loss_det of -1.5
            # multiplied by prec=54.6 gives ~-82 per detection step → divergence.
            # Full zero-floor: GIoU<0 gets 0 gradient (no negative signal to log_var).
            NEG_SLOPE = 0.0
            loss_det = torch.where(
                loss_det < 0,
                NEG_SLOPE * loss_det,
                loss_det,
            )
            # NaN/inf guard on detection loss
            if not torch.isfinite(loss_det).all():
                loss_det = torch.tensor(1e-4, device=device, dtype=output_dtype)
        else:
            cls_loss = reg_loss = loss_det = zero

        # === Pose (Wing Loss on keypoints) ===
        # Note: When TRAIN_HEAD_POSE=True, train_pose=True controls head pose head
        # (9-DoF MSE), NOT body keypoints (COCO-style). The model generates pseudo-
        # keypoints from detection outputs for PoseFiLM, but there are no real
        # keypoint annotations in IndustReal. Wing Loss block is only for the rare
        # body-keypoint case where train_pose=True AND keypoints are actually present.
        if self.train_pose and 'keypoints' in targets:
            loss_pose = self.pose_loss_fn(
                outputs['keypoints'],
                outputs['pose_confidence'],
                targets['keypoints'],
                targets['pose_confidence'],
            ) * 0.001  # Kendall exp(-lv_pose)=exp(1)≈2.7 amplifies
        else:
            loss_pose = zero

        # === Activity ===
        if self.train_act:
            # C.2: LDAM-DRW needs epoch for DRW decision
            if self.use_ldam:
                drw_epoch = int(getattr(C, 'LDAM_DRW_EPOCH', 60))
                loss_act = self.act_loss_fn(
                    outputs['act_logits'],
                    targets['activity'],
                    epoch=self._current_epoch,
                    drw_epoch=drw_epoch,
                )
            else:
                loss_act = self.act_loss_fn(outputs['act_logits'], targets['activity'])
        else:
            loss_act = zero

        # [FIX #1] Preventive NaN guard — catch inf before smooth cap propagates it.
        # LDAM can produce inf if logits are extreme (e.g., all zeros → softamax → NaN from log).
        # LDAM forward: F.cross_entropy with label_smoothing=0.1 and s=30 — any
        # inf in logits will create NaN in cross_entropy output.
        # This guard is BEFORE the smooth cap so NaN cannot corrupt the cap formula.
        if not torch.isfinite(loss_act).all():
            loss_act = torch.where(
                torch.isfinite(loss_act),
                loss_act,
                zero.expand_as(loss_act) if loss_act.numel() > 1 else zero,
            )
            # Fallback: if entire tensor is non-finite, use zero (activity contributes nothing this batch).
            if not torch.isfinite(loss_act).all():
                loss_act = zero

        # Activity warmup ramp
        # NOTE: +1 so epoch 0 gets ramp=1/5=0.2 instead of 0/5=0 (which zeroed loss_act entirely)
        act_ramp = min(1.0, (self._current_epoch + 1) / max(self._act_warmup_epochs, 1))
        loss_act = loss_act * act_ramp

        # --- FIX: Activity loss cap to prevent NaN cascade at Stage 3 entry ---
        # Prior runs showed activity loss spiking to 40.8 when head_pose + PSR activate
        # simultaneously at epoch 16, causing log_var explosion and NaN cascade.
        # Cap activity loss to a safe threshold; Kendall will still learn from lower values.
        #
        # PROBLEM: hard clamp(max=40.0) zeroes the gradient when loss > 40.
        # PyTorch clamp backward: at the boundary, subgradient = 0 (hard max).
        # LDAM loss at epoch 16 ~= 55 > 40 → gradient zeroed → activity head can't learn.
        #
        # SOLUTION: fully-differentiable smooth cap that preserves gradient above cap.
        # loss_capped(x, cap) = x for x <= cap, cap * (1 + log(x/cap)) for x > cap
        # - Below cap: gradient = 1.0 (passthrough)
        # - Above cap: gradient = cap/x > 0 (never zeroed)
        # torch.where preserves the autograd graph through both branches.
        # [FIX #2] Clamp smooth cap log input to prevent extreme value instability.
        # log(x) is only defined for x > 0. clamp(min=1e-6) guards against x ≤ 0 from prior
        # numerical errors. clamp(max=1e6) prevents overflow in exp(log(x)) downstream.
        # The outer torch.where still passes gradient through when loss_act <= act_cap.
        act_cap = float(getattr(C, 'ACTIVITY_LOSS_CAP', 40.0))
        loss_act_safe = loss_act.clamp(min=1e-6, max=1e6)
        loss_act = torch.where(
            loss_act > act_cap,
            act_cap * (1 + torch.log(loss_act_safe / act_cap)),
            loss_act
        )

        # === PSR ===
        if self.train_psr:
            # C.3: Binary focal loss for PSR (Doc 01 §D.4: per-component alpha)
            if self.use_psr_focal:
                loss_psr = binary_focal_loss(
                    outputs['psr_logits'],
                    targets['psr_labels'],
                    alpha=self.psr_focal_alpha,
                    gamma=self.psr_focal_gamma,
                    per_component_alpha=self._psr_per_component_alpha,
                )
            else:
                loss_psr = self.psr_loss_fn(
                    outputs['psr_logits'],
                    targets['psr_labels'],
                )

            if self._psr_temporal_smooth_weight > 0:
                preds = torch.sigmoid(outputs['psr_logits'])
                labels = targets['psr_labels']

                smooth_loss = torch.tensor(0.0, device=device)
                bs = preds.shape[0]

                for i in range(bs):
                    p_i = preds[i]
                    l_i = labels[i]

                    diff_p = (p_i[1:] - p_i[:-1]).abs().mean()
                    diff_l = (l_i[1:] - l_i[:-1]).abs().mean()

                    pred_change = torch.sigmoid(diff_p)
                    label_change = diff_l
                    smooth_loss = smooth_loss + (
                        (pred_change - label_change) ** 2
                    )
                smooth_loss = smooth_loss / max(bs, 1)

                loss_psr = loss_psr + self._psr_temporal_smooth_weight * smooth_loss
        else:
            loss_psr = zero

        # === Head Pose (MSE on 9-DoF) ===
        if 'head_pose' in outputs and outputs['head_pose'] is not None:
            loss_head_pose = self.head_pose_loss_fn(
                outputs['head_pose'],
                targets['head_pose'],
            ) * 0.001  # Head pose 9-DoF MSE
        else:
            loss_head_pose = zero

        # === Kendall weighting ===
        if self.use_kendall:
            lv_det = self.log_var_det.clamp(-4.0, 2.0)
            lv_hp = self.log_var_pose.clamp(-4.0, 2.0)
            lv_act = self.log_var_act.clamp(-4.0, 2.0)
            lv_psr = self.log_var_psr.clamp(-4.0, 2.0)

            prec_det = torch.exp(-lv_det)
            prec_hp = torch.exp(-lv_hp)
            prec_act = torch.exp(-lv_act)
            prec_psr = torch.exp(-lv_psr)

            # Stage-aware Kendall: zero precision AND log_var of frozen tasks to prevent
            # gradient corruption in their log_vars during staged training.
            # Epoch 0: no staging (backward-compat for resumed runs).
            # Epoch 1-5 (stage 1): detection only.
            # Epoch 6-15 (stage 2): detection + head_pose.
            # Epoch 16+ (stage 3): all tasks.
            if bool(getattr(C, 'STAGED_TRAINING', True)) and self._current_epoch >= 1:
                stage = _get_kendall_stage(self._current_epoch)
                if stage == 1:
                    # Zero BOTH precision and log_var for frozen tasks
                    prec_hp = prec_hp * 0
                    lv_hp = lv_hp * 0
                    prec_act = prec_act * 0
                    lv_act = lv_act * 0
                    prec_psr = prec_psr * 0
                    lv_psr = lv_psr * 0
                elif stage == 2:
                    prec_act = prec_act * 0
                    lv_act = lv_act * 0
                    prec_psr = prec_psr * 0
                    lv_psr = lv_psr * 0

            total = torch.tensor(0.0, device=device)
            if self.train_det:
                total = total + prec_det * loss_det + lv_det
            # Sanity floor: ensure no component loss is negative enough to cause
            # Kendall divergence. Soft floor already applied to loss_det above, but
            # add this as last-resort guard for all components.
            if self.train_pose:
                loss_pose = loss_pose.clamp(min=0.0)
                total = total + prec_hp * loss_pose + lv_hp
            if self.train_act:
                loss_head_pose = loss_head_pose.clamp(min=0.0)
                loss_act = loss_act.clamp(min=0.0)
                total = total + prec_act * loss_head_pose + lv_act + prec_act * loss_act + lv_act
            if self.train_psr:
                loss_psr = loss_psr.clamp(min=0.0)
                total = total + prec_psr * loss_psr + lv_psr
            total = total.squeeze()

            # --- NaN guard in Kendall total ---
            # Handle both scalar and 1-element tensor cases
            total_val = total.item() if total.numel() == 1 else total
            if not math.isfinite(total_val):
                parts = []
                if self.train_det:
                    parts.append(loss_det)
                if self.train_pose:
                    parts.append(loss_pose)
                if self.train_act:
                    parts.append(loss_head_pose)
                    parts.append(loss_act)
                if self.train_psr:
                    parts.append(loss_psr)
                finite_parts = [p for p in parts if torch.isfinite(p) and p >= 0]
                if finite_parts:
                    total = torch.stack(finite_parts).sum()
                else:
                    total = loss_det
        else:
            prec_det = prec_hp = prec_act = prec_psr = torch.tensor(1.0, device=device)
            _loss_act_staged = loss_act
            _loss_psr_staged = loss_psr
            _loss_pose_staged = loss_pose if self.train_pose else loss_head_pose
            if bool(getattr(C, 'STAGED_TRAINING', True)) and self._current_epoch >= 1:
                stage = _get_kendall_stage(self._current_epoch)
                if stage == 1:
                    _loss_act_staged = zero
                    _loss_psr_staged = zero
                    _loss_pose_staged = zero
                elif stage == 2:
                    _loss_act_staged = zero
                    _loss_psr_staged = zero
                    if self.train_pose:
                        _loss_pose_staged = zero
            total = loss_det + _loss_pose_staged + _loss_act_staged + _loss_psr_staged

        # Normalized weights for logging — use the ACTUAL precision values
        # (already zeroed for staged training), not the clamped pre-zeroing values
        with torch.no_grad():
            a_det = prec_det.item() if isinstance(prec_det, torch.Tensor) else prec_det
            a_hp = prec_hp.item() if isinstance(prec_hp, torch.Tensor) else prec_hp
            a_act = prec_act.item() if isinstance(prec_act, torch.Tensor) else prec_act
            a_psr = prec_psr.item() if isinstance(prec_psr, torch.Tensor) else prec_psr
            ws = a_det + a_hp + a_act + a_psr + 1e-8
            wd = a_det / ws
            wp = a_hp / ws
            wa = a_act / ws
            wps = a_psr / ws

        loss_dict = {
            'total': total.item(),
            'det_cls': cls_loss.item(),
            'det_reg': reg_loss.item(),
            'det': loss_det.item(),
            'pose': loss_pose.item(),
            'activity': loss_act.item(),
            'psr': loss_psr.item(),
            'head_pose': loss_head_pose.item(),
            'w_det': wd / ws,
            'w_pose': wp / ws,
            'w_act': wa / ws,
            'w_psr': wps / ws,
            'log_var_det': self.log_var_det.item(),
            'log_var_pose': self.log_var_pose.item(),
            'log_var_act': self.log_var_act.item(),
            'log_var_psr': self.log_var_psr.item(),
            'act_ramp': act_ramp,
        }

        return total, loss_dict