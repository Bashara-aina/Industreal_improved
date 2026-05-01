import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
POPW Loss Functions — Matches the exact diagram architecture
============================================================
L_det  = Focal Loss (α=0.25, γ=2) + GIoU for bounding boxes (Doc 02 C.1)
L_pose = Wing Loss (ω=0.05, ε=0.005) for keypoint regression
L_act  = LDAM-DRW (Doc 02 C.2) or CB-Focal Loss (β=0.999, γ=2.0, 74 classes)
L_psr  = Binary Focal Loss (α=0.25, γ=2.0) (Doc 02 C.3) — replaces BCE
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

import config as C


# ===========================================================================
# Detection Losses
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for ASD detection (Lin et al., 2017).
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    With α=0.25, γ=2 for class imbalance handling.

    Doc 02 C.1: GIoU loss replaces SmoothL1 for box regression.
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
        g_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        g_h = gt_boxes[:, 3] - gt_boxes[:, 1]

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

    def forward(self, cls_preds: torch.Tensor, reg_preds: torch.Tensor,
                anchors: torch.Tensor, targets: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
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

            # C.1: GIoU loss replaces SmoothL1 — directly optimizes IoU metric
            if pos_mask.sum() > 0:
                pos_anchors = anchors[pos_mask]
                pos_deltas = reg_preds[i][pos_mask]
                pred_boxes = self._decode_boxes(pos_anchors, pos_deltas)
                gt_boxes_pos = matched_boxes[pos_mask]
                total_reg = total_reg + generalized_box_iou_loss(
                    pred_boxes, gt_boxes_pos, reduction='sum'
                ) / num_pos

        return total_cls / B, total_reg / B


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
        # Wing loss on keypoints — weighted by per-joint confidence (Doc 02 C.4)
        # Invalid joints (confidence=0) contribute zero loss
        loss_kp = self.wing_loss(pred_keypoints, target_keypoints, weight=target_confidence)

        # Confidence regularization: encourage high confidence when target is valid
        conf_loss = F.mse_loss(pred_confidence, target_confidence.clamp(0.0, 1.0))

        return loss_kp + 0.1 * conf_loss


# ===========================================================================
# Activity Loss — LDAM-DRW + CB-Focal (Doc 02 C.2)
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

    def _compute_margins(self, cls_num_list: np.ndarray) -> torch.Tensor:
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (self.max_m / m_list.max())
        return torch.tensor(m_list, dtype=torch.float32)

    def set_class_counts(self, counts):
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

        m_list = self._compute_margins(
            self.class_weights.cpu().numpy()
        ).to(device)

        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        batch_m = m_list[targets].view(-1, 1)
        x_m = logits - batch_m * index.float()

        if epoch >= drw_epoch and self.cb_weights is not None:
            w = self.cb_weights.to(device)[targets]
        else:
            w = torch.ones(B, device=device)

        return (w * F.cross_entropy(self.s * x_m, targets, reduction='none')).mean()


class ClassBalancedFocalLoss(nn.Module):
    """
    CB Focal Loss (Cui et al., 2019) for 74-class activity recognition.
    Effective number of samples: E(n) = (1 - β^n) / (1 - β)
    Weights: w_c = 1 / E(n_c)
    FL = (1 - p_t)^γ * CE_with_weights
    With optional label smoothing for better generalization.

    Doc 02 C.2: LDAM-DRW is preferred for long-tail IndustReal classes.
    """
    def __init__(self, num_classes: int = 74, beta: float = 0.999, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
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


def binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Binary Focal Loss for PSR (Doc 02 C.3).

    PSR has heavy class imbalance per component (component appears in <30% of frames).
    BCE WithLogitsLoss struggles with this; focal loss down-weights easy negatives.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    where p_t = p if y=1 else 1-p
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
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

    Doc 02 improvements:
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

        # Kendall log variances (log σ²)
        self.log_var_det = nn.Parameter(torch.zeros(1))
        self.log_var_pose = nn.Parameter(torch.tensor([-1.0]))
        self.log_var_act = nn.Parameter(torch.zeros(1))
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

        self.head_pose_loss_fn = nn.MSELoss(reduction='mean')

        self._psr_temporal_smooth_weight = float(getattr(C, 'PSR_TEMPORAL_SMOOTH_WEIGHT', 0.05))
        self._psr_temporal_history: Dict[str, List[torch.Tensor]] = {}

        self._act_warmup_epochs = int(getattr(C, 'ACT_WARMUP_EPOCHS', 5))
        self._current_epoch = 0

    def set_class_counts(self, counts):
        self.act_loss_fn.set_class_counts(counts)

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        device = outputs['cls_preds'].device
        zero = torch.tensor(0.0, device=device)

        # === Detection ===
        if self.train_det:
            cls_loss, reg_loss = self.det_loss_fn(
                outputs['cls_preds'], outputs['reg_preds'],
                outputs['anchors'], targets['detection'],
            )
            loss_det = cls_loss + reg_loss
        else:
            cls_loss = reg_loss = loss_det = zero

        # === Pose (Wing Loss on keypoints) ===
        if self.train_pose and 'keypoints' in targets:
            loss_pose = self.pose_loss_fn(
                outputs['keypoints'],
                outputs['pose_confidence'],
                targets['keypoints'],
                targets['pose_confidence'],
            )
        else:
            loss_pose = zero

        # === Activity ===
        if self.train_act and 'activity' in targets:
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

        # Activity warmup ramp
        act_ramp = min(1.0, self._current_epoch / max(self._act_warmup_epochs, 1))
        loss_act = loss_act * act_ramp

        # === PSR ===
        if self.train_psr and 'psr_labels' in targets:
            # C.3: Binary focal loss for PSR
            if self.use_psr_focal:
                loss_psr = binary_focal_loss(
                    outputs['psr_logits'],
                    targets['psr_labels'],
                    alpha=self.psr_focal_alpha,
                    gamma=self.psr_focal_gamma,
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
        if 'head_pose' in outputs and outputs['head_pose'] is not None and 'head_pose' in targets:
            loss_head_pose = self.head_pose_loss_fn(
                outputs['head_pose'],
                targets['head_pose'],
            ) * 0.001
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

            total = torch.tensor(0.0, device=device)
            if self.train_det:
                total = total + prec_det * loss_det + lv_det
            if self.train_pose:
                total = total + prec_hp * loss_pose + lv_hp
            else:
                total = total + prec_hp * loss_head_pose + lv_hp
            if self.train_act:
                total = total + prec_act * loss_act + lv_act
            if self.train_psr:
                total = total + prec_psr * loss_psr + lv_psr
            total = total.squeeze()
        else:
            prec_det = prec_pose = prec_act = prec_psr = torch.tensor(1.0, device=device)
            total = loss_det + loss_act + loss_psr + loss_head_pose

        # Normalized weights for logging
        with torch.no_grad():
            wd = prec_det.item()
            wp = prec_hp.item()
            wa = prec_act.item()
            wps = prec_psr.item()
            ws = wd + wp + wa + wps + 1e-8

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