import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Detection Anchor Matching Probe (drop-in diagnostic)
# =============================================================================
def probe_anchor_matching(
    matched_labels, max_iou=None, num_gt: int = 0, img_idx: int = 0, every: int = 200, _state: dict = None,
) -> None:
    if _state is None:
        _state = {}
    _state["n"] = _state.get("n", 0) + 1
    if every > 0 and _state["n"] % every != 0:
        return
    pos = int((matched_labels >= 0).sum())
    neg = int((matched_labels == -2).sum())
    ign = int((matched_labels == -1).sum())
    extra = ""
    if max_iou is not None:
        extra = f" max_iou[p50/p99/max]={float(max_iou.float().median()):.3f}/" \
                f"{float(max_iou.float().quantile(0.99)):.3f}/{float(max_iou.float().max()):.3f}"
    msg = (f"[MATCH_PROBE call={_state['n']} img={img_idx}] "
           f"num_gt={num_gt} pos={pos} neg={neg} ignore={ign}{extra}")
    logger.info(msg)
    print(msg, flush=True)

"""
POPW Loss Functions — Matches paper architecture
==================================================
L_det  = Focal Loss (α=0.25, γ=2) + GIoU (paper §3.2)
L_pose = Wing Loss (ω=0.05, ε=0.005) (paper §3.3)
L_act  = CE + label_smooth(0.1) (paper §3.7.1)
L_psr  = Binary Focal Loss (α=0.25, γ=2.0) (paper §3.6)
L_total = Kendall(s_det, s_pose, s_act, s_psr) with act_ramp = min(1, epoch/5)
init: s_det=0, s_pose=-1, s_act=0, s_psr=0. Clamp[-4,2].

Author: Bashara
Date: April 2026
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# MTL loss weighting modules (optional alternatives to Kendall weighting)
from src.losses.famo import FAMOWeighter
from src.losses.imtl_l import imtl_l_loss
from src.losses.rlw import RLWWeighter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss

from src import config as C
from src.losses.varifocal_loss import VarifocalLoss
from src.losses.wiou_loss import wiou_v3_loss


def _get_kendall_stage(epoch: int) -> int:
    """Use train.get_stage() so reinit_epoch_offset is respected.

    This was a duplicate of train.get_stage() that didn't incorporate
    the _REINIT_EPOCH_OFFSET, causing Kendall weighting to apply Stage 3
    when the training loop applied Stage 1 — Kendall log_vars drifted
    incorrectly during recovery retraining.
    """
    from training.train import get_stage
    return get_stage(epoch)


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
                 pos_iou_thresh: float = 0.5, neg_iou_thresh: float = 0.4,
                 class_alphas: Optional[Dict[int, float]] = None,
                 use_varifocal: bool = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        # [FIX 2026-06-20] Per-class alpha for fine-grained detection classes.
        # Stored as {class_id: alpha}. Applied at the alpha_t step in forward().
        self.class_alphas = class_alphas or {}
        # [VFL 2026-07-14] VarifocalLoss replaces focal for detection cls.
        self.use_varifocal = use_varifocal
        if use_varifocal:
            self.varifocal_loss = VarifocalLoss(alpha=alpha, gamma=gamma)

    def _match_anchors(self, anchors: torch.Tensor, gt_boxes: torch.Tensor,
                      gt_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign anchors to GT boxes. Returns labels, matched boxes, and max IoU.

        BUG FIX #1: Normalize both anchors and GT boxes to [0,1] image coords
        before IoU matching. GT boxes from COCO are in pixel coordinates
        (absolute pixels), but anchors are generated at full resolution (1280x720).
        Without normalization, max IoU = 0.0001 << 0.5 threshold → zero positive
        matches → detection loss = 0.000 → no learning.
        """
        N = anchors.shape[0]
        device = anchors.device

        if gt_boxes.shape[0] == 0:
            return (torch.full((N,), -2, dtype=torch.long, device=device),
                    torch.zeros((N, 4), device=device),
                    torch.zeros(N, device=device))

        # --- Normalize anchors and GT boxes to [0,1] before IoU matching ---
        # Anchors: shifts in pixels → divide by image dimensions
        anchors_norm = anchors.clone()
        anchors_norm[:, [0, 2]] = anchors[:, [0, 2]] / C.IMG_WIDTH   # x coords
        anchors_norm[:, [1, 3]] = anchors[:, [1, 3]] / C.IMG_HEIGHT  # y coords

        # GT boxes: already in pixel xyxy → normalize to [0,1]
        gt_boxes_norm = gt_boxes.clone()
        gt_boxes_norm[:, [0, 2]] = gt_boxes[:, [0, 2]] / C.IMG_WIDTH
        gt_boxes_norm[:, [1, 3]] = gt_boxes[:, [1, 3]] / C.IMG_HEIGHT

        ious = box_iou(anchors_norm, gt_boxes_norm)
        max_iou, max_idx = ious.max(dim=1)

        labels = torch.full((N,), -1, dtype=torch.long, device=device)
        matched_boxes = gt_boxes[max_idx]

        labels[max_iou < self.neg_iou_thresh] = -2
        pos_mask = max_iou >= self.pos_iou_thresh
        labels[pos_mask] = gt_labels[max_idx[pos_mask]]

        # [FIX 2026-06-20 (Opus v8 §3 Fix 2)] Top-k force-match per GT.
        # Standard RetinaNet force-matches the single best anchor per GT (~1 pos/GT).
        # For small assembly parts (h≈156px at 720p) at IoU≥0.4, most GT boxes only
        # match 1-2 anchors above threshold. Top-k force-match gives ~6-10 pos/GT,
        # fixing the supply-side root cause of gradient starvation at source.
        _topk = int(getattr(C, 'DET_POS_IOU_TOP_K', 9))
        _iou_floor = float(getattr(C, 'DET_POS_IOU_IOU_FLOOR', 0.0))
        for gi in range(gt_boxes.shape[0]):
            gi_ious = ious[:, gi]
            # Always assign the best anchor (standard RetinaNet behavior, no floor)
            labels[gi_ious.argmax()] = gt_labels[gi]
            # Then assign top-k above IoU floor (minus the best, already assigned above)
            # [OPUS v9 §R2] Without the IoU floor, top-k can label near-zero-IoU anchors
            # as "positive", injecting label noise into the cls head. At 0.2, only anchors
            # with ≥20% box overlap get positive labels from the force-match.
            if _topk > 1 and gi_ious.numel() > _topk:
                _, topk_idx = gi_ious.topk(min(_topk, gi_ious.numel()), largest=True)
                for idx in topk_idx.tolist():
                    if labels[idx] < 0:  # don't overwrite an already-positive anchor
                        if _iou_floor <= 0 or gi_ious[idx].item() >= _iou_floor:
                            labels[idx] = gt_labels[gi]

        return labels, matched_boxes, max_iou

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
        total_reg_sum = torch.tensor(0.0, device=device)  # [A5] Accumulate sum, not per-image mean
        total_reg_cnt = 0                                 # [A5] Count of positive boxes for global mean
        n_img_with_gt = 0

        for i in range(B):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)

            # [RC-28 FIX 2026-06-12 + 2026-06-15 v2] Bounded background loss for
            # empty frames instead of skip. The original RC-28 fix (skip via
            # `continue`) eliminated the massive ~130-200 negative gradient from
            # 173K anchors, but left the detection head with gradient on only
            # ~0.7% of batches — causing weight drift and DEAD det loss. The v2
            # fix subsamples DET_EMPTY_SAMPLE (512) anchor locations and computes
            # a scaled background focal loss, producing ~0.005-0.9 loss per
            # empty frame — enough to prevent weight drift without the collapse
            # that the original RC-28 was protecting against.
            if gt_boxes.shape[0] == 0:
                n_anc = anchors.shape[0]
                n_sample = min(n_anc, C.DET_EMPTY_SAMPLE)
                idx = torch.randperm(n_anc, device=device)[:n_sample]
                bg_cls = cls_preds[i][idx]
                bg_target = torch.zeros_like(bg_cls)
                bg_p = torch.sigmoid(bg_cls).clamp(1e-7, 1.0 - 1e-7)
                bg_ce = F.binary_cross_entropy_with_logits(bg_cls, bg_target, reduction='none')
                bg_p_t = bg_p * bg_target + (1 - bg_p) * (1 - bg_target)
                bg_focal = (1 - self.alpha) * (1 - bg_p_t).pow(self.gamma) * bg_ce
                bg_loss = bg_focal.sum() * C.DET_EMPTY_BG_SCALE
                total_cls = total_cls + bg_loss
                # [A5] No regression loss for empty frames — skip reg accumulation entirely.
                continue
            n_img_with_gt += 1

            matched_labels, matched_boxes, matched_iou = self._match_anchors(anchors, gt_boxes, gt_labels)
            probe_anchor_matching(matched_labels, num_gt=gt_boxes.shape[0], img_idx=i)

            pos_mask = matched_labels >= 0
            neg_mask = matched_labels == -2
            valid_mask = pos_mask | neg_mask
            num_pos = max(pos_mask.sum().item(), 1)

            cls_pred = cls_preds[i][valid_mask]
            cls_target = torch.zeros_like(cls_pred)

            # [OPUS v9 §R3] Positive-anchor score probe: log sigmoid scores of positive anchors
            # to detect cls head collapse (scores → 0 despite force-match positives).
            _pos_probe_every = int(getattr(C, 'DET_POS_ANCHOR_PROBE_EVERY', 0))
            if _pos_probe_every > 0 and pos_mask.sum() > 0:
                _ppc = self._probe_ctr = getattr(self, '_probe_ctr', 0) + 1
                if _ppc % _pos_probe_every == 0:
                    with torch.no_grad():
                        _pp_iv = pos_mask[valid_mask]
                        _pp_lab = matched_labels[valid_mask][_pp_iv]
                        _pp_scores = torch.sigmoid(cls_pred[_pp_iv]).gather(1, _pp_lab.unsqueeze(1)).squeeze(1)
                        _pp_msg = (f'[POS_ANCHOR_PROBE img={i} call={_ppc}] '
                                   f'n_pos={int(pos_mask.sum().item())} '
                                   f'mean={_pp_scores.mean().item():.4f} '
                                   f'med={_pp_scores.median().item():.4f} '
                                   f'max={_pp_scores.max().item():.4f} '
                                   f'min={_pp_scores.min().item():.4f}')
                        logger.warning(_pp_msg)
                        print(_pp_msg, flush=True)

            if pos_mask.sum() > 0:
                pos_in_valid = pos_mask[valid_mask]
                pos_labels = matched_labels[valid_mask][pos_in_valid]
                num_det_classes = cls_target.shape[1]
                if pos_labels.max() >= num_det_classes or pos_labels.min() < 0:
                    logger.warning(
                        f'Out-of-range detection labels: min={pos_labels.min().item()}, '
                        f'max={pos_labels.max().item()}, num_classes={num_det_classes}'
                    )
                    pos_labels = pos_labels.clamp(0, num_det_classes - 1)
                cls_target[pos_in_valid, pos_labels] = 1.0

                # [VFL 2026-07-14] VarifocalLoss uses IoU for positive targets
                if self.use_varifocal:
                    _pos_iou_vals = matched_iou[valid_mask][pos_in_valid].clamp(0.0, 1.0)
                    cls_target[pos_in_valid, pos_labels] = _pos_iou_vals

            # --- OHEM: Subsample negatives to prevent class imbalance collapse ---
            # With ~0.01% positive anchors in detection, cumulative gradient from 173K
            # negatives per image drives cls logits to -16 over ~850 steps (collapse).
            # Keep all positives + top-K hardest negatives (K = n_pos * DET_OHEM_RATIO).
            if getattr(C, 'DET_OHEM_ENABLED', False):
                ohem_ratio = getattr(C, 'DET_OHEM_RATIO', 3.0)
                ohem_min_neg = getattr(C, 'DET_OHEM_MIN_NEG', 64)
                neg_in_valid = neg_mask[valid_mask]
                n_pos_val = pos_mask.sum().item()
                n_neg_all = neg_in_valid.sum().item()
                with torch.no_grad():
                    ce_pre = F.binary_cross_entropy_with_logits(
                        cls_pred, cls_target, reduction='none'
                    )
                    per_anchor_loss = ce_pre.sum(dim=1)  # [valid_anchors] sum over 24 classes
                n_keep = max(n_pos_val * ohem_ratio, ohem_min_neg)
                n_keep = min(int(n_keep), n_neg_all)
                if n_neg_all > n_keep:
                    valid_indices = torch.arange(cls_pred.shape[0], device=cls_pred.device)
                    neg_indices = valid_indices[neg_in_valid]
                    neg_losses = per_anchor_loss[neg_in_valid]
                    _, topk_idx = torch.topk(neg_losses, n_keep)
                    keep_neg = neg_indices[topk_idx]
                    ohem_mask = torch.zeros(cls_pred.shape[0], dtype=torch.bool,
                                            device=cls_pred.device)
                    pos_in_valid = pos_mask[valid_mask]
                    ohem_mask[pos_in_valid] = True
                    ohem_mask[keep_neg] = True
                    cls_pred = cls_pred[ohem_mask]
                    cls_target = cls_target[ohem_mask]

            # [VFL 2026-07-14] VarifocalLoss (Zhang et al.) replaces focal for detection cls.
            # VFL only down-weights negative samples; positive samples get full weight
            # with IoU-based target values (already set in target construction above).
            # Asymmetric gamma and per-class alpha do not apply in VFL mode.
            if self.use_varifocal:
                total_cls = total_cls + self.varifocal_loss(cls_pred, cls_target)
            else:
                # --- FIX #1: Clamp sigmoid inputs to prevent NaN in focal loss ---
                # p_t near 0 causes (1-p_t)^gamma → inf and log(0) → NaN
                p = torch.sigmoid(cls_pred).clamp(1e-7, 1.0 - 1e-7)
                ce = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='none')
                p_t = p * cls_target + (1 - p) * (1 - cls_target)

                # Asymmetric gamma: positives get less/no suppression to prevent cls mean collapse
                if getattr(C, 'DET_ASYMMETRIC_GAMMA', False):
                    gamma_pos = getattr(C, 'DET_GAMMA_POS', 0.0)
                    gamma_neg = getattr(C, 'DET_GAMMA_NEG', self.gamma)
                    gamma_eff = gamma_pos * cls_target + gamma_neg * (1 - cls_target)
                else:
                    gamma_eff = self.gamma

                # [FIX 2026-06-20] Per-class alpha: override default alpha for specific classes
                # to break gradient conflicts from fine-grained class ambiguity (e.g., class_6 vs class_7).
                # alpha_per_class[class_id] when set, else self.alpha.
                if self.class_alphas:
                    num_det_classes = cls_target.shape[1]
                    base_alpha_arr = torch.full((num_det_classes,), self.alpha, device=cls_target.device)
                    for cid, ca in self.class_alphas.items():
                        if 0 <= cid < num_det_classes:
                            base_alpha_arr[cid] = ca
                    base_alpha_arr = base_alpha_arr.unsqueeze(0)  # [1, 24]
                    alpha_t = base_alpha_arr * cls_target + (1 - base_alpha_arr) * (1 - cls_target)
                else:
                    alpha_t = self.alpha * cls_target + (1 - self.alpha) * (1 - cls_target)
                total_cls = total_cls + (alpha_t * (1 - p_t) ** gamma_eff * ce).sum() / num_pos

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

                if getattr(C, 'USE_WIOU', False):
                    # WIoU v3 — dynamic non-monotonic focusing (Tong et al. 2023)
                    reg_loss = wiou_v3_loss(pred_boxes, gt_boxes_pos, pos_anchors)
                    # wiou_v3_loss returns mean; scale to sum for consistent accumulation
                    reg_loss = reg_loss * num_pos
                else:
                    reg_loss = generalized_box_iou_loss(
                        pred_boxes, gt_boxes_pos, reduction='sum'
                    )
                    # Guard NaN GIoU (happens when boxes don't overlap at all)
                    reg_loss = torch.where(
                        torch.isfinite(reg_loss),
                        reg_loss,
                        torch.tensor(0.0, device=device),
                    )
                # [A5 FIX 2026-06-17] Accumulate sum and count for single global mean.
                # Per-image mean + mean-across-images dilutes gradient from dense-positive
                # frames (2 positives get same weight as 50). Single global mean preserves
                # proportional contribution from each positive box.
                total_reg_sum = total_reg_sum + reg_loss
                total_reg_cnt = total_reg_cnt + num_pos

        # [RC-28 FIX 2026-06-12] Normalize by the number of images that
        # actually contributed (GT-bearing), not the full batch size — with
        # empty images skipped, dividing by B would shrink the det gradient
        # by the empty-frame fraction (~85%) for no reason.
        # [A5 FIX 2026-06-17] Regression uses single global mean (sum/count) instead of
        # two sequential means, preserving proportional contribution from dense-positive frames.
        return total_cls / max(n_img_with_gt, 1), total_reg_sum / max(total_reg_cnt, 1)


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
        """Set per-class sample counts for LDAM margin / CB-Focal reweighting.

        Accepts counts of length self.num_classes-1, self.num_classes, or
        self.num_classes+1 (with explicit reconciliation in each case). The
        old code silently passed an over-long array through to copy_(),
        producing a Torch size-mismatch error on the full dataset whenever
        the data contained the highest action id.
        """
        if counts is None:
            self.class_weights.data.fill_(1.0 / self.num_classes)
            self._raw_counts = None
            self._margins = None
            return

        counts = np.asarray(counts, dtype=np.float64).reshape(-1)
        n_in, n_want = counts.shape[0], self.num_classes

        if n_in == n_want - 1:
            # caller forgot the NA slot; prepend a sentinel of 1 (not 0; the
            # margin formula 1/√√n needs n > 0 to avoid huge spurious margins)
            counts = np.concatenate([[1.0], counts])
        elif n_in != n_want:
            # Length disagrees with self.num_classes (the NUM_CLASSES_ACT
            # 74-vs-75 hazard). This is NOT fatal: forward() resizes every
            # per-class vector to the actual logits width via _fit_to_width,
            # so we keep the counts at their natural length and only warn.
            logger.warning(
                f'LDAMLoss.set_class_counts: got {n_in} entries but '
                f'num_classes={n_want}. Keeping natural length; margins/weights '
                f'are re-aligned to the logits width at forward time.'
            )

        effective = np.where(
            counts > 0,
            (1.0 - np.power(0.999, counts)) / (1.0 - 0.999),
            1.0,
        )
        weights = 1.0 / np.maximum(effective, 1e-8)
        weights = weights / weights.sum() * counts.shape[0]
        # The class_weights buffer is fixed-size; only copy when shapes match
        # (it is informational — forward() reads _raw_counts / cb_weights).
        if weights.shape[0] == self.class_weights.shape[0]:
            self.class_weights.data.copy_(torch.tensor(weights, dtype=torch.float32))

        self._raw_counts = counts.copy()
        self._raw_counts[0] = max(self._raw_counts[0], 1.0)  # avoid 1/√√0 at NA
        self._margins = None  # force recompute on next forward

        # [OPUS FIX #2] Wire cb_weights for DRW when LDAM_USE_DRW is True.
        # DRW needs class-balanced re-weighting: w_c = 1/E(n_c).
        # When LDAM_USE_DRW=False, LDAM uses margins only (no CB re-weighting) — for A/B testing.
        if bool(C.LDAM_USE_DRW):
            # Store as torch tensor — forward() calls .to(device) on it directly.
            self.cb_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.cb_weights = None

    @staticmethod
    def _fit_to_width(arr: np.ndarray, width: int, pad_value: float = 1.0) -> np.ndarray:
        """Return ``arr`` resized to exactly ``width`` entries.

        Pads with ``pad_value`` if too short, truncates if too long. This keeps
        per-class vectors (margins, weights) aligned with the *actual* logits
        width at forward time, even when ``self.num_classes`` was computed from a
        different source than the data (the NUM_CLASSES_ACT 74-vs-75 hazard).
        """
        n = arr.shape[0]
        if n == width:
            return arr
        if n < width:
            return np.concatenate([arr, np.full(width - n, pad_value, dtype=arr.dtype)])
        return arr[:width]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch: int = 0, drw_epoch: int = 60) -> torch.Tensor:
        device = logits.device
        B, C = logits.shape

        is_soft_labels = (targets.dim() == 2 and targets.shape[1] == C)
        if is_soft_labels:
            hard_targets = targets.argmax(dim=1)
        else:
            hard_targets = targets.long()

        # [ROBUSTNESS FIX — activity label range] ----------------------------
        # On the full IndustReal dataset, NUM_CLASSES_ACT is computed by scanning
        # AR_labels.csv on disk; the raw bincount that feeds set_class_counts and
        # the model's output width can disagree by one (the 37/64 dead-channel /
        # NA-prepend ambiguity). If any target lands one-past the logits width,
        # the original `m_list[hard_targets]` and `scatter_` calls raise a cryptic
        # CUDA device-side assert that kills the 100-epoch run. We instead size
        # every per-class vector to the *observed* logits width C and clamp any
        # out-of-range target into valid range, warning exactly once so the data
        # mapping bug stays visible without aborting training.
        if hard_targets.numel() > 0:
            t_max = int(hard_targets.max().item())
            t_min = int(hard_targets.min().item())
            if t_max >= C or t_min < 0:
                if not getattr(self, '_warned_oob_target', False):
                    logger.warning(
                        'LDAMLoss: activity target out of range '
                        f'(min={t_min}, max={t_max}, logits width C={C}, '
                        f'num_classes={self.num_classes}). Clamping to [0, {C - 1}]. '
                        'This indicates a NUM_CLASSES_ACT / label-mapping mismatch — '
                        'see AUDIT_REPORT.md "Activity class count".'
                    )
                    self._warned_oob_target = True
                hard_targets = hard_targets.clamp_(0, C - 1)

        # Margins are built to width C (not self.num_classes) so they always
        # align with the logits, regardless of how counts were registered.
        raw = self._raw_counts if self._raw_counts is not None else np.ones(C)
        m_list = self._compute_margins(self._fit_to_width(raw, C)).to(device)

        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, hard_targets.view(-1, 1), True)
        batch_m = m_list[hard_targets].view(-1, 1)
        x_m = logits - batch_m * index.float()
        # Clamp x_m to prevent overflow when s=30 and logits are large
        x_m = x_m.clamp(-10.0, 10.0)

        if epoch >= drw_epoch and self.cb_weights is not None:
            cb = self.cb_weights.to(device)
            if cb.shape[0] != C:  # keep DRW weights aligned with logits width
                cb = torch.from_numpy(
                    self._fit_to_width(cb.cpu().numpy(), C)
                ).to(device)
            w = cb[hard_targets]
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
        self._warned_oob_target = False

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
            # [ROBUSTNESS FIX] — clamp targets to valid range before scatter_
            # Prevents CUDA ScatterGatherKernel.cu:409 assertion when validation
            # batch contains activity labels outside the logits width (same root
            # cause as LDAMLoss guard at line 522).
            tgt = targets.long()
            if tgt.numel() > 0:
                C = logits.shape[1]
                if tgt.max() >= C or tgt.min() < 0:
                    if not getattr(self, '_warned_oob_target', False):
                        logger.warning(
                            'CBFocalLoss: activity target out of range '
                            f'(min={tgt.min().item()}, max={tgt.max().item()}, '
                            f'C={C}). Clamping to [0, {C - 1}].'
                        )
                        self._warned_oob_target = True
                    tgt = tgt.clamp(0, C - 1)
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / self.num_classes)
                smooth_targets.scatter_(1, tgt.unsqueeze(1), 1.0 - self.label_smoothing)

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

        # Clamp max weight ratio to prevent catastrophic reweighting (beta=0.999 caused 10,000x amplification)
        w_min = self.class_weights.min()
        ratios = self.class_weights / w_min.clamp(min=1e-8)
        clamp_mask = ratios > 50.0
        if clamp_mask.any():
            w = torch.where(clamp_mask[targets], w_min * 50.0, w)

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
                       per_component_alpha: torch.Tensor = None,
                       comp_weights: torch.Tensor = None) -> torch.Tensor:
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
        comp_weights: [C] tensor of per-component loss multipliers. Applied after
            focal computation, before reduction. Normalized by mean weight so
            total loss scale is preserved. Rare components (low prevalence) get
            higher weight to prevent DEAD bias gradients.
    """
    # --- FIX: Clamp logits to gradient-safe range BEFORE sigmoid.
    # Logits beyond ±8 produce saturated sigmoid (≈0 or ≈1) that destabilizes
    # focal loss: p_t≈0 → log(p_t)=-inf → NaN. ±8 is safe: exp(-8)≈0.00034,
    # exp(8)≈2981 (well below fp16 overflow ~65504). Gradient flows through clamp.
    _logit_safe = logits.clamp(min=-8.0, max=8.0)
    p = torch.sigmoid(_logit_safe)
    ce = F.binary_cross_entropy_with_logits(_logit_safe, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)

    # --- FIX: Mask -1 (error/aborted state) targets as ignore ---
    # PSR labels contain -1 for error/aborted states (_parse_psr_raw fills -1 for
    # error transitions). With target=-1: p_t = 2-3p (can be negative), making
    # (1-p_t)^gamma numerically unstable. Also, alpha_t = 2*target - 1 would produce
    # alpha_t=1 (not 0) for target=-1. Mask these as ignore BEFORE computing alpha_t.
    if (targets < 0).any():
        ignore_mask = (targets < 0).float()
        # Temporarily set -1 → 0 so alpha_t formula works correctly on masked entries
        targets_safe = targets.clone().masked_fill_(ignore_mask.bool(), 0)
    else:
        ignore_mask = None
        targets_safe = targets

    if per_component_alpha is not None:
        alpha_c = per_component_alpha.to(logits.device).unsqueeze(0).clamp(max=1.0)
        alpha_t = alpha_c * targets_safe + (1 - alpha_c) * (1 - targets_safe)
    else:
        alpha_t = alpha * targets_safe + (1 - alpha) * (1 - targets_safe)

    # Apply ignore mask: zero out alpha_t and CE contribution for -1 entries
    if ignore_mask is not None:
        alpha_t = alpha_t * (1 - ignore_mask)  # 0 for -1 targets
        # p_t=1 → (1-1)^gamma=0, ce=0 → loss=0 for -1 targets
        p_t = p_t.masked_fill(ignore_mask.bool(), 1.0)
        ce = ce.masked_fill(ignore_mask.bool(), 0.0)

    # --- FIX: Clamp p_t to prevent NaN in focal loss ---
    # p_t near 0 → (1-0)^gamma = 1, but -log(0) = inf → NaN
    # p_t near 1 → (1-1)^gamma = 0, log(1) = 0, but clamp is safe
    p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)
    per_elem = alpha_t * (1 - p_t) ** gamma * ce

    # [FIX 2026-06-15] Per-component PSR loss weighting
    # Multiply per-element loss by per-component weights so rare steps
    # contribute more gradient. Normalize by mean weight to preserve scale.
    if comp_weights is not None and per_elem.dim() > 1:
        cw = comp_weights.to(logits.device).unsqueeze(0)  # [1, C] broadcast over batch
        weight_mean = cw.mean()
        per_elem = per_elem * cw / weight_mean

    if ignore_mask is not None:
        valid_mask = (1 - ignore_mask).bool()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=logits.requires_grad)
        loss = per_elem.masked_select(valid_mask).mean()
    else:
        loss = per_elem.mean()

    # --- DIAGNOSTIC: expose real PSR loss before any sentinel ---
    # 18,615/18,635 steps show psr=0.0001000 but zero PSR_NAN warnings.
    # Diagnostic: print whenever loss is non-finite OR suspiciously small.
    # Remove this block after confirming the -1 fraction on real data.
    _n_neg1 = int((targets < 0).sum())
    _total = targets.numel()
    _valid = _total - _n_neg1
    _suspicious = (not torch.isfinite(loss).all()) or (float(loss.detach()) < 1e-4)
    if _suspicious:
        _msg = (
            f"[PSR_DIAG] loss={float(loss.detach()):.3e} finite={bool(torch.isfinite(loss).all())} | "
            f"shape={tuple(targets.shape)} total={_total} valid={_valid} neg1={_n_neg1} "
            f"(neg1_frac={_n_neg1 / max(_total, 1):.3f}) | "
            f"logits[min/max/mean]={float(logits.min()):.3f}/"
            f"{float(logits.max()):.3f}/{float(logits.mean()):.3f} | "
            f"target counts: zeros={int((targets == 0).sum())} "
            f"ones={int((targets == 1).sum())} neg1={_n_neg1} | "
            f"per_elem[min/max/sum]={float(per_elem.min()):.3e}/"
            f"{float(per_elem.max()):.3e}/{float(per_elem.sum()):.3e}"
        )
        logger.warning(_msg)
        print(_msg, flush=True)

    return loss


# =============================================================================
# Head Pose Split Loss (two-term: position MSE + normalized direction MSE)
# =============================================================================
def head_pose_loss_split(
    pred: torch.Tensor, target: torch.Tensor,
    pos_weight: float = 1.0, dir_weight: float = 1.0,
    norm_reg_weight: float = 0.0, eps: float = 1e-6,
) -> torch.Tensor:
    """
    Two-term head-pose loss. Position target must be standardized to O(1)
    via HEAD_POSE_POS_SCALE in the dataset (industreal_dataset.py _parse_pose).
    Direction term uses L2-normalized vectors so it is scale-invariant.
    """
    fwd_p, pos_p, up_p = pred[:, 0:3], pred[:, 3:6], pred[:, 6:9]
    fwd_t, pos_t, up_t = target[:, 0:3], target[:, 3:6], target[:, 6:9]
    pos_loss = F.mse_loss(pos_p, pos_t)
    fwd_pn = F.normalize(fwd_p, dim=1, eps=eps)
    up_pn = F.normalize(up_p, dim=1, eps=eps)
    fwd_tn = F.normalize(fwd_t, dim=1, eps=eps)
    up_tn = F.normalize(up_t, dim=1, eps=eps)
    dir_loss = F.mse_loss(fwd_pn, fwd_tn) + F.mse_loss(up_pn, up_tn)
    total = pos_weight * pos_loss + dir_weight * dir_loss
    if norm_reg_weight > 0.0:
        fwd_norm = fwd_p.norm(dim=1)
        up_norm = up_p.norm(dim=1)
        norm_reg = ((fwd_norm - 1.0) ** 2 + (up_norm - 1.0) ** 2).mean()
        total = total + norm_reg_weight * norm_reg
    return total


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
        use_famo: bool = False,
        use_imtl_l: bool = False,
        use_rlw: bool = False,
    ):
        super().__init__()
        self.train_det = train_det
        self.train_pose = train_pose
        self.train_act = train_act
        self.train_psr = train_psr
        self.use_kendall = use_kendall
        self.use_famo = use_famo
        self.use_imtl_l = use_imtl_l
        self.use_rlw = use_rlw

        # MTL loss weighter instances (created on first forward to know device)
        self.famo_weighter: Optional[FAMOWeighter] = None
        self.rlw_weighter: Optional[RLWWeighter] = None
        self._mtl_task_names: List[str] = []

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
        # Fix (Opus #1): s_pose=0 instead of -1 to prevent Kendall clamp(min=0) from
        # zeroing the pose Kendall term at init. With s=-1: exp(-(-1))=exp(1)=2.7 prec
        # and lv=-1 → term = 2.7*loss + (-1) which can be negative when loss is tiny
        # (e.g. loss_pose ~0.0001 → 2.7*0.0001-1 = -0.9997). clamp(min=0) zeroes it.
        # s=0 gives prec=1.0 and lv=0 → term = loss + 0 (always ≥ 0).
        self.log_var_pose = nn.Parameter(torch.tensor([0.0]))
        self.log_var_act = nn.Parameter(torch.zeros(1))  # Paper §Multi-Task Loss: init [0,-1,0,0] → use zeros(1) per spec
        self.log_var_psr = nn.Parameter(torch.zeros(1))

        # Sub-losses
        self.det_loss_fn = FocalLoss(alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA,
                                      pos_iou_thresh=C.DET_POS_IOU_THRESH,
                                      neg_iou_thresh=C.DET_NEG_IOU_THRESH,
                                      class_alphas=getattr(C, 'DET_CLASS_ALPHAS', {}),
                                      use_varifocal=bool(getattr(C, 'USE_VARIFOCAL', False)))
        self.pose_loss_fn = PoseLoss(wing_omega=C.WING_OMEGA, wing_epsilon=C.WING_EPSILON)

        # C.2: Per paper §3.7.1 — "CE (label_smooth=0.1)" for activity.
        # LDAM-DRW is available as an ablation via USE_LDAM_DRW=True.
        # [OPUS DECISION 2] USE_CB_FOCAL_ACT=True switches to class-balanced focal
        # loss (Cui et al., 2019) for the activity head when CE collapses to 1 class.
        use_ldam = bool(getattr(C, 'USE_LDAM_DRW', False))
        use_cb_focal = bool(getattr(C, 'USE_CB_FOCAL_ACT', False))
        if use_ldam:
            self.act_loss_fn = LDAMLoss(
                num_classes=num_classes_act,
                max_m=float(getattr(C, 'LDAM_MAX_M', 0.5)),
                s=float(getattr(C, 'LDAM_S', 30)),
            )
        elif use_cb_focal:
            self.act_loss_fn = ClassBalancedFocalLoss(
                num_classes=num_classes_act,
                beta=float(getattr(C, 'CB_FOCAL_BETA', 0.999)),
                gamma=float(getattr(C, 'CB_FOCAL_GAMMA', 2.0)),
                label_smoothing=getattr(C, 'CB_LABEL_SMOOTHING', 0.1),
            )
        else:
            self.act_loss_fn = nn.CrossEntropyLoss(
                label_smoothing=getattr(C, 'CB_LABEL_SMOOTHING', 0.1),
            )
        self.use_ldam = use_ldam
        self.use_cb_focal = use_cb_focal

        # C.3: Binary focal loss for PSR (instead of BCE)
        self.psr_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.use_psr_focal = bool(getattr(C, 'PSR_FOCAL_GAMMA', 0) > 0)
        self.use_psr_transition = bool(getattr(C, 'USE_PSR_TRANSITION', False))  # [OPUS v5] transition objective
        self.psr_focal_alpha = float(getattr(C, 'PSR_FOCAL_ALPHA', 0.25))
        self.psr_focal_gamma = float(getattr(C, 'PSR_FOCAL_GAMMA', 2.0))
        self._psr_per_component_alpha: torch.Tensor = None
        self._psr_num_components = num_psr_components
        # [FIX 2026-06-15] Per-component PSR loss weights (11 components)
        # Inverse prevalence weighting: rarer steps get higher weight.
        _cw = torch.tensor(getattr(C, 'PSR_COMP_WEIGHTS', [1.0] * num_psr_components), dtype=torch.float32)
        self.register_buffer('_psr_comp_weights', _cw)

        self.head_pose_loss_fn = nn.MSELoss(reduction='mean')

        self._psr_temporal_smooth_weight = float(getattr(C, 'PSR_TEMPORAL_SMOOTH_WEIGHT', 0.05))
        self._psr_temporal_history: Dict[str, List[torch.Tensor]] = {}

        self._act_warmup_epochs = int(getattr(C, 'ACT_RAMP_EPOCHS', 5))
        self._current_epoch = 0
        # [FIX 2026-06-28 20-agent] Stage-local epoch counter for activity ramp.
        # Resets to 0 when train_act becomes True, so the ramp works regardless
        # of global epoch (RF3 starts at global epoch 50+, ramp would be 1.0).
        self._act_epoch_counter = -1

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state['_act_epoch_counter'] = self._act_epoch_counter
        state['_current_epoch'] = self._current_epoch
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        self._act_epoch_counter = state_dict.pop('_act_epoch_counter', -1)
        self._current_epoch = state_dict.pop('_current_epoch', 0)
        super().load_state_dict(state_dict, *args, **kwargs)

    def set_class_counts(self, counts):
        # Per paper §3.7.1: CE + label_smooth(0.1) does not use class counts.
        # Only LDAM-DRW path needs them; CE loss ignores set_class_counts.
        if hasattr(self.act_loss_fn, 'set_class_counts'):
            # counts has length 74 (excludes NA class 0); act_loss_fn expects 75
            # The FocalLoss and LDAMLoss use num_classes=75 internally but only
            # active classes 1-74 get weights; class 0 (NA/NA) gets weight 0.
            # We pass the 74 active-class counts directly; the loss handles the mismatch.
            self.act_loss_fn.set_class_counts(counts)
        # [FIX 2026-06-28 20-agent] Add class-balanced weights to CE loss.
        # Plain CE on 74-class long-tail data lets head classes dominate.
        # Compute effective number (CB beta=0.99) and invert for loss weight.
        # counts has 74 elements (excludes NA class 0).
        # NOTE 2026-06-30: The IndustReal dataset uses action_id=0 as a real action
        # ("take_short_brace", 63 frames). The config NUM_CLASSES_ACT=75 with slot 0
        # designated as "NA" is a MAPPING BUG — action_id 0 maps to slot 0 which has
        # weight=0, wasting 63 training frames. Fixed by NOT zero-weighting slot 0:
        # the CB-balanced formula naturally gives low weight to rare classes, and
        # frames with label=-1 (no annotation) are already excluded from CE loss.
        if not self.use_ldam and isinstance(self.act_loss_fn, nn.CrossEntropyLoss) and counts is not None:
            counts = torch.as_tensor(counts, dtype=torch.float32)
            _beta = float(getattr(C, 'CB_BETA', 0.99))
            _eff_num = (1.0 - _beta ** counts) / (1.0 - _beta)
            _eff_num = _eff_num.clamp(min=1.0)
            _weights = 1.0 / _eff_num
            _weights = _weights / _weights.sum() * len(_weights)  # normalize
            self.act_loss_fn = nn.CrossEntropyLoss(
                weight=_weights.to(self.log_var_det.device) if hasattr(self, 'log_var_det') else _weights,
                label_smoothing=float(getattr(C, 'CB_LABEL_SMOOTHING', 0.1)),
            )

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
        # [FIX Bug #11] Clamp alpha_c to min=0.1 to prevent starvation.
        # Component0 has prevalence=1.0 → clamped prev=0.99 → alpha_c=0.02.
        # With alpha_c=0.02, focal loss gives near-zero gradient for that component,
        # causing PSR head to collapse to trivial all-zero predictions.
        # Min-clamp ensures every component gets at least 0.1 focal weight.
        alpha_c = alpha_c.clamp(min=0.1)
        self._psr_per_component_alpha = alpha_c
        logger.debug(
            f'PSR per-component alpha: {alpha_c.numpy().round(3).tolist()} '
            f'(from prevalence {prev.numpy().round(3).tolist()})'
        )

    def set_epoch(self, epoch: int):
        # [FIX 2026-06-28 v2] Only increment _act_epoch_counter when epoch
        # actually changes (not every batch call). Otherwise set_epoch being
        # called at line 3831 + 1362 every batch completes the 5-epoch ramp
        # in 4 batches instead of 5 epochs.
        _epoch_changed = epoch != self._current_epoch
        self._current_epoch = epoch
        # [FIX 2026-06-28 20-agent] Stage-local epoch counter for activity ramp.
        # Resets to 0 when train_act first becomes True, so the 5-epoch ramp
        # works at RF3 boundary (global epoch 50+). Without this, the ramp
        # condition `epoch < ACT_RAMP_EPOCHS` is never met.
        if self.train_act:
            if self._act_epoch_counter < 0:
                self._act_epoch_counter = 0
            elif _epoch_changed:
                self._act_epoch_counter += 1
        else:
            self._act_epoch_counter = -1

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
            # [FIX 2026-06-16] Regression gradient warmup for --reinit-heads
            # When --reinit-heads is active, the regression head is freshly initialized.
            # For the first ~750 steps after reinit, no regression targets exist (only
            # classification trains). When GT boxes first appear (~step 751), reg_preds
            # are still random and produce a huge regression loss. The resulting gradient
            # shock propagates back through shared FPN features and collapses both
            # regression and classification (cls_mean drops from -2.3 to -16.68 in ~500
            # steps). This warmup linearly ramps reg_loss from init_mult → 1.0 over the
            # first N steps, allowing the regression head to stabilize gradually.
            _reinit_reg_ws = int(getattr(C, 'REINIT_REG_WARMUP_STEPS', 0))
            if _reinit_reg_ws > 0 and hasattr(self, '_step_counter') and self._step_counter < _reinit_reg_ws:
                _reinit_reg_wm = float(getattr(C, 'REINIT_REG_WARMUP_INIT_MULT', 0.01))
                _reg_ramp = _reinit_reg_wm + (1.0 - _reinit_reg_wm) * float(self._step_counter) / _reinit_reg_ws
                reg_loss = reg_loss * _reg_ramp
            loss_det = cls_loss + giou_weight * reg_loss
            # --- Historical note: NEG_SLOPE was designed to prevent a Kendall divergence
            # scenario where GIoU < 0 (poor box overlap) combined with large loss_det *
            # prec_det products. In practice generalized_box_iou_loss returns 1-GIoU
            # (always >= 0), so loss_det >= 0 always and NEG_SLOPE never fires. The
            # code is retained as a safety floor in case a future loss component changes.
            NEG_SLOPE = 0.01
            loss_det = torch.where(
                loss_det < 0,
                NEG_SLOPE * loss_det,
                loss_det,
            )
        else:
            cls_loss = reg_loss = loss_det = zero

        # NaN/inf guard on ALL individual losses before smooth cap.
        # A single batch with corrupted loss (e.g., from extreme GIoU or LDAM overflow)
        # propagates NaN into Kendall and crashes training.
        #
        # FIX: Use locals() check instead of dir() — loss_pose and loss_head_pose are
        # assigned AFTER this guard block (lines 831 and 950), so dir() always returns
        # False and they get incorrectly zeroed. locals() gives us the current vars
        # actually in scope at this point.
        _local = locals()
        loss_pose = _local.get('loss_pose', zero)
        loss_head_pose = _local.get('loss_head_pose', zero)
        # Initialize loss_act and loss_psr before the NaN guard loop
        # (they may not be assigned yet if train_act/train_psr are False,
        # but we still include them in the loop to avoid UnboundLocalError)
        loss_act = zero
        loss_psr = zero
        for _loss, _name, _zero in [
            (loss_det, 'det', zero),
            (loss_pose, 'pose', zero),
            (loss_act, 'act', zero),
            (loss_psr, 'psr', zero),
            (loss_head_pose, 'head_pose', zero),
        ]:
            if not torch.isfinite(_loss).all():
                # [A4 FIX 2026-06-17] Use torch.where to maintain grad graph connectivity.
                # Bare tensor(1e-4) creates a leaf without grad_fn, crashing backward().
                # torch.where always produces a result with grad_fn even for ALL-NaN losses.
                _fallback = torch.tensor(1e-4, device=device, dtype=output_dtype) if _name in ('det', 'pose', 'psr', 'head_pose') else _zero
                if _name == 'det':
                    loss_det = torch.where(torch.isfinite(loss_det), loss_det, _fallback)
                elif _name == 'pose':
                    loss_pose = torch.where(torch.isfinite(loss_pose), loss_pose, _fallback)
                elif _name == 'act':
                    loss_act = _fallback  # activity uses zero (not 1e-4) — already a leaf, OK
                elif _name == 'psr':
                    if getattr(C, 'ASSERT_AND_CRASH', False):
                        raise FloatingPointError(
                            f'[ASSERT_AND_CRASH] loss_psr is non-finite before Kendall assembly '
                            f'(epoch {self._current_epoch}). The 1e-4 sentinel would silently replace it.'
                        )
                    loss_psr = torch.where(torch.isfinite(loss_psr), loss_psr, _fallback)
                    logger.warning(
                        f'  [PSR_NAN_GUARD_L1041] loss_psr was non-finite before Kendall assembly '
                        f'— replaced with 1e-4 fallback (epoch {self._current_epoch})'
                    )
                elif _name == 'head_pose':
                    loss_head_pose = torch.where(torch.isfinite(loss_head_pose), loss_head_pose, _fallback)

        # --- FIX: Per-component smooth loss caps to prevent NaN cascade into Kendall.
        # A single batch with extreme loss (e.g., det=100+, pose=200+) can cause
        # Kendall divergence: exp(-lv) * loss can produce inf, which then propagates
        # through all log_vars. Soft cap formula: x if x<=cap, cap*(1+log(x/cap)) if x>cap.
        # Gradient: 1.0 below cap, cap/x above cap (never zero → no gradient death).
        def _smooth_cap(x, cap):
            # [OPUS v5 AUDIT] SIMPLIFY_LOSS (#49): bypass caps during bring-up
            if getattr(C, 'SIMPLIFY_LOSS', False):
                return x
            x_safe = x.clamp(min=1e-6, max=1e6)
            return torch.where(x > cap, cap * (1 + torch.log(x_safe / cap)), x.clamp(min=1e-6))

        det_cap = float(getattr(C, 'DET_LOSS_CAP', 50.0))
        loss_det = _smooth_cap(loss_det, det_cap)

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
            ) * float(getattr(C, 'POSE_LOSS_WEIGHT', 0.02))  # [INTERVENTION 2026-06-14] 0.02 = 20x from 0.001
        else:
            loss_pose = zero

        # --- FIX: Pose loss cap to prevent NaN cascade ---
        pose_cap = float(getattr(C, 'POSE_LOSS_CAP', 30.0))
        loss_pose = _smooth_cap(loss_pose, pose_cap)

        # [FIX 2026-07-04 Opus 111 SS3.2] Zero body-pose loss when the branch
        # is frozen — prevents dead-code loss from distorting Kendall head-pose weight.
        if getattr(C, 'FREEZE_BODY_POSE_BRANCH', False):
            loss_pose = zero  # body-pose frozen — no gradient contribution

        # === Activity ===
        if self.train_act:
            act_logits = outputs['act_logits']
            act_targets = targets['activity']
            act_mask = targets.get('activity_mask', torch.ones_like(act_targets, dtype=torch.float32))
            valid_mask = act_mask.bool()
            if valid_mask.any():
                # Filter to labeled frames only (-1 sentinel = unlabeled, excluded)
                if valid_mask.all():
                    filt_logits, filt_targets = act_logits, act_targets
                else:
                    filt_logits = act_logits[valid_mask]
                    filt_targets = act_targets[valid_mask].clamp(0, act_logits.shape[1] - 1)
                # C.2: LDAM-DRW needs epoch for DRW decision
                if self.use_ldam:
                    drw_epoch = int(getattr(C, 'LDAM_DRW_EPOCH', 60))
                    loss_act = self.act_loss_fn(
                        filt_logits, filt_targets,
                        epoch=self._current_epoch,
                        drw_epoch=drw_epoch,
                    )
                else:
                    loss_act = self.act_loss_fn(filt_logits, filt_targets)
            else:
                loss_act = zero
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

        # [FIX 2026-06-28 20-agent] Activity warmup ramp using stage-local epoch.
        # Uses _act_epoch_counter (resets to 0 when train_act becomes True) instead
        # of _current_epoch (global epoch). Without this, RF3 starting at global
        # epoch 50 would get ramp=1.0 immediately, bypassing the 5-epoch warmup.
        act_ramp = 1.0
        if self.train_act and self._act_epoch_counter >= 0:
            _ramp_ep = self._act_epoch_counter
            act_ramp = min(1.0, (_ramp_ep + 1) / max(self._act_warmup_epochs, 1))
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
        preds = None
        # [F3 2026-07-02] True when the PSR loss is structurally zero (per-frame
        # batch under the transition objective) — the Kendall block then skips
        # the "+ lv_psr" term so log_var_psr only receives gradient on batches
        # that actually computed a PSR task loss.
        _psr_structurally_zero = False
        if self.train_psr:
            # C.3: Binary focal loss for PSR (Doc 01 §D.4: per-component alpha)
            # [OPUS v5] USE_PSR_TRANSITION: convert fill-forward labels to transition targets
            # before computing loss. Per-frame focal on 95%-static labels makes constant
            # output near-optimal; transition targets (Gaussian-smeared 0→1 events) force
            # the model to learn changepoints. psr_transition.py implements the conversion.
            _psr_targets = targets['psr_labels']
            # [OPUS v5 BLOCKER-A FIX] When transition objective is enabled, skip PSR
            # loss on per-frame batches (dim==2). Per-frame focal on 95%-static
            # fill-forward labels produces a constant-output gradient that drowns the
            # transition signal 9:1 (sequence batches are 1-in-10). This gives the
            # transition objective 100% of the PSR gradient.
            if self.use_psr_transition:
                if outputs['psr_logits'].dim() == 3:
                    try:
                        from src.models.psr_transition import build_transition_targets
                        _psr_targets = build_transition_targets(
                            targets['psr_labels'].to(outputs['psr_logits'].device),
                            sigma=float(getattr(C, 'PSR_TRANSITION_SIGMA', 3.0)),
                        )
                    except Exception as e:
                        logger.warning(f'  [PSR_TRANSITION] build_transition_targets failed: {e} — falling back')
                    if self.use_psr_focal:
                        loss_psr = binary_focal_loss(
                            outputs['psr_logits'], _psr_targets,
                            alpha=self.psr_focal_alpha, gamma=self.psr_focal_gamma,
                            per_component_alpha=self._psr_per_component_alpha,
                            comp_weights=self._psr_comp_weights,
                        )
                    else:
                        loss_psr = self.psr_loss_fn(outputs['psr_logits'], _psr_targets)
                else:
                    # Per-frame batch (dim==2) under transition objective: skip PSR loss.
                    # The per-frame static labels teach constant output which drowns the
                    # transition signal on sequence batches.
                    loss_psr = zero
                    # [FIX 2026-07-02 Fable RF4 consult — F3] Mark PSR loss as
                    # structurally zero so the Kendall block below can skip the
                    # "+ lv_psr" regularizer term. Without this, every non-seq
                    # batch added a constant +1 gradient to log_var_psr (task loss
                    # is zero, only the log-sigma term remains), pushing it toward
                    # -4 (54.6x precision) on evidence-free batches. In practice
                    # the seq-batch equilibrium + the MAX_PSR clamp masked this,
                    # but the log_var dynamics (and any future clamp retuning)
                    # were corrupted by the spurious signal.
                    _psr_structurally_zero = True
                    # Skip sensitivity penalty + smooth-cap for per-frame batches under transition objective.
                    # The transition signal only flows on sequence (dim==3) batches.
            elif self.use_psr_focal:
                loss_psr = binary_focal_loss(
                    outputs['psr_logits'], _psr_targets,
                    alpha=self.psr_focal_alpha, gamma=self.psr_focal_gamma,
                    per_component_alpha=self._psr_per_component_alpha,
                    comp_weights=self._psr_comp_weights,
                )
            else:
                loss_psr = self.psr_loss_fn(outputs['psr_logits'], _psr_targets)

            # --- FIX 1 (2026-06-06): PSR input-sensitivity penalty ---
            # Stage 3 epoch 16 collapsed: per-frame logit std/mean=0.12% on cap100
            # (vs 7% at 200b baseline). Root cause: focal loss alone doesn't penalize
            # constant output when the per-component positives ratio is in [0, 1]
            # and the model finds a single threshold that fits all frames.
            # Penalty: -log(mean(per-component-std)) keeps std > 1e-3 in log-space.
            # Only fires at T=1 (dim==2); sequence mode (dim==3) handled by temporal smooth.
            # NOTE: requires batch > 1 — std(dim=0) on a single element has grad = (x-mean)/(n*std)
            # which divides by zero (std=0) and produces NaN in backward → LogBackward0 crash.
            # [F3b 2026-07-02 Fable RF4 consult] `and not _psr_structurally_zero`:
            # the transition-objective branch above documents "Skip sensitivity
            # penalty ... for per-frame batches under transition objective", but
            # this block sat OUTSIDE that branch and fired anyway, silently
            # re-adding a per-frame gradient the BLOCKER-A design explicitly
            # removed (usually clamped to 0 by the Kendall min-clamp since
            # -log(std) is negative for std>1, so training logs still showed
            # psr=0.00 — but near collapse it injected undocumented gradient).
            if (outputs['psr_logits'].dim() == 2 and outputs['psr_logits'].shape[0] > 1
                    and not _psr_structurally_zero):
                _per_comp_std = outputs['psr_logits'].std(dim=0, correction=0).mean()
                _sens = -torch.log(_per_comp_std + 1e-3)
                _sens = torch.where(
                    torch.isfinite(_sens), _sens, torch.tensor(0.0, device=device)
                )
                _sens_w = float(getattr(C, 'PSR_SENSITIVITY_WEIGHT', 0.01))
                loss_psr = loss_psr + _sens_w * _sens

            if self._psr_temporal_smooth_weight > 0 and outputs['psr_logits'].dim() == 3:
                preds = torch.sigmoid(outputs['psr_logits'])
                labels = targets['psr_labels']
                smooth_loss = torch.tensor(0.0, device=device)
                bs = preds.shape[0]
                for i in range(bs):
                    p_i = preds[i]
                    l_i = labels[i]
                    # --- FIX: tanh(.abs()) destroys sign. Use signed diff
                    # so tanh sees real direction. Labels also get a -1 mask
                    # so oscillating labels (signed mean = 0) don't inflate
                    # smooth_loss for a consistent-but-different pred.
                    diff_p = (p_i[1:] - p_i[:-1]).mean()
                    diff_l = (l_i[1:] - l_i[:-1]).mean()
                    pred_change = torch.tanh(diff_p)
                    label_change = diff_l
                    smooth_loss = smooth_loss + (
                        (pred_change - label_change) ** 2
                    )
                smooth_loss = smooth_loss / max(bs, 1)
                smooth_loss = torch.where(
                    torch.isfinite(smooth_loss),
                    smooth_loss,
                    torch.tensor(0.0, device=device),
                )
                loss_psr = loss_psr + self._psr_temporal_smooth_weight * smooth_loss
            # --- FIX: PSR NaN guard BEFORE smooth_cap (lines 1187-1189).
            # loss_psr can be NaN even after binary_focal_loss returns if the model
            # spits extreme logits. Adding smooth_loss weight to NaN gives NaN.
            # Catch it here so _smooth_cap never sees x<=0 (its log would give NaN).
            if not torch.isfinite(loss_psr).all():
                if getattr(C, 'ASSERT_AND_CRASH', False):
                    raise FloatingPointError(
                        f'[ASSERT_AND_CRASH] loss_psr is non-finite before smooth_cap '
                        f'(epoch {self._current_epoch}). The 1e-4 sentinel would silently replace it.'
                    )
                logger.warning(
                    f'  [PSR_NAN] loss_psr={loss_psr.item() if loss_psr.numel()==1 else loss_psr} '
                    f'— replacing with 1e-4 before smooth_cap (epoch {self._current_epoch})'
                )
                # [A4 FIX 2026-06-17] Use torch.where to maintain grad_fn for backward().
                loss_psr = torch.where(
                    torch.isfinite(loss_psr),
                    loss_psr,
                    torch.tensor(1e-4, device=device, dtype=loss_psr.dtype),
                )
        else:
            loss_psr = zero

        # === Head Pose (split loss: position MSE + normalized direction MSE) ===
        if 'head_pose' in outputs and outputs['head_pose'] is not None:
            loss_head_pose = head_pose_loss_split(
                outputs['head_pose'],
                targets['head_pose'],
                pos_weight=1.0,
                dir_weight=1.0,
            )  # No *0.001: Kendall handles weighting; O(1) inputs give balanced gradient
        else:
            loss_head_pose = zero

        # --- FIX: PSR loss cap to prevent NaN cascade ---
        psr_cap = float(getattr(C, 'PSR_LOSS_CAP', 20.0))
        loss_psr = _smooth_cap(loss_psr, psr_cap)

        # --- Per paper §3.7.1: "ℒ_hp = MSE × 5.0" ---
        _hp_weight = float(getattr(C, 'HEAD_POSE_LOSS_WEIGHT', 5.0))
        loss_head_pose = loss_head_pose * _hp_weight

        # --- FIX: Head pose loss cap to prevent NaN cascade ---
        hp_cap = float(getattr(C, 'HEAD_POSE_LOSS_CAP', 30.0))
        loss_head_pose = _smooth_cap(loss_head_pose, hp_cap)

        # [FIX 2026-06-15] Per-component PSR loss breakdown for liveness logging
        _psr_per_component = None
        if self.train_psr and 'psr_logits' in outputs:
            with torch.no_grad():
                _pl = outputs['psr_logits'].clamp(-8.0, 8.0)
                _pt = targets['psr_labels'].to(_pl.device).float()
                _p = torch.sigmoid(_pl)
                _ce = F.binary_cross_entropy_with_logits(_pl, _pt, reduction='none')
                _p_t = (_p * _pt + (1 - _p) * (1 - _pt)).clamp(1e-6, 1 - 1e-6)
                _alpha_t = self.psr_focal_alpha * _pt + (1 - self.psr_focal_alpha) * (1 - _pt)
                _reduce_dims = tuple(range(_ce.ndim - 1))
                _psr_per_component = (_alpha_t * (1 - _p_t) ** self.psr_focal_gamma * _ce).mean(dim=_reduce_dims)

        # === Final NaN guard BEFORE Kendall — covers ALL losses.
        # Each loss can produce NaN via: loss function overflow, temporal smooth
        # overflow, smooth cap log(0), or numerical instability in any component.
        # Replace any NaN/inf with 1e-4 (tiny but > 0 — ensures gradient flows
        # and no division-by-zero in Kendall normalization).
        # [A4 FIX 2026-06-17] _safe must always produce a tensor with grad_fn.
        # torch.where creates a WhereBackward node even when ALL elements are NaN,
        # preventing "element 0 does not require grad" crash during loss.backward().
        _safe = lambda l, z: torch.where(
            torch.isfinite(l),
            torch.where(l < 0, z, l) if l.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64] else l,
            torch.tensor(1e-4, device=device, dtype=l.dtype),
        )
        loss_det = _safe(loss_det, zero)
        loss_pose = _safe(loss_pose, zero)
        loss_act = _safe(loss_act, zero)
        # [FIX Bug #9] PSR NaN diagnostic — log when PSR is silently replaced
        if not torch.isfinite(loss_psr).all():
            logger.warning(
                f'  [PSR_NAN] loss_psr={loss_psr.item() if loss_psr.numel()==1 else loss_psr} '
                f'— replacing with 1e-4 (epoch {self._current_epoch})'
            )
        loss_psr = _safe(loss_psr, zero)
        loss_head_pose = _safe(loss_head_pose, zero)

        # === [OPUS v5] Per-head liveness probe — checks I1 (non-NaN) + I2 (non-zero) ===
        # Prints every LIVENESS_EVERY steps (default 200). A head is ALIVE iff:
        # loss > 10× its floor, and the loss is finite.
        _liveness_every = int(getattr(C, 'LIVENESS_EVERY', 200))
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
        else:
            self._step_counter = 1
        if self._step_counter % _liveness_every == 0:
            _floor = {'det': 1e-2, 'act': 1e-3, 'psr': 1e-4, 'head_pose': 1e-4, 'pose': 1e-5}
            _heads = [
                ('det', loss_det, 'det'),
                ('act', loss_act if self.train_act else zero, 'act'),
                ('psr', loss_psr if self.train_psr else zero, 'psr'),
                ('head_pose', loss_head_pose, 'head_pose'),
                ('pose', loss_pose, 'pose'),
            ]
            _parts = []
            for _hname, _hloss, _hkey in _heads:
                _hf = _floor.get(_hkey, 1e-4)
                _hval = float(_hloss.item()) if isinstance(_hloss, torch.Tensor) and _hloss.numel() == 1 else float(_hloss)
                _hfin = torch.isfinite(_hloss).all() if isinstance(_hloss, torch.Tensor) else True
                _halive = _hfin and _hval > 10 * _hf
                _parts.append(f'{_hname}={_hval:.2e} {"ALIVE" if _halive else ("NaN" if not _hfin else "DEAD")}')
            # [FIX 2026-06-15] Per-component PSR breakdown
            if _psr_per_component is not None:
                _comp_min = float(_psr_per_component.min())
                _comp_max = float(_psr_per_component.max())
                _comp_mean = float(_psr_per_component.mean())
                _parts.append(f'psr_c={_comp_min:.2e}/{_comp_max:.2e}/{_comp_mean:.2e}')
            # [FIX 2026-06-15] GPU memory
            if torch.cuda.is_available():
                _mem_alloc = torch.cuda.memory_allocated() / 1e9
                _mem_res = torch.cuda.memory_reserved() / 1e9
                _parts.append(f'mem={_mem_alloc:.2f}/{_mem_res:.2f}G')
            _msg = f'  [LIVENESS step={self._step_counter}] ' + ' | '.join(_parts)
            logger.warning(_msg)
            print(_msg, flush=True)

        # === MTL alternative weighting (FAMO / IMTL-L / RLW) ===
        # These optional modules replace Kendall weighting entirely for the total loss.
        # Precision vars in loss_dict are zeroed (not meaningful for non-Kendall MTL).
        if self.use_famo or self.use_imtl_l or self.use_rlw:
            mtl_losses = {}
            if self.train_det:
                mtl_losses["det"] = loss_det
            mtl_losses["pose"] = loss_pose + loss_head_pose
            if self.train_act:
                mtl_losses["act"] = loss_act
            if self.train_psr and not _psr_structurally_zero:
                mtl_losses["psr"] = loss_psr

            if self.use_famo:
                if self.famo_weighter is None:
                    self.famo_weighter = FAMOWeighter(num_tasks=len(mtl_losses))
                    self._mtl_task_names = list(mtl_losses.keys())
                total = self.famo_weighter.forward(mtl_losses)
                self._last_mtl_losses = mtl_losses
            elif self.use_imtl_l:
                total = imtl_l_loss(mtl_losses)
            elif self.use_rlw:
                if self.rlw_weighter is None:
                    self.rlw_weighter = RLWWeighter(num_tasks=len(mtl_losses))
                    self._mtl_task_names = list(mtl_losses.keys())
                loss_tensor = torch.stack(list(mtl_losses.values()))
                weights = self.rlw_weighter.get_weights(loss_tensor.device)
                total = (loss_tensor * weights).sum()

            # MTL paths don't use Kendall precisions; zero for logging
            prec_det = prec_hp = prec_act = prec_psr = torch.tensor(0.0, device=device)

        # === Kendall weighting ===
        # Init precision vars before branching so logging at line 1772 never hits UnboundLocalError.
        elif self.use_kendall:
            # [FIX 2026-06-20 (Opus v8 §3 Fix 1)] Fixed-weight path for RF1-RF2.
            # Bypasses learned Kendall log_vars entirely; uses fixed lambda weights so
            # detection drives the backbone and head_pose just stabilizes it.
            # Re-enable standard Kendall at RF3+ once detection is real.
            if bool(getattr(C, 'KENDALL_FIXED_WEIGHTS', False)):
                _hp_lambda = float(getattr(C, 'KENDALL_HP_FIXED_LAMBDA', 0.2))
                total = torch.tensor(0.0, device=device)
                if self.train_det:
                    total = total + loss_det
                if self.train_pose:
                    loss_pose = loss_pose.clamp(min=0.0)
                if self.train_act:
                    loss_act = loss_act.clamp(min=0.0)
                if self.train_pose or self.train_act:
                    loss_head_pose = loss_head_pose.clamp(min=0.0)
                    if self.train_pose and self.train_act:
                        total = total + _hp_lambda * (loss_pose + loss_head_pose)
                    elif self.train_pose:
                        total = total + _hp_lambda * (loss_pose + loss_head_pose)
                    else:
                        total = total + _hp_lambda * loss_head_pose
                if self.train_act:
                    _act_w = float(getattr(C, 'ACTIVITY_LOSS_WEIGHT', 1.0))
                    total = total + loss_act * _act_w
                if self.train_psr:
                    loss_psr = loss_psr.clamp(min=0.0)
                    _psr_w = float(getattr(C, 'PSR_WEIGHT', 20.0))
                    total = total + loss_psr * _psr_w
                total = total.squeeze()
            else:
                # Standard Kendall precision-weighted path (activated when
                # KENDALL_FIXED_WEIGHTS=False). Each task contributes
                # prec_task * loss_task + lv_task.
                # [FIX 2026-06-15] Per-task Kendall bounds to prevent multi-task collapse.
                # Activity log_var FLOOR (min) prevents precision-boosting above exp(0)=1.0.
                # PSR/pose log_var CEILING (max) prevents suppression below exp(0)=1.0.
                # Matches _clamp_kendall_log_vars in train.py (parameter-level guard).
                _act_min = float(getattr(C, 'KENDALL_LOG_VAR_MIN_ACT', -4.0))
                _psr_max = float(getattr(C, 'KENDALL_LOG_VAR_MAX_PSR', 2.0))
                _pose_max = float(getattr(C, 'KENDALL_LOG_VAR_MAX_POSE', 2.0))
                lv_det = self.log_var_det.clamp(-4.0, 2.0)
                lv_hp = self.log_var_pose.clamp(-4.0, _pose_max)
                lv_act = self.log_var_act.clamp(_act_min, 2.0)
                lv_psr = self.log_var_psr.clamp(-4.0, _psr_max)

                # [FIX 2026-06-20 (Opus v8 §3 Fix 1)] Head-pose precision cap:
                # head_pose precision can never exceed detection precision. Without this,
                # head_pose (loss ≈ 0.01) gets Kendall-optimal precision ~54.6× while detection
                # (loss ≈ 0.5) gets ~1.4×, and the shared backbone is optimized for head_pose.
                # The `detach()` ensures detection's log_var is not affected by the asymmetry.
                if bool(getattr(C, 'KENDALL_HP_PREC_CAP', True)):
                    lv_hp = torch.maximum(lv_hp, lv_det.detach())

                prec_det = torch.exp(-lv_det)
                prec_hp = torch.exp(-lv_hp)
                prec_act = torch.exp(-lv_act)
                prec_psr = torch.exp(-lv_psr)

                # Stage-aware Kendall: zero precision AND log_var of frozen tasks to prevent
                # gradient corruption in their log_vars during staged training.
                # Epoch 0: no staging (backward-compat for resumed runs).
                # Epoch 1-5 (stage 1): detection + activity (ramped) only.
                #   head_pose + psr frozen (zeroed).
                # Epoch 6-15 (stage 2): detection + head_pose + activity (ramp completed).
                #   psr frozen (zeroed).
                # Epoch 16+ (stage 3): all tasks (psr ramped per PSR_WARMUP_EPOCHS).
                #
                # [USER-AUTH FIX 2026-06-05] Activity is now trained from epoch 0 via
                # ACT_RAMP_EPOCHS ramp (mirrors prec_psr_ramp below). The log_var_act
                # is preserved (not zeroed) so the learnable parameter can adapt to the
                # ramping gradient magnitude. Zeroing lv_act would have starved log_var_act
                # of all gradient signal.
                # [F18 2026-07-02 Fable RF4 consult] DOUBLE-RAMP FIX. The activity
                # ramp was applied TWICE: once to the RAW loss in the activity
                # section above (`loss_act = loss_act * act_ramp`, the canonical
                # site — it covers Kendall, fixed-weight, and non-Kendall paths
                # alike), and AGAIN here to the Kendall precision. Effective
                # activity supervision during warmup was ramp^2: 4% (not 20%) at
                # epoch 0, 36% (not 60%) at epoch 2 — a compounding factor in
                # every historical activity-collapse episode. The precision-side
                # multiplication below is removed; the loss-level ramp is the
                # single source of truth.
                # (was: prec_act scaled by (counter+1)/ACT_RAMP_EPOCHS right here)
                if bool(getattr(C, 'STAGED_TRAINING', True)) and self._current_epoch >= 1:
                    stage = _get_kendall_stage(self._current_epoch)
                    # Activity ramp — applies in BOTH stage 1 and stage 2 (ramp completes
                    # by epoch ACT_RAMP_EPOCHS-1, so stage 2 sees ramp=1.0 unless config
                    # is changed). In stage 3 the ramp is naturally 1.0 and prec_act is
                    # left unchanged below.
                    act_ramp_epochs = int(getattr(C, 'ACT_RAMP_EPOCHS', 5))
                    if act_ramp_epochs > 0 and self._current_epoch < act_ramp_epochs:
                        act_ramp = (self._current_epoch + 1) / act_ramp_epochs
                    else:
                        act_ramp = 1.0
                    # [FIX 2026-07-07 File-157 F-1 v2] KENDALL_FIXED_WEIGHTS=1 guard:
                    # when fixed weights are in use, the staging zero-out of prec_psr
                    # /lv_psr is suppressed so PSR keeps receiving gradient in stages
                    # 1-2. Without this, every multi-task run with KENDALL_FIXED_WEIGHTS=1
                    # would have a dead PSR head until stage 3 (epoch >= 16), reproducing
                    # the original V3 pathology on fresh starts.
                    _kendall_fixed = bool(getattr(C, 'KENDALL_FIXED_WEIGHTS', False))
                    if stage == 1 and not _kendall_fixed:
                        # Detection-only stage: head_pose and psr frozen.
                        prec_hp = prec_hp * 0
                        lv_hp = lv_hp * 0
                        prec_psr = prec_psr * 0
                        lv_psr = lv_psr * 0
                        # [F18] Activity ramp handled ONCE at the loss level
                        # (activity section above) — the old prec_act *= act_ramp
                        # here made staged warmup ramp^2 as well.
                    elif stage == 2 and not _kendall_fixed:
                        # Det + head_pose + activity (ramp at 1.0 by epoch >= ACT_RAMP_EPOCHS)
                        prec_psr = prec_psr * 0
                        lv_psr = lv_psr * 0
                        # [F18] prec_act ramp removed here too — see stage 1 note.
                    elif stage == 3:
                        stage1_end = int(getattr(C, 'STAGE1_EPOCHS', 5))
                        stage2_end = stage1_end + int(getattr(C, 'STAGE2_EPOCHS', 10))
                        stage3_start = stage2_end + 1
                        warmup_epochs = int(getattr(C, 'PSR_WARMUP_EPOCHS', 5))
                        if warmup_epochs > 0:
                            prec_psr_ramp = min(1.0, (self._current_epoch - stage3_start + 1) / warmup_epochs)
                            prec_psr = prec_psr * prec_psr_ramp

                total = torch.tensor(0.0, device=device)
                if self.train_det:
                    total = total + prec_det * loss_det + lv_det
                # Sanity floor: ensure no component loss is negative enough to cause
                # Kendall divergence. Soft floor already applied to loss_det above, but
                # add this as last-resort guard for all components.
                if self.train_pose:
                    loss_pose = loss_pose.clamp(min=0.0)
                if self.train_act:
                    loss_act = loss_act.clamp(min=0.0)
                # Build Kendall total — each task adds its precision-weighted loss + log_var.
                # NOTE: loss_pose (body keypoint Wing Loss) and loss_head_pose (head pose 9-DoF MSE)
                # share log_var_pose (intentional per paper §Multi-Task Loss: both are pose tasks).
                # lv_hp is included ONCE for pose+head_pose when EITHER branch runs, since they share log_var_pose.
                if self.train_pose or self.train_act:
                    loss_head_pose = loss_head_pose.clamp(min=0.0)
                    pose_contribution = prec_hp * loss_pose + lv_hp if self.train_pose else prec_hp * loss_head_pose + lv_hp
                    if self.train_pose and self.train_act:
                        pose_contribution = prec_hp * loss_pose + prec_hp * loss_head_pose + lv_hp  # both body+head, one lv_hp
                    elif self.train_pose:
                        # [FIX 2026-06-17] Include loss_head_pose in Kendall total!
                        # Bug: IndustReal has NO keypoint annotations → loss_pose is always ZERO,
                        # making pose_contribution = lv_hp (just log_var reg, zero grad for head_pose_head).
                        # Head pose (9-DoF MSE, ~1.7) was computed in forward pass but excluded from total loss,
                        # neutralizing the entire train_head_pose=True fix (Opus RF1 prescription).
                        pose_contribution = prec_hp * loss_pose + prec_hp * loss_head_pose + lv_hp
                    else:  # train_act only
                        pose_contribution = prec_hp * loss_head_pose + lv_hp
                    total = total + pose_contribution
                if self.train_act:
                    # [FIX 2026-06-15] ACTIVITY_LOSS_WEIGHT prevents activity from dominating
                    # the Kendall total. Activity loss is ~1-5 vs PSR loss ~0.01 — without
                    # this down-weight, activity dominates even when both heads have equal
                    # precision, and Kendall log_vars learn to suppress PSR/pose to compensate.
                    _act_w = float(getattr(C, 'ACTIVITY_LOSS_WEIGHT', 1.0))
                    total = total + prec_act * (loss_act * _act_w) + lv_act
                if self.train_psr and not _psr_structurally_zero:
                    loss_psr = loss_psr.clamp(min=0.0)
                    # [INTERVENTION 2026-06-14] PSR_WEIGHT applied BEFORE Kendall precision
                    # so the learned s_psr (log_var) can still modulate the effective weight.
                    _psr_w = float(getattr(C, 'PSR_WEIGHT', 20.0))
                    # [FIX 2026-06-15] Step-based PSR warmup: extra precision multiplier that
                    # decays from PSR_WARMUP_INIT_MULT → 1.0 over PSR_WARMUP_STEPS.
                    # Gives PSR a gradient head start before activity loss dominates.
                    _psr_warmup_steps = int(getattr(C, 'PSR_WARMUP_STEPS', 0))
                    if _psr_warmup_steps > 0 and self._step_counter < _psr_warmup_steps:
                        _psr_warmup_init = float(getattr(C, 'PSR_WARMUP_INIT_MULT', 3.0))
                        _ramp = _psr_warmup_init + (1.0 - _psr_warmup_init) * self._step_counter / _psr_warmup_steps
                        prec_psr = prec_psr * _ramp
                    total = total + prec_psr * (loss_psr * _psr_w) + lv_psr
                total = total.squeeze()

            # NaN guard in Kendall total — fires on inf/NaN only, not on negative
            # Handle both scalar and 1-element tensor cases
            total_val = total.item() if total.numel() == 1 else total
            if not math.isfinite(total_val):
                parts = []
                if self.train_det:
                    parts.append(loss_det)
                if self.train_pose:
                    parts.append(loss_pose)  # body keypoint (prec_hp * loss_pose in Kendall total)
                if self.train_act:
                    parts.append(loss_head_pose)  # head pose (prec_hp * loss_head_pose + lv_hp)
                    parts.append(loss_act)  # activity (prec_act * loss_act)
                if self.train_psr:
                    parts.append(loss_psr)
                finite_parts = [p for p in parts if torch.isfinite(p) and p >= 0]
                if finite_parts:
                    total = torch.stack(finite_parts).sum()
                else:
                    # [A4 FIX 2026-06-17] Maintain grad connectivity through Kendall log_vars
                    total = loss_det + 0.0 * (lv_det + lv_hp + lv_act + lv_psr)
                # [A4 FIX 2026-06-17] Guard: if total somehow lacks grad_fn (e.g., all
                # components were replaced with leaf tensors), reconnect through log_vars.
                if total.grad_fn is None:
                    total = total + 0.0 * (self.log_var_det + self.log_var_pose + self.log_var_act + self.log_var_psr).sum()
        else:
            prec_det = prec_hp = prec_act = prec_psr = torch.tensor(1.0, device=device)
            _loss_act_staged = loss_act
            _loss_psr_staged = loss_psr
            # [FIX 2026-06-17] Same bug as Kendall path: loss_head_pose (9-DoF MSE)
            # was excluded when train_pose=True. IndustReal has no keypoint annotations,
            # so loss_pose=zero → head_pose_head got zero gradient even in non-Kendall path.
            _loss_pose_staged = loss_pose + loss_head_pose if self.train_pose else loss_head_pose
            if bool(getattr(C, 'STAGED_TRAINING', True)) and self._current_epoch >= 1:
                stage = _get_kendall_stage(self._current_epoch)
                if stage == 1:
                    _loss_act_staged = zero
                    _loss_psr_staged = zero
                    _loss_pose_staged = zero
                elif stage == 2:
                    _loss_act_staged = zero
                    _loss_psr_staged = zero
                    _loss_pose_staged = zero
            total = loss_det + _loss_pose_staged + _loss_act_staged + _loss_psr_staged
            # [A4 FIX 2026-06-17] Guard total NaN in non-Kendall path.
            # Use torch.where to maintain grad_fn (prevents backward crash on disconnected graph).
            if not torch.isfinite(total).all():
                total = torch.where(
                    torch.isfinite(total),
                    total,
                    torch.zeros_like(total),
                )

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

        # Safe scalar extractor — handles 0D tensor, 1D tensor, or plain Python float.
        # Returns 0.0 if the value is NaN/inf or not a scalar.
        def _s(x):
            try:
                if hasattr(x, 'item'):
                    v = x.item()
                else:
                    v = float(x)
                return 0.0 if not math.isfinite(v) else v
            except Exception:
                return 0.0

        loss_dict = {
            'total': _s(total),
            'det_cls': _s(cls_loss) if self.train_det else 0.0,
            'det_reg': _s(reg_loss) if self.train_det else 0.0,
            'det': _s(loss_det),
            'pose': _s(loss_pose),
            'activity': _s(loss_act),
            'psr': _s(loss_psr),
            'head_pose': _s(loss_head_pose),
            'w_det': _s(wd),
            'w_pose': _s(wp),
            'w_act': _s(wa),
            'w_psr': _s(wps),
            'log_var_det': _s(self.log_var_det),
            'log_var_pose': _s(self.log_var_pose),
            'log_var_act': _s(self.log_var_act),
            'log_var_psr': _s(self.log_var_psr),
            'act_ramp': _s(act_ramp),
        }

        return total, loss_dict

    def famo_step(self) -> None:
        """Update FAMO weights after optimizer.step().

        Must be called AFTER optimizer.step() each training step when
        ``use_famo=True``. Delegates to ``FAMOWeighter.step()`` with the
        per-task losses captured in the preceding forward pass.

        Safe no-op when not in FAMO mode, weighter not initialized, or
        no cached losses available.
        """
        if not self.use_famo or self.famo_weighter is None:
            return
        if not hasattr(self, '_last_mtl_losses') or self._last_mtl_losses is None:
            return
        self.famo_weighter.step(self._last_mtl_losses)
        self._last_mtl_losses = None  # prevent double-step on same batch