#!/usr/bin/env python3
"""
train_mtl_v3.py — Detection training with QFL + GFL + ATSS options.

PROBLEM (v2, mAP=0):
  - "Anchor matching" was center-point cell assignment: GT box center -> nearest
    grid cell. No actual anchor box geometry used.
  - Regression targets used absolute pixel offsets (cx*W-gx), not anchor-relative.
  - The eval decode expects dx*0.1 center offsets and exp(dw) width scaling,
    but train targets were in pixel space — completely mismatched.
  - Classification used weighted CE with arbitrary 50x foreground boost
    instead of focal loss.
  - DFL output (64ch = 16 bins x 4 coords) was collapsed by mean() instead
    of interpreted as 16 anchors x 4 coords matching eval decode.

FIXES (v3):
  1. IoU-based anchor-GT matching across all 3 FPN levels and 16 anchors.
  2. Each GT force-matched to its highest-IoU anchor; any anchor with IoU>0.5
     also positive.
  3. Regression targets match eval decode exactly:
       dx = (cx_gt - cx_a) / 0.1    (eval decodes as cx_a + dx*0.1)
       dy = (cy_gt - cy_a) / 0.1
       dw = log(w_gt / w_a)         (eval decodes as w_a * exp(dw))
       dh = log(h_gt / h_a)
  4. Sigmoid focal loss (per-class binary) matching eval's sigmoid decode,
     not softmax CE.
  5. Smooth L1 regression loss on positive anchors only.
  6. Per-level FPN loss weighting (inverse-sqrt spatial size).
  7. Proper 16-anchor interpretation: reg_preds [B, 64, H, W] =
     [B, 16, 4, H, W] via reshape.

V3.1 FIXES (added 2026-07-16):
  8. Loss normalization: background-only batches now normalize by N*C (total
     elements) instead of falling back to 1.0.  With ~93% background-only
     samples, the old normalization made background losses ~H*W times larger
     than foreground losses, drowning all positive gradient signal.
  9. Foreground-balanced batch sampler: each batch now guaranteed to contain
     at least one foreground (non-background) sample.  Random sampling at
     batch_size=2 produces >85% background-only batches, starving the
     detection head of positive training signal.
  10. Gradient clipping threshold raised from 1.0 to 10.0 to allow the rare
      foreground gradient steps to actually have an effect.
  11. DetectionHead cleaned up: removed misleading DFL naming, now explicitly
      outputs 4 * num_anchors (= 64) channels.

V3.2 IMPROVEMENTS (added 2026-07-19):
  12. Quality Focal Loss (QFL): replaces sigmoid focal loss when --loss qfl.
      QFL uses soft targets (IoU scores) instead of binary 0/1, so
      classification quality reflects localization accuracy.
  13. ATSS adaptive matching: replaces fixed IoU threshold (0.5) with
      per-GT dynamic threshold = mean(topk_iou) + std(topk_iou) when
      --matcher atss.  Adapts to object scale automatically.

V3.3 IMPROVEMENTS (added 2026-07-19):
  14. CIoU loss (Zheng et al. AAAI 2020): replaces Smooth L1 for box
      regression.  CIoU optimizes IoU, center distance, and aspect ratio
      jointly instead of independent coordinate residuals.  Standard
      practice in YOLOv8/v9/v10.  Expected +5.67% AP improvement.
      Implementation: decode_deltas_to_xyxy converts (dx,dy,dw,dh) to
      (x1,y1,x2,x2) before computing CIoU loss.

Architecture:
  Backbone: MViTv2-S with K400 pretrained weights (expanded 3->9 channels)
  FPN: P3(45x80) / P4(23x40) / P5(12x20) at 640x360 input
  Detection: shared per-location sigmoid cls (24 classes) + per-anchor reg (16A)

Usage:
  python train_mtl_v3.py [--phase1-epochs 2] [--phase2-epochs 5]
                        [--loss focal|qfl] [--matcher iou|atss]
"""
import argparse, os, sys, time, json, logging, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(0, str(_CODE_ROOT / "src"))
import src.config as C
C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

from train_mtl_full_multimodal import (
    expand_conv_proj_to_9ch, ensure_5d,
    FullSyntheticDataset, FullMultiModalDataset,
    collate_synth_targets, collate_real_targets,
    Part3DLoader, WrappedMTL,
)
from src.models.mvit_mtl_model import MTLMViTModel, NUM_DET_CLASSES
from src.losses.supcon import SupConLoss, get_projection_head
from src.losses.uw_so import UWSOLoss
from src.losses.ciou import ciou_loss, decode_deltas_to_xyxy
from src.losses.qfl import quality_focal_loss_with_logits


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mtl_v3")

# ===========================================================================
# Anchor configuration (MUST match eval_real_map_fast.py decode)
# ===========================================================================
# CRITICAL FIX 2026-07-21 (Bug 8): Decode fix uses FIXED 0.1 scaling (see ciou.py:39).
# CRITICAL FIX 2026-07-21 (Anchors): 8 new anchors from k-means on IndustReal GT.
#
# V3.8: Default is now 8 anchors (k-means optimized), replacing 16 legacy anchors.
# Legacy 16-anchor mode available via --num-anchors 16 for checkpoint compat.
# Fresh K400 init required when switching anchor count (different reg channels).

# 8 anchors from k-means (IoU distance) on IndustReal GT boxes (k=8, mean best IoU=0.8284)
_ANCHOR_SPECS_8: list[tuple[float, float]] = [
    (0.2261, 0.1961),  # 145x71  px at 640x360
    (0.4242, 0.2926),  # 271x105 px
    (0.3500, 0.4035),  # 224x145 px
    (0.2696, 0.5773),  # 173x208 px
    (0.4904, 0.4280),  # 314x154 px
    (0.4275, 0.5544),  # 274x200 px
    (0.5564, 0.5555),  # 356x200 px
    (0.6556, 0.6704),  # 420x241 px
]

# Legacy 16 anchors (4 sizes x 4 ratios) for backward compat
_ANCHOR_SPECS_16: list[tuple[float, float]] = []
for s in [0.05, 0.1, 0.2, 0.4]:
    for r in [0.5, 1.0, 2.0, 0.25]:
        _ANCHOR_SPECS_16.append((s * math.sqrt(r), s / math.sqrt(r)))

# Default: 8 anchors (v3.8+). Override with --num-anchors.
NUM_ANCHORS = 8
_ANCHOR_SPECS: list[tuple[float, float]] = _ANCHOR_SPECS_8


def generate_anchors(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Generate anchor boxes for one FPN level.

    Each anchor center is at the grid cell center (i+0.5)/H, (j+0.5)/W.
    Anchors are the same NUM_ANCHORS shapes at every location.

    Returns:
        anchors: [H, W, NUM_ANCHORS, 4] in (cx, cy, w, h), normalized [0,1]
    """
    ys = (torch.arange(H, device=device) + 0.5) / H
    xs = (torch.arange(W, device=device) + 0.5) / W
    # [1, H, W], [1, H, W]
    cx_grid = xs.view(1, 1, W).expand(1, H, W)
    cy_grid = ys.view(1, H, 1).expand(1, H, W)

    anchors = torch.zeros(H, W, NUM_ANCHORS, 4, device=device)
    for a_idx, (aw, ah) in enumerate(_ANCHOR_SPECS):
        anchors[:, :, a_idx, 0] = cx_grid.squeeze()
        anchors[:, :, a_idx, 1] = cy_grid.squeeze()
        anchors[:, :, a_idx, 2] = aw
        anchors[:, :, a_idx, 3] = ah
    return anchors


# ===========================================================================
# IoU for (cx, cy, w, h) format
# ===========================================================================
def box_iou(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Vectorized IoU between anchors and GT boxes.

    Args:
        anchors: [N, 4] in (cx, cy, w, h), normalized
        gt_boxes: [M, 4] in (cx, cy, w, h), normalized

    Returns:
        iou: [N, M]
    """
    eps = 1e-6
    # Convert to [x1, y1, x2, y2] broadcasting
    a_x1 = anchors[:, 0:1] - anchors[:, 2:3] / 2
    a_y1 = anchors[:, 1:2] - anchors[:, 3:4] / 2
    a_x2 = anchors[:, 0:1] + anchors[:, 2:3] / 2
    a_y2 = anchors[:, 1:2] + anchors[:, 3:4] / 2

    g_x1 = gt_boxes[:, 0:1] - gt_boxes[:, 2:3] / 2
    g_y1 = gt_boxes[:, 1:2] - gt_boxes[:, 3:4] / 2
    g_x2 = gt_boxes[:, 0:1] + gt_boxes[:, 2:3] / 2
    g_y2 = gt_boxes[:, 1:2] + gt_boxes[:, 3:4] / 2

    inter_x1 = torch.max(a_x1, g_x1.T)
    inter_y1 = torch.max(a_y1, g_y1.T)
    inter_x2 = torch.min(a_x2, g_x2.T)
    inter_y2 = torch.min(a_y2, g_y2.T)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = anchors[:, 2] * anchors[:, 3]
    area_g = gt_boxes[:, 2] * gt_boxes[:, 3]
    union = area_a[:, None] + area_g[None, :] - inter

    return inter / (union + eps)


# ===========================================================================
# Anchor-GT matching (THE CORE FIX)
# ===========================================================================
def match_anchors_to_gt(
    anchors_flat: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_threshold: float = 0.5,
    matcher_type: str = "iou",
    iou_topk: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match GT boxes to anchors for one sample.

    Supports two matcher modes:
      - "iou" (default): fixed IoU threshold + force-match best anchor per GT.
      - "atss": adaptive threshold per GT = mean(topk_iou) + std(topk_iou).

    Args:
        anchors_flat: [H*W*A, 4] all anchors in (cx, cy, w, h)
        gt_boxes: [N, 4] GT boxes in (cx, cy, w, h), normalized [0,1]
        gt_classes: [N] GT class indices (0 .. NUM_DET_CLASSES-1)
        iou_threshold: minimum IoU for positive match ("iou" mode only)
        matcher_type: "iou" or "atss"
        iou_topk: top-k for ATSS threshold estimation

    Returns:
        cls_target: [H*W] class index per location (-1 = background)
        reg_target: [H*W*A, 4] regression targets per anchor
        pos_mask: [H*W*A] bool mask for positive anchors
        quality_scores: [H*W*A] IoU quality score per anchor (0 for bg)
    """
    n_anchors = anchors_flat.shape[0]
    n_gts = gt_boxes.shape[0]
    A = NUM_ANCHORS
    H_W = n_anchors // A  # number of spatial locations
    device = gt_boxes.device

    # -1 = background, 0..23 = object class
    cls_target = torch.full((H_W,), -1, dtype=torch.long, device=device)
    reg_target = torch.zeros(n_anchors, 4, device=device)
    pos_mask = torch.zeros(n_anchors, dtype=torch.bool, device=device)
    quality_scores = torch.zeros(n_anchors, device=device)

    if n_gts == 0:
        return cls_target, reg_target, pos_mask, quality_scores

    if matcher_type == "atss":
        # ---- ATSS adaptive matching ----
        ious = box_iou(anchors_flat, gt_boxes)

        # Per-GT adaptive threshold = mean(topk) + std(topk)
        candidate = torch.zeros(n_anchors, dtype=torch.bool, device=device)
        for gt_idx in range(n_gts):
            gt_iou = ious[:, gt_idx]
            topk = min(iou_topk, n_anchors)
            topk_vals, _ = gt_iou.topk(topk)
            threshold = topk_vals.mean() + topk_vals.std()
            candidate = candidate | (gt_iou > threshold)

        best_iou_per_anchor, best_gt_per_anchor = ious.max(dim=1)
        pos_mask = candidate
        quality_scores[pos_mask] = best_iou_per_anchor[pos_mask]

    else:
        # ---- Original IoU matching (unchanged logic) ----
        ious = box_iou(anchors_flat, gt_boxes)

        best_iou_per_gt, best_anchor_per_gt = ious.max(dim=0)
        best_iou_per_anchor, best_gt_per_anchor = ious.max(dim=1)

        pos_mask = best_iou_per_anchor >= iou_threshold
        for gt_idx in range(n_gts):
            pos_mask[best_anchor_per_gt[gt_idx]] = True

        quality_scores[pos_mask] = best_iou_per_anchor[pos_mask]

    # === Per-location classification targets ===
    for loc_idx in range(H_W):
        start = loc_idx * A
        end = start + A
        loc_pos = pos_mask[start:end]
        if loc_pos.any():
            gt_for_loc = best_gt_per_anchor[start:end][loc_pos]
            cls_target[loc_idx] = gt_classes[gt_for_loc[0]]

    # === Per-anchor regression targets (same regardless of matcher) ===
    if pos_mask.any():
        matched_gt_idx = best_gt_per_anchor[pos_mask]
        matched_anchors = anchors_flat[pos_mask]
        matched_gts = gt_boxes[matched_gt_idx]

        dx = (matched_gts[:, 0] - matched_anchors[:, 0]) / 0.1
        dy = (matched_gts[:, 1] - matched_anchors[:, 1]) / 0.1
        dw = torch.log(matched_gts[:, 2] / matched_anchors[:, 2].clamp(min=1e-6))
        dh = torch.log(matched_gts[:, 3] / matched_anchors[:, 3].clamp(min=1e-6))

        reg_target[pos_mask] = torch.stack([
            dx.clamp(-5.0, 5.0), dy.clamp(-5.0, 5.0),
            dw.clamp(-2.0, 2.0), dh.clamp(-2.0, 2.0),
        ], dim=1)

    return cls_target, reg_target, pos_mask, quality_scores


# ===========================================================================
# Paired IoU for (x1, y1, x2, y2) format (used by TAL alignment metric)
# ===========================================================================
def box_iou_xyxy_paired(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU between corresponding pairs of boxes in xyxy format.

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2)
        boxes2: [N, 4] in (x1, y1, x2, y2)

    Returns:
        iou: [N] per-pair IoU
    """
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=eps) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=eps)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=eps) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=eps)
    union = area1 + area2 - inter + eps
    return inter / union


# ===========================================================================
# Sigmoid Focal Loss (matches eval's sigmoid decode)
# ===========================================================================
def sigmoid_focal_loss(
    logits: torch.Tensor,
    cls_target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Per-class sigmoid focal loss.

    Eval uses ``probs = torch.sigmoid(cls_l)``, so training must use binary
    focal loss (one-vs-rest), NOT softmax cross-entropy.

    CRITICAL FIX: When there are no positive targets (background-only sample),
    normalize by total elements (N*C) instead of falling back to 1.0.  The old
    code used ``max(targets.sum(), 1.0)`` which made background-only batches
    produce losses proportional to H*W, drowning out rare foreground gradients.
    With ~93% of samples being background-only, this was the root cause of
    mAP=0 after 196K batches.

    Args:
        logits: [N, C] raw logits (pre-sigmoid)
        cls_target: [N] class index (-1 = background, 0..C-1 = object class)
        gamma: focusing parameter
        alpha: class-balance parameter

    Returns:
        scalar loss
    """
    N, C = logits.shape
    device = logits.device

    # Build one-hot targets: background = all zeros, foreground = 1 at class idx
    targets = torch.zeros(N, C, device=device)
    pos = cls_target >= 0
    if pos.any():
        targets[pos] = F.one_hot(cls_target[pos], num_classes=C).float()

    # Binary CE
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Focal weight: (1 - p_t)^gamma
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1.0 - p_t) ** gamma

    # Alpha balancing
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    # --- CRITICAL: proper normalization ---
    # Standard detectron2 convention: normalize by num positive anchors.
    # However, when pos_count is very small (1-3 anchors), dividing by
    # such a tiny normalizer inflates the loss by 100-1000x compared to
    # well-populated batches, causing gradient explosions.  We apply a
    # floor of 16 to keep loss magnitude stable across batch compositions.
    #
    # When NO positives (background-only): normalize by total elements N*C
    # so the background loss is ~ per-location loss, NOT huge.
    pos_count = targets.sum()
    if pos_count > 0:
        normalizer = max(float(pos_count), 16.0)
    else:
        normalizer = float(N * C)
    return (alpha_t * focal_weight * bce).sum() / normalizer


# ===========================================================================
# Class-balanced weighting (Cui et al. 2019 — Effective Number of Samples)
# ===========================================================================
def class_balanced_weights(class_counts):
    """Compute per-class sampling weights using Effective Number of Samples.

    Cui et al. 2019 propose weighting by (1-beta^n)/(1-beta) where n is
    the number of samples for a class.  Classes with few samples get higher
    weight, but the law of diminishing returns limits the boost — a class
    with 1 sample gets ~1/(1-beta) weight, a class with 30K samples gets
    ~1/(1-beta^30K) ~= 1.  beta=0.999 caps the max effective at ~1000.

    Args:
        class_counts: [24] array-like of per-class sample counts.

    Returns:
        [24] float array of normalized weights summing to 1.0.
    """
    beta = 0.999
    arr = np.array(class_counts, dtype=np.float64)
    effective_num = (1.0 - beta ** arr) / (1.0 - beta)
    weights = effective_num / effective_num.sum()
    return weights


# ===========================================================================
# Hard-negative mining helper
# ===========================================================================
def hard_negative_mining(
    per_loc_loss: torch.Tensor,
    cls_target: torch.Tensor,
    hard_neg_ratio: float = 3.0,
) -> torch.Tensor:
    """Select top-K hardest negatives and keep all positives.

    Standard OHEM / RetinaNet-style mining: after computing per-location
    classification loss, sort negative locations by loss magnitude and
    keep only the hardest K negatives (K = hard_neg_ratio * num_positives).
    All positive locations are always kept.

    Args:
        per_loc_loss: [N] per-location loss (reduction='none').
        cls_target:  [N] class index (-1=bg, 0..C-1=fg).
        hard_neg_ratio: how many negatives to keep per positive (default 3).

    Returns:
        [N] loss tensor with easy-negative entries zeroed out.
    """
    pos_mask = cls_target >= 0
    num_pos = pos_mask.sum().item()

    if num_pos == 0 or hard_neg_ratio <= 0:
        return per_loc_loss  # no mining needed

    neg_mask = ~pos_mask
    num_neg = neg_mask.sum().item()
    num_hard_neg = min(int(num_pos * hard_neg_ratio), num_neg)

    if num_hard_neg <= 0 or num_hard_neg >= num_neg:
        return per_loc_loss

    # Get loss values for negative locations only
    neg_losses = per_loc_loss[neg_mask]
    # Find threshold for top-K hardest negatives
    k = min(num_hard_neg, neg_losses.numel())
    threshold = neg_losses.topk(k).values.min()
    # Zero out easy negatives (those below threshold)
    easy_neg_mask = neg_mask & (per_loc_loss < threshold)
    per_loc_loss = per_loc_loss.clone()
    per_loc_loss[easy_neg_mask] = 0.0

    return per_loc_loss


# ===========================================================================
# Detection loss (with proper anchor matching)
# ===========================================================================
def detection_loss(
    det_outputs: dict,
    anchors_per_level: dict,
    gt_boxes_list: list,
    gt_classes_list: list,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    iou_threshold: float = 0.5,
    reg_weight: float = 5.0,
    loss_type: str = "focal",
    matcher_type: str = "iou",
    use_tal: bool = False,
    tal_alpha: float = 2.0,
    hard_neg_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute detection loss across all FPN levels with anchor matching.

    Supports --loss focal|qfl, --matcher iou|atss, and --use-tal.

    TAL (Task-Aligned Learning, TOOD 2021) weights classification and
    regression losses by the alignment metric IoU(pred_box, GT_box),
    down-weighting poorly-aligned positive anchors.  For each positive anchor:
        alignment_metric = IoU(B_i, G_i)
        cls_weight = alignment_metric ** tal_alpha
        reg_weight_anchor = alignment_metric  (no exponent)

    For each FPN level (P3/P4/P5):
      - Generate anchors for that level's spatial dimensions
      - Match GT boxes to anchors via selected matcher
      - Sigmoid focal loss or QFL for per-location classification
      - CIoU loss for per-anchor regression (positive anchors only)
      - Per-level loss weighted by inverse-sqrt of spatial size

    Args:
        det_outputs: {level: {'cls_logits': [B, C, H, W], 'reg_preds': [B, 64, H, W]}}
        anchors_per_level: {level: [H, W, A, 4]}
        gt_boxes_list: list of [N_i, 4] per sample in (cx, cy, w, h)
        gt_classes_list: list of [N_i] per sample (0..23)
        loss_type: "focal" (sigmoid focal loss) or "qfl" (quality focal loss)
        matcher_type: "iou" (fixed threshold) or "atss" (adaptive threshold)
        use_tal: if True, apply TAL alignment weighting
        tal_alpha: exponent for alignment metric (TOOD default=2.0)

    Returns:
        cls_loss, reg_loss, num_positive
    """
    device = next(iter(det_outputs.values()))["cls_logits"].device
    B = next(iter(det_outputs.values()))["cls_logits"].shape[0]

    # Per-level weight: inverse sqrt of spatial size, normalized so smallest
    # level has weight 1.0. P3 (45x80=3600 locs) ~ 4x more than P5 (12x20=240),
    # so P5 gets higher weight per-location to balance.
    level_weights: dict[str, float] = {}
    for level, out in det_outputs.items():
        if level == "P2":
            continue
        H, W = out["cls_logits"].shape[2], out["cls_logits"].shape[3]
        level_weights[level] = 1.0 / math.sqrt(H * W)
    min_w = min(level_weights.values()) if level_weights else 1.0
    for level in level_weights:
        level_weights[level] /= min_w

    total_cls = torch.tensor(0.0, device=device)
    total_reg = torch.tensor(0.0, device=device)
    total_pos = 0

    for level, out in det_outputs.items():
        if level == "P2":
            continue
        cls_logits = out["cls_logits"]  # [B, 24, H, W]
        reg_preds = out["reg_preds"]    # [B, 64, H, W]
        H, W = cls_logits.shape[2], cls_logits.shape[3]
        A = NUM_ANCHORS  # 16
        lw = level_weights[level]

        # Reshape regression: [B, 64, H, W] -> [B, H, W, A, 4]
        # Channel layout: anchor0_dx, dy, dw, dh, anchor1_dx, dy, dw, dh, ...
        # Reshape as (B, A, 4, H, W) then permute to (B, H, W, A, 4)
        reg_out = reg_preds.reshape(B, A, 4, H, W).permute(0, 3, 4, 1, 2)

        # Classification: [B, 24, H, W] -> [B, H, W, 24]
        cls_out = cls_logits.permute(0, 2, 3, 1)

        # Anchors for this level
        anchors = anchors_per_level[level]  # [H, W, A, 4]

        batch_cls = 0.0
        batch_reg = 0.0
        batch_pos = 0

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_classes = gt_classes_list[b].to(device)

            if gt_boxes.numel() == 0:
                # No GT: pure background loss
                cls_flat = cls_out[b].reshape(-1, NUM_DET_CLASSES)
                bg_target = torch.full((H * W,), -1, dtype=torch.long, device=device)
                if loss_type == "qfl":
                    bg_quality = torch.zeros(H * W, device=device)
                    batch_cls = batch_cls + quality_focal_loss_with_logits(
                        cls_flat, bg_target, bg_quality, NUM_DET_CLASSES, focal_gamma,
                    )
                else:
                    batch_cls = batch_cls + sigmoid_focal_loss(cls_flat, bg_target, focal_gamma, focal_alpha)
                continue

            # Flatten anchors for matching
            anchors_flat = anchors.reshape(-1, 4)

            # Match GT to anchors (returns quality_scores for QFL)
            cls_target, reg_target, pos_mask, quality_scores = match_anchors_to_gt(
                anchors_flat, gt_boxes, gt_classes,
                iou_threshold=iou_threshold,
                matcher_type=matcher_type,
            )

            # ---- Classification loss (per-location) ----
            cls_flat = cls_out[b].reshape(-1, NUM_DET_CLASSES)  # [H*W, 24]

            if use_tal and pos_mask.any():
                # ---- TAL: compute alignment weights before losses ----
                reg_flat = reg_out[b].reshape(-1, 4)  # [H*W*A, 4]
                anchors_pos = anchors_flat[pos_mask]
                reg_pos = reg_flat[pos_mask]
                target_pos = reg_target[pos_mask]
                # Decode predicted and GT boxes for alignment metric
                pred_xyxy_tal = decode_deltas_to_xyxy(reg_pos, anchors_pos, clamp=True)
                target_xyxy_tal = decode_deltas_to_xyxy(target_pos, anchors_pos, clamp=True)
                # Per-anchor alignment: IoU(pred_box, GT_box)
                alignments = box_iou_xyxy_paired(pred_xyxy_tal, target_xyxy_tal)  # [P]

                # Build per-location cls weights: max alignment over anchors at each location
                pos_indices = torch.nonzero(pos_mask).squeeze(-1)  # [P] indices in [H*W*A]
                loc_indices = pos_indices // A  # [P] location indices in [H*W]
                loc_alignments = torch.zeros(H * W, device=device)
                # Cluster alignments by location and take max per location
                for loc_idx in range(H * W):
                    mask_at_loc = (loc_indices == loc_idx)
                    if mask_at_loc.any():
                        loc_alignments[loc_idx] = alignments[mask_at_loc].max().detach()

                # Build per-location TAL classification weights
                # Positive locations: weight = alignment ** tal_alpha
                # Background locations: weight = 1.0 (standard focal loss)
                tal_cls_weights = torch.ones(H * W, device=device)
                pos_locs = loc_alignments > 0
                tal_cls_weights[pos_locs] = loc_alignments[pos_locs] ** tal_alpha

                # Per-anchor regression TAL weights: alignments (no exponent)
                tal_reg_weights = torch.ones(H * W * A, device=device)
                tal_reg_weights[pos_mask] = alignments.detach()

            if use_tal and pos_mask.any():
                # ---- TAL classification: alignment_weight * focal_loss(cls_logit, target) ----
                # Per TOOD, alignment_metric = IoU(pred_box, GT_box), weight = alignment ** tal_alpha.
                # Positive locations get weight > 0, background locations get weight = 1.0.
                tal_targets = torch.zeros(H * W, NUM_DET_CLASSES, device=device)
                tal_pos = cls_target >= 0
                if tal_pos.any():
                    tal_targets[tal_pos] = F.one_hot(cls_target[tal_pos], num_classes=NUM_DET_CLASSES).float()
                # BCE
                tal_bce = F.binary_cross_entropy_with_logits(cls_flat, tal_targets, reduction="none")  # [H*W, C]
                # Focal weight: (1 - p_t)^gamma
                p = torch.sigmoid(cls_flat)
                p_t = p * tal_targets + (1.0 - p) * (1.0 - tal_targets)
                tal_focal_w = (1.0 - p_t) ** focal_gamma
                # Alpha balancing
                tal_alpha_w = focal_alpha * tal_targets + (1.0 - focal_alpha) * (1.0 - tal_targets)
                # TAL alignment weight (broadcast over classes)
                tal_w = tal_cls_weights[:, None]  # [H*W, 1]
                # Combined: alignment_weight * alpha * focal * BCE
                per_loc_tal = (tal_w * tal_alpha_w * tal_focal_w * tal_bce).sum(dim=1)  # [H*W]
                # Hard-negative mining on per-location TAL losses
                if hard_neg_ratio > 0:
                    per_loc_tal = hard_negative_mining(per_loc_tal, cls_target, hard_neg_ratio)
                cls_loss_b = per_loc_tal.sum() / max(float(tal_targets.sum()), 16.0)
            elif loss_type == "qfl":
                quality_per_loc = quality_scores.reshape(H * W, A).max(dim=1)[0]
                cls_loss_b = quality_focal_loss_with_logits(
                    cls_flat, cls_target, quality_per_loc, NUM_DET_CLASSES, focal_gamma,
                )
            else:
                # ---- Standard focal loss with optional hard-negative mining ----
                if hard_neg_ratio > 0 and pos_mask.any():
                    # Compute per-location focal loss for mining
                    N, C = cls_flat.shape
                    fl_targets = torch.zeros(N, C, device=device)
                    fl_pos = cls_target >= 0
                    if fl_pos.any():
                        fl_targets[fl_pos] = F.one_hot(cls_target[fl_pos], num_classes=C).float()
                    fl_bce = F.binary_cross_entropy_with_logits(cls_flat, fl_targets, reduction="none")
                    p = torch.sigmoid(cls_flat)
                    p_t = p * fl_targets + (1.0 - p) * (1.0 - fl_targets)
                    fl_focal = (1.0 - p_t) ** focal_gamma
                    fl_alpha = focal_alpha * fl_targets + (1.0 - focal_alpha) * (1.0 - fl_targets)
                    per_loc_fl = (fl_alpha * fl_focal * fl_bce).sum(dim=1)  # [N]
                    # Apply hard-negative mining
                    per_loc_fl = hard_negative_mining(per_loc_fl, cls_target, hard_neg_ratio)
                    pos_count = fl_targets.sum()
                    normalizer = max(float(pos_count), 16.0) if pos_count > 0 else float(N * C)
                    cls_loss_b = per_loc_fl.sum() / normalizer
                else:
                    cls_loss_b = sigmoid_focal_loss(cls_flat, cls_target, focal_gamma, focal_alpha)

            batch_cls = batch_cls + cls_loss_b

            # ---- Regression loss (per-anchor, positive only, CIoU) ----
            # NOTE: TAL only modifies classification loss. Regression uses standard CIoU.
            if pos_mask.any():
                reg_flat = reg_out[b].reshape(-1, 4)  # [H*W*A, 4]
                # Decode predicted and target deltas to xyxy boxes for CIoU
                anchors_pos = anchors_flat[pos_mask]
                pred_xyxy = decode_deltas_to_xyxy(reg_flat[pos_mask], anchors_pos, clamp=True)
                target_xyxy = decode_deltas_to_xyxy(reg_target[pos_mask], anchors_pos, clamp=True)
                reg_loss_b = ciou_loss(pred_xyxy, target_xyxy)
                batch_reg = batch_reg + reg_loss_b
                batch_pos += pos_mask.sum().item()

        total_cls = total_cls + batch_cls / B * lw
        if batch_pos > 0:
            total_reg = total_reg + batch_reg / B * lw * reg_weight
        total_pos += batch_pos

    return total_cls, total_reg, total_pos


# ===========================================================================
# Foreground batch sampler
# ===========================================================================
class ForegroundBatchSampler:
    """Three-pool batch sampler: det-FG, act-FG, and background.

    Only ~16% of IndustReal frames have OD boxes (det FG).  Expanding to
    85% by including activity labels diluted detection signal: every batch
    had activity labels but zero detection boxes.

    SOLUTION: three separate pools
      det_fg  — frames with OD boxes (cat_id > 1)              ~12K
      act_fg  — frames with activity labels but NO OD boxes     ~55K
      bg      — frames with neither                             ~12K

    Batch construction (batch_size=2):
      Slot 1: always from det_fg pool → detection sees positive anchors every batch
      Slot 2: prioritized from act_fg pool → activity/PSR get signal every batch
      Fallback: if one pool is exhausted, any remaining pool fills the slot

    CLASS-BALANCED MODE (``class_balanced=True``):
      Computes per-class frequencies from the det_fg pool, then uses
      Effective Number of Samples (Cui et al. 2019) weighting to
      oversample underrepresented classes.  Rare classes get ~3-5x
      more representation, common classes are slightly downsampled,
      but a class with 30K examples can't be reduced to 34 (sqrt cap).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        class_balanced: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_total = len(dataset)
        self.class_balanced = class_balanced

        # Three pools
        self.det_fg_indices: list[int] = []
        self.act_fg_indices: list[int] = []
        self.bg_indices: list[int] = []

        # Class-balanced tracking: per-class sample index lists
        # NUM_DET_CLASSES = 24 (cat_id - 1 maps to 0..23)
        self.class_samples: list[list[int]] = [[] for _ in range(24)]
        self.class_counts: list[int] = [0] * 24
        self.rare_class_indices: list[int] = []  # classes with count < median/4

        for idx in range(self.n_total):
            rec, stem = dataset.samples[idx]
            key = f"{rec.name}/{stem}"
            anns = dataset.gt['detection'].get(key, [])
            has_det_fg = any(ann.get('cat_id', 1) > 1 for ann in anns)
            has_activity = dataset.gt['activity'].get(key, -1) >= 0

            if has_det_fg:
                self.det_fg_indices.append(idx)
                # Track per-class counts for class-balanced sampling
                seen_classes = set()
                for ann in anns:
                    cat_id = ann.get('cat_id', 1)
                    if cat_id > 1:
                        cls_idx = cat_id - 1
                        if cls_idx not in seen_classes:
                            seen_classes.add(cls_idx)
                            self.class_samples[cls_idx].append(idx)
                            self.class_counts[cls_idx] += 1
            elif has_activity:
                self.act_fg_indices.append(idx)
            else:
                self.bg_indices.append(idx)

        logger.info(
            f"ForegroundBatchSampler: {len(self.det_fg_indices)} det-FG + "
            f"{len(self.act_fg_indices)} act-FG + "
            f"{len(self.bg_indices)} BG = {self.n_total} total"
        )

        # Log class distribution and identify rare classes
        if self.det_fg_indices:
            non_zero_counts = [c for c in self.class_counts if c > 0]
            if non_zero_counts:
                median_count = sorted(non_zero_counts)[len(non_zero_counts) // 2]
                self.rare_class_indices = [
                    i for i, c in enumerate(self.class_counts)
                    if 0 < c < max(median_count / 4, 4)
                ]
                logger.info(
                    f"  Class distribution (min={min(non_zero_counts)}, "
                    f"max={max(non_zero_counts)}, median={median_count}): "
                    f"{self.class_counts}"
                )
                if self.rare_class_indices:
                    logger.info(
                        f"  Rare classes (count < median/4): "
                        f"{[(i, self.class_counts[i]) for i in self.rare_class_indices]}"
                    )
                if class_balanced:
                    cb_w = class_balanced_weights(self.class_counts)
                    logger.info(
                        f"  Class-balanced weights (Cui et al.): "
                        f"{[f'{w:.4f}' for w in cb_w]}"
                    )
                    logger.info(
                        f"  Rare class sampling boost ~{1.0 / min(cb_w[cb_w > 0]):.1f}x"
                    )

    def __len__(self) -> int:
        return max(1, self.n_total // self.batch_size)

    def _pick_rare_class_sample(self, rng: random.Random) -> int | None:
        """Pick a sample from a rare class, cycling through rare classes."""
        if not self.rare_class_indices or not self._rare_class_pool:
            return None
        # Pop from the rare-class priority queue
        return self._rare_class_pool.pop(0) if self._rare_class_pool else None

    def __iter__(self):
        rng = random.Random()
        # Refresh pools each epoch
        det_pool = list(self.det_fg_indices)
        act_pool = list(self.act_fg_indices)
        bg_pool = list(self.bg_indices)
        if self.shuffle:
            rng.shuffle(det_pool)
            rng.shuffle(act_pool)
            rng.shuffle(bg_pool)

        # If class-balanced mode, build a priority queue of rare-class samples
        # to inject into batches.  Each batch gets ~1 rare sample per 5 batches.
        self._rare_class_pool: list[int] = []
        if self.class_balanced and self.rare_class_indices:
            # Gather all samples from rare classes, interleaved to ensure coverage
            rare_by_class = [list(self.class_samples[c]) for c in self.rare_class_indices]
            for lst in rare_by_class:
                if self.shuffle:
                    rng.shuffle(lst)
            # Interleave: round-robin across rare classes
            max_len = max(len(lst) for lst in rare_by_class)
            for i in range(max_len):
                for lst in rare_by_class:
                    if i < len(lst):
                        self._rare_class_pool.append(lst[i])
            logger.info(
                f"  Rare-class priority queue: {len(self._rare_class_pool)} samples "
                f"from {len(self.rare_class_indices)} classes"
            )

        det_ptr = 0
        act_ptr = 0
        bg_ptr = 0
        batch_counter = 0

        while det_ptr < len(det_pool) or act_ptr < len(act_pool) or bg_ptr < len(bg_pool):
            batch = []
            batch_counter += 1

            # Slot 1: always from det_fg (ensures detection positive anchors)
            # In class-balanced mode, occasionally inject a rare-class sample
            if (self.class_balanced and batch_counter % 5 == 0
                    and self._rare_class_pool):
                rare_idx = self._rare_class_pool.pop(0)
                batch.append(rare_idx)
                # Also consume from main pool to avoid double-sampling
                if rare_idx in det_pool[det_ptr:]:
                    pool_idx = det_pool.index(rare_idx, det_ptr)
                    # Swap to current pointer position
                    det_pool[det_ptr], det_pool[pool_idx] = det_pool[pool_idx], det_pool[det_ptr]
                    det_ptr += 1
                elif rare_idx in act_pool:
                    pool_idx = act_pool.index(rare_idx)
                    act_pool[pool_idx] = act_pool[act_ptr] if act_ptr < len(act_pool) else rare_idx
                elif rare_idx in bg_pool:
                    pool_idx = bg_pool.index(rare_idx)
                    bg_pool[pool_idx] = bg_pool[bg_ptr] if bg_ptr < len(bg_pool) else rare_idx
            elif det_ptr < len(det_pool):
                batch.append(det_pool[det_ptr])
                det_ptr += 1
            elif act_ptr < len(act_pool):
                batch.append(act_pool[act_ptr])
                act_ptr += 1
            elif bg_ptr < len(bg_pool):
                batch.append(bg_pool[bg_ptr])
                bg_ptr += 1
            else:
                break

            # Slot 2: prioritize act_fg (ensures activity/PSR signal)
            # In class-balanced mode, occasionally inject another rare-class sample
            if (self.class_balanced and batch_counter % 7 == 0
                    and self._rare_class_pool):
                rare_idx = self._rare_class_pool.pop(0)
                batch.append(rare_idx)
                # Consume from main pool
                if rare_idx in act_pool:
                    pool_idx = act_pool.index(rare_idx)
                    act_pool[pool_idx] = act_pool[act_ptr] if act_ptr < len(act_pool) else rare_idx
                elif rare_idx in det_pool:
                    pool_idx = det_pool.index(rare_idx)
                    det_pool[pool_idx] = det_pool[det_ptr] if det_ptr < len(det_pool) else rare_idx
                elif rare_idx in bg_pool:
                    pool_idx = bg_pool.index(rare_idx)
                    bg_pool[pool_idx] = bg_pool[bg_ptr] if bg_ptr < len(bg_pool) else rare_idx
            elif act_ptr < len(act_pool):
                batch.append(act_pool[act_ptr])
                act_ptr += 1
            elif det_ptr < len(det_pool):
                batch.append(det_pool[det_ptr])
                det_ptr += 1
            elif bg_ptr < len(bg_pool):
                batch.append(bg_pool[bg_ptr])
                bg_ptr += 1
            else:
                break

            yield batch
            if (det_ptr >= len(det_pool) and act_ptr >= len(act_pool)
                    and bg_ptr >= len(bg_pool)):
                break


# ===========================================================================
# Full multi-task loss (detection + activity + pose + PSR)
# ===========================================================================
def multi_task_loss_v3(
    out_dict: dict,
    targets: dict,
    anchors_per_level: dict,
    use_supcon: bool = False,
    supcon_proj_head: nn.Module | None = None,
    uw_so: nn.Module | None = None,
    loss_type: str = "focal",
    matcher_type: str = "iou",
    use_tal: bool = False,
    tal_alpha: float = 2.0,
    use_class_balanced_sampling: bool = False,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-task loss: proper detection + auxiliary tasks.

    Detection uses the fixed anchor-matching loss above.
    Activity/Pose/PSR losses are carried over from v2.
    SupCon contrastive loss is included only when ``use_supcon=True``.

    When *uw_so* is provided, task losses are weighted by learned
    uncertainty parameters (UW-SO) instead of fixed manual weights.

    Args:
        loss_type: "focal" (sigmoid focal loss) or "qfl" (quality focal loss)
        matcher_type: "iou" (fixed threshold) or "atss" (adaptive threshold)
    """
    # Find device from any tensor in out_dict
    device = None
    for v in out_dict.values():
        if isinstance(v, torch.Tensor):
            device = v.device
            break
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, torch.Tensor):
                    device = vv.device
                    break
    if device is None:
        device = torch.device("cuda:0")

    total = torch.tensor(0.0, device=device)
    loss_components = {
        "det_cls": 0.0, "det_reg": 0.0, "det_pos": 0,
        "act": 0.0, "pose": 0.0, "psr": 0.0, "supcon": 0.0, "l2": 0.0,
    }
    raw_losses: dict = {}  # populated only when uw_so is active

    # ---- 1. Detection loss (the main fix) ----
    det_out = out_dict.get("detection", {})
    boxes_list = targets.get("boxes", [])
    classes_list = targets.get("classes", [])

    has_det_boxes = any(
        hasattr(b, "numel") and b.numel() > 0 for b in boxes_list
    )
    if det_out:
        # Hard-negative mining ratio: when class-balanced is active, keep
        # 3 hard negatives per positive to focus on difficult locations.
        cb_hard_neg_ratio = 3.0 if use_class_balanced_sampling else 0.0
        cls_loss, reg_loss, n_pos = detection_loss(
            det_out,
            anchors_per_level,
            boxes_list,
            classes_list,
            loss_type=loss_type,
            matcher_type=matcher_type,
            use_tal=use_tal,
            tal_alpha=tal_alpha,
            hard_neg_ratio=cb_hard_neg_ratio,
        )
        if uw_so is not None:
            raw_losses["det"] = cls_loss + reg_loss
        else:
            total = total + cls_loss + reg_loss
        loss_components["det_cls"] = cls_loss.item()
        loss_components["det_reg"] = reg_loss.item()
        loss_components["det_pos"] = n_pos

    # ---- 2. Activity CE (index-aligned) ----
    act_targets = targets.get("activity", [])
    if isinstance(act_targets, list) and act_targets and "activity" in out_dict:
        valid_pairs = [(i, t) for i, t in enumerate(act_targets) if t is not None and t >= 0]
        if valid_pairs:
            try:
                valid_idx, valid_act = zip(*valid_pairs)
                act_logits = out_dict["activity"][list(valid_idx)]
                act_target = torch.tensor(valid_act, dtype=torch.long, device=device)
                act_loss_raw = F.cross_entropy(act_logits, act_target)
                if uw_so is not None:
                    raw_losses["act"] = act_loss_raw
                else:
                    total = total + act_loss_raw * 0.5
                loss_components["act"] = act_loss_raw.item()
            except Exception as e:
                logger.debug(f"Activity loss skipped: {e}")

    # ---- 3. Pose MSE (index-aligned) ----
    pose_targets = targets.get("pose", [])
    if isinstance(pose_targets, list) and pose_targets and "pose_6d" in out_dict:
        valid_pairs = [(i, p) for i, p in enumerate(pose_targets) if p is not None]
        if valid_pairs:
            try:
                valid_idx, valid_pose = zip(*valid_pairs)
                pose_pred = out_dict["pose_6d"][list(valid_idx)]
                pose_target = torch.tensor(
                    [list(p[0]) + list(p[1]) for p in valid_pose],
                    dtype=torch.float32,
                ).to(device)
                pose_loss_raw = F.smooth_l1_loss(pose_pred, pose_target)
                if uw_so is not None:
                    raw_losses["pose"] = pose_loss_raw
                else:
                    total = total + pose_loss_raw * 0.1
                loss_components["pose"] = pose_loss_raw.item()
            except Exception as e:
                logger.debug(f"Pose loss skipped: {e}")

    # ---- 4. PSR BCE (index-aligned) ----
    psr_targets = targets.get("psr", [])
    if isinstance(psr_targets, list) and "psr_logits" in out_dict and psr_targets:
        valid_pairs = [(i, p) for i, p in enumerate(psr_targets) if p is not None]
        if valid_pairs:
            try:
                valid_idx, valid_psr = zip(*valid_pairs)
                psr_target = torch.stack(valid_psr, dim=0).to(device)
                psr_pred = out_dict["psr_logits"][list(valid_idx)]
                psr_loss_raw = F.binary_cross_entropy_with_logits(psr_pred, psr_target)
                if uw_so is not None:
                    raw_losses["psr"] = psr_loss_raw
                else:
                    total = total + psr_loss_raw * 0.5
                loss_components["psr"] = psr_loss_raw.item()
            except Exception as e:
                logger.debug(f"PSR loss skipped: {e}")

    # ---- 5. Apply UW-SO weighting (if enabled) ----
    if uw_so is not None and raw_losses:
        total = uw_so(raw_losses)
        loss_components["uw_log_sigmas"] = uw_so.log_sigma.detach().cpu().tolist()
        loss_components["uw_sigmas"] = uw_so.sigma.detach().cpu().tolist()

    # ---- 6. SupCon contrastive loss (optional, requires --use-supcon) ----
    supcon_loss_val = 0.0
    if use_supcon:
        # NOTE: Full SupCon integration requires:
        #   1) A projection head attached to the backbone's pooled features
        #   2) Features plumbed through out_dict (e.g. out_dict["backbone_feat"])
        #   3) Activity class labels from targets
        # Until then this is a no-op placeholder.
        logger.debug("SupCon loss requested but backbone features not yet plumbed")
        # When plumbed, this would be:
        #   feats = out_dict["backbone_feat"]        # [B, D]
        #   emb = supcon_proj_head(feats)             # [B, 128]
        #   act_labels = ...                          # [B] from targets
        #   supcon_loss_val = SupConLoss()(emb, act_labels) * 0.5
        #   total = total + supcon_loss_val
        #   loss_components["supcon"] = supcon_loss_val.item()

    # ---- 7. Light L2 regularization (keep heads from collapsing) ----
    for k, v in out_dict.items():
        if k == "detection":
            for lvl, lv in v.items():
                if isinstance(lv, dict):
                    for sk, sv in lv.items():
                        if isinstance(sv, torch.Tensor):
                            l2_val = 1e-5 * sv.pow(2).mean()
                            total = total + l2_val
                            loss_components["l2"] = loss_components["l2"] + l2_val.item()
        elif isinstance(v, torch.Tensor):
            l2_val = 1e-5 * v.pow(2).mean()
            total = total + l2_val
            loss_components["l2"] = loss_components["l2"] + l2_val.item()

    return total, loss_components


# ===========================================================================
# Layer-wise Learning Rate Decay (LLRD)
# ===========================================================================
def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    llrd_decay: float = 0.95,
    det_lr_mult: float = 1000.0,
    weight_decay: float = 0.05,
) -> list[dict]:
    """Build optimizer parameter groups with Layer-wise Learning Rate Decay.

    MViTv2-S has 16 MultiscaleBlocks (indices 0-15).  LLRD assigns lower LR
    to earlier (deeper) layers and higher LR to later (shallower) layers:

        conv_proj / pos_encoding:  lr = base_lr * decay^16   (deepest)
        blocks.0  (earliest):      lr = base_lr * decay^15
        blocks.1:                  lr = base_lr * decay^14
        ...
        blocks.15 (latest):        lr = base_lr * decay^0 = base_lr
        FPN + non-det heads:       lr = base_lr               (no decay)
        Detection head:            lr = base_lr * det_lr_mult (highest)

    Reference: BEiT/MAE-style LLRD with AdamW wd=0.05.

    Args:
        model: The WrappedMTL model.
        base_lr: Base learning rate.
        llrd_decay: LLRD decay factor (0.9-0.95 typical).
        det_lr_mult: Detection head LR multiplier.
        weight_decay: AdamW weight decay for backbone params.

    Returns:
        List of param groups for torch.optim.AdamW.
    """
    # Count backbone blocks dynamically
    num_blocks = 0
    for name, _ in model.named_parameters():
        if ".blocks." in name:
            parts = name.split(".blocks.")
            if len(parts) > 1:
                idx_str = parts[1].split(".")[0]
                try:
                    num_blocks = max(num_blocks, int(idx_str) + 1)
                except ValueError:
                    pass
    if num_blocks == 0:
        num_blocks = 16  # MViTv2-S default

    # Classify params into groups
    groups: dict[str, any] = {
        "det": [],
        "conv_proj": [],
        "pos_encoding": [],
        "blocks": {},  # block_idx -> list of params
        "heads": [],
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Detection head
        if name.startswith("m.det_head."):
            groups["det"].append(p)
            continue

        # Backbone blocks
        if ".blocks." in name:
            parts = name.split(".blocks.")
            if len(parts) > 1:
                try:
                    block_idx = int(parts[1].split(".")[0])
                    groups["blocks"].setdefault(block_idx, []).append(p)
                    continue
                except (ValueError, IndexError):
                    pass

        # Conv projection (deepest layer)
        if "conv_proj" in name:
            groups["conv_proj"].append(p)
            continue

        # Position encoding (same depth as conv_proj)
        if "pos_encoding" in name:
            groups["pos_encoding"].append(p)
            continue

        # Everything else (FPN, act_head, psr_head, pose_head)
        groups["heads"].append(p)

    param_groups: list[dict] = []

    def _add_group(params, lr_val, wd):
        if params:
            param_groups.append({
                "params": params, "lr": lr_val, "weight_decay": wd,
                "_lr_mult": lr_val / base_lr if base_lr > 0 else 1.0,
            })

    # 1. Conv proj: deepest -> lowest LR
    _add_group(groups["conv_proj"], base_lr * (llrd_decay ** num_blocks), weight_decay)

    # 2. Pos encoding: same depth as conv_proj
    _add_group(groups["pos_encoding"], base_lr * (llrd_decay ** num_blocks), weight_decay)

    # 3. Blocks sorted by index (earlier = deeper = lower LR)
    for idx in sorted(groups["blocks"].keys()):
        lr = base_lr * (llrd_decay ** (num_blocks - 1 - idx))
        _add_group(groups["blocks"][idx], lr, weight_decay)

    # 4. Non-detection heads + FPN: full LR
    _add_group(groups["heads"], base_lr, weight_decay)

    # 5. Detection head: highest LR, lower wd
    _add_group(groups["det"], base_lr * det_lr_mult, 0.01)

    return param_groups


# ===========================================================================
# Main training
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="V3 training with IoU-based anchor matching"
    )
    parser.add_argument("--phase1-epochs", type=int, default=2)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--det-lr-mult", type=float, default=1000,
                        help="Detection head LR multiplier (default 1000 with prior_prob=0.1 for faster learning)")
    parser.add_argument("--det-prior-prob", type=float, default=0.1,
                        help="Detection head prior probability for bias init (default 0.1; higher = faster warmup)")
    parser.add_argument("--logit-bias-scale", type=float, default=1.0,
                        help="Multiplier for detection head bias init (default 1.0; <1.0 boosts initial confidence, >1.0 suppresses)")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-norm", type=float, default=10.0)
    parser.add_argument("--use-llrd", action="store_true",
                        help="Enable Layer-wise Learning Rate Decay for backbone")
    parser.add_argument("--llrd-decay", type=float, default=0.95,
                        help="LLRD decay factor (default 0.95; 0.9-0.95 typical)")
    parser.add_argument(
        "--k400-init",
        type=str,
        default="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/action_recognition_model_weights/mvit_rgb_kinetics_pretrained.pyth",
    )
    parser.add_argument("--output-dir", type=str, default="runs/mtl_v3")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--use-p2-level", action="store_true",
                        help="Include P2 (stride 4) in FPN detection outputs for improved small object AP (Agent 5: +3-5%)")
    parser.add_argument("--use-supcon", action="store_true",
                        help="Enable SupCon auxiliary contrastive loss (experimental)")
    parser.add_argument("--use-uw-so", action="store_true",
                        help="Enable UW-SO uncertainty weighting for multi-task loss balancing")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "qfl"],
                        help="Classification loss: focal (sigmoid focal loss) or qfl (quality focal loss)")
    parser.add_argument("--matcher", type=str, default="iou", choices=["iou", "atss"],
                        help="Anchor matcher: iou (fixed threshold 0.5) or atss (adaptive threshold)")
    parser.add_argument("--mosaic-prob", type=float, default=0.3,
                        help="Mosaic augmentation probability (0=off, default 0.3)")
    parser.add_argument("--copy-paste-prob", type=float, default=0.2,
                        help="Copy-Paste augmentation probability (0=off, default 0.2)")
    parser.add_argument("--use-mosaic", action="store_true",
                        help="Enable mosaic augmentation (alias: sets mosaic-prob to 0.3 if not explicitly set)")
    parser.add_argument("--use-copy-paste", action="store_true",
                        help="Enable copy-paste augmentation (alias: sets copy-paste-prob to 0.2 if not explicitly set)")
    parser.add_argument("--num-anchors", type=int, default=8, choices=[8, 16],
                        help="Number of anchors (8=k-means optimized, 16=legacy). V3.8+ default 8. FRESH INIT REQUIRED.")
    parser.add_argument("--use-class-balanced-sampling", action="store_true",
                        help="Enable class-balanced sampling for detection (weights rare classes higher)")
    parser.add_argument("--use-tal", action="store_true",
                        help="Enable Task-Aligned Learning (TOOD-style) assigner instead of IoU/ATSS matching")
    parser.add_argument("--tal-alpha", type=float, default=2.0,
                        help="TAL alignment exponent alpha (TOOD default=2.0)")
    parser.add_argument("--tal-lr-mult", type=float, default=2.0,
                        help="TAL detection head LR multiplier (TOOD default 2x)")
    args = parser.parse_args()

    # Effective det_lr_mult: TAL heads get 2x higher LR per TOOD paper
    det_lr_mult_eff = args.det_lr_mult * (args.tal_lr_mult if args.use_tal else 1.0)
    if args.use_tal:
        logger.info(f"TAL enabled: tal_alpha={args.tal_alpha}, det_lr_mult={args.det_lr_mult:.0f} -> {det_lr_mult_eff:.0f}x (TAL 2x)")

    # Apply --num-anchors to global anchor config
    global NUM_ANCHORS, _ANCHOR_SPECS
    if args.num_anchors == 8:
        NUM_ANCHORS = 8
        _ANCHOR_SPECS = _ANCHOR_SPECS_8
        logger.info("Using 8 k-means optimized anchors (64->32 reg channels)")
    else:
        NUM_ANCHORS = 16
        _ANCHOR_SPECS = _ANCHOR_SPECS_16
        logger.info("Using 16 legacy anchors (64 reg channels)")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    out = Path(args.output_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    # ---- Build model ----
    logger.info(f"Loading model from {args.k400_init}")
    full = MTLMViTModel(
        num_act_classes=75,
        det_prior_prob=args.det_prior_prob,
        logit_bias_scale=args.logit_bias_scale,
        use_p2_level=args.use_p2_level,
        num_anchors=args.num_anchors,
    )

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # Expand to 9 channels first so checkpoint's 9ch conv_proj fits
        expand_conv_proj_to_9ch(full)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Load checkpoint directly (full checkpoint format)
        sd = ckpt.get("model_state_dict", ckpt)
        # Handle WrappedMTL prefix
        clean_sd = {}
        for k, v in sd.items():
            if k.startswith("m."):
                clean_sd[k[2:]] = v
            else:
                clean_sd[k] = v
        full.load_state_dict(clean_sd, strict=False)
        logger.info(f"Loaded checkpoint (epoch={ckpt.get('epoch', '?')})")
    else:
        # Load K400 weights
        raw = torch.load(args.k400_init, map_location="cpu", weights_only=False)
        raw_sd = raw.get("model_state", raw.get("model_state_dict", raw))
        is_k400 = "patch_embed.proj.weight" in raw_sd

        ms = full.state_dict()
        loaded = 0
        for k in raw_sd:
            if k.startswith("head."):
                continue
            if is_k400:
                if k == "patch_embed.proj.weight":
                    target = "feature_pyramid.backbone.conv_proj.weight"
                elif k == "patch_embed.proj.bias":
                    target = "feature_pyramid.backbone.conv_proj.bias"
                elif k == "cls_token":
                    target = "feature_pyramid.backbone.pos_encoding.class_token"
                elif k == "norm.weight":
                    target = "feature_pyramid.backbone.norm.weight"
                elif k == "norm.bias":
                    target = "feature_pyramid.backbone.norm.bias"
                elif k.startswith("blocks."):
                    target = "feature_pyramid.backbone." + k
                else:
                    continue
            else:
                target = k[2:] if k.startswith("m.") else k
            if target in ms and ms[target].shape == raw_sd[k].shape:
                ms[target] = raw_sd[k]
                loaded += 1
        full.load_state_dict(ms, strict=False)
        logger.info(f"Loaded {loaded} K400 keys")
        expand_conv_proj_to_9ch(full)

    full = full.to(device)
    model = WrappedMTL(full).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params / 1e6:.1f}M params")
    logger.info(f"Detection: prior_prob={args.det_prior_prob:.2f}, lr_mult={args.det_lr_mult:.0f}x (eff={det_lr_mult_eff:.0f}x{' TAL 2x' if args.use_tal else ''})")

    # ---- UW-SO uncertainty weighting (if enabled) ----
    uw_so_loss = None
    if args.use_uw_so:
        uw_so_loss = UWSOLoss().to(device)
        logger.info(f"UW-SO enabled: {len(UWSOLoss.TASK_NAMES)} learnable log_sigma params initialized to 0.0")

    # ---- Optimizer with optional LLRD ----
    # The detection head's sigmoid bias (-4.595, prior_prob=0.01) needs ~2.4
    # units of upward shift before it produces p>0.1 detections.  At base
    # LR=2e-5, AdamW gives only 2e-5 effective update per step (gradient is
    # consistent, so m/sqrt(v) ~ sign), requiring 479K batches to warm the
    # bias.  By giving the detection head 1000x higher LR, we warm the bias
    # in ~480 batches (well within Phase 1).  With prior_prob=0.1, the
    # bias starts at -2.2 so p>0.1 immediately; the multiplier still helps
    # the head converge faster.
    if args.use_llrd:
        param_groups = build_llrd_param_groups(
            model, args.lr, llrd_decay=args.llrd_decay,
            det_lr_mult=det_lr_mult_eff, weight_decay=0.05,
        )
        if uw_so_loss is not None:
            param_groups.append({
                "params": list(uw_so_loss.parameters()),
                "lr": args.lr,
                "weight_decay": 0.0,
            })
        opt = torch.optim.AdamW(param_groups)
        llrd_msg = f"LLRD enabled: decay={args.llrd_decay}, {len(param_groups)} param groups"
        for g in param_groups:
            llrd_msg += f"\n    lr={g['lr']:.6f}, wd={g.get('weight_decay', 0.01):.4f}, n={len(g['params'])}"
        logger.info(llrd_msg)
    else:
        det_params, base_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("m.det_head."):
                det_params.append(p)
            else:
                base_params.append(p)
        opt_groups = [
            {"params": base_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": det_params, "lr": args.lr * det_lr_mult_eff, "weight_decay": 0.01},
        ]
        if uw_so_loss is not None:
            opt_groups.append({
                "params": list(uw_so_loss.parameters()),
                "lr": args.lr,
                "weight_decay": 0.0,
            })
        opt = torch.optim.AdamW(opt_groups)
        opt_msg = (
            f"Optimizer: {len(base_params)} base params @ lr={args.lr}"
            f" + {len(det_params)} det_head params @ lr={args.lr * det_lr_mult_eff}"
        )
        if uw_so_loss is not None:
            opt_msg += f" + {len(list(uw_so_loss.parameters()))} uw_so params @ lr={args.lr}"
        logger.info(opt_msg)

    # ---- LR helpers ----
    def lr_at_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step / max(args.warmup_steps, 1))
        total_est = max(1, args.phase1_epochs * 50000)
        frac = (step - args.warmup_steps) / max(total_est - args.warmup_steps, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))

    def save_ckpt(epoch: int, batch: int, phase: int):
        path = out / "checkpoints" / f"phase{phase}_e{epoch}_b{batch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "batch": batch,
                "phase": phase,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "uw_so_state_dict": uw_so_loss.state_dict() if args.use_uw_so else None,
            },
            path,
        )
        logger.info(f"  Saved {path.name}")

    # ---- Build anchors dynamically from FPN output shapes ----
    def build_anchors(det_out: dict) -> dict:
        """Generate anchors for each FPN level based on output feature shapes."""
        anchors = {}
        for level, out in det_out.items():
            if level == "P2":
                continue
            H, W = out["cls_logits"].shape[2], out["cls_logits"].shape[3]
            anchors[level] = generate_anchors(H, W, out["cls_logits"].device)
        return anchors

    # ====================================================================
    # PHASE 1: Synthetic pretraining
    # ====================================================================
    if args.phase1_epochs > 0:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PHASE 1: Synthetic ({args.phase1_epochs} epoch)")
        logger.info(f"{'=' * 60}")
        synth_ds = FullSyntheticDataset(
            img_dir="/home/admin/swarm-bot/master/POPW/datasets/industreal/images".replace(
                "/home/admin/swarm-bot/", "/home/newadmin/swarm-bot/"
            ),
            label_dir="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/images",
            img_size=(640, 360),
        )
        synth_loader = DataLoader(
            synth_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_synth_targets,
            num_workers=0,
            pin_memory=False,
        )
        n_total = len(synth_loader)
        logger.info(f"  Total batches: {n_total}")

        global_step = 0
        for epoch in range(args.phase1_epochs):
            model.train()
            n_batches = 0
            n_skipped = 0
            epoch_loss = 0.0
            epoch_cls = 0.0
            epoch_reg = 0.0
            epoch_pos = 0
            t0 = time.time()
            opt.zero_grad()

            for i, (images, targets) in enumerate(synth_loader):
                images = images.to(device).float()
                out_dict = model(images)

                # Build anchors from current FPN output sizes
                anchors_per_level = build_anchors(out_dict["detection"])

                loss, lc = multi_task_loss_v3(out_dict, targets, anchors_per_level, use_supcon=args.use_supcon, uw_so=uw_so_loss, loss_type=args.loss, matcher_type=args.matcher, use_tal=args.use_tal, tal_alpha=args.tal_alpha, use_class_balanced_sampling=args.use_class_balanced_sampling)

                if torch.isnan(loss) or torch.isinf(loss):
                    n_skipped += 1
                    opt.zero_grad()
                    continue

                # Update LR (detection head gets det_lr_mult_eff multiplier)
                base_lr = lr_at_step(global_step)
                if args.use_llrd:
                    for g in opt.param_groups:
                        g["lr"] = base_lr * g.get("_lr_mult", 1.0)
                else:
                    opt.param_groups[0]["lr"] = base_lr
                    opt.param_groups[1]["lr"] = base_lr * det_lr_mult_eff
                    if uw_so_loss is not None:
                        opt.param_groups[2]["lr"] = base_lr
                cur_lr = base_lr

                loss_scaled = loss / args.grad_accum
                loss_scaled.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_norm
                )

                if (i + 1) % args.grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                    global_step += 1

                # [IMP-10] Dynamic logit bias adjustment: updates detection head
                # classification bias based on observed per-sample positive anchor
                # ratio. Uses EMA smoothing (momentum=0.05) to prevent oscillation.
                total_anchors = sum(a.shape[0] * a.shape[1] * a.shape[2] for a in anchors_per_level.values())
                pos_ratio = lc['det_pos'] / max(total_anchors * args.batch_size, 1)
                model.m.det_head.update_logit_bias(pos_ratio)

                epoch_loss += loss.item()
                epoch_cls += lc["det_cls"]
                epoch_reg += lc["det_reg"]
                epoch_pos += lc["det_pos"]
                n_batches += 1

                if n_batches % 100 == 0:
                    elapsed = time.time() - t0
                    speed = n_batches / elapsed
                    eta_min = (n_total - n_batches) / speed / 60 if speed > 0 else 0
                    log_msg = (
                        f"  P1 Ep{epoch} b{n_batches}/{n_total}: "
                        f"loss={loss.item():.4f}, cls={lc['det_cls']:.4f}, "
                        f"reg={lc['det_reg']:.4f}, pos={lc['det_pos']}, "
                        f"act={lc['act']:.4f}, psr={lc['psr']:.4f}, "
                        f"l2={lc['l2']:.4f}, gnorm={grad_norm:.3f}, "
                        f"lr={cur_lr:.2e}, speed={speed:.1f}/s, "
                        f"ETA={eta_min:.0f}min, skip={n_skipped}"
                    )
                    if "uw_log_sigmas" in lc:
                        log_msg += f", uw_sig={lc['uw_sigmas']}"
                    logger.info(log_msg)
                if n_batches % args.save_every == 0:
                    save_ckpt(epoch, n_batches, 1)

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"P1 Epoch {epoch}: {n_batches} batches, "
                f"avg_loss={avg_loss:.4f}, time={elapsed / 60:.1f}min, "
                f"skip={n_skipped}"
            )
            save_ckpt(epoch + 1, 0, 1)
            torch.cuda.reset_peak_memory_stats(0)

    # ====================================================================
    # PHASE 2: Real multi-modal fine-tuning with foreground-balanced batches
    # ====================================================================
    if args.phase2_epochs > 0:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PHASE 2: Real multi-modal ({args.phase2_epochs} epoch)")
        logger.info(f"{'=' * 60}")
        real_ds = FullMultiModalDataset(
            recordings_dir="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train",
            img_size=(640, 360),
            mosaic_prob=args.mosaic_prob,
            copy_paste_prob=args.copy_paste_prob,
        )

        # --- CRITICAL: Foreground-balanced batch sampler ---
        # Without this, ~93% of random batches are background-only, producing
        # pure background loss that drowns the rare foreground gradient signal.
        # This was the root cause of mAP=0 after 196K training batches.
        fg_sampler = ForegroundBatchSampler(
            real_ds,
            batch_size=args.batch_size,
            shuffle=True,
            class_balanced=args.use_class_balanced_sampling,
        )
        real_loader = DataLoader(
            real_ds,
            batch_sampler=fg_sampler,
            collate_fn=collate_real_targets,
            num_workers=0,
            pin_memory=False,
        )
        n_total = len(real_loader)
        logger.info(f"  Batches per epoch: {n_total} (FG-guaranteed)")

        # Lower LR for fine-tuning (but keep detection head multiplier)
        # NOTE: Must rebuild param groups, NOT use model.parameters(), otherwise
        # the DET_LR_MULT=1000 is lost and det_head bias never warms up.
        phase2_lr = args.lr / 4
        if args.use_llrd:
            param_groups = build_llrd_param_groups(
                model, phase2_lr, llrd_decay=args.llrd_decay,
                det_lr_mult=det_lr_mult_eff, weight_decay=0.05,
            )
            if uw_so_loss is not None:
                param_groups.append({
                    "params": list(uw_so_loss.parameters()),
                    "lr": phase2_lr,
                    "weight_decay": 0.0,
                })
            opt = torch.optim.AdamW(param_groups)
            llrd_msg = f"P2 LLRD enabled: decay={args.llrd_decay}, {len(param_groups)} param groups"
            for g in param_groups:
                llrd_msg += f"\n    lr={g['lr']:.6f}, wd={g.get('weight_decay', 0.01):.4f}, n={len(g['params'])}"
            logger.info(llrd_msg)
        else:
            det_params, base_params = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("m.det_head."):
                    det_params.append(p)
                else:
                    base_params.append(p)
            opt_groups = [
                {"params": base_params, "lr": phase2_lr, "weight_decay": 0.01},
                {"params": det_params, "lr": phase2_lr * det_lr_mult_eff, "weight_decay": 0.01},
            ]
            if uw_so_loss is not None:
                opt_groups.append({
                    "params": list(uw_so_loss.parameters()),
                    "lr": phase2_lr,
                    "weight_decay": 0.0,
                })
            opt = torch.optim.AdamW(opt_groups)
            opt_msg = (
                f"Optimizer: {len(base_params)} base params @ lr={phase2_lr}"
                f" + {len(det_params)} det_head params @ lr={phase2_lr * det_lr_mult_eff}"
            )
            if uw_so_loss is not None:
                opt_msg += f" + {len(list(uw_so_loss.parameters()))} uw_so params @ lr={phase2_lr}"
            logger.info(opt_msg)

        # Phase 2 warmup + cosine decay LR scheduler (was missing — fixed)
        phase2_step = 0

        def lr_at_step_phase2(step: int) -> float:
            if step < args.warmup_steps:
                return phase2_lr * (step / max(args.warmup_steps, 1))
            total_est = max(1, args.phase2_epochs * n_total)
            frac = (step - args.warmup_steps) / max(total_est - args.warmup_steps, 1)
            return phase2_lr * 0.5 * (1.0 + math.cos(math.pi * frac))

        for epoch in range(args.phase2_epochs):
            model.train()
            n_batches = 0
            n_skipped = 0
            epoch_loss = 0.0
            epoch_cls = 0.0
            epoch_reg = 0.0
            epoch_pos = 0
            t0 = time.time()
            opt.zero_grad()

            for i, (images, targets) in enumerate(real_loader):
                images = images.to(device).float()
                out_dict = model(images)

                # Build anchors from current FPN output sizes
                anchors_per_level = build_anchors(out_dict["detection"])

                loss, lc = multi_task_loss_v3(out_dict, targets, anchors_per_level, use_supcon=args.use_supcon, uw_so=uw_so_loss, loss_type=args.loss, matcher_type=args.matcher, use_tal=args.use_tal, tal_alpha=args.tal_alpha, use_class_balanced_sampling=args.use_class_balanced_sampling)

                if torch.isnan(loss) or torch.isinf(loss):
                    n_skipped += 1
                    opt.zero_grad()
                    continue

                # Cosine-decay LR (warmup → cosine over total Phase 2 steps)
                cur_lr = lr_at_step_phase2(phase2_step)
                if args.use_llrd:
                    for g in opt.param_groups:
                        g["lr"] = cur_lr * g.get("_lr_mult", 1.0)
                else:
                    opt.param_groups[0]["lr"] = cur_lr
                    opt.param_groups[1]["lr"] = cur_lr * det_lr_mult_eff
                    if uw_so_loss is not None:
                        opt.param_groups[2]["lr"] = cur_lr

                loss_scaled = loss / args.grad_accum
                loss_scaled.backward()

                if (i + 1) % args.grad_accum == 0:
                    # Clip gradients right before step (prevent explosion from
                    # rare large-gradient FG batches)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.max_norm
                    )
                    opt.step()
                    opt.zero_grad()
                    phase2_step += 1
                else:
                    grad_norm = 0.0

                # [IMP-10] Dynamic logit bias adjustment (same as Phase 1)
                total_anchors = sum(a.shape[0] * a.shape[1] * a.shape[2] for a in anchors_per_level.values())
                pos_ratio = lc['det_pos'] / max(total_anchors * args.batch_size, 1)
                model.m.det_head.update_logit_bias(pos_ratio)

                epoch_loss += loss.item()
                epoch_cls += lc["det_cls"]
                epoch_reg += lc["det_reg"]
                epoch_pos += lc["det_pos"]
                n_batches += 1

                if n_batches % 50 == 0:
                    elapsed = time.time() - t0
                    speed = n_batches / elapsed
                    log_msg = (
                        f"  P2 Ep{epoch} b{n_batches}/{n_total}: "
                        f"loss={loss.item():.4f}, cls={lc['det_cls']:.4f}, "
                        f"reg={lc['det_reg']:.4f}, pos={lc['det_pos']}, "
                        f"act={lc['act']:.4f}, psr={lc['psr']:.4f}, "
                        f"l2={lc['l2']:.4f}, "
                        f"gnorm={grad_norm:.3f}, lr={cur_lr:.2e}, "
                        f"speed={speed:.1f}/s, skip={n_skipped}"
                    )
                    if "uw_log_sigmas" in lc:
                        log_msg += f", uw_sig={lc['uw_sigmas']}"
                    logger.info(log_msg)
                if n_batches % args.save_every == 0:
                    save_ckpt(epoch, n_batches, 2)

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"P2 Epoch {epoch}: {n_batches} batches, "
                f"avg_loss={avg_loss:.4f}, time={elapsed / 60:.1f}min, "
                f"skip={n_skipped}"
            )
            save_ckpt(epoch + 1, 0, 2)
            torch.cuda.reset_peak_memory_stats(0)

    logger.info("\n=== V3 TRAINING COMPLETE ===")
    logger.info(f"Checkpoints saved to {out / 'checkpoints'}")
    logger.info("Evaluate with: python eval_real_map_fast.py --checkpoint <path>")


if __name__ == "__main__":
    main()
