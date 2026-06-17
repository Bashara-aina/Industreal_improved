"""
compute_metrics() — single-batch multi-task metric dispatcher
==============================================================
Routes a single batch's model outputs (pred) and targets (target) to the
correct evaluation functions in evaluate.py.

Metrics returned: mAP50, F1_action, MAE, F1_psr, combined

Author: Bashara
Date: May 2026
"""

from __future__ import annotations

import sys
import os

# Ensure src/ is in path so that evaluate.py can import config
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import torch

from evaluation.evaluate import (
    compute_activity_metrics,
    compute_det_metrics_extended,
    compute_head_pose_metrics,
    compute_psr_metrics,
)


# ---------------------------------------------------------------------------
# Helper: detection boxes from heatmaps
# ---------------------------------------------------------------------------

def _heatmaps_to_detection(
    pred_heatmaps: torch.Tensor,
    gt_heatmap: torch.Tensor,
    score_thresh: float = 0.3,
    nms_thresh: float = 0.5,
) -> tuple:
    """
    [FIX D8] GUARDED PLACEHOLDER — DO NOT USE FOR REAL EVALUATION.

    This is a legacy placeholder that produces fake 64x64 boxes centered on
    heatmap peaks. If called with cls_preds (24-channel classification scores)
    instead of actual heatmaps, it silently reports mAP values that are MEANINGLESS.
    Use compute_det_metrics_extended with real cls_preds + reg_preds from the model.

    Returns: (dp_boxes, dp_scores, dp_labels, dg_boxes, dg_labels)

    Raises:
        RuntimeError: always — this placeholder must never be silently called.
    """
    import logging as _lg
    _lg.getLogger(__name__).warning(
        '[FIX D8] _heatmaps_to_detection placeholder called — producing FAKE 64x64 boxes! '
        'This will result in meaningless mAP values. Fix caller to use real cls_preds/reg_preds.'
    )
    raise RuntimeError(
        '_heatmaps_to_detection is a placeholder and must not be silently called. '
        'Use compute_det_metrics_extended with real cls_preds + reg_preds from the model.'
    )
    # Placeholder: treat each spatial location as a detection candidate
    B, C, H, W = pred_heatmaps.shape
    device = pred_heatmaps.device

    dp_boxes, dp_scores, dp_labels = [], [], []
    dg_boxes, dg_labels = [], []

    for b in range(B):
        # Per-class peak detection (simple argmax over spatial dims)
        cls_sigmoid = torch.sigmoid(pred_heatmaps[b])  # [C, H, W]
        max_scores, max_locs = cls_sigmoid.max(dim=1)   # [H, W], [H, W]

        keep = max_scores > score_thresh
        scores = max_scores[keep].cpu().numpy()
        classes = max_locs[keep].cpu().numpy()

        # Fake boxes around peak locations
        h_ratio, w_ratio = H / 720.0, W / 1280.0
        ys, xs = torch.where(keep)
        if len(ys) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            ys_np, xs_np = ys.cpu().numpy(), xs.cpu().numpy()
            boxes = np.stack([
                xs_np - 32, ys_np - 32,
                xs_np + 32, ys_np + 32,
            ], axis=1).astype(np.float32)

        dp_boxes.append(boxes)
        dp_scores.append(scores)
        dp_labels.append(classes)

        # GT boxes from gt_heatmap (if available)
        if gt_heatmap is not None:
            # gt_heatmap shape: [B, 17, 64, 64] (COCO-keypoint style) or [B, ...]
            # Just use zeros as placeholder GT
            dg_boxes.append(np.zeros((0, 4), dtype=np.float32))
            dg_labels.append(np.zeros(0, dtype=np.int64))
        else:
            dg_boxes.append(np.zeros((0, 4), dtype=np.float32))
            dg_labels.append(np.zeros(0, dtype=np.int64))

    return dp_boxes, dp_scores, dp_labels, dg_boxes, dg_labels


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: dict,
    target: dict,
) -> dict:
    """
    Compute all multi-task metrics for a single batch.

    Args:
        pred: dict with keys:
            - act_logits  : torch.Tensor [B, 75] activity logits
            - heatmaps     : torch.Tensor [B, 24, H, W] detection heatmaps (or cls_preds)
            - psr_logits   : torch.Tensor [B, 11] PSR logits
            - head_pose    : torch.Tensor [B, 9] head pose predictions
        target: dict with keys:
            - activity     : torch.Tensor [B] activity labels
            - heatmap      : torch.Tensor [B, ...] detection GT (optional)
            - psr_labels   : torch.Tensor [B, 11] PSR binary labels
            - head_pose    : torch.Tensor [B, 9] head pose GT

    Returns:
        dict with keys:
            - mAP50        : detection mAP@0.5
            - F1_action    : activity macro-F1
            - MAE          : head pose MAE (raw)
            - F1_psr       : PSR overall F1
            - combined     : combined multi-task score
    """
    results = {}

    # ---- Activity F1 ----
    act_logits = pred.get('act_logits')
    activity_labels = target.get('activity')
    if act_logits is not None and activity_labels is not None:
        act_logits_np = act_logits.detach().cpu().numpy()
        activity_labels_np = activity_labels.detach().cpu().numpy()
        act_pred_np = act_logits_np.argmax(axis=1)
        act_metrics = compute_activity_metrics(
            all_gt=activity_labels_np,
            all_pred=act_pred_np,
            all_logits=act_logits_np,
        )
        results['F1_action'] = act_metrics.get('act_macro_f1', 0.0)
    else:
        results['F1_action'] = 0.0

    # ---- Detection mAP50 ----
    # Fallback: compute from heatmaps if cls_preds/reg_preds not available
    heatmaps = pred.get('heatmaps')
    gt_heatmap = target.get('heatmap')
    if heatmaps is not None:
        try:
            # Try compute_det_metrics_extended if we have proper box format
            # Since we may not, use heatmap-based detection fallback
            dp_boxes, dp_scores, dp_labels, dg_boxes, dg_labels = _heatmaps_to_detection(
                heatmaps, gt_heatmap
            )
            det_metrics = compute_det_metrics_extended(
                dp_boxes, dp_scores, dp_labels,
                dg_boxes, dg_labels,
            )
            results['mAP50'] = det_metrics.get('det_mAP50', 0.0)
        except Exception:
            results['mAP50'] = 0.0
    else:
        results['mAP50'] = 0.0

    # ---- Head Pose MAE ----
    head_pose_pred = pred.get('head_pose')
    head_pose_gt = target.get('head_pose')
    if head_pose_pred is not None and head_pose_gt is not None:
        hp_pred_np = head_pose_pred.detach().cpu().numpy()
        hp_gt_np = head_pose_gt.detach().cpu().numpy()
        hp_metrics = compute_head_pose_metrics(hp_pred_np, hp_gt_np)
        results['MAE'] = hp_metrics.get('head_pose_MAE', 0.0)
    else:
        results['MAE'] = 0.0

    # ---- PSR F1 ----
    psr_logits = pred.get('psr_logits')
    psr_labels = target.get('psr_labels')
    if psr_logits is not None and psr_labels is not None:
        psr_logits_np = psr_logits.detach().cpu().numpy()
        psr_labels_np = psr_labels.detach().cpu().numpy()
        psr_metrics = compute_psr_metrics(psr_logits_np, psr_labels_np)
        results['F1_psr'] = psr_metrics.get('psr_overall_f1', 0.0)
    else:
        results['F1_psr'] = 0.0

    # ---- Combined score ----
    # Weighted combination (higher is better):
    # mAP50 (det) * 0.25 + F1_action * 0.25 + (1 - MAE/10) * 0.25 + F1_psr * 0.25
    # Normalize MAE so it's 0-1 where 1 is best (inverse, clamped)
    mae_component = max(0.0, 1.0 - results['MAE'] / 10.0)
    combined = (
        results['mAP50'] * 0.25 +
        results['F1_action'] * 0.25 +
        mae_component * 0.25 +
        results['F1_psr'] * 0.25
    )
    results['combined'] = combined

    return results