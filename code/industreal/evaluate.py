import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Evaluation Metrics for Multi-Task IndustReal Model
====================================================
4 tasks:
  - Activity Recognition (AR): 74 classes — reuse IKEA compute_activity_metrics
  - Assembly State Detection (ASD): 24 classes — reuse IKEA compute_det_metrics_extended
  - Head Pose: 9-DoF MAE — new compute_head_pose_metrics
  - Procedure Step Recognition (PSR): 11-component F1 — new compute_psr_metrics

Author: Bashara
Date: April 2026
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, classification_report,
)

import config as C

logger = logging.getLogger(__name__)


# =============================================================================
# Image Helpers
# =============================================================================

def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
        mean = torch.tensor(C.IMAGENET_MEAN, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


# =============================================================================
# Activity Recognition (AR) — 74 classes
# =============================================================================

def compute_activity_metrics(
    all_gt,
    all_pred,
    all_logits=None,
    class_names=None,
    save_dir=None,
):
    """
    Comprehensive activity recognition metrics.
    Identical interface to IKEA evaluate.py — just pass C.ACT_CLASS_NAMES.

    Args:
        all_gt      : np.ndarray [N] -- ground truth class ids
        all_pred    : np.ndarray [N] -- predicted class ids
        all_logits  : np.ndarray [N, C] or None -- raw logits for top-k
        class_names : list of str or None
        save_dir    : str or None -- if provided, saves confusion matrix image

    Returns:
        dict with activity metrics
    """
    all_gt = np.asarray(all_gt)
    all_pred = np.asarray(all_pred)
    num_classes = len(class_names) if class_names else C.NUM_CLASSES_ACT
    labels = list(range(num_classes))

    # 1. Frame accuracy (all classes)
    fa_all = float(accuracy_score(all_gt, all_pred))

    # 2. Frame accuracy excluding NA (class 0)
    mask_no_na = all_gt != 0
    fa_no_na = float(accuracy_score(all_gt[mask_no_na], all_pred[mask_no_na])) if mask_no_na.sum() > 0 else 0.0

    # 3. Macro-F1
    macro_f1 = float(f1_score(all_gt, all_pred, average='macro',
                               zero_division=0, labels=labels))
    present_labels = [i for i in labels if np.sum(all_gt == i) > 0]
    macro_f1_present = float(f1_score(all_gt, all_pred, average='macro',
                                      zero_division=0, labels=present_labels))

    # 4. Weighted-F1
    weighted_f1 = float(f1_score(all_gt, all_pred, average='weighted',
                                  zero_division=0))

    # 5. Macro-Recall
    macro_recall = float(recall_score(all_gt, all_pred, average='macro',
                                       zero_division=0, labels=labels))

    # 6. Mean per-class accuracy
    cm = confusion_matrix(all_gt, all_pred, labels=labels)
    row_sums = cm.sum(axis=1).clip(min=1)
    per_class_acc = cm.diagonal() / row_sums
    mean_per_class_acc = float(per_class_acc.mean())

    # 7. Top-5 accuracy (requires raw logits)
    top5_acc = 0.0
    if all_logits is not None:
        all_logits = np.asarray(all_logits)
        top5_indices = np.argsort(all_logits, axis=1)[:, -5:]
        top5_correct = np.any(top5_indices == all_gt[:, None], axis=1)
        top5_acc = float(top5_correct.mean())

    # 8. Per-class report
    report = {}
    if class_names is not None:
        report = classification_report(
            all_gt, all_pred,
            target_names=class_names,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )

    # 9. Save confusion matrix
    if save_dir is not None and class_names is not None:
        _save_confusion_matrix(cm, class_names, save_dir)

    return {
        'act_accuracy': fa_all,
        'act_accuracy_no_na': fa_no_na,
        'act_macro_f1': macro_f1,
        'act_macro_f1_present': macro_f1_present,
        'act_weighted_f1': weighted_f1,
        'act_macro_recall': macro_recall,
        'act_mean_per_class_acc': mean_per_class_acc,
        'act_top5_accuracy': top5_acc,
        'act_per_class_acc': per_class_acc.tolist(),
        'act_per_class_report': report,
        'act_confusion_matrix': cm.tolist(),
    }


def report_per_class_accuracy(cm_list, class_names=None, k: int = 5):
    """Log top-k worst and best per-class activity accuracy."""
    cm = np.asarray(cm_list, dtype=np.float64)
    if cm.size == 0:
        logger.info('Per-class activity report skipped: empty confusion matrix.')
        return

    row_sums = cm.sum(axis=1).clip(min=1.0)
    per_class_acc = cm.diagonal() / row_sums
    names = class_names if class_names is not None else [f'class_{i}' for i in range(len(per_class_acc))]

    sorted_idx = np.argsort(per_class_acc)
    worst_idx = sorted_idx[:k]
    best_idx = sorted_idx[-k:][::-1]

    logger.info('  📉 %d Worst Classes:', k)
    for idx in worst_idx:
        logger.info(f'    {names[idx]:30s}: {per_class_acc[idx]:.1%}')

    logger.info('  📈 %d Best Classes:', k)
    for idx in best_idx:
        logger.info(f'    {names[idx]:30s}: {per_class_acc[idx]:.1%}')

    logger.info(
        f'  Per-class accuracy summary: '
        f'macro={per_class_acc.mean():.1%} '
        f'min={per_class_acc.min():.1%} '
        f'max={per_class_acc.max():.1%}'
    )


def _save_confusion_matrix(cm, class_names, save_dir):
    """Save confusion matrix as PNG. Fails silently if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning('matplotlib/seaborn not available, skipping confusion matrix plot')
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Activity Confusion Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    logger.info(f'  Saved confusion matrix to {save_dir / "confusion_matrix.png"}')


# =============================================================================
# Assembly State Detection (ASD) — COCO-format detection, 24 classes
# =============================================================================

def compute_iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-6)


def decode_boxes(anchors, deltas):
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = np.clip(deltas[:, 2], -4, 4)
    dh = np.clip(deltas[:, 3], -4, 4)
    pw, ph = np.exp(dw) * a_w, np.exp(dh) * a_h
    cx, cy = dx * a_w + a_cx, dy * a_h + a_cy
    return np.stack([cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2], axis=1)


def nms_numpy(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep, dtype=np.int64)


def compute_ap_per_class(
    pred_boxes, pred_scores, pred_labels,
    gt_boxes, gt_labels,
    iou_thresh=0.5,
    num_classes=C.NUM_DET_CLASSES,
):
    """Per-class AP with 11-point interpolation."""
    aps = {}
    for cls in range(num_classes):
        all_tp, all_sc = [], []
        total_gt = 0
        for idx in range(len(gt_boxes)):
            gm = gt_labels[idx] == cls
            gb = gt_boxes[idx][gm]
            total_gt += len(gb)
            pm = pred_labels[idx] == cls
            pb = pred_boxes[idx][pm]
            ps = pred_scores[idx][pm]
            if len(pb) == 0:
                continue
            if len(gb) == 0:
                all_tp.extend([0] * len(pb))
                all_sc.extend(ps.tolist())
                continue
            ious = compute_iou_matrix(pb, gb)
            matched = set()
            for j in ps.argsort()[::-1]:
                bi = ious[j].argmax()
                if ious[j, bi] >= iou_thresh and bi not in matched:
                    all_tp.append(1)
                    matched.add(bi)
                else:
                    all_tp.append(0)
                all_sc.append(ps[j])
        if total_gt == 0:
            continue
        tp = np.array(all_tp)[np.array(all_sc).argsort()[::-1]]
        tc = np.cumsum(tp)
        fc = np.cumsum(1 - tp)
        rec = tc / total_gt
        prec = tc / (tc + fc)
        ap = sum(prec[rec >= t].max() if (rec >= t).any() else 0.0
                 for t in np.linspace(0, 1, 11)) / 11
        aps[cls] = float(ap)
    return {'mAP': float(np.mean(list(aps.values()))) if aps else 0.0, 'per_class_ap': aps}


def compute_det_metrics_extended(
    pred_boxes, pred_scores, pred_labels,
    gt_boxes, gt_labels,
    num_classes=C.NUM_DET_CLASSES,
):
    """
    Extended detection metrics: mAP@0.5 and mAP@[0.5:0.95].

    Returns:
        dict with det_mAP50, det_mAP_50_95, det_per_class_ap
    """
    r50 = compute_ap_per_class(
        pred_boxes, pred_scores, pred_labels,
        gt_boxes, gt_labels, 0.5, num_classes,
    )

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps_at_thresholds = []
    for iou_t in iou_thresholds:
        r = compute_ap_per_class(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, float(iou_t), num_classes,
        )
        maps_at_thresholds.append(r['mAP'])

    return {
        'det_mAP50': r50['mAP'],
        'det_mAP_50_95': float(np.mean(maps_at_thresholds)),
        'det_per_class_ap': r50['per_class_ap'],
    }


# =============================================================================
# Head Pose Metrics — 9-DoF (forward[3] + pos[3] + up[3])
# =============================================================================

def compute_head_pose_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """
    Compute head pose Mean Absolute Error per DoF and overall.

    The 9 DoFs are ordered as:
        0-2: forward_vector (forward_x, forward_y, forward_z)
        3-5: position       (pos_x, pos_y, pos_z)
        6-8: up_vector      (up_x, up_y, up_z)

    Args:
        pred: np.ndarray [N, 9] predicted head pose
        gt:   np.ndarray [N, 9] ground-truth head pose

    Returns:
        dict with per-DoF MAE, overall MAE, and std
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if pred.shape[0] == 0:
        return {
            'head_pose_MAE': float('nan'),
            'head_pose_MAE_std': float('nan'),
            'forward_x_MAE': float('nan'),
            'forward_y_MAE': float('nan'),
            'forward_z_MAE': float('nan'),
            'pos_x_MAE': float('nan'),
            'pos_y_MAE': float('nan'),
            'pos_z_MAE': float('nan'),
            'up_x_MAE': float('nan'),
            'up_y_MAE': float('nan'),
            'up_z_MAE': float('nan'),
        }

    abs_err = np.abs(pred - gt)  # [N, 9]

    dof_names = [
        'forward_x', 'forward_y', 'forward_z',
        'pos_x', 'pos_y', 'pos_z',
        'up_x', 'up_y', 'up_z',
    ]

    result = {}
    for i, name in enumerate(dof_names):
        result[f'{name}_MAE'] = float(abs_err[:, i].mean())

    result['head_pose_MAE'] = float(abs_err.mean())
    result['head_pose_MAE_std'] = float(abs_err.std())
    result['n_samples'] = int(pred.shape[0])

    return result


# =============================================================================
# Procedure Step Recognition (PSR) — Multi-label binary
# =============================================================================

def compute_psr_metrics(
    pred_logits: np.ndarray,
    gt_labels: np.ndarray,
    tolerance_frames: int = 3,
) -> Dict[str, float]:
    """
    Compute PSR metrics for 11 assembly components.

    PSR is multi-label: each component is either done (1) or not (0).
    We compute:
      - Per-component F1 (macro across thresholded predictions)
      - Overall F1 (macro over components)
      - F1@T (tolerance matching: allow ±T frames shift on state transitions)
      - Edit Score (Levenshtein distance on state-change sequences)
      - POS (Percentage of Ordering Success): for each component, segment
        the binary sequence into runs; for every consecutive run-pair in GT,
        check that prediction preserves the same order; average across components

    Args:
        pred_logits: np.ndarray [N, 11] sigmoid logits
        gt_labels:   np.ndarray [N, 11] binary labels (0/1, -1 for unknown/error)
        tolerance_frames: frames to tolerate on state transitions for F1@T

    Returns:
        dict with all PSR metrics
    """
    pred_logits = np.asarray(pred_logits)
    gt_labels = np.asarray(gt_labels)

    num_components = pred_logits.shape[1]

    # Mask out unknown/error labels (-1 in gt)
    valid_mask = gt_labels != -1  # [N, 11]
    gt_safe = gt_labels.copy()
    gt_safe[~valid_mask] = 0

    # Binarize predictions (threshold=0.5)
    pred_binary = (pred_logits > 0.5).astype(np.int64)

    # --- Per-component F1 ---
    per_component_f1 = {}
    component_names = [f'comp{i}' for i in range(num_components)]

    for c in range(num_components):
        vm = valid_mask[:, c]
        if vm.sum() == 0:
            per_component_f1[component_names[c]] = float('nan')
            continue
        tp = int(((pred_binary[vm, c] == 1) & (gt_safe[vm, c] == 1)).sum())
        fp = int(((pred_binary[vm, c] == 1) & (gt_safe[vm, c] == 0)).sum())
        fn = int(((pred_binary[vm, c] == 0) & (gt_safe[vm, c] == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            per_component_f1[component_names[c]] = 0.0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if prec + rec == 0:
                per_component_f1[component_names[c]] = 0.0
            else:
                per_component_f1[component_names[c]] = 2 * prec * rec / (prec + rec)

    valid_components = [c for c in range(num_components) if not np.isnan(per_component_f1[component_names[c]])]
    overall_f1 = float(np.nanmean([per_component_f1[component_names[c]] for c in valid_components])) if valid_components else 0.0

    # --- F1@T (tolerance matching) ---
    # For each component, allow ±tolerance_frames error on state change frames.
    # Treat as correct if predicted state change within T frames of GT change.
    f1_at_t_values = []
    for c in range(num_components):
        vm = valid_mask[:, c]
        if vm.sum() == 0:
            continue
        gt_c = gt_safe[vm, c]  # [V]
        pred_c = pred_binary[vm, c]

        # Find state change frames in GT
        gt_changes = np.where(np.diff(gt_c.astype(np.int32)) != 0)[0]
        pred_changes = np.where(np.diff(pred_c.astype(np.int32)) != 0)[0]

        if len(gt_changes) == 0:
            # No changes: any prediction is fine
            continue

        matched_gt = set()
        matched_pred = set()
        for pg in pred_changes:
            for tg in gt_changes:
                if abs(pg - tg) <= tolerance_frames and tg not in matched_gt:
                    matched_gt.add(tg)
                    matched_pred.add(pg)
                    break

        tp_t = len(matched_gt)
        fp_t = len(pred_changes) - len(matched_pred)
        fn_t = len(gt_changes) - len(matched_gt)

        if tp_t + fp_t == 0 or tp_t + fn_t == 0:
            f1_at_t_values.append(0.0)
        else:
            prec_t = tp_t / (tp_t + fp_t)
            rec_t = tp_t / (tp_t + fn_t)
            f1_at_t_values.append(2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0)

    psr_f1_at_t = float(np.mean(f1_at_t_values)) if f1_at_t_values else 0.0

    # --- Edit Score (Levenshtein on state-change sequences) ---
    edit_distances = []
    for c in range(num_components):
        vm = valid_mask[:, c]
        if vm.sum() == 0:
            continue
        gt_c = gt_safe[vm, c]
        pred_c = pred_binary[vm, c]

        # Build state change sequences as strings
        # Only include frames where state differs from initial
        init_state = int(gt_c[0])
        gt_seq = ''.join(str(int(v)) for v in gt_c)
        pred_seq = ''.join(str(int(v)) for v in pred_c)

        # Simple Levenshtein distance
        m, n = len(gt_seq), len(pred_seq)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if gt_seq[i - 1] == pred_seq[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        max_len = max(m, n, 1)
        edit_distances.append(dp[m][n] / max_len)

    edit_score = 1.0 - float(np.mean(edit_distances)) if edit_distances else 0.0

    # --- POS: Percentage of Ordering Success ---
    # For each component: segment GT binary sequence into runs.
    # For every consecutive run-pair (A,B) in GT, check if prediction has A before B.
    # POS = fraction of correctly-ordered run-pairs, averaged across valid components.

    def _find_runs(seq: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find runs in a binary sequence. Returns list of (start_idx, end_idx, value)."""
        if len(seq) == 0:
            return []
        runs = []
        start = 0
        current_val = int(seq[0])
        for i in range(1, len(seq)):
            if seq[i] != current_val:
                runs.append((start, i, current_val))
                start = i
                current_val = int(seq[i])
        runs.append((start, len(seq), current_val))
        return runs

    pos_values = []
    for c in range(num_components):
        vm = valid_mask[:, c]
        if vm.sum() == 0:
            continue
        gt_c = gt_safe[vm, c]
        pred_c = pred_binary[vm, c]

        gt_runs = _find_runs(gt_c)
        # POS requires at least 2 runs (i.e., at least one run-pair)
        if len(gt_runs) < 2:
            continue

        # Build a simplified prediction sequence: for each position, map to run index
        # Actually, we need to check: for each consecutive run-pair (A,B) in GT,
        # does pred have A appearing before B?
        pred_seq = pred_c  # binary sequence

        # For each run-pair (run_k, run_{k+1}) in GT, check ordering in prediction
        correct_pairs = 0
        total_pairs = len(gt_runs) - 1
        for k in range(total_pairs):
            run_a_start, run_a_end, val_a = gt_runs[k]
            run_b_start, run_b_end, val_b = gt_runs[k + 1]
            # In prediction, find any occurrence of val_a before any occurrence of val_b
            pred_a_positions = np.where(pred_seq == val_a)[0]
            pred_b_positions = np.where(pred_seq == val_b)[0]
            if len(pred_a_positions) == 0 or len(pred_b_positions) == 0:
                # If either value doesn't appear, order is not preserved
                continue
            # Check if there's at least one A before at least one B
            # (i.e., exists i in pred_a and j in pred_b where i < j)
            if np.any(pred_a_positions < pred_b_positions.min()):
                correct_pairs += 1

        pos_values.append(correct_pairs / total_pairs if total_pairs > 0 else 0.0)

    psr_pos = float(np.mean(pos_values)) if pos_values else 0.0

    return {
        'psr_overall_f1': overall_f1,
        'psr_f1_at_t': psr_f1_at_t,
        'psr_edit_score': edit_score,
        'psr_pos': psr_pos,
        'psr_per_component_f1': per_component_f1,
        'psr_num_valid_components': len(valid_components),
        'psr_num_samples': int(pred_logits.shape[0]),
    }


# =============================================================================
# Main Evaluation Loop
# =============================================================================

@torch.no_grad()
def evaluate_all(
    model: nn.Module,
    criterion,
    loader,
    device: torch.device,
    max_batches: int = 2500,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full evaluation returning all metrics across 4 IndustReal tasks.

    Args:
        model       : MultiTaskIndustReal
        criterion   : MultiTaskLoss
        loader      : DataLoader (val or test)
        device      : torch.device
        max_batches : int -- cap for speed during training validation
        save_dir    : str or None -- where to save confusion matrix

    Returns:
        dict with all metrics
    """
    model.eval()
    total_loss = 0.0
    lc = 0

    act_preds, act_labels, act_logits_all = [], [], []
    head_pose_preds, head_pose_gts = [], []
    psr_preds_logits, psr_labels = [], []
    dp_boxes, dp_scores, dp_labels = [], [], []
    dg_boxes, dg_labels = [], []

    _cached_anchors_np = None

    for bi, (images, targets) in enumerate(loader):
        if bi >= max_batches:
            break

        images = _prepare_images(images, device)

        # Move targets to device
        detection_list = targets['detection']
        for i in range(len(detection_list)):
            detection_list[i]['boxes'] = detection_list[i]['boxes'].to(device)
            detection_list[i]['labels'] = detection_list[i]['labels'].to(device)
        targets['head_pose'] = targets['head_pose'].to(device)
        targets['psr_labels'] = targets['psr_labels'].to(device)
        targets['activity'] = targets['activity'].to(device)

        # Forward pass: no autocast to avoid dtype mismatch (model weights are fp32, inputs should be fp32)
        # Autocast causes mixed fp16/fp32 that breaks RetinaNet on RTX 3060
        outputs_raw = model(images)

        # Cast all FP16 outputs to FP32 to prevent dtype mismatch in loss
        outputs = {}
        for _k, _v in outputs_raw.items():
            if isinstance(_v, torch.Tensor):
                outputs[_k] = _v.float()
            else:
                outputs[_k] = _v

        loss, _ = criterion(outputs, targets)
        if torch.isfinite(loss):
            total_loss += loss.float().item()
            lc += 1

        # --- Activity ---
        act_logits_batch = outputs['act_logits'].cpu().numpy()
        if act_logits_all is not None:
            act_logits_all.append(act_logits_batch)
        act_preds.append(act_logits_batch.argmax(axis=1))
        act_labels.append(targets['activity'].cpu().numpy())

        # --- Head Pose ---
        head_pose_preds.append(outputs['head_pose'].cpu().numpy())
        head_pose_gts.append(targets['head_pose'].cpu().numpy())

        # --- PSR ---
        psr_preds_logits.append(outputs['psr_logits'].cpu().numpy())
        psr_labels.append(targets['psr_labels'].cpu().numpy())

        # --- Detection ---
        if _cached_anchors_np is None:
            _cached_anchors_np = outputs['anchors'].cpu().numpy()

        cls_sigmoid = torch.sigmoid(outputs['cls_preds'])  # [B, N, 24] on GPU
        B = images.shape[0]

        for i in range(B):
            scores_i = cls_sigmoid[i]  # [N, 24] on GPU
            max_scores = scores_i.max(dim=1).values  # [N] on GPU
            score_thresh = float(getattr(C, 'DET_EVAL_SCORE_THRESH', 0.5))
            keep_mask = max_scores > score_thresh  # [N] bool on GPU

            max_keep = int(getattr(C, 'DET_EVAL_MAX_PER_IMAGE', 300))
            if max_keep > 0 and keep_mask.sum().item() > max_keep:
                topk_idx = torch.topk(max_scores, k=max_keep, largest=True, sorted=False).indices
                topk_mask = torch.zeros_like(keep_mask)
                topk_mask[topk_idx] = True
                keep_mask = keep_mask & topk_mask

            if keep_mask.sum().item() == 0:
                dp_boxes.append(np.zeros((0, 4)))
                dp_scores.append(np.zeros(0))
                dp_labels.append(np.zeros(0, dtype=np.int64))
            else:
                keep_np = keep_mask.cpu().numpy()
                kept_cls = scores_i[keep_mask].cpu().numpy()
                kept_reg = outputs['reg_preds'][i][keep_mask].cpu().numpy()
                kept_anc = _cached_anchors_np[keep_np]

                ms = kept_cls.max(axis=1)
                ml = kept_cls.argmax(axis=1)
                pb = decode_boxes(kept_anc, kept_reg)
                pb[:, 0] = np.clip(pb[:, 0], 0, C.IMG_WIDTH)
                pb[:, 1] = np.clip(pb[:, 1], 0, C.IMG_HEIGHT)
                pb[:, 2] = np.clip(pb[:, 2], 0, C.IMG_WIDTH)
                pb[:, 3] = np.clip(pb[:, 3], 0, C.IMG_HEIGHT)

                fb, fs, fl = [], [], []
                for c in range(C.NUM_DET_CLASSES):
                    cm = ml == c
                    if cm.sum() == 0:
                        continue
                    nk = nms_numpy(pb[cm], ms[cm], 0.5)
                    fb.append(pb[cm][nk])
                    fs.append(ms[cm][nk])
                    fl.append(np.full(len(nk), c, dtype=np.int64))
                if fb:
                    dp_boxes.append(np.concatenate(fb))
                    dp_scores.append(np.concatenate(fs))
                    dp_labels.append(np.concatenate(fl))
                else:
                    dp_boxes.append(np.zeros((0, 4)))
                    dp_scores.append(np.zeros(0))
                    dp_labels.append(np.zeros(0, dtype=np.int64))

            dg_boxes.append(detection_list[i]['boxes'].cpu().numpy())
            dg_labels.append(detection_list[i]['labels'].cpu().numpy())

        del images, outputs, cls_sigmoid

    if not act_preds:
        dataset_len = len(loader.dataset) if hasattr(loader, 'dataset') else -1
        raise RuntimeError(
            f'No batches were produced by DataLoader (dataset_len={dataset_len}). '
            f'Check split paths and filtering logic.'
        )

    results: Dict[str, Any] = {'loss': total_loss / max(lc, 1)}

    # -------------------------------------------------------------------------
    # Activity Metrics
    # -------------------------------------------------------------------------
    all_act_pred = np.concatenate(act_preds)
    all_act_gt = np.concatenate(act_labels)
    all_act_logits = np.concatenate(act_logits_all) if act_logits_all else None
    del act_preds, act_labels, act_logits_all

    act_metrics = compute_activity_metrics(
        all_act_gt, all_act_pred, all_act_logits,
        class_names=C.ACT_CLASS_NAMES,
        save_dir=save_dir,
    )
    results.update(act_metrics)
    report_per_class_accuracy(
        act_metrics.get('act_confusion_matrix', []),
        class_names=C.ACT_CLASS_NAMES,
        k=5,
    )
    logger.info(
        f'  Activity — Acc: {results["act_accuracy"]:.4f}  '
        f'Macro-F1: {results["act_macro_f1"]:.4f}  '
        f'Top-5: {results["act_top5_accuracy"]:.4f}'
    )

    # -------------------------------------------------------------------------
    # Head Pose Metrics
    # -------------------------------------------------------------------------
    all_hp_pred = np.concatenate(head_pose_preds)
    all_hp_gt = np.concatenate(head_pose_gts)
    del head_pose_preds, head_pose_gts

    hp_metrics = compute_head_pose_metrics(all_hp_pred, all_hp_gt)
    results.update(hp_metrics)

    logger.info(
        f'  Head Pose (9-DoF) — Overall MAE: {results["head_pose_MAE"]:.4f}  '
        f'std: {results["head_pose_MAE_std"]:.4f}  '
        f'forward_z: {results["forward_z_MAE"]:.4f}'
    )

    # -------------------------------------------------------------------------
    # PSR Metrics
    # -------------------------------------------------------------------------
    all_psr_logits = np.concatenate(psr_preds_logits)
    all_psr_labels = np.concatenate(psr_labels)
    del psr_preds_logits, psr_labels

    psr_metrics = compute_psr_metrics(all_psr_logits, all_psr_labels)
    results.update(psr_metrics)

    logger.info(
        f'  PSR — Overall F1: {results["psr_overall_f1"]:.4f}  '
        f'F1@T: {results["psr_f1_at_t"]:.4f}  '
        f'Edit Score: {results["psr_edit_score"]:.4f}  '
        f'POS: {results["psr_pos"]:.4f}'
    )

    # -------------------------------------------------------------------------
    # Detection Metrics
    # -------------------------------------------------------------------------
    gt_box_total = int(sum(len(x) for x in dg_boxes))
    if gt_box_total == 0:
        logger.warning('Detection evaluation skipped: no GT boxes found in this split.')
        det_metrics = {
            'det_mAP50': float('nan'),
            'det_mAP_50_95': float('nan'),
            'det_per_class_ap': {},
        }
    else:
        det_metrics = compute_det_metrics_extended(
            dp_boxes, dp_scores, dp_labels,
            dg_boxes, dg_labels,
        )
    results.update(det_metrics)

    logger.info(
        f'  ASD — mAP@0.5: {results.get("det_mAP50", float("nan")):.4f}  '
        f'mAP@[0.5:0.95]: {results.get("det_mAP_50_95", float("nan")):.4f}'
    )

    model.train()
    return results


# =============================================================================
# Standalone CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description='Evaluate Multi-Task IndustReal Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --checkpoint path/to/checkpoint.pth

  python evaluate.py --checkpoint model.pt --split val

  python evaluate.py --checkpoint model.pt --max-batches 100
        """
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split', type=str, default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--max-batches', type=int, default=9999,
        help='Maximum number of batches to evaluate'
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from model import MultiTaskIndustReal
    from losses import MultiTaskLoss
    from industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    save_dir = args.save_dir or str(C.EVAL_SAVE_DIR)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ds = IndustRealMultiTaskDataset(
        split=args.split,
        img_size=C.IMG_SIZE,
        augment=False,
        seed=C.SEED,
    )
    loader = DataLoader(
        ds,
        batch_size=C.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=C.VAL_NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = MultiTaskIndustReal(pretrained=False).to(device)
    criterion = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
    ).to(device)
    criterion.set_class_counts(ds.class_counts)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    results: Dict[str, Any] = evaluate_all(
        model, criterion, loader, device,
        max_batches=args.max_batches, save_dir=save_dir,
    )

    print('\n' + '=' * 60)
    print(f'IndustReal Evaluation Results ({args.split})')
    print('=' * 60)

    print('\nACTIVITY RECOGNITION')
    print('-' * 40)
    print(f'  Frame Accuracy (all)    : {results["act_accuracy"]:.4f}')
    print(f'  Frame Accuracy (no NA) : {results["act_accuracy_no_na"]:.4f}')
    print(f'  Macro-F1               : {results["act_macro_f1"]:.4f}')
    print(f'  Weighted-F1            : {results["act_weighted_f1"]:.4f}')
    print(f'  Macro-Recall           : {results["act_macro_recall"]:.4f}')
    print(f'  Mean Per-Class Accuracy: {results["act_mean_per_class_acc"]:.4f}')
    print(f'  Top-5 Accuracy         : {results["act_top5_accuracy"]:.4f}')

    print('\nHEAD POSE (9-DoF)')
    print('-' * 40)
    print(f'  Overall MAE            : {results["head_pose_MAE"]:.4f}')
    print(f'  MAE Std                : {results["head_pose_MAE_std"]:.4f}')
    print(f'  forward_x MAE          : {results["forward_x_MAE"]:.4f}')
    print(f'  forward_y MAE          : {results["forward_y_MAE"]:.4f}')
    print(f'  forward_z MAE          : {results["forward_z_MAE"]:.4f}')
    print(f'  pos_x MAE              : {results["pos_x_MAE"]:.4f}')
    print(f'  pos_y MAE              : {results["pos_y_MAE"]:.4f}')
    print(f'  pos_z MAE              : {results["pos_z_MAE"]:.4f}')
    print(f'  up_x MAE               : {results["up_x_MAE"]:.4f}')
    print(f'  up_y MAE               : {results["up_y_MAE"]:.4f}')
    print(f'  up_z MAE               : {results["up_z_MAE"]:.4f}')
    print(f'  N samples              : {results.get("n_samples", "N/A")}')

    print('\nPROCEDURE STEP RECOGNITION (PSR)')
    print('-' * 40)
    print(f'  Overall F1             : {results["psr_overall_f1"]:.4f}')
    print(f'  F1@T (±3 frames)       : {results["psr_f1_at_t"]:.4f}')
    print(f'  Edit Score             : {results["psr_edit_score"]:.4f}')
    print(f'  PSR POS                : {results["psr_pos"]:.4f}')
    print(f'  Valid components       : {results["psr_num_valid_components"]}/11')
    print(f'  N samples              : {results["psr_num_samples"]}')

    psr_per_comp = cast(Dict[str, float], results.get('psr_per_component_f1', {}))
    print('  Per-component F1:')
    for comp_name in sorted(psr_per_comp.keys()):
        val = psr_per_comp[comp_name]
        print(f'    {comp_name:12s}: {val:.4f}')

    print('\nASSEMBLY STATE DETECTION (ASD)')
    print('-' * 40)
    print(f'  mAP@0.5                : {results["det_mAP50"]:.4f}')
    print(f'  mAP@[0.5:0.95]         : {results["det_mAP_50_95"]:.4f}')

    det_per_class = cast(Dict[int, float], results.get('det_per_class_ap', {}))
    if det_per_class:
        print('\n  Per-class AP@0.5:')
        for cls_id, ap in sorted(det_per_class.items()):
            name = C.DET_CLASS_NAMES.get(cls_id + 1, f'class_{cls_id}')
            print(f'    {name:20s}: {ap:.4f}')

    print('\n' + '=' * 60)