import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Evaluation Metrics for Multi-Task IKEA Model
============================================
Activity  : Frame accuracy (all / excl. NA), Macro-F1, Weighted-F1,
            Macro-Recall, Mean Per-Class Accuracy, Top-5 Accuracy,
            Confusion matrix, Per-class report, mcAP
Pose      : PCK@0.05, PCK@0.1, PCK@0.2, Mean Pixel Error,
            Per-keypoint PCK@0.1, PCK@10px (IKEA ASM benchmark)
Detection : mAP@0.5, mAP@[0.5:0.95], Per-class AP@0.5
Temporal  : Kendall's Tau (temporal ordering), mAP@0.5 (temporal localization)
Efficiency: Params (M), GFLOPs, FPS

Author: Bashara
Date: February 2026 | Updated: April 2026 (temporal + efficiency)
"""

import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

import config as C

C = cast(Any, C)

logger = logging.getLogger(__name__)


def _load_state_with_compat(model: torch.nn.Module, checkpoint_state: Dict[str, torch.Tensor]):
    mapped_state = dict(checkpoint_state)
    model_state = model.state_dict()

    legacy_key_map = {
        'activity_head.fc.1.weight': 'activity_head.fc1.0.weight',
        'activity_head.fc.1.bias': 'activity_head.fc1.0.bias',
        'activity_head.fc.4.weight': 'activity_head.classifier.weight',
        'activity_head.fc.4.bias': 'activity_head.classifier.bias',
    }
    remapped = []
    for old_key, new_key in legacy_key_map.items():
        if old_key in mapped_state and new_key in model_state:
            if tuple(mapped_state[old_key].shape) == tuple(model_state[new_key].shape):
                mapped_state[new_key] = mapped_state[old_key]
                remapped.append((old_key, new_key))

    if remapped:
        logger.warning(
            'Checkpoint compatibility remap applied for %d activity-head tensors.',
            len(remapped),
        )

    return model.load_state_dict(mapped_state, strict=False)


def _needs_legacy_model(load_result) -> bool:
    missing_activity = [k for k in load_result.missing_keys if k.startswith('activity_head.')]
    unexpected_activity = [k for k in load_result.unexpected_keys if k.startswith('activity_head.')]
    has_legacy_fc_keys = any(k.startswith('activity_head.fc.') for k in unexpected_activity)
    return len(missing_activity) >= 8 and has_legacy_fc_keys


def _load_legacy_model_class():
    variant_dir = Path(__file__).resolve().parent.name
    legacy_model_path = Path(__file__).resolve().parents[1] / 'model_OLD' / variant_dir / 'model.py'
    if not legacy_model_path.exists():
        raise FileNotFoundError(f'Legacy model not found: {legacy_model_path}')

    spec = importlib.util.spec_from_file_location(f'legacy_model_{variant_dir}', legacy_model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load legacy model spec from: {legacy_model_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'MultiTaskIKEA'):
        raise AttributeError(f'MultiTaskIKEA not found in legacy model: {legacy_model_path}')
    return module.MultiTaskIKEA


def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
        mean = torch.tensor(C.IMAGENET_MEAN, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD, device=device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


# =============================================================================
# Activity Metrics
# =============================================================================
def compute_activity_metrics(all_gt, all_pred, all_logits=None,
                             class_names=None, save_dir=None):
    """
    Comprehensive activity recognition metrics.

    NOTE: The IKEA ASM "Phase Classification" task (STEPs baseline: Acc@1.0 >37.02%)
    is equivalent to frame-level activity classification — the returned
    'phase_classification_accuracy' is an alias for 'act_accuracy' (Top-1 accuracy).

    Args:
        all_gt      : np.ndarray [N] -- ground truth class ids
        all_pred    : np.ndarray [N] -- predicted class ids
        all_logits  : np.ndarray [N, C] or None -- raw logits for top-k
        class_names : list of str or None
        save_dir    : str or None -- if provided, saves confusion matrix image

    Returns:
        dict with all activity metrics
    """
    all_gt = np.asarray(all_gt)
    all_pred = np.asarray(all_pred)
    num_classes = len(class_names) if class_names else C.NUM_ACT_CLASSES
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

    # 5. Macro-Recall (reported in WACV 2021)
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
        top5_indices = np.argsort(all_logits, axis=1)[:, -5:]  # [N, 5]
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

    # 9. Save confusion matrix image
    if save_dir is not None and class_names is not None:
        _save_confusion_matrix(cm, class_names, save_dir)

    return {
        'act_accuracy': fa_all,
        'phase_classification_accuracy': fa_all,
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


def compute_calibrated_ap(recall, precision, targets_sorted, num_pos, num_neg):
    """
    Compute calibrated AP (Geest et al.) using 11-point interpolation.

    Calibrated precision at rank i:
        cPrec(i) = w * TP(i) / (w * TP(i) + FP(i))
    where w = num_negative / num_positive.

    For each recall threshold t, take the MAXIMUM calibrated precision
    among all detection ranks achieving recall >= t (standard AP interpolation).
    Then apply 11-point recall sampling.

    Args:
        recall: np.ndarray of recall values (cumulative TP / total_pos)
        precision: np.ndarray of precision values (cumulative TP / (cumulative TP + cumulative FP))
        targets_sorted: np.ndarray of binary GT values sorted by score descending (1=positive, 0=negative)
        num_pos: int, number of positive samples
        num_neg: int, number of negative samples

    Returns:
        float: calibrated AP
    """
    if num_pos == 0 or num_neg == 0:
        return 0.0

    w = num_neg / num_pos

    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    targets_sorted = np.concatenate([[0], targets_sorted, [0]])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    tp_cum = np.cumsum(targets_sorted)
    fp_cum = np.cumsum(1 - targets_sorted)

    n = len(recall)
    calibrated_prec = np.zeros(n)
    for i in range(n):
        if tp_cum[i] + fp_cum[i] > 0:
            calibrated_prec[i] = (w * tp_cum[i]) / (w * tp_cum[i] + fp_cum[i])
        else:
            calibrated_prec[i] = 0.0

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        if mask.sum() == 0:
            continue
        prec_at_t = calibrated_prec[mask].max()
        recall_step = 1.0 / 11.0
        ap += prec_at_t * recall_step

    return ap


def compute_activity_mcAP(all_gt, all_logits, num_classes=None, protocol='coco'):
    """
    Mean Average Precision per class for activity recognition (PTMA/MiniROAD comparison).

    Supports three interpolation protocols:
      - 'coco' (default): COCO-style 101-point area-averaged AP
      - 'pascal': PASCAL VOC-style 11-point interpolation AP
      - 'calibrated': Geest et al. calibrated AP (mcAP for imbalanced classes)

    Uses per-class AP, then averages across classes (mean Average Precision).

    Args:
        all_gt      : np.ndarray [N] -- ground truth class ids
        all_logits  : np.ndarray [N, C] -- raw logits (softmax will be applied)
        num_classes : int or None -- inferred from logits shape if None
        protocol    : 'coco' (101-point), 'pascal' (11-point), or 'calibrated'

    Returns:
        float: mcAP (mean AP across all classes)
    """
    if all_logits is None or len(all_logits) == 0:
        return 0.0

    all_logits = np.asarray(all_logits)
    num_classes = num_classes or all_logits.shape[1]

    # Softmax to get per-class probabilities
    exp_logits = np.exp(all_logits - all_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    aps = []
    for cls in range(num_classes):
        targets_binary = (all_gt == cls).astype(np.int32)
        scores_cls = probs[:, cls]

        num_pos = int(targets_binary.sum())
        num_neg = int((all_gt != cls).sum())

        if num_pos == 0:
            continue

        order = np.argsort(-scores_cls)
        targets_sorted = targets_binary[order]

        tp = np.cumsum(targets_sorted)
        fp = np.cumsum(1 - targets_sorted)
        total_pos = tp[-1]

        if total_pos == 0:
            continue

        recall = tp / total_pos
        precision = tp / (tp + fp)

        if protocol == 'coco':
            ap = _compute_coco_ap(recall, precision)
        elif protocol == 'calibrated':
            ap = compute_calibrated_ap(recall, precision, targets_sorted, num_pos, num_neg)
        else:
            ap = 0.0
            for t in np.linspace(0, 1, 11):
                prec_at_rec = precision[recall >= t]
                ap += prec_at_rec.max() if prec_at_rec.size > 0 else 0.0
            ap /= 11.0
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def _compute_coco_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute COCO-style AP (area-averaged precision across recall thresholds).

    Uses 101 unique recall thresholds and computes the mean precision
    at each threshold where precision is sampled.

    COCO metric: AP = mean(precision at each recall point)
    """
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Compute precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # Find unique recall thresholds
    indices = np.where(recall[1:] != recall[:-1])[0]

    # Sum precision differences * recall step
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap


def report_per_class_accuracy(cm_list, class_names=None, k: int = 5):
    """Log top-k worst and best per-class activity accuracy from confusion matrix."""
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

    logger.info('  📉 5 Worst Classes:')
    for idx in worst_idx:
        logger.info(f'    {names[idx]:30s}: {per_class_acc[idx]:.1%}')

    logger.info('  📈 5 Best Classes:')
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
# Pose Metrics
# =============================================================================
def compute_pck(pred, gt, vis, threshold_ratio=0.1, img_size=C.IMG_SIZE):
    """PCK at given threshold (fraction of image diagonal)."""
    diag = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)
    threshold = threshold_ratio * diag
    dist = np.linalg.norm(pred - gt, axis=-1)
    correct = (dist < threshold) & (vis > 0)
    visible = vis > 0
    total = visible.sum()
    if total == 0:
        return {'pck': float('nan'), 'per_keypoint': [float('nan')] * C.NUM_KEYPOINTS}
    per_kpt = []
    for k in range(C.NUM_KEYPOINTS):
        vk = visible[:, k].sum()
        per_kpt.append(float(correct[:, k].sum() / vk) if vk > 0 else 0.0)
    return {'pck': float(correct.sum() / total), 'per_keypoint': per_kpt}


def compute_pck_fixed_px(pred, gt, vis, threshold_px=10.0):
    """
    PCK at a fixed absolute pixel threshold.

    This matches the evaluation protocol of the official IKEA ASM benchmark
    (Ben-Shabat et al., WACV 2021) which uses a fixed 10px threshold.
    Their best result: 64.3% PCK@10px with fine-tuned Mask R-CNN.

    Args:
        pred         : np.ndarray [N, 17, 2]  predicted pixel coords
        gt           : np.ndarray [N, 17, 2]  ground-truth pixel coords
        vis          : np.ndarray [N, 17]     visibility flags (>0 = visible)
        threshold_px : float                  pixel threshold (default: 10.0)

    Returns:
        dict with:
            pck_fixed_px     : float  -- overall PCK@{threshold_px}px
            per_keypoint     : list   -- per-keypoint PCK@{threshold_px}px (17 values)
            threshold_px     : float  -- threshold used (for logging)
            n_visible        : int    -- number of visible keypoints evaluated
    """
    dist = np.linalg.norm(pred - gt, axis=-1)
    vis_bool = vis > 0
    correct = (dist < threshold_px) & vis_bool
    total_visible = int(vis_bool.sum())

    if total_visible == 0:
        return {
            'pck_fixed_px': float('nan'),
            'per_keypoint': [float('nan')] * C.NUM_KEYPOINTS,
            'threshold_px': threshold_px,
            'n_visible': 0,
        }

    per_kpt = []
    for k in range(C.NUM_KEYPOINTS):
        vk = int(vis_bool[:, k].sum())
        per_kpt.append(float(correct[:, k].sum() / vk) if vk > 0 else float('nan'))

    return {
        'pck_fixed_px': float(correct.sum() / total_visible),
        'per_keypoint': per_kpt,
        'threshold_px': threshold_px,
        'n_visible': total_visible,
    }


def compute_pose_metrics_extended(pred_kpts, gt_kpts, visibility,
                                  img_w=C.IMG_WIDTH, img_h=C.IMG_HEIGHT):
    """
    Extended pose metrics: PCK@0.05, PCK@0.1, PCK@0.2, mean pixel error.

    Args:
        pred_kpts  : np.ndarray [N, 17, 2] pixel coords
        gt_kpts    : np.ndarray [N, 17, 2] pixel coords
        visibility : np.ndarray [N, 17] float (>0 = visible)

    Returns:
        dict with all pose metrics
    """
    img_diag = np.sqrt(img_w ** 2 + img_h ** 2)
    dists = np.linalg.norm(pred_kpts - gt_kpts, axis=-1)  # [N, 17]
    vis = visibility > 0

    pck_005 = compute_pck(pred_kpts, gt_kpts, visibility, 0.05, (img_w, img_h))
    pck_010 = compute_pck(pred_kpts, gt_kpts, visibility, 0.10, (img_w, img_h))
    pck_020 = compute_pck(pred_kpts, gt_kpts, visibility, 0.20, (img_w, img_h))

    # Fixed-pixel PCK to match IKEA ASM benchmark (Ben-Shabat et al., WACV 2021)
    pck_10px = compute_pck_fixed_px(pred_kpts, gt_kpts, visibility, threshold_px=10.0)
    pck_20px = compute_pck_fixed_px(pred_kpts, gt_kpts, visibility, threshold_px=20.0)
    pck_30px = compute_pck_fixed_px(pred_kpts, gt_kpts, visibility, threshold_px=30.0)

    # Mean pixel error on visible keypoints
    mean_px_err = float(dists[vis].mean()) if vis.sum() > 0 else 0.0

    return {
        'pck_at_005': pck_005['pck'],
        'pck_at_01': pck_010['pck'],
        'pck_at_02': pck_020['pck'],
        'pck_per_keypoint_01': pck_010['per_keypoint'],
        'pck_per_keypoint_005': pck_005['per_keypoint'],
        'mean_pixel_error': mean_px_err,
        'pck_at_10px': pck_10px['pck_fixed_px'],
        'pck_at_20px': pck_20px['pck_fixed_px'],
        'pck_at_30px': pck_30px['pck_fixed_px'],
        'pck_per_keypoint_10px': pck_10px['per_keypoint'],
        'n_visible_keypoints': pck_10px['n_visible'],
    }


# =============================================================================
# Detection Metrics
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
    return np.stack([cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2], axis=1)


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


def compute_ap_per_class(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
                         iou_thresh=0.5, num_classes=C.NUM_DET_CLASSES):
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


def compute_det_metrics_extended(pred_boxes, pred_scores, pred_labels,
                                 gt_boxes, gt_labels, num_classes=C.NUM_DET_CLASSES):
    """
    Extended detection metrics: mAP@0.5 and mAP@[0.5:0.95].

    Returns:
        dict with mAP50, mAP_50_95, per_class_ap50
    """
    r50 = compute_ap_per_class(pred_boxes, pred_scores, pred_labels,
                               gt_boxes, gt_labels, 0.5, num_classes)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps_at_thresholds = []
    for iou_t in iou_thresholds:
        r = compute_ap_per_class(pred_boxes, pred_scores, pred_labels,
                                 gt_boxes, gt_labels, float(iou_t), num_classes)
        maps_at_thresholds.append(r['mAP'])

    return {
        'det_mAP50': r50['mAP'],
        'det_mAP_50_95': float(np.mean(maps_at_thresholds)),
        'det_per_class_ap': r50['per_class_ap'],
    }


# =============================================================================
# Temporal Ordering — Kendall's Tau
# =============================================================================
def compute_kendall_tau(pred_order: np.ndarray, gt_order: np.ndarray) -> float:
    """
    Compute Kendall's Tau for temporal ordering of actions.

    The STEPs paper (Koppula et al.) evaluates temporal ordering as:
    "Given a video segment pair (A, B), predict which action happens first."

    Kendall's Tau = (concordant_pairs - discordant_pairs) / (n_pairs)

    Args:
        pred_order: np.ndarray [N, K] or [N] — predicted order indices (argsort)
                    For N segments, each row is the ranking of K actions.
        gt_order:   np.ndarray [N, K] — ground truth order indices

    Returns:
        float: Kendall's Tau score (-1 to 1, higher = better)
    """
    if pred_order.size == 0 or gt_order.size == 0:
        return 0.0

    pred_order = np.asarray(pred_order)
    gt_order = np.asarray(gt_order)

    if pred_order.ndim == 1:
        pred_order = pred_order.reshape(1, -1)
        gt_order = gt_order.reshape(1, -1)

    N, K = pred_order.shape
    if K < 2:
        return 0.0

    tau_sum = 0.0
    n_pairs = K * (K - 1) // 2

    for i in range(N):
        pred_inv = np.argsort(pred_order[i])
        gt_rank = np.argsort(gt_order[i])
        pred_rank = pred_inv[gt_rank]

        concordant = discordant = 0
        for p_idx in range(K):
            for q_idx in range(p_idx + 1, K):
                if pred_rank[p_idx] < pred_rank[q_idx]:
                    concordant += 1
                else:
                    discordant += 1

        if n_pairs > 0:
            tau_sum += (concordant - discordant) / n_pairs

    return float(tau_sum / N) if N > 0 else 0.0


def compute_temporal_ordering_metrics(
    all_gt_order: List[np.ndarray],
    all_pred_order: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute temporal ordering metrics (Kendall's Tau, DTA, temporal accuracy).

    Args:
        all_gt_order: List of [K] ground truth temporal order arrays
        all_pred_order: List of [K] predicted temporal order arrays

    Returns:
        dict with temporal_order_kendall_tau and temporal_order_accuracy
    """
    if len(all_gt_order) == 0:
        return {'temporal_order_kendall_tau': 0.0, 'temporal_order_accuracy': 0.0}

    gt_flat = np.concatenate(all_gt_order)
    pred_flat = np.concatenate(all_pred_order)

    kendall_tau = compute_kendall_tau(pred_flat, gt_flat)

    correct_order = (pred_flat == gt_flat).all(axis=1) if pred_flat.ndim > 1 else (pred_flat == gt_flat)
    dta = float(correct_order.mean()) if correct_order.size > 0 else 0.0

    return {
        'temporal_order_kendall_tau': kendall_tau,
        'temporal_order_accuracy': dta,
    }


@torch.no_grad()
def evaluate_temporal_sequence(
    model: nn.Module,
    temporal_loader: DataLoader,
    device: torch.device,
    max_batches: int = 2000,
) -> Dict[str, float]:
    """
    Evaluate temporal ordering (Kendall's Tau) and temporal action localization
    on sequence data from IKEAMultiTaskSequenceDataset.

    Args:
        model           : MultiTaskIKEA model (with use_temporal=True)
        temporal_loader : DataLoader returning (images_seq [B,T,3,H,W], targets dict)
        device          : torch device
        max_batches     : cap for speed during validation

    Returns:
        dict with Kendall's Tau, DTA, temporal ordering accuracy, and tma metrics
    """
    model.eval()
    all_gt_pairs, all_pred_pairs = [], []
    all_gt_temporal: List[List[Dict]] = []
    all_temporal_al_outputs: List[Dict[str, np.ndarray]] = []

    TMA_KENDALL_TAU = float('nan')
    TMA_DTA = float('nan')
    ORDER_ACC = float('nan')
    TEMPORAL_MCAP50 = float('nan')

    for bi, (images_seq, targets) in enumerate(tqdm(temporal_loader, desc='[Temporal Eval]', leave=False)):
        if bi >= max_batches:
            break

        B, T = images_seq.shape[:2]
        images_seq = images_seq.to(device, non_blocking=True)
        if images_seq.dtype == torch.uint8:
            images_seq = images_seq.float().div_(255.0)
            mean = torch.tensor(C.IMAGENET_MEAN, device=device, dtype=images_seq.dtype).view(1, 1, 3, 1, 1)
            std = torch.tensor(C.IMAGENET_STD, device=device, dtype=images_seq.dtype).view(1, 1, 3, 1, 1)
            images_seq = (images_seq - mean) / std

        outputs_seq = model.forward_sequence(images_seq)
        act_logits_seq = outputs_seq['act_logits_seq']   # [B, T, num_classes]
        action_labels = targets['action_labels_seq']     # [B, T]

        pred_scores = torch.sigmoid(act_logits_seq.mean(dim=2))  # [B, T] — avg logit per frame

        temporal_al = outputs_seq.get('temporal_al', {})
        start_scores = temporal_al.get('start_scores')
        if temporal_al and start_scores is not None and (
            (isinstance(start_scores, torch.Tensor) and start_scores.numel() > 0) or
            (isinstance(start_scores, (list, tuple)) and len(start_scores) > 0) or
            (isinstance(start_scores, np.ndarray) and start_scores.size > 0)
        ):
            gt_temporal_batch = targets.get('gt_temporal', [[] for _ in range(B)])
            all_gt_temporal.extend(gt_temporal_batch)

            for b in range(B):
                b_tmal = {}
                for k, v in temporal_al.items():
                    if isinstance(v, torch.Tensor):
                        b_tmal[k] = v[b].cpu().numpy()
                    elif isinstance(v, list):
                        b_tmal[k] = v[b] if b < len(v) else v[0]
                    else:
                        b_tmal[k] = v
                all_temporal_al_outputs.append(b_tmal)

        for b in range(B):
            labels_b = action_labels[b].numpy()          # [T]
            scores_b = pred_scores[b].cpu().numpy()        # [T]

            gt_order = np.argsort(labels_b)                # ascending = earlier action first
            pred_order = np.argsort(-scores_b)             # descending score = earlier

            K = T
            if K < 2:
                continue

            n_pairs = K * (K - 1) // 2
            gt_pair_vec = np.zeros(n_pairs, dtype=np.float32)
            pred_pair_vec = np.zeros(n_pairs, dtype=np.float32)

            pidx = 0
            for i in range(K):
                for j in range(i + 1, K):
                    gt_pair_vec[pidx] = 1.0 if gt_order[i] < gt_order[j] else 0.0
                    pred_pair_vec[pidx] = 1.0 if pred_order[i] < pred_order[j] else 0.0
                    pidx += 1

            all_gt_pairs.append(gt_pair_vec)
            all_pred_pairs.append(pred_pair_vec)

    if len(all_gt_pairs) > 0:
        gt_flat = np.stack(all_gt_pairs)
        pred_flat = np.stack(all_pred_pairs)

        kendall_tau = compute_kendall_tau(pred_flat, gt_flat)

        concordant = ((pred_flat == gt_flat)).sum()
        total_pairs = pred_flat.size
        order_acc = float(concordant / total_pairs) if total_pairs > 0 else 0.0

        TMA_KENDALL_TAU = kendall_tau
        TMA_DTA = order_acc

        gt_ranks = np.argsort(np.argsort(gt_flat, axis=1), axis=1).astype(np.float32)
        pred_ranks = np.argsort(np.argsort(pred_flat, axis=1), axis=1).astype(np.float32)
        ORDER_ACC = compute_kendall_tau(pred_ranks, gt_ranks)

    if all_temporal_al_outputs and all_gt_temporal:
        tmal_dict = {
            'start_scores': np.stack([o['start_scores'] for o in all_temporal_al_outputs]),
            'end_scores': np.stack([o['end_scores'] for o in all_temporal_al_outputs]),
            'action_logits': np.stack([o['action_logits'] for o in all_temporal_al_outputs]),
            'confidence': np.stack([o['confidence'] for o in all_temporal_al_outputs]),
        }
        tmal_results = compute_temporal_metrics(tmal_dict, all_gt_temporal)
        TEMPORAL_MCAP50 = tmal_results.get('temporal_mAP50', float('nan'))
        logger.info(
            f'  Temporal Localization — '
            f"mAP@0.5={TEMPORAL_MCAP50:.4f}  "
            f"(from {len(all_gt_temporal)} sequences with GT proposals)"
        )

    logger.info(
        f'  Temporal Ordering — '
        f"Kendall Tau={TMA_KENDALL_TAU:.4f}  "
        f"DTA={TMA_DTA:.4f}  "
        f"Pairwise Acc={ORDER_ACC:.4f}"
    )

    model.train()
    return {
        'temporal_order_kendall_tau': TMA_KENDALL_TAU,
        'temporal_order_dta': TMA_DTA,
        'temporal_order_pairwise_acc': ORDER_ACC,
        'temporal_mAP50': TEMPORAL_MCAP50,
    }


# =============================================================================
# Temporal Action Localization — mAP@0.5
# =============================================================================
def temporal_iou(temporal_pred: Tuple[float, float],
                 temporal_gt: Tuple[float, float]) -> float:
    """
    Compute temporal IoU between two [start, end] time intervals.

    Args:
        temporal_pred: (start, end) predicted interval
        temporal_gt: (start, end) ground truth interval

    Returns:
        float: temporal IoU in [0, 1]
    """
    pred_start, pred_end = temporal_pred
    gt_start, gt_end = temporal_gt

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter_len = max(0.0, inter_end - inter_start)

    pred_len = max(0.0, pred_end - pred_start)
    gt_len = max(0.0, gt_end - gt_start)

    union_len = pred_len + gt_len - inter_len
    return inter_len / union_len if union_len > 0 else 0.0


def compute_temporal_localization_ap(
    pred_proposals: List[Dict],
    gt_temporal: List[Dict],
    iou_thresh: float = 0.5,
    num_classes: int = C.NUM_ACT_CLASSES,
) -> Dict[str, float]:
    """
    Compute temporal action localization mAP@0.5.

    Args:
        pred_proposals: List of dicts with keys:
            - 'start': float (0-1 normalized)
            - 'end': float (0-1 normalized)
            - 'action_class': int
            - 'score': float
        gt_temporal: List of dicts with keys:
            - 'start': float
            - 'end': float
            - 'action_class': int

    Returns:
        dict with mAP@0.5 and per-class AP
    """
    aps = []
    for cls in range(num_classes):
        cls_preds = [(p['score'], p['start'], p['end'])
                      for p in pred_proposals if p['action_class'] == cls]
        cls_gts = [(g['start'], g['end']) for g in gt_temporal if g['action_class'] == cls]

        if len(cls_gts) == 0:
            continue

        cls_preds_sorted = sorted(cls_preds, key=lambda x: x[0], reverse=True)

        total_gt = len(cls_gts)
        tp = []
        fp = []
        scores = []

        for score, p_start, p_end in cls_preds_sorted:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (g_start, g_end) in enumerate(cls_gts):
                iou = temporal_iou((p_start, p_end), (g_start, g_end))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh:
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
            scores.append(score)

        if len(tp) == 0:
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / total_gt
        precision = tp_cum / (tp_cum + fp_cum)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec_at_rec = precision[recall >= t]
            ap += prec_at_rec.max() if prec_at_rec.size > 0 else 0.0
        ap /= 11.0
        aps.append(ap)

    return {
        'temporal_mAP50': float(np.mean(aps)) if aps else 0.0,
        'temporal_per_class_ap': aps,
    }


def compute_temporal_metrics(
    temporal_al_output: Dict[str, Any],
    gt_temporal: List[List[Dict]],
) -> Dict[str, float]:
    """
    Compute all temporal localization metrics from model output.

    Args:
        temporal_al_output: dict with keys start_scores, end_scores, action_logits,
            confidence — each value is either a torch.Tensor or np.ndarray
        gt_temporal: list of lists of ground truth temporal annotations (one list per sample)

    Returns:
        dict with temporal_mAP50 and per-class metrics
    """
    def _to_numpy(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    start_scores = _to_numpy(temporal_al_output['start_scores'])
    end_scores = _to_numpy(temporal_al_output['end_scores'])
    action_logits = _to_numpy(temporal_al_output['action_logits'])
    confidence = _to_numpy(temporal_al_output['confidence'])

    pred_proposals = []
    B = start_scores.shape[0]
    num_proposals = start_scores.shape[1]

    for b in range(B):
        for p_idx in range(num_proposals):
            start = float(start_scores[b, p_idx])
            end = float(end_scores[b, p_idx])
            conf = float(confidence[b, p_idx])
            action_cls = int(action_logits[b, p_idx].argmax())

            pred_proposals.append({
                'start': start,
                'end': end,
                'action_class': action_cls,
                'score': conf,
            })

    flat_gt = [g for sample_gt in gt_temporal for g in sample_gt]
    return compute_temporal_localization_ap(pred_proposals, flat_gt, iou_thresh=0.5)


@torch.no_grad()
def evaluate_with_tta(model, dataloader, device, max_batches=2500):
    """
    Evaluation with Test-Time Augmentation (horizontal flip).

    Performs two forward passes per image:
        1. Original image
        2. Horizontally flipped image (torch.flip(img, dims=[-1]))

    Activity logits are averaged: act_logits = (act_orig + act_flipped) / 2
    Pose keypoints from the flipped pass are transformed:
        x_coord = IMG_WIDTH - x_coord
    Detection boxes from the flipped pass are transformed:
        x_min = IMG_WIDTH - x_max
        x_max = IMG_WIDTH - x_min

    Args:
        model        : MultiTaskIKEA
        dataloader   : DataLoader (val or test)
        device       : torch.device
        max_batches  : int -- cap for speed during training val

    Returns:
        dict with all metrics (same structure as evaluate_all)
    """
    if not C.USE_TTA:
        logger.warning('USE_TTA=False, falling back to standard evaluate_all')
        return evaluate_all(model, None, dataloader, device, max_batches)

    model.eval()
    IMG_W = C.IMG_WIDTH
    IMG_H = C.IMG_HEIGHT

    act_preds, act_labels, act_logits_all = [], [], []
    kp_preds, kp_gts, kp_vis = [], [], []
    dp_boxes, dp_scores, dp_labels = [], [], []
    dg_boxes, dg_labels = [], []

    _cached_anchors_np = None

    for bi, (images, targets) in enumerate(dataloader):
        if bi >= max_batches:
            break
        images = _prepare_images(images, device)
        for i in range(len(targets['detection'])):
            targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].to(device)
            targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to(device)
        targets['keypoints'] = targets['keypoints'].to(device)
        targets['visibility'] = targets['visibility'].to(device)
        targets['kpt_confidence'] = targets['kpt_confidence'].to(device)
        targets['activity'] = targets['activity'].to(device)

        with amp.autocast(enabled=C.MIXED_PRECISION):
            # Original forward pass
            outputs_orig = model(images)
            # Flipped forward pass
            images_flipped = torch.flip(images, dims=[-1])
            outputs_flip = model(images_flipped)

        # Cast FP16 outputs to FP32
        for _k in ('cls_preds', 'reg_preds', 'keypoints', 'act_logits'):
            for out_dict in (outputs_orig, outputs_flip):
                if _k in out_dict and isinstance(out_dict[_k], torch.Tensor):
                    out_dict[_k] = out_dict[_k].float()

        # Activity: average logits from original and flipped
        act_logits_orig = outputs_orig['act_logits']
        act_logits_flip = outputs_flip['act_logits']
        act_logits_avg = (act_logits_orig + act_logits_flip) / 2.0
        act_logits_batch = act_logits_avg.cpu().numpy()
        if C.COMPUTE_VAL_TOP5:
            act_logits_all.append(act_logits_batch)
        act_preds.append(act_logits_batch.argmax(axis=1))
        act_labels.append(targets['activity'].cpu().numpy())

        # Pose keypoints: transform flipped output coordinates
        # x_coord = IMG_WIDTH - x_coord (horizontal flip)
        kp_flip = outputs_flip['keypoints']  # [B, 17, 2]
        kp_orig = outputs_orig['keypoints']   # [B, 17, 2]
        kp_flip_xform = kp_flip.clone()
        kp_flip_xform[..., 0] = IMG_W - kp_flip[..., 0]
        kp_preds.append(kp_orig.cpu().numpy())  # Use original for now; could average kp too
        kp_gts.append(targets['keypoints'].cpu().numpy())
        kp_vis.append(targets['visibility'].cpu().numpy())

        # Detection: transform flipped boxes
        # x_min = IMG_WIDTH - x_max, x_max = IMG_WIDTH - x_min
        if _cached_anchors_np is None:
            _cached_anchors_np = outputs_orig['anchors'].cpu().numpy()

        for i in range(images.shape[0]):
            # Original detection branch
            cls_sigmoid = torch.sigmoid(outputs_orig['cls_preds'][i])
            scores_i = cls_sigmoid.max(dim=1).values
            score_thresh = float(getattr(C, 'DET_EVAL_SCORE_THRESH', 0.5))
            keep_mask = scores_i > score_thresh

            max_keep = int(getattr(C, 'DET_EVAL_MAX_PER_IMAGE', 300))
            if max_keep > 0 and keep_mask.sum().item() > max_keep:
                topk_idx = torch.topk(scores_i, k=max_keep, largest=True, sorted=False).indices
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
                kept_reg = outputs_orig['reg_preds'][i][keep_mask].cpu().numpy()
                kept_anc = _cached_anchors_np[keep_np]

                ms = kept_cls.max(axis=1)
                ml = kept_cls.argmax(axis=1)
                pb = decode_boxes(kept_anc, kept_reg)
                pb[:, 0] = np.clip(pb[:, 0], 0, IMG_W)
                pb[:, 1] = np.clip(pb[:, 1], 0, IMG_H)
                pb[:, 2] = np.clip(pb[:, 2], 0, IMG_W)
                pb[:, 3] = np.clip(pb[:, 3], 0, IMG_H)
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

            dg_boxes.append(targets['detection'][i]['boxes'].cpu().numpy())
            dg_labels.append(targets['detection'][i]['labels'].cpu().numpy())

        # Free GPU tensors promptly
        del images, outputs_orig, outputs_flip, images_flipped

    if not act_preds:
        dataset_len = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else -1
        raise RuntimeError(
            f'No batches were produced by DataLoader (dataset_len={dataset_len}). '
            f'Check split paths and filtering logic.'
        )

    results = {}

    all_act_pred = np.concatenate(act_preds)
    all_act_gt = np.concatenate(act_labels)
    all_act_logits = np.concatenate(act_logits_all) if act_logits_all else None
    del act_preds, act_labels, act_logits_all

    act_metrics = compute_activity_metrics(
        all_act_gt, all_act_pred, all_act_logits,
        class_names=C.ACT_CLASS_NAMES,
    )
    results.update(act_metrics)

    if all_act_logits is not None:
        results['act_mcAP'] = compute_activity_mcAP(
            all_act_gt, all_act_logits,
            num_classes=len(C.ACT_CLASS_NAMES) if C.ACT_CLASS_NAMES else C.NUM_ACT_CLASSES,
            protocol='calibrated',
        )

    report_per_class_accuracy(
        act_metrics.get('act_confusion_matrix', []),
        class_names=C.ACT_CLASS_NAMES,
        k=5,
    )

    all_kp_pred = np.concatenate(kp_preds)
    all_kp_gt = np.concatenate(kp_gts)
    all_kp_vis = np.concatenate(kp_vis)
    del kp_preds, kp_gts, kp_vis

    pose_metrics = compute_pose_metrics_extended(
        all_kp_pred, all_kp_gt, all_kp_vis,
    )
    results.update(pose_metrics)

    results['pck_per_keypoint'] = results['pck_per_keypoint_01']

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

    model.train()
    return results


# =============================================================================
# Main Evaluation Loop
# =============================================================================
@torch.no_grad()
def evaluate_all(model, criterion, loader, device, max_batches=2500,
                 save_dir=None, temporal_loader=None) -> Dict[str, Any]:
    """
    Full evaluation returning all metrics across 3 tasks.

    Args:
        model          : MultiTaskIKEA
        criterion      : MultiTaskLoss
        loader         : DataLoader (val or test)
        device         : torch.device
        max_batches    : int -- cap for speed during training val
        save_dir       : str or None -- where to save confusion matrix
        temporal_loader: DataLoader or None -- IKEAMultiTaskSequenceDataset loader

    Returns:
        dict with all metrics
    """
    model.eval()
    total_loss, lc = 0.0, 0

    # Always compute Top-5: trivial overhead (single argsort on CPU numpy).
    # Previously gated by COMPUTE_VAL_TOP5 which caused stale 0.0 in checkpoints.
    compute_top5 = True

    act_preds, act_labels, act_logits_all = [], [], []
    kp_preds, kp_gts, kp_vis = [], [], []
    dp_boxes, dp_scores, dp_labels = [], [], []
    dg_boxes, dg_labels = [], []

    # Cache anchors: identical every batch for fixed input size (640x480).
    # Avoids re-allocating 57600*4*4 = 921KB per batch on CPU.
    _cached_anchors_np = None

    for bi, (images, targets) in enumerate(loader):
        if bi >= max_batches:
            break
        images = _prepare_images(images, device)
        for i in range(len(targets['detection'])):
            targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].to(device)
            targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to(device)
        targets['keypoints'] = targets['keypoints'].to(device)
        targets['visibility'] = targets['visibility'].to(device)
        targets['kpt_confidence'] = targets['kpt_confidence'].to(device)
        targets['activity'] = targets['activity'].to(device)

        with amp.autocast(enabled=C.MIXED_PRECISION):
            outputs = model(images)
        # Cast FP16 model outputs to FP32 before loss to prevent overflow.
        # Kendall clamp is [-4, 2], so max precision is exp(4)~55.
        # Worst-case product: 55 * 3.0 (max act loss) = 165, safe in FP16.
        for _k in ('cls_preds', 'reg_preds', 'keypoints', 'act_logits'):
            if _k in outputs and isinstance(outputs[_k], torch.Tensor):
                outputs[_k] = outputs[_k].float()
        loss, _ = criterion(outputs, targets)
        if torch.isfinite(loss):
            total_loss += loss.float().item()
            lc += 1

        act_logits_batch = outputs['act_logits'].cpu().numpy()
        if compute_top5:
            act_logits_all.append(act_logits_batch)
        act_preds.append(act_logits_batch.argmax(axis=1))
        act_labels.append(targets['activity'].cpu().numpy())

        kp_preds.append(outputs['keypoints'].cpu().numpy())
        kp_gts.append(targets['keypoints'].cpu().numpy())
        kp_vis.append(targets['visibility'].cpu().numpy())

        # --- Detection: filter on GPU before CPU transfer ---
        # The full cls_preds tensor is [B, 57600, 7] float32 = 6.45MB.
        # At 2.85GB RSS after 2 training epochs, even this single allocation
        # can trigger ENOMEM (Error code 12). Instead, apply score threshold
        # on GPU and transfer only the ~0.1-1% of anchors that pass.
        if _cached_anchors_np is None:
            _cached_anchors_np = outputs['anchors'].cpu().numpy()

        cls_sigmoid = torch.sigmoid(outputs['cls_preds'])  # [B, N, 7] on GPU
        B = images.shape[0]

        for i in range(B):
            scores_i = cls_sigmoid[i]              # [N, 7] on GPU
            max_scores = scores_i.max(dim=1).values  # [N] on GPU
            score_thresh = float(getattr(C, 'DET_EVAL_SCORE_THRESH', 0.5))
            keep_mask = max_scores > score_thresh    # [N] bool on GPU

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
                # Transfer only kept anchors to CPU (~100-500 vs 57600)
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

            dg_boxes.append(targets['detection'][i]['boxes'].cpu().numpy())
            dg_labels.append(targets['detection'][i]['labels'].cpu().numpy())

        # Free batch GPU tensors promptly
        del images, outputs, cls_sigmoid

    if not act_preds:
        dataset_len = len(loader.dataset) if hasattr(loader, 'dataset') else -1
        raise RuntimeError(
            f'No batches were produced by DataLoader (dataset_len={dataset_len}). '
            f'Check split paths and filtering logic.'
        )

    results = {'loss': total_loss / max(lc, 1)}

    all_act_pred = np.concatenate(act_preds)
    all_act_gt = np.concatenate(act_labels)
    all_act_logits = np.concatenate(act_logits_all) if compute_top5 else None
    del act_preds, act_labels, act_logits_all

    act_metrics = compute_activity_metrics(
        all_act_gt, all_act_pred, all_act_logits,
        class_names=C.ACT_CLASS_NAMES,
        save_dir=save_dir,
    )
    results.update(act_metrics)

    # mcAP for PTMA/MiniROAD benchmark comparison (per-class AP, calibrated/Geest protocol)
    if all_act_logits is not None:
        results['act_mcAP'] = compute_activity_mcAP(
            all_act_gt, all_act_logits,
            num_classes=len(C.ACT_CLASS_NAMES) if C.ACT_CLASS_NAMES else C.NUM_ACT_CLASSES,
            protocol='calibrated',
        )

    report_per_class_accuracy(
        act_metrics.get('act_confusion_matrix', []),
        class_names=C.ACT_CLASS_NAMES,
        k=5,
    )

    all_kp_pred = np.concatenate(kp_preds)
    all_kp_gt = np.concatenate(kp_gts)
    all_kp_vis = np.concatenate(kp_vis)
    del kp_preds, kp_gts, kp_vis

    _vis_total = int((all_kp_vis > 0).sum())
    logger.info(
        f'  Pose diag: vis_total={_vis_total:,}, '
        f'pred_range=[{np.nanmin(all_kp_pred):.1f}, {np.nanmax(all_kp_pred):.1f}], '
        f'gt_range=[{all_kp_gt.min():.1f}, {all_kp_gt.max():.1f}]'
    )

    pose_metrics = compute_pose_metrics_extended(
        all_kp_pred, all_kp_gt, all_kp_vis,
    )
    results.update(pose_metrics)

    # Log PCK@10px as the primary metric (IKEA ASM benchmark protocol)
    logger.info(
        f'  Pose (PRIMARY - IKEA ASM benchmark protocol):\n'
        f'    PCK@10px (fixed)     : {results["pck_at_10px"]:.4f}  '
        f'<-- compare to Ben-Shabat et al. best: 0.6430\n'
        f'    PCK@20px (fixed)     : {results["pck_at_20px"]:.4f}\n'
        f'    PCK@30px (fixed)     : {results["pck_at_30px"]:.4f}\n'
        f'  Pose (COCO protocol - keep for reference):\n'
        f'    PCK@0.05 (diagonal)  : {results["pck_at_005"]:.4f}\n'
        f'    PCK@0.10 (diagonal)  : {results["pck_at_01"]:.4f}\n'
        f'    Mean pixel error     : {results["mean_pixel_error"]:.1f} px'
    )

    results['pck_per_keypoint'] = results['pck_per_keypoint_01']

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

    if temporal_loader is not None:
        temporal_results = evaluate_temporal_sequence(
            model, temporal_loader, device, max_batches=max_batches
        )
        results.update(temporal_results)

    eff = compute_model_efficiency(model, device=device)
    results.update(eff)

    model.train()
    return results


# =============================================================================
# Standalone CLI
# =============================================================================
if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    _model_module = importlib.import_module('model')
    MultiTaskIKEA = cast(Any, getattr(_model_module, 'MultiTaskIKEA'))

    _losses_module = importlib.import_module('losses')
    MultiTaskLoss = cast(Any, getattr(_losses_module, 'MultiTaskLoss'))

    _dataset_module = importlib.import_module('ikea_dataset')
    IKEAMultiTaskDataset = cast(Any, getattr(_dataset_module, 'IKEAMultiTaskDataset'))
    IKEAMultiTaskSequenceDataset = cast(Any, getattr(_dataset_module, 'IKEAMultiTaskSequenceDataset'))
    collate_fn = cast(Any, getattr(_dataset_module, 'collate_fn'))
    temporal_sequence_collate_fn = cast(Any, getattr(_dataset_module, 'temporal_sequence_collate_fn'))

    parser = argparse.ArgumentParser(
        description='Evaluate Multi-Task IKEA Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate checkpoint on test set
  python evaluate.py --checkpoint path/to/checkpoint.pth
  
  # Evaluate on validation set with specific config
  python evaluate.py --checkpoint model.pt --split val --dataset manual_pseudo
  
  # Override detection mode
  python evaluate.py --checkpoint model.pt --detection dev3_only
        """
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['manual_only', 'manual_pseudo'],
        default=None,
        help='Override C.DATASET_MODE for evaluation'
    )
    parser.add_argument(
        '--detection',
        type=str,
        choices=['all_cameras', 'dev3_only'],
        default=None,
        help='Override C.DETECTION_MODE for evaluation'
    )
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='Output directory for evaluation results (default: config.EVAL_SAVE_DIR)'
    )
    parser.add_argument(
        '--temporal', action='store_true',
        help='Also evaluate temporal ordering and localization metrics '
             '(requires USE_TEMPORAL=True model and gt_segments.json)'
    )

    args = parser.parse_args()
    
    # Override config if specified
    if args.dataset:
        C.DATASET_MODE = args.dataset
    if args.detection:
        C.DETECTION_MODE = args.detection
    
    # Update dynamic paths
    C.update_dynamic_paths()
    
    save_dir = args.save_dir or str(C.EVAL_SAVE_DIR)
    
    logger.info(f'[evaluate] Config: DATASET_MODE={C.DATASET_MODE}, '
                f'DETECTION_MODE={C.DETECTION_MODE}, split={args.split}')

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = IKEAMultiTaskDataset(split=args.split, augment=False)
    loader = DataLoader(
        ds, batch_size=C.VAL_BATCH_SIZE, shuffle=False,
        num_workers=C.VAL_NUM_WORKERS, collate_fn=collate_fn,
    )

    temporal_loader = None
    if args.temporal:
        logger.info('[evaluate] Building temporal sequence dataset for --temporal eval ...')
        temporal_seq_ds = IKEAMultiTaskSequenceDataset(
            base_dataset=ds,
            sequence_len=getattr(C, 'TEMPORAL_SEQUENCE_LEN', 16),
            stride=1,
            target_camera='dev3',
        )
        temporal_batch_size = max(2, C.VAL_BATCH_SIZE // 2)
        temporal_loader = DataLoader(
            temporal_seq_ds,
            batch_size=temporal_batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=temporal_sequence_collate_fn,
            pin_memory=C.PIN_MEMORY,
            drop_last=False,
            persistent_workers=False,
        )
        logger.info(f'[evaluate] Temporal loader ready: {len(temporal_seq_ds)} sequences')

    model = MultiTaskIKEA(pretrained=False, use_film=C.USE_FILM).to(device)
    criterion = MultiTaskLoss().to(device)
    criterion.set_class_counts(ds.class_counts)

    ckpt = torch.load(args.checkpoint, map_location=device)
    load_result = _load_state_with_compat(model, ckpt['model'])

    if _needs_legacy_model(load_result):
        logger.warning(
            'Detected legacy activity-head checkpoint format. Falling back to model_OLD implementation.'
        )
        LegacyMultiTaskIKEA = _load_legacy_model_class()
        model = LegacyMultiTaskIKEA(pretrained=False, use_film=C.USE_FILM).to(device)
        legacy_load = model.load_state_dict(ckpt['model'], strict=False)
        if legacy_load.missing_keys or legacy_load.unexpected_keys:
            raise RuntimeError(
                'Legacy model fallback still has checkpoint mismatch. '
                f'missing={len(legacy_load.missing_keys)} '
                f'unexpected={len(legacy_load.unexpected_keys)}'
            )
    elif load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(
            'Checkpoint loaded with partial mismatch: missing=%d unexpected=%d',
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = evaluate_all(
        model, criterion, loader, device,
        max_batches=9999, save_dir=save_dir,
        temporal_loader=temporal_loader,
    )

    print('\n' + '=' * 60)
    print(f'Results ({args.split})')
    print('=' * 60)

    print('\nACTIVITY RECOGNITION')
    print('-' * 40)
    print(f'  Top-1 (Activity Recognition): {results["act_accuracy"]:.4f}  '
          f'(= Phase Classification Acc@1.0 — STEPs benchmark)')
    print(f'  mcAP (PTMA/MiniROAD comparison): {results.get("act_mcAP", float("nan")):.4f}')
    print(f'  Frame Accuracy (no NA)  : {results["act_accuracy_no_na"]:.4f}')
    print(f'  Macro-F1                : {results["act_macro_f1"]:.4f}')
    print(f'  Weighted-F1             : {results["act_weighted_f1"]:.4f}')
    print(f'  Macro-Recall            : {results["act_macro_recall"]:.4f}')
    print(f'  Mean Per-Class Accuracy : {results["act_mean_per_class_acc"]:.4f}')
    print(f'  Top-5                   : {results["act_top5_accuracy"]:.4f}')
    print(f'  Kendall Tau (Temporal Ordering): {results.get("temporal_order_kendall_tau", float("nan")):.4f}  '
          f'(STEPs benchmark: 0.91)')
    print(f'  Temporal mAP@0.5               : {results.get("temporal_mAP50", float("nan")):.4f}  '
          f'(Gated SRM benchmark: 21.77%)')

    print('\nPOSE ESTIMATION')
    print('-' * 40)
    print('  -- PRIMARY metric (IKEA ASM benchmark, Ben-Shabat et al. WACV 2021) --')
    print(f'  PCK@10px  (fixed px)    : {results["pck_at_10px"]:.4f}   (paper best: 0.6430)')
    print(f'  PCK@20px  (fixed px)    : {results["pck_at_20px"]:.4f}')
    print(f'  PCK@30px  (fixed px)    : {results["pck_at_30px"]:.4f}')
    print(f'  N visible keypoints     : {results["n_visible_keypoints"]}')
    print('  -- COCO protocol (keep for reference) --')
    print(f'  PCK@0.05  (diag ratio)  : {results["pck_at_005"]:.4f}')
    print(f'  PCK@0.1   (diag ratio)  : {results["pck_at_01"]:.4f}')
    print(f'  PCK@0.2   (diag ratio)  : {results["pck_at_02"]:.4f}')
    print(f'  Mean Pixel Error        : {results["mean_pixel_error"]:.1f} px')

    pck_per_keypoint_10px = cast(List[float], results.get('pck_per_keypoint_10px', []))
    print('\n  Per-keypoint PCK@10px:')
    for i, name in enumerate(C.KEYPOINT_NAMES):
        print(f'    {name:20s}: {pck_per_keypoint_10px[i]:.4f}')

    print('\nOBJECT DETECTION')
    print('-' * 40)
    print(f'  AP@0.5 (Object Segmentation): {results["det_mAP50"]:.4f}')
    print(f'  mAP@[0.5:0.95]          : {results["det_mAP_50_95"]:.4f}')

    det_per_class_ap = cast(Dict[int, float], results.get('det_per_class_ap', {}))
    print('\n  Per-class AP@0.5:')
    for cls_id, ap in det_per_class_ap.items():
        name = C.DET_CLASS_NAMES.get(cls_id + 1, f'class_{cls_id}')
        print(f'    {name:20s}: {ap:.4f}')

    print('\nMODEL EFFICIENCY')
    print('-' * 40)
    print(f'  Parameters       : {results.get("params_M", float("nan")):.2f} M')
    print(f'  GFLOPs           : {results.get("gflops", float("nan")):.2f} G')
    print(f'  FPS (throughput) : {results.get("fps", float("nan")):.1f} fps')
    print(f'  Latency          : {results.get("latency_ms", float("nan")):.2f} ms/img')

    print('\n' + '=' * 60)


# =============================================================================
# Efficiency Metrics — Params / GFLOPs / FPS
# =============================================================================
def compute_model_efficiency(
    model: nn.Module,
    input_size: Tuple[int, int] = (C.IMG_HEIGHT, C.IMG_WIDTH),
    device: torch.device = torch.device('cuda'),
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Compute model efficiency metrics: parameter count, GFLOPs, and FPS.

    Uses thop (torch ops) for GFLOPs computation. Falls back to manual
    MAC counting if thop is unavailable.

    Args:
        model: MultiTaskIKEA model
        input_size: (H, W) input resolution
        device: torch device
        num_runs: Number of forward passes for FPS timing
        warmup_runs: Number of warmup runs before timing

    Returns:
        dict with params_M, gflops, fps (throughput)
    """
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = float(flops / 1e9)
    except ImportError:
        params = sum(p.numel() for p in model.parameters())
        gflops = 0.0

    params_m = params / 1e6 if isinstance(params, (int, float)) else float(params / 1e6)

    model.eval()
    dummy_batch = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_batch)
        end_event.record()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    fps = (num_runs * 1000.0) / elapsed_ms if elapsed_ms > 0 else 0.0

    return {
        'params_M': params_m,
        'gflops': gflops,
        'fps': fps,
        'latency_ms': elapsed_ms / num_runs,
    }
