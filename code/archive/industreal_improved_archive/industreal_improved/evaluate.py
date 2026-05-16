"""
Evaluation script for POPW multi-task model.
Computes all metrics for benchmarking against paper targets:

IndustReal targets:
- ASD mAP (b-boxed) > 0.838, mAP@0.5 > 0.838 (YOLOv8m)
- Activity Top-1 > 66.45%, Top-5 > 88.43% (MViTv2)
- PSR F1 > 0.901 (±3 frames), > 0.883 (±5 frames) (STORM-PSR / B3)
- PSR POS > 0.812 (±3 frames) (STORM-PSR)
- Assembly State F1@1 > 0.85 (SupCon+ISIL)
- Error Verification AP > 0.58 (GCA)
- Head pose MAE (establish baseline)
"""

import os
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import C
from model import POPWModel
from industreal_dataset import IndustRealDataset, collate_fn, Transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate POPW model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--flip-tta", action="store_true", help="Use horizontal flip TTA")
    parser.add_argument("--crop-tta", action="store_true", help="Use 5-crop TTA")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


# =============================================================================
# Detection Metrics — Doc 01 §E / Doc 03
# =============================================================================

def compute_bbox_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format (center-based)."""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)


def compute_detection_metrics(predictions: List, ground_truth: List,
                               iou_threshold: float = 0.5,
                               all_frame_count: int = None) -> Dict:
    """
    Compute detection mAP following Doc 01 §E / Doc 03 protocol.

    Args:
        predictions: list of dicts with keys: image_id, bbox, score, category_id
        ground_truth: list of dicts with keys: image_id, bbox, category_id
        iou_threshold: IoU threshold for matching (default 0.5)
        all_frame_count: total number of frames in dataset (annotated + unannotated)
                        Used to compute mAP_all_frames. If None, only mAP_annotated is computed.

    Returns:
        dict with keys:
            - mAP_annotated_only: COCO-style AP over frames that have GT bboxes
            - mAP_all_frames: mAP over ALL frames (unannotated frames with no pred = TN)
    """
    num_classes = C.NUM_CLASSES_DET  # 24 ASD classes

    # Get set of annotated frames from ground truth
    annotated_frames = set(gt['image_id'] for gt in ground_truth)
    n_annotated = len(annotated_frames)

    # Build frame-level GT index per class
    frame_to_gts = defaultdict(list)
    for gt in ground_truth:
        frame_to_gts[gt['image_id']].append(gt)

    aps_per_class = []

    for class_id in range(1, num_classes + 1):
        class_preds = [p for p in predictions if p['category_id'] == class_id]
        class_gts = [g for g in ground_truth if g['category_id'] == class_id]

        if len(class_gts) == 0:
            continue

        # Sort predictions by score descending
        class_preds.sort(key=lambda x: x['score'], reverse=True)

        # Build frame-level GT index for this class
        class_frame_to_gts = defaultdict(list)
        for gt in class_gts:
            class_frame_to_gts[gt['image_id']].append(gt)

        # Track matched GTs
        matched = set()

        tp_list = []
        fp_list = []

        for pred in class_preds:
            frame_id = pred['image_id']

            if frame_id not in annotated_frames:
                # Prediction on unannotated frame = false positive
                tp_list.append(0)
                fp_list.append(1)
                continue

            preds_in_frame = class_frame_to_gts.get(frame_id, [])
            best_iou = 0
            best_gt_idx = None

            for gt_idx, gt in enumerate(preds_in_frame):
                iou = compute_bbox_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx is not None:
                key = (frame_id, best_gt_idx)
                if key not in matched:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched.add(key)
                else:
                    # Duplicate detection — GT already matched
                    tp_list.append(0)
                    fp_list.append(1)
            else:
                # No match above threshold
                tp_list.append(0)
                fp_list.append(1)

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)

        total_gt = len(class_gts)
        recalls = tp_cumsum / (total_gt + 1e-10)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # VOC-style 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.05, 0.1):
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask])
        ap /= 11.0

        aps_per_class.append(ap)

    # mAP_annotated_only: mean AP over classes where we have GT
    mAP_annotated_only = np.mean(aps_per_class) * 100.0 if aps_per_class else 0.0

    # mAP_all_frames: scale by fraction of frames that are annotated
    # Unannotated frames with no prediction are vacuously correct (TN)
    # Unannotated frames with prediction are FP (already counted above)
    # The mAP drop comes from the fact that unannotated frames add denominator
    # but contribute neither TP nor FP to the annotated frames' calculation
    if all_frame_count is not None and all_frame_count > 0 and n_annotated > 0:
        mAP_all_frames = mAP_annotated_only * (n_annotated / all_frame_count)
    elif all_frame_count is not None:
        mAP_all_frames = 0.0
    else:
        mAP_all_frames = mAP_annotated_only

    return {
        'mAP_annotated_only': round(mAP_annotated_only, 2),
        'mAP_all_frames': round(mAP_all_frames, 2),
    }


# =============================================================================
# Activity Recognition — Clip-level Top-1/Top-5 — Doc 03 §3
# =============================================================================

def compute_activity_metrics(cls_preds, activity_labels, recording_ids, frame_ids,
                               num_classes=74):
    """
    Compute clip-level activity recognition metrics.

    Protocol per Doc 03:
    - Clip = 64 consecutive frames from same recording
    - Average softmax logits over clip → clip prediction
    - Top-1: clip prediction matches video-level label
    - Top-5: true label in top-5 clip predictions

    Args:
        cls_preds: [N_frames, num_classes] softmax probabilities (torch.Tensor)
        activity_labels: [N_frames] integer labels (torch.Tensor)
        recording_ids: [N_frames] recording ID strings
        frame_ids: [N_frames] frame indices (int)
        num_classes: number of activity classes

    Returns:
        dict with top1_accuracy, top5_accuracy (percentages)
    """
    # Group frames by (recording_id, clip_id)
    clips = {}  # (rec_id, clip_id) -> {'logits': [], 'labels': set()}

    for i in range(len(cls_preds)):
        rec_id = recording_ids[i]
        clip_id = int(frame_ids[i]) // 64  # 64 frames per clip
        key = (rec_id, clip_id)
        if key not in clips:
            clips[key] = {'logits': [], 'labels': set()}
        clips[key]['logits'].append(cls_preds[i])
        clips[key]['labels'].add(activity_labels[i].item())

    top1_correct = 0
    top5_correct = 0
    total_clips = 0

    for (rec_id, clip_id), data in clips.items():
        clip_logits = torch.stack(data['logits']).mean(dim=0)  # [num_classes]
        top5_preds = clip_logits.topk(5).indices  # [5]
        true_label = list(data['labels'])[0]  # all frames in clip should have same label
        total_clips += 1

        if top5_preds[0] == true_label:
            top1_correct += 1
        if true_label in top5_preds:
            top5_correct += 1

    return {
        'top1_accuracy': round(top1_correct / total_clips * 100, 2) if total_clips > 0 else 0.0,
        'top5_accuracy': round(top5_correct / total_clips * 100, 2) if total_clips > 0 else 0.0,
    }


# =============================================================================
# PSR — F1 and POS with Frame Tolerance — Doc 03 §4
# =============================================================================

def compute_psr_metrics(pred_psr_scores, true_psr_labels, pred_timestamps,
                         true_timestamps, tolerance_frames=3):
    """
    Compute PSR F1 and POS (Procedure Order Similarity) with tolerance.

    Protocol per STORM-PSR (Doc 03):
    - Predicted transition is correct if within tolerance_frames of true transition
    - Greedy matching (each true transition matched once)
    - F1: standard precision/recall on transition matching
    - POS: average normalized proximity for correctly matched transitions

    Args:
        pred_psr_scores: [N] sigmoid probabilities (torch.Tensor)
        true_psr_labels: [N] binary 0/1 labels (torch.Tensor)
        pred_timestamps: [N] frame indices of predictions (torch.Tensor)
        true_timestamps: [N] frame indices (only for positive events) (torch.Tensor)
        tolerance_frames: frame tolerance (±N frames)

    Returns:
        dict with f1, precision, recall, pos
    """
    # Convert to numpy
    pred_binary = (pred_psr_scores > 0.5).cpu().numpy().astype(int)
    true_labels = true_psr_labels.cpu().numpy().astype(int)
    pred_ts = pred_timestamps.cpu().numpy()
    true_ts = true_timestamps.cpu().numpy()

    # Find predicted transitions (0→1 edges)
    pred_transitions = []
    for i in range(1, len(pred_binary)):
        if pred_binary[i] == 1 and pred_binary[i - 1] == 0:
            pred_transitions.append(pred_ts[i])

    # Find true transitions
    true_transitions = []
    for i in range(1, len(true_labels)):
        if true_labels[i] == 1 and true_labels[i - 1] == 0:
            true_transitions.append(true_ts[i])

    # Greedy matching with tolerance
    matched_pred = set()
    matched_true = set()

    for tt in true_transitions:
        best_dist = tolerance_frames + 1
        best_pred = None
        for pt in pred_transitions:
            if pt in matched_pred:
                continue
            dist = abs(pt - tt)
            if dist <= tolerance_frames and dist < best_dist:
                best_dist = dist
                best_pred = pt
        if best_pred is not None:
            matched_pred.add(best_pred)
            matched_true.add(tt)

    tp = len(matched_true)
    fp = len(pred_transitions) - tp
    fn = len(true_transitions) - tp

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # POS: For correctly matched transitions, normalized proximity
    pos_scores = []
    for tt in true_transitions:
        if tt in matched_true:
            # Find matching pred
            for pt in pred_transitions:
                if abs(pt - tt) <= tolerance_frames:
                    # Normalized: 1.0 = perfect, 0.0 = at tolerance boundary
                    pos_scores.append(1.0 - abs(pt - tt) / tolerance_frames)
                    break

    pos = np.mean(pos_scores) if pos_scores else 0.0

    return {
        'f1': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'pos': round(pos, 4),
    }


# =============================================================================
# Head Pose MAE — Doc 03 §5
# =============================================================================

def compute_head_pose_mae(pred_poses, gt_poses):
    """
    Compute head pose angular MAE.

    Protocol per Doc 03:
    - Forward vector: angular error (degrees)
    - Up vector: angular error (degrees)
    - Position: MSE (mm)

    Args:
        pred_poses: [N, 9] array (forward, position, up)
        gt_poses: [N, 9] array

    Returns:
        dict with forward_angular_mae_deg, up_angular_mae_deg, position_mae_mm
    """
    forward_errors = []
    up_errors = []
    position_errors = []

    for pred, gt in zip(pred_poses, gt_poses):
        # Forward vector (first 3)
        pred_forward = pred[:3] / (np.linalg.norm(pred[:3]) + 1e-6)
        gt_forward = gt[:3] / (np.linalg.norm(gt[:3]) + 1e-6)
        forward_dot = np.clip(np.dot(pred_forward, gt_forward), -1, 1)
        forward_mae = np.arccos(forward_dot) * 180 / np.pi
        forward_errors.append(forward_mae)

        # Up vector (last 3)
        pred_up = pred[6:] / (np.linalg.norm(pred[6:]) + 1e-6)
        gt_up = gt[6:] / (np.linalg.norm(gt[6:]) + 1e-6)
        up_dot = np.clip(np.dot(pred_up, gt_up), -1, 1)
        up_mae = np.arccos(up_dot) * 180 / np.pi
        up_errors.append(up_mae)

        # Position (middle 3) - convert to mm (assuming 1 unit = 1 mm)
        pos_mae = np.linalg.norm(pred[3:6] - gt[3:6])
        position_errors.append(pos_mae)

    return {
        'forward_angular_mae_deg': round(np.mean(forward_errors), 2),
        'up_angular_mae_deg': round(np.mean(up_errors), 2),
        'position_mae_mm': round(np.mean(position_errors), 2),
    }


# =============================================================================
# Assembly State F1 — Doc 03 §6
# =============================================================================

def compute_assembly_state_f1(pred_detections, gt_states, threshold=0.5):
    """
    Compute Assembly State F1@1 (per-frame top-1 match).

    Args:
        pred_detections: list of dicts with 'boxes', 'scores', 'labels'
        gt_states: list of int class labels
    """
    tp = fp = fn = 0

    for pred, gt in zip(pred_detections, gt_states):
        if len(pred.get('scores', [])) == 0:
            if gt == 0:
                continue
            else:
                fn += 1
                continue

        top_class = pred['labels'][0].item() if hasattr(pred['labels'][0], 'item') else pred['labels'][0]
        top_score = pred['scores'][0].item() if hasattr(pred['scores'][0], 'item') else pred['scores'][0]

        if top_score < threshold:
            if gt == 0:
                continue
            else:
                fn += 1
        else:
            if top_class == gt:
                tp += 1
            else:
                fp += 1
                fn += 1

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return round(f1, 4)


# =============================================================================
# Error Verification AP — Doc 03 §7
# =============================================================================

def compute_error_verification_ap(pred_scores, gt_errors):
    """
    Compute Error Verification AP.

    Score = 1 - confidence(expected_state)
    """
    from sklearn.metrics import average_precision_score

    y_pred = [float(s) for s in pred_scores]
    y_true = [1 if e else 0 for e in gt_errors]

    if len(set(y_true)) < 2:
        return 0.0

    ap = average_precision_score(y_true, y_pred)
    return round(ap, 4)


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def evaluate_model(model, dataset, device, mode='val', sequence_mode=False):
    """
    Run evaluation on dataset.

    Args:
        model: POPWModel
        dataset: IndustRealDataset
        device: torch device
        mode: 'val' or 'test'
        sequence_mode: whether dataset is in sequence mode

    Returns:
        dict with all collected predictions for metric computation
    """
    model.eval()

    all_det_predictions = []
    all_det_ground_truths = []
    all_activity_logits = []
    all_activity_labels = []
    all_recording_ids = []
    all_frame_ids = []
    all_psr_scores = []  # sigmoid probabilities
    all_psr_labels = []
    all_psr_frame_idx = []
    all_head_pose_preds = []
    all_head_pose_gts = []

    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 500 == 0:
                print(f"  Evaluation batch {batch_idx}/{len(data_loader)}")

            images = batch['images'].to(device)

            # Forward pass
            outputs = model(
                images,
                video_id=batch['recording_ids'][0] if batch['recording_ids'] else "eval"
            )

            B = images.size(0)

            # ---- Detection ----
            cls_preds = outputs.get('cls_preds', None)
            reg_preds = outputs.get('reg_preds', None)

            if cls_preds is not None and len(cls_preds) > 0:
                # cls_preds[0]: [B, N, 24], reg_preds[0]: [B, N, 4]
                cls_pred = cls_preds[0][0]  # [N, 24]
                reg_pred = reg_preds[0][0]  # [N, 4]

                scores, labels = cls_pred.max(dim=-1)  # [N], [N]

                # Filter by confidence
                keep = scores > C.DET_CONF_THRESH
                boxes = reg_pred[keep].cpu().numpy()
                sc = scores[keep].cpu().numpy()
                lbls = labels[keep].cpu().numpy()

                frame_name = batch['frame_names'][0] if 'frame_names' in batch else batch.get('frame_name', 'unknown')
                frame_idx = batch.get('frame_indices', [0])[0]

                for j in range(len(boxes)):
                    all_det_predictions.append({
                        'image_id': frame_name,
                        'frame_idx': frame_idx,
                        'bbox': boxes[j].tolist(),
                        'score': float(sc[j]),
                        'category_id': int(lbls[j]) + 1,  # 1-indexed
                    })

            # Ground truth detection
            det_labels = batch.get('det_labels', [{}])[0]
            if det_labels and len(det_labels.get('boxes', [])) > 0:
                boxes_gt = det_labels['boxes'].cpu()
                labels_gt = det_labels['labels'].cpu()
                frame_name = batch['frame_names'][0] if 'frame_names' in batch else batch.get('frame_name', 'unknown')
                for k in range(len(boxes_gt)):
                    all_det_ground_truths.append({
                        'image_id': frame_name,
                        'bbox': boxes_gt[k].tolist(),
                        'category_id': int(labels_gt[k]) + 1,
                    })

            # ---- Activity Recognition ----
            act_logits = outputs.get('act_logits', None)
            if act_logits is not None:
                act_softmax = F.softmax(act_logits[0], dim=-1).cpu()  # [num_classes]
                all_activity_logits.append(act_softmax)
                all_activity_labels.append(batch['activity_labels'][0].item())
                all_recording_ids.append(batch['recording_ids'][0] if batch['recording_ids'] else 'unknown')
                all_frame_ids.append(batch.get('frame_indices', [batch_idx])[0])

            # ---- PSR ----
            psr_logits = outputs.get('psr_logits', None)
            if psr_logits is not None:
                psr_sig = torch.sigmoid(psr_logits[0]).cpu()  # [T, 11]
                psr_lbl = batch['psr_labels'][0].cpu()  # [T, 11]
                frame_indices = torch.arange(psr_sig.size(0))

                for t in range(psr_sig.size(0)):
                    for c in range(psr_sig.size(1)):
                        all_psr_scores.append(psr_sig[t, c].item())
                        all_psr_labels.append(psr_lbl[t, c].item())
                        all_psr_frame_idx.append(batch_idx * 100 + t)  # approximate global frame index

            # ---- Head Pose ----
            head_pose = outputs.get('head_pose', None)
            if head_pose is not None:
                all_head_pose_preds.append(head_pose[0].cpu())
                all_head_pose_gts.append(batch['head_pose'][0].cpu())

    return {
        'det_predictions': all_det_predictions,
        'det_ground_truths': all_det_ground_truths,
        'activity_logits': all_activity_logits,
        'activity_labels': all_activity_labels,
        'recording_ids': all_recording_ids,
        'frame_ids': all_frame_ids,
        'psr_scores': all_psr_scores,
        'psr_labels': all_psr_labels,
        'psr_frame_idx': all_psr_frame_idx,
        'head_pose_preds': all_head_pose_preds,
        'head_pose_gts': all_head_pose_gts,
        'total_frames': len(dataset),
    }


def compute_all_metrics(results):
    """Compute all metrics from evaluation results."""
    metrics = {}

    # ---- Detection mAP (two numbers) ----
    det_preds = results['det_predictions']
    det_gts = results['det_ground_truths']
    total_frames = results.get('total_frames', None)

    if len(det_gts) > 0:
        det_metrics = compute_detection_metrics(det_preds, det_gts, iou_threshold=0.5,
                                                  all_frame_count=total_frames)
        metrics['mAP_annotated_only'] = det_metrics['mAP_annotated_only']
        metrics['mAP_all_frames'] = det_metrics['mAP_all_frames']
    else:
        metrics['mAP_annotated_only'] = 0.0
        metrics['mAP_all_frames'] = 0.0

    # ---- Activity Recognition (clip-level) ----
    if len(results['activity_logits']) > 0:
        cls_preds = torch.stack(results['activity_logits'])  # [N, 74]
        activity_labels = torch.tensor(results['activity_labels'])
        recording_ids = results['recording_ids']
        frame_ids = torch.tensor(results['frame_ids'])

        act_metrics = compute_activity_metrics(
            cls_preds, activity_labels, recording_ids, frame_ids, num_classes=C.NUM_CLASSES_ACT
        )
        metrics['activity_top1_clip'] = act_metrics['top1_accuracy']
        metrics['activity_top5_clip'] = act_metrics['top5_accuracy']
    else:
        metrics['activity_top1_clip'] = 0.0
        metrics['activity_top5_clip'] = 0.0

    # ---- PSR F1 + POS (both tolerances) ----
    if len(results['psr_scores']) > 0:
        psr_scores = torch.tensor(results['psr_scores'])
        psr_labels = torch.tensor(results['psr_labels'])
        psr_frame_idx = torch.tensor(results['psr_frame_idx'])

        r3 = compute_psr_metrics(psr_scores, psr_labels, psr_frame_idx, psr_frame_idx,
                                  tolerance_frames=3)
        r5 = compute_psr_metrics(psr_scores, psr_labels, psr_frame_idx, psr_frame_idx,
                                  tolerance_frames=5)

        metrics['psr_f1_at3'] = r3['f1']
        metrics['psr_precision_at3'] = r3['precision']
        metrics['psr_recall_at3'] = r3['recall']
        metrics['psr_pos_at3'] = r3['pos']

        metrics['psr_f1_at5'] = r5['f1']
        metrics['psr_precision_at5'] = r5['precision']
        metrics['psr_recall_at5'] = r5['recall']
        metrics['psr_pos_at5'] = r5['pos']
    else:
        metrics['psr_f1_at3'] = metrics['psr_f1_at5'] = 0.0
        metrics['psr_pos_at3'] = metrics['psr_pos_at5'] = 0.0

    # ---- Head Pose MAE ----
    if len(results['head_pose_preds']) > 0:
        hp_preds = torch.stack(results['head_pose_preds']).numpy()
        hp_gts = torch.stack(results['head_pose_gts']).numpy()
        pose_metrics = compute_head_pose_mae(hp_preds, hp_gts)
        metrics.update(pose_metrics)
    else:
        metrics['forward_angular_mae_deg'] = 0.0
        metrics['up_angular_mae_deg'] = 0.0
        metrics['position_mae_mm'] = 0.0

    return metrics


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = POPWModel(config=C)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('model_state', checkpoint), strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Dataset
    transform = Transforms(is_train=False)
    dataset = IndustRealDataset(split=args.split, transform=transform, sequence_mode=False)

    print(f"Dataset: {len(dataset)} samples")

    # Evaluate
    print("Running evaluation...")
    results = evaluate_model(model, dataset, device, mode=args.split)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(results)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Detection (ASD) ---")
    print(f"mAP@0.5 (annotated frames only): {metrics.get('mAP_annotated_only', 0):.2f}%")
    print(f"mAP@0.5 (all frames): {metrics.get('mAP_all_frames', 0):.2f}%")
    print(f"Benchmark: YOLOv8m = 83.80%")

    print("\n--- Activity Recognition (Clip-level) ---")
    print(f"Top-1: {metrics.get('activity_top1_clip', 0):.2f}%")
    print(f"Top-5: {metrics.get('activity_top5_clip', 0):.2f}%")
    print(f"Benchmark: MViTv2 Top-1=66.45%, Top-5=88.43%")

    print("\n--- PSR (Procedure Step Recognition) ---")
    print(f"PSR@±3: F1={metrics.get('psr_f1_at3', 0):.4f} POS={metrics.get('psr_pos_at3', 0):.4f}")
    print(f"PSR@±5: F1={metrics.get('psr_f1_at5', 0):.4f} POS={metrics.get('psr_pos_at5', 0):.4f}")
    print(f"Benchmark: STORM-PSR F1@±3=0.901, POS=0.812")
    print(f"          B3 baseline F1@±5=0.883")

    print("\n--- Head Pose (9-DoF) ---")
    print(f"Forward angular MAE: {metrics.get('forward_angular_mae_deg', 0):.2f}°")
    print(f"Up angular MAE: {metrics.get('up_angular_mae_deg', 0):.2f}°")
    print(f"Position MAE: {metrics.get('position_mae_mm', 0):.2f} mm")

    # Save metrics
    output_path = Path(C.LOG_DIR) / f"eval_results_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    main()