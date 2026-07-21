#!/usr/bin/env python3
"""Comprehensive multi-head evaluation for MTLMViTModel.

Evaluates all 4 heads (Detection, Activity, Pose, PSR) on a given checkpoint
and reports metrics vs paper SOTA targets.

Usage:
    python scripts/eval_all_heads.py \
        --checkpoint runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
        --output /tmp/all_heads_eval.json
"""
import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

from train_mtl_full_multimodal import (
    FullMultiModalDataset,
    expand_conv_proj_to_9ch,
    WrappedMTL,
    ensure_5d,
    collate_real_targets,
)
from src.models.mvit_mtl_model import MTLMViTModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('eval_all_heads')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paper SOTA targets (from FULL_AUDIT_REPORT.md)
SOTA_TARGETS = {
    'detection_mAP50': 0.70,
    'activity_top1': 0.95,
    'activity_top5': None,
    'pose_MAE_deg': 5.0,
    'pose_MAE_pos_mm': None,
    'psr_F1': 0.80,
    'psr_edit_OSA': 0.85,
}


def build_model(num_anchors=16, num_classes_det=24, num_classes_act=75, num_psr=11):
    base = MTLMViTModel(
        num_act_classes=num_classes_act,
        num_det_classes=num_classes_det,
        num_psr_components=num_psr,
        num_anchors=num_anchors,
    )
    expand_conv_proj_to_9ch(base)
    return base.to(DEVICE)


def load_checkpoint(path, num_anchors=16):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    # Strip 'm.' prefix if present
    sd = {k[2:] if k.startswith('m.') else k: v for k, v in sd.items()}
    model = build_model(num_anchors=num_anchors)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f'Missing keys: {len(missing)} (examples: {missing[:3]})')
    if unexpected:
        logger.warning(f'Unexpected keys: {len(unexpected)}')
    return model


def normalize_inputs(images):
    """Normalize [B, 9, H, W] to model input."""
    images = images.float().to(DEVICE)
    images = ensure_5d(images)  # [B, 9, 1, H, W]
    mean = torch.tensor([0.45]*9, device=DEVICE).view(1, 9, 1, 1, 1)
    std = torch.tensor([0.225]*9, device=DEVICE).view(1, 9, 1, 1, 1)
    return (images - mean) / std


def evaluate_activity(model, loader, max_frames=2000):
    """Activity Recognition: Top-1 and Top-5 accuracy."""
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if total >= max_frames:
                break
            images_norm = normalize_inputs(images)
            out = model(images_norm)
            activity_logits = out['activity']  # [B, 75]

            # Get ground truth activity labels
            for i in range(activity_logits.shape[0]):
                act_gt = targets['activity'][i]
                if act_gt is None or act_gt < 0:
                    continue
                # Top-1
                pred_top1 = activity_logits[i].argmax().item()
                if pred_top1 == act_gt:
                    top1_correct += 1
                # Top-5
                top5_preds = activity_logits[i].topk(min(5, activity_logits.shape[1]))[1].tolist()
                if act_gt in top5_preds:
                    top5_correct += 1
                total += 1

    dt = time.time() - t0
    top1 = top1_correct / max(total, 1)
    top5 = top5_correct / max(total, 1)
    logger.info(f'  Activity: {total} samples in {dt:.1f}s')
    logger.info(f'  Top-1: {top1:.4f} ({top1_correct}/{total})')
    logger.info(f'  Top-5: {top5:.4f} ({top5_correct}/{total})')
    return {'top1': top1, 'top5': top5, 'correct_top1': top1_correct, 'correct_top5': top5_correct, 'total': total}


def evaluate_pose(model, loader, max_frames=2000):
    """Head Pose: MAE in degrees and mm."""
    model.eval()
    angle_errors = []
    pos_errors_mm = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if len(angle_errors) >= max_frames:
                break
            images_norm = normalize_inputs(images)
            out = model(images_norm)
            pose_6d = out['pose_6d']  # [B, 6] - fwd(3) + up(3), Tanh-bounded

            for i in range(pose_6d.shape[0]):
                pose_gt = targets['pose'][i]
                if pose_gt is None:
                    continue
                # pose_gt format: (fwd_angle_tuple[3], up_angle_tuple[3]) or similar
                # Following the train_mtl_v3.py format
                if not isinstance(pose_gt, (list, tuple)) or len(pose_gt) < 2:
                    continue
                fwd_gt = np.array(pose_gt[0])
                up_gt = np.array(pose_gt[1])
                if fwd_gt.shape != (3,) or up_gt.shape != (3,):
                    continue

                # Predicted: Tanh-bounded, normalize
                pred_fwd = F.normalize(pose_6d[i, :3].unsqueeze(0), dim=1)[0].cpu().numpy()
                pred_up = F.normalize(pose_6d[i, 3:].unsqueeze(0), dim=1)[0].cpu().numpy()

                # Angular error via dot product
                fwd_err = np.arccos(np.clip(np.dot(pred_fwd, fwd_gt), -1, 1)) * 180 / np.pi
                up_err = np.arccos(np.clip(np.dot(pred_up, up_gt), -1, 1)) * 180 / np.pi
                angle_errors.append((fwd_err + up_err) / 2)

    if angle_errors:
        mae = float(np.mean(angle_errors))
        logger.info(f'  Pose: {len(angle_errors)} samples')
        logger.info(f'  Angular MAE: {mae:.2f}°')
    else:
        mae = None
        logger.warning('  Pose: no valid samples')
    return {'angular_MAE_deg': mae, 'n_samples': len(angle_errors)}


def evaluate_psr(model, loader, max_frames=2000):
    """PSR (Procedure State Recognition): F1 and Edit Score."""
    model.eval()
    psr_preds = []
    psr_gts = []
    valid_count = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if valid_count >= max_frames:
                break
            images_norm = normalize_inputs(images)
            out = model(images_norm)
            psr_logits = out['psr_logits']  # [B, 11]
            psr_pred = (torch.sigmoid(psr_logits) > 0.5).cpu().numpy()

            for i in range(psr_pred.shape[0]):
                psr_gt = targets['psr'][i]
                if psr_gt is None:
                    continue
                psr_preds.append(psr_pred[i])
                psr_gts.append(psr_gt.numpy() if torch.is_tensor(psr_gt) else np.array(psr_gt))
                valid_count += 1

    if not psr_gts:
        return {'f1': None, 'edit_OSA': None, 'n_samples': 0}

    psr_preds = np.array(psr_preds)
    psr_gts = np.array(psr_gts)

    # Per-component F1
    f1_scores = []
    for c in range(psr_preds.shape[1]):
        y_true = psr_gts[:, c]
        y_pred = psr_preds[:, c]
        if y_true.sum() == 0 and y_pred.sum() == 0:
            continue
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1_scores.append(f1)

    f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    logger.info(f'  PSR: {valid_count} samples, {len(f1_scores)} valid components')
    logger.info(f'  Mean F1: {f1:.4f}')

    # Edit score (OSA) per sequence - simple version
    edit_scores = []
    for pred_seq, gt_seq in zip(psr_preds, psr_gts):
        # Treat each sequence as a string of 0/1
        pred_str = ''.join(map(str, pred_seq.astype(int)))
        gt_str = ''.join(map(str, gt_seq.astype(int)))
        # Simple edit distance (Hamming) - OSA is more complex but this is a proxy
        if len(pred_str) > 0:
            dist = sum(c1 != c2 for c1, c2 in zip(pred_str, gt_str))
            edit_scores.append(1.0 - dist / len(pred_str))
    edit_OSA = float(np.mean(edit_scores)) if edit_scores else 0.0
    logger.info(f'  Edit (Hamming): {edit_OSA:.4f}')

    return {
        'f1': f1,
        'edit_OSA': edit_OSA,
        'n_samples': valid_count,
        'per_component_f1': [float(f) for f in f1_scores],
    }


def evaluate_detection(model, loader, max_frames=2000, num_anchors=16):
    """Detection: mAP@0.5 using training-correct decode."""
    from train_mtl_v3 import _ANCHOR_SPECS_16, generate_anchors
    from src.losses.ciou import decode_deltas_to_xyxy
    from src.evaluation.evaluate import nms_numpy, compute_ap_per_class

    _ANCHOR_SPECS = _ANCHOR_SPECS_16

    def generate_proper_anchors(H, W):
        ys = (np.arange(H) + 0.5) / H
        xs = (np.arange(W) + 0.5) / W
        anchors_list = []
        for y in range(H):
            for x in range(W):
                for aw, ah in _ANCHOR_SPECS:
                    anchors_list.append([xs[x], ys[y], aw, ah])
        return np.array(anchors_list, dtype=np.float32)

    model.eval()
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_gt_boxes, all_gt_labels = [], []
    n_frames = 0
    t0 = time.time()

    with torch.no_grad():
        for images, gt_boxes_list, gt_labels_list in loader:
            if n_frames >= max_frames:
                break
            images_norm = normalize_inputs(images)
            out = model(images_norm)
            det_out = out['detection']
            B = images.shape[0]

            for b in range(B):
                image_pred_boxes, image_pred_scores, image_pred_labels = [], [], []

                for level_key, stride in [('P3', 8), ('P4', 16), ('P5', 32)]:
                    if level_key not in det_out:
                        continue
                    level = det_out[level_key]
                    cls_logits = level['cls_logits'][b]
                    reg_preds = level['reg_preds'][b]
                    H, W = cls_logits.shape[1], cls_logits.shape[2]

                    anchors_arr = generate_proper_anchors(H, W)
                    scores_map = torch.sigmoid(cls_logits)
                    max_scores, max_classes = scores_map.max(dim=0)

                    scores_flat = max_scores.reshape(-1).cpu().numpy()
                    classes_flat = max_classes.reshape(-1).cpu().numpy()

                    n_anchors = reg_preds.shape[0] // 4
                    reg_flat = reg_preds.reshape(4 * n_anchors, H * W).permute(1, 0).reshape(-1, 4).cpu().numpy()
                    scores_per_anchor = np.repeat(scores_flat, n_anchors)
                    classes_per_anchor = np.repeat(classes_flat, n_anchors)

                    keep = scores_per_anchor > 0.01
                    if keep.sum() > 5000:
                        topk_idx = np.argsort(scores_per_anchor)[-5000:]
                        keep_mask = np.zeros_like(keep)
                        keep_mask[topk_idx] = True
                        keep = keep & keep_mask
                    if keep.sum() == 0:
                        continue

                    kept_anchors = anchors_arr[keep]
                    kept_reg = reg_flat[keep]
                    kept_scores = scores_per_anchor[keep]
                    kept_classes = classes_per_anchor[keep]

                    decoded = decode_deltas_to_xyxy(
                        torch.from_numpy(kept_reg).float(),
                        torch.from_numpy(kept_anchors).float(),
                    )
                    pb = decoded.cpu().numpy()
                    pb[:, 0::2] *= C.IMG_WIDTH
                    pb[:, 1::2] *= C.IMG_HEIGHT

                    image_pred_boxes.append(pb)
                    image_pred_scores.append(kept_scores)
                    image_pred_labels.append(kept_classes)

                if image_pred_boxes:
                    all_pb = np.concatenate(image_pred_boxes, axis=0)
                    all_ps = np.concatenate(image_pred_scores, axis=0)
                    all_pl = np.concatenate(image_pred_labels, axis=0)
                    fb, fs, fl = [], [], []
                    for c in range(C.NUM_DET_CLASSES):
                        cm = all_pl == c
                        if cm.sum() == 0:
                            continue
                        nk = nms_numpy(all_pb[cm], all_ps[cm], 0.5)
                        fb.append(all_pb[cm][nk])
                        fs.append(all_ps[cm][nk])
                        fl.append(np.full(len(nk), c, dtype=np.int64))
                    if fb:
                        all_pred_boxes.append(np.concatenate(fb))
                        all_pred_scores.append(np.concatenate(fs))
                        all_pred_labels.append(np.concatenate(fl))
                    else:
                        all_pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
                        all_pred_scores.append(np.zeros(0, dtype=np.float32))
                        all_pred_labels.append(np.zeros(0, dtype=np.int32))
                else:
                    all_pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
                    all_pred_scores.append(np.zeros(0, dtype=np.float32))
                    all_pred_labels.append(np.zeros(0, dtype=np.int32))

                all_gt_boxes.append(gt_boxes_list[b].cpu().numpy())
                all_gt_labels.append(gt_labels_list[b].cpu().numpy())
                n_frames += 1

    # mAP@0.5 (annotated frames only)
    has_gt = [len(gb) > 0 for gb in all_gt_boxes]
    af_boxes = [all_pred_boxes[i] for i in range(len(has_gt)) if has_gt[i]]
    af_scores = [all_pred_scores[i] for i in range(len(has_gt)) if has_gt[i]]
    af_labels = [all_pred_labels[i] for i in range(len(has_gt)) if has_gt[i]]
    af_gtb = [all_gt_boxes[i] for i in range(len(has_gt)) if has_gt[i]]
    af_gtl = [all_gt_labels[i] for i in range(len(has_gt)) if has_gt[i]]

    if af_boxes:
        af_result = compute_ap_per_class(af_boxes, af_scores, af_labels, af_gtb, af_gtl, iou_thresh=0.5)
        mAP = float(af_result['mAP'])
    else:
        mAP = 0.0

    dt = time.time() - t0
    n_preds = sum(len(pb) for pb in all_pred_boxes)
    logger.info(f'  Detection: {n_frames} frames in {dt:.1f}s')
    logger.info(f'  mAP@0.5 (annotated-frames): {mAP:.4f}')
    logger.info(f'  Total predictions: {n_preds:,}')

    return {
        'mAP50': mAP,
        'n_frames': n_frames,
        'n_preds': n_preds,
    }


def collate_eval(batch):
    """Collate batch into (images, gt_boxes, gt_labels) for detection eval.

    FullMultiModalDataset returns (x, targets) tuples where x is [9, H, W]
    image tensor and targets is a dict with 'boxes' and 'labels' lists.
    """
    images = torch.stack([b[0] for b in batch])  # [B, 9, H, W]
    targets = [b[1] for b in batch]
    gt_boxes = [t.get('boxes', torch.zeros(0, 4)) for t in targets]
    gt_labels = [t.get('classes', torch.zeros(0, dtype=torch.long)) for t in targets]
    return images, gt_boxes, gt_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-anchors', type=int, default=16)
    parser.add_argument('--max-frames', type=int, default=2000)
    parser.add_argument('--output', type=str, default='/tmp/all_heads_eval.json')
    args = parser.parse_args()

    logger.info(f'Loading checkpoint: {args.checkpoint}')
    model = load_checkpoint(args.checkpoint, num_anchors=args.num_anchors)
    model.eval()
    logger.info(f'Model loaded. Device: {DEVICE}')

    # Build dataset
    logger.info('Loading validation dataset...')
    val_ds = FullMultiModalDataset(
        recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/val',
        img_size=(640, 360),
        mosaic_prob=0.0,
        copy_paste_prob=0.0,
    )
    logger.info(f'Dataset: {len(val_ds)} frames')

    # Run all 4 evaluations
    logger.info('='*60)
    logger.info('Evaluating all 4 heads...')
    logger.info('='*60)

    # 1. Activity
    logger.info('\n[1/4] Activity Recognition')
    act_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            collate_fn=collate_real_targets, num_workers=0)
    activity_results = evaluate_activity(model, act_loader, args.max_frames)

    # 2. Pose
    logger.info('\n[2/4] Head Pose')
    pose_results = evaluate_pose(model, act_loader, args.max_frames)

    # 3. PSR
    logger.info('\n[3/4] PSR')
    psr_results = evaluate_psr(model, act_loader, args.max_frames)

    # 4. Detection (uses its own collate to get GT boxes/labels)
    logger.info('\n[4/4] Detection')
    det_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            collate_fn=collate_eval, num_workers=0)
    det_results = evaluate_detection(model, det_loader, args.max_frames, args.num_anchors)

    # Summary
    logger.info('\n' + '='*60)
    logger.info('SUMMARY: All 4 heads')
    logger.info('='*60)

    summary = {
        'checkpoint': args.checkpoint,
        'n_frames': det_results.get('n_frames', 0),
        'detection': {
            'mAP50': det_results['mAP50'],
            'n_preds': det_results['n_preds'],
            'SOTA_target': SOTA_TARGETS['detection_mAP50'],
            'gap_to_SOTA': SOTA_TARGETS['detection_mAP50'] - det_results['mAP50'],
            'status': 'PASS' if det_results['mAP50'] >= SOTA_TARGETS['detection_mAP50'] else 'FAIL',
        },
        'activity': {
            'top1': activity_results['top1'],
            'top5': activity_results['top5'],
            'n_samples': activity_results['total'],
            'SOTA_target': SOTA_TARGETS['activity_top1'],
            'gap_to_SOTA': SOTA_TARGETS['activity_top1'] - activity_results['top1'],
            'status': 'PASS' if activity_results['top1'] >= SOTA_TARGETS['activity_top1'] else 'FAIL',
        },
        'pose': {
            'angular_MAE_deg': pose_results['angular_MAE_deg'],
            'n_samples': pose_results['n_samples'],
            'SOTA_target': SOTA_TARGETS['pose_MAE_deg'],
            'gap_to_SOTA': (pose_results['angular_MAE_deg'] - SOTA_TARGETS['pose_MAE_deg']) if pose_results['angular_MAE_deg'] else None,
            'status': 'PASS' if pose_results['angular_MAE_deg'] is not None and pose_results['angular_MAE_deg'] <= SOTA_TARGETS['pose_MAE_deg'] else 'FAIL',
        },
        'psr': {
            'f1': psr_results['f1'],
            'edit_OSA': psr_results['edit_OSA'],
            'n_samples': psr_results['n_samples'],
            'SOTA_target': SOTA_TARGETS['psr_F1'],
            'gap_to_SOTA': SOTA_TARGETS['psr_F1'] - (psr_results['f1'] or 0),
            'status': 'PASS' if psr_results['f1'] and psr_results['f1'] >= SOTA_TARGETS['psr_F1'] else 'FAIL',
        },
    }

    for head in ['detection', 'activity', 'pose', 'psr']:
        h = summary[head]
        gap_str = f'{h["gap_to_SOTA"]:+.4f}' if h['gap_to_SOTA'] is not None else 'N/A'
        logger.info(f'  {head:12s}: {h["status"]:4s} (gap: {gap_str})')

    n_pass = sum(1 for h in ['detection', 'activity', 'pose', 'psr'] if summary[h]['status'] == 'PASS')
    n_fail = 4 - n_pass
    logger.info(f'\nOverall: {n_pass}/4 heads PASS, {n_fail}/4 FAIL')

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'Results saved: {args.output}')


if __name__ == '__main__':
    main()