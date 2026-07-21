#!/usr/bin/env python3
"""Fixed mAP@0.5 eval for MTLMViTModel.

CRITICAL FIXES (2026-07-21):
1. Uses TRAINING decode formula: cx = anchor_cx + dx * 0.1  (NOT dx * anchor_w)
2. Uses _ANCHOR_SPECS anchors matching training config (NOT stride-sized squares)
3. Anchors are in (cx, cy, w, h) normalized format, decoded to (x1, y1, x2, y2) absolute px
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parent.parent / "src"
for _p in [str(_SRC), str(_SRC.parent), str(_SRC / "evaluation"), str(_SRC / "data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset
from src.models.mvit_mtl_model import MTLMViTModel
from src.evaluation.evaluate import nms_numpy, compute_ap_per_class, compute_ap_per_class_all_frames
from src.losses.ciou import decode_deltas_to_xyxy

from train_mtl_full_multimodal import expand_conv_proj_to_9ch, WrappedMTL

# Import anchor specs from train_mtl_v3
import train_mtl_v3 as mtl_mod
NUM_ANCHORS = mtl_mod.NUM_ANCHORS  # 8 or 16
_ANCHOR_SPECS = mtl_mod._ANCHOR_SPECS

C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360
C.NUM_DET_CLASSES = 24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_images(images, device):
    """Normalize to ~0 mean unit var, make [B,9,1,H,W]."""
    images = images.float().to(device)
    mean = torch.tensor([0.45], device=device).view(1, 1, 1, 1)
    std = torch.tensor([0.225], device=device).view(1, 1, 1, 1)
    images = (images / 255.0 - mean) / std
    images = images.repeat(1, 3, 1, 1)  # [B, 9, H, W]
    images = images.unsqueeze(2)  # [B, 9, 1, H, W]
    return images


def collate_eval(batch):
    """Collate batch from IndustRealMultiTaskDataset for eval."""
    images = torch.stack([b['images']['rgb'] for b in batch])
    det_list = [b.get('detection', {}) for b in batch]
    gt_boxes = [d.get('boxes', torch.zeros(0, 4)) for d in det_list]
    gt_labels = [d.get('labels', torch.zeros(0, dtype=torch.long)) for d in det_list]
    return images, gt_boxes, gt_labels


def generate_proper_anchors(H, W, device=None):
    """Generate anchors matching training config.

    Uses _ANCHOR_SPECS (from train_mtl_v3.py) — either k-means 8 or legacy 16.
    Anchors are in (cx, cy, w, h) normalized [0, 1] format, matching training.

    Returns:
        anchors_arr: [H*W*NUM_ANCHORS, 4] in (cx, cy, w, h) normalized
        anchor_cell_map: [H*W*NUM_ANCHORS] mapping each anchor to its grid cell
    """
    if device is None:
        device = torch.device('cpu')
    ys = (torch.arange(H, device=device) + 0.5) / H
    xs = (torch.arange(W, device=device) + 0.5) / W

    anchors_list = []
    cell_map_list = []
    for y in range(H):
        for x in range(W):
            for a_idx, (aw, ah) in enumerate(_ANCHOR_SPECS):
                anchors_list.append([xs[x].item(), ys[y].item(), aw, ah])
                cell_map_list.append(y * W + x)

    return np.array(anchors_list, dtype=np.float32), np.array(cell_map_list, dtype=np.int64)


def get_decoded_predictions(model, loader, max_frames=None, score_thresh=0.01):
    """Run inference, decode using training-correct formula, return results."""
    model.eval()
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_gt_boxes, all_gt_labels = [], []
    n_frames = 0
    t0 = time.time()

    with torch.no_grad():
        for images, gt_boxes, gt_labels in loader:
            if max_frames and n_frames >= max_frames:
                break
            images_norm = normalize_images(images, DEVICE)
            out = model(images_norm)
            det_out = out['detection']
            B = images.shape[0]

            for b in range(B):
                image_pred_boxes, image_pred_scores, image_pred_labels = [], [], []

                for level_key, stride in [('P3', 8), ('P4', 16), ('P5', 32)]:
                    if level_key not in det_out:
                        continue
                    level = det_out[level_key]
                    cls_logits = level['cls_logits'][b]  # [C, H, W]
                    reg_preds = level['reg_preds'][b]    # [4*A, H, W]
                    H, W = cls_logits.shape[1], cls_logits.shape[2]

                    # Generate proper training-matching anchors
                    anchors_arr, _ = generate_proper_anchors(H, W)
                    # anchors_arr: [H*W*A, 4] in (cx, cy, w, h) normalized [0,1]

                    # Scores: sigmoid(cls_logits) max over classes
                    scores_map = torch.sigmoid(cls_logits)  # [C, H, W]
                    max_scores, max_classes = scores_map.max(dim=0)  # [H, W]

                    # Flatten: [H, W] -> [H*W]
                    scores_flat = max_scores.reshape(-1).cpu().numpy()
                    classes_flat = max_classes.reshape(-1).cpu().numpy()

                    # Reg: [4*A, H, W] -> [H*W, A, 4] -> [H*W*A, 4]
                    n_anchors = reg_preds.shape[0] // 4
                    reg_4d = reg_preds.reshape(4 * n_anchors, H * W)
                    reg_flat = reg_4d.permute(1, 0).reshape(H * W * n_anchors, 4).cpu().numpy()

                    # Replicate scores and classes for each anchor at each position
                    scores_per_anchor = np.repeat(scores_flat, n_anchors)  # [H*W*A]
                    classes_per_anchor = np.repeat(classes_flat, n_anchors)  # [H*W*A]

                    # Threshold
                    keep = scores_per_anchor > score_thresh
                    if keep.sum() > 5000:
                        topk_idx = np.argsort(scores_per_anchor)[-5000:]
                        keep_mask = np.zeros_like(keep)
                        keep_mask[topk_idx] = True
                        keep = keep & keep_mask
                    if keep.sum() == 0:
                        continue

                    # Apply decode with TRAINING-correct formula
                    kept_anchors = anchors_arr[keep]  # [K, 4] in (cx, cy, w, h) normalized
                    kept_reg = reg_flat[keep]  # [K, 4] = (dx, dy, dw, dh)
                    kept_scores = scores_per_anchor[keep]
                    kept_classes = classes_per_anchor[keep]

                    # Decode using training formula: cx = anchor_cx + dx*0.1, w = anchor_w * exp(dw)
                    decoded = decode_deltas_to_xyxy(
                        torch.from_numpy(kept_reg).float(),
                        torch.from_numpy(kept_anchors).float(),
                    )  # [K, 4] in (x1, y1, x2, y2) normalized [0,1]

                    # Convert to absolute pixels
                    pb = decoded.cpu().numpy()
                    pb[:, 0::2] *= C.IMG_WIDTH   # x1, x2
                    pb[:, 1::2] *= C.IMG_HEIGHT  # y1, y2

                    image_pred_boxes.append(pb)
                    image_pred_scores.append(kept_scores)
                    image_pred_labels.append(kept_classes)

                if image_pred_boxes:
                    all_pb = np.concatenate(image_pred_boxes, axis=0)
                    all_ps = np.concatenate(image_pred_scores, axis=0)
                    all_pl = np.concatenate(image_pred_labels, axis=0)

                    # Per-class NMS
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

                all_gt_boxes.append(gt_boxes[b].cpu().numpy())
                all_gt_labels.append(gt_labels[b].cpu().numpy())
                n_frames += 1

            if n_frames % 500 == 0:
                elapsed = time.time() - t0
                logger.info(f"  ... {n_frames} frames in {elapsed:.0f}s ({n_frames/elapsed:.1f}/s)")

    return {
        'boxes': all_pred_boxes,
        'scores': all_pred_scores,
        'labels': all_pred_labels,
        'gt_boxes': all_gt_boxes,
        'gt_labels': all_gt_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to mtl_v3.x checkpoint')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--score-thresh', type=float, default=0.01)
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit frames for quick eval')
    parser.add_argument('--num-anchors', type=int, default=16, choices=[8, 16],
                        help='Must match checkpoint training config')
    parser.add_argument('--output', type=str, default='/tmp/mvit_eval_results.json')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    global logger
    logger = logging.getLogger('eval_mvit_mAP')

    # Override anchor config to match checkpoint
    global NUM_ANCHORS, _ANCHOR_SPECS
    if args.num_anchors == 8:
        mtl_mod.NUM_ANCHORS = 8
        mtl_mod._ANCHOR_SPECS = mtl_mod._ANCHOR_SPECS_8
    else:
        mtl_mod.NUM_ANCHORS = 16
        mtl_mod._ANCHOR_SPECS = mtl_mod._ANCHOR_SPECS_16
    NUM_ANCHORS = mtl_mod.NUM_ANCHORS
    _ANCHOR_SPECS = mtl_mod._ANCHOR_SPECS
    logger.info(f"Anchor config: {args.num_anchors} anchors, {len(_ANCHOR_SPECS)} specs")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Build model with matching anchor count
    model = MTLMViTModel(num_act_classes=75, num_det_classes=24, num_psr_components=11,
                         num_anchors=args.num_anchors)
    expand_conv_proj_to_9ch(model)

    # Load state dict
    sd = ckpt['model_state_dict']
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('m.'):
            new_sd[k[2:]] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
    model = model.to(DEVICE)
    model.eval()
    logger.info("Model loaded.")

    # Build dataset
    ds = IndustRealMultiTaskDataset(
        split=args.split,
        img_size=(640, 360),
        augment=False,
        sequence_mode=False,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_eval, num_workers=0)
    logger.info(f"Eval dataset: {len(ds)} frames, {len(loader)} batches")

    # Run inference
    logger.info(f"Running inference (score_thresh={args.score_thresh})...")
    results = get_decoded_predictions(
        model, loader,
        max_frames=args.max_frames,
        score_thresh=args.score_thresh,
    )

    # Compute mAP
    logger.info("Computing mAP@0.5...")
    has_gt = [len(gb) > 0 for gb in results['gt_boxes']]
    af_boxes = [results['boxes'][i] for i in range(len(has_gt)) if has_gt[i]]
    af_scores = [results['scores'][i] for i in range(len(has_gt)) if has_gt[i]]
    af_labels = [results['labels'][i] for i in range(len(has_gt)) if has_gt[i]]
    af_gtb = [results['gt_boxes'][i] for i in range(len(has_gt)) if has_gt[i]]
    af_gtl = [results['gt_labels'][i] for i in range(len(has_gt)) if has_gt[i]]

    logger.info(f"  Annotated frames: {len(af_gtb)}/{len(results['gt_boxes'])}")
    logger.info(f"  Total predictions: {sum(len(pb) for pb in results['boxes'])}")
    logger.info(f"  Total GT boxes: {sum(len(gb) for gb in results['gt_boxes'])}")

    af_result = compute_ap_per_class(af_boxes, af_scores, af_labels, af_gtb, af_gtl, iou_thresh=0.5)
    logger.info(f"  mAP@0.5 (annotated-frames): {af_result['mAP']:.4f}")

    ev_result = compute_ap_per_class_all_frames(
        results['boxes'], results['scores'], results['labels'],
        results['gt_boxes'], results['gt_labels'],
        iou_thresh=0.5,
    )
    logger.info(f"  mAP@0.5 (entire-video): {ev_result['mAP']:.4f}")

    logger.info("Per-class AP@0.5 (annotated-frames):")
    for c in sorted(af_result['per_class_ap'].keys()):
        ap = af_result['per_class_ap'][c]
        logger.info(f"    Class {c}: AP={ap:.4f}")

    output = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'n_frames': len(results['gt_boxes']),
        'n_gt_boxes': sum(len(gb) for gb in results['gt_boxes']),
        'n_preds': sum(len(pb) for pb in results['boxes']),
        'mAP50_annotated_frames': float(af_result['mAP']),
        'mAP50_entire_video': float(ev_result['mAP']),
        'per_class_AP50': {str(k): float(v) for k, v in af_result['per_class_ap'].items()},
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved: {args.output}")


if __name__ == '__main__':
    main()
