"""
eval_tta.py — Test-Time Augmentation (TTA) Wrapper for POPW Multi-Task Model
=============================================================================
Applies multi-scale {0.8, 1.0, 1.2} + horizontal flip at inference time,
merges predictions with Soft-NMS, and computes detection mAP metrics.

Expected: +0.02-0.07 mAP over single-pass inference (Opus Q50).
Zero training required. ~2-3 hours on one GPU.

Usage:
    python3 src/evaluation/eval_tta.py \\
        --ckpt src/runs/rf_stages/checkpoints/epoch_11.pth \\
        --batch_size 2

Reference (inference loop):
    src/evaluation/evaluate.py:3158-3425  (evaluate_all loop body)
    src/evaluation/evaluate.py:1344-1363  (nms_numpy)
    src/evaluation/soft_nms.py:37-114   (Soft-NMS implementation)
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup (identical to evaluate.py) ────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent  # src/
for _sub in ["models", "training", "evaluation", "data", str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src import config as C
from src.evaluation.evaluate import (
    decode_boxes,
    compute_det_metrics_extended,
    _prepare_images,
)
from src.evaluation.soft_nms import soft_nms
from src.models.model import POPWMultiTaskModel
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

logger = logging.getLogger("eval_tta")

# ── Constants ────────────────────────────────────────────────────────────
_TTA_SCALES = [0.8, 1.0, 1.2]
_TTA_FLIPS = [False, True]  # no flip, horizontal flip
_OUTPUT_PATH = Path("src/runs/rf_stages/checkpoints/eval_tta_results.json")


def _build_model(
    ckpt_path: str,
    device: torch.device,
) -> POPWMultiTaskModel:
    """Load the POPWMultiTaskModel from a checkpoint file.

    Args:
        ckpt_path: Path to the .pth checkpoint.
        device: Target device (cuda or cpu).

    Returns:
        model in eval() mode on the target device.
    """
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})

    backbone_type = cfg.get("backbone_type", "convnext_tiny")
    model = (
        POPWMultiTaskModel(
            pretrained=False,
            backbone_type=str(backbone_type),
            use_headpose_film=False,
            use_hand_film=False,
            use_videomae=False,
            train_pose=False,
            use_backbone_checkpoint=False,
        )
        .to(device)
        .eval()
    )

    result = model.load_state_dict(state["model"], strict=False)
    if result.missing_keys:
        logger.warning("Missing keys: %s", result.missing_keys)
    if result.unexpected_keys:
        logger.warning("Unexpected keys: %s", result.unexpected_keys)

    return model


def _build_val_loader(
    batch_size: int,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Build the validation DataLoader with no augmentation."""
    val_ds = IndustRealMultiTaskDataset(
        split="val",
        img_size=(C.IMG_WIDTH, C.IMG_HEIGHT),
    )
    return torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def _tta_resize(
    image: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Resize a batched image tensor [B, 3, H, W] by a scale factor.

    Uses bilinear interpolation for downscaling (scale < 1) and upscaling
    (scale > 1). When scale == 1.0, returns the input unchanged.
    """
    if abs(scale - 1.0) < 1e-6:
        return image
    B, C, H, W = image.shape
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    return F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _tta_horizontal_flip(image: torch.Tensor) -> torch.Tensor:
    """Flip a batched image tensor [B, 3, H, W] horizontally."""
    return torch.flip(image, dims=[3])


def _decode_batch_predictions(
    cls_preds: torch.Tensor,
    reg_preds: torch.Tensor,
    anchors_np: np.ndarray,
    img_w: int,
    img_h: int,
    score_thresh: float,
    nms_iou: float,
    num_classes: int,
    max_per_image: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Decode model outputs into per-image detection lists.

    This is adapted from evaluate.py:3365-3415 but uses Soft-NMS instead of
    standard NMS (controlled by the ``nms_iou`` parameter which is passed to
    the Soft-NMS sigma via a heuristic mapping; see build below).

    Returns:
        (boxes_list, scores_list, labels_list) each of length B,
        where each element is a numpy array of shape [M, 4], [M], [M].
    """
    cls_sigmoid = torch.sigmoid(cls_preds)  # [B, N, num_classes]
    B = cls_preds.shape[0]
    boxes_list, scores_list, labels_list = [], [], []

    for i in range(B):
        scores_i = cls_sigmoid[i]
        max_scores = scores_i.max(dim=1).values
        keep_mask = max_scores > score_thresh

        if max_per_image > 0 and keep_mask.sum().item() > max_per_image:
            topk_idx = torch.topk(max_scores, k=max_per_image, largest=True, sorted=False).indices
            topk_mask = torch.zeros_like(keep_mask)
            topk_mask[topk_idx] = True
            keep_mask = keep_mask & topk_mask

        if keep_mask.sum().item() == 0:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            scores_list.append(np.zeros(0, dtype=np.float32))
            labels_list.append(np.zeros(0, dtype=np.int64))
            continue

        keep_np = keep_mask.cpu().numpy()
        kept_cls = scores_i[keep_mask].float().cpu().numpy()
        kept_reg = reg_preds[i][keep_mask].cpu().numpy()
        kept_anc = anchors_np[keep_np]

        ms = kept_cls.max(axis=1)
        ml = kept_cls.argmax(axis=1)
        pb = decode_boxes(kept_anc, kept_reg)
        pb[:, 0] = np.clip(pb[:, 0], 0, img_w)
        pb[:, 1] = np.clip(pb[:, 1], 0, img_h)
        pb[:, 2] = np.clip(pb[:, 2], 0, img_w)
        pb[:, 3] = np.clip(pb[:, 3], 0, img_h)

        fb, fs, fl = [], [], []
        for c in range(num_classes):
            cm = ml == c
            if cm.sum() == 0:
                continue
            # Use Soft-NMS per class instead of standard NMS.
            sigma = max(0.3, nms_iou * 0.7)  # heuristic: map IoU to sigma
            keep_snms = soft_nms(pb[cm], ms[cm], sigma=sigma, score_thresh=0.001)
            if len(keep_snms) == 0:
                continue
            fb.append(pb[cm][keep_snms])
            fs.append(ms[cm][keep_snms])
            fl.append(np.full(len(keep_snms), c, dtype=np.int64))

        if fb:
            boxes_list.append(np.concatenate(fb))
            scores_list.append(np.concatenate(fs))
            labels_list.append(np.concatenate(fl))
        else:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            scores_list.append(np.zeros(0, dtype=np.float32))
            labels_list.append(np.zeros(0, dtype=np.int64))

    return boxes_list, scores_list, labels_list


def _rescale_boxes_to_original(
    boxes_list: List[np.ndarray],
    model_w: int,
    model_h: int,
    orig_w: int,
    orig_h: int,
    is_flipped: bool,
) -> List[np.ndarray]:
    """Rescale boxes from model coordinate space back to original image space.

    Args:
        boxes_list: Per-image list of [M, 4] boxes in model-space coordinates.
        model_w, model_h: Dimensions of the image the model processed.
        orig_w, orig_h: Original image dimensions (typically 1280, 720).
        is_flipped: Whether the model processed a horizontally flipped image.

    Returns:
        boxes_list with coordinates in the original (orig_w, orig_h) space.
    """
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    rescaled = []
    for boxes in boxes_list:
        if boxes.shape[0] == 0:
            rescaled.append(boxes)
            continue
        boxes = boxes.copy()
        if is_flipped:
            # Unflip: the model saw a flipped image of size (model_w, model_h).
            boxes[:, [0, 2]] = model_w - boxes[:, [2, 0]]
        # Rescale to original coordinates.
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
        # Clip to original bounds.
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
        rescaled.append(boxes)
    return rescaled


def _merge_tta_predictions(
    all_boxes: List[List[np.ndarray]],
    all_scores: List[List[np.ndarray]],
    all_labels: List[List[np.ndarray]],
    num_images: int,
    num_classes: int,
    soft_nms_sigma: float = 0.5,
    final_score_thresh: float = 0.001,
    max_per_image: int = 300,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Merge predictions across TTA augmentations using Soft-NMS.

    For each image, concatenate predictions from all TTA variants, then apply
    per-class Soft-NMS to produce the final merged detection set.

    Args:
        all_boxes:  [num_augs] list of per-image box lists each of length num_images.
        all_scores: [num_augs] list of per-image score lists.
        all_labels: [num_augs] list of per-image label lists.
        num_images: number of images in the batch / dataset.
        num_classes: number of detection classes (24 for ASD).
        soft_nms_sigma: Soft-NMS sigma parameter (Gaussian spread).
        final_score_thresh: minimum score to keep after merging.
        max_per_image: max detections per image after merging.

    Returns:
        merged_boxes, merged_scores, merged_labels: per-image lists.
    """
    merged_boxes: List[np.ndarray] = []
    merged_scores: List[np.ndarray] = []
    merged_labels: List[np.ndarray] = []

    for img_idx in range(num_images):
        # Concatenate all TTA predictions for this image.
        non_empty_boxes = [aug[img_idx] for aug in all_boxes if aug[img_idx].shape[0] > 0]
        non_empty_scores = [aug[img_idx] for aug in all_scores if aug[img_idx].shape[0] > 0]
        non_empty_labels = [aug[img_idx] for aug in all_labels if aug[img_idx].shape[0] > 0]

        if not non_empty_boxes:
            merged_boxes.append(np.zeros((0, 4), dtype=np.float32))
            merged_scores.append(np.zeros(0, dtype=np.float32))
            merged_labels.append(np.zeros(0, dtype=np.int64))
            continue

        cat_boxes = np.concatenate(non_empty_boxes, axis=0)
        cat_scores = np.concatenate(non_empty_scores, axis=0)
        cat_labels = np.concatenate(non_empty_labels, axis=0)

        # Apply Soft-NMS per class.
        final_boxes, final_scores, final_labels = [], [], []
        for c in range(num_classes):
            cm = cat_labels == c
            if cm.sum() == 0:
                continue
            keep = soft_nms(
                cat_boxes[cm],
                cat_scores[cm],
                sigma=soft_nms_sigma,
                score_thresh=final_score_thresh,
            )
            if len(keep) == 0:
                continue
            final_boxes.append(cat_boxes[cm][keep])
            final_scores.append(cat_scores[cm][keep])
            final_labels.append(np.full(len(keep), c, dtype=np.int64))

        if final_boxes:
            img_boxes = np.concatenate(final_boxes)
            img_scores = np.concatenate(final_scores)
            img_labels = np.concatenate(final_labels)
            # Cap at max_per_image.
            if max_per_image > 0 and len(img_scores) > max_per_image:
                topk = np.argsort(img_scores)[::-1][:max_per_image]
                img_boxes = img_boxes[topk]
                img_scores = img_scores[topk]
                img_labels = img_labels[topk]
        else:
            img_boxes = np.zeros((0, 4), dtype=np.float32)
            img_scores = np.zeros(0, dtype=np.float32)
            img_labels = np.zeros(0, dtype=np.int64)

        merged_boxes.append(img_boxes)
        merged_scores.append(img_scores)
        merged_labels.append(img_labels)

    return merged_boxes, merged_scores, merged_labels


def run_tta_eval(
    ckpt_path: str,
    batch_size: int,
    max_batches: int = 0,
    device: torch.device = None,
) -> Dict[str, float]:
    """Run TTA evaluation and return detection metrics.

    Args:
        ckpt_path: Path to the epoch-11 checkpoint.
        batch_size: Batch size for inference.
        max_batches: Cap on number of batches (0 = unlimited).
        device: Target device.

    Returns:
        dict with det_mAP50, det_mAP_50_95, det_per_class_ap, and metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────
    model = _build_model(ckpt_path, device)

    # ── Build val loader ─────────────────────────────────────────────────
    loader = _build_val_loader(batch_size=batch_size)

    # ── Accumulators ─────────────────────────────────────────────────────
    gt_boxes_global: List[np.ndarray] = []
    gt_labels_global: List[np.ndarray] = []

    logger.info(
        "Starting TTA eval with scales=%s, flips=%s, batch_size=%d, device=%s",
        _TTA_SCALES,
        _TTA_FLIPS,
        batch_size,
        device,
    )

    for bi, (images, targets) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        B = images.shape[0]
        logger.info("Batch %d: %d images", bi, B)

        # Ground truth for all images in this batch (collected once, used
        # for all TTA variants).
        detection_list = targets["detection"]
        for img_idx in range(B):
            gt_boxes_global.append(detection_list[img_idx]["boxes"].cpu().numpy())
            gt_labels_global.append(detection_list[img_idx]["labels"].cpu().numpy())

        # ── Run all TTA variants ────────────────────────────────────────
        # all_preds[aug_idx] = (boxes_list, scores_list, labels_list)
        all_preds: List = []  # each element is a tuple of 3 lists

        with torch.no_grad():
            for scale in _TTA_SCALES:
                for do_flip in _TTA_FLIPS:
                    # Prepare image at the current scale.
                    img_input = _prepare_images(images, device)
                    if abs(scale - 1.0) > 1e-6:
                        img_input = _tta_resize(img_input, scale)
                    if do_flip:
                        img_input = _tta_horizontal_flip(img_input)

                    # Forward pass.
                    outputs = model(img_input, video_ids=None, clip_rgb=None)

                    # Decode boxes in model coordinate space.
                    anchors_np = outputs["anchors"].cpu().numpy()
                    cls_preds = outputs["cls_preds"]
                    reg_preds = outputs["reg_preds"]

                    boxes_list, scores_list, labels_list = _decode_batch_predictions(
                        cls_preds=cls_preds,
                        reg_preds=reg_preds,
                        anchors_np=anchors_np,
                        img_w=img_input.shape[3],
                        img_h=img_input.shape[2],
                        score_thresh=float(getattr(C, "DET_EVAL_SCORE_THRESH", 0.001)),
                        nms_iou=float(getattr(C, "DET_EVAL_NMS_IOU_THRESH", 0.5)),
                        num_classes=C.NUM_DET_CLASSES,
                        max_per_image=int(getattr(C, "DET_EVAL_MAX_PER_IMAGE", 300)),
                    )

                    # Rescale boxes from model space to original (1280, 720).
                    boxes_list = _rescale_boxes_to_original(
                        boxes_list,
                        model_w=img_input.shape[3],
                        model_h=img_input.shape[2],
                        orig_w=C.IMG_WIDTH,
                        orig_h=C.IMG_HEIGHT,
                        is_flipped=do_flip,
                    )

                    all_preds.append((boxes_list, scores_list, labels_list))

                    del outputs, cls_preds, reg_preds
                    gc.collect()

        # ── Merge TTA predictions for the batch ─────────────────────────
        aug_boxes = [p[0] for p in all_preds]  # [num_augs, B] each
        aug_scores = [p[1] for p in all_preds]
        aug_labels = [p[2] for p in all_preds]

        merged_boxes, merged_scores, merged_labels = _merge_tta_predictions(
            aug_boxes,
            aug_scores,
            aug_labels,
            num_images=B,
            num_classes=C.NUM_DET_CLASSES,
            soft_nms_sigma=0.5,
            final_score_thresh=float(getattr(C, "DET_EVAL_SCORE_THRESH", 0.001)),
            max_per_image=int(getattr(C, "DET_EVAL_MAX_PER_IMAGE", 300)),
        )

        # ── Store per-image predictions for mAP computation ─────────────
        # We accumulate into the evaluate.py-compatible collectors
        # so we can call compute_det_metrics_extended at the end.
        if bi == 0:
            # Initialize global accumulators on the first batch.
            global_dp_boxes = merged_boxes
            global_dp_scores = merged_scores
            global_dp_labels = merged_labels
        else:
            global_dp_boxes.extend(merged_boxes)
            global_dp_scores.extend(merged_scores)
            global_dp_labels.extend(merged_labels)

        # Cleanup
        del images, targets, img_input
        gc.collect()

        if bi % 10 == 0 and torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
            logger.info("  GPU alloc: %.2f GB", alloc_gb)

    # ── Compute detection metrics ───────────────────────────────────────
    logger.info("Computing detection metrics (mAP@0.5, mAP@[0.5:0.95])...")
    det_metrics = compute_det_metrics_extended(
        global_dp_boxes,
        global_dp_scores,
        global_dp_labels,
        gt_boxes_global,
        gt_labels_global,
        num_classes=C.NUM_DET_CLASSES,
    )

    # Add per-class mAP50_pc (present-class average).
    mAP_per_thresh_pc = det_metrics.get("mAP_per_thresh_pc", {})
    det_metrics["det_mAP50_pc"] = mAP_per_thresh_pc.get(0.5, 0.0)

    # Add metadata about the TTA configuration used.
    det_metrics["_tta_scales"] = _TTA_SCALES
    det_metrics["_tta_flips"] = [f"flip={f}" for f in _TTA_FLIPS]
    det_metrics["_tta_num_augs"] = len(_TTA_SCALES) * len(_TTA_FLIPS)
    det_metrics["_soft_nms_sigma"] = 0.5
    det_metrics["_checkpoint"] = str(ckpt_path)
    det_metrics["_num_images"] = len(global_dp_boxes)

    logger.info(
        "TTA results — mAP@0.5: %.4f  mAP@[0.5:0.95]: %.4f  mAP50_pc: %.4f",
        det_metrics.get("det_mAP50", 0.0),
        det_metrics.get("det_mAP_50_95", 0.0),
        det_metrics.get("det_mAP50_pc", 0.0),
    )

    return det_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="TTA eval: multi-scale + flip + Soft-NMS")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="src/runs/rf_stages/checkpoints/epoch_11.pth",
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Max batches to evaluate (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_OUTPUT_PATH),
        help="Output JSON path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(name)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    device = torch.device(args.device)
    metrics = run_tta_eval(
        ckpt_path=args.ckpt,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=device,
    )

    # Save results.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean serialisable dict.
    clean = {}
    for k, v in metrics.items():
        try:
            json.dumps(v)
            clean[k] = v
        except (TypeError, OverflowError):
            clean[k] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)

    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
