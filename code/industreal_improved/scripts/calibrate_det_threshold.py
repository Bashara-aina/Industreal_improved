#!/usr/bin/env python3
"""calibrate_det_threshold.py — detection score-threshold calibration (Task #262).

Sweeps the score threshold to find the optimal operating point for detection
mAP@0.5. Strategy: single inference pass saving top-K raw predictions, then
re-threshold / re-NMS in post-processing without re-inference.

Usage:
    python scripts/calibrate_det_threshold.py \\
        --checkpoint code/industreal_improved/src/runs/rf_stages/checkpoints/best.pth \\
        --save-dir runs/calib_run \\
        --max-images 5000

Output:
    optimal_threshold.json     — best global and per-class thresholds
    sweep_results.json         — full sweep table with mAP vs threshold
    per_class_thresholds.json  — per-class optimal thresholds (for inference override)

Reference: Opus Q6 / Task #262 — detection operating-point calibration.
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
for _p in [_SRC, _SRC.parent, _SRC / "models", _SRC / "data", _SRC / "evaluation"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import src.config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import POPWMultiTaskModel

# Imports from evaluation package
from evaluate import decode_boxes, nms_numpy, compute_ap_per_class

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes (Nx4, Mx4, xyxy format)."""
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xi1 = np.maximum(x11, x21.T)
    yi1 = np.maximum(y11, y21.T)
    xi2 = np.minimum(x12, x22.T)
    yi2 = np.minimum(y12, y22.T)
    inter = np.clip(xi2 - xi1, 0, None) * np.clip(yi2 - yi1, 0, None)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter
    return inter / np.clip(union, 1e-9, None)


# ---------------------------------------------------------------------------
# Model loading (mirrors full_eval_inprocess.load_model)
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, int, dict]:
    """Load POPWMultiTaskModel from checkpoint. Returns (model, epoch, config)."""
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})

    backbone_type = cfg.get("BACKBONE_TYPE", getattr(C, "BACKBONE_TYPE", "convnext_tiny"))
    use_hand_film = bool(cfg.get("USE_HAND_FILM", getattr(C, "USE_HAND_FILM", True)))
    use_headpose_film = bool(cfg.get("USE_HEADPOSE_FILM", getattr(C, "USE_HEADPOSE_FILM", False)))
    use_videomae = bool(cfg.get("USE_VIDEOMAE", getattr(C, "USE_VIDEOMAE", False)))
    train_pose = bool(cfg.get("TRAIN_HEAD_POSE", getattr(C, "TRAIN_HEAD_POSE", True)))

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type=backbone_type,
        use_hand_film=use_hand_film,
        use_headpose_film=use_headpose_film,
        use_videomae=use_videomae,
        train_pose=train_pose,
    ).to(device).eval()

    seq_len = cfg.get("PSR_SEQUENCE_LENGTH", getattr(C, "PSR_SEQUENCE_LENGTH", 1))
    model._seq_len = seq_len if cfg.get("USE_PSR_SEQUENCE_MODE", False) else 1

    result = model.load_state_dict(state["model"], strict=False)
    if result.missing_keys:
        print(f"  Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {len(result.unexpected_keys)} keys (harmless)")

    epoch = state.get("epoch", -1)
    print(f"  Epoch={epoch}, step={state.get('step', '?')}")
    return model, epoch, cfg


def prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize image batch from [0,255] uint8 to ImageNet-normalized float."""
    images_f = images.to(device).float()
    if images_f.max() > 1.0:
        images_f = images_f.div_(255.0)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (images_f - mean) / std


# ---------------------------------------------------------------------------
# Threshold sweep post-processing
# ---------------------------------------------------------------------------

def compute_map_for_threshold(
    detections: list[dict],
    gt_data: list[dict],
    score_thresh: float,
    nms_thresh: float = 0.5,
    num_classes: int = 24,
) -> dict:
    """Re-filter decoded detections at `score_thresh`, NMS, then compute mAP.

    Args:
        detections: list of dicts with keys 'boxes', 'scores', 'labels' per image.
        gt_data: list of dicts with keys 'boxes', 'labels' per image.
        score_thresh: candidate score threshold.
        nms_thresh: IoU threshold for NMS.
        num_classes: number of detection classes.

    Returns:
        dict with 'mAP50', 'mAP75', 'per_class_ap50', 'n_total_preds'.
    """
    filtered_boxes: list[np.ndarray] = []
    filtered_scores: list[np.ndarray] = []
    filtered_labels: list[np.ndarray] = []

    total_preds = 0
    for det in detections:
        boxes, scores, labels = det["boxes"], det["scores"], det["labels"]

        # Score threshold
        keep = scores >= score_thresh
        if not keep.any():
            filtered_boxes.append(np.zeros((0, 4), dtype=np.float32))
            filtered_scores.append(np.zeros(0, dtype=np.float32))
            filtered_labels.append(np.zeros(0, dtype=np.int64))
            continue

        pb, ps, pl = boxes[keep], scores[keep], labels[keep]

        # Class-wise NMS
        fb, fs, fl = [], [], []
        for c in range(num_classes):
            cm = pl == c
            if not cm.any():
                continue
            nk = nms_numpy(pb[cm], ps[cm], nms_thresh)
            fb.append(pb[cm][nk])
            fs.append(ps[cm][nk])
            fl.append(np.full(len(nk), c, dtype=np.int64))

        if fb:
            filtered_boxes.append(np.concatenate(fb))
            filtered_scores.append(np.concatenate(fs))
            filtered_labels.append(np.concatenate(fl))
            total_preds += len(filtered_boxes[-1])
        else:
            filtered_boxes.append(np.zeros((0, 4), dtype=np.float32))
            filtered_scores.append(np.zeros(0, dtype=np.float32))
            filtered_labels.append(np.zeros(0, dtype=np.int64))

    gt_boxes_list = [g["boxes"] for g in gt_data]
    gt_labels_list = [g["labels"] for g in gt_data]

    # mAP@0.5
    ap50 = compute_ap_per_class(
        filtered_boxes, filtered_scores, filtered_labels,
        gt_boxes_list, gt_labels_list,
        iou_thresh=0.5,
        num_classes=num_classes,
    )
    # mAP@0.75 (surrogate for mAP@0.5-95)
    ap75 = compute_ap_per_class(
        filtered_boxes, filtered_scores, filtered_labels,
        gt_boxes_list, gt_labels_list,
        iou_thresh=0.75,
        num_classes=num_classes,
    )

    # Compute F1@threshold: count TP/FP/FN by matching pred→GT at IoU=0.5
    tp, fp, fn = 0, 0, 0
    for pi in range(len(filtered_boxes)):
        pb, ps, pl = filtered_boxes[pi], filtered_scores[pi], filtered_labels[pi]
        gb, gl = gt_boxes_list[pi], gt_labels_list[pi]
        if len(pb) == 0:
            fn += len(gb)
            continue
        if len(gb) == 0:
            fp += len(pb)
            continue
        # Compute IoU matrix [N_pred, N_gt]
        ious = box_iou_numpy(pb, gb)
        matched_gt = set()
        for pi2 in range(len(pb)):
            if len(ious[pi2]) == 0:
                continue
            best_gi = int(ious[pi2].argmax())
            if best_gi not in matched_gt and ious[pi2, best_gi] >= 0.5 and pl[pi2] == gl[best_gi]:
                matched_gt.add(best_gi)
                tp += 1
            else:
                fp += 1
        fn += len(gb) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "mAP50": float(ap50["mAP"]),
        "mAP75": float(ap75["mAP"]),
        "per_class_ap50": {str(k): float(v) for k, v in ap50["per_class_ap"].items()},
        "n_total_preds": total_preds,
        "precision": float(precision),
        "recall": float(recall),
        "f1_at_threshold": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def find_optimal_per_class(
    detections: list[dict],
    gt_data: list[dict],
    candidate_thresholds: np.ndarray,
    nms_thresh: float = 0.5,
    num_classes: int = 24,
) -> dict[int, float]:
    """Find per-class optimal thresholds that maximize each class's AP@0.5."""
    nms_thresh = float(getattr(C, "DET_EVAL_NMS_IOU_THRESH", 0.5))

    # For each class, find the threshold that maximizes AP@0.5 for that class
    best_thresh: dict[int, float] = {}
    best_ap: dict[int, float] = {}

    for t in candidate_thresholds:
        result = compute_map_for_threshold(
            detections, gt_data, score_thresh=t,
            nms_thresh=nms_thresh, num_classes=num_classes,
        )
        for cls_str, ap_val in result["per_class_ap50"].items():
            cls_int = int(cls_str)
            if cls_int not in best_ap or ap_val > best_ap[cls_int]:
                best_ap[cls_int] = ap_val
                best_thresh[cls_int] = float(t)

    return best_thresh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detection score-threshold calibration (Task #262)"
    )
    parser.add_argument(
        "--checkpoint",
        default="code/industreal_improved/src/runs/rf_stages/checkpoints/best.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--save-dir",
        default="code/industreal_improved/src/runs/rf_stages/checkpoints/calib_run",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--max-images", type=int, default=5000,
        help="Max images to process (default: 5000, ~13% of val set)",
    )
    parser.add_argument(
        "--top-k", type=int, default=2000,
        help="Save top-K anchor predictions per image (default: 2000)",
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=0.5,
        help="NMS IoU threshold (default: 0.5, matches config)",
    )
    parser.add_argument(
        "--per-class", action="store_true",
        help="Also compute per-class optimal thresholds (slower, ~5x)",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    # ── Candidate thresholds (log-scale) ─────────────────────────────────
    # Dense near 0.001 (current config value), sparse at extremes
    thresholds = np.sort(np.unique(np.concatenate([
        np.logspace(np.log10(0.0003), np.log10(0.5), num=25),
        [0.001],  # ensure current config value is included
    ])))
    print(f"Candidate thresholds: {len(thresholds)} values from {thresholds[0]:.5f} to {thresholds[-1]:.3f}")

    # ── Load model ───────────────────────────────────────────────────────
    model, epoch, cfg = load_model(args.checkpoint, device)

    # ── Data ─────────────────────────────────────────────────────────────
    print("Loading validation dataset...")
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, num_workers=0,
        collate_fn=collate_fn, shuffle=False,
    )
    print(f"  Dataset size: {len(val_ds)} frames")

    # ── Inference pass: save raw detections ─────────────────────────────
    print("\nRunning inference pass (saving raw predictions)...")
    detections: list[dict] = []   # per-image filtered predictions
    gt_data: list[dict] = []      # per-image ground truth
    n_processed = 0
    _anchors_np = None
    t_start = time.time()

    with torch.no_grad():
        for bi, (images, targets) in enumerate(val_loader):
            if bi >= args.max_images:
                break
            if images.shape[0] == 0:
                continue

            images = prepare_images(images, device)
            B = images.shape[0]
            outputs = model(images)

            # Cache anchors (same for all images)
            if _anchors_np is None:
                _anchors_np = outputs["anchors"].cpu().numpy()

            cls_sigmoid = torch.sigmoid(outputs["cls_preds"])  # [B, N, 24]
            detection_list = targets["detection"]

            for i in range(B):
                scores_i = cls_sigmoid[i].cpu().numpy()  # [N, 24]
                reg_i = outputs["reg_preds"][i].cpu().numpy()  # [N, 4]

                # Max score per anchor
                max_scores = scores_i.max(axis=1)
                argmax_classes = scores_i.argmax(axis=1)

                # Top-K filter (keep enough for recall at low thresholds)
                topk_idx = np.argsort(max_scores)[::-1][:args.top_k]
                topk_scores = max_scores[topk_idx]
                topk_classes = argmax_classes[topk_idx]
                topk_reg = reg_i[topk_idx]
                topk_anc = _anchors_np[topk_idx]

                # Decode boxes once (save CPU work later)
                boxes = decode_boxes(topk_anc, topk_reg)
                boxes[:, 0] = np.clip(boxes[:, 0], 0, C.IMG_WIDTH)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, C.IMG_HEIGHT)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, C.IMG_WIDTH)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, C.IMG_HEIGHT)

                detections.append({
                    "boxes": boxes,
                    "scores": topk_scores,
                    "labels": topk_classes.astype(np.int64),
                })

                # GT
                gt_boxes = detection_list[i]["boxes"].cpu().numpy()
                gt_labels = detection_list[i]["labels"].cpu().numpy()
                gt_data.append({"boxes": gt_boxes, "labels": gt_labels})

                n_processed += 1

            # Cleanup
            del images, outputs, cls_sigmoid
            gc.collect()

            if (bi + 1) % 500 == 0:
                elapsed = time.time() - t_start
                print(f"  processed {bi+1} batches ({n_processed} images) in {elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"\nInference complete: {n_processed} images in {elapsed:.0f}s ({elapsed/max(n_processed,1):.3f}s/img)")

    # Count total GT boxes
    total_gt = sum(len(g["boxes"]) for g in gt_data)
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total saved predictions: {sum(len(d['boxes']) for d in detections)}")

    # ── Threshold sweep ──────────────────────────────────────────────────
    print(f"\nSweeping {len(thresholds)} thresholds...")
    sweep_results = []
    best_mAP50 = -1.0
    best_threshold = thresholds[0]
    best_result = None

    t_sweep = time.time()
    for ti, t in enumerate(thresholds):
        result = compute_map_for_threshold(
            detections, gt_data,
            score_thresh=t,
            nms_thresh=args.nms_thresh,
            num_classes=C.NUM_DET_CLASSES,
        )
        result["threshold"] = float(t)
        sweep_results.append(result)

        if result["mAP50"] > best_mAP50:
            best_mAP50 = result["mAP50"]
            best_threshold = t
            best_result = result

        if (ti + 1) % 5 == 0 or ti == len(thresholds) - 1:
            print(f"  thresh={t:.5f}  mAP50={result['mAP50']:.4f}  F1={result['f1_at_threshold']:.4f}  P={result['precision']:.3f}  R={result['recall']:.3f}  TP/FP/FN={result['tp']}/{result['fp']}/{result['fn']}")

    sweep_elapsed = time.time() - t_sweep
    print(f"\nSweep complete in {sweep_elapsed:.0f}s")

    # ── Select optimal threshold: maximize F1@threshold, tiebreak on pred count ──
    # NOTE: mAP50 is rank-based and generally invariant across thresholds.
    # F1@threshold (precision/recall at operating point) is the real metric to optimize.
    best_f1_result = max(sweep_results, key=lambda r: (r["f1_at_threshold"], -r["n_total_preds"]))
    best_f1_threshold = best_f1_result["threshold"]
    print(f"\n  [F1-based] Optimal threshold: {best_f1_threshold:.5f}  F1={best_f1_result['f1_at_threshold']:.4f}  "
          f"P={best_f1_result['precision']:.3f}  R={best_f1_result['recall']:.3f}  preds={best_f1_result['n_total_preds']}")

    # ── Per-class optimal thresholds (optional) ──────────────────────────
    per_class_thresholds = None
    if args.per_class:
        print("\nComputing per-class optimal thresholds...")
        per_class_thresholds = find_optimal_per_class(
            detections, gt_data, thresholds,
            nms_thresh=args.nms_thresh,
            num_classes=C.NUM_DET_CLASSES,
        )
        n_found = sum(1 for v in per_class_thresholds.values() if v is not None)
        print(f"  Found optimal thresholds for {n_found}/{C.NUM_DET_CLASSES} classes")

    # ── Print headline ──────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("DETECTION THRESHOLD CALIBRATION (Task #262)")
    print("=" * 64)
    print(f"  Checkpoint:         {args.checkpoint}")
    print(f"  Epoch:              {epoch}")
    print(f"  Images evaluated:   {n_processed}")
    print(f"  Total GT boxes:     {total_gt}")
    print(f"  NMS IoU threshold:  {args.nms_thresh}")
    print()
    print(f"  Current config threshold (DET_EVAL_SCORE_THRESH): {C.DET_EVAL_SCORE_THRESH}")
    print(f"  Current mAP@0.5 at config threshold:              {sweep_results[list(thresholds).index(C.DET_EVAL_SCORE_THRESH) if C.DET_EVAL_SCORE_THRESH in thresholds else 0]['mAP50']:.4f}")
    print()
    print(f"  ⚠  NOTE: mAP@0.5 is rank-based and invariant across thresholds.")
    print(f"     F1@threshold is the real metric for operating-point selection.")
    print()
    print(f"  BEST (by mAP@0.5): threshold={best_threshold:.5f}  mAP50={best_mAP50:.4f}")
    print(f"  BEST (by F1):      threshold={best_f1_threshold:.5f}  F1={best_f1_result['f1_at_threshold']:.4f}  "
          f"P={best_f1_result['precision']:.3f}  R={best_f1_result['recall']:.3f}  preds={best_f1_result['n_total_preds']}")
    print()
    print("  Sweep summary (sorted by F1@threshold):")
    sorted_by_f1 = sorted(sweep_results, key=lambda r: (r["f1_at_threshold"], -r["n_total_preds"]), reverse=True)
    for rank, sr in enumerate(sorted_by_f1[:5]):
        marker = " *" if abs(sr["threshold"] - best_f1_threshold) < 1e-6 else ""
        print(f"    {rank+1}. thresh={sr['threshold']:.5f}  F1={sr['f1_at_threshold']:.4f}  "
              f"mAP50={sr['mAP50']:.4f}  P={sr['precision']:.3f}  R={sr['recall']:.3f}  "
              f"preds={sr['n_total_preds']}{marker}")
    print()
    print(f"  Config recommendation: DET_EVAL_SCORE_THRESH = {best_f1_threshold:.5f}")
    print("=" * 64)

    # ── Save ─────────────────────────────────────────────────────────────
    summary = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "n_images": n_processed,
        "n_total_gt_boxes": total_gt,
        "nms_threshold": args.nms_thresh,
        "candidate_thresholds": thresholds.tolist(),
        "config_current_threshold": C.DET_EVAL_SCORE_THRESH,
        "config_current_mAP50": float(sweep_results[list(thresholds).index(C.DET_EVAL_SCORE_THRESH) if C.DET_EVAL_SCORE_THRESH in thresholds else 0]["mAP50"]),
        "optimal_global_threshold": float(best_threshold),
        "optimal_mAP50": best_mAP50,
        "optimal_mAP75": float(best_result["mAP75"]) if best_result else 0.0,
        "optimal_f1_threshold": float(best_f1_threshold),
        "optimal_f1": float(best_f1_result["f1_at_threshold"]),
        "optimal_precision": float(best_f1_result["precision"]),
        "optimal_recall": float(best_f1_result["recall"]),
        "n_preds_at_optimal_f1": best_f1_result["n_total_preds"],
        "note": "mAP@0.5 is rank-based and invariant across thresholds. F1@threshold is the real metric for operating-point selection.",
        "selection_criterion": "F1@threshold (tiebreak: lowest prediction count)",
        "sweep_results": sweep_results,
    }

    # Save optimal threshold (F1-based)
    opt_path = save_dir / "optimal_threshold.json"
    with open(opt_path, "w") as f:
        json.dump({
            "optimal_threshold": float(best_f1_threshold),
            "optimal_criterion": "F1@threshold",
            "config_current": C.DET_EVAL_SCORE_THRESH,
            "config_current_mAP50": float(sweep_results[list(thresholds).index(C.DET_EVAL_SCORE_THRESH) if C.DET_EVAL_SCORE_THRESH in thresholds else 0]["mAP50"]),
            "mAP50_at_optimal": float(best_f1_result["mAP50"]),
            "mAP75_at_optimal": float(best_f1_result["mAP75"]),
            "f1_at_optimal": float(best_f1_result["f1_at_threshold"]),
            "precision_at_optimal": float(best_f1_result["precision"]),
            "recall_at_optimal": float(best_f1_result["recall"]),
            "n_preds_at_optimal": best_f1_result["n_total_preds"],
            "n_images": n_processed,
            "n_total_gt_boxes": total_gt,
            "checkpoint": str(args.checkpoint),
            "note": "mAP@0.5 is rank-based and invariant across thresholds. F1@threshold is the real metric. Optimal threshold maximizes F1 then minimises pred count.",
            "recommendation": f"Set DET_EVAL_SCORE_THRESH = {best_f1_threshold:.5f}",
        }, f, indent=2)
    print(f"\nOptimal threshold saved to {opt_path}")

    # Save full sweep
    sweep_path = save_dir / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full sweep results saved to {sweep_path}")

    # Save per-class thresholds
    if per_class_thresholds:
        pc_path = save_dir / "per_class_thresholds.json"
        with open(pc_path, "w") as f:
            json.dump({
                "optimal_thresholds": {str(k): float(v) for k, v in per_class_thresholds.items()},
                "default_threshold": float(best_threshold),
            }, f, indent=2)
        print(f"Per-class thresholds saved to {pc_path}")

    print(f"\nDone. {'✅' if best_mAP50 > 0 else '❌'} Best mAP@0.5 = {best_mAP50:.4f} at threshold = {best_threshold:.5f}")


if __name__ == "__main__":
    main()
