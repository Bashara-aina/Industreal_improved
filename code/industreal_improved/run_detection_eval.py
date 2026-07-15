#!/usr/bin/env python3
"""Fast detection mAP evaluation — streaming, no NaN.

Usage:
    python run_detection_eval.py --ckpt runs/aa_path_b/checkpoints/best.pth --device cuda:1
"""
import json, logging, os, sys, time
from pathlib import Path
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent / "src"
for p in [_SRC, _SRC / "evaluation", _SRC / "models", _SRC / "training", _SRC / "data"]:
    sys.path.insert(0, str(p))

import src.config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import POPWMultiTaskModel
from evaluate import decode_boxes, nms_numpy, compute_ap_per_class, compute_ap_per_class_all_frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("det_eval")

def compute_det_mAP(metrics, batch_size):
    """Compute detection mAP from collected dp/dg arrays."""
    dpb, dps, dpl = metrics["dp_boxes"], metrics["dp_scores"], metrics["dp_labels"]
    dgb, dgl = metrics["dg_boxes"], metrics["dg_labels"]

    gt_box_total = sum(len(b) for b in dgb)
    if gt_box_total == 0:
        logger.warning("No GT boxes in eval set")
        return {"det_mAP50": 0.0, "det_mAP_50_95": 0.0, "det_n_present_classes": 0}

    logger.info("Computing mAP over %d frames, %d GT boxes...", len(dpb), gt_box_total)

    # Standard mAP (eval frames only)
    ap_result = compute_ap_per_class(dpb, dps, dpl, dgb, dgl, num_classes=24)
    r50 = compute_ap_per_class_all_frames(dpb, dps, dpl, dgb, dgl, num_classes=24)

    results = {
        "det_mAP50": float(ap_result["mAP"]),
        "det_mAP_50_95": float(ap_result.get("mAP_50_95", 0.0)),
        "det_n_present_classes": int(ap_result.get("n_present", 0)),
        "det_mAP50_all_frames": float(r50["mAP"]),
        "det_per_class_ap_all_frames": {str(k): float(v) for k, v in r50["per_class_ap"].items()},
        "_n_frames": len(dpb),
        "_n_gt_boxes": gt_box_total,
    }

    # Per-class breakdown
    per_class = ap_result.get("per_class_ap", {})
    class_aps = []
    for c in range(24):
        ap = per_class.get(c, 0.0)
        n_gt = sum((gl == c).sum() for gl in dgl) if any((gl == c).sum() > 0 for gl in dgl) else 0
        class_aps.append({"class": c, "AP": float(ap), "n_gt": int(n_gt)})
    results["det_per_class"] = class_aps

    # Log summary
    n_present = sum(1 for c in class_aps if c["n_gt"] > 0)
    logger.info("det_mAP50=%.4f (present-class=%.4f, %d/%d classes present)",
                results["det_mAP50"], results["det_mAP50_all_frames"], n_present, 24)

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/aa_path_b/checkpoints/best.pth")
    parser.add_argument("--save-dir", type=str, default="runs/aa_path_b/full_eval")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max-batches", type=int, default=5000,
                        help="0 = full, else cap")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", args.ckpt)
    logger.info("max_batches=%s batch_size=%d",
                "FULL" if args.max_batches == 0 else args.max_batches, args.batch_size)

    # Load model
    model = POPWMultiTaskModel().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()
    logger.info("Model loaded (epoch %s)", ckpt.get("epoch", "?"))

    # Dataset
    val_ds = IndustRealMultiTaskDataset(split="val", img_size=(C.IMG_HEIGHT, C.IMG_WIDTH))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )
    logger.info("Dataset: %d samples, %d batches", len(val_ds), len(val_loader))

    # Streaming eval
    dp_boxes, dp_scores, dp_labels = [], [], []
    dg_boxes, dg_labels = [], []
    _cached_anchors = None
    n_batches = 0
    t0 = time.time()

    with torch.no_grad():
        for bi, (images, targets) in enumerate(val_loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            images = images.to(device, non_blocking=True).float()
            images = images / 255.0 if images.max() > 1.0 else images
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            images = (images - mean) / std

            outputs = model(images)
            B = images.shape[0]

            if _cached_anchors is None:
                _cached_anchors = outputs["anchors"].cpu().numpy()

            cls_sig = torch.sigmoid(outputs["cls_preds"])
            score_thresh = float(getattr(C, "DET_EVAL_SCORE_THRESH", 0.5))
            nms_thresh = float(getattr(C, "DET_EVAL_NMS_IOU_THRESH", 0.5))

            for i in range(B):
                scores_i = cls_sig[i]
                max_scores = scores_i.max(dim=1).values
                keep = max_scores > score_thresh

                if keep.sum() == 0:
                    dp_boxes.append(np.zeros((0, 4)))
                    dp_scores.append(np.zeros(0))
                    dp_labels.append(np.zeros(0, dtype=np.int64))
                else:
                    keep_np = keep.cpu().numpy()
                    kept_cls = scores_i[keep].cpu().numpy()
                    kept_reg = outputs["reg_preds"][i][keep].cpu().numpy()
                    kept_anc = _cached_anchors[keep_np]
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
                        nk = nms_numpy(pb[cm], ms[cm], nms_thresh)
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

                det_list = targets["detection"]
                dg_boxes.append(det_list[i]["boxes"].numpy())
                dg_labels.append(det_list[i]["labels"].numpy())

            n_batches += 1
            if n_batches % 200 == 0:
                elapsed = time.time() - t0
                logger.info("  processed %d batches (%d frames) in %.0fm",
                            n_batches, n_batches * args.batch_size, elapsed / 60)

    elapsed = time.time() - t0
    logger.info("Inference done: %d batches (%d frames) in %.0fm",
                n_batches, n_batches * args.batch_size, elapsed / 60)

    # Compute mAP
    det_metrics = compute_det_mAP(
        {"dp_boxes": dp_boxes, "dp_scores": dp_scores, "dp_labels": dp_labels,
         "dg_boxes": dg_boxes, "dg_labels": dg_labels},
        args.batch_size,
    )

    # Save
    out_path = save_dir / "metrics.json"
    clean = {}
    for k, v in det_metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, OverflowError):
                clean[k] = str(v)
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logger.info("Saved to %s", out_path)

    # Headlines
    logger.info("=" * 50)
    logger.info("DETECTION METRICS (%d frames, %d GT boxes)",
                n_batches * args.batch_size, sum(len(b) for b in dg_boxes))
    logger.info("  det_mAP50        = %.4f", det_metrics["det_mAP50"])
    logger.info("  det_mAP50_all    = %.4f", det_metrics["det_mAP50_all_frames"])
    logger.info("  det_mAP_50_95    = %.4f", det_metrics["det_mAP_50_95"])
    logger.info("  n_present_classes = %d/24", det_metrics["det_n_present_classes"])
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
