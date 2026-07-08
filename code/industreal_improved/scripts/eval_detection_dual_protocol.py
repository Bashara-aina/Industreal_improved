#!/usr/bin/env python3
"""
eval_detection_dual_protocol.py — Standalone Dual-Protocol mAP@0.5 Evaluation.

Evaluates detection mAP@0.5 under two protocols (174 §3.1, 175 §7.2):
  1. Annotated-frames protocol  — subset to frames with >= 1 GT box  (WACV 0.838)
  2. Entire-video protocol     — full video sequence, 99.9% empty      (WACV 0.641)

Loads a saved per_frame_predictions.json (detection outputs + GT) and runs
both mAP computations using the existing evaluate.py helpers.

Usage:
    python3 scripts/eval_detection_dual_protocol.py \
        [--predictions PATH] [--out PATH]

Without arguments, uses the d3_full_eval predictions file and writes to
src/runs/rf_stages/checkpoints/detection_dual_protocol/metrics.json.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
for _sub in ["evaluation", "training", "data", "models", str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src import config as C
from src.evaluation.evaluate import (
    compute_ap_per_class,
    compute_ap_per_class_all_frames,
)

logger = logging.getLogger("eval_detection_dual_protocol")

# ── Default paths ─────────────────────────────────────────────────────────────
_DEFAULT_PREDICTIONS = (
    Path(__file__).resolve().parent.parent
    / "src/runs/rf_stages/checkpoints/d3_full_eval/per_frame_predictions.json"
)
_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "src/runs/rf_stages/checkpoints/detection_dual_protocol/metrics.json"
)


def load_predictions(path: Path) -> dict:
    """Load per-frame detection predictions JSON and convert lists to numpy arrays."""
    logger.info("Loading predictions from %s", path)
    with open(path) as f:
        data = json.load(f)

    # Convert nested lists to ndarray-of-lists for the eval helpers.
    # The helpers iterate with len(), indexing, and np.array() calls internally.
    boxes = [np.asarray(b, dtype=np.float32) for b in data["det_pred_boxes"]]
    scores = [np.asarray(s, dtype=np.float32) for s in data["det_pred_scores"]]
    labels = [np.asarray(l, dtype=np.int32) for l in data["det_pred_labels"]]
    gt_boxes = [np.asarray(gb, dtype=np.float32) for gb in data["det_gt_boxes"]]
    gt_labels = [np.asarray(gl, dtype=np.int32) for gl in data["det_gt_labels"]]

    logger.info(
        "Loaded %d frames, %d total GT boxes, %d total predictions",
        len(gt_boxes),
        sum(len(b) for b in gt_boxes),
        sum(len(b) for b in boxes),
    )
    return {"boxes": boxes, "scores": scores, "labels": labels,
            "gt_boxes": gt_boxes, "gt_labels": gt_labels}


def filter_annotated_frames(pred_boxes, pred_scores, pred_labels,
                            gt_boxes, gt_labels):
    """Subset to frames where at least one GT box exists."""
    indices = [i for i, gb in enumerate(gt_boxes) if len(gb) > 0]
    logger.info(
        "Annotated-frames subset: %d / %d frames (%.1f%%)",
        len(indices), len(gt_boxes), 100 * len(indices) / max(len(gt_boxes), 1),
    )
    return (
        [pred_boxes[i] for i in indices],
        [pred_scores[i] for i in indices],
        [pred_labels[i] for i in indices],
        [gt_boxes[i] for i in indices],
        [gt_labels[i] for i in indices],
    )


def compute_annotated_frames_mAP(pred_boxes, pred_scores, pred_labels,
                                 gt_boxes, gt_labels, num_classes):
    """Compute mAP@0.5 on annotated frames only (WACV 0.838 protocol)."""
    pb_f, ps_f, pl_f, gb_f, gl_f = filter_annotated_frames(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
    )
    gt_box_total = sum(len(b) for b in gb_f)
    if gt_box_total == 0:
        raise AssertionError(
            "gt_box_total == 0 in annotated-frames subset — "
            "no GT boxes found. Cannot compute mAP."
        )
    result = compute_ap_per_class(
        pb_f, ps_f, pl_f, gb_f, gl_f,
        iou_thresh=0.5,
        num_classes=num_classes,
    )
    # Present-class average: mean over classes with GT > 0.
    per_class = result["per_class_ap"]
    present_aps = [v for v in per_class.values() if v > 0] or [0.0]
    mAP_pc = float(np.mean(present_aps))
    return {
        "det_mAP50": float(result["mAP"]),
        "det_mAP50_pc": mAP_pc,
        "per_class_ap": {str(k): float(v) for k, v in per_class.items()},
        "n_frames": len(pb_f),
        "gt_box_total": gt_box_total,
    }


def compute_entire_video_mAP(pred_boxes, pred_scores, pred_labels,
                             gt_boxes, gt_labels, num_classes):
    """Compute mAP@0.5 on ALL frames (WACV 0.641 protocol)."""
    gt_box_total = sum(len(b) for b in gt_boxes)
    if gt_box_total == 0:
        raise AssertionError(
            "gt_box_total == 0 in entire eval set — no GT boxes found. "
            "Cannot compute mAP."
        )
    result = compute_ap_per_class_all_frames(
        pred_boxes, pred_scores, pred_labels,
        gt_boxes, gt_labels,
        iou_thresh=0.5,
        num_classes=num_classes,
    )
    return {
        "det_mAP50_all_frames": float(result["mAP"]),
        "per_class_ap_all_frames": {str(k): float(v) for k, v in result["per_class_ap"].items()},
        "n_frames": len(gt_boxes),
        "gt_box_total": gt_box_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dual-protocol detection mAP@0.5 evaluation"
    )
    parser.add_argument(
        "--predictions", type=str, default=str(_DEFAULT_PREDICTIONS),
        help="Path to per_frame_predictions.json",
    )
    parser.add_argument(
        "--out", type=str, default=str(_DEFAULT_OUTPUT),
        help="Output JSON path",
    )
    parser.add_argument(
        "--num_classes", type=int, default=C.NUM_DET_CLASSES,
        help="Number of detection classes (default: %d)" % C.NUM_DET_CLASSES,
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress INFO logging (errors still print)",
    )
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    preds_path = Path(args.predictions)
    if not preds_path.exists():
        logger.error("Predictions file not found: %s", preds_path)
        sys.exit(1)

    num_classes = args.num_classes
    out_path = Path(args.out)

    # ── Load ────────────────────────────────────────────────────────────────
    t0 = time.time()
    data = load_predictions(preds_path)

    # ── Assert: we have GT boxes ────────────────────────────────────────────
    total_gt = sum(len(b) for b in data["gt_boxes"])
    if total_gt == 0:
        raise AssertionError(
            "gt_box_total == 0 in the full evaluation set. "
            "Cannot compute any detection mAP. "
            "This is the P3 empty-subsample bug — refusing to produce 0.0."
        )
    logger.info("Total GT boxes in eval set: %d (good)", total_gt)

    # ── Protocol 1: Annotated frames (WACV 0.838) ──────────────────────────
    logger.info("=" * 60)
    logger.info("Protocol 1: ANNOTATED-FRAMES (subset to frames with >= 1 GT)")
    t1 = time.time()
    af_result = compute_annotated_frames_mAP(
        data["boxes"], data["scores"], data["labels"],
        data["gt_boxes"], data["gt_labels"],
        num_classes=num_classes,
    )
    t_af = time.time() - t1
    logger.info("  mAP@0.5 (COCO-24):          %.4f", af_result["det_mAP50"])
    logger.info("  mAP@0.5 (present-class avg): %.4f", af_result["det_mAP50_pc"])
    logger.info("  Frames evaluated: %d  GT boxes: %d  Time: %.1fs",
                af_result["n_frames"], af_result["gt_box_total"], t_af)

    # ── Protocol 2: Entire video (WACV 0.641) ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Protocol 2: ENTIRE-VIDEO (all %d frames, including empty)", len(data["gt_boxes"]))
    t2 = time.time()
    ev_result = compute_entire_video_mAP(
        data["boxes"], data["scores"], data["labels"],
        data["gt_boxes"], data["gt_labels"],
        num_classes=num_classes,
    )
    t_ev = time.time() - t2
    logger.info("  mAP@0.5 (all frames):       %.4f", ev_result["det_mAP50_all_frames"])
    logger.info("  Frames evaluated: %d  GT boxes: %d  Time: %.1fs",
                ev_result["n_frames"], ev_result["gt_box_total"], t_ev)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DUAL-PROTOCOL DETECTION mAP@0.5 SUMMARY")
    logger.info("  Annotated-frames mAP@0.5  (WACV 0.838)  →  %.4f",
                af_result["det_mAP50"])
    logger.info("  Annotated-frames PC mAP@0.5 (WACV 0.838) →  %.4f",
                af_result["det_mAP50_pc"])
    logger.info("  Entire-video mAP@0.5      (WACV 0.641)  →  %.4f",
                ev_result["det_mAP50_all_frames"])
    logger.info("  Total time: %.1fs", time.time() - t0)

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "protocol": "detection_dual_protocol",
        "reference": "AAIML 174 §3.1 / 175 §7.2",
        "predictions_source": str(preds_path),
        "num_classes": num_classes,
        "annotated_frames": {
            "description": "Subset to frames with >= 1 GT box (WACV 0.838)",
            "n_frames": af_result["n_frames"],
            "gt_box_total": af_result["gt_box_total"],
            "det_mAP50": af_result["det_mAP50"],
            "det_mAP50_pc": af_result["det_mAP50_pc"],
            "per_class_ap": af_result["per_class_ap"],
        },
        "entire_video": {
            "description": "All frames including empty (WACV 0.641)",
            "n_frames": ev_result["n_frames"],
            "gt_box_total": ev_result["gt_box_total"],
            "det_mAP50_all_frames": ev_result["det_mAP50_all_frames"],
            "per_class_ap_all_frames": ev_result["per_class_ap_all_frames"],
        },
        "sota_anchor": {
            "WACV_annotated_frames": 0.838,
            "WACV_entire_video": 0.641,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved dual-protocol metrics to %s", out_path)

    # ── human-readable one-liner ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DUAL-PROTOCOL mAP@0.5  (annotated, entire-video)")
    print(f"  ({af_result['det_mAP50']:.4f}, {ev_result['det_mAP50_all_frames']:.4f})")
    print(f"  WACV anchors:          (0.838, 0.641)")
    print("=" * 60)


if __name__ == "__main__":
    main()
