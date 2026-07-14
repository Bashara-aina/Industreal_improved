"""
YOLOv8m Class-Mapping Sanity Check — D1/D4 Experiment Bootstrap
================================================================
Load YOLOv8m weights, run inference on N random validation images,
verify the predicted class IDs match the IndustReal 24-class taxonomy,
and save per-image predictions + mapping verification to JSON.

Reference: Opus answer rank 3 — "Download now, verify class mapping on
10 frames before the full run."

Usage::
    python -m src.evaluation.yolov8m_sanity_check.py \
        --weights weights/yolov8m.pt \
        --num_samples 10 \
        --out_path sanity_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("yolov8m_sanity")

# ---------------------------------------------------------------------------
# IndustReal 24-class detection taxonomy (from config.py DET_CLASS_NAMES)
# ---------------------------------------------------------------------------
INDUSTREAL_CLASS_NAMES: dict[int, str] = {
    1: "background",
    2: "10000000000",
    3: "10010010000",
    4: "10010100000",
    5: "10010110000",
    6: "11100000000",
    7: "11110010000",
    8: "11110100000",
    9: "11110110000",
    10: "11110111100",
    11: "11110111110",
    12: "11110110001",
    13: "11110111101",
    14: "11110111111",
    15: "11110101111",
    16: "11110011111",
    17: "11110011110",
    18: "11110101110",
    19: "11100001110",
    20: "11101101110",
    21: "11101011110",
    22: "11101111110",
    23: "11101111111",
    24: "error_state",
}

# ---------------------------------------------------------------------------
# Validation data paths (matching industreal_dataset.py + config.py)
# ---------------------------------------------------------------------------
INDUSTREAL_ROOT = Path("/media/newadmin/master/POPW/datasets/industreal")
RECORDINGS_ROOT = INDUSTREAL_ROOT / "recordings"
VAL_DIR = RECORDINGS_ROOT / "val"


def _find_val_images() -> list[Path]:
    """Collect all JPEG paths from val recordings."""
    images: list[Path] = []
    if not VAL_DIR.is_dir():
        logger.warning("Val dir not found: %s", VAL_DIR)
        return images
    for rec_dir in sorted(VAL_DIR.iterdir()):
        rgb_dir = rec_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        for f in sorted(rgb_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append(f)
    logger.info(
        "Found %d val images across %d recordings", len(images), len(list(VAL_DIR.iterdir()))
    )
    return images


def load_yolov8m(weights_path: str):
    """Load YOLOv8m model via ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info("Loading YOLOv8m weights from %s", weights_path)
    model = YOLO(weights_path)
    logger.info("Model loaded. Task=%s, names=%s", model.task, model.names)
    return model


def run_sanity_check(
    model,
    images: list[Path],
    num_samples: int = 10,
    out_path: str = "sanity_check.json",
) -> dict:
    """Run inference on random sample images, verify class IDs, save results.

    Returns a summary dict with per-image predictions + class-mapping verification.
    """
    if num_samples > len(images):
        logger.warning(
            "num_samples=%d > available images=%d, using %d",
            num_samples,
            len(images),
            len(images),
        )
        num_samples = len(images)

    sampled = random.sample(images, num_samples)
    logger.info("Running inference on %d random val images...", num_samples)

    per_image_results: list[dict] = []
    all_predicted_classes: set[int] = set()
    max_det = 0

    for img_path in sampled:
        # Read image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("Could not read %s, skipping", img_path)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img_rgb, verbose=False)
        result = results[0]  # first (only) image

        # Extract predictions
        boxes = result.boxes
        pred_classes: list[int] = []
        pred_scores: list[float] = []
        pred_boxes_xyxy: list[list[float]] = []

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                score = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()
                pred_classes.append(cls_id)
                pred_scores.append(score)
                pred_boxes_xyxy.append(xyxy)
                all_predicted_classes.add(cls_id)

        max_det = max(max_det, len(pred_classes))

        rel_path = str(img_path.relative_to(RECORDINGS_ROOT))
        entry = {
            "image": rel_path,
            "num_detections": len(pred_classes),
            "classes": pred_classes,
            "scores": pred_scores,
            "boxes_xyxy": pred_boxes_xyxy,
        }
        per_image_results.append(entry)

        logger.info(
            "  [%s] %d detections — classes=%s scores=%s",
            rel_path,
            len(pred_classes),
            pred_classes[:5],
            [f"{s:.3f}" for s in pred_scores[:5]],
        )

    # -----------------------------------------------------------------------
    # Class-mapping verification
    # -----------------------------------------------------------------------
    industreal_class_ids = set(INDUSTREAL_CLASS_NAMES.keys())
    valid_classes = all_predicted_classes.intersection(industreal_class_ids)
    out_of_range = all_predicted_classes - industreal_class_ids
    missing_industreal = industreal_class_ids - all_predicted_classes

    mapping_ok = len(out_of_range) == 0
    # If COCO-pretrained (80 classes), class IDs 0..79 are expected; only
    # IDs that match IndustReal (1..24, excl. 0) indicate correct mapping.
    coco_ids_in_industreal_range = {c for c in all_predicted_classes if 1 <= c <= 24}
    mapping_summary = {
        "predicted_classes_sorted": sorted(all_predicted_classes),
        "num_unique_predicted_classes": len(all_predicted_classes),
        "in_industreal_range_1_to_24": sorted(coco_ids_in_industreal_range),
        "out_of_industreal_range": sorted(out_of_range),
        "industreal_classes_not_predicted": sorted(missing_industreal),
        "class_names_sampled": {
            str(c): INDUSTREAL_CLASS_NAMES.get(c, f"OUT_OF_RANGE({c})")
            for c in sorted(all_predicted_classes)
        },
        "mapping_ok": mapping_ok,
        "note": (
            "All predicted classes within 1..24: OK for IndustReal-trained weights. "
            "Out-of-range classes are expected for COCO-pretrained (80-class) weights "
            "and do NOT indicate a bug — only classes 1..24 are relevant."
        ),
    }

    payload = {
        "metadata": {
            "weights": str(model.ckpt_path) if hasattr(model, "ckpt_path") else "unknown",
            "num_samples": len(per_image_results),
            "max_detections_per_image": max_det,
            "total_images_available": len(images),
        },
        "mapping_verification": mapping_summary,
        "per_image": per_image_results,
    }

    # Save to JSON
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("  Samples: %d", len(per_image_results))
    logger.info("  Max detections per image: %d", max_det)
    logger.info("  Unique predicted class IDs: %s", sorted(all_predicted_classes))
    logger.info("  Within IndustReal range (1..24): %s", sorted(coco_ids_in_industreal_range))
    logger.info("  Out of range: %s", sorted(out_of_range))
    logger.info("  Mapping OK: %s", mapping_ok)
    if not mapping_ok:
        logger.warning(
            "  OUT-OF-RANGE classes detected: %s. "
            "This is EXPECTED for COCO-pretrained YOLOv8m (80 classes). "
            "For IndustReal-finetuned weights, all classes should be in 1..24.",
            sorted(out_of_range),
        )
    logger.info("=" * 60)

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8m class-mapping sanity check on IndustReal val set.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to YOLOv8m weights (.pt)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random val images to sample (default: 10)",
    )
    parser.add_argument(
        "--out_path",
        default="sanity_check.json",
        help="Path to write results JSON (default: sanity_check.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Find val images
    val_images = _find_val_images()
    if not val_images:
        logger.error("No validation images found at %s", VAL_DIR)
        sys.exit(1)

    # Load model
    model = load_yolov8m(args.weights)

    # Run sanity check
    run_sanity_check(
        model,
        val_images,
        num_samples=args.num_samples,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
