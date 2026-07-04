"""
eval_yolov8m.py — YOLOv8m Evaluation on IndustReal Validation Set
==================================================================
Downloads YOLOv8m weights (IndustReal-specific or COCO-pretrained fallback),
runs inference on the IndustReal validation set, and computes mAP@0.5 against
the 24-class ASD taxonomy.

Expected mAP: 0.78-0.82 (Opus Q41 / D1). ~2 hours inference.

Class mapping verification:
    - Before full inference, runs detection on 10 validation frames and prints
      the distribution of predicted class IDs to verify that the YOLOv8m output
      channels align with the 24-class IndustReal taxonomy (Opus rank 3 step).

Usage:
    python3 src/evaluation/eval_yolov8m.py --batch_size 16

Reference (inference loop):
    src/evaluation/evaluate.py:3158-3425  (evaluate_all detection loop)
    src/evaluation/evaluate.py:1680-1740  (compute_det_metrics_extended)
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

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
from src.evaluation.evaluate import compute_det_metrics_extended, decode_boxes
from src.data.industreal_dataset import IndustRealMultiTaskDataset as IndustRealDataset, collate_fn

logger = logging.getLogger("eval_yolov8m")

_OUTPUT_PATH = Path(
    "src/runs/rf_stages/checkpoints/eval_yolov8m_results.json"
)

# ── YOLOv8m weight URLs ─────────────────────────────────────────────────
INDUSTREAL_WEIGHT_URL = (
    "https://github.com/microsoft/IndustReal/raw/main/weights/yolov8m_industreal.pt"
)
COCO_WEIGHT_NAME = "yolov8m.pt"  # Ultralytics will download this


def _download_weights(
    url: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Download weights from a URL, falling back to Ultralytics default.

    First tries the IndustReal-specific weights. If that fails, falls back
    to the COCO-pretrained YOLOv8m.pt from Ultralytics.

    Args:
        url: URL of the IndustReal weights.
        save_path: Where to save the downloaded weights. If None, uses
                   a default under the runs directory.

    Returns:
        Path to the downloaded weight file.
    """
    if save_path is None:
        save_path = Path(
            "src/runs/rf_stages/checkpoints/yolov8m_industreal.pt"
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Try IndustReal-specific weights first.
    if not save_path.exists():
        logger.info("Downloading IndustReal weights from %s ...", url)
        try:
            import urllib.request

            urllib.request.urlretrieve(url, str(save_path))
            logger.info("Downloaded to %s", save_path)
        except Exception as exc:
            logger.warning(
                "IndustReal download failed: %s. Falling back to COCO-pretrained %s.",
                exc,
                COCO_WEIGHT_NAME,
            )
            # Don't save a local copy; let Ultralytics handle its cache.
            save_path = None
    else:
        logger.info("Using cached IndustReal weights: %s", save_path)

    return save_path


def _build_yolo_model(
    weight_path: Optional[Path] = None,
) -> "YOLO":
    """Load or download the YOLOv8m model via the ultralytics API.

    Args:
        weight_path: Path to a custom weight file, or None for COCO-pretrained.

    Returns:
        YOLO model instance.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required. Install: pip install ultralytics"
        )

    if weight_path is not None and weight_path.exists():
        logger.info("Loading YOLOv8m from: %s", weight_path)
        model = YOLO(str(weight_path))
    else:
        logger.info(
            "Loading COCO-pretrained YOLOv8m (Ultralytics cache): %s",
            COCO_WEIGHT_NAME,
        )
        model = YOLO(COCO_WEIGHT_NAME)

    return model


def _verify_class_mapping(
    yolo_model: "YOLO",
    val_dataset: IndustRealDataset,
    num_verify_frames: int = 10,
) -> bool:
    """Verify YOLOv8m class predictions on a small sample of validation frames.

    Runs inference on the first ``num_verify_frames`` frames and prints the
    observed class ID distribution. This is a sanity check to confirm that the
    YOLOv8m output channels match our 24-class IndustReal taxonomy before
    running the full evaluation.

    Args:
        yolo_model: YOLOv8m model instance.
        val_dataset: IndustReal validation dataset.
        num_verify_frames: Number of frames to test (default: 10).

    Returns:
        True if verification passed (classes detected as expected).
    """
    logger.info(
        "Verifying class mapping on %d frames (Opus rank 3)...",
        num_verify_frames,
    )
    all_det_classes = []
    for idx in range(min(num_verify_frames, len(val_dataset))):
        sample = val_dataset[idx]
        # The dataset returns uint8 [3, H, W] images under images['rgb'].
        # Convert to numpy for YOLOv8 inference (HWC uint8).
        img_tensor = sample["images"]["rgb"]  # [3, H, W]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        results = yolo_model(img_np, verbose=False)
        if len(results) > 0 and results[0].boxes is not None:
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            all_det_classes.extend(cls_ids.tolist())

    if not all_det_classes:
        logger.warning(
            "No detections found on verification frames! "
            "This may indicate a class mismatch or score threshold issue."
        )
        return False

    unique_classes = set(all_det_classes)
    logger.info(
        "Verification: %d detections across %d frames. "
        "Unique class IDs detected: %s",
        len(all_det_classes),
        num_verify_frames,
        sorted(unique_classes),
    )

    # Check that class IDs are within the expected range 0-23 (24 classes).
    max_class = max(unique_classes) if unique_classes else -1
    min_class = min(unique_classes) if unique_classes else -1
    if min_class >= 0 and max_class < C.NUM_DET_CLASSES:
        logger.info(
            "Class mapping OK: detected classes in [%d, %d) within [0, %d)",
            min_class,
            max_class + 1,
            C.NUM_DET_CLASSES,
        )
        return True
    else:
        logger.warning(
            "Class mapping MISMATCH: detected classes in [%d, %d) "
            "but expecting [0, %d)",
            min_class,
            max_class + 1,
            C.NUM_DET_CLASSES,
        )
        return False


def _yolo_to_eval_format(
    yolo_results,
    img_w: int,
    img_h: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Convert YOLO Results objects to the evaluate.py detection format.

    YOLOv8 outputs are [N, 6] tensors: [x1, y1, x2, y2, confidence, class_id].
    We convert to per-image lists of boxes (xyxy), scores, and labels.

    Args:
        yolo_results: List of ultralytics Results objects (one per image).
        img_w, img_h: Image dimensions for clipping.

    Returns:
        (boxes_list, scores_list, labels_list) per-image lists.
    """
    boxes_list: List[np.ndarray] = []
    scores_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for result in yolo_results:
        if result.boxes is None or len(result.boxes) == 0:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            scores_list.append(np.zeros(0, dtype=np.float32))
            labels_list.append(np.zeros(0, dtype=np.int64))
            continue

        # YOLOv8 boxes are already in xyxy format, [N, 4].
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        # Class IDs: YOLOv8 uses 0-indexed COCO classes (80 classes).
        # For IndustReal weights, these should be 0-23.
        labels = result.boxes.cls.cpu().numpy().astype(np.int64)

        # Clip to image bounds.
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return boxes_list, scores_list, labels_list


def run_yolov8m_eval(
    weight_url: str = INDUSTREAL_WEIGHT_URL,
    batch_size: int = 16,
    max_batches: int = 0,
    device: str = "cuda",
    verify_frames: int = 10,
) -> Dict[str, float]:
    """Run YOLOv8m evaluation on the IndustReal validation set.

    Args:
        weight_url: URL of the IndustReal-specific YOLOv8m weights.
        batch_size: YOLOv8 inference batch size.
        max_batches: Cap on number of batches (0 = unlimited).
        device: Target device string.
        verify_frames: Number of frames to use for class mapping verification.

    Returns:
        dict with detection metrics (det_mAP50, det_per_class_ap, etc.).
    """
    # ── Download / load YOLOv8m ─────────────────────────────────────────
    weight_path = _download_weights(weight_url)
    yolo = _build_yolo_model(weight_path)

    # ── Build val dataset ───────────────────────────────────────────────
    val_dataset = IndustRealDataset(
        split="val",
        img_size=(C.IMG_WIDTH, C.IMG_HEIGHT),
    )

    # ── Class mapping verification (step 3 per Opus) ─────────────────────
    mapping_ok = _verify_class_mapping(
        yolo, val_dataset, num_verify_frames=verify_frames
    )
    if not mapping_ok:
        logger.warning(
            "Class mapping verification flagged potential issues. "
            "Proceeding with full eval nonetheless."
        )

    # ── Accumulators ─────────────────────────────────────────────────────
    dp_boxes: List[np.ndarray] = []
    dp_scores: List[np.ndarray] = []
    dp_labels: List[np.ndarray] = []
    dg_boxes: List[np.ndarray] = []
    dg_labels: List[np.ndarray] = []

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    logger.info(
        "Starting YOLOv8m eval, batch_size=%d, device=%s",
        batch_size,
        device,
    )

    for bi, (images, targets) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        B = images.shape[0]

        # Convert dataset's [B, 3, H, W] uint8 tensors to numpy HWC images.
        batch_imgs_np = []
        for i in range(B):
            img = images[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            batch_imgs_np.append(img)

        # YOLOv8 inference on the batch.
        results = yolo(batch_imgs_np, verbose=False)

        # Convert to evaluation format.
        boxes_list, scores_list, labels_list = _yolo_to_eval_format(
            results, img_w=C.IMG_WIDTH, img_h=C.IMG_HEIGHT,
        )

        dp_boxes.extend(boxes_list)
        dp_scores.extend(scores_list)
        dp_labels.extend(labels_list)

        # Ground truth.
        detection_list = targets["detection"]
        for i in range(B):
            dg_boxes.append(
                detection_list[i]["boxes"].cpu().numpy()
            )
            dg_labels.append(
                detection_list[i]["labels"].cpu().numpy()
            )

        if bi % 10 == 0:
            _dp_total = sum(len(b) for b in boxes_list)
            logger.info(
                "Batch %d: %d images, %d total detections",
                bi, B, _dp_total,
            )

        del images, targets, results
        gc.collect()

    # ── Compute detection metrics ───────────────────────────────────────
    logger.info("Computing detection metrics...")
    det_metrics = compute_det_metrics_extended(
        dp_boxes, dp_scores, dp_labels,
        dg_boxes, dg_labels,
        num_classes=C.NUM_DET_CLASSES,
    )

    # Add per-class-present mAP50.
    mAP_per_thresh_pc = det_metrics.get("mAP_per_thresh_pc", {})
    det_metrics["det_mAP50_pc"] = mAP_per_thresh_pc.get(0.5, 0.0)

    # Metadata.
    det_metrics["_model"] = "yolov8m"
    det_metrics["_weight_source"] = (
        str(weight_path) if weight_path else "ultralytics_coco"
    )
    det_metrics["_num_images"] = len(dp_boxes)
    det_metrics["_class_mapping_verified"] = mapping_ok

    logger.info(
        "YOLOv8m results — mAP@0.5: %.4f  mAP@[0.5:0.95]: %.4f  mAP50_pc: %.4f",
        det_metrics.get("det_mAP50", 0.0),
        det_metrics.get("det_mAP_50_95", 0.0),
        det_metrics.get("det_mAP50_pc", 0.0),
    )

    return det_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8m evaluation on IndustReal validation set"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="YOLOv8 inference batch size (default: 16)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Max batches to evaluate (0 = all)",
    )
    parser.add_argument(
        "--weight_url",
        type=str,
        default=INDUSTREAL_WEIGHT_URL,
        help="URL for IndustReal-specific YOLOv8m weights",
    )
    parser.add_argument(
        "--verify_frames",
        type=int,
        default=10,
        help="Number of frames for class mapping verification",
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

    metrics = run_yolov8m_eval(
        weight_url=args.weight_url,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
        verify_frames=args.verify_frames,
    )

    # Save results.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean = {}
    for k, v in metrics.items():
        try:
            json.dumps(v)
            clean[k] = v
        except (TypeError, OverflowError):
            clean[k] = (
                float(v) if isinstance(v, (int, float, np.floating)) else str(v)
            )

    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
