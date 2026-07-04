"""
eval_yolov8m_psr.py — YOLOv8m Detections fed through MonotonicDecoder for PSR
================================================================================
Runs YOLOv8m detection on the validation set, converts per-frame detection
outputs to spatial-semantic (s2) features, and feeds them through the
MonotonicDecoder (from ``src.models.psr_transition``) to produce PSR state
predictions. Computes PSR F1, POS, and edit distance.

Expected F1: 0.45-0.65 (Opus Q16 / D4).

The s2 conversion relies on the observation that each of the 24 ASD detection
classes (DET_CLASS_NAMES) encodes a specific 11-bit PSR component state as a
binary string. YOLOv8m detections are therefore a proxy for the assembly state:
the per-component logit is the maximum detection-confidence-logit across all
detected ASD classes that activate that component.

Usage:
    python3 src/evaluation/eval_yolov8m_psr.py --batch_size 16

Reference (s2 feature conversion):
    src/evaluation/eval_yolov8m_psr.py:180-207  (s2_from_yolo_detections)
    src/models/psr_transition.py:76-163          (MonotonicDecoder)
    src/evaluation/evaluate.py:379-421           (decode_and_score_psr)
    src/evaluation/evaluate.py:324-376           (_group_psr_by_recording)
"""

import argparse
import gc
import json
import logging
import math
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
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

logger = logging.getLogger("eval_yolov8m_psr")

_OUTPUT_PATH = Path(
    "src/runs/rf_stages/checkpoints/eval_yolov8m_psr_results.json"
)

INDUSTREAL_WEIGHT_URL = (
    "https://github.com/microsoft/IndustReal/raw/main/weights/yolov8m_industreal.pt"
)

# ── ASD class -> PSR component mapping ──────────────────────────────────
# Each DET_CLASS_NAME (1-indexed, COCO convention) is an 11-char binary
# string where the i-th character is '1' if that PSR component is active
# in the assembly state represented by the detection class.
# We build a mapping matrix: PSR_MASK[det_class_0idx, psr_component] = 1
# if detection class implies component is assembled.
def _build_psr_mask() -> np.ndarray:
    """Build the [24, 11] binary mask: PSR_MASK[c, comp] = 1

    Maps each 0-indexed detection class (0 = background, 1-22 = assembly
    states, 23 = error_state) to the 11 PSR components it activates.
    """
    mask = np.zeros((C.NUM_DET_CLASSES, C.NUM_PSR_COMPONENTS), dtype=np.float32)
    names = getattr(C, "DET_CLASS_NAMES", {})
    for one_idx, name in names.items():
        zero_idx = one_idx - 1
        if name == "background" or name == "error_state":
            continue  # skip non-PSR classes: no component mapping
        if len(name) != C.NUM_PSR_COMPONENTS:
            logger.warning(
                "DET_CLASS_NAMES[%d] = '%s' has length %d, expected %d. Skipping.",
                one_idx, name, len(name), C.NUM_PSR_COMPONENTS,
            )
            continue
        for comp in range(C.NUM_PSR_COMPONENTS):
            if name[comp] == "1":
                mask[zero_idx, comp] = 1.0
    return mask


PSR_MASK = _build_psr_mask()


def _download_weights(
    url: str = INDUSTREAL_WEIGHT_URL,
) -> Optional[Path]:
    """Download YOLOv8m weights (IndustReal-specific or COCO fallback)."""
    save_path = Path(
        "src/runs/rf_stages/checkpoints/yolov8m_industreal.pt"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        logger.info("Using cached weights: %s", save_path)
        return save_path

    logger.info("Downloading IndustReal weights from %s ...", url)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(save_path))
        logger.info("Downloaded to %s", save_path)
        return save_path
    except Exception as exc:
        logger.warning(
            "IndustReal download failed: %s. Using COCO-pretrained YOLOv8m.",
            exc,
        )
        return None


def _build_yolo_model(
    weight_path: Optional[Path] = None,
):
    """Load YOLOv8m via the ultralytics API."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required. Install: pip install ultralytics"
        )

    if weight_path is not None and weight_path.exists():
        logger.info("Loading YOLOv8m from: %s", weight_path)
        return YOLO(str(weight_path))
    logger.info("Loading COCO-pretrained YOLOv8m.")
    return YOLO("yolov8m.pt")


# =====================================================================
# s2 Feature Conversion  (src/evaluation/eval_yolov8m_psr.py:180-207)
# =====================================================================
def s2_from_yolo_detections(
    yolo_results,
    detection_thresh: float = 0.1,
) -> np.ndarray:
    """Convert YOLOv8m per-image detection results to PSR logits.

    For each image, builds an 11-component logit vector by aggregating
    detection confidences over the PSR_MASK mapping. The logit for each
    PSR component c is::

        logit_c = sum_{detection d with class c_d}
                      logit(confidence_d) * PSR_MASK[c_d, c]

    where logit(p) = log(p / (1 - p)). If no detections activate a
    component, the logit defaults to -3.0 (sigmoid(-3) ≈ 0.047).

    Args:
        yolo_results: List of ultralytics Results objects (one per image).
        detection_thresh: Minimum confidence to consider a detection.

    Returns:
        psr_logits: [B, 11] numpy array — per-frame PSR logits.
    """
    B = len(yolo_results)
    batch_logits = np.full(
        (B, C.NUM_PSR_COMPONENTS),
        fill_value=-3.0,
        dtype=np.float32,
    )

    for img_idx, result in enumerate(yolo_results):
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # [N]
        confs = boxes.conf.cpu().numpy().astype(np.float32)  # [N]

        # For each detection, convert confidence to logit and aggregate
        # over the PSR mask.
        det_logits = np.full(C.NUM_PSR_COMPONENTS, -3.0, dtype=np.float32)
        for cls_id, conf in zip(cls_ids, confs):
            if cls_id < 0 or cls_id >= C.NUM_DET_CLASSES:
                continue
            if conf < detection_thresh:
                continue

            # Only consider ASD classes that have a PSR mapping.
            component_mask = PSR_MASK[cls_id]  # [11]
            if component_mask.sum() == 0:
                continue

            # Convert confidence to logit: logit(p) = log(p / (1-p)).
            p = np.clip(conf, 1e-7, 1.0 - 1e-7)
            logit_val = math.log(p / (1.0 - p))

            # Take the top-K contribution for this component.
            # We use max-accumulation: the highest-confidence detection
            # for a component determines its logit.
            for comp in range(C.NUM_PSR_COMPONENTS):
                if component_mask[comp] > 0:
                    det_logits[comp] = max(det_logits[comp], logit_val)

        batch_logits[img_idx] = det_logits

    return batch_logits


# =====================================================================
# PSR Eval (adapted from evaluate.py:379-421)
# =====================================================================
def decode_and_score_psr_from_logits(
    psr_logits_by_rec: Dict[str, np.ndarray],
    gt_states_by_rec: Dict[str, np.ndarray],
    tol_frames: int = 3,
) -> Dict[str, float]:
    """Score PSR by running the MonotonicDecoder on aggregated logits.

    This mirrors ``evaluate.decode_and_score_psr`` but accepts numpy
    logit arrays directly (since they come from s2 features, not model
    outputs).  The decoder enforces the monotone fill-forward constraint
    to produce binary state sequences, then computes transition F1, POS,
    and edit distance.

    Args:
        psr_logits_by_rec: {recording_id: [T, 11] numpy logit array}
        gt_states_by_rec:  {recording_id: [T, 11] numpy binary GT array}
        tol_frames: tolerance for bi-directional event matching.

    Returns:
        dict with psr_f1, psr_pos, psr_edit.
    """
    try:
        from src.models.psr_transition import MonotonicDecoder
        decoder = MonotonicDecoder(num_components=C.NUM_PSR_COMPONENTS)
    except ImportError:
        logger.warning(
            "MonotonicDecoder not available — PSR metrics will be zeros."
        )
        return {"psr_f1": 0.0, "psr_pos": 0.0, "psr_edit": 0.0}

    # Same scoring helpers as evaluate.py.
    def _event_f1(pred_tr, gt_tr, tol):
        if not pred_tr.any() and not gt_tr.any():
            return 1.0
        if not pred_tr.any() or not gt_tr.any():
            return 0.0
        n_comp = pred_tr.shape[1]
        tp, fp, fn_tot = 0, 0, 0
        for c in range(n_comp):
            p_frames = np.where(pred_tr[:, c])[0]
            g_frames = np.where(gt_tr[:, c])[0]
            matched = set()
            for pf in p_frames:
                for gi, gf in enumerate(g_frames):
                    if gi not in matched and abs(pf - gf) <= tol:
                        matched.add(gi)
                        tp += 1
                        break
                else:
                    fp += 1
            fn_tot += len(g_frames) - len(matched)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn_tot, 1)
        return 2 * prec * rec / max(prec + rec, 1e-9)

    def _ordered_pair_fraction(pred_states, gt_states):
        pred_pairs = pred_states[1:] - pred_states[:-1]
        gt_pairs = gt_states[1:] - gt_states[:-1]
        return float((np.sign(pred_pairs) == np.sign(gt_pairs)).mean())

    def _psr_edit_score(pred_states, gt_states):
        pred_events = "".join(
            str(int(b))
            for b in (pred_states[1:] != pred_states[:-1]).any(axis=1).astype(int)
        )
        gt_events = "".join(
            str(int(b))
            for b in (gt_states[1:] != gt_states[:-1]).any(axis=1).astype(int)
        )
        if not gt_events:
            return 1.0 if not pred_events else 0.0
        m, n = len(pred_events), len(gt_events)
        dp = np.zeros((m + 1, n + 1))
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if pred_events[i - 1] == gt_events[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        return 1.0 - dp[m, n] / max(m, n, 1)

    f1s, poss, edits = [], [], []
    for rec, logits_np in psr_logits_by_rec.items():
        gt = gt_states_by_rec.get(rec)
        if gt is None or len(logits_np) < 2:
            continue

        events = torch.sigmoid(torch.as_tensor(logits_np)).unsqueeze(0).float()
        pred_states = decoder(events).squeeze(0)  # [T, 11]

        pred_tr = (pred_states[1:] - pred_states[:-1]).clamp(min=0).cpu().numpy()
        gt_np = np.asarray(gt)
        gt_tr = (gt_np[1:] - gt_np[:-1]).clamp(min=0)

        f1s.append(_event_f1(pred_tr, gt_tr, tol=tol_frames))
        poss.append(
            _ordered_pair_fraction(pred_states.cpu().numpy(), gt_np)
        )
        edits.append(
            _psr_edit_score(pred_states.cpu().numpy(), gt_np)
        )

    if not f1s:
        return {"psr_f1": 0.0, "psr_pos": 0.0, "psr_edit": 0.0}

    return {
        "psr_f1": float(np.mean(f1s)),
        "psr_pos": float(np.mean(poss)),
        "psr_edit": float(np.mean(edits)),
    }


def run_yolov8m_psr_eval(
    weight_url: str = INDUSTREAL_WEIGHT_URL,
    batch_size: int = 16,
    max_batches: int = 0,
    device: str = "cuda",
    detection_thresh: float = 0.1,
) -> Dict[str, float]:
    """Run YOLOv8m -> PSR evaluation.

    Pipeline:
        1. Run YOLOv8m detection on all val frames
        2. Convert per-frame detections to [T, 11] logits (s2 features)
        3. Group frames by recording, sort temporally
        4. Run MonotonicDecoder to produce state predictions
        5. Compute PSR F1, POS, edit distance

    Args:
        weight_url: URL for IndustReal YOLOv8m weights.
        batch_size: YOLOv8 inference batch size.
        max_batches: Cap on number of batches (0 = unlimited).
        device: Target device.
        detection_thresh: Minimum detection confidence for PSR aggregation.

    Returns:
        dict with psr_f1, psr_pos, psr_edit, and metadata.
    """
    # ── Load YOLOv8m ────────────────────────────────────────────────────
    weight_path = _download_weights(weight_url)
    yolo = _build_yolo_model(weight_path)

    # ── Build val dataset ───────────────────────────────────────────────
    val_dataset = IndustRealMultiTaskDataset(
        split="val",
        img_size=(C.IMG_WIDTH, C.IMG_HEIGHT),
    )

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # ── Accumulators ────────────────────────────────────────────────────
    all_psr_logits: List[np.ndarray] = []     # per-batch [B, 11]
    all_psr_labels: List[np.ndarray] = []     # per-batch [B, 11]
    all_rec_ids: List[str] = []               # per-frame
    all_frame_nums: List[int] = []            # per-frame

    logger.info(
        "Starting YOLOv8m -> PSR eval, batch_size=%d, detection_thresh=%.3f",
        batch_size,
        detection_thresh,
    )

    for bi, (images, targets) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        B = images.shape[0]

        # Convert to numpy HWC for YOLOv8.
        batch_imgs_np = []
        for i in range(B):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            batch_imgs_np.append(img)

        # YOLOv8m inference.
        results = yolo(batch_imgs_np, verbose=False)

        # s2 feature conversion: detections -> PSR logits [B, 11].
        psr_logits_batch = s2_from_yolo_detections(
            results, detection_thresh=detection_thresh,
        )
        all_psr_logits.append(psr_logits_batch)

        # Ground truth PSR labels [B, 11].
        psr_labels_batch = targets["psr_labels"].cpu().numpy()  # [B, 11]
        all_psr_labels.append(psr_labels_batch)

        # Recording IDs and frame numbers for temporal grouping.
        for i in range(B):
            metadata_item = targets["metadata"][i] if i < len(targets["metadata"]) else {}
            rec_id = metadata_item.get(
                "recording_id",
                metadata_item.get("rec_id", f"batch{bi}_i{i}"),
            )
            if isinstance(rec_id, torch.Tensor):
                rec_id = str(rec_id.item())
            else:
                rec_id = str(rec_id)
            all_rec_ids.append(rec_id)

            frame_num = metadata_item.get("frame_num", metadata_item.get("frame_idx", 0))
            if isinstance(frame_num, torch.Tensor):
                frame_num = frame_num.item()
            all_frame_nums.append(int(frame_num))

        if bi % 10 == 0:
            n_det = sum((r.boxes is not None and len(r.boxes)) for r in results)
            logger.info(
                "Batch %d: %d images, %d total detections",
                bi, B, n_det,
            )

        del images, targets, results
        gc.collect()

    # ── Group by recording (mirrors evaluate.py _group_psr_by_recording) ──
    logger.info("Grouping %d frames by recording...", len(all_rec_ids))
    by_rec_logits: Dict[str, List[np.ndarray]] = {}
    by_rec_gt: Dict[str, List[np.ndarray]] = {}
    by_rec_fn: Dict[str, List[int]] = {}

    flat_i = 0
    for batch_logits, batch_labels in zip(all_psr_logits, all_psr_labels):
        bl = np.asarray(batch_logits)
        lb = np.asarray(batch_labels)
        if bl.ndim == 1:
            bl = bl[None, :]
        if lb.ndim == 1:
            lb = lb[None, :]
        for row in range(bl.shape[0]):
            rec = all_rec_ids[flat_i] if flat_i < len(all_rec_ids) else f"rec_{flat_i}"
            fn = (
                all_frame_nums[flat_i]
                if flat_i < len(all_frame_nums)
                else flat_i
            )
            by_rec_logits.setdefault(rec, []).append(bl[row, :C.NUM_PSR_COMPONENTS])
            by_rec_gt.setdefault(rec, []).append(
                lb[row, :C.NUM_PSR_COMPONENTS] if row < lb.shape[0] else None
            )
            by_rec_fn.setdefault(rec, []).append(fn)
            flat_i += 1

    psr_logits_by_rec: Dict[str, np.ndarray] = {}
    gt_states_by_rec: Dict[str, np.ndarray] = {}
    for rec, rows in by_rec_logits.items():
        gts = by_rec_gt[rec]
        if any(g is None for g in gts) or len(rows) < 2:
            continue
        order = np.argsort(np.asarray(by_rec_fn[rec], dtype=np.int64), kind="stable")
        psr_logits_by_rec[rec] = np.stack([rows[k] for k in order]).astype(np.float32)
        gt_states_by_rec[rec] = np.stack([gts[k] for k in order]).astype(np.float32)

    logger.info(
        "Grouped into %d recordings (min frames=%d, max frames=%d)",  # noqa: G004
        len(psr_logits_by_rec),
        min(v.shape[0] for v in psr_logits_by_rec.values()) if psr_logits_by_rec else 0,
        max(v.shape[0] for v in psr_logits_by_rec.values()) if psr_logits_by_rec else 0,
    )

    # ── Decode and score PSR ───────────────────────────────────────────
    logger.info("Running MonotonicDecoder on %d recordings...", len(psr_logits_by_rec))
    psr_metrics = decode_and_score_psr_from_logits(
        psr_logits_by_rec, gt_states_by_rec, tol_frames=3,
    )

    # Add metadata.
    psr_metrics["_model"] = "yolov8m -> s2 -> MonotonicDecoder"
    psr_metrics["_detection_thresh"] = detection_thresh
    psr_metrics["_num_frames"] = flat_i
    psr_metrics["_num_recordings"] = len(psr_logits_by_rec)
    psr_metrics["_weight_source"] = (
        str(weight_path) if weight_path else "ultralytics_coco"
    )

    logger.info(
        "YOLOv8m -> PSR results — F1: %.4f  POS: %.4f  Edit: %.4f",
        psr_metrics.get("psr_f1", 0.0),
        psr_metrics.get("psr_pos", 0.0),
        psr_metrics.get("psr_edit", 0.0),
    )

    return psr_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8m -> PSR evaluation (D4 / Q16)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="YOLOv8 batch size"
    )
    parser.add_argument(
        "--max_batches", type=int, default=0, help="Max batches (0 = all)"
    )
    parser.add_argument(
        "--detection_thresh",
        type=float,
        default=0.1,
        help="Min detection confidence for s2 aggregation",
    )
    parser.add_argument(
        "--weight_url", type=str, default=INDUSTREAL_WEIGHT_URL, help="Weight URL"
    )
    parser.add_argument(
        "--output", type=str, default=str(_OUTPUT_PATH), help="Output JSON path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(name)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    metrics = run_yolov8m_psr_eval(
        weight_url=args.weight_url,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
        detection_thresh=args.detection_thresh,
    )

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
