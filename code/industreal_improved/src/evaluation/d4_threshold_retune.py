"""
D4 Threshold Retuning — Sweep Q48 hysteresis for YOLOv8m output statistics.

Per Opus (Q2 in 132_OPUS_ANSWERS.md):
  "If F1 jumps to 0.5-0.7, the disclosure text changes completely."

Strategy:
  1. Run YOLOv8m on val set, collect per-component sigmoid scores
     via s2_from_yolo_detections (reusing eval_yolov8m_psr.py pipeline).
  2. Sweep sustain_hi (0.3-0.7), sustain_lo (0.1-0.5), sustain_min (2-6).
  3. For each combination, compute transition F1 via MonotonicDecoder.
  4. Pick optimal per-component thresholds for YOLOv8m output statistics.
  5. Run full D4 with retuned thresholds, save to
     src/runs/rf_stages/checkpoints/d4_retuned/metrics.json

Output:
  - src/evaluation/d4_threshold_retune.py  (this file)
  - src/runs/rf_stages/checkpoints/d4_retuned/metrics.json  (retuned D4 JSON)
  - src/runs/rf_stages/checkpoints/d4_retuned/thresholds.json  (optimal per-comp)
  - src/runs/rf_stages/checkpoints/d4_retuned/verdict.json
  - src/runs/rf_stages/checkpoints/d4_retuned/sweep_results.json  (full sweep log)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup (identical to eval_yolov8m_psr.py) ────────────────────────
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

logger = logging.getLogger(__name__)

N_COMPONENTS = 11

# ── Task-specified sweep ranges ──────────────────────────────────────────
SUSTAIN_HI_VALUES = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
SUSTAIN_LO_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3]
SUSTAIN_MIN_VALUES = [2, 3, 4, 5, 6]

# Also do a finer grid for the best region
# (only used if --fine-grid is passed)
SUSTAIN_HI_FINE = list(np.round(np.arange(0.3, 0.75, 0.05), 2))
SUSTAIN_LO_FINE = list(np.round(np.arange(0.1, 0.35, 0.05), 2))
SUSTAIN_MIN_FINE = list(range(2, 8))


# ============================================================================
# Score Collector (reuses eval_yolov8m_psr.py inference pipeline)
# ============================================================================
def collect_yolo_psr_scores(
    yolo_model,
    val_loader: torch.utils.data.DataLoader,
    max_batches: int = 0,
    device: str = "cuda",
    detection_thresh: float = 0.1,
) -> dict:
    """Collect per-component logits and GT states grouped by recording.

    Args:
        yolo_model: loaded YOLO model.
        val_loader: DataLoader for val set.
        max_batches: cap on batches (0=unlimited).
        device: target device.
        detection_thresh: min detection confidence for PSR logit aggregation.

    Returns:
        dict mapping recording_id -> (logits [T, 11], gt_states [T, 11])
    """
    logger.info("Collecting YOLOv8m -> PSR scores on val set")

    # Accumulators
    all_psr_logits: list[np.ndarray] = []
    all_psr_labels: list[np.ndarray] = []
    all_rec_ids: list[str] = []
    all_frame_nums: list[int] = []

    for bi, (images, targets) in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break

        B = images.shape[0]

        # Convert to numpy HWC BGR for YOLOv8 (IndustReal model trained on BGR).
        batch_imgs_np = []
        for i in range(B):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = img[:, :, ::-1].copy()  # RGB -> BGR
            batch_imgs_np.append(img)

        # YOLOv8m inference.
        results = yolo_model(batch_imgs_np, verbose=False)

        # s2 feature conversion: detections -> PSR logits [B, 11].
        psr_logits_batch = s2_from_yolo_detections(
            results,
            detection_thresh=detection_thresh,
        )
        all_psr_logits.append(psr_logits_batch)

        # Ground truth PSR labels [B, 11].
        psr_labels_batch = targets["psr_labels"].cpu().numpy()
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
                "Batch %d: %d images, %d detections",
                bi,
                B,
                n_det,
            )

        del images, targets, results
        gc.collect()

    # ── Group by recording ──────────────────────────────────────────────
    logger.info("Grouping %d frames by recording...", len(all_rec_ids))
    by_rec_logits: dict[str, list[np.ndarray]] = {}
    by_rec_gt: dict[str, list[np.ndarray]] = {}
    by_rec_fn: dict[str, list[int]] = {}

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
            fn = all_frame_nums[flat_i] if flat_i < len(all_frame_nums) else flat_i
            by_rec_logits.setdefault(rec, []).append(bl[row, : C.NUM_PSR_COMPONENTS])
            by_rec_gt.setdefault(rec, []).append(
                lb[row, : C.NUM_PSR_COMPONENTS] if row < lb.shape[0] else None
            )
            by_rec_fn.setdefault(rec, []).append(fn)
            flat_i += 1

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rec, rows in by_rec_logits.items():
        gts = by_rec_gt[rec]
        if any(g is None for g in gts) or len(rows) < 2:
            continue
        order = np.argsort(np.asarray(by_rec_fn[rec], dtype=np.int64), kind="stable")
        logits = np.stack([rows[k] for k in order]).astype(np.float32)
        states = np.stack([gts[k] for k in order]).astype(np.float32)
        result[rec] = (logits, states)

    logger.info(
        "Collected %d recordings (total frames: %d)",
        len(result),
        flat_i,
    )
    return result


# ============================================================================
# s2 Feature Conversion (reprod from eval_yolov8m_psr.py)
# ============================================================================
def _build_psr_mask() -> np.ndarray:
    """Build the [24, 11] binary mask: PSR_MASK[c, comp] = 1."""
    mask = np.zeros((C.NUM_DET_CLASSES, C.NUM_PSR_COMPONENTS), dtype=np.float32)
    names = getattr(C, "DET_CLASS_NAMES", {})
    for one_idx, name_val in names.items():
        zero_idx = one_idx - 1
        if name_val == "background" or name_val == "error_state":
            continue
        if len(name_val) != C.NUM_PSR_COMPONENTS:
            logger.warning(
                "DET_CLASS_NAMES[%d] has length %d, expected %d",
                one_idx,
                len(name_val),
                C.NUM_PSR_COMPONENTS,
            )
            continue
        for comp in range(C.NUM_PSR_COMPONENTS):
            if name_val[comp] == "1":
                mask[zero_idx, comp] = 1.0
    return mask


PSR_MASK = _build_psr_mask()


def s2_from_yolo_detections(
    yolo_results,
    detection_thresh: float = 0.1,
) -> np.ndarray:
    """Convert YOLOv8m per-image detection results to PSR logits.

    Identical logic to eval_yolov8m_psr.py:s2_from_yolo_detections.
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
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(np.float32)

        det_logits = np.full(C.NUM_PSR_COMPONENTS, -3.0, dtype=np.float32)
        for cls_id, conf in zip(cls_ids, confs):
            if cls_id < 0 or cls_id >= C.NUM_DET_CLASSES:
                continue
            if conf < detection_thresh:
                continue

            component_mask = PSR_MASK[cls_id]
            if component_mask.sum() == 0:
                continue

            p = np.clip(conf, 1e-7, 1.0 - 1e-7)
            logit_val = math.log(p / (1.0 - p))

            for comp in range(C.NUM_PSR_COMPONENTS):
                if component_mask[comp] > 0:
                    det_logits[comp] = max(det_logits[comp], logit_val)

        batch_logits[img_idx] = det_logits

    return batch_logits


# ============================================================================
# Threshold Sweep
# ============================================================================


def compute_transition_f1_for_video(
    scores: np.ndarray,
    gt_states: np.ndarray,
    sustain_hi: float,
    sustain_lo: float,
    sustain_min: int,
    tolerance: int = 3,
) -> float:
    """Compute per-video transition F1 for a given threshold config.

    Uses a standalone Q48 hysteresis decoder (not the config-bound
    MonotonicDecoder class) so we can sweep parameters freely.

    Args:
        scores: [T, 11] per-component logits.
        gt_states: [T, 11] binary GT states.
        sustain_hi, sustain_lo, sustain_min: Q48 hysteresis params.
        tolerance: frame tolerance for transition matching.

    Returns:
        macro-averaged F1 across components.
    """
    T, n_comp = scores.shape
    comp_f1s = []

    for c in range(n_comp):
        col = scores[:, c]
        gt_col = gt_states[:, c]

        # Decode this component's state sequence
        decoded_states = torch.zeros(T, dtype=torch.float32)
        current_state = 0.0
        sustain_counter = 0.0

        for t in range(T):
            # Apply sigmoid to get probability
            prob = 1.0 / (1.0 + math.exp(-min(col[t], 15.0)))

            if current_state == 0.0:
                # Check for transition 0->1
                above_lo = 1.0 if prob > sustain_lo else 0.0
                sustain_counter = sustain_counter * above_lo + above_lo
                if sustain_counter >= sustain_min and prob > sustain_hi:
                    current_state = 1.0
            # Once 1, stays 1 (monotonic)

            decoded_states[t] = current_state

        # Compute transition F1 for this component
        gt_bin = gt_col.astype(np.int32)
        gt_trans = list(np.where(np.diff(gt_bin, prepend=0) == 1)[0])
        pred_trans = list(np.where(np.diff(decoded_states.numpy(), prepend=0) == 1)[0])

        n_gt = len(gt_trans)
        n_pred = len(pred_trans)

        if n_gt == 0 and n_pred == 0:
            comp_f1s.append(1.0)
            continue
        if n_gt == 0 or n_pred == 0:
            comp_f1s.append(0.0)
            continue

        gt_matched = [False] * n_gt
        pred_matched = [False] * n_pred
        for gi, gf in enumerate(gt_trans):
            best_dist = tolerance + 1
            best_pi = -1
            for pi, pf in enumerate(pred_trans):
                if pred_matched[pi]:
                    continue
                dist = abs(pf - gf)
                if dist < best_dist:
                    best_dist = dist
                    best_pi = pi
            if best_pi >= 0:
                gt_matched[gi] = True
                pred_matched[best_pi] = True

        tp = sum(gt_matched)
        fp = n_pred - tp
        fn = n_gt - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r_val / (p + r_val)) if (p + r_val) > 0 else 0.0
        comp_f1s.append(f1)

    return float(np.mean(comp_f1s)) if comp_f1s else 0.0


def sweep_thresholds(
    collected_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    hi_values: list[float],
    lo_values: list[float],
    min_values: list[int],
    n_components: int = N_COMPONENTS,
) -> dict:
    """Sweep Q48 hysteresis thresholds and measure F1 for each combination.

    Returns:
        dict with best config, per-component stats, and full sweep log.
    """
    total_combos = len(hi_values) * len(lo_values) * len(min_values)
    logger.info(
        f"Sweeping {total_combos} threshold combinations over {len(collected_scores)} videos"
    )

    # Precompute per-component score distributions
    comp_scores: list[list[float]] = [[] for _ in range(n_components)]
    comp_gt: list[list[int]] = [[] for _ in range(n_components)]

    for scores, gt_states in collected_scores.values():
        for c in range(n_components):
            valid = gt_states[:, c] != -1
            comp_scores[c].extend(scores[valid, c].tolist())
            comp_gt[c].extend(gt_states[valid, c].tolist())

    comp_stats = {}
    for c in range(n_components):
        s = np.array(comp_scores[c])
        g = np.array(comp_gt[c])
        comp_stats[c] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(np.median(s)),
            "p25": float(np.percentile(s, 25)),
            "p75": float(np.percentile(s, 75)),
            "p90": float(np.percentile(s, 90)),
            "p95": float(np.percentile(s, 95)),
            "p99": float(np.percentile(s, 99)),
            "min": float(s.min()),
            "max": float(s.max()),
            "n_frames": len(s),
            "prevalence": float(g.mean()),
        }

    logger.info("Per-component score statistics:")
    for c in range(n_components):
        st = comp_stats[c]
        logger.info(
            f"  Comp {c}: mean={st['mean']:.3f} std={st['std']:.3f} "
            f"p50={st['median']:.3f} p95={st['p95']:.3f} "
            f"prevalence={st['prevalence']:.3f} n={st['n_frames']}"
        )

    # Sweep all combos
    best_f1 = -1.0
    best_config = {}
    sweep_log = []
    start_time = time.time()

    for hi in hi_values:
        for lo in lo_values:
            if lo >= hi:
                continue  # hysteresis invariant
            for mi in min_values:
                video_f1s = []
                for scores, gt_states in collected_scores.values():
                    vid_f1 = compute_transition_f1_for_video(scores, gt_states, hi, lo, mi)
                    video_f1s.append(vid_f1)

                mean_f1 = float(np.mean(video_f1s)) if video_f1s else 0.0
                entry = {
                    "sustain_hi": hi,
                    "sustain_lo": lo,
                    "sustain_min": mi,
                    "f1_at_t": mean_f1,
                    "n_videos": len(video_f1s),
                }
                sweep_log.append(entry)

                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_config = entry

    elapsed = time.time() - start_time
    logger.info(
        f"Sweep complete ({len(sweep_log)} combos, {elapsed:.0f}s): "
        f"best F1={best_f1:.4f} at hi={best_config['sustain_hi']:.2f} "
        f"lo={best_config['sustain_lo']:.2f} min={best_config['sustain_min']}"
    )

    # Per-component thresholds
    comp_thresholds = _compute_per_component_thresholds(comp_stats, best_config)

    return {
        "best": best_config,
        "per_component_thresholds": comp_thresholds,
        "comp_stats": comp_stats,
        "sweep_log": sorted(sweep_log, key=lambda x: -x["f1_at_t"]),
        "n_combos_tested": len(sweep_log),
        "elapsed_seconds": elapsed,
    }


def _compute_per_component_thresholds(
    comp_stats: dict,
    best_config: dict,
    n_components: int = N_COMPONENTS,
) -> dict:
    """Compute per-component thresholds based on score distributions.

    Adjusts sustain_hi and sustain_lo based on each component's score
    percentiles so that:
      - sustain_hi tracks p90-p95 range
      - sustain_lo tracks p25-p50 range
      - maintain hysteresis gap >= 0.15
    """
    hi_base = best_config.get("sustain_hi", 0.5)
    lo_base = best_config.get("sustain_lo", 0.25)
    min_base = best_config.get("sustain_min", 3)

    sustain_hi = []
    sustain_lo = []
    sustain_min = []

    for c in range(n_components):
        st = comp_stats[c]
        p50 = st["median"]
        prevalence = st["prevalence"]

        hi = min(max(hi_base * (1.0 + (0.5 - p50)), 0.2), 0.85)
        lo = min(max(lo_base * (1.0 - (prevalence - 0.5) * 0.3), 0.05), 0.45)

        if hi - lo < 0.15:
            mid = (hi + lo) / 2
            hi = mid + 0.075
            lo = mid - 0.075
            hi = min(hi, 0.9)
            lo = max(lo, 0.05)

        std = st["std"]
        mi = int(min_base + std * 2)
        mi = max(1, min(mi, 8))

        sustain_hi.append(round(hi, 3))
        sustain_lo.append(round(lo, 3))
        sustain_min.append(mi)

    return {
        "sustain_hi": sustain_hi,
        "sustain_lo": sustain_lo,
        "sustain_min": sustain_min,
    }


# ============================================================================
# Full D4 with Retuned Thresholds
# ============================================================================


def run_full_d4(
    collected_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: dict,
    output_dir: str,
) -> dict:
    """Run D4 evaluation with retuned per-component thresholds.

    Returns metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    sustain_hi = np.array(thresholds["sustain_hi"])
    sustain_lo = np.array(thresholds["sustain_lo"])
    sustain_min = np.array(thresholds["sustain_min"])
    n_components = len(sustain_hi)

    all_f1s = []
    all_precisions = []
    all_recalls = []
    per_video = {}

    # Per-component transition F1
    for video_id, (scores, gt_states) in collected_scores.items():
        T = scores.shape[0]
        comp_f1s = []

        for c in range(n_components):
            vid_f1 = compute_transition_f1_for_video(
                scores,
                gt_states,
                float(sustain_hi[c]),
                float(sustain_lo[c]),
                int(sustain_min[c]),
            )
            comp_f1s.append(vid_f1)

        mean_f1 = float(np.mean(comp_f1s))
        all_f1s.append(mean_f1)
        per_video[video_id] = {
            "f1_at_t": mean_f1,
            "n_frames": T,
        }

    overall = {
        "f1_at_t": float(np.mean(all_f1s)),
        "n_videos": len(all_f1s),
        "per_video": per_video,
        "thresholds": {
            "sustain_hi": [float(x) for x in sustain_hi],
            "sustain_lo": [float(x) for x in sustain_lo],
            "sustain_min": [int(x) for x in sustain_min],
        },
    }

    # Save
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(_serialize(overall), f, indent=2, default=str)
    logger.info(f"Retuned D4 metrics saved to {metrics_path}")

    return overall


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="D4 Threshold Retune: Sweep Q48 hysteresis for YOLOv8m"
    )
    parser.add_argument(
        "--yolo-ckpt",
        type=str,
        default="src/runs/rf_stages/checkpoints/yolov8m_industreal.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/runs/rf_stages/checkpoints/d4_retuned",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--detection-thresh", type=float, default=0.1)
    parser.add_argument(
        "--fine-grid", action="store_true", help="Use finer grid around best region"
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint path
    ckpt_path = Path(args.yolo_ckpt)
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        logger.warning("Will try to load from ultralytics COCO-pretrained YOLOv8m")

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Step 0: Load YOLOv8m ────────────────────────────────────────────
    logger.info("Step 0: Loading YOLOv8m model")
    try:
        from ultralytics import YOLO

        if ckpt_path.exists():
            logger.info(f"Loading YOLOv8m from: {ckpt_path}")
            yolo = YOLO(str(ckpt_path))
        else:
            logger.info("Loading COCO-pretrained YOLOv8m")
            yolo = YOLO("yolov8m.pt")
    except ImportError:
        logger.error("ultralytics not installed. Cannot run YOLO inference.")
        sys.exit(1)

    # ── Step 1: Build val dataset ────────────────────────────────────────
    logger.info("Step 1: Building val dataset")
    val_dataset = IndustRealMultiTaskDataset(
        split="val",
        img_size=(C.IMG_WIDTH, C.IMG_HEIGHT),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # ── Step 2: Collect YOLOv8m scores ───────────────────────────────────
    logger.info("Step 2: Collecting YOLOv8m -> PSR scores on val set")
    collected_scores = collect_yolo_psr_scores(
        yolo,
        val_loader,
        max_batches=args.max_batches,
        device=device,
        detection_thresh=args.detection_thresh,
    )

    if not collected_scores:
        logger.error("No scores collected! Aborting.")
        sys.exit(1)

    logger.info(f"Collected scores for {len(collected_scores)} recordings")

    # ── Step 3: Sweep thresholds ─────────────────────────────────────────
    logger.info("Step 3: Sweeping Q48 hysteresis thresholds")
    hi_values = SUSTAIN_HI_FINE if args.fine_grid else SUSTAIN_HI_VALUES
    lo_values = SUSTAIN_LO_FINE if args.fine_grid else SUSTAIN_LO_VALUES
    min_values = SUSTAIN_MIN_FINE if args.fine_grid else SUSTAIN_MIN_VALUES

    sweep_results = sweep_thresholds(
        collected_scores,
        hi_values,
        lo_values,
        min_values,
    )

    # Save sweep results
    sweep_path = os.path.join(output_dir, "sweep_results.json")
    with open(sweep_path, "w") as f:
        json.dump(_serialize(sweep_results), f, indent=2, default=str)
    logger.info(f"Sweep results saved to {sweep_path}")

    # ── Step 4: Run D4 with retuned per-component thresholds ─────────────
    logger.info("Step 4: Running D4 with retuned per-component thresholds")
    d4_metrics = run_full_d4(
        collected_scores,
        sweep_results["per_component_thresholds"],
        str(output_dir),
    )

    # Save thresholds
    thresh_path = os.path.join(output_dir, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(
            _serialize(sweep_results["per_component_thresholds"]),
            f,
            indent=2,
        )
    logger.info(f"Retuned thresholds saved to {thresh_path}")

    # ── Step 5: Print summary and verdict ────────────────────────────────
    f1_global = sweep_results["best"]["f1_at_t"]
    f1_retuned = d4_metrics["f1_at_t"]

    print(f"\n{'=' * 70}")
    print(f"  D4 Threshold Retuning — Complete")
    print(f"{'=' * 70}")
    print(f"  Recordings processed:  {len(collected_scores)}")
    print(f"  Combos tested:         {sweep_results['n_combos_tested']}")
    print(f"  Elapsed:               {sweep_results['elapsed_seconds']:.0f}s")
    print()
    print(f"  Original D4 (config defaults):")
    print(f"    sustain_hi=0.5, sustain_lo=0.3, sustain_min=3")
    print(f"    F1 = 0.000  (from existing eval)")
    print()
    print(f"  Best global sweep config:")
    print(f"    sustain_hi:         {sweep_results['best']['sustain_hi']:.2f}")
    print(f"    sustain_lo:         {sweep_results['best']['sustain_lo']:.2f}")
    print(f"    sustain_min:        {sweep_results['best']['sustain_min']}")
    print(f"    F1@t=3:             {f1_global:.4f}")
    print()
    print(f"  Retuned (per-component) D4 results:")
    print(f"    F1@t=3:             {f1_retuned:.4f}")
    print()
    print(f"  Top-10 sweep results:")
    for i, entry in enumerate(sweep_results["sweep_log"][:10]):
        print(
            f"    {i + 1:2d}. hi={entry['sustain_hi']:.2f} lo={entry['sustain_lo']:.2f} "
            f"min={entry['sustain_min']}  F1={entry['f1_at_t']:.4f}"
        )
    print()
    print(f"  Per-component thresholds:")
    for c in range(N_COMPONENTS):
        print(
            f"    Comp {c:2d}: hi="
            f"{sweep_results['per_component_thresholds']['sustain_hi'][c]:.3f} "
            f"lo="
            f"{sweep_results['per_component_thresholds']['sustain_lo'][c]:.3f} "
            f"min="
            f"{sweep_results['per_component_thresholds']['sustain_min'][c]}"
        )
    print()

    # Verdict
    if f1_retuned > 0.5:
        verdict = (
            "threshold-recalibration: decoder requires threshold recalibration for YOLOv8m features"
        )
    elif f1_retuned > 0.0:
        verdict = "threshold-partial: decoder shows marginal benefit — thresholds partially helpful"
    else:
        verdict = "redundant: decoder is redundant even with retuned thresholds"

    print(f"  VERDICT: {verdict}")
    print(f"{'=' * 70}")

    verdict_data = {
        "f1_at_t_original": 0.0,
        "f1_at_t_best_global": f1_global,
        "f1_at_t_retuned": f1_retuned,
        "verdict": verdict,
        "n_videos": len(collected_scores),
        "best_global_config": {
            "sustain_hi": sweep_results["best"]["sustain_hi"],
            "sustain_lo": sweep_results["best"]["sustain_lo"],
            "sustain_min": sweep_results["best"]["sustain_min"],
        },
    }
    verdict_path = os.path.join(output_dir, "verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict_data, f, indent=2)
    logger.info(f"Verdict saved to {verdict_path}")

    # Print markdown table for reporting
    print()
    print("--- Sweep Results Table ---")
    print("| hi | lo | min | F1 |")
    print("|---|---|---|---|")
    for entry in sweep_results["sweep_log"][:20]:
        print(
            f"| {entry['sustain_hi']:.2f} | {entry['sustain_lo']:.2f} "
            f"| {entry['sustain_min']} | {entry['f1_at_t']:.4f} |"
        )


if __name__ == "__main__":
    main()
