"""
ConvNeXt -> decoder PSR evaluation (Opus 141 Q38).
=====================================================================
Loads the multi-task ConvNeXt-Tiny model, forward-passes the
validation set, extracts PSR head logits, and runs them through the
MonotonicDecoder with Q48 hysteresis thresholds (same as D4).

Three evaluations:
  1. Default thresholds (hi=0.5, lo=0.3, min=3)
  2. Threshold sweep across sustain_hi/lo/min (same grid as d4_threshold_retune.py)
  3. Re-tuned per-component thresholds

OOM SAFEGUARD: defaults to max_batches=500 (not full 38k), runs CPU-only.

Output:
  metrics.json       — consolidated F1 scores (default, best-global, per-comp retuned)
  2x2_table.md        — comparison table with YOLOv8m (D4) and D1R cells
  per_video.json      — per-recording breakdown
  sweep_results.json  — full sweep log
  thresholds.json     — per-component retuned thresholds
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# -- Path setup ---------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent
for _sub in ["models", "training", "evaluation", "data", str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src import config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.psr_transition import MonotonicDecoder

logger = logging.getLogger("convnext_psr_decoder")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

N_COMPONENTS = 11

# -- Q48 Default thresholds (same as D4) --------------------------------------
DEFAULT_HI = float(getattr(C, "PSR_TRANSITION_THRESHOLD_HI", 0.5))
DEFAULT_LO = float(getattr(C, "PSR_TRANSITION_THRESHOLD_LO", 0.3))
DEFAULT_MIN = int(getattr(C, "PSR_TRANSITION_MIN_SUSTAINED", 3))

# Sweep ranges (same as d4_threshold_retune.py)
SUSTAIN_HI_VALUES = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
SUSTAIN_LO_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3]
SUSTAIN_MIN_VALUES = [2, 3, 4, 5, 6]

SUSTAIN_HI_FINE = list(np.round(np.arange(0.3, 0.75, 0.05), 2))
SUSTAIN_LO_FINE = list(np.round(np.arange(0.1, 0.35, 0.05), 2))
SUSTAIN_MIN_FINE = list(range(2, 8))


# =============================================================================
# Model loading
# =============================================================================
def load_model(ckpt_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load ConvNeXt multi-task model onto device."""
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info(
        "Epoch: %s, best_metric: %s",
        ckpt.get("epoch"), ckpt.get("best_metric", "?"),
    )

    from src.models.model import POPWMultiTaskModel
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type="convnext_tiny",
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=False,
    )
    state_dict = {
        k: v for k, v in ckpt["model"].items()
        if "total_ops" not in k and "total_params" not in k
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))
    model._seq_len = 1
    model = model.to(device).eval()
    return model


# =============================================================================
# Score collection from ConvNeXt PSR head
# =============================================================================
def collect_convnext_psr_scores(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    max_batches: int = 500,
    device: str = "cpu",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Collect PSR logits and GT states grouped by recording.

    Args:
        model: POPWMultiTaskModel (eval mode).
        val_loader: DataLoader for val set.
        max_batches: cap on batches (0 = unlimited).
        device: target device.

    Returns:
        Dict mapping recording_id -> (logits [T, 11], gt_states [T, 11])
    """
    all_psr_logits: List[np.ndarray] = []
    all_psr_labels: List[np.ndarray] = []
    all_rec_ids: List[str] = []
    all_frame_nums: List[int] = []

    logger.info("Collecting ConvNeXt PSR scores on val set (device=%s)", device)

    for bi, (images, targets) in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break

        images = images.to(device).float()
        if images.max() > 1.0:
            images = images.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            outputs = model(images)

        pl = outputs.get("psr_logits")
        if pl is None:
            logger.warning("Batch %d: no psr_logits in model output", bi)
            gc.collect()
            continue

        psr_logits_batch = pl.cpu().numpy().astype(np.float32)
        all_psr_logits.append(psr_logits_batch)

        psr_labels_batch = targets["psr_labels"].cpu().numpy().astype(np.float32)
        all_psr_labels.append(psr_labels_batch)

        for i in range(images.shape[0]):
            metadata_item = (
                targets["metadata"][i]
                if i < len(targets.get("metadata", []))
                else {}
            )
            rec_id = metadata_item.get(
                "recording_id",
                metadata_item.get("rec_id", f"batch{bi}_i{i}"),
            )
            if isinstance(rec_id, torch.Tensor):
                rec_id = str(rec_id.item())
            else:
                rec_id = str(rec_id)
            all_rec_ids.append(rec_id)

            frame_num = metadata_item.get(
                "frame_num", metadata_item.get("frame_idx", 0)
            )
            if isinstance(frame_num, torch.Tensor):
                frame_num = frame_num.item()
            all_frame_nums.append(int(frame_num))

        if bi % 50 == 0:
            logger.info("  Batch %d: %d images", bi, images.shape[0])

        del images, targets, outputs
        gc.collect()

    # -- Group by recording --------------------------------------------------
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
            rec = (
                all_rec_ids[flat_i]
                if flat_i < len(all_rec_ids)
                else f"rec_{flat_i}"
            )
            fn = (
                all_frame_nums[flat_i]
                if flat_i < len(all_frame_nums)
                else flat_i
            )
            by_rec_logits.setdefault(rec, []).append(bl[row, :N_COMPONENTS])
            by_rec_gt.setdefault(rec, []).append(
                lb[row, :N_COMPONENTS] if row < lb.shape[0] else None
            )
            by_rec_fn.setdefault(rec, []).append(fn)
            flat_i += 1

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for rec, rows in by_rec_logits.items():
        gts = by_rec_gt[rec]
        if any(g is None for g in gts) or len(rows) < 2:
            continue
        order = np.argsort(
            np.asarray(by_rec_fn[rec], dtype=np.int64), kind="stable"
        )
        logits = np.stack([rows[k] for k in order]).astype(np.float32)
        states = np.stack([gts[k] for k in order]).astype(np.float32)
        result[rec] = (logits, states)

    logger.info(
        "Collected %d recordings (total frames: %d)", len(result), flat_i
    )
    return result


# =============================================================================
# Transition F1 helpers (same as d4_threshold_retune.py / eval_convnext_psr.py)
# =============================================================================
def compute_transition_f1_for_video(
    scores: np.ndarray,
    gt_states: np.ndarray,
    sustain_hi: float,
    sustain_lo: float,
    sustain_min: int,
    tolerance: int = 3,
) -> float:
    """Compute per-video macro-averaged transition F1 for a given threshold config.

    Uses the same standalone Q48 hysteresis decoder as d4_threshold_retune.py.
    """
    T, n_comp = scores.shape
    comp_f1s = []

    for c in range(n_comp):
        col = scores[:, c]
        gt_col = gt_states[:, c]

        decoded_states = torch.zeros(T, dtype=torch.float32)
        current_state = 0.0
        sustain_counter = 0.0

        for t in range(T):
            prob = 1.0 / (1.0 + math.exp(-min(col[t], 15.0)))

            if current_state == 0.0:
                above_lo = 1.0 if prob > sustain_lo else 0.0
                sustain_counter = sustain_counter * above_lo + above_lo
                if sustain_counter >= sustain_min and prob > sustain_hi:
                    current_state = 1.0

            decoded_states[t] = current_state

        gt_bin = gt_col.astype(np.int32)
        gt_trans = list(np.where(np.diff(gt_bin, prepend=0) == 1)[0])
        pred_trans = list(
            np.where(np.diff(decoded_states.numpy(), prepend=0) == 1)[0]
        )

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
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
        comp_f1s.append(f1)

    return float(np.mean(comp_f1s)) if comp_f1s else 0.0


def compute_per_component_f1(
    collected_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sustain_hi: float = DEFAULT_HI,
    sustain_lo: float = DEFAULT_LO,
    sustain_min: int = DEFAULT_MIN,
    tolerance: int = 3,
) -> Dict[str, Any]:
    """Compute per-component and macro F1 with given thresholds."""
    comp_f1s_sum = [0.0] * N_COMPONENTS
    n_videos = 0
    all_f1s = []
    per_video: Dict[str, Dict[str, Any]] = {}

    for video_id, (scores, gt_states) in collected_scores.items():
        n_videos += 1
        video_comp_f1s = []
        for c in range(N_COMPONENTS):
            vf1 = compute_transition_f1_for_video(
                scores, gt_states, sustain_hi, sustain_lo, sustain_min, tolerance,
            )
            video_comp_f1s.append(vf1)
            comp_f1s_sum[c] += vf1
        mean_f1 = float(np.mean(video_comp_f1s))
        all_f1s.append(mean_f1)
        per_video[video_id] = {"f1_at_t": mean_f1, "n_frames": scores.shape[0]}

    per_comp = {
        f"comp_{c}": round(s / max(n_videos, 1), 4)
        for c, s in enumerate(comp_f1s_sum)
    }

    return {
        "f1_at_t": float(np.mean(all_f1s)) if all_f1s else 0.0,
        "per_component_f1": per_comp,
        "per_video": per_video,
        "n_videos": n_videos,
        "thresholds": {
            "sustain_hi": sustain_hi,
            "sustain_lo": sustain_lo,
            "sustain_min": sustain_min,
        },
    }


def sweep_thresholds(
    collected_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    hi_values: List[float],
    lo_values: List[float],
    min_values: List[int],
) -> Dict[str, Any]:
    """Sweep Q48 hysteresis thresholds and find best global config."""
    total_combos = len(hi_values) * len(lo_values) * len(min_values)
    logger.info(
        "Sweeping %d threshold combinations over %d recordings",
        total_combos, len(collected_scores),
    )

    # Precompute per-component score statistics
    comp_scores: List[List[float]] = [[] for _ in range(N_COMPONENTS)]
    comp_gt: List[List[int]] = [[] for _ in range(N_COMPONENTS)]

    for scores, gt_states in collected_scores.values():
        for c in range(N_COMPONENTS):
            valid = gt_states[:, c] != -1
            comp_scores[c].extend(scores[valid, c].tolist())
            comp_gt[c].extend(gt_states[valid, c].tolist())

    comp_stats = {}
    for c in range(N_COMPONENTS):
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
    for c in range(N_COMPONENTS):
        st = comp_stats[c]
        logger.info(
            "  Comp %d: mean=%.3f std=%.3f p50=%.3f p95=%.3f prevalence=%.3f n=%d",
            c, st["mean"], st["std"], st["median"], st["p95"],
            st["prevalence"], st["n_frames"],
        )

    best_f1 = -1.0
    best_config: Dict[str, Any] = {}
    sweep_log: List[Dict[str, Any]] = []
    start_time = time.time()

    for hi in hi_values:
        for lo in lo_values:
            if lo >= hi:
                continue
            for mi in min_values:
                video_f1s = []
                for scores, gt_states in collected_scores.values():
                    vid_f1 = compute_transition_f1_for_video(
                        scores, gt_states, hi, lo, mi,
                    )
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
        "Sweep complete (%d combos, %.0fs): best F1=%.4f at hi=%.2f lo=%.2f min=%d",
        len(sweep_log), elapsed, best_f1,
        best_config["sustain_hi"], best_config["sustain_lo"],
        best_config["sustain_min"],
    )

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
    comp_stats: Dict,
    best_config: Dict,
) -> Dict:
    """Compute per-component thresholds based on score distributions."""
    hi_base = best_config.get("sustain_hi", 0.5)
    lo_base = best_config.get("sustain_lo", 0.25)
    min_base = best_config.get("sustain_min", 3)

    sustain_hi = []
    sustain_lo = []
    sustain_min = []

    for c in range(N_COMPONENTS):
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

        std_val = st["std"]
        mi = int(min_base + std_val * 2)
        mi = max(1, min(mi, 8))

        sustain_hi.append(round(hi, 3))
        sustain_lo.append(round(lo, 3))
        sustain_min.append(mi)

    return {
        "sustain_hi": sustain_hi,
        "sustain_lo": sustain_lo,
        "sustain_min": sustain_min,
    }


# =============================================================================
# Serialization helper
# =============================================================================
def _serialize(obj: Any) -> Any:
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


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="ConvNeXt -> decoder PSR evaluation (Opus 141 Q38)"
    )
    parser.add_argument(
        "--ckpt", type=str,
        default="src/runs/rf_stages/checkpoints/best.pth",
        help="Path to multi-task ConvNeXt checkpoint",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="src/runs/rf_stages/checkpoints/convnext_psr_decoder",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-batches", type=int, default=500,
        help="Max batches (default 500 for OOM safety; 0 = full 38k)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (use 'cuda:0' or 'cuda:1' for GPU)",
    )
    parser.add_argument("--fine-grid", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(args.ckpt):
        logger.error("Checkpoint not found: %s", args.ckpt)
        sys.exit(1)

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # -- Step 1: Load model ---------------------------------------------------
    logger.info("Step 1: Loading ConvNeXt multi-task model on %s", device)
    model = load_model(args.ckpt, device=device)

    # -- Step 2: Build val dataset --------------------------------------------
    logger.info("Step 2: Building val dataset")
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
    logger.info(
        "Val dataset: %d batches (batch_size=%d, max_batches=%s, full=%d)",
        len(val_loader), args.batch_size,
        "all" if args.max_batches == 0 else str(args.max_batches),
        len(val_loader) * args.batch_size,
    )

    # -- Step 3: Collect ConvNeXt PSR scores ----------------------------------
    logger.info("Step 3: Collecting ConvNeXt PSR scores on val set")
    collected_scores = collect_convnext_psr_scores(
        model, val_loader, max_batches=args.max_batches, device=device,
    )

    if not collected_scores:
        logger.error("No scores collected! Aborting.")
        sys.exit(1)

    logger.info("Collected scores for %d recordings", len(collected_scores))
    total_frames = sum(s.shape[0] for s, _ in collected_scores.values())
    logger.info("Total frames collected: %d", total_frames)

    # -- Step 4a: Score with default decoder ----------------------------------
    logger.info(
        "Step 4a: Default decoder thresholds (hi=%.1f, lo=%.1f, min=%d)",
        DEFAULT_HI, DEFAULT_LO, DEFAULT_MIN,
    )
    default_explicit = compute_per_component_f1(
        collected_scores,
        sustain_hi=DEFAULT_HI,
        sustain_lo=DEFAULT_LO,
        sustain_min=DEFAULT_MIN,
        tolerance=3,
    )
    logger.info("Default decoder - F1: %.4f", default_explicit["f1_at_t"])

    # -- Step 5: Sweep thresholds ---------------------------------------------
    logger.info("Step 5: Sweeping Q48 hysteresis thresholds")
    hi_values = SUSTAIN_HI_FINE if args.fine_grid else SUSTAIN_HI_VALUES
    lo_values = SUSTAIN_LO_FINE if args.fine_grid else SUSTAIN_LO_VALUES
    min_values = SUSTAIN_MIN_FINE if args.fine_grid else SUSTAIN_MIN_VALUES

    sweep_results = sweep_thresholds(
        collected_scores, hi_values, lo_values, min_values,
    )

    sweep_path = save_dir / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(_serialize(sweep_results), f, indent=2, default=str)
    logger.info("Sweep results saved to %s", sweep_path)

    # -- Step 6: Best global threshold ----------------------------------------
    logger.info("Step 6: Running with best global sweep threshold")
    best_config = sweep_results["best"]
    best_global = compute_per_component_f1(
        collected_scores,
        sustain_hi=best_config["sustain_hi"],
        sustain_lo=best_config["sustain_lo"],
        sustain_min=best_config["sustain_min"],
        tolerance=3,
    )

    # -- Step 7: Per-component retuned thresholds -----------------------------
    logger.info("Step 7: Running with per-component retuned thresholds")
    comp_thresholds = sweep_results["per_component_thresholds"]
    retuned_f1s = []
    retuned_per_video: Dict[str, Dict[str, Any]] = {}
    for video_id, (scores, gt_states) in collected_scores.items():
        comp_f1s = []
        for c in range(N_COMPONENTS):
            vf1 = compute_transition_f1_for_video(
                scores, gt_states,
                float(comp_thresholds["sustain_hi"][c]),
                float(comp_thresholds["sustain_lo"][c]),
                int(comp_thresholds["sustain_min"][c]),
                tolerance=3,
            )
            comp_f1s.append(vf1)
        mean_f1 = float(np.mean(comp_f1s))
        retuned_f1s.append(mean_f1)
        retuned_per_video[video_id] = {
            "f1_at_t": mean_f1,
            "n_frames": scores.shape[0],
        }
    retuned_f1 = float(np.mean(retuned_f1s)) if retuned_f1s else 0.0

    retuned_metrics = {
        "f1_at_t": retuned_f1,
        "n_videos": len(retuned_f1s),
        "per_video": retuned_per_video,
        "thresholds": {
            "sustain_hi": [float(x) for x in comp_thresholds["sustain_hi"]],
            "sustain_lo": [float(x) for x in comp_thresholds["sustain_lo"]],
            "sustain_min": [int(x) for x in comp_thresholds["sustain_min"]],
        },
    }

    # Save thresholds
    thresh_path = save_dir / "thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(_serialize(sweep_results["per_component_thresholds"]), f, indent=2)
    logger.info("Retuned thresholds saved to %s", thresh_path)

    # -- Step 8: Save consolidated metrics ------------------------------------
    logger.info("Step 8: Saving consolidated metrics")
    f1_default = default_explicit["f1_at_t"]
    f1_best_global = best_global["f1_at_t"]
    f1_retuned = retuned_f1

    metrics = {
        "checkpoint": args.ckpt,
        "device": device,
        "max_batches": args.max_batches,
        "total_frames": total_frames,
        "n_recordings": len(collected_scores),
        "default_decoder": {
            "f1_at_t": f1_default,
            "per_component_f1": default_explicit["per_component_f1"],
            "thresholds": {
                "sustain_hi": DEFAULT_HI,
                "sustain_lo": DEFAULT_LO,
                "sustain_min": DEFAULT_MIN,
            },
        },
        "best_global_sweep": {
            "f1_at_t": f1_best_global,
            "thresholds": {
                "sustain_hi": best_config["sustain_hi"],
                "sustain_lo": best_config["sustain_lo"],
                "sustain_min": best_config["sustain_min"],
            },
        },
        "retuned_per_component": {
            "f1_at_t": f1_retuned,
            "n_videos": retuned_metrics["n_videos"],
            "thresholds": retuned_metrics["thresholds"],
        },
        "sweep_summary": {
            "n_combos_tested": sweep_results["n_combos_tested"],
            "elapsed_seconds": sweep_results["elapsed_seconds"],
        },
    }

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_serialize(metrics), f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    # -- Step 9: Per-video breakdown ------------------------------------------
    per_video_all: Dict[str, Any] = {}
    for vid in collected_scores:
        pv = {}
        if vid in default_explicit.get("per_video", {}):
            pv["default"] = default_explicit["per_video"][vid]
        if vid in best_global.get("per_video", {}):
            pv["best_global"] = best_global["per_video"][vid]
        if vid in retuned_metrics.get("per_video", {}):
            pv["retuned"] = retuned_metrics["per_video"][vid]
        if pv:
            per_video_all[vid] = pv
    pv_path = save_dir / "per_video.json"
    with open(pv_path, "w") as f:
        json.dump(_serialize(per_video_all), f, indent=2, default=str)

    # -- Step 10: Generate 2x2 comparison table -------------------------------
    logger.info("Step 10: Generating 2x2 comparison table")

    # Reference values from D4 / D4+D1R:
    # D4 (YOLOv8m) default = 0.000
    # D4 retuned per-comp = 0.2614 (from d4_retuned/metrics.json)
    # D4+D1R default = 0.000
    # D4+D1R best global = 0.6364 (hi=0.3, lo=0.1, min=2)
    # D4+D1R retuned per-comp = 0.1956
    d4_default_f1 = 0.000
    d4_retuned_f1 = None
    d4_d1r_default_f1 = 0.000
    d4_d1r_best_f1 = 0.6364
    d4_d1r_retuned_f1 = 0.1956

    # Try to load actual retuned value from d4_retuned
    d4_retuned_path = Path(
        "src/runs/rf_stages/checkpoints/d4_retuned/metrics.json"
    )
    if d4_retuned_path.exists():
        try:
            with open(d4_retuned_path) as f:
                d4r_data = json.load(f)
            d4_retuned_f1 = d4r_data.get("f1_at_t", None)
        except Exception:
            pass

    # Try to load d4_d1r metrics
    d4_d1r_metrics_path = Path(
        "src/runs/rf_stages/checkpoints/d4_d1r/metrics.json"
    )
    if d4_d1r_metrics_path.exists():
        try:
            with open(d4_d1r_metrics_path) as f:
                dd = json.load(f)
            d4_d1r_default_f1 = dd.get("psr_f1_default", d4_d1r_default_f1)
            d4_d1r_best_f1 = dd.get(
                "psr_f1_retuned_global", d4_d1r_best_f1
            )
            d4_d1r_retuned_f1 = dd.get(
                "psr_f1_retuned_per_component", d4_d1r_retuned_f1
            )
        except Exception:
            pass

    table_lines = [
        "# 2x2 PSR Decoder Comparison Table\n",
        "\n",
        "Generated by `convnext_psr_decoder.py` (Opus 141 Q38)\n",
        "\n",
        "## Compact 2x2 (Backbone x Decoder-strictness)\n",
        "\n",
        "| | Default decoder (hi=0.5, lo=0.3, min=3) | Re-tuned decoder |\n",
        "|---|---|---|\n",
    ]

    d4_retuned_str = "%.4f" % d4_retuned_f1 if d4_retuned_f1 is not None else "?"
    table_lines.append(
        "| YOLOv8m (D4) | %.4f | %s |\n" % (d4_default_f1, d4_retuned_str)
    )
    table_lines.append(
        "| ConvNeXt-Tiny | %.4f | %.4f |\n" % (f1_default, f1_retuned)
    )
    table_lines.append(
        "| D1R (oracle det.) | %.4f | %.4f |\n" % (
            d4_d1r_default_f1, d4_d1r_best_f1
        )
    )

    table_lines.append("\n")
    table_lines.append("## Full Results\n")
    table_lines.append("\n")
    table_lines.append("| Backbone | Decoder config | F1@t=3 |\n")
    table_lines.append("|---|---|---|\n")

    table_lines.append(
        "| YOLOv8m (D4) | Default (hi=0.5, lo=0.3, min=3)"
        " | %.4f |\n" % d4_default_f1
    )
    if d4_retuned_f1 is not None:
        table_lines.append(
            "| YOLOv8m (D4) | Per-component retuned"
            " | %.4f |\n" % d4_retuned_f1
        )
    table_lines.append(
        "| ConvNeXt-Tiny | Default (hi=0.5, lo=0.3, min=3)"
        " | %.4f |\n" % f1_default
    )
    table_lines.append(
        "| ConvNeXt-Tiny | Best global (hi=%.2f, lo=%.2f, min=%d)"
        " | %.4f |\n" % (
            best_config["sustain_hi"], best_config["sustain_lo"],
            best_config["sustain_min"], f1_best_global,
        )
    )
    table_lines.append(
        "| ConvNeXt-Tiny | Per-component retuned"
        " | %.4f |\n" % f1_retuned
    )
    table_lines.append(
        "| D1R (oracle det.) | Default | %.4f |\n" % d4_d1r_default_f1
    )
    table_lines.append(
        "| D1R (oracle det.) | Best global (hi=0.30, lo=0.10, min=2)"
        " | %.4f |\n" % d4_d1r_best_f1
    )
    table_lines.append(
        "| D1R (oracle det.) | Per-component retuned"
        " | %.4f |\n" % d4_d1r_retuned_f1
    )

    # -- Attribution analysis ------------------------------------------------
    table_lines.append("\n")
    table_lines.append("## Attribution Analysis\n")
    table_lines.append("\n")

    gap_yolo_default_to_convnext_default = f1_default - d4_default_f1
    gap_convnext_default_to_convnext_retuned = f1_retuned - f1_default
    gap_total = f1_retuned - d4_default_f1

    pct_backbone = (
        100 * gap_yolo_default_to_convnext_default / max(gap_total, 1e-9)
    )
    pct_decoder = (
        100 * gap_convnext_default_to_convnext_retuned / max(gap_total, 1e-9)
    )

    table_lines.append(
        "- Backbone effect (YOLOv8m to ConvNeXt, same default decoder): "
        "%+.4f\n" % gap_yolo_default_to_convnext_default
    )
    table_lines.append(
        "- Decoder-strictness effect (ConvNeXt default to retuned): "
        "%+.4f\n" % gap_convnext_default_to_convnext_retuned
    )
    table_lines.append(
        "- Total improvement (YOLOv8m default to ConvNeXt retuned): "
        "%+.4f\n" % gap_total
    )
    table_lines.append(
        "- Backbone contribution: %.1f%%\n" % pct_backbone
    )
    table_lines.append(
        "- Decoder-strictness contribution: %.1f%%\n" % pct_decoder
    )

    # -- Per-component F1 -----------------------------------------------------
    table_lines.append("\n")
    table_lines.append("## Per-Component F1 (Default Decoder)\n")
    table_lines.append("\n")
    table_lines.append("| Component | F1 |\n")
    table_lines.append("|---|---|\n")
    for c in range(N_COMPONENTS):
        cf1 = default_explicit["per_component_f1"].get("comp_%d" % c, 0.0)
        table_lines.append("| %d | %.4f |\n" % (c, cf1))

    # -- Top sweep results ----------------------------------------------------
    table_lines.append("\n")
    table_lines.append("## Top Sweep Results\n")
    table_lines.append("\n")
    table_lines.append("| hi | lo | min | F1@t=3 |\n")
    table_lines.append("|---|---|---|---|\n")
    for entry in sweep_results["sweep_log"][:15]:
        table_lines.append(
            "| %.2f | %.2f | %d | %.4f |\n" % (
                entry["sustain_hi"], entry["sustain_lo"],
                entry["sustain_min"], entry["f1_at_t"],
            )
        )

    table_path = save_dir / "2x2_table.md"
    with open(table_path, "w") as f:
        f.writelines(table_lines)
    logger.info("2x2 table saved to %s", table_path)

    # -- Print summary --------------------------------------------------------
    print("\n%s" % ("=" * 70))
    print("  ConvNeXt -> decoder PSR Evaluation - Complete")
    print("%s" % ("=" * 70))
    print("  Checkpoint:            %s" % args.ckpt)
    print("  Device:                %s" % device)
    print("  Frames processed:     %d" % total_frames)
    print("  Recordings processed:  %d" % len(collected_scores))
    print()
    print("  Default decoder (hi=%.1f, lo=%.1f, min=%d):" % (
        DEFAULT_HI, DEFAULT_LO, DEFAULT_MIN,
    ))
    print("    F1@t=3:  %.4f" % f1_default)
    print()
    print("  Best global sweep:")
    print("    hi=%.2f, lo=%.2f, min=%d" % (
        best_config["sustain_hi"], best_config["sustain_lo"],
        best_config["sustain_min"],
    ))
    print("    F1@t=3:  %.4f" % f1_best_global)
    print()
    print("  Retuned per-component:")
    print("    F1@t=3:  %.4f" % f1_retuned)
    print()
    print("  Sweep: %d combos in %.0fs" % (
        sweep_results["n_combos_tested"],
        sweep_results["elapsed_seconds"],
    ))
    print()
    print("  Per-component thresholds:")
    for c in range(N_COMPONENTS):
        print("    Comp %2d: hi=%.3f lo=%.3f min=%d" % (
            c, comp_thresholds["sustain_hi"][c],
            comp_thresholds["sustain_lo"][c],
            comp_thresholds["sustain_min"][c],
        ))
    print()
    print("  Per-component F1 (default decoder):")
    for c in range(N_COMPONENTS):
        cf1 = default_explicit["per_component_f1"].get("comp_%d" % c, 0.0)
        print("    Comp %2d: %.4f" % (c, cf1))
    print()

    print("  Top-10 sweep results:")
    for i, entry in enumerate(sweep_results["sweep_log"][:10]):
        print("    %2d. hi=%.2f lo=%.2f min=%d  F1=%.4f" % (
            i + 1, entry["sustain_hi"], entry["sustain_lo"],
            entry["sustain_min"], entry["f1_at_t"],
        ))
    print()
    print("  === 2x2 TABLE ===")
    print("  | | Default | Retuned |")
    print("  |---|---|---|")
    d4_retuned_print = "%.4f" % d4_retuned_f1 if d4_retuned_f1 is not None else "?"
    print("  | YOLOv8m (D4) | %.4f | %s |" % (d4_default_f1, d4_retuned_print))
    print("  | ConvNeXt-Tiny | %.4f | %.4f |" % (f1_default, f1_retuned))
    print("  | D1R (oracle) | %.4f | %.4f |" % (d4_d1r_default_f1, d4_d1r_best_f1))
    print()
    print("%s" % ("=" * 70))


if __name__ == "__main__":
    main()
