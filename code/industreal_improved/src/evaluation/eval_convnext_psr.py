"""
eval_convnext_psr.py — ConvNeXt->decoder PSR evaluation (Opus 141 Q38)
=======================================================================
Loads the multi-task ConvNeXt-Tiny model (best.pth), forward-passes the
validation set, extracts PSR head logits, and runs them through the
MonotonicDecoder with configurable hysteresis thresholds (Q48).

Performs three evaluations:
  1. Default thresholds (hi=0.5, lo=0.3, min=3)
  2. Threshold sweep across sustain_hi/lo/min
  3. Re-tuned per-component thresholds (adapted from d4_threshold_retune.py)

Usage:
    python3 src/evaluation/eval_convnext_psr.py --ckpt <best.pth> [--max-batches 0]

Output saved to:
    src/runs/rf_stages/checkpoints/convnext_psr_decoder/metrics.json
    src/runs/rf_stages/checkpoints/convnext_psr_decoder/2x2_table.md
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# - Path setup ------------------------------------------------------------
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

logger = logging.getLogger("eval_convnext_psr")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

N_COMPONENTS = 11

# - Default thresholds (Q48 defaults from config) -------------------------
DEFAULT_HI = float(getattr(C, "PSR_TRANSITION_THRESHOLD_HI", 0.5))
DEFAULT_LO = float(getattr(C, "PSR_TRANSITION_THRESHOLD_LO", 0.3))
DEFAULT_MIN = int(getattr(C, "PSR_TRANSITION_MIN_SUSTAINED", 3))

# - Sweep ranges (from d4_threshold_retune.py) ----------------------------
SUSTAIN_HI_VALUES = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
SUSTAIN_LO_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3]
SUSTAIN_MIN_VALUES = [2, 3, 4, 5, 6]

SUSTAIN_HI_FINE = list(np.round(np.arange(0.3, 0.75, 0.05), 2))
SUSTAIN_LO_FINE = list(np.round(np.arange(0.1, 0.35, 0.05), 2))
SUSTAIN_MIN_FINE = list(range(2, 8))


# =========================================================================
# Model loading
# =========================================================================
def load_model(ckpt_path: str) -> torch.nn.Module:
    """Load ConvNeXt multi-task model from checkpoint."""
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    logger.info(
        "Epoch: %s, best_metric: %s",
        ckpt.get("epoch"),
        ckpt.get("best_metric", "?"),
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
        k: v for k, v in ckpt["model"].items() if "total_ops" not in k and "total_params" not in k
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))
    model._seq_len = 1
    model = model.cuda().eval()
    return model


# =========================================================================
# Score collection from ConvNeXt PSR head
# =========================================================================
def collect_convnext_psr_scores(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    max_batches: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Collect PSR logits and GT states grouped by recording.

    Args:
        model: POPWMultiTaskModel (eval mode, cuda).
        val_loader: DataLoader for val set.
        max_batches: cap on batches (0 = unlimited).

    Returns:
        Dict mapping recording_id -> (logits [T, 11], gt_states [T, 11])
    """
    all_psr_logits: List[np.ndarray] = []
    all_psr_labels: List[np.ndarray] = []
    all_rec_ids: List[str] = []
    all_frame_nums: List[int] = []

    logger.info("Collecting ConvNeXt PSR scores on val set")

    for bi, (images, targets) in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break

        images = images.cuda().float()
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

        # psr_logits: [B, 11]
        psr_logits_batch = pl.cpu().numpy().astype(np.float32)
        all_psr_logits.append(psr_logits_batch)

        # Ground truth PSR labels [B, 11]
        psr_labels_batch = targets["psr_labels"].cpu().numpy().astype(np.float32)
        all_psr_labels.append(psr_labels_batch)

        # Recording IDs and frame numbers
        for i in range(images.shape[0]):
            metadata_item = targets["metadata"][i] if i < len(targets.get("metadata", [])) else {}
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

        if bi % 50 == 0:
            logger.info("  Batch %d: %d images", bi, images.shape[0])

        del images, targets, outputs
        gc.collect()

    # - Group by recording -------------------------------------------------
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
            fn = all_frame_nums[flat_i] if flat_i < len(all_frame_nums) else flat_i
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
        order = np.argsort(np.asarray(by_rec_fn[rec], dtype=np.int64), kind="stable")
        logits = np.stack([rows[k] for k in order]).astype(np.float32)
        states = np.stack([gts[k] for k in order]).astype(np.float32)
        result[rec] = (logits, states)

    logger.info("Collected %d recordings (total frames: %d)", len(result), flat_i)
    return result


# =========================================================================
# PSR scoring helpers (from eval_yolov8m_psr.py)
# =========================================================================
def _event_f1(pred_tr: np.ndarray, gt_tr: np.ndarray, tol: int = 3) -> float:
    """Bi-directional greedy match of transition events within +/-tol frames."""
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


def _ordered_pair_fraction(pred_states: np.ndarray, gt_states: np.ndarray) -> float:
    pred_pairs = pred_states[1:] - pred_states[:-1]
    gt_pairs = gt_states[1:] - gt_states[:-1]
    return float((np.sign(pred_pairs) == np.sign(gt_pairs)).mean())


def _psr_edit_score(pred_states: np.ndarray, gt_states: np.ndarray) -> float:
    pred_events = "".join(
        str(int(b)) for b in (pred_states[1:] != pred_states[:-1]).any(axis=1).astype(int)
    )
    gt_events = "".join(
        str(int(b)) for b in (gt_states[1:] != gt_states[:-1]).any(axis=1).astype(int)
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


# =========================================================================
# MonotonicDecoder scoring (uses true MonotonicDecoder class)
# =========================================================================
def score_with_monotonic_decoder(
    collected_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    tol_frames: int = 3,
) -> Dict[str, Any]:
    """Score PSR using the MonotonicDecoder with its built-in thresholds."""
    decoder = MonotonicDecoder(num_components=N_COMPONENTS)

    f1s, poss, edits = [], [], []
    per_video: Dict[str, Dict[str, float]] = {}

    for rec, (logits_np, gt_np) in collected_scores.items():
        if len(logits_np) < 2:
            continue

        events = torch.sigmoid(torch.as_tensor(logits_np)).unsqueeze(0).float()
        pred_states = decoder(events).squeeze(0)

        pred_tr = (pred_states[1:] - pred_states[:-1]).clamp(min=0).cpu().numpy()
        gt_tr = np.clip(gt_np[1:] - gt_np[:-1], a_min=0, a_max=None)

        rec_f1 = _event_f1(pred_tr, gt_tr, tol=tol_frames)
        rec_pos = _ordered_pair_fraction(pred_states.cpu().numpy(), gt_np)
        rec_edit = _psr_edit_score(pred_states.cpu().numpy(), gt_np)

        f1s.append(rec_f1)
        poss.append(rec_pos)
        edits.append(rec_edit)
        per_video[rec] = {
            "f1_at_t": rec_f1,
            "pos": rec_pos,
            "edit": rec_edit,
            "n_frames": len(logits_np),
        }

    if not f1s:
        return {"psr_f1": 0.0, "psr_pos": 0.0, "psr_edit": 0.0, "per_video": {}}

    return {
        "psr_f1": float(np.mean(f1s)),
        "psr_pos": float(np.mean(poss)),
        "psr_edit": float(np.mean(edits)),
        "per_video": per_video,
    }


# =========================================================================
# Standalone threshold sweep (copy of d4_threshold_retune.py logic)
# =========================================================================
def compute_transition_f1_for_video(
    scores: np.ndarray,
    gt_states: np.ndarray,
    sustain_hi: float,
    sustain_lo: float,
    sustain_min: int,
    tolerance: int = 3,
) -> float:
    """Compute per-video transition F1 for a given threshold config (standalone)."""
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


def compute_per_component_f1(
    collected_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sustain_hi: float = DEFAULT_HI,
    sustain_lo: float = DEFAULT_LO,
    sustain_min: int = DEFAULT_MIN,
    tolerance: int = 3,
) -> Dict[str, Any]:
    """Compute per-component and macro F1 with given thresholds."""
    n_comp = N_COMPONENTS
    comp_f1s_sum = [0.0] * n_comp
    n_videos = 0
    all_f1s = []
    per_video = {}

    for video_id, (scores, gt_states) in collected_scores.items():
        n_videos += 1
        video_comp_f1s = []
        for c in range(n_comp):
            vf1 = compute_transition_f1_for_video(
                scores, gt_states, sustain_hi, sustain_lo, sustain_min, tolerance
            )
            video_comp_f1s.append(vf1)
            comp_f1s_sum[c] += vf1
        mean_f1 = float(np.mean(video_comp_f1s))
        all_f1s.append(mean_f1)
        per_video[video_id] = {"f1_at_t": mean_f1, "n_frames": scores.shape[0]}

    per_comp = {f"comp_{c}": round(s / max(n_videos, 1), 4) for c, s in enumerate(comp_f1s_sum)}

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
    """Sweep Q48 hysteresis thresholds and find best global config.

    Returns:
        dict with best config, comp_stats, and full sweep log.
    """
    total_combos = len(hi_values) * len(lo_values) * len(min_values)
    logger.info(
        "Sweeping %d threshold combinations over %d recordings",
        total_combos,
        len(collected_scores),
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
            c,
            st["mean"],
            st["std"],
            st["median"],
            st["p95"],
            st["prevalence"],
            st["n_frames"],
        )

    # Sweep all combos
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
        "Sweep complete (%d combos, %.0fs): best F1=%.4f at hi=%.2f lo=%.2f min=%d",
        len(sweep_log),
        elapsed,
        best_f1,
        best_config["sustain_hi"],
        best_config["sustain_lo"],
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


# =========================================================================
# Serialization helper
# =========================================================================
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


# =========================================================================
# Main
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="ConvNeXt -> decoder PSR evaluation (Opus 141 Q38)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="src/runs/rf_stages/checkpoints/best.pth",
        help="Path to multi-task ConvNeXt checkpoint (best.pth)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="src/runs/rf_stages/checkpoints/convnext_psr_decoder",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--fine-grid",
        action="store_true",
        help="Use finer grid around best region",
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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(args.ckpt):
        logger.error("Checkpoint not found: %s", args.ckpt)
        sys.exit(1)

    # -- Step 1: Load model ------------------------------------------------
    logger.info("Step 1: Loading ConvNeXt multi-task model")
    model = load_model(args.ckpt)

    # -- Step 2: Build val dataset and dataloader --------------------------
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

    # -- Step 3: Collect ConvNeXt PSR scores -------------------------------
    logger.info("Step 3: Collecting ConvNeXt PSR scores on val set")
    collected_scores = collect_convnext_psr_scores(
        model,
        val_loader,
        max_batches=args.max_batches,
    )

    if not collected_scores:
        logger.error("No scores collected! Aborting.")
        sys.exit(1)

    logger.info("Collected scores for %d recordings", len(collected_scores))

    # -- Step 4a: Score with default decoder -------------------------------
    logger.info("Step 4a: Scoring with default MonotonicDecoder thresholds")
    default_metrics = score_with_monotonic_decoder(collected_scores, tol_frames=3)
    logger.info(
        "Default decoder - F1: %.4f, POS: %.4f, Edit: %.4f",
        default_metrics.get("psr_f1", 0.0),
        default_metrics.get("psr_pos", 0.0),
        default_metrics.get("psr_edit", 0.0),
    )

    # Also score using the explicit compute function
    default_explicit = compute_per_component_f1(
        collected_scores,
        sustain_hi=DEFAULT_HI,
        sustain_lo=DEFAULT_LO,
        sustain_min=DEFAULT_MIN,
        tolerance=3,
    )
    logger.info("Default explicit - F1: %.4f", default_explicit["f1_at_t"])

    # -- Step 4b: Per-component breakdown at default thresholds ------------
    logger.info("Step 4b: Scoring with per-component breakdown at default thresholds")
    per_comp_default = compute_per_component_f1(
        collected_scores,
        sustain_hi=DEFAULT_HI,
        sustain_lo=DEFAULT_LO,
        sustain_min=DEFAULT_MIN,
        tolerance=3,
    )

    # -- Step 5: Sweep thresholds ------------------------------------------
    logger.info("Step 5: Sweeping Q48 hysteresis thresholds")
    hi_values = SUSTAIN_HI_FINE if args.fine_grid else SUSTAIN_HI_VALUES
    lo_values = SUSTAIN_LO_FINE if args.fine_grid else SUSTAIN_LO_VALUES
    min_values = SUSTAIN_MIN_FINE if args.fine_grid else SUSTAIN_MIN_VALUES

    sweep_results = sweep_thresholds(
        collected_scores,
        hi_values,
        lo_values,
        min_values,
    )

    sweep_path = save_dir / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(_serialize(sweep_results), f, indent=2, default=str)
    logger.info("Sweep results saved to %s", sweep_path)

    # -- Step 6: Run with best global threshold ----------------------------
    logger.info("Step 6: Running with best global sweep threshold")
    best_config = sweep_results["best"]
    best_global = compute_per_component_f1(
        collected_scores,
        sustain_hi=best_config["sustain_hi"],
        sustain_lo=best_config["sustain_lo"],
        sustain_min=best_config["sustain_min"],
        tolerance=3,
    )

    # -- Step 7: Run with per-component retuned thresholds -----------------
    logger.info("Step 7: Running with per-component retuned thresholds")
    comp_thresholds = sweep_results["per_component_thresholds"]
    retuned_f1s = []
    retuned_per_video = {}
    for video_id, (scores, gt_states) in collected_scores.items():
        comp_f1s = []
        for c in range(N_COMPONENTS):
            vf1 = compute_transition_f1_for_video(
                scores,
                gt_states,
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

    # -- Step 8: Save consolidated metrics ---------------------------------
    logger.info("Step 8: Saving consolidated metrics")
    f1_default = default_metrics["psr_f1"]
    f1_best_global = best_global["f1_at_t"]
    f1_retuned = retuned_f1

    metrics = {
        "default_decoder": {
            "f1_at_t": f1_default,
            "pos": default_metrics.get("psr_pos", 0.0),
            "edit": default_metrics.get("psr_edit", 0.0),
            "thresholds": {
                "sustain_hi": DEFAULT_HI,
                "sustain_lo": DEFAULT_LO,
                "sustain_min": DEFAULT_MIN,
            },
            "per_component_f1": per_comp_default["per_component_f1"],
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
        "config": {
            "ckpt": args.ckpt,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "n_recordings": len(collected_scores),
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

    # -- Step 9: Per-video breakdown ---------------------------------------
    per_video_all: Dict = {}
    for vid in collected_scores:
        pv = {}
        if vid in default_metrics.get("per_video", {}):
            pv["default"] = default_metrics["per_video"][vid]
        if vid in best_global.get("per_video", {}):
            pv["best_global"] = best_global["per_video"][vid]
        if vid in retuned_metrics.get("per_video", {}):
            pv["retuned"] = retuned_metrics["per_video"][vid]
        if pv:
            per_video_all[vid] = pv
    pv_path = save_dir / "per_video.json"
    with open(pv_path, "w") as f:
        json.dump(_serialize(per_video_all), f, indent=2, default=str)

    # -- Step 10: Generate 2x2 comparison table ----------------------------
    logger.info("Step 10: Generating 2x2 comparison table")

    # Reference values from Agent-6:
    d4_default_f1 = 0.000
    d4_d1r_default_f1 = 0.000
    d4_retuned_f1 = None
    d4_d1r_best_f1 = 0.347
    d4_d1r_retuned_f1 = 0.6364

    d4_retuned_path = Path("src/runs/rf_stages/checkpoints/d4_retuned/metrics.json")
    if d4_retuned_path.exists():
        try:
            with open(d4_retuned_path) as f:
                d4_retuned_data = json.load(f)
            d4_retuned_f1 = d4_retuned_data.get(
                "f1_at_t_retuned",
                d4_retuned_data.get("retuned_per_component", {}).get("f1_at_t"),
            )
        except Exception:
            pass

    d4_d1r_metrics_path = Path("src/runs/rf_stages/checkpoints/d4_d1r/metrics.json")
    if d4_d1r_metrics_path.exists():
        try:
            with open(d4_d1r_metrics_path) as f:
                dd = json.load(f)
            d4_d1r_default_f1 = dd.get("default_decoder", {}).get("f1_at_t", 0.000)
            d4_d1r_best_f1 = dd.get("best_global_sweep", {}).get("f1_at_t", d4_d1r_best_f1)
            d4_d1r_retuned_f1 = dd.get("retuned_per_component", {}).get(
                "f1_at_t", d4_d1r_retuned_f1
            )
        except Exception:
            pass

    table_lines = [
        "# 2x2 PSR Decoder Comparison Table\n",
        "\n",
        "## Full Table\n",
        "\n",
        "| Backbone | Decoder config | F1@t=3 | Notes |\n",
        "|---|---|---|---|\n",
    ]

    # Row 1: YOLOv8m + default
    table_lines.append(
        "| YOLOv8m (D4) | Default (hi=0.5, lo=0.3, min=3) | %.4f | Original D4 |\n" % d4_default_f1
    )

    # Row 2: YOLOv8m + retuned
    if d4_retuned_f1 is not None:
        table_lines.append(
            "| YOLOv8m (D4 retuned) | Per-component retuned "
            "| %.4f | Threshold recalibration |\n" % d4_retuned_f1
        )
    else:
        table_lines.append("| YOLOv8m (D4 retuned) | Per-component retuned | ? |\n")

    # Row 3: ConvNeXt + default
    table_lines.append(
        "| ConvNeXt-Tiny (this eval) | Default (hi=0.5, lo=0.3, min=3) "
        "| %.4f | Backbone effect |\n" % f1_default
    )

    # Row 4: ConvNeXt + best global
    table_lines.append(
        "| ConvNeXt-Tiny (this eval) | Best global "
        "(hi=%.2f, lo=%.2f, min=%d) "
        "| %.4f | Decoder strictness effect |\n"
        % (
            best_config["sustain_hi"],
            best_config["sustain_lo"],
            best_config["sustain_min"],
            f1_best_global,
        )
    )

    # Row 5: ConvNeXt + per-component
    table_lines.append(
        "| ConvNeXt-Tiny (this eval) | Per-component retuned | %.4f | Combined |\n" % f1_retuned
    )

    # Rows for D1R (reference)
    table_lines.append(
        "| D1R (oracle detection) | Default "
        "| %.4f | Oracle backbone (Agent-6) |\n" % d4_d1r_default_f1
    )
    table_lines.append(
        "| D1R (oracle detection) | Best global sweep "
        "| %.4f | Oracle backbone (Agent-6) |\n" % d4_d1r_best_f1
    )
    table_lines.append(
        "| D1R (oracle detection) | Per-component retuned "
        "| %.4f | Oracle backbone (Agent-6) |\n" % d4_d1r_retuned_f1
    )

    table_lines.append("\n")
    table_lines.append("## Compact 2x2 (Backbone x Decoder-strictness)\n")
    table_lines.append("\n")
    table_lines.append("| | Default decoder | Re-tuned decoder |\n")
    table_lines.append("|---|---|---|\n")
    d4_retuned_str = "%.4f" % d4_retuned_f1 if d4_retuned_f1 is not None else "?"
    table_lines.append("| YOLOv8m | %.4f | %s |\n" % (d4_default_f1, d4_retuned_str))
    table_lines.append("| ConvNeXt-Tiny | %.4f | %.4f |\n" % (f1_default, f1_retuned))
    table_lines.append(
        "| D1R (oracle det.) | %.4f | %.4f |\n" % (d4_d1r_default_f1, d4_d1r_retuned_f1)
    )

    table_lines.append("\n")
    table_lines.append("## Attribution Analysis\n")
    table_lines.append("\n")

    gap_yolo_default_to_convnext_default = f1_default - d4_default_f1
    gap_convnext_default_to_convnext_retuned = f1_retuned - f1_default
    gap_total = f1_retuned - d4_default_f1

    pct_backbone = 100 * gap_yolo_default_to_convnext_default / max(gap_total, 1e-9)
    pct_decoder = 100 * gap_convnext_default_to_convnext_retuned / max(gap_total, 1e-9)

    table_lines.append(
        "- Backbone effect (YOLOv8m to ConvNeXt, same default decoder): "
        "%+.4f\n" % gap_yolo_default_to_convnext_default
    )
    table_lines.append(
        "- Decoder-strictness effect (ConvNeXt default to retuned): "
        "%+.4f\n" % gap_convnext_default_to_convnext_retuned
    )
    table_lines.append(
        "- Total improvement (YOLOv8m default to ConvNeXt retuned): %+.4f\n" % gap_total
    )
    table_lines.append("- Backbone attribution: %.1f%%\n" % pct_backbone)
    table_lines.append("- Decoder-strictness attribution: %.1f%%\n" % pct_decoder)

    table_lines.append("\n")
    table_lines.append("## Per-Component F1 (Default Decoder)\n")
    table_lines.append("\n")
    table_lines.append("| Component | F1 |\n")
    table_lines.append("|---|---|\n")
    for c in range(N_COMPONENTS):
        cf1 = per_comp_default["per_component_f1"].get("comp_%d" % c, 0.0)
        table_lines.append("| %d | %.4f |\n" % (c, cf1))

    table_lines.append("\n")
    table_lines.append("## Top Sweep Results\n")
    table_lines.append("\n")
    table_lines.append("| hi | lo | min | F1@t=3 |\n")
    table_lines.append("|---|---|---|---|\n")
    for entry in sweep_results["sweep_log"][:15]:
        table_lines.append(
            "| %.2f | %.2f | %d | %.4f |\n"
            % (
                entry["sustain_hi"],
                entry["sustain_lo"],
                entry["sustain_min"],
                entry["f1_at_t"],
            )
        )

    table_path = save_dir / "2x2_table.md"
    with open(table_path, "w") as f:
        f.writelines(table_lines)
    logger.info("2x2 table saved to %s", table_path)

    # -- Print summary -----------------------------------------------------
    print("\n%s" % ("=" * 70))
    print("  ConvNeXt -> decoder PSR Evaluation - Complete")
    print("%s" % ("=" * 70))
    print("  Recordings processed:  %d" % len(collected_scores))
    print("  Checkpoint:            %s" % args.ckpt)
    print()
    print(
        "  Default decoder (hi=%.1f, lo=%.1f, min=%d):"
        % (
            DEFAULT_HI,
            DEFAULT_LO,
            DEFAULT_MIN,
        )
    )
    print("    F1@t=3:  %.4f" % f1_default)
    print("    POS:     %.4f" % default_metrics.get("psr_pos", 0.0))
    print("    Edit:    %.4f" % default_metrics.get("psr_edit", 0.0))
    print()
    print("  Best global sweep:")
    print(
        "    hi=%.2f, lo=%.2f, min=%d"
        % (
            best_config["sustain_hi"],
            best_config["sustain_lo"],
            best_config["sustain_min"],
        )
    )
    print("    F1@t=3:  %.4f" % f1_best_global)
    print()
    print("  Retuned per-component:")
    print("    F1@t=3:  %.4f" % f1_retuned)
    print()
    print(
        "  Sweep: %d combos in %.0fs"
        % (
            sweep_results["n_combos_tested"],
            sweep_results["elapsed_seconds"],
        )
    )
    print()
    print("  Per-component thresholds:")
    for c in range(N_COMPONENTS):
        print(
            "    Comp %2d: hi=%.3f lo=%.3f min=%d"
            % (
                c,
                comp_thresholds["sustain_hi"][c],
                comp_thresholds["sustain_lo"][c],
                comp_thresholds["sustain_min"][c],
            )
        )
    print()
    print("  Per-component F1 (default decoder):")
    for c in range(N_COMPONENTS):
        cf1 = per_comp_default["per_component_f1"].get("comp_%d" % c, 0.0)
        print("    Comp %2d: %.4f" % (c, cf1))
    print()

    print("  Top-10 sweep results:")
    for i, entry in enumerate(sweep_results["sweep_log"][:10]):
        print(
            "    %2d. hi=%.2f lo=%.2f min=%d  F1=%.4f"
            % (
                i + 1,
                entry["sustain_hi"],
                entry["sustain_lo"],
                entry["sustain_min"],
                entry["f1_at_t"],
            )
        )
    print()
    print("%s" % ("=" * 70))


if __name__ == "__main__":
    main()
