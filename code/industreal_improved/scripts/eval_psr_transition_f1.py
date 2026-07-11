#!/usr/bin/env python3
"""eval_psr_transition_f1.py — end-to-end PSR transition event-F1 evaluation.

Loads a trained model checkpoint, runs inference on the validation set,
and computes all three PSR headline metrics at once:
    - event_f1@±3   transition-event F1 with greedy matching within tolerance
    - POS            ordered-pair fraction (directional sign agreement)
    - tau (delay)    average frame offset between matched pred and GT events

Saves per-recording and aggregate results to JSON.

Usage:
    python scripts/eval_psr_transition_f1.py \\
        --checkpoint code/industreal_improved/src/runs/rf_stages/checkpoints/best.pth \\
        --tolerance 3 --max-batches 5000

Reference: 175 §7.2 (PSR metric), 174 §3.3 (transition-F1 protocol).
"""

import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import median_filter

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "code" / "industreal_improved" / "src"
for _p in [_SRC, _SRC.parent, _SRC / "models", _SRC / "data"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.evaluation.decoder_oracle_bound import event_f1

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_tau(pred_tr: np.ndarray, gt_tr: np.ndarray, tol: int = 3) -> float:
    """Compute mean delay tau (frame offset) for matched transition events.

    For each component, iterates predicted frames (0-to-1 events) and
    matches them greedily to the nearest GT frame within tolerance.
    tau = mean(pred_frame - gt_frame) across all matches.

    Positive tau = lag (prediction after GT event).
    Negative tau = anticipation (prediction before GT event).

    Returns:
        Mean delay in frames. NaN if no matches.
    """
    n_comp = pred_tr.shape[1]
    delays = []
    for c in range(n_comp):
        p_frames = np.where(pred_tr[:, c])[0]
        g_frames = np.where(gt_tr[:, c])[0]
        matched_gt = set()
        for pf in p_frames:
            best_delay = None
            best_gi = None
            for gi, gf in enumerate(g_frames):
                if gi not in matched_gt and abs(pf - gf) <= tol:
                    d = int(pf) - int(gf)
                    if best_delay is None or abs(d) < abs(best_delay):
                        best_delay = d
                        best_gi = gi
            if best_gi is not None:
                matched_gt.add(best_gi)
                delays.append(best_delay)
    if not delays:
        return float("nan")
    return float(np.mean(delays))


def compute_tau_abs(pred_tr: np.ndarray, gt_tr: np.ndarray, tol: int = 3) -> float:
    """Mean absolute delay for matched events (STORM-style, always non-negative)."""
    n_comp = pred_tr.shape[1]
    delays = []
    for c in range(n_comp):
        p_frames = np.where(pred_tr[:, c])[0]
        g_frames = np.where(gt_tr[:, c])[0]
        matched_gt = set()
        for pf in p_frames:
            best_delay = None
            best_gi = None
            for gi, gf in enumerate(g_frames):
                if gi not in matched_gt and abs(pf - gf) <= tol:
                    d = abs(int(pf) - int(gf))
                    if best_delay is None or d < best_delay:
                        best_delay = d
                        best_gi = gi
            if best_gi is not None:
                matched_gt.add(best_gi)
                delays.append(best_delay)
    if not delays:
        return float("nan")
    return float(np.mean(delays))


def compute_pos(pred_tr: np.ndarray, gt_tr: np.ndarray) -> float:
    """Ordered-pair fraction: fraction of frame-pairs where sign agrees.

    POS = mean( sign(pred_diff) == sign(gt_diff) ).

    NOTE: A null model (all-zeros) scores ~0.9995 because most frames have
    no transition in either direction (0 == 0). POS is reliable ONLY when
    the dataset has dense transitions. Always report alongside null-model
    baseline. See 174 §3.3 POS caveat.
    """
    pred_sign = np.sign(pred_tr)
    gt_sign = np.sign(gt_tr)
    return float((pred_sign == gt_sign).mean())


def per_frame_f1(pred: np.ndarray, label: np.ndarray) -> float:
    """Per-component macro F1 at the frame level (secondary metric)."""
    f1s = []
    for c in range(pred.shape[1]):
        tp = ((pred[:, c] == 1) & (label[:, c] == 1)).sum()
        fp = ((pred[:, c] == 1) & (label[:, c] == 0)).sum()
        fn = ((pred[:, c] == 0) & (label[:, c] == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def frames_to_seconds(n_frames: int, fps: float = 30.0) -> float:
    """Convert frame-count to real seconds."""
    return n_frames / fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PSR transition event-F1 evaluation (175 §7.2)"
    )
    parser.add_argument(
        "--checkpoint",
        default="code/industreal_improved/src/runs/rf_stages/checkpoints/best.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=5000,
        help="Max batches to evaluate (default: 5000, ~full val set at batch=1)",
    )
    parser.add_argument(
        "--tolerance", type=int, default=3,
        help="Frame tolerance for event matching (default: 3, matches B3/STORM)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Global sigmoid threshold for PSR binary prediction (default: 0.10, matches PSR headline protocol)",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Path to optimal_thresholds.json (per-component thresholds, overrides --threshold)",
    )
    parser.add_argument(
        "--save-dir",
        default="code/industreal_improved/src/runs/rf_stages/checkpoints/psr_event_f1_run",
        help="Output directory for metrics.json",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ─────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", "unknown")
    print(f"  Epoch: {epoch}")

    # Per-component thresholds
    per_comp_thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thr_data = json.load(f)
        per_comp_thresholds = np.array(thr_data["optimal_thresholds"], dtype=np.float32)
        print(f"  Per-component thresholds: {per_comp_thresholds.tolist()}")
    else:
        print(f"  Global threshold: {args.threshold}")

    # ── Build model ─────────────────────────────────────────────────────
    from src.models.model import POPWMultiTaskModel
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    model = POPWMultiTaskModel(
        pretrained=True, backbone_type="convnext_tiny",
        use_hand_film=True, use_headpose_film=True,
        use_videomae=False, train_pose=False,
    )
    state_dict = {
        k: v for k, v in ckpt["model"].items()
        if "total_ops" not in k and "total_params" not in k
    }
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()

    # ── Data ────────────────────────────────────────────────────────────
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False,
    )

    # Per-recording accumulators
    rec_preds: dict[str, list] = defaultdict(list)
    rec_labels: dict[str, list] = defaultdict(list)
    rec_frame_nums: dict[str, list] = defaultdict(list)

    n_frames = 0
    for i, batch in enumerate(val_loader):
        if i >= args.max_batches:
            break
        images, targets = batch
        if images.shape[0] == 0:
            continue

        images_f = images.cuda().float()
        if images_f.max() > 1.0:
            images_f = images_f.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images_f.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images_f.device).view(1, 3, 1, 1)
        images_n = (images_f - mean) / std

        with torch.no_grad():
            outputs = model(images_n)

        pl = outputs.get("psr_logits")
        pl_lbl = targets.get("psr_labels")
        if pl is None or pl_lbl is None:
            continue

        # Sigmoid probabilities (threshold+monotonicity per-recording)
        sig = torch.sigmoid(pl[0]).cpu().numpy()
        lbl = pl_lbl[0].cpu().numpy()

        meta = targets.get("metadata", [])
        rec_id = meta[0].get("recording_id", "unknown")
        frame_num = meta[0].get("frame_num", 0)

        rec_preds[rec_id].append(sig)
        rec_labels[rec_id].append(lbl)
        rec_frame_nums[rec_id].append(frame_num)
        n_frames += 1
        if n_frames % 2000 == 0:
            print(f"  processed {n_frames} frames...")

    print(f"\nCollected {n_frames} frames across {len(rec_preds)} recordings")

    # ── Compute metrics per recording ───────────────────────────────────
    per_rec = {}
    event_f1s = []
    taus = []
    taus_abs = []
    poss = []

    for rec, preds in rec_preds.items():
        gt = rec_labels[rec]
        frames = np.array(rec_frame_nums[rec])
        sort_idx = np.argsort(frames)
        preds_sorted = np.array(preds)[sort_idx]   # [T, 11] sigmoid probabilities
        gt_sorted = np.array(gt)[sort_idx]

        # Keep only frames with valid GT labels
        valid_mask = gt_sorted.max(axis=1) >= 0
        n_valid = int(valid_mask.sum())
        if n_valid < 2:
            continue
        vp_prob = preds_sorted[valid_mask]
        vl = gt_sorted[valid_mask]

        # Monotonicity constraint: smooth then running-max (once-on stays-on)
        _vp_smooth = median_filter(vp_prob, size=(5, 1), mode="nearest")
        _vp_mono = np.maximum.accumulate(_vp_smooth, axis=0)
        if per_comp_thresholds is not None:
            vp = (_vp_mono > per_comp_thresholds[np.newaxis, :]).astype(np.int32)
        else:
            vp = (_vp_mono > args.threshold).astype(np.int32)

        # Per-frame (secondary)
        pf1 = per_frame_f1(vp, vl)

        # Transition events (0-to-1)
        pred_tr = np.clip(vp[1:] - vp[:-1], a_min=0, a_max=None)
        gt_tr = np.clip(vl[1:] - vl[:-1], a_min=0, a_max=None)

        # Filter to valid transition pairs
        valid_tr = vl[1:].max(axis=1) >= 0
        pred_tr_v = pred_tr[valid_tr]
        gt_tr_v = gt_tr[valid_tr]

        # Transition counts
        n_pred_trans = int(pred_tr_v.sum())
        n_gt_trans = int(gt_tr_v.sum())

        # event_f1@±tolerance
        ef1 = event_f1(pred_tr_v, gt_tr_v, tol=args.tolerance)

        # tau (delay)
        tau = compute_tau(pred_tr_v, gt_tr_v, tol=args.tolerance)
        tau_abs = compute_tau_abs(pred_tr_v, gt_tr_v, tol=args.tolerance)

        # POS (ordered-pair)
        pos = compute_pos(pred_tr_v, gt_tr_v)

        per_rec[rec] = {
            "n_frames": n_valid,
            "n_gt_transitions": n_gt_trans,
            "n_pred_transitions": n_pred_trans,
            "per_frame_f1": pf1,
            "event_f1": ef1,
            "tau_frames": tau,
            "tau_abs_frames": tau_abs,
            "pos": pos,
        }
        event_f1s.append(ef1)
        if not np.isnan(tau):
            taus.append(tau)
            taus_abs.append(tau_abs)
        poss.append(pos)

    # ── Aggregate ───────────────────────────────────────────────────────
    FPS = 30.0  # IndustReal recording framerate
    tau_mean_frames = float(np.nanmean(taus)) if taus else float("nan")
    tau_abs_mean_frames = float(np.nanmean(taus_abs)) if taus_abs else float("nan")

    summary = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "n_frames": n_frames,
        "n_recordings": len(per_rec),
        "tolerance": args.tolerance,
        "threshold": (
            "per_component_optimal"
            if per_comp_thresholds is not None
            else args.threshold
        ),
        "event_f1_macro": float(np.mean(event_f1s)) if event_f1s else 0.0,
        "tau_mean_frames": tau_mean_frames,
        "tau_abs_mean_frames": tau_abs_mean_frames,
        "tau_mean_seconds": tau_mean_frames / FPS if not np.isnan(tau_mean_frames) else float("nan"),
        "tau_abs_mean_seconds": tau_abs_mean_frames / FPS if not np.isnan(tau_abs_mean_frames) else float("nan"),
        "pos_macro": float(np.mean(poss)) if poss else 0.0,
        "per_recording": per_rec,
    }

    # ── Print headline ──────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("PSR TRANSITION EVENT-F1 (175 §7.2 / 174 §3.3)")
    print("=" * 64)
    print(f"  Checkpoint:          {args.checkpoint}")
    print(f"  Epoch:               {epoch}")
    print(f"  Tolerance:           +/-{args.tolerance} frames")
    print(f"  Threshold:           {summary['threshold']}")
    print(f"  Frames evaluated:    {n_frames}")
    print(f"  Recordings:          {len(per_rec)}")
    print()
    print(f"  event_f1@+/-{args.tolerance}:  {summary['event_f1_macro']:.4f}  (macro avg)")
    print(f"  tau (mean delay):     {tau_mean_frames:.2f} frames  ({summary['tau_mean_seconds']:.1f}s)")
    print(f"  tau_abs (abs delay):  {tau_abs_mean_frames:.2f} frames  ({summary['tau_abs_mean_seconds']:.1f}s)")
    print(f"  POS (macro):          {summary['pos_macro']:.4f}")
    print()
    print("  Reference (STORM CVIU 2025, test split):")
    print("    event_f1 = 0.901, POS = 0.812, tau = 15.5s")
    print("  Reference (B3/WACV 2024, test split):")
    print("    event_f1 = 0.883, POS = 0.797, tau = 22.4s")
    print("=" * 64)

    # ── Save ────────────────────────────────────────────────────────────
    out_path = save_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
