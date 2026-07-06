"""Head pose Kalman smoothing eval — single-frame vs smoothed MAE.

Runs model inference over the full val dataset, computes single-frame
angular MAE for forward and up-vector, then applies per-recording
Kalman smoothing (RTS smoother) and recomputes MAE.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 src/evaluation/eval_pose_kalman.py

Caches per-recording predictions to {save_dir}/cache/ as .npz files for
subsequent parameter sweeps (skip_inference=True).
"""
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def kalman_smooth(seq: np.ndarray, process_noise: float = 0.01,
                  measurement_noise: float = 0.05) -> np.ndarray:
    """Rauch-Tung-Striebel (RTS) Kalman smoother per channel.

    Proper implementation: saves filtered and predicted states/covariances
    from the forward pass, then uses them in the backward pass.

    Args:
        seq: [N, D] sequence of D-dimensional unit vectors
        process_noise: Q (per-step variance)
        measurement_noise: R (observation variance)

    Returns:
        smoothed: [N, D] smoothed unit vectors
    """
    N, D = seq.shape
    if N < 3:
        return seq.copy()
    smoothed = np.zeros_like(seq)
    for c in range(D):
        z = seq[:, c]
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.array([[process_noise, 0.0], [0.0, process_noise * 0.1]])
        R = np.array([[measurement_noise]])

        # Forward pass — save filtered and predicted states/covariances
        x = np.array([z[0], 0.0])
        P = np.eye(2) * 1.0
        x_filt = np.zeros((N, 2))
        P_filt = np.zeros((N, 2, 2))
        x_pred = np.zeros((N, 2))
        P_pred = np.zeros((N, 2, 2))

        for t in range(N):
            # Predict
            x_p = F @ x
            P_p = F @ P @ F.T + Q
            x_pred[t] = x_p
            P_pred[t] = P_p
            # Update
            y = z[t] - (H @ x_p)[0]
            S = (H @ P_p @ H.T)[0, 0] + R[0, 0]
            K = (P_p @ H.T) / S
            x = x_p + (K.flatten() * y)
            P = (np.eye(2) - K @ H) @ P_p
            x_filt[t] = x
            P_filt[t] = P

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((N, 2))
        x_smooth[-1] = x_filt[-1]
        for t in range(N - 2, -1, -1):
            C = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
            x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])

        smoothed[:, c] = x_smooth[:, 0]

    # Re-normalize to unit length
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    smoothed = smoothed / np.maximum(norms, 1e-6)
    return smoothed


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean angular error in degrees between predicted and GT unit vectors."""
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-6)
    gt_n = gt / np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-6)
    cos = np.sum(pred_n * gt_n, axis=1).clip(-1.0, 1.0)
    return float(np.degrees(np.arccos(cos).mean()))


def run_inference(args):
    """Run model inference and return recording_preds dict."""
    from src.models.model import POPWMultiTaskModel
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  global_step: {ckpt.get('global_step', 'unknown')}")

    print("Building model...")
    model = POPWMultiTaskModel(
        pretrained=True, backbone_type='convnext_tiny',
        use_hand_film=True, use_headpose_film=True,
        use_videomae=False, train_pose=True,
    )
    state_dict = {k: v for k, v in ckpt["model"].items()
                  if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()
    print("Model loaded and in eval mode.")

    print("Loading val dataset...")
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False,
    )
    print(f"Val dataset: {len(val_ds)} frames")

    recording_preds = defaultdict(list)
    n = 0
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

        hp = outputs.get("head_pose")
        gt_hp = targets.get("head_pose")
        if hp is None or gt_hp is None:
            continue

        # head_pose is [B, 9] = forward[0:3] + position[3:6] + up[6:9]
        pred_fwd = hp[0, :3].cpu().numpy()
        pred_up = hp[0, 6:9].cpu().numpy()
        gt_fwd = gt_hp[0, :3].cpu().numpy()
        gt_up = gt_hp[0, 6:9].cpu().numpy()

        meta = targets.get("metadata", [])
        rec_id = meta[0].get("recording_id", "unknown")

        recording_preds[rec_id].append({
            "pred_fwd": pred_fwd,
            "pred_up": pred_up,
            "gt_fwd": gt_fwd,
            "gt_up": gt_up,
        })
        n += 1
        if n % 2000 == 0:
            print(f"  processed {n} frames across {len(recording_preds)} recordings...")

    print(f"\nProcessed {n} total frames across {len(recording_preds)} recordings.")
    return recording_preds, n


def load_cached_predictions(cache_dir: Path):
    """Load per-recording predictions from npz cache."""
    recording_preds = defaultdict(list)
    for fpath in sorted(cache_dir.glob("*.npz")):
        data = np.load(fpath)
        rec_id = data["recording_id"].item() if data["recording_id"].ndim == 0 else str(data["recording_id"])
        n_frames = data["pred_fwd"].shape[0]
        for t in range(n_frames):
            recording_preds[rec_id].append({
                "pred_fwd": data["pred_fwd"][t],
                "pred_up": data["pred_up"][t],
                "gt_fwd": data["gt_fwd"][t],
                "gt_up": data["gt_up"][t],
            })
    total = sum(len(v) for v in recording_preds.values())
    return recording_preds, total


def cache_predictions(recording_preds, cache_dir: Path):
    """Save per-recording predictions to npz files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for rec_id, frames in recording_preds.items():
        n = len(frames)
        pred_fwd = np.array([f["pred_fwd"] for f in frames])
        pred_up = np.array([f["pred_up"] for f in frames])
        gt_fwd = np.array([f["gt_fwd"] for f in frames])
        gt_up = np.array([f["gt_up"] for f in frames])
        safe_id = rec_id.replace("/", "_").replace(" ", "_")
        np.savez_compressed(
            cache_dir / f"{safe_id}.npz",
            recording_id=rec_id,
            pred_fwd=pred_fwd, pred_up=pred_up,
            gt_fwd=gt_fwd, gt_up=gt_up,
        )
    print(f"Cached {len(recording_preds)} recordings to {cache_dir}")


def eval_smoothing(recording_preds, n_total, args):
    """Evaluate single-frame and Kalman-smoothed MAE."""
    per_recording_results = {}
    all_single_fwd_errors = []
    all_single_up_errors = []
    all_smoothed_fwd_errors = []
    all_smoothed_up_errors = []

    for rec_id in sorted(recording_preds.keys()):
        frames = recording_preds[rec_id]
        N = len(frames)
        if N < 3:
            print(f"  {rec_id}: only {N} frames, skipping smoothing")
            continue

        pred_fwd = np.array([f["pred_fwd"] for f in frames])
        pred_up = np.array([f["pred_up"] for f in frames])
        gt_fwd = np.array([f["gt_fwd"] for f in frames])
        gt_up = np.array([f["gt_up"] for f in frames])

        sf_fwd = angular_error_deg(pred_fwd, gt_fwd)
        sf_up = angular_error_deg(pred_up, gt_up)

        smooth_fwd = kalman_smooth(pred_fwd, args.process_noise, args.measurement_noise)
        smooth_up = kalman_smooth(pred_up, args.process_noise, args.measurement_noise)
        sm_fwd = angular_error_deg(smooth_fwd, gt_fwd)
        sm_up = angular_error_deg(smooth_up, gt_up)

        all_single_fwd_errors.append(sf_fwd)
        all_single_up_errors.append(sf_up)
        all_smoothed_fwd_errors.append(sm_fwd)
        all_smoothed_up_errors.append(sm_up)

        per_recording_results[rec_id] = {
            "n": N,
            "single_frame_fwd_MAE_deg": float(sf_fwd),
            "single_frame_up_MAE_deg": float(sf_up),
            "smoothed_fwd_MAE_deg": float(sm_fwd),
            "smoothed_up_MAE_deg": float(sm_up),
            "fwd_improvement_deg": float(sf_fwd - sm_fwd),
            "up_improvement_deg": float(sf_up - sm_up),
        }

    sf_fwd_arr = np.array(all_single_fwd_errors)
    sf_up_arr = np.array(all_single_up_errors)
    sm_fwd_arr = np.array(all_smoothed_fwd_errors)
    sm_up_arr = np.array(all_smoothed_up_errors)

    total_frames_all = sum(r["n"] for r in per_recording_results.values()) or 1
    weighted_sf_fwd = sum(r["single_frame_fwd_MAE_deg"] * r["n"]
                          for r in per_recording_results.values()) / total_frames_all
    weighted_sf_up = sum(r["single_frame_up_MAE_deg"] * r["n"]
                         for r in per_recording_results.values()) / total_frames_all
    weighted_sm_fwd = sum(r["smoothed_fwd_MAE_deg"] * r["n"]
                          for r in per_recording_results.values()) / total_frames_all
    weighted_sm_up = sum(r["smoothed_up_MAE_deg"] * r["n"]
                         for r in per_recording_results.values()) / total_frames_all

    summary = {
        "checkpoint": args.checkpoint if hasattr(args, 'checkpoint') else "cached",
        "n_frames_total": n_total,
        "n_recordings": len(per_recording_results),
        "kalman_params": {
            "process_noise": args.process_noise,
            "measurement_noise": args.measurement_noise,
        },
        "per_recording": per_recording_results,
        "single_frame": {
            "forward_MAE_deg": float(np.mean(sf_fwd_arr)),
            "forward_MAE_deg_weighted": float(weighted_sf_fwd),
            "forward_MAE_deg_median": float(np.median(sf_fwd_arr)),
            "forward_MAE_deg_std": float(np.std(sf_fwd_arr)),
            "up_MAE_deg": float(np.mean(sf_up_arr)),
            "up_MAE_deg_weighted": float(weighted_sf_up),
            "up_MAE_deg_median": float(np.median(sf_up_arr)),
            "up_MAE_deg_std": float(np.std(sf_up_arr)),
        },
        "smoothed": {
            "forward_MAE_deg": float(np.mean(sm_fwd_arr)),
            "forward_MAE_deg_weighted": float(weighted_sm_fwd),
            "forward_MAE_deg_median": float(np.median(sm_fwd_arr)),
            "forward_MAE_deg_std": float(np.std(sm_fwd_arr)),
            "up_MAE_deg": float(np.mean(sm_up_arr)),
            "up_MAE_deg_weighted": float(weighted_sm_up),
            "up_MAE_deg_median": float(np.median(sm_up_arr)),
            "up_MAE_deg_std": float(np.std(sm_up_arr)),
        },
        "improvement": {
            "forward_deg": float(weighted_sf_fwd - weighted_sm_fwd),
            "forward_pct": float((weighted_sf_fwd - weighted_sm_fwd) / weighted_sf_fwd * 100)
                          if weighted_sf_fwd > 0 else 0.0,
            "up_deg": float(weighted_sf_up - weighted_sm_up),
            "up_pct": float((weighted_sf_up - weighted_sm_up) / weighted_sf_up * 100)
                      if weighted_sf_up > 0 else 0.0,
        },
    }
    return summary, per_recording_results


def print_summary(summary, n_total, per_recording_results,
                  process_noise, measurement_noise):
    """Print results summary."""
    sf = summary["single_frame"]
    sm = summary["smoothed"]
    imp = summary["improvement"]

    print("\n" + "=" * 60)
    print("HEAD POSE KALMAN SMOOTHING RESULTS")
    print("=" * 60)
    print(f"\nSingle-frame (deployment-honest) — per-recording mean:")
    print(f"  Forward MAE:  {sf['forward_MAE_deg']:.2f} deg  "
          f"(weighted: {sf['forward_MAE_deg_weighted']:.2f} deg)")
    print(f"  Up-vector MAE: {sf['up_MAE_deg']:.2f} deg  "
          f"(weighted: {sf['up_MAE_deg_weighted']:.2f} deg)")
    print(f"  Forward median: {sf['forward_MAE_deg_median']:.2f} deg")
    print(f"  Up-vector median: {sf['up_MAE_deg_median']:.2f} deg")

    print(f"\nKalman-smoothed — per-recording mean:")
    print(f"  Forward MAE:  {sm['forward_MAE_deg']:.2f} deg  "
          f"(weighted: {sm['forward_MAE_deg_weighted']:.2f} deg)")
    print(f"  Up-vector MAE: {sm['up_MAE_deg']:.2f} deg  "
          f"(weighted: {sm['up_MAE_deg_weighted']:.2f} deg)")
    print(f"  Forward median: {sm['forward_MAE_deg_median']:.2f} deg")
    print(f"  Up-vector median: {sm['up_MAE_deg_median']:.2f} deg")

    print(f"\nImprovement from smoothing:")
    print(f"  Forward:  {imp['forward_deg']:+.2f} deg ({imp['forward_pct']:+.1f}%)")
    print(f"  Up-vector: {imp['up_deg']:+.2f} deg ({imp['up_pct']:+.1f}%)")

    print(f"\nProcessed {n_total} frames across {len(per_recording_results)} recordings")
    print(f"Kalman params: Q={process_noise}, R={measurement_noise}")

    print(f"\n{'='*60}")
    print(f"Per-recording breakdown:")
    print(f"{'Recording':>20s}  {'N':>5s}  {'SF-fwd':>8s}  {'SM-fwd':>8s}  "
          f"{'SF-up':>8s}  {'SM-up':>8s}")
    print(f"{'-'*60}")
    for rid in sorted(per_recording_results.keys()):
        r = per_recording_results[rid]
        print(f"{rid:>20s}  {r['n']:5d}  {r['single_frame_fwd_MAE_deg']:8.2f}  "
              f"{r['smoothed_fwd_MAE_deg']:8.2f}  {r['single_frame_up_MAE_deg']:8.2f}  "
              f"{r['smoothed_up_MAE_deg']:8.2f}")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="src/runs/rf_stages/checkpoints/best.pth")
    parser.add_argument("--max-batches", type=int, default=200000)
    parser.add_argument("--save-dir",
                        default="src/runs/rf_stages/checkpoints/pose_kalman_eval")
    parser.add_argument("--process-noise", type=float, default=0.01)
    parser.add_argument("--measurement-noise", type=float, default=0.05)
    parser.add_argument("--cache-only", action="store_true",
                        help="Only cache predictions, skip eval")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Load cached predictions instead of running inference")
    parser.add_argument("--cache-dir",
                        default="")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else save_dir / "cache"

    print("=" * 60)
    print("Head Pose Kalman Smoothing Eval")
    print("=" * 60)
    print(f"Save dir: {save_dir}")
    print(f"Cache dir: {cache_dir}")
    print(f"Process noise (Q): {args.process_noise}")
    print(f"Measurement noise (R): {args.measurement_noise}")
    print()

    if args.skip_inference:
        if not cache_dir.exists():
            print(f"ERROR: cache dir {cache_dir} not found. Run without --skip-inference first.")
            sys.exit(1)
        print("Loading cached predictions...")
        recording_preds, n_total = load_cached_predictions(cache_dir)
        print(f"Loaded {n_total} frames across {len(recording_preds)} recordings.")
    else:
        recording_preds, n_total = run_inference(args)
        cache_predictions(recording_preds, cache_dir)

    if args.cache_only:
        print("Cache-only mode. Exiting.")
        return

    summary, per_recording_results = eval_smoothing(recording_preds, n_total, args)

    out_path = save_dir / "pose_kalman_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_summary(summary, n_total, per_recording_results,
                  args.process_noise, args.measurement_noise)

    # Per-recording eval
    per_recording_results = {}
    all_single_fwd_errors = []
    all_single_up_errors = []
    all_smoothed_fwd_errors = []
    all_smoothed_up_errors = []

    for rec_id in sorted(recording_preds.keys()):
        frames = recording_preds[rec_id]
        N = len(frames)
        if N < 3:
            print(f"  {rec_id}: only {N} frames, skipping smoothing")
            continue

        pred_fwd = np.array([f["pred_fwd"] for f in frames])
        pred_up = np.array([f["pred_up"] for f in frames])
        gt_fwd = np.array([f["gt_fwd"] for f in frames])
        gt_up = np.array([f["gt_up"] for f in frames])

        # Single-frame errors
        sf_fwd = angular_error_deg(pred_fwd, gt_fwd)
        sf_up = angular_error_deg(pred_up, gt_up)

        # Smoothed errors
        smooth_fwd = kalman_smooth(pred_fwd, args.process_noise, args.measurement_noise)
        smooth_up = kalman_smooth(pred_up, args.process_noise, args.measurement_noise)
        sm_fwd = angular_error_deg(smooth_fwd, gt_fwd)
        sm_up = angular_error_deg(smooth_up, gt_up)

        all_single_fwd_errors.append(sf_fwd)
        all_single_up_errors.append(sf_up)
        all_smoothed_fwd_errors.append(sm_fwd)
        all_smoothed_up_errors.append(sm_up)

        per_recording_results[rec_id] = {
            "n": N,
            "single_frame_fwd_MAE_deg": float(sf_fwd),
            "single_frame_up_MAE_deg": float(sf_up),
            "smoothed_fwd_MAE_deg": float(sm_fwd),
            "smoothed_up_MAE_deg": float(sm_up),
            "fwd_improvement_deg": float(sf_fwd - sm_fwd),
            "up_improvement_deg": float(sf_up - sm_up),
        }

    sf_fwd_arr = np.array(all_single_fwd_errors)
    sf_up_arr = np.array(all_single_up_errors)
    sm_fwd_arr = np.array(all_smoothed_fwd_errors)
    sm_up_arr = np.array(all_smoothed_up_errors)

    # Aggregate: weighted by frame count per recording
    total_frames_all = sum(r["n"] for r in per_recording_results.values()) or 1
    weighted_sf_fwd = sum(r["single_frame_fwd_MAE_deg"] * r["n"]
                          for r in per_recording_results.values()) / total_frames_all
    weighted_sf_up = sum(r["single_frame_up_MAE_deg"] * r["n"]
                         for r in per_recording_results.values()) / total_frames_all
    weighted_sm_fwd = sum(r["smoothed_fwd_MAE_deg"] * r["n"]
                          for r in per_recording_results.values()) / total_frames_all
    weighted_sm_up = sum(r["smoothed_up_MAE_deg"] * r["n"]
                         for r in per_recording_results.values()) / total_frames_all

    summary = {
        "checkpoint": args.checkpoint,
        "n_frames_total": sum(rec["n"] for rec in per_recording_results.values()),
        "n_recordings": len(per_recording_results),
        "kalman_params": {
            "process_noise": args.process_noise,
            "measurement_noise": args.measurement_noise,
        },
        "per_recording": per_recording_results,
        "single_frame": {
            "forward_MAE_deg": float(np.mean(sf_fwd_arr)),
            "forward_MAE_deg_weighted": float(weighted_sf_fwd),
            "forward_MAE_deg_median": float(np.median(sf_fwd_arr)),
            "forward_MAE_deg_std": float(np.std(sf_fwd_arr)),
            "up_MAE_deg": float(np.mean(sf_up_arr)),
            "up_MAE_deg_weighted": float(weighted_sf_up),
            "up_MAE_deg_median": float(np.median(sf_up_arr)),
            "up_MAE_deg_std": float(np.std(sf_up_arr)),
        },
        "smoothed": {
            "forward_MAE_deg": float(np.mean(sm_fwd_arr)),
            "forward_MAE_deg_weighted": float(weighted_sm_fwd),
            "forward_MAE_deg_median": float(np.median(sm_fwd_arr)),
            "forward_MAE_deg_std": float(np.std(sm_fwd_arr)),
            "up_MAE_deg": float(np.mean(sm_up_arr)),
            "up_MAE_deg_weighted": float(weighted_sm_up),
            "up_MAE_deg_median": float(np.median(sm_up_arr)),
            "up_MAE_deg_std": float(np.std(sm_up_arr)),
        },
        "improvement": {
            "forward_deg": float(weighted_sf_fwd - weighted_sm_fwd),
            "forward_pct": float((weighted_sf_fwd - weighted_sm_fwd) / weighted_sf_fwd * 100)
                          if weighted_sf_fwd > 0 else 0.0,
            "up_deg": float(weighted_sf_up - weighted_sm_up),
            "up_pct": float((weighted_sf_up - weighted_sm_up) / weighted_sf_up * 100)
                      if weighted_sf_up > 0 else 0.0,
        },
    }

    # Save
    out_path = save_dir / "pose_kalman_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print report
    print("\n" + "=" * 60)
    print("HEAD POSE KALMAN SMOOTHING RESULTS")
    print("=" * 60)

    print(f"\nSingle-frame (deployment-honest) — per-recording mean:")
    print(f"  Forward MAE:  {np.mean(sf_fwd_arr):.2f} deg  (weighted: {weighted_sf_fwd:.2f} deg)")
    print(f"  Up-vector MAE: {np.mean(sf_up_arr):.2f} deg  (weighted: {weighted_sf_up:.2f} deg)")
    print(f"  Forward median: {np.median(sf_fwd_arr):.2f} deg")
    print(f"  Up-vector median: {np.median(sf_up_arr):.2f} deg")

    print(f"\nKalman-smoothed — per-recording mean:")
    print(f"  Forward MAE:  {np.mean(sm_fwd_arr):.2f} deg  (weighted: {weighted_sm_fwd:.2f} deg)")
    print(f"  Up-vector MAE: {np.mean(sm_up_arr):.2f} deg  (weighted: {weighted_sm_up:.2f} deg)")
    print(f"  Forward median: {np.median(sm_fwd_arr):.2f} deg")
    print(f"  Up-vector median: {np.median(sm_up_arr):.2f} deg")

    print(f"\nImprovement from smoothing:")
    fwd_imp = weighted_sf_fwd - weighted_sm_fwd
    up_imp = weighted_sf_up - weighted_sm_up
    fwd_pct = fwd_imp / weighted_sf_fwd * 100 if weighted_sf_fwd > 0 else 0
    up_pct = up_imp / weighted_sf_up * 100 if weighted_sf_up > 0 else 0
    print(f"  Forward:  -{fwd_imp:.2f} deg ({fwd_pct:.1f}%)")
    print(f"  Up-vector: -{up_imp:.2f} deg ({up_pct:.1f}%)")

    print(f"\nProcessed {n_total} frames across {len(per_recording_results)} recordings")
    print(f"Kalman params: Q={args.process_noise}, R={args.measurement_noise}")

    # Print per-recording table
    print(f"\n{'='*60}")
    print(f"Per-recording breakdown:")
    print(f"{'Recording':>20s}  {'N':>5s}  {'SF-fwd':>8s}  {'SM-fwd':>8s}  {'SF-up':>8s}  {'SM-up':>8s}")
    print(f"{'-'*60}")
    for rid in sorted(per_recording_results.keys()):
        r = per_recording_results[rid]
        print(f"{rid:>20s}  {r['n']:5d}  {r['single_frame_fwd_MAE_deg']:8.2f}  "
              f"{r['smoothed_fwd_MAE_deg']:8.2f}  {r['single_frame_up_MAE_deg']:8.2f}  "
              f"{r['smoothed_up_MAE_deg']:8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
