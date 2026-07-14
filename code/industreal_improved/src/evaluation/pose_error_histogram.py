"""Pose frame-level error histogram + fwd/up error correlation.

Plots histogram of frame-level forward and up-vector errors (using per-frame
errors from the Kalman eval cache), and computes Pearson correlation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 src/evaluation/pose_error_histogram.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame angular error in degrees."""
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-6)
    gt_n = gt / np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-6)
    cos = np.sum(pred_n * gt_n, axis=1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", default="src/runs/rf_stages/checkpoints/pose_kalman_eval/cache"
    )
    parser.add_argument("--save-dir", default="src/runs/rf_stages/checkpoints/pose_kalman_eval")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cached predictions from {cache_dir}...")
    recording_preds = {}
    for fpath in sorted(cache_dir.glob("*.npz")):
        data = np.load(fpath)
        rec_id = str(data["recording_id"])
        n_frames = data["pred_fwd"].shape[0]
        recording_preds[rec_id] = {
            "pred_fwd": data["pred_fwd"],
            "pred_up": data["pred_up"],
            "gt_fwd": data["gt_fwd"],
            "gt_up": data["gt_up"],
        }
        print(f"  {rec_id}: {n_frames} frames")

    # Per-frame errors
    all_fwd_errors = []
    all_up_errors = []
    for rec_id, frames in recording_preds.items():
        fwd_err = angular_error_deg(frames["pred_fwd"], frames["gt_fwd"])
        up_err = angular_error_deg(frames["pred_up"], frames["gt_up"])
        all_fwd_errors.extend(fwd_err.tolist())
        all_up_errors.extend(up_err.tolist())

    all_fwd_errors = np.array(all_fwd_errors)
    all_up_errors = np.array(all_up_errors)

    # Pearson correlation between fwd and up errors per frame
    from scipy.stats import pearsonr, spearmanr

    corr, corr_p = pearsonr(all_fwd_errors, all_up_errors)
    sp_corr, sp_corr_p = spearmanr(all_fwd_errors, all_up_errors)

    print(f"\nFrame-level error statistics:")
    print(
        f"  Forward: mean={all_fwd_errors.mean():.2f} deg, "
        f"median={np.median(all_fwd_errors):.2f} deg, "
        f"std={all_fwd_errors.std():.2f} deg"
    )
    print(
        f"  Up:      mean={all_up_errors.mean():.2f} deg, "
        f"median={np.median(all_up_errors):.2f} deg, "
        f"std={all_up_errors.std():.2f} deg"
    )
    print(f"  Pearson r: {corr:.4f} (p={corr_p:.2e})")
    print(f"  Spearman rho: {sp_corr:.4f} (p={sp_corr_p:.2e})")

    stats = {
        "n_frames": len(all_fwd_errors),
        "forward_error": {
            "mean_deg": float(all_fwd_errors.mean()),
            "median_deg": float(np.median(all_fwd_errors)),
            "std_deg": float(all_fwd_errors.std()),
            "p25_deg": float(np.percentile(all_fwd_errors, 25)),
            "p75_deg": float(np.percentile(all_fwd_errors, 75)),
        },
        "up_error": {
            "mean_deg": float(all_up_errors.mean()),
            "median_deg": float(np.median(all_up_errors)),
            "std_deg": float(all_up_errors.std()),
            "p25_deg": float(np.percentile(all_up_errors, 25)),
            "p75_deg": float(np.percentile(all_up_errors, 75)),
        },
        "correlation": {
            "pearson_r": float(corr),
            "pearson_p": float(corr_p),
            "spearman_rho": float(sp_corr),
            "spearman_p": float(sp_corr_p),
        },
    }

    with open(save_dir / "pose_error_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, data, color, label in [
        (axes[0], all_fwd_errors, "#2196F3", "Forward"),
        (axes[1], all_up_errors, "#FF5722", "Up-vector"),
    ]:
        ax.hist(data, bins=80, color=color, alpha=0.7, density=True)
        ax.axvline(
            np.median(data),
            color="black",
            ls="--",
            lw=1.5,
            label=f"median={np.median(data):.1f} deg",
        )
        ax.set_xlabel("Angular Error (deg)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Frame-Level MAE")
        ax.legend(fontsize=8)
        ax.set_xlim(0, min(data.max(), 60))

    # Scatter plot fwd vs up
    # Subsample for scatter (50k max)
    scatter_n = min(50000, len(all_fwd_errors))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(all_fwd_errors), scatter_n, replace=False)
    axes[2].scatter(all_fwd_errors[idx], all_up_errors[idx], s=1, alpha=0.3, c="#4CAF50")
    axes[2].set_xlabel("Forward Error (deg)")
    axes[2].set_ylabel("Up-Vector Error (deg)")
    axes[2].set_title(f"Fwd vs Up Error (r={corr:.2f})")
    axes[2].set_xlim(0, 50)
    axes[2].set_ylim(0, 50)

    plt.tight_layout()
    plot_path = save_dir / "pose_error_histogram.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"\nHistogram saved to {plot_path}")
    plt.close()

    print(f"Stats saved to {save_dir / 'pose_error_stats.json'}")


if __name__ == "__main__":
    main()
