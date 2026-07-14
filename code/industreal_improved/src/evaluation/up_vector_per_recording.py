"""
Up-Vector Angular MAE per Recording
====================================
Computes per-recording head-pose up-vector angular MAE breakdown
across the full validation set.

The head pose output is [B, 6] — [fwd_angle_rad, up_angle_rad, pos_x_mm, pos_y_mm, pos_z_mm].
Up-vector (pitch) is index 1.

Reports:
  - Per-recording mean, median, IQR
  - Full-eval median with IQR
  - Identifies outlier recording (highest MAE)

Saves to: src/runs/rf_stages/checkpoints/up_vector_per_recording.json

Usage:
  python src/evaluation/up_vector_per_recording.py         # real eval
  python src/evaluation/up_vector_per_recording.py --demo  # synthetic demo (no model needed)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = REPO_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "best.pth"
OUTPUT_PATH = (
    REPO_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "up_vector_per_recording.json"
)

# Add industreal source to path
INDUSTREAL_SRC = REPO_ROOT / ".wiki" / "archive-research" / "industreal_improved"
if INDUSTREAL_SRC.exists():
    sys.path.insert(0, str(INDUSTREAL_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_recording_id(video_id: str) -> str:
    """Extract recording ID from a video_id string.

    Expected formats:
      "rec_003_seq_02"        -> "rec_003"
      "recording_003/frame_42" -> "recording_003"
    Falls back to first 12 characters.
    """
    if video_id.startswith("rec_"):
        parts = video_id.split("_")
        return f"rec_{parts[1]}"
    if video_id.startswith("recording_"):
        return video_id.split("/")[0]
    return video_id[:12]


def compute_up_vector_mae_per_recording(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
) -> dict[str, dict]:
    """
    Compute per-recording up-vector angular MAE.

    Args:
        model: POPWMultiTaskModel
        val_loader: DataLoader yielding batches
        device: "cuda" or "cpu"

    Returns:
        dict[recording_id, {"errors": list[float], "mean": float, "median": float,
                            "q25": float, "q75": float, "count": int}]
    """
    model.eval()
    recording_errors: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["images"].to(device)
            video_ids = batch["video_ids"]
            gt_pose = batch["head_pose_labels"].to(device)

            B, T = images.shape[:2]
            images_flat = images.view(B * T, images.shape[2], images.shape[3], images.shape[4])
            gt_pose_flat = gt_pose.view(B * T, 6)

            outputs = model(images=images_flat, video_ids=None, clip_rgb=None)
            pred_pose = outputs["head_pose"]

            pred_up = pred_pose[:, 1].abs() * 180.0 / np.pi
            gt_up = gt_pose_flat[:, 1].abs() * 180.0 / np.pi
            up_mae = (pred_up - gt_up).abs().cpu().numpy()

            idx = 0
            for b in range(B):
                rec_id = extract_recording_id(video_ids[b])
                recording_errors[rec_id].extend(up_mae[idx : idx + T].tolist())
                idx += T

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    results = {}
    for rec_id, errors in recording_errors.items():
        arr = np.array(errors)
        results[rec_id] = {
            "errors": errors,
            "count": len(errors),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }
    return results


def compute_aggregate_stats(
    per_recording: dict[str, dict],
) -> dict:
    """Compute full-eval aggregate statistics from per-recording data."""
    records = list(per_recording.values())
    medians = np.array([r["median"] for r in records])

    outlier_idx = int(np.argmax(medians))
    outlier_id = list(per_recording.keys())[outlier_idx]

    return {
        "per_recording": {
            rid: {
                "mean": r["mean"],
                "median": r["median"],
                "q25": r["q25"],
                "q75": r["q75"],
                "iqr": r["iqr"],
                "count": r["count"],
            }
            for rid, r in per_recording.items()
        },
        "full_eval": {
            "median_of_medians": float(np.median(medians)),
            "iqr_of_medians": float(np.percentile(medians, 75) - np.percentile(medians, 25)),
        },
        "outlier": {
            "recording_id": outlier_id,
            "median_mae": float(medians[outlier_idx]),
        },
        "n_recordings": len(records),
    }


def print_report(stats: dict) -> None:
    """Print formatted report."""
    print("\n" + "=" * 60)
    print("Per-Recording Up-Vector Angular MAE (deg)")
    print("=" * 60)
    for rid in sorted(stats["per_recording"].keys()):
        r = stats["per_recording"][rid]
        print(
            f"  {rid:>12s}:  median={r['median']:6.2f}  "
            f"IQR=[{r['q25']:5.2f}, {r['q75']:5.2f}]  "
            f"mean={r['mean']:6.2f}  n={r['count']:4d}"
        )

    fe = stats["full_eval"]
    full_med = fe["median_of_medians"]
    iqr_half = fe["iqr_of_medians"] / 2.0
    print(f"\n--- Full Evaluation ---")
    print(f"  Median of per-recording medians: {full_med:.2f} deg")
    print(f"  IQR: [{full_med - iqr_half:.2f}, {full_med + iqr_half:.2f}]")

    out = stats["outlier"]
    print(f"\n--- Outlier (highest median MAE) ---")
    print(f"  Recording: {out['recording_id']}")
    print(f"  Median MAE: {out['median_mae']:.2f} deg")

    print(f"\n=== VERDICT ===")
    print(
        f"  Full-eval median with IQR: {full_med:.2f} deg "
        f"([{full_med - iqr_half:.2f}, {full_med + iqr_half:.2f}])"
    )
    print(f"  Outlier: {out['recording_id']} @ {out['median_mae']:.2f} deg")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo mode (no model needed)
# ---------------------------------------------------------------------------


def run_demo():
    """Run with demo data to verify logic without a trained model."""
    print("[DEMO] Running with synthetic data (no model needed)")
    rng = np.random.RandomState(42)

    recordings = {}
    for i in range(8):
        n = 64
        if i == 3:
            errors = rng.exponential(scale=5.0, size=n)
        else:
            errors = rng.exponential(scale=1.0, size=n)
        rid = f"rec_{i:03d}"
        recordings[rid] = {
            "errors": errors.tolist(),
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "q25": float(np.percentile(errors, 25)),
            "q75": float(np.percentile(errors, 75)),
            "iqr": float(np.percentile(errors, 75) - np.percentile(errors, 25)),
            "count": len(errors),
        }

    stats = compute_aggregate_stats(recordings)

    print_report(stats)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "metadata": {
                    "checkpoint": str(CHECKPOINT_PATH),
                    "status": "demo_mode",
                    "description": "Per-recording up-vector angular MAE breakdown (SYNTHETIC DEMO)",
                    "warning": "No checkpoint found at best.pth. These are synthetic placeholder results.",
                },
                **stats,
            },
            f,
            indent=2,
        )
    print(f"\n[OK] Demo results saved to: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Per-recording up-vector angular MAE breakdown")
    parser.add_argument(
        "--demo", action="store_true", help="Run with synthetic data (no model/checkpoint needed)"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, cpu)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not CHECKPOINT_PATH.exists():
        print(f"[FATAL] Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"        Train a model first, or run with --demo for a synthetic demonstration.")
        sys.exit(1)

    # Only import torch when needed (avoids OOM during --demo)
    import torch
    from torch.utils.data import DataLoader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Import model
    try:
        from model import POPWMultiTaskModel
    except ImportError as e:
        print(f"[FATAL] Cannot import POPWMultiTaskModel: {e}")
        print(f"        Expected source at: {INDUSTREAL_SRC}")
        sys.exit(1)

    try:
        import config as C  # noqa: F401
    except ImportError as e:
        print(f"[FATAL] Cannot import config: {e}")
        sys.exit(1)

    # Build validation DataLoader
    try:
        from industreal_dataset import IndustRealDataset

        dataset = IndustRealDataset(
            split="val",
            data_root=str(REPO_ROOT / "data" / "industreal"),
            num_frames=16,
            stride=2,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        print(f"Validation dataset: {len(dataset)} samples, {len(val_loader)} batches")
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"[FATAL] Cannot load validation dataset: {e}")
        print(f"        Expected IndustRealDataset at: industreal_dataset.py")
        print(f"        Data root: {REPO_ROOT / 'data' / 'industreal'}")
        sys.exit(1)

    # Load model and checkpoint
    print("Loading model...")
    model = POPWMultiTaskModel(pretrained=False)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model = model.to(device)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    # Evaluate
    per_recording = compute_up_vector_mae_per_recording(model, val_loader, device=device)
    print(f"Recordings found: {len(per_recording)}")

    stats = compute_aggregate_stats(per_recording)
    print_report(stats)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "metadata": {
                    "checkpoint": str(CHECKPOINT_PATH),
                    "status": "completed",
                    "batch_size": args.batch_size,
                    "device": device,
                    "description": "Per-recording up-vector angular MAE breakdown",
                },
                **stats,
            },
            f,
            indent=2,
        )
    print(f"\n[OK] Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
