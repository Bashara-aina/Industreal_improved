"""Up-vector angular MAE per recording breakdown.

Per Opus Q7: "Treat 26.20° (full eval) as the number of record until P2.4's
per-recording breakdown says otherwise. Report the full-eval median with IQR
per debate 4.2's resolution."

Outputs:
  - per-recording mean, median, q25, q75, IQR
  - full-eval median of medians with IQR
  - identifies outlier recording
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def angular_mae_3d(pred, gt):
    """Compute angular MAE in degrees between two 3D unit vectors."""
    cos_angle = torch.clamp(
        (pred * gt).sum() / (torch.norm(pred) * torch.norm(gt) + 1e-12), -1.0, 1.0
    )
    return torch.acos(cos_angle) * (180.0 / np.pi)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="src/runs/rf_stages/checkpoints/best.pth")
    parser.add_argument("--max-batches", type=int, default=20000)
    parser.add_argument(
        "--save-dir", default="src/runs/rf_stages/checkpoints/up_vector_per_recording_v2"
    )
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    from src.models.model import POPWMultiTaskModel
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type="convnext_tiny",
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=True,
    )
    state_dict = {
        k: v for k, v in ckpt["model"].items() if "total_ops" not in k and "total_params" not in k
    }
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()

    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=False,
    )

    recording_errors = defaultdict(list)
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
        # up-vector is indices [6:9] — 3D unit vector
        pred_up = hp[0, 6:9]
        gt_up = gt_hp[0, 6:9].to(device=pred_up.device)

        # Compute angular MAE in degrees
        mae = angular_mae_3d(pred_up, gt_up).item()

        meta = targets.get("metadata", [])
        rec_id = meta[0].get("recording_id", "unknown")
        recording_errors[rec_id].append(mae)
        n += 1
        if n % 2000 == 0:
            print(f"  processed {n} frames across {len(recording_errors)} recordings...")

    print(f"\nProcessed {n} frames across {len(recording_errors)} recordings")

    results = {}
    for rec_id, errors in recording_errors.items():
        arr = np.array(errors)
        results[rec_id] = {
            "errors": errors[:100] if len(errors) > 100 else errors,  # cap for json size
            "count": int(len(errors)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }

    medians = np.array([r["median"] for r in results.values()])
    if len(medians) > 0:
        outlier_idx = int(np.argmax(medians))
        outlier_id = list(results.keys())[outlier_idx]
    else:
        outlier_id = "N/A"

    summary = {
        "checkpoint": args.checkpoint,
        "n_frames": n,
        "n_recordings": len(results),
        "per_recording": {
            rid: {
                "mean": r["mean"],
                "median": r["median"],
                "q25": r["q25"],
                "q75": r["q75"],
                "iqr": r["iqr"],
                "count": r["count"],
            }
            for rid, r in results.items()
        },
        "full_eval": {
            "median_of_medians": float(np.median(medians)) if len(medians) > 0 else 0.0,
            "iqr_of_medians": float(np.percentile(medians, 75) - np.percentile(medians, 25))
            if len(medians) > 0
            else 0.0,
        },
        "outlier": {
            "recording_id": outlier_id,
            "median_mae": float(medians[outlier_idx]) if len(medians) > 0 else 0.0,
        },
        "interpretation": (
            "Per Opus Q7: report full-eval median with IQR. "
            "Outlier identified separately. The 26.20° full-eval number may be skewed by the outlier recording."
        ),
    }

    out = Path(args.save_dir) / "up_vector_per_recording.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"\n{'=' * 60}")
    print(f"Per-Recording Up-Vector Angular MAE (deg)")
    print(f"{'=' * 60}")
    for rid in sorted(results.keys()):
        r = summary["per_recording"][rid]
        print(
            f"  {rid:>16s}:  median={r['median']:6.2f}  "
            f"IQR=[{r['q25']:5.2f}, {r['q75']:5.2f}]  "
            f"mean={r['mean']:6.2f}  n={r['count']:5d}"
        )

    fe = summary["full_eval"]
    full_med = fe["median_of_medians"]
    iqr_half = fe["iqr_of_medians"] / 2.0
    print(f"\n--- Full Evaluation ---")
    print(f"  Median of per-recording medians: {full_med:.2f} deg")
    print(f"  IQR: [{full_med - iqr_half:.2f}, {full_med + iqr_half:.2f}]")
    print(f"\n--- Outlier ---")
    print(f"  Recording: {summary['outlier']['recording_id']}")
    print(f"  Median MAE: {summary['outlier']['median_mae']:.2f} deg")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
