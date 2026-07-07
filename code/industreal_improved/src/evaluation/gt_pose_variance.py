"""GT pose variance analysis — forward vs up-vector angular range.

Loads pose.csv directly from val recording directories (no GPU, no image
loading) to compute angular range of GT forward vectors vs up-vectors,
testing whether the up-vector advantage is partially mechanical.

Usage:
    python3 src/evaluation/gt_pose_variance.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Match config RECORDINGS_ROOT
RECORDINGS_ROOT = Path('/media/newadmin/master/POPW/datasets/industreal/recordings')
VAL_ROOT = RECORDINGS_ROOT / 'val'

# head_pose columns: forward_x, forward_y, forward_z, pos_x, pos_y, pos_z, up_x, up_y, up_z
POSE_COLS = ['fwd_x', 'fwd_y', 'fwd_z', 'pos_x', 'pos_y', 'pos_z', 'up_x', 'up_y', 'up_z']


def angular_range_deg(vectors: np.ndarray, max_pairs: int = 5000) -> float:
    """Maximum angular separation between any two unit vectors via sampling."""
    n = vectors.shape[0]
    if n < 2:
        return 0.0
    actual_pairs = min(max_pairs, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    max_angle = 0.0
    for _ in range(actual_pairs):
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j:
            cos = float(np.dot(vectors[i], vectors[j]).clip(-1.0, 1.0))
            angle = np.degrees(np.arccos(cos))
            if angle > max_angle:
                max_angle = angle
    return max_angle


def main():
    rec_dirs = sorted(VAL_ROOT.iterdir())
    rec_dirs = [d for d in rec_dirs if d.is_dir()]
    print(f"Found {len(rec_dirs)} val recording directories")

    all_fwd = []
    all_up = []
    per_recording = {}

    for rec_dir in rec_dirs:
        rec_id = rec_dir.name
        pose_csv = rec_dir / 'pose.csv'
        if not pose_csv.exists():
            print(f"  {rec_id}: no pose.csv, skipping")
            continue
        df = pd.read_csv(pose_csv, header=None, names=POSE_COLS)
        n = len(df)
        fwd = df[['fwd_x', 'fwd_y', 'fwd_z']].values.astype(np.float32)
        up = df[['up_x', 'up_y', 'up_z']].values.astype(np.float32)

        # Normalize to unit vectors
        fwd_n = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-6)
        up_n = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-6)

        # Per-recording stats
        fwd_range = angular_range_deg(fwd_n)
        up_range = angular_range_deg(up_n)

        per_recording[rec_id] = {
            "n_frames": n,
            "fwd_angular_range_deg": float(fwd_range),
            "up_angular_range_deg": float(up_range),
        }

        all_fwd.append(fwd_n)
        all_up.append(up_n)
        print(f"  {rec_id}: {n} frames, fwd_range={fwd_range:.1f} deg, up_range={up_range:.1f} deg")

    all_fwd = np.concatenate(all_fwd, axis=0)
    all_up = np.concatenate(all_up, axis=0)
    print(f"\nTotal: {all_fwd.shape[0]} frames")

    # Overall angular range
    print("Computing overall angular ranges...")
    fwd_range = angular_range_deg(all_fwd, max_pairs=20000)
    up_range = angular_range_deg(all_up, max_pairs=20000)
    fwd_ratio = fwd_range / up_range if up_range > 0 else float('inf')

    # Mean direction and dispersion (angular std)
    fwd_mean = all_fwd.mean(axis=0)
    fwd_mean_unit = fwd_mean / np.linalg.norm(fwd_mean)
    fwd_angles_to_mean = np.degrees(np.arccos(
        (all_fwd @ fwd_mean_unit).clip(-1.0, 1.0)
    ))
    fwd_dispersion = float(fwd_angles_to_mean.std())

    up_mean = all_up.mean(axis=0)
    up_mean_unit = up_mean / np.linalg.norm(up_mean)
    up_angles_to_mean = np.degrees(np.arccos(
        (all_up @ up_mean_unit).clip(-1.0, 1.0)
    ))
    up_dispersion = float(up_angles_to_mean.std())

    results = {
        "n_frames": all_fwd.shape[0],
        "forward": {
            "angular_range_deg": float(fwd_range),
            "mean_direction": fwd_mean_unit.tolist(),
            "mean_angle_deg": float(fwd_angles_to_mean.mean()),
            "std_angle_deg": fwd_dispersion,
        },
        "up_vector": {
            "angular_range_deg": float(up_range),
            "mean_direction": up_mean_unit.tolist(),
            "mean_angle_deg": float(up_angles_to_mean.mean()),
            "std_angle_deg": up_dispersion,
        },
        "comparison": {
            "range_ratio_fwd_over_up": float(fwd_ratio),
            "fwd_range_larger_by_factor": float(fwd_ratio),
            "interpretation": (
                "forward range is larger than up range" if fwd_ratio > 1.2
                else "forward and up ranges are comparable"
            ),
        },
        "per_recording": per_recording,
    }

    print("\n" + "=" * 60)
    print("GT POSE VARIANCE ANALYSIS (pose.csv, no GPU)")
    print("=" * 60)
    print(f"Total frames: {results['n_frames']}")
    print(f"\nForward vectors:")
    print(f"  Angular range: {results['forward']['angular_range_deg']:.2f} deg")
    print(f"  Mean angle from centroid: {results['forward']['mean_angle_deg']:.2f} deg")
    print(f"  Angular std: {results['forward']['std_angle_deg']:.2f} deg")
    print(f"\nUp vectors:")
    print(f"  Angular range: {results['up_vector']['angular_range_deg']:.2f} deg")
    print(f"  Mean angle from centroid: {results['up_vector']['mean_angle_deg']:.2f} deg")
    print(f"  Angular std: {results['up_vector']['std_angle_deg']:.2f} deg")
    print(f"\nRatio (fwd_range / up_range): {results['comparison']['range_ratio_fwd_over_up']:.2f}")
    print(f"Interpretation: {results['comparison']['interpretation']}")
    if fwd_ratio > 1.3:
        print("  -> Up-vector advantage IS partially mechanical (constrained up movement)")
    else:
        print("  -> Up-vector advantage is NOT explained by constrained movement alone")

    save_dir = Path("src/runs/rf_stages/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "gt_pose_variance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import sys
    main()
