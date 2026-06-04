"""
Calibrate anchor sizes using k-means clustering on ground truth boxes.
Derived from paper: anchor sizes (24, 48, 96, 192, 384).
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import argparse

from config import C


def load_gt_boxes(data_root: str, split: str = "train"):
    """Load all ground truth boxes from dataset."""
    recordings_dir = Path(data_root) / "recordings" / split
    all_boxes = []

    for rec_dir in recordings_dir.iterdir():
        if not rec_dir.is_dir():
            continue

        od_labels_path = rec_dir / "OD_labels.json"
        if not od_labels_path.exists():
            continue

        with open(od_labels_path, 'r') as f:
            od_data = json.load(f)

        for ann in od_data.get("annotations", []):
            bbox = ann.get("bbox", [])
            if len(bbox) >= 4:
                x, y, w, h = bbox
                # Convert to center format for clustering
                cx = x + w / 2
                cy = y + h / 2
                all_boxes.append([cx, cy, w, h])

    return np.array(all_boxes)


def kmeans_anchors(boxes: np.ndarray, n_anchors: int = 5, size_threshold: float = 0.05):
    """
    Run k-means to find anchor sizes.
    boxes: [N, 4] in (cx, cy, w, h) format
    """
    # Use area and aspect ratio for clustering
    areas = boxes[:, 2] * boxes[:, 3]
    aspect_ratios = boxes[:, 2] / (boxes[:, 3] + 1e-6)

    # Filter outliers
    valid_mask = (areas > np.percentile(areas, size_threshold)) & \
                 (areas < np.percentile(areas, 100 - size_threshold))
    boxes_filtered = boxes[valid_mask]

    print(f"Using {len(boxes_filtered)} boxes for clustering (filtered from {len(boxes)})")

    # Cluster on (w, h) - ignore position
    features = boxes_filtered[:, 2:4]  # width, height

    kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=10)
    kmeans.fit(features)

    anchor_sizes = sorted(kmeans.cluster_centers_[:, 0])  # Sort by width

    print(f"K-means anchor sizes (width): {anchor_sizes}")
    print(f"K-means anchor heights: {sorted(kmeans.cluster_centers_[:, 1])}")

    return anchor_sizes


def main():
    parser = argparse.ArgumentParser(description="Calibrate anchor sizes with k-means")
    parser.add_argument("--data-root", type=str, default=C.DATA_ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-anchors", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Loading ground truth boxes...")
    boxes = load_gt_boxes(args.data_root, args.split)
    print(f"Loaded {len(boxes)} boxes")

    if len(boxes) == 0:
        print("No boxes found. Check data path.")
        return

    print(f"\nBox statistics:")
    print(f"  Width: min={boxes[:, 2].min():.1f}, max={boxes[:, 2].max():.1f}, mean={boxes[:, 2].mean():.1f}")
    print(f"  Height: min={boxes[:, 3].min():.1f}, max={boxes[:, 3].max():.1f}, mean={boxes[:, 3].mean():.1f}")

    print(f"\nRunning k-means with {args.n_anchors} clusters...")
    anchor_sizes = kmeans_anchors(boxes, args.n_anchors)

    print(f"\nCalibrated anchor sizes: {anchor_sizes}")

    # Compare with default
    default_sizes = C.ANCHOR_SIZES
    print(f"Default anchor sizes: {default_sizes}")

    # Save results
    result = {
        'anchor_sizes': anchor_sizes,
        'n_anchors': args.n_anchors,
        'split': args.split,
        'n_boxes_used': len(boxes)
    }

    output_path = args.output or Path(C.RUN_DIR) / "anchor_calibration.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()