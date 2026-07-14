#!/usr/bin/env python3
"""
Anchor Calibration Script (Doc 01 B.3)

Clusters ground-truth bounding box areas from training set using k-means
to find 5 anchor sizes that minimize average IoU loss.

Run after synthetic pretrain data collection:
    python calibrate_anchors.py --split train --output anchors_calibrated.py

The recommended anchor sizes are written to config.py as ANCHOR_SIZES.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate anchors via k-means on GT boxes")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/newadmin/swarm-bot/project/popw/working/data/datasets/industreal",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=5, help="Number of anchor size clusters (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anchors_calibrated.txt",
        help="Output file for calibrated sizes",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_gt_boxes(data_root: str, split: str) -> List[Tuple[float, float]]:
    """
    Load all ground-truth bounding boxes from OD_labels.json files.

    Returns:
        List of (width, height) tuples for each box.
    """
    split_root = Path(data_root) / split
    if not split_root.exists():
        print(f"[calibrate_anchors] ERROR: split root not found: {split_root}")
        sys.exit(1)

    all_boxes: List[Tuple[float, float]] = []
    rec_dirs = [d for d in sorted(split_root.iterdir()) if d.is_dir()]
    print(f"[calibrate_anchors] Scanning {len(rec_dirs)} recordings...")

    for rec_dir in rec_dirs:
        od_path = rec_dir / "OD_labels.json"
        if not od_path.exists():
            continue

        try:
            with open(od_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] Failed to load {od_path}: {e}")
            continue

        for ann in coco.get("annotations", []):
            bbox = ann.get("bbox", [])
            if len(bbox) >= 4:
                w, h = float(bbox[2]), float(bbox[3])
                if w > 0 and h > 0:
                    all_boxes.append((w, h))

    print(f"[calibrate_anchors] Collected {len(all_boxes)} GT boxes")
    return all_boxes


def kmeans_anchor_sizes(
    boxes: List[Tuple[float, float]],
    num_clusters: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Run k-means on (w, h) to find representative anchor sizes.
    Uses log-scale clustering for better area-based sizing.
    """
    if len(boxes) < num_clusters:
        print(f"[calibrate_anchors] WARNING: only {len(boxes)} boxes, using min(boxes) as fallback")
        areas = np.array([w * h for w, h in boxes])
        return np.array(sorted(set(np.sqrt(areas))))[:num_clusters]

    X = np.log2(np.array(boxes) + 1e-6)  # log-scale for better clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    kmeans.fit(X)

    sizes = []
    for center in kmeans.cluster_centers_:
        w = 2 ** center[0]
        h = 2 ** center[1]
        area = math.sqrt(w * h)
        sizes.append(area)

    sizes = sorted(set(round(s) for s in sizes))
    while len(sizes) < num_clusters:
        sizes.append(sizes[-1] * 2 if sizes else 32)

    return np.array(sizes[:num_clusters])


def analyze_box_distribution(boxes: List[Tuple[float, float]]) -> dict:
    """Compute statistics about the GT box distribution."""
    areas = np.array([w * h for w, h in boxes])
    widths = np.array([w for w, h in boxes])
    heights = np.array([h for w, h in boxes])

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    stats = {}
    for p in percentiles:
        stats[f"area_p{p}"] = float(np.percentile(areas, p))
        stats[f"width_p{p}"] = float(np.percentile(widths, p))
        stats[f"height_p{p}"] = float(np.percentile(heights, p))
    stats["count"] = len(boxes)
    return stats


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    boxes = load_gt_boxes(args.data_root, args.split)
    if not boxes:
        print("[calibrate_anchors] ERROR: No GT boxes found")
        sys.exit(1)

    stats = analyze_box_distribution(boxes)
    print(f"\n[calibrate_anchors] Box distribution stats:")
    for k, v in stats.items():
        if k != "count":
            print(f"  {k}: {v:.1f}")

    anchor_sizes = kmeans_anchor_sizes(boxes, num_clusters=args.num_clusters, seed=args.seed)

    print(f"\n[calibrate_anchors] Recommended anchor sizes (area-based):")
    for i, s in enumerate(anchor_sizes):
        print(f"  level {i}: {s:.1f}px (sqrt-area)")

    anchor_areas = anchor_sizes**2
    print(f"\n[calibrate_anchors] As (w, h) anchor sizes (aspect=1:1):")
    for i, s in enumerate(anchor_sizes):
        print(f"  level {i}: w={s:.1f}, h={s:.1f}")

    with open(args.output, "w") as f:
        f.write(f"# Calibrated anchor sizes from k-means on {stats['count']} GT boxes\n")
        f.write(f"# Generated by calibrate_anchors.py --split {args.split}\n")
        f.write(f"ANCHOR_SIZES = ({', '.join(str(int(round(s))) for s in anchor_sizes)})\n")

    print(f"\n[calibrate_anchors] Written to {args.output}")
    print(f"\nRecommended config update:")
    print(f"  ANCHOR_SIZES = ({', '.join(str(int(round(s))) for s in anchor_sizes)})")


if __name__ == "__main__":
    main()
