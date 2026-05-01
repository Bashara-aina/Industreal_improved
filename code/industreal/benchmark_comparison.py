#!/usr/bin/env python3
"""
Benchmark Comparison Script for IndustReal POPW Model
=====================================================
Compares POPW evaluation results against published baselines.

Usage:
    python benchmark_comparison.py --checkpoint <path> [--split val] [--save-json path]

Metrics compared:
  1. Activity Top-1: >66.45% (MViTv2 Kinetics, WACV 2024)
  2. Activity Top-5: >88.43% (MViTv2 Kinetics, WACV 2024)
  3. ASD Detection mAP@0.5: >83.8% (YOLOv8m COCO+synth+real, WACV 2024)
  4. Head Pose MAE: vs raw GT (⚠️ N/A — report as "evaluate vs GT")
  5. PSR F1: >0.901 (STORM-PSR, CVIU 2025 arXiv:2510.12385)
  6. PSR POS: >0.812 (STORM-PSR, CVIU 2025 arXiv:2510.12385)

.. note::
  IndustReal has two published papers:
  - WACV 2024: B3 rule-based PSR, F1=0.883. Uses hand-crafted segmentation rules.
  - CVIU 2025 (arXiv:2510.12385): STORM-PSR learned approach, F1=0.901/0.812.
    This is the stronger method and is used as the benchmark target above.

Author: POPW Team
Date: April 2026
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from evaluate import evaluate_all
from losses import MultiTaskLoss
from model import MultiTaskIndustReal
from industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Targets
# =============================================================================
BENCHMARK_TARGETS = {
    "Activity Top-1": {
        "target": 66.45,
        "unit": "%",
        "source": "MViTv2 Kinetics (WACV 2024)",
        "higher_is_better": True,
    },
    "Activity Top-5": {
        "target": 88.43,
        "unit": "%",
        "source": "MViTv2 Kinetics (WACV 2024)",
        "higher_is_better": True,
    },
    "ASD Detection mAP@0.5": {
        "target": 83.8,
        "unit": "%",
        "source": "YOLOv8m COCO+synth+real (WACV 2024)",
        "higher_is_better": True,
    },
    "Head Pose MAE": {
        "target": None,  # No baseline - compare against GT directly
        "unit": "MAE",
        "source": "Ground Truth (⚠️ N/A — report as evaluate vs GT)",
        "higher_is_better": False,
        "na_mode": True,
    },
    "PSR F1": {
        "target": 0.901,
        "unit": "",
        "source": "STORM-PSR (CVIU 2025 arXiv:2510.12385)",
        "higher_is_better": True,
    },
    "PSR POS": {
        "target": 0.812,
        "unit": "",
        "source": "STORM-PSR (CVIU 2025 arXiv:2510.12385)",
        "higher_is_better": True,
    },
}


def format_value(value: float, unit: str) -> str:
    """Format a metric value for display."""
    if unit == "%":
        return f"{value:.2f}%"
    elif unit == "MAE":
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"


def determine_beat(value: float, target: Optional[float], higher_is_better: bool, na_mode: bool = False) -> tuple:
    """
    Determine if a metric beats the target.

    Returns: (status_icon, status_text)
    """
    if na_mode:
        return "⚠️", "N/A"

    if target is None:
        return "⚠️", "N/A"

    if higher_is_better:
        if value >= target:
            return "✅", "BEAT"
        else:
            return "❌", "BELOW"
    else:
        if value <= target:
            return "✅", "BEAT"
        else:
            return "❌", "BELOW"


def print_comparison_table(results: Dict[str, Any], save_json: Optional[str] = None) -> int:
    """
    Print the benchmark comparison table.

    Returns: number of benchmarks beaten
    """
    # Extract metrics from results
    # Activity Top-1 is act_accuracy (frame accuracy)
    # Activity Top-5 is act_top5_accuracy
    # ASD mAP@0.5 is det_mAP50
    # Head Pose MAE is head_pose_MAE
    # PSR F1 is psr_overall_f1
    # PSR POS is psr_pos

    metrics_map = {
        "Activity Top-1": {
            "value": results.get("act_accuracy", 0.0) * 100,  # Convert to percentage
            "key": "act_accuracy",
        },
        "Activity Top-5": {
            "value": results.get("act_top5_accuracy", 0.0) * 100,  # Convert to percentage
            "key": "act_top5_accuracy",
        },
        "ASD Detection mAP@0.5": {
            "value": results.get("det_mAP50", 0.0) * 100,  # Convert to percentage
            "key": "det_mAP50",
        },
        "Head Pose MAE": {
            "value": results.get("head_pose_MAE", float('nan')),
            "key": "head_pose_MAE",
        },
        "PSR F1": {
            "value": results.get("psr_overall_f1", 0.0),
            "key": "psr_overall_f1",
        },
        "PSR POS": {
            "value": results.get("psr_pos", 0.0),
            "key": "psr_pos",
        },
    }

    # Print header
    print("\n" + "=" * 100)
    print("IndustReal Benchmark Comparison Table")
    print("=" * 100)
    print(f"{'Metric':<30} {'POPW':<15} {'Target':<15} {'Baseline':<35} {'Beat?':<10}")
    print("-" * 100)

    benchmarks_beaten = 0
    table_rows = []

    for metric_name, benchmark in BENCHMARK_TARGETS.items():
        target = benchmark["target"]
        unit = benchmark["unit"]
        source = benchmark["source"]
        higher_is_better = benchmark["higher_is_better"]
        na_mode = benchmark.get("na_mode", False)

        metric_info = metrics_map.get(metric_name, {})
        value = metric_info.get("value", float('nan'))
        key = metric_info.get("key", "")

        # Format target string
        if na_mode:
            target_str = "N/A"
        elif target is not None:
            target_str = format_value(target, unit)
        else:
            target_str = "N/A"

        # Format baseline (source)
        baseline_str = source

        # Format POPW value
        if na_mode:
            popw_str = format_value(value, "MAE")
        else:
            popw_str = format_value(value, unit)

        # Determine beat status
        icon, status = determine_beat(value, target, higher_is_better, na_mode)

        if status == "BEAT":
            benchmarks_beaten += 1

        # Format for table
        target_display = f"{target_str}" if not na_mode else "vs GT"

        row = {
            "metric": metric_name,
            "popw": value,
            "target": target,
            "baseline": source,
            "status": status,
            "icon": icon,
        }
        table_rows.append(row)

        print(f"{metric_name:<30} {popw_str:<15} {target_display:<15} {baseline_str:<35} {icon} {status}")

    print("-" * 100)

    # Summary
    summary = f"POPW beats {benchmarks_beaten}/6 benchmarks"
    print(f"\n{summary}")
    print("=" * 100 + "\n")

    # Save JSON if requested
    if save_json:
        output_data = {
            "summary": summary,
            "benchmarks_beaten": benchmarks_beaten,
            "total_benchmarks": 6,
            "results": table_rows,
            "raw_metrics": {
                "act_accuracy": results.get("act_accuracy"),
                "act_top5_accuracy": results.get("act_top5_accuracy"),
                "det_mAP50": results.get("det_mAP50"),
                "head_pose_MAE": results.get("head_pose_MAE"),
                "psr_overall_f1": results.get("psr_overall_f1"),
                "psr_pos": results.get("psr_pos"),
            },
        }
        with open(save_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {save_json}")

    return benchmarks_beaten


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison for IndustReal POPW model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_comparison.py --checkpoint path/to/checkpoint.pth

  python benchmark_comparison.py --checkpoint model.pt --split val

  python benchmark_comparison.py --checkpoint model.pt --save-json results.json
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: val)"
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save results as JSON"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=9999,
        help="Maximum number of batches to evaluate (default: 9999)"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = MultiTaskIndustReal(pretrained=False).to(device)

    criterion = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    logger.info("Checkpoint loaded successfully")

    # Load dataset
    logger.info(f"Loading {args.split} dataset...")
    ds = IndustRealMultiTaskDataset(
        split=args.split,
        img_size=C.IMG_SIZE,
        augment=False,
        seed=C.SEED,
    )
    loader = DataLoader(
        ds,
        batch_size=C.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=C.VAL_NUM_WORKERS,
        collate_fn=collate_fn,
    )
    criterion.set_class_counts(ds.class_counts)
    logger.info(f"Dataset loaded: {len(ds)} samples")

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluate_all(
        model, criterion, loader, device,
        max_batches=args.max_batches, save_dir=None,
    )

    # Print comparison table
    benchmarks_beaten = print_comparison_table(results, args.save_json)

    # Log detailed results
    logger.info("\nDetailed Results:")
    logger.info(f"  Activity Top-1: {results.get('act_accuracy', 0) * 100:.2f}%")
    logger.info(f"  Activity Top-5: {results.get('act_top5_accuracy', 0) * 100:.2f}%")
    logger.info(f"  ASD mAP@0.5: {results.get('det_mAP50', 0) * 100:.2f}%")
    logger.info(f"  Head Pose MAE: {results.get('head_pose_MAE', float('nan')):.4f}")
    logger.info(f"  PSR F1: {results.get('psr_overall_f1', 0):.4f}")
    logger.info(f"  PSR POS: {results.get('psr_pos', 0):.4f}")

    logger.info(f"\nBenchmarks Beaten: {benchmarks_beaten}/6")

    # Exit with appropriate code
    sys.exit(0)


if __name__ == "__main__":
    main()
