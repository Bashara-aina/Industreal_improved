#!/usr/bin/env python3
"""
Ablation Table Generator (Doc 03 C.3)

Reads experiment results from JSON files and generates a formatted markdown
ablation table comparing multiple configurations.

Usage:
    python generate_ablation_table.py \
        --runs runs/exp1/best.json runs/exp2/best.json \
        --names "Baseline" "Full Recipe" \
        --output ablation_table.md
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_DISPLAY_NAMES = {
    "det_mAP50": "ASD mAP@0.5",
    "det_mAP_50_95": "ASD mAP",
    "act_accuracy": "AR Top-1",
    "act_top1": "AR Top-1",
    "act_macro_f1": "AR Macro F1",
    "head_pose_MAE": "Pose MAE",
    "head_pose_angular_MAE_deg": "Pose Ang. MAE (°)",
    "psr_macro_f1": "PSR Macro F1",
    "psr_micro_f1": "PSR Micro F1",
    "combined_metric": "Combined",
    "streaming_fps": "Stream FPS",
    "fps": "Batch FPS",
    "gflops": "GFLOPs",
    "peak_memory_mb": "GPU Mem (MB)",
    "total_params_m": "Params (M)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ablation table from run results")
    parser.add_argument("--runs", nargs="+", required=True, help="Paths to result JSON files")
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Display names for each run (default: extract from path)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "det_mAP50",
            "act_accuracy",
            "act_macro_f1",
            "head_pose_MAE",
            "psr_macro_f1",
            "fps",
        ],
        help="Metrics to include in the table",
    )
    parser.add_argument(
        "--output", type=str, default="ablation_table.md", help="Output markdown file"
    )
    parser.add_argument(
        "--sort-by", type=str, default=None, help="Metric to sort rows by (default: first metric)"
    )
    parser.add_argument(
        "--higher-is-better",
        type=str,
        default="det_mAP50,act_accuracy,act_macro_f1,psr_macro_f1",
        help="Comma-separated metrics where higher is better",
    )
    return parser.parse_args()


def load_result(path: str) -> Dict[str, Any]:
    """Load a result JSON file. Supports both full results and best-checkpoint dicts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "val_metrics" in data:
        return data["val_metrics"]
    if "metrics" in data:
        return data["metrics"]
    if "result" in data:
        return data["result"]
    return data


def extract_metric(data: Dict[str, Any], metric: str) -> Optional[float]:
    """Extract a metric value from a result dict, checking nested structures."""
    if metric in data:
        return float(data[metric])

    for section in ["val", "test", "metrics", "evaluation"]:
        if section in data and isinstance(data[section], dict):
            if metric in data[section]:
                return float(data[section][metric])

    return None


def format_metric(value: float, metric: str, higher_is_better: bool) -> str:
    """Format a metric value for display."""
    if math.isnan(value) or math.isinf(value):
        return "—"

    if metric in ("fps", "streaming_fps"):
        return f"{value:.1f}"
    if metric in ("gflops",):
        return f"{value:.1f}"
    if metric in ("peak_memory_mb",):
        return f"{value:.0f}"
    if metric in ("total_params_m",):
        return f"{value:.2f}"
    if metric in ("head_pose_MAE", "head_pose_angular_MAE_deg"):
        return f"{value:.3f}"
    return f"{value:.4f}"


def get_best_run(runs: List[Dict[str, Any]], metric: str, higher_is_better: bool) -> int:
    """Return index of the best run for a given metric."""
    values = []
    for r in runs:
        v = extract_metric(r["data"], metric)
        values.append(v if v is not None else float("-inf"))

    if higher_is_better:
        return int(max(range(len(values)), key=lambda i: values[i]))
    else:
        return int(min(range(len(values)), key=lambda i: values[i]))


def render_table(
    runs: List[Dict[str, Any]],
    metric_names: List[str],
    higher_is_better: List[bool],
    sort_idx: Optional[int] = None,
) -> str:
    """Render the ablation table as a markdown string."""
    lines = []
    lines.append(
        "| Configuration | "
        + " | ".join(METRIC_DISPLAY_NAMES.get(m, m) for m in metric_names)
        + " |"
    )
    lines.append("|" + "|".join(["---"] * (len(metric_names) + 1)) + "|")

    if sort_idx is not None:
        runs = sorted(
            runs,
            key=lambda r: extract_metric(r["data"], metric_names[sort_idx]) or float("-inf"),
            reverse=higher_is_better[sort_idx] if sort_idx < len(higher_is_better) else True,
        )

    for run in runs:
        cells = [run["display_name"]]
        for metric in metric_names:
            val = extract_metric(run["data"], metric)
            if val is None:
                cells.append("—")
            else:
                cells.append(format_metric(val, metric, True))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    higher_is_better_set = set(args.higher_is_better.split(","))
    metric_higher = [m in higher_is_better_set for m in args.metrics]

    if len(args.runs) != len(args.names):
        if args.names is None:
            args.names = [Path(p).stem for p in args.runs]
        else:
            sys.exit("ERROR: --runs and --names must have the same length")

    runs = []
    for path_str, name in zip(args.runs, args.names):
        path = Path(path_str)
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue
        try:
            data = load_result(str(path))
            runs.append({"path": str(path), "display_name": name, "data": data})
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")

    if not runs:
        sys.exit("ERROR: No valid result files found")

    sort_idx = None
    if args.sort_by:
        if args.sort_by in args.metrics:
            sort_idx = args.metrics.index(args.sort_by)

    table_md = render_table(runs, args.metrics, metric_higher, sort_idx)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        f.write(f"<!-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} -->\n")
        metrics_sep = " | ".join(args.metrics)
        f.write(f"<!-- Metrics: {metrics_sep} -->\n\n")
        f.write("# Ablation Study\n\n")
        f.write(table_md)
        f.write("\n\n")

        f.write("## Per-Run Details\n\n")
        for run in runs:
            f.write(f"### {run['display_name']}\n")
            f.write(f"Source: `{run['path']}`\n\n")
            for metric in args.metrics:
                val = extract_metric(run["data"], metric)
                if val is not None:
                    name = METRIC_DISPLAY_NAMES.get(metric, metric)
                    f.write(f"- {name}: {format_metric(val, metric, True)}\n")
            f.write("\n")

    print(f"Written: {output}")
    print()
    print(table_md)


if __name__ == "__main__":
    main()
