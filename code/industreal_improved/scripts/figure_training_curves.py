#!/usr/bin/env python3
"""Figure: per-task metric vs. epoch training curves (AAIML Fig 7).

Parses metrics.jsonl files across multiple run directories and produces
a 4-panel figure with detection (mAP50), activity (macro F1),
head pose (MAE, inverted so up=better), and PSR (macro F1) over epochs.

Pathology markers (NaN/divergent epochs) are highlighted.

Usage:
    python scripts/figure_training_curves.py \\
        --runs runs/run1 runs/run2 \\
        [--labels "Run A" "Run B"] \\
        [--output paper/figures/training_curves.png]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Metric key mapping: (panel_label, jsonl_key, invert, unit)
PANELS = [
    ("Detection (mAP@50)", "det_mAP50", False, ""),
    ("Activity (Macro F1)", "act_macro_f1", False, ""),
    ("Head Pose (MAE, inv.)", "head_pose_MAE", True, "deg"),
    ("PSR (Macro F1)", "psr_macro_f1", False, ""),
]

# Pathology detection: flag if metric is NaN or outside reasonable bounds
REASONABLE_RANGES = {
    "det_mAP50": (0.0, 1.0),
    "act_macro_f1": (0.0, 1.0),
    "head_pose_MAE": (0.0, 180.0),
    "psr_macro_f1": (0.0, 1.0),
}


def parse_metrics_jsonl(path: Path):
    """Parse a metrics.jsonl file, return list of dicts per epoch."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def is_pathological(value, key):
    """Check if a metric value is pathological (NaN or out of range)."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return True
    lo, hi = REASONABLE_RANGES.get(key, (-1e9, 1e9))
    return not (lo <= value <= hi)


def main():
    parser = argparse.ArgumentParser(description="Plot per-task training curves")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to run directories containing metrics.jsonl",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Legend labels for each run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures/training_curves.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving average window for smoothing (default: 5, 0=off)",
    )
    args = parser.parse_args()

    # Resolve paths
    run_paths = []
    for r in args.runs:
        p = Path(r)
        if p.is_dir():
            p = p / "metrics.jsonl"
        run_paths.append(p)

    for p in run_paths:
        if not p.exists():
            print(f"ERROR: metrics.jsonl not found: {p}")
            sys.exit(1)

    labels = args.labels if args.labels else [str(p.parent) for p in run_paths]
    if len(labels) != len(run_paths):
        print("ERROR: --labels count must match --runs count")
        sys.exit(1)

    # Parse all runs
    all_records = []
    for path in run_paths:
        records = parse_metrics_jsonl(path)
        all_records.append(records)
        print(f"  {path}: {len(records)} epochs")

    # --- Build 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    colors = plt.cm.Set1(np.linspace(0, 1, len(run_paths)))

    for panel_idx, (panel_label, metric_key, invert, unit) in enumerate(PANELS):
        ax = axes_flat[panel_idx]

        for run_idx, records in enumerate(all_records):
            epochs = np.arange(1, len(records) + 1, dtype=float)
            values = np.array([r.get(metric_key, float("nan")) for r in records], dtype=float)

            # Replace None with NaN
            values = np.array([float("nan") if v is None else v for v in values])

            # Invert if needed (lower is better -> negate for display)
            plot_values = -values if invert else values

            # Moving average
            window = args.window
            if window > 1 and len(values) > window:
                smoothed = np.convolve(plot_values, np.ones(window) / window, mode="valid")
                # Align x-axis: smoothed starts at index window-1
                smooth_epochs = epochs[window - 1 :]
                ax.plot(
                    smooth_epochs,
                    smoothed,
                    color=colors[run_idx],
                    label=f"{labels[run_idx]} (smoothed)",
                    linewidth=1.5,
                    alpha=0.85,
                )
            else:
                ax.plot(
                    epochs,
                    plot_values,
                    color=colors[run_idx],
                    label=labels[run_idx],
                    linewidth=1.0,
                    alpha=0.7,
                )

            # Mark pathological epochs
            patho_epochs = []
            patho_vals = []
            for i, rec in enumerate(records):
                val = rec.get(metric_key)
                if val is not None and is_pathological(val, metric_key):
                    patho_epochs.append(i + 1)
                    pv = -val if invert else val
                    patho_vals.append(pv)

            if patho_epochs:
                ax.scatter(
                    patho_epochs,
                    patho_vals,
                    color=colors[run_idx],
                    marker="x",
                    s=40,
                    zorder=5,
                    label=f"{labels[run_idx]} (pathological)" if run_idx == 0 else None,
                )

        ax.set_xlabel("Epoch", fontsize=11)
        ylabel = panel_label
        if unit:
            ylabel += f" ({unit})"
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(panel_label, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Add shared pathology legend entry
    fig.suptitle("Training Curves (AAIML Fig 7)", fontsize=14, y=1.01)

    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.close(fig)

    # Also save individual panels for paper sub-figures
    stem = out_path.stem
    suffix = out_path.suffix
    for panel_idx, (panel_label, metric_key, invert, unit) in enumerate(PANELS):
        panel_fig, panel_ax = plt.subplots(figsize=(6, 4.5))
        for run_idx, records in enumerate(all_records):
            epochs = np.arange(1, len(records) + 1, dtype=float)
            values = np.array([r.get(metric_key, float("nan")) for r in records], dtype=float)
            values = np.array([float("nan") if v is None else v for v in values])
            plot_values = -values if invert else values

            window = args.window
            if window > 1 and len(values) > window:
                smoothed = np.convolve(plot_values, np.ones(window) / window, mode="valid")
                smooth_epochs = epochs[window - 1 :]
                panel_ax.plot(
                    smooth_epochs,
                    smoothed,
                    color=colors[run_idx],
                    label=labels[run_idx],
                    linewidth=1.5,
                )
            else:
                panel_ax.plot(
                    epochs,
                    plot_values,
                    color=colors[run_idx],
                    label=labels[run_idx],
                    linewidth=1.0,
                )

            # Pathological markers
            for i, rec in enumerate(records):
                val = rec.get(metric_key)
                if val is not None and is_pathological(val, metric_key):
                    pv = -val if invert else val
                    panel_ax.scatter(
                        i + 1,
                        pv,
                        color=colors[run_idx],
                        marker="x",
                        s=60,
                        zorder=5,
                    )

        panel_ax.set_xlabel("Epoch", fontsize=11)
        panel_ax.set_ylabel(ylabel, fontsize=11)
        panel_ax.set_title(panel_label, fontsize=12)
        panel_ax.grid(alpha=0.3)
        panel_ax.legend(fontsize=9)
        plt.tight_layout()

        safe_name = metric_key.replace("_", "-")
        panel_out = out_path.parent / f"{stem}_{safe_name}{suffix}"
        panel_fig.savefig(panel_out, dpi=150)
        print(f"  Panel saved: {panel_out}")
        plt.close(panel_fig)


if __name__ == "__main__":
    main()
