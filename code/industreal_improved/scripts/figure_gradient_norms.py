#!/usr/bin/env python3
"""Figure: per-head backbone gradient norm bar chart (AAIML Fig 5).

Parses [GRAD-NORM step=N] lines from a training log and produces a
bar chart of per-head gradient norms averaged over the final N
measurements.  Annotates each bar with the mean value.

Usage:
    python scripts/figure_gradient_norms.py --log <training_log> [--tail 20]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Regex for gradient norm log lines
GRAD_NORM_RE = re.compile(
    r"\[GRAD-NORM step=(\d+)\]\s+"
    r"backbone=([\d.eE+\-]+)\s+"
    r"det=([\d.eE+\-]+)\s+"
    r"hp=([\d.eE+\-]+)\s+"
    r"act=([\d.eE+\-]+)\s+"
    r"psr=([\d.eE+\-]+)"
)

HEAD_LABELS = ["detection", "head_pose", "activity", "psr"]
HEAD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
HEAD_KEYS = ["det", "hp", "act", "psr"]


def parse_log(path: Path):
    """Parse [GRAD-NORM] lines from training log.

    Returns list of dicts with keys: step, backbone, det, hp, act, psr.
    """
    records = []
    with open(path) as f:
        for line in f:
            m = GRAD_NORM_RE.search(line)
            if m:
                records.append(
                    {
                        "step": int(m.group(1)),
                        "backbone": float(m.group(2)),
                        "det": float(m.group(3)),
                        "hp": float(m.group(4)),
                        "act": float(m.group(5)),
                        "psr": float(m.group(6)),
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser(description="Plot per-head gradient norm bar chart")
    parser.add_argument("--log", type=str, required=True, help="Path to training log")
    parser.add_argument(
        "--tail",
        type=int,
        default=20,
        help="Number of most recent measurements to average (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures/gradient_norm_imbalance.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}")
        sys.exit(1)

    records = parse_log(log_path)
    if not records:
        print("No [GRAD-NORM] entries found in log.")
        sys.exit(1)

    # Average over the last N measurements
    tail = records[-args.tail :]

    means = {}
    stds = {}
    for key in HEAD_KEYS:
        vals = [r[key] for r in tail]
        means[key] = float(np.mean(vals))
        stds[key] = float(np.std(vals))

    backbone_mean = float(np.mean([r["backbone"] for r in tail]))

    print(f"Parsed {len(records)} [GRAD-NORM] entries from {log_path}")
    print(f"Averaging last {len(tail)} measurements:")
    print(f"  backbone: {backbone_mean:.4f}")
    for key, label in zip(HEAD_KEYS, HEAD_LABELS):
        print(f"  {label}: {means[key]:.4f} +/- {stds[key]:.4f}")

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(HEAD_LABELS))
    bar_vals = [means[k] for k in HEAD_KEYS]
    bar_errs = [stds[k] for k in HEAD_KEYS]

    bars = ax.bar(
        x_pos,
        bar_vals,
        yerr=bar_errs,
        color=HEAD_COLORS,
        edgecolor="black",
        capsize=5,
        width=0.6,
    )

    # Annotate each bar with its value
    for bar, val in zip(bars, bar_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(bar_errs) * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Reference line for backbone mean
    ax.axhline(
        y=backbone_mean,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"backbone mean = {backbone_mean:.2f}",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(HEAD_LABELS, fontsize=12)
    ax.set_ylabel("Gradient Norm", fontsize=12)
    ax.set_title("Per-Head Backbone Gradient Norm (Final Measurements)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add imbalance annotation
    max_head = max(bar_vals)
    min_head = min(bar_vals)
    ratio = max_head / max(min_head, 1e-12)
    ax.text(
        0.98,
        0.95,
        f"max/min ratio: {ratio:.1f}x",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
