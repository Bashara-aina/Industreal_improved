#!/usr/bin/env python3
"""Figure: Kendall log-var trajectories (AAIML Fig 6).

Parses [KENDALL step=N] lines from a training log and plots the
evolution of each task's log_var (lv) over training steps/epochs.
Overlays capped vs uncapped regions per the Kendall clamp bounds.

Usage:
    python scripts/figure_logvar_trajectories.py --log <training_log> [--output ...]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Regex for Kendall log lines
KENDALL_RE = re.compile(
    r"\[KENDALL step=(\d+)\]\s+"
    r"lv:\s+det=([\d.\-]+)\s+pose=([\d.\-]+)\s+act=([\d.\-]+)\s+psr=([\d.\-]+)\s+\|"
)

# Clamp bounds (matching train.py _clamp_kendall_log_vars)
CLAMP_BOUNDS = {
    "det": (-4.0, 2.0),
    "pose": (-4.0, 2.0),
    "act": (-4.0, 2.0),
    "psr": (-4.0, 2.0),
}

TASK_COLORS = {
    "det": "#4C72B0",
    "pose": "#DD8452",
    "act": "#55A868",
    "psr": "#C44E52",
}

TASK_LABELS = {
    "det": "Detection",
    "pose": "Head Pose",
    "act": "Activity",
    "psr": "PSR",
}

TASK_ORDER = ["det", "pose", "act", "psr"]


def parse_log(path: Path):
    """Parse [KENDALL] lines from training log.

    Returns list of dicts with keys: step, det, pose, act, psr.
    """
    records = []
    with open(path) as f:
        for line in f:
            m = KENDALL_RE.search(line)
            if m:
                records.append(
                    {
                        "step": int(m.group(1)),
                        "det": float(m.group(2)),
                        "pose": float(m.group(3)),
                        "act": float(m.group(4)),
                        "psr": float(m.group(5)),
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser(description="Plot Kendall log-var trajectories")
    parser.add_argument("--log", type=str, required=True, help="Path to training log")
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures/logvar_trajectories.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Steps per epoch (for x-axis in epochs). Auto-detected if not set.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}")
        sys.exit(1)

    records = parse_log(log_path)
    if not records:
        print("No [KENDALL] entries found in log.")
        sys.exit(1)

    print(f"Parsed {len(records)} [KENDALL] entries from {log_path}")

    # Auto-detect steps per epoch from step spacing
    if args.steps_per_epoch is None and len(records) > 1:
        steps = [r["step"] for r in records]
        diffs = [steps[i + 1] - steps[i] for i in range(min(10, len(steps) - 1))]
        typical_gap = max(1, int(np.median(diffs)))
        # Estimate: assume last step ~= total steps trained
        max_step = max(steps)
        # Common step counts per epoch: look for round numbers
        candidates = [100, 200, 250, 500, 1000]
        steps_per_epoch = None
        for c in candidates:
            # Check if max_step is roughly a multiple of c
            ratio = max_step / c
            if 1 <= ratio <= 200 and abs(ratio - round(ratio)) < 0.05:
                steps_per_epoch = c
                break
        if steps_per_epoch is None:
            # Fallback: use 1000 as default
            steps_per_epoch = 1000
            print(f"  (auto: steps_per_epoch={steps_per_epoch} by fallback)")
        else:
            print(f"  (auto: steps_per_epoch={steps_per_epoch})")
    else:
        steps_per_epoch = args.steps_per_epoch or 1000

    # Convert steps to epoch (1-indexed)
    steps = np.array([r["step"] for r in records])
    epochs = steps / steps_per_epoch + 1.0

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Plot each task
    for task in TASK_ORDER:
        values = np.array([r[task] for r in records])
        ax.plot(
            epochs,
            values,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
            linewidth=1.5,
            alpha=0.85,
        )

    # Shade clamp bound regions
    for task in TASK_ORDER:
        lo, hi = CLAMP_BOUNDS[task]
        color = TASK_COLORS[task]
        # Lower bound zone
        ax.axhspan(
            lo - 5,
            lo + 0.05,
            xmin=0,
            xmax=1,
            facecolor=color,
            alpha=0.06,
        )
        # Upper bound zone
        ax.axhspan(
            hi - 0.05,
            hi + 5,
            xmin=0,
            xmax=1,
            facecolor=color,
            alpha=0.06,
        )

    # Draw dashed lines for clamp bounds
    for task in TASK_ORDER:
        lo, hi = CLAMP_BOUNDS[task]
        color = TASK_COLORS[task]
        ax.axhline(y=lo, color=color, linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axhline(y=hi, color=color, linestyle=":", linewidth=0.7, alpha=0.5)

    # Add zero line for reference
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("log_var", fontsize=12)
    ax.set_title("Kendall log-var Trajectories (AAIML Fig 6)", fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)

    # Annotate clamp regions
    ax.text(
        0.02,
        0.04,
        "Shaded bands: clamp bounds [-4, 2]",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        va="bottom",
    )

    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
