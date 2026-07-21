#!/usr/bin/env python3
"""Plot training curves from the log.

Reads a training log and produces an ASCII plot of per-task loss over batches.
Also produces a CSV that can be imported into any plotting tool.

Usage:
    python scripts/plot_training_curves.py --log /tmp/mtl_mvit_run9.log
    python scripts/plot_training_curves.py --log /tmp/mtl_mvit_run9.log --csv curves.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path


BATCH_RE = re.compile(
    r"\[batch\s+(\d+)/\d+\s+accum=\d+/\d+\]\s+"
    r"loss=([\d.\-+eE]+)\s+det=([\d.\-+eE]+)\s+"
    r"act=([\d.\-+eE]+)\s+psr=([\d.\-+eE]+)\s+pose=([\d.\-+eE]+)"
)
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+loss=([\d.\-+eE]+)\s+"
    r"det=([\d.\-+eE]+)\s+act=([\d.\-+eE]+)\s+"
    r"psr=([\d.\-+eE]+)\s+pose=([\d.\-+eE]+)\s+\|\s+"
    r"lv=\[([\d.\-+eE]+),([\d.\-+eE]+),([\d.\-+eE]+),([\d.\-+eE]+)\]"
)


def parse_log(path: Path):
    batches = []
    epochs = []
    with open(path) as f:
        for line in f:
            m = BATCH_RE.search(line)
            if m:
                batches.append(
                    {
                        "batch": int(m.group(1)),
                        "loss": float(m.group(2)),
                        "det": float(m.group(3)),
                        "act": float(m.group(4)),
                        "psr": float(m.group(5)),
                        "pose": float(m.group(6)),
                    }
                )
                continue
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(
                    {
                        "epoch": int(m.group(1)),
                        "loss": float(m.group(3)),
                        "det": float(m.group(4)),
                        "act": float(m.group(5)),
                        "psr": float(m.group(6)),
                        "pose": float(m.group(7)),
                        "lv_det": float(m.group(8)),
                        "lv_act": float(m.group(9)),
                        "lv_psr": float(m.group(10)),
                        "lv_pose": float(m.group(11)),
                    }
                )
    return batches, epochs


def moving_average(values, window=50):
    if len(values) < window:
        return values
    return [
        sum(values[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for i in range(len(values))
    ]


def ascii_sparkline(values, width=60, height=10):
    """Render an ASCII sparkline of values."""
    if not values:
        return ""
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return " " * width
    # Downsample to width
    n = len(values)
    if n > width:
        step = n / width
        downsampled = [values[int(i * step)] for i in range(width)]
    else:
        downsampled = values + [values[-1]] * (width - n)
    # Build rows
    rows = []
    for h in range(height, 0, -1):
        threshold = v_min + (v_max - v_min) * h / height
        row = ""
        for v in downsampled:
            row += "█" if v >= threshold else " "
        rows.append(row)
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--log", type=str, required=True, help="Training log")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--window", type=int, default=100, help="Moving average window")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}")
        sys.exit(1)

    batches, epochs = parse_log(log_path)
    if not batches:
        print("No batch data found in log")
        sys.exit(1)

    print(f"Parsed {len(batches)} batches and {len(epochs)} epochs from {log_path}")
    print()

    # CSV output
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["batch", "loss", "det", "act", "psr", "pose"])
            for b in batches:
                w.writerow([b["batch"], b["loss"], b["det"], b["act"], b["psr"], b["pose"]])
        print(f"CSV saved: {args.csv}")

    # Per-task moving averages
    print("=" * 80)
    print("TRAINING CURVES (smoothed, last 100 batches each task)")
    print("=" * 80)
    print()

    for task in ("loss", "det", "act", "psr", "pose"):
        raw = [b[task] for b in batches]
        smoothed = moving_average(raw, args.window)
        print(f"  {task:5s} (range: {min(raw):.4f} — {max(raw):.4f}, current: {raw[-1]:.4f})")
        print(ascii_sparkline(smoothed, width=70, height=8))
        print()

    # Per-epoch summary
    if epochs:
        print("=" * 80)
        print("PER-EPOCH SUMMARY (with log_var caps)")
        print("=" * 80)
        print(
            f"  {'Epoch':>5s}  {'loss':>8s}  {'det':>8s}  {'act':>8s}  {'psr':>8s}  {'pose':>8s}  | "
            f"{'lv_det':>7s}  {'lv_act':>7s}  {'lv_psr':>7s}  {'lv_pose':>7s}"
        )
        for ep in epochs:
            print(
                f"  {ep['epoch']:5d}  {ep['loss']:8.4f}  {ep['det']:8.4f}  {ep['act']:8.4f}  "
                f"{ep['psr']:8.4f}  {ep['pose']:8.4f}  | "
                f"{ep['lv_det']:7.2f}  {ep['lv_act']:7.2f}  {ep['lv_psr']:7.2f}  {ep['lv_pose']:7.2f}"
            )


if __name__ == "__main__":
    main()
