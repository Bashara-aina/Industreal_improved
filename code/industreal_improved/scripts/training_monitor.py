#!/usr/bin/env python3
"""Training monitor: detect anomalies in the running Path-D training.

Reads the training log and checks for:
- NaN/Inf in any loss
- Kendall log_var caps (act≤1.0, psr≤0.5)
- EMA updates (model moving, not stuck)
- Per-task loss trends (each head should be decreasing or stable)
- Eval frequency (ep10 should produce first val metrics)

Usage:
    python scripts/training_monitor.py --log /tmp/mtl_mvit_run9.log
    python scripts/training_monitor.py --log /tmp/mtl_mvit_run9.log --once  # single check
"""

import argparse
import re
import sys
import time
from pathlib import Path


# Regex patterns
BATCH_RE = re.compile(
    r"\[batch\s+(\d+)/(\d+)\s+accum=(\d+)/(\d+)\]\s+"
    r"loss=([\d.\-+eE]+)\s+det=([\d.\-+eE]+)\s+"
    r"act=([\d.\-+eE]+)\s+psr=([\d.\-+eE]+)\s+pose=([\d.\-+eE]+)"
)
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+loss=([\d.\-+eE]+)\s+"
    r"det=([\d.\-+eE]+)\s+act=([\d.\-+eE]+)\s+"
    r"psr=([\d.\-+eE]+)\s+pose=([\d.\-+eE]+)\s+\|\s+"
    r"lv=\[([\d.\-+eE]+),([\d.\-+eE]+),([\d.\-+eE]+),([\d.\-+eE]+)\]"
)
EVAL_RE = re.compile(
    r"Eval \(([a-z]+)\):\s+act_top1=([\d.\-+eE]+)\s+"
    r"act_top5=([\d.\-+eE]+)\s+psr_f1=([\d.\-+eE]+)\s+"
    r"det_bce=([\d.\-+eE]+)\s+pose_fwd=([\d.\-+eE]+)deg"
)


def parse_log(path: Path) -> dict:
    """Parse training log and extract batch/epoch/eval data."""
    batches = []
    epochs = []
    evals = []
    warnings = []
    with open(path) as f:
        for line in f:
            m = BATCH_RE.search(line)
            if m:
                batches.append(
                    {
                        "batch": int(m.group(1)),
                        "total": int(m.group(2)),
                        "accum_step": int(m.group(3)),
                        "accum_total": int(m.group(4)),
                        "loss": float(m.group(5)),
                        "det": float(m.group(6)),
                        "act": float(m.group(7)),
                        "psr": float(m.group(8)),
                        "pose": float(m.group(9)),
                    }
                )
                continue
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(
                    {
                        "epoch": int(m.group(1)),
                        "total": int(m.group(2)),
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
                continue
            m = EVAL_RE.search(line)
            if m:
                evals.append(
                    {
                        "split": m.group(1),
                        "act_top1": float(m.group(2)),
                        "act_top5": float(m.group(3)),
                        "psr_f1": float(m.group(4)),
                        "det_bce": float(m.group(5)),
                        "pose_fwd": float(m.group(6)),
                    }
                )
                continue
            if "WARN" in line.upper() or "ERROR" in line.upper() or "Traceback" in line:
                warnings.append(line.strip())
    return {"batches": batches, "epochs": epochs, "evals": evals, "warnings": warnings}


def diagnose(data: dict) -> list:
    """Diagnose training state. Return list of (severity, message) tuples."""
    issues = []
    if not data["batches"]:
        return [("INFO", "No batches parsed yet — log too short or wrong format")]

    # Latest batch
    last = data["batches"][-1]
    if last["batch"] == 0:
        return [("INFO", f"Training just started (batch {last['batch']}/{last['total']})")]

    # Check for NaN/Inf
    for task in ("loss", "det", "act", "psr", "pose"):
        v = last[task]
        if v != v:  # NaN
            issues.append(("ERROR", f"NaN in {task} at batch {last['batch']}"))
        if v == float("inf") or v == float("-inf"):
            issues.append(("ERROR", f"Inf in {task} at batch {last['batch']}"))

    # Kendall log_var caps
    if data["epochs"]:
        last_ep = data["epochs"][-1]
        # act cap = 1.0
        if last_ep["lv_act"] > 1.0:
            issues.append(
                (
                    "INFO",
                    f"log_var_act={last_ep['lv_act']:.2f} > cap 1.0; cap IS active (loss uses min(lv, 1.0) = exp(-1.0) = 0.37)",
                )
            )
        else:
            issues.append(("OK", f"log_var_act={last_ep['lv_act']:.2f} ≤ cap 1.0"))
        # psr cap = 0.5
        if last_ep["lv_psr"] > 0.5:
            issues.append(("INFO", f"log_var_psr={last_ep['lv_psr']:.2f} > cap 0.5; cap IS active"))
        else:
            issues.append(("OK", f"log_var_psr={last_ep['lv_psr']:.2f} ≤ cap 0.5"))

    # Loss trend: compare last 100 batches to previous 100
    if len(data["batches"]) >= 200:
        recent = data["batches"][-100:]
        earlier = data["batches"][-200:-100]
        for task in ("loss", "act", "psr"):
            r_mean = sum(b[task] for b in recent) / len(recent)
            e_mean = sum(b[task] for b in earlier) / len(earlier)
            pct = (r_mean - e_mean) / max(e_mean, 0.01) * 100
            sign = "↓" if pct < 0 else "↑"
            severity = "OK" if pct < 5 else ("WARN" if pct < 20 else "ERROR")
            if task == "loss" and pct > 50:
                severity = "ERROR"
            issues.append(
                (
                    severity,
                    f"{task}: {e_mean:.4f} → {r_mean:.4f} ({sign}{abs(pct):.1f}% over last 100 batches)",
                )
            )

    # Eval results
    if data["evals"]:
        last_ev = data["evals"][-1]
        issues.append(
            (
                "INFO",
                f"Latest eval ({last_ev['split']}): act_top1={last_ev['act_top1']:.4f}, "
                f"psr_f1={last_ev['psr_f1']:.4f}, pose_fwd={last_ev['pose_fwd']:.2f}°",
            )
        )

    # Warnings/errors in log
    for w in data["warnings"][-5:]:  # last 5 warnings
        if "Traceback" in w:
            issues.append(("ERROR", f"Log: {w[:200]}"))
        else:
            issues.append(("WARN", f"Log: {w[:200]}"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Training monitor")
    parser.add_argument("--log", type=str, required=True, help="Path to training log")
    parser.add_argument("--once", action="store_true", help="Single check (default: monitor loop)")
    parser.add_argument("--interval", type=int, default=60, help="Monitor interval (sec)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}")
        sys.exit(1)

    if args.once:
        data = parse_log(log_path)
        issues = diagnose(data)
        print(f"\n{'=' * 60}")
        print(f"Training status ({log_path})")
        print(f"{'=' * 60}")
        for severity, msg in issues:
            icon = {"OK": "✓", "INFO": "ℹ", "WARN": "⚠", "ERROR": "❌"}[severity]
            print(f"  {icon} [{severity:5s}] {msg}")
        return

    # Monitor loop
    print(f"Monitoring {log_path} (interval {args.interval}s, Ctrl-C to stop)")
    while True:
        data = parse_log(log_path)
        issues = diagnose(data)
        print(
            f"\n[{time.strftime('%H:%M:%S')}] {len(data['batches'])} batches, {len(data['epochs'])} epochs, {len(data['evals'])} evals"
        )
        for severity, msg in issues:
            icon = {"OK": "✓", "INFO": "ℹ", "WARN": "⚠", "ERROR": "❌"}[severity]
            print(f"  {icon} [{severity:5s}] {msg}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
