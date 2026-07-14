#!/usr/bin/env python3
"""Generate the paper's headline table from existing checkpoint metrics.

[OPUS 192 §5 step 10] After all runs complete, this script reads the
metrics.json files from each run and produces the headline table:

  | Head   | ST (Phase 2) | MTL (Phase 3) | MTL/ST ratio | SOTA | MTL/SOTA |

Usage:
    python scripts/generate_paper_table.py \
        --mtl-runs runs/mtl_mvit_run \
        --st-runs runs/st_det runs/st_act runs/st_psr \
        --output paper_table.md
"""

import argparse
import json
from pathlib import Path


def load_metrics(run_dir: Path) -> dict:
    """Load metrics.json from a run directory."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        return json.load(f)


def extract_metric(metrics: dict, key: str, default: float = 0.0) -> float:
    """Extract a metric from a metrics dict, checking multiple locations."""
    if not metrics:
        return default
    # Check val_metrics (list of dicts) — take last
    val = metrics.get("val_metrics", [])
    if isinstance(val, list) and val:
        for k in [key, f"val_{key}"]:
            if k in val[-1]:
                return val[-1][k]
    # Check test_metrics (dict)
    test = metrics.get("test_metrics", {})
    if isinstance(test, dict):
        for k in [key, f"test_{key}"]:
            if k in test:
                return test[k]
    # Check top-level
    if key in metrics:
        return metrics[key]
    return default


def fmt_metric(value: float, format: str = ".3f") -> str:
    """Format a metric value, with - for missing."""
    if value is None:
        return "-"
    return f"{value:{format}}"


def main():
    parser = argparse.ArgumentParser(description="Generate paper headline table from metrics")
    parser.add_argument(
        "--mtl-runs",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to MTL run directories (with metrics.json)",
    )
    parser.add_argument(
        "--st-runs",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to ST run directories (with metrics.json)",
    )
    parser.add_argument(
        "--sota", type=str, default=None, help="Path to SOTA numbers JSON (WACV paper)"
    )
    parser.add_argument("--output", type=str, default="paper_table.md")
    args = parser.parse_args()

    # SOTA numbers (from WACV paper Tables 2-3, verified by Opus 186)
    # Updated per Opus 192 FC-6: detection should compare to IndustReal-only (0.779 boxed)
    # not synthetic-augmented (0.838).
    SOTA = {
        "det_mAP50": 0.779,  # WACV: YOLOv8-m on IndustReal-only (boxed)
        "act_top1": 0.6525,  # WACV: MViTv2-S on AR (Kinetics pretrained, RGB)
        "psr_event_f1_at_3": 0.901,  # WACV: STORM
        "pose_fwd_mae": None,  # No SOTA (per Opus 186)
    }
    # 80% bar (low bar; the user-requested target)
    SOTA_80PCT = {k: (0.8 * v if v else None) for k, v in SOTA.items()}

    # Load MTL metrics
    mtl_metrics_list = [load_metrics(Path(d)) for d in args.mtl_runs]
    # Use the LAST run's metrics (most recent)
    mtl = mtl_metrics_list[-1] if mtl_metrics_list else {}
    # Use test_metrics_ema if available, else test_metrics
    mtl_test_ema = mtl.get("test_metrics_ema")
    mtl_test_raw = mtl.get("test_metrics")
    mtl_val = mtl.get("val_metrics", [])
    mtl_val_last = mtl_val[-1] if mtl_val else {}

    # Load ST metrics
    st_metrics = {}
    for run_dir in args.st_runs:
        path = Path(run_dir)
        if not path.exists():
            continue
        # Try to infer task from directory name
        task = path.name.replace("st_", "").replace("runs_", "")
        st_metrics[task] = load_metrics(path)

    # Build table
    print("=" * 80)
    print("PAPER HEADLINE TABLE")
    print("=" * 80)
    print()

    md = ["# Headline Table (auto-generated)", ""]
    md.append(
        "| Head | Metric | ST (Phase 2) | MTL (Phase 3) | MTL/ST | SOTA | MTL/SOTA | 80% bar |"
    )
    md.append(
        "|------|--------|---------------|---------------|--------|------|----------|---------|"
    )

    rows = [
        ("Detection", "mAP@0.5", "det_mAP50"),
        ("Activity", "top-1", "act_top1"),
        ("PSR", "event F1@±3", "psr_event_f1_at_3"),
        ("Pose", "fwd MAE (deg)", "pose_fwd_mae"),
    ]

    for head_name, metric_label, key in rows:
        # ST value
        st_val = None
        for task, m in st_metrics.items():
            if task in head_name.lower() or head_name.lower() in task:
                st_val = extract_metric(m, key)
                break
        # MTL value: prefer test_ema, then test, then val
        mtl_val = None
        if mtl_test_ema and key in mtl_test_ema:
            mtl_val = mtl_test_ema[key]
        elif mtl_test_raw and key in mtl_test_raw:
            mtl_val = mtl_test_raw[key]
        elif key in mtl_val_last:
            mtl_val = mtl_val_last[key]

        # MTL/ST ratio (only for higher-is-better metrics)
        sota = SOTA.get(key)
        sota_80 = SOTA_80PCT.get(key)
        ratio = ""
        if st_val and mtl_val and st_val > 0:
            if key == "pose_fwd_mae":
                # Lower is better; ratio inverted
                ratio = f"{st_val / mtl_val:.2f}" if mtl_val > 0 else "-"
            else:
                ratio = f"{mtl_val / st_val:.2f}"
        mtl_sota = ""
        if mtl_val and sota:
            if key == "pose_fwd_mae":
                mtl_sota = f"{sota / mtl_val:.2f}" if mtl_val > 0 else "-"
            else:
                mtl_sota = f"{mtl_val / sota:.2f}"

        # Format values
        if key == "pose_fwd_mae":
            st_str = fmt_metric(st_val, ".2f") if st_val else "-"
            mtl_str = fmt_metric(mtl_val, ".2f") if mtl_val else "-"
            sota_str = "-"
            sota_80_str = "-"
        else:
            st_str = fmt_metric(st_val, ".3f") if st_val else "-"
            mtl_str = fmt_metric(mtl_val, ".3f") if mtl_val else "-"
            sota_str = fmt_metric(sota, ".3f") if sota else "-"
            sota_80_str = fmt_metric(sota_80, ".3f") if sota_80 else "-"

        md.append(
            f"| {head_name} | {metric_label} | {st_str} | {mtl_str} | {ratio} | {sota_str} | {mtl_sota} | {sota_80_str} |"
        )
        print(
            f"  {head_name:12s} {metric_label:18s} | ST: {st_str:>8s} | MTL: {mtl_str:>8s} | "
            f"SOTA: {sota_str:>8s} | 80% bar: {sota_80_str:>8s}"
        )

    print()
    md.append("")
    md.append("## Notes")
    md.append("- **MTL/ST ratio** > 1.0 = MTL beats single-task (positive transfer).")
    md.append("- **MTL/SOTA** > 0.8 = MTL clears 80% of SOTA.")
    md.append("- **Pose**: no SOTA exists; reported as MAE (lower is better).")
    md.append(
        "- **Detection SOTA**: 0.779 is IndustReal-only (boxed); 0.838 is with synthetic-data augmentation. We compare to 0.779 per Opus 192 FC-6."
    )
    md.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(md))
    print(f"\nTable saved: {args.output}")
    print()
    print("Verdict (Opus 192 §3 Q-Strategy):")
    print("  MTL/ST ≥ 0.9 across all heads → 'MTL is helping'")
    print("  MTL/SOTA ≥ 0.8 on 3/4 heads → publishable L2+L3+method paper")


if __name__ == "__main__":
    main()
