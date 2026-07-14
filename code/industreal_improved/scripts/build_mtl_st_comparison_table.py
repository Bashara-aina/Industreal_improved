#!/usr/bin/env python3
"""
build_mtl_st_comparison_table.py — MTL vs ST comparison table for IndustReal paper.

Reads metrics from MTL and single-task (ST) baseline runs, computes per-task
statistics (mean, std, bootstrap CI), retention ratios, and a geometry-mean
composite score. Outputs both a human-readable markdown table and a LaTeX
booktabs table ready for IEEEtran insertion.

Usage:
    python scripts/build_mtl_st_comparison_table.py \
        --mtl-dir runs/aa_main_mtl \
        --st-dir runs/aa_st_baselines \
        --output-dir paper/tables

    # Dry-run with diagnostic output:
    python scripts/build_mtl_st_comparison_table.py \
        --mtl-dir runs/aa_main_mtl \
        --st-dir runs/aa_st_baselines \
        --output-dir paper/tables \
        --dry-run

Input expectations:
    MTL directory:  <mtl-dir>/metrics.json  (output of evaluate_all())
    ST directory:   <st-dir>/<task>/seed_<N>/metrics.json
        where task is one of: pose, act, det, psr
        and N is a seed number (e.g., 103, 104, 105)
    Each metrics.json should contain a flat dict with numeric keys like:
        det_mAP50, act_top1, psr_f1_at_t, forward_angular_MAE_deg
    Or nested under "test_metrics": { ... }

    If metrics are missing, the script fills cells with "TBD" and continues.

Outputs:
    <output-dir>/main_results.md     — Markdown table
    <output-dir>/main_results.tex    — LaTeX booktabs table
"""

import argparse
import json
import logging
import sys
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("build_mtl_st_comparison_table")

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
# (task_id, display_name, metric_display, metric_key_aliases, higher_is_better)
TASKS = [
    ("det", "Detection", "mAP@0.5", ["det_mAP50", "det_mAP50_pc"], True),
    ("act", "Activity", "clip top-1", ["act_top1", "act_accuracy", "act_seg_top1"], True),
    ("psr", "PSR", "F1@+-3", ["psr_f1_at_t", "psr_overall_f1", "psr_macro_f1", "psr_event_f1_at_3"], True),
    ("pose", "Head pose", "fwd MAE (deg)", ["forward_angular_MAE_deg", "pose_fwd_mae"], False),
]

LOWER_IS_BETTER_TASKS = {tid for tid, _, _, _, hib in TASKS if not hib}


def load_metrics(path: Path) -> dict:
    """Load a metrics.json file, returning a flat dict of metric key -> value.

    Handles both flat dicts and nested {"test_metrics": {...}} structures.
    Returns empty dict if file is missing or unparseable.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"  Cannot parse {path}: {e}")
        return {}

    if not isinstance(raw, dict):
        return {}

    # Flatten test_metrics into top-level
    flat = {}
    for k, v in raw.items():
        if k == "test_metrics" and isinstance(v, dict):
            flat.update(v)
        elif isinstance(v, (int, float)):
            flat[k] = v
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            # val_metrics list — take last entry
            last_entry = v[-1]
            for vk, vv in last_entry.items():
                if isinstance(vv, (int, float)):
                    flat[vk] = vv
    return flat


def extract_metric(metrics: dict, aliases: list[str]) -> Optional[float]:
    """Try metric key aliases in order; return first found value or None."""
    for key in aliases:
        val = metrics.get(key)
        if val is not None and isinstance(val, (int, float)) and not math.isnan(val):
            return float(val)
    return None


def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float, float, float]:
    """Compute mean, std, and bootstrap confidence interval for a list of values.

    Returns (mean, std, ci_low, ci_high).
    If fewer than 2 values, returns (mean_or_zero, 0, mean_or_zero, mean_or_zero).
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    if n == 1:
        return float(arr[0]), 0.0, float(arr[0]), float(arr[0])

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # sample std

    # Bootstrap
    rng = np.random.default_rng(seed=42)
    boot_means = np.array([
        float(np.mean(rng.choice(arr, size=n, replace=True)))
        for _ in range(n_resamples)
    ])

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1.0 - alpha))
    return mean, std, ci_low, ci_high


def compute_ratio(mtl_val: float, st_val: float, higher_is_better: bool) -> Optional[float]:
    """Compute MTL/ST retention ratio.

    For higher-is-better metrics: ratio = MTL / ST  (>1 means MTL improves)
    For lower-is-better metrics: ratio = ST / MTL  (>1 means MTL improves)

    Returns None if either value is missing, zero, or negative (for lower-better).
    """
    if mtl_val is None or st_val is None or st_val <= 0:
        return None
    if higher_is_better:
        return mtl_val / st_val
    else:
        # Lower is better: invert so >1 still means MTL is better
        if mtl_val <= 0:
            return None
        return st_val / mtl_val


def fmt_val(val: Optional[float], decimals: int = 2) -> str:
    """Format a metric value, or '--' for missing."""
    if val is None:
        return "--"
    if decimals == 0:
        return f"{val:.0f}"
    return f"{val:.{decimals}f}"


def fmt_ratio(val: Optional[float]) -> str:
    """Format a ratio value, or '--' for missing."""
    if val is None:
        return "--"
    return f"{val:.2f}"


def find_metrics_files(base_dir: Path, task: str, seed_glob: str = "seed_*") -> list[Path]:
    """Find all metrics.json files for a task in the ST directory structure.

    Searches patterns:
        <base_dir>/<task>/<seed_glob>/metrics.json
        <base_dir>/aa_st_<task>/<seed_glob>/metrics.json
        <base_dir>/st_<task>/<seed_glob>/metrics.json
        <base_dir>/<task>/metrics.json
        <base_dir>/aa_st_<task>/metrics.json
    """
    patterns = [
        base_dir / task / seed_glob / "metrics.json",
        base_dir / f"aa_st_{task}" / seed_glob / "metrics.json",
        base_dir / f"st_{task}" / seed_glob / "metrics.json",
        base_dir / f"aa_st_{task}" / "metrics.json",
        base_dir / task / "metrics.json",
    ]

    found = []
    seen = set()
    for pattern in patterns:
        parent_glob = str(pattern.parent)
        if parent_glob in seen:
            continue
        seen.add(parent_glob)
        matched = sorted(Path(base_dir).glob(str(pattern.relative_to(base_dir))))
        for m in matched:
            if m.exists() and m not in found:
                found.append(m)

    return found


def find_mtl_metrics(mtl_dir: Path) -> tuple[Optional[Path], list[Path]]:
    """Find MTL metrics.json, returning (primary_path, fallback_paths).

    Primary: <mtl_dir>/metrics.json
    Fallbacks: <mtl_dir>/<subdir>/metrics.json for any immediate subdirectories
    """
    primary = mtl_dir / "metrics.json"
    if primary.exists():
        return primary, []

    # Check immediate subdirectories
    fallbacks = sorted(mtl_dir.glob("*/metrics.json"))
    if fallbacks:
        return fallbacks[0], fallbacks

    return None, []


def load_st_metrics(
    st_dir: Path,
    n_seeds: int = 3,
    bootstrap_samples: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Load all ST baseline metrics and compute per-task statistics.

    Returns dict:
    {
        "det": { "mean": ..., "std": ..., "ci_low": ..., "ci_high": ..., "values": [...], "n_seeds": ... },
        "act": { ... },
        "psr": { ... },
        "pose": { ... },
    }
    Missing tasks have None for each field.
    """
    results = {}
    for task_id, display_name, metric_label, key_aliases, hib in TASKS:
        metric_files = find_metrics_files(st_dir, task_id)
        if not metric_files:
            logger.info(f"  ST [{task_id}]: No metrics files found in {st_dir}")
            results[task_id] = None
            continue

        values = []
        for mf in metric_files[:n_seeds]:  # respect n_seeds
            metrics = load_metrics(mf)
            val = extract_metric(metrics, key_aliases)
            if val is not None:
                values.append(val)
                logger.info(f"  ST [{task_id}]: {mf} -> {val:.4f}")
            else:
                logger.info(f"  ST [{task_id}]: {mf} -> no matching key (aliases={key_aliases})")

        if not values:
            logger.warning(f"  ST [{task_id}]: No valid metrics extracted from {len(metric_files)} file(s)")
            results[task_id] = None
            continue

        mean, std, ci_low, ci_high = bootstrap_ci(values, n_resamples=bootstrap_samples, ci=ci_level)
        results[task_id] = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "values": values,
            "n_seeds": len(values),
        }
        logger.info(f"  ST [{task_id}]: mean={mean:.4f} std={std:.4f} "
                     f"95%CI=[{ci_low:.4f}, {ci_high:.4f}] n={len(values)}")

    return results


def load_mtl_metrics(mtl_dir: Path) -> dict:
    """Load MTL metrics and return per-task dict.

    Returns dict like ST output:
    {
        "det": { "mean": ..., "std": ..., "ci_low": ..., "ci_high": ..., "values": [...] },
        "act": { ... },
        ...
    }
    For MTL (single run), std/CI reflect frame-level uncertainty if available,
    otherwise std=0 and CI spans the point estimate.
    """
    primary, fallbacks = find_mtl_metrics(mtl_dir)

    if primary is not None:
        logger.info(f"  MTL: Found metrics.json at {primary}")
        return _extract_mtl_metrics(primary)

    logger.info(f"  MTL: No primary metrics.json in {mtl_dir}")
    for fb in fallbacks:
        logger.info(f"  MTL: Trying fallback {fb}")
        result = _extract_mtl_metrics(fb)
        if any(v is not None for v in result.values()):
            return result

    logger.warning(f"  MTL: No metrics found in {mtl_dir}")
    return {tid: None for tid, _, _, _, _ in TASKS}


def _extract_mtl_metrics(path: Path) -> dict:
    """Extract per-task metrics from a single metrics.json."""
    raw_metrics = load_metrics(path)
    if not raw_metrics:
        return {tid: None for tid, _, _, _, _ in TASKS}

    results = {}
    for task_id, display_name, metric_label, key_aliases, hib in TASKS:
        val = extract_metric(raw_metrics, key_aliases)
        if val is not None:
            # Check for CI keys (e.g., pose_fwd_mae_ci95 in mtl_mvit_run)
            ci_low = raw_metrics.get(f"{key_aliases[0]}_ci95_low")
            ci_high = raw_metrics.get(f"{key_aliases[0]}_ci95_high")

            # Also check list-style CI:  {"pose_fwd_mae_ci95": [low, high]}
            ci_list = raw_metrics.get(f"{key_aliases[0]}_ci95")
            if ci_list is not None and isinstance(ci_list, (list, tuple)) and len(ci_list) == 2:
                ci_low = ci_low if ci_low is not None else ci_list[0]
                ci_high = ci_high if ci_high is not None else ci_list[1]

            results[task_id] = {
                "mean": val,
                "std": 0.0,  # single run
                "ci_low": ci_low if ci_low is not None else val,
                "ci_high": ci_high if ci_high is not None else val,
                "values": [val],
                "n_seeds": 1,
            }
            logger.info(f"  MTL [{task_id}]: {val:.4f}" +
                         (f" CI=[{ci_low:.4f},{ci_high:.4f}]" if ci_low is not None else ""))
        else:
            results[task_id] = None
            logger.info(f"  MTL [{task_id}]: No metric found (aliases={key_aliases})")

    return results


def find_mediapipe_baseline() -> Optional[dict]:
    """Load MediaPipe baseline results if available.

    Checks the expected output path from mediapipe_pose_baseline.py.
    Returns the parsed JSON dict or None.
    """
    candidates = [
        Path(__file__).resolve().parent.parent
        / "src" / "runs" / "rf_stages" / "checkpoints" / "efficiency_measured"
        / "mediapipe_baseline.json",
        Path(__file__).resolve().parent.parent
        / "runs" / "mediapipe_baseline.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def compute_geo_mean(ratios: dict[str, Optional[float]]) -> Optional[float]:
    """Compute geometry-mean of per-task MTL/ST retention ratios.

    Only includes tasks where ratio is available.
    For pose (lower-is-better), the ratio was already inverted by compute_ratio
    so all ratios >1 mean MTL improvement.
    """
    valid = [r for r in ratios.values() if r is not None and r > 0]
    if not valid:
        return None
    product = 1.0
    for r in valid:
        product *= r
    return product ** (1.0 / len(valid))


def build_table_data(
    mtl_metrics: dict,
    st_metrics: dict,
    mediapipe: Optional[dict],
) -> list[dict]:
    """Build table rows from MTL and ST metrics.

    Returns list of dicts, one per task, with keys:
        task_id, display_name, metric_label,
        st_mean, st_std, st_ci_str,
        mtl_mean, mtl_std, mtl_ci_str,
        ratio, higher_is_better
    """
    rows = []
    for task_id, display_name, metric_label, key_aliases, hib in TASKS:
        st = st_metrics.get(task_id)
        mtl = mtl_metrics.get(task_id)

        st_mean = st["mean"] if st else None
        st_std = st["std"] if st else None
        mtl_mean = mtl["mean"] if mtl else None
        mtl_std = mtl["std"] if mtl else None

        # Confidence interval strings
        st_ci_str = _ci_string(st)
        mtl_ci_str = _ci_string(mtl)

        # Ratio
        ratio = compute_ratio(mtl_mean, st_mean, hib)

        rows.append({
            "task_id": task_id,
            "display_name": display_name,
            "metric_label": metric_label,
            "st_mean": st_mean,
            "st_std": st_std,
            "st_ci_low": st["ci_low"] if st else None,
            "st_ci_high": st["ci_high"] if st else None,
            "st_ci_str": st_ci_str,
            "mtl_mean": mtl_mean,
            "mtl_std": mtl_std,
            "mtl_ci_low": mtl["ci_low"] if mtl else None,
            "mtl_ci_high": mtl["ci_high"] if mtl else None,
            "mtl_ci_str": mtl_ci_str,
            "ratio": ratio,
            "higher_is_better": hib,
        })

    return rows


def _ci_string(metrics: Optional[dict]) -> str:
    """Format confidence interval as string, or '--'."""
    if metrics is None:
        return "--"
    ci_low = metrics.get("ci_low")
    ci_high = metrics.get("ci_high")
    if ci_low is None or ci_high is None:
        return "--"
    return f"[{ci_low:.2f}, {ci_high:.2f}]"


def build_pose_coverage_row(mediapipe: Optional[dict]) -> Optional[dict]:
    """Build a summary row from MediaPipe baseline data.

    Returns dict with coverage_pct, forward_mae, up_mae or None.
    """
    if mediapipe is None:
        return None

    overall = mediapipe.get("overall", {})
    if not overall:
        return None

    return {
        "coverage_pct": overall.get("n_frames_processed", 0),
        "forward_mae": overall.get("forward_angular_mae_deg"),
        "up_mae": overall.get("up_angular_mae_deg"),
        "n_frames_total": overall.get("n_frames_processed", 0)
        + overall.get("n_frames_failed", 0),
    }


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------
def render_markdown(
    rows: list[dict],
    geo_mean: Optional[float],
    pose_coverage: Optional[dict],
    mtl_metrics: dict,
    st_metrics: dict,
) -> str:
    """Render the comparison table as markdown."""
    lines = [
        "# Main Results: Multi-Task vs Single-Task (auto-generated)",
        "",
        "| Task | Metric | ST (mean +/- std) | ST 95% CI | MTL | MTL 95% CI | MTL/ST |",
        "|------|--------|-------------------|-----------|-----|------------|--------|",
    ]

    for row in rows:
        st_str = _mean_std_str(row["st_mean"], row["st_std"], row["task_id"])
        mtl_str = _mean_std_str(row["mtl_mean"], row["mtl_std"], row["task_id"])
        ratio_str = fmt_ratio(row["ratio"])

        lines.append(
            f"| {row['display_name']} | {row['metric_label']} "
            f"| {st_str} | {row['st_ci_str']} "
            f"| {mtl_str} | {row['mtl_ci_str']} "
            f"| {ratio_str} |"
        )

    # Geometry-mean row
    lines.append(
        f"| **Geo-mean** | **composite** "
        f"| -- | -- | -- | -- | **{fmt_ratio(geo_mean)}** |"
    )

    # Pose coverage if available
    if pose_coverage is not None:
        lines.extend([
            "",
            "### Pose Coverage (MediaPipe baseline)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Forward angular MAE | {fmt_val(pose_coverage.get('forward_mae'), 2)} deg |",
            f"| Up angular MAE | {fmt_val(pose_coverage.get('up_mae'), 2)} deg |",
            f"| Coverage | {pose_coverage.get('coverage_pct', 0)} / {pose_coverage.get('n_frames_total', 0)} frames |",
        ])

    lines.extend([
        "",
        "### Notes",
        "- **MTL/ST ratio** > 1.0 = MTL beats single-task (positive transfer).",
        "- For pose (lower-is-better), ratio is inverted: ST/MTL, so >1 still means MTL improves.",
        "- **Geo-mean composite** is the geometric mean of per-task retention ratios.",
        "- Missing values (--) mean the corresponding run has not completed evaluation.",
        "- Bootstrap 95% CI computed from 1000 resamples (ST) or from frame-level metrics (MTL).",
    ])

    return "\n".join(lines)


def _mean_std_str(mean: Optional[float], std: Optional[float], task_id: str) -> str:
    """Format mean +/- std string, with task-appropriate decimal places."""
    if mean is None:
        return "--"
    if task_id == "pose":
        return f"{mean:.2f} +/- {std:.2f}" if std and std > 0 else f"{mean:.2f}"
    else:
        # Probabilities: 3 decimal places
        return f"{mean:.3f} +/- {std:.3f}" if std and std > 0 else f"{mean:.3f}"


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------
def render_latex(rows: list[dict], geo_mean: Optional[float], pose_coverage: Optional[dict]) -> str:
    """Render the comparison table as LaTeX booktabs, matching IEEEtran format."""
    lines = [
        "% Auto-generated by build_mtl_st_comparison_table.py",
        "% Do not edit manually — re-run the script to update.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Main results: multi-task vs.\ single-task across four heads. "
        r"ST baselines are single-head training from scratch (50~epochs, 3~seeds). "
        r"MTL is joint training with Kendall uncertainty weighting. "
        r"Bootstrap 95\%~CI from 1000 resamples. "
        r"MTL/ST ratio $>1$ indicates positive transfer.}",
        r"\label{tab:main_results}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccc}\toprule",
        r"\textbf{Task} & \textbf{Metric} & \textbf{ST} & \textbf{ST 95\%~CI} "
        r"& \textbf{MTL} & \textbf{MTL 95\%~CI} & \textbf{MTL/ST}\\",
        r"\midrule",
    ]

    for i, row in enumerate(rows):
        st_str = _latex_mean_std(row["st_mean"], row["st_std"], row["task_id"])
        mtl_str = _latex_mean_std(row["mtl_mean"], row["mtl_std"], row["task_id"])
        ratio_str = fmt_ratio(row["ratio"])
        name = row["display_name"]
        ml = row["metric_label"].replace("%", r"\%")

        lines.append(
            rf"{name} & {ml} & {st_str} & {row['st_ci_str']} "
            rf"& {mtl_str} & {row['mtl_ci_str']} & {ratio_str} \\"
        )

    # Geo-mean row
    lines.append(r"\midrule")
    lines.append(
        rf"\textbf{{Geo-mean}} & \textbf{{composite}} "
        rf"& -- & -- & -- & -- & \textbf{{{fmt_ratio(geo_mean)}}} \\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    # MediaPipe coverage table
    if pose_coverage is not None:
        lines.extend([
            "",
            r"\begin{table}[htbp]\centering\small",
            r"\caption{MediaPipe Face Mesh head pose baseline on the IndustReal "
            r"test split. Coverage indicates the fraction of frames where "
            r"MediaPipe successfully detected a face.}",
            r"\label{tab:mediapipe_baseline}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lcc}\toprule",
            r"\textbf{Metric} & \textbf{Value} \\",
            r"\midrule",
            rf"Forward angular MAE & {fmt_val(pose_coverage.get('forward_mae'), 2)} deg \\",
            rf"Up angular MAE & {fmt_val(pose_coverage.get('up_mae'), 2)} deg \\",
            rf"Coverage & {pose_coverage.get('coverage_pct', 0)} / {pose_coverage.get('n_frames_total', 0)} \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ])

    return "\n".join(lines)


def _latex_mean_std(mean: Optional[float], std: Optional[float], task_id: str) -> str:
    """Format mean +/- std for LaTeX, with task-appropriate decimal places."""
    if mean is None:
        return "--"
    if task_id == "pose":
        return rf"${mean:.2f}\pm{std:.2f}$" if std and std > 0 else f"${mean:.2f}$"
    else:
        return rf"${mean:.3f}\pm{std:.3f}$" if std and std > 0 else f"${mean:.3f}$"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build MTL vs ST comparison table for IndustReal paper."
    )
    parser.add_argument(
        "--mtl-dir",
        type=str,
        default="runs/aa_main_mtl",
        help="MTL run directory (default: runs/aa_main_mtl)",
    )
    parser.add_argument(
        "--st-dir",
        type=str,
        default="runs/aa_st_baselines",
        help="ST baseline root directory (default: runs/aa_st_baselines)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper/tables",
        help="Output directory for generated tables (default: paper/tables)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of ST seeds to use (default: 3)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CI (default: 0.95)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diagnostic info without writing output files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve paths relative to project root (script parent's parent)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    mtl_dir = project_root / args.mtl_dir
    st_dir = project_root / args.st_dir
    output_dir = project_root / args.output_dir

    logger.info("=" * 60)
    logger.info("MTL vs ST Comparison Table Builder")
    logger.info("=" * 60)
    logger.info(f"  Project root: {project_root}")
    logger.info(f"  MTL dir:      {mtl_dir}")
    logger.info(f"  ST dir:       {st_dir}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Seeds:        {args.seeds}")
    logger.info(f"  Bootstrap:    {args.bootstrap_samples} resamples @ {args.ci:.0%} CI")
    logger.info("")

    # Load metrics
    logger.info("Loading ST baselines...")
    st_metrics = load_st_metrics(
        st_dir,
        n_seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
        ci_level=args.ci,
    )

    logger.info("")
    logger.info("Loading MTL metrics...")
    mtl_metrics = load_mtl_metrics(mtl_dir)

    # MediaPipe baseline
    logger.info("")
    logger.info("Loading MediaPipe baseline...")
    mediapipe = find_mediapipe_baseline()
    if mediapipe:
        logger.info(f"  Found: {mediapipe.get('overall', {}).get('forward_angular_mae_deg', '?')} deg "
                     f"forward MAE")
    else:
        logger.info("  Not found (MediaPipe baseline not yet run)")

    # Build table data
    rows = build_table_data(mtl_metrics, st_metrics, mediapipe)

    # Compute per-task ratios for geo-mean
    ratios_dict = {r["task_id"]: r["ratio"] for r in rows}
    geo_mean = compute_geo_mean(ratios_dict)

    # Pose coverage
    pose_coverage = build_pose_coverage_row(mediapipe)

    # Render
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 60)

    for row in rows:
        st_str = _mean_std_str(row["st_mean"], row["st_std"], row["task_id"])
        mtl_str = _mean_std_str(row["mtl_mean"], row["mtl_std"], row["task_id"])
        logger.info(
            f"  {row['display_name']:12s} | ST: {st_str:>12s} "
            f"| MTL: {mtl_str:>12s} | MTL/ST: {fmt_ratio(row['ratio']):>6s}"
        )
    logger.info(f"  {'Geo-mean':12s} | {'':>12s} | {'':>12s} | Geo: {fmt_ratio(geo_mean):>6s}")

    if pose_coverage is not None:
        logger.info("")
        logger.info(f"  MediaPipe coverage: "
                     f"{pose_coverage['coverage_pct']}/{pose_coverage['n_frames_total']} frames")

    # Write outputs
    md_content = render_markdown(rows, geo_mean, pose_coverage, mtl_metrics, st_metrics)
    tex_content = render_latex(rows, geo_mean, pose_coverage)

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — skipping file writes")
        logger.info("")
        logger.info("Markdown output:")
        logger.info("-" * 40)
        print(md_content)
        logger.info("")
        logger.info("LaTeX output:")
        logger.info("-" * 40)
        print(tex_content)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / "main_results.md"
        with open(md_path, "w") as f:
            f.write(md_content)
        logger.info(f"  Markdown table written: {md_path}")

        tex_path = output_dir / "main_results.tex"
        with open(tex_path, "w") as f:
            f.write(tex_content)
        logger.info(f"  LaTeX table written:    {tex_path}")

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
