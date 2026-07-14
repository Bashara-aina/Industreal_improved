#!/usr/bin/env python3
"""Compare two checkpoints: parameter diff, EMA diff, and metric diff.

[OPUS 192 utility] Compare best.pt vs latest.pt to see how much the model
moved during training. Also useful for comparing pre-Opus-192 to
post-Opus-192 checkpoints.

Usage:
    python scripts/compare_checkpoints.py --a best.pt --b latest.pt
"""

import argparse
import sys
from pathlib import Path

import torch


def load_state(ckpt_path: Path) -> dict:
    """Load a checkpoint, return (state_dict, metrics, ema_state, log_vars)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    val_metrics = ckpt.get("val_metrics", [])
    if isinstance(val_metrics, list) and val_metrics:
        metrics = val_metrics[-1]
    elif isinstance(val_metrics, dict):
        metrics = val_metrics
    else:
        metrics = {}
    return {
        "state_dict": ckpt.get("model_state_dict", {}),
        "ema_state": ckpt.get("ema_model_state", {}),
        "metrics": metrics,
        "log_vars": ckpt.get("log_vars", {}),
        "epoch": ckpt.get("epoch", "?"),
    }


def compare_states(sd_a: dict, sd_b: dict, name_a: str, name_b: str) -> dict:
    """Compare two state_dicts: per-key L2 distance, % identical keys, % changed keys."""
    common = set(sd_a.keys()) & set(sd_b.keys())
    only_a = set(sd_a.keys()) - set(sd_b.keys())
    only_b = set(sd_b.keys()) - set(sd_a.keys())

    diffs = {}
    for k in common:
        if sd_a[k].shape != sd_b[k].shape:
            diffs[k] = {"shape_mismatch": True}
            continue
        a, b = sd_a[k].float(), sd_b[k].float()
        l2 = (a - b).norm().item()
        rel_l2 = l2 / max(a.norm().item(), 1e-9)
        cos = float(
            torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
            ).item()
        )
        diffs[k] = {"l2": l2, "rel_l2": rel_l2, "cosine": cos}

    n_changed = sum(1 for d in diffs.values() if d.get("rel_l2", 0) > 0.01)
    n_unchanged = len(common) - n_changed
    n_identical = sum(1 for d in diffs.values() if d.get("rel_l2", 0) < 1e-6)

    return {
        "common": len(common),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "n_changed": n_changed,
        "n_unchanged": n_unchanged,
        "n_identical": n_identical,
        "per_key": diffs,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two checkpoints")
    parser.add_argument("--a", type=str, required=True, help="First checkpoint")
    parser.add_argument("--b", type=str, required=True, help="Second checkpoint")
    parser.add_argument(
        "--top-keys", type=int, default=10, help="Show top N keys with largest L2 distance"
    )
    args = parser.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    if not a_path.exists() or not b_path.exists():
        print(f"ERROR: checkpoint(s) not found: {a_path}, {b_path}")
        sys.exit(1)

    print(f"Loading A: {a_path}")
    a = load_state(a_path)
    print(f"Loading B: {b_path}")
    b = load_state(b_path)

    print()
    print("=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)
    print(f"  A epoch: {a['epoch']}, B epoch: {b['epoch']}")
    print()
    print(f"  Common keys:  {len(set(a['state_dict'].keys()) & set(b['state_dict'].keys()))}")
    print(f"  Only in A:    {len(set(a['state_dict'].keys()) - set(b['state_dict'].keys()))}")
    print(f"  Only in B:    {len(set(b['state_dict'].keys()) - set(a['state_dict'].keys()))}")
    print()

    # Raw state_dict
    raw_cmp = compare_states(a["state_dict"], b["state_dict"], "A", "B")
    print(f"  Raw model:")
    print(f"    Changed (rel_l2 > 0.01): {raw_cmp['n_changed']}/{raw_cmp['common']}")
    print(f"    Unchanged:               {raw_cmp['n_unchanged']}/{raw_cmp['common']}")
    print(f"    Identical:               {raw_cmp['n_identical']}/{raw_cmp['common']}")

    # EMA
    ema_cmp = compare_states(a["ema_state"], b["ema_state"], "A ema", "B ema")
    print(f"  EMA model:")
    print(
        f"    Common: {ema_cmp['common']}, Only A: {ema_cmp['only_a']}, Only B: {ema_cmp['only_b']}"
    )
    print(
        f"    Changed: {ema_cmp['n_changed']}/{ema_cmp['common']}, Identical: {ema_cmp['n_identical']}"
    )

    # Log vars
    print()
    print("  Kendall log_vars (per task):")
    for task in ("det", "act", "psr", "pose"):
        if task in a["log_vars"] and task in b["log_vars"]:
            a_lv = float(a["log_vars"][task])
            b_lv = float(b["log_vars"][task])
            print(f"    {task}: A={a_lv:+.3f}, B={b_lv:+.3f}, delta={b_lv - a_lv:+.3f}")

    # Top changed keys
    print()
    print(f"  Top {args.top_keys} keys with largest L2 distance:")
    sorted_keys = sorted(
        [(k, d.get("l2", 0)) for k, d in raw_cmp["per_key"].items() if "l2" in d],
        key=lambda x: -x[1],
    )[: args.top_keys]
    for k, l2 in sorted_keys:
        d = raw_cmp["per_key"][k]
        if "cosine" in d:
            print(f"    {k:60s} L2={d['l2']:.4e}  rel_l2={d['rel_l2']:.4f}  cos={d['cosine']:+.4f}")

    # Metrics
    print()
    print("  Last val metrics:")
    if a["metrics"] and b["metrics"]:
        for key in sorted(set(a["metrics"].keys()) | set(b["metrics"].keys())):
            a_v = a["metrics"].get(key)
            b_v = b["metrics"].get(key)
            if a_v is not None or b_v is not None:
                a_str = f"{a_v:.4f}" if a_v is not None else "-"
                b_str = f"{b_v:.4f}" if b_v is not None else "-"
                print(f"    {key:20s}: A={a_str:>8s}  B={b_str:>8s}")


if __name__ == "__main__":
    main()
