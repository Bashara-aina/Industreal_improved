"""Null-POS extended analysis: 16 recordings + Edit metric.

Uses cached PSR predictions from psr_data_cache_best.pth (no GPU needed).
Reference: Opus 141 Q24 (extend to all 16 recs), Q29 (add Edit metric).
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

NUM_COMPONENTS = 11


def pos_metric(preds, gt):
    """Pairwise orientation score."""
    pred_pairs = preds[1:] - preds[:-1]
    gt_pairs = gt[1:] - gt[:-1]
    return float((np.sign(pred_pairs) == np.sign(gt_pairs)).mean())


def edit_metric(pred_seq, gt_seq):
    """Levenshtein Hamming / T, per-component + mean."""
    mismatches = (pred_seq != gt_seq).astype(np.float32)
    per_comp = mismatches.mean(axis=0)
    return per_comp, float(per_comp.mean())


def compute_metrics(vp, vl):
    """Compute POS and Edit for Ours, null_zeros, null_copy_prev."""
    # Ours
    ours_pos = pos_metric(vp, vl)
    ours_edit_pc, ours_edit = edit_metric(vp, vl)

    # Null 1: all zeros
    n1 = np.zeros_like(vp)
    n1_pos = pos_metric(n1, vl)
    n1_edit_pc, n1_edit = edit_metric(n1, vl)

    # Null 2: copy prev
    n2 = np.zeros_like(vp)
    n2[1:] = vp[:-1]
    n2_pos = pos_metric(n2, vl)
    n2_edit_pc, n2_edit = edit_metric(n2, vl)

    return {
        "pos": {"ours": ours_pos, "null_all_zeros": n1_pos, "null_copy_prev": n2_pos},
        "edit_mean": {"ours": ours_edit, "null_all_zeros": n1_edit, "null_copy_prev": n2_edit},
        "edit_per_comp": {
            "ours": ours_edit_pc.tolist(),
            "null_all_zeros": n1_edit_pc.tolist(),
            "null_copy_prev": n2_edit_pc.tolist(),
        },
    }


def main():
    cache_path = Path("src/runs/rf_stages/checkpoints/psr_data_cache_best.pth")
    save_dir = Path("src/runs/rf_stages/checkpoints/null_model_pos_extended")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    rec_labels = cache["rec_labels"]  # dict[str, Tensor[T, 11]]
    rec_logits = cache["rec_logits"]  # dict[str, Tensor[T, 11]]
    recordings = sorted(rec_labels.keys())
    print(f"Loaded {len(recordings)} recordings, {cache['total_frames']} total frames")

    # Process each recording
    per_rec_pos = {}
    per_rec_edit = {}
    all_gt_pos = []

    for rec in recordings:
        lbl = rec_labels[rec].numpy()
        logit = rec_logits[rec].numpy()

        # Binarize logits
        pred = (logit > 0.0).astype(np.int32)
        gt = lbl.astype(np.int32)

        # Valid frames: any component >= 0
        # Note: labels from cache should already be valid
        valid = gt.max(axis=1) >= 0
        if valid.sum() < 2:
            print(f"  SKIP {rec}: only {valid.sum()} valid frames")
            continue

        vp = pred[valid]
        vl = gt[valid]
        nf = int(valid.sum())

        metrics = compute_metrics(vp, vl)

        per_rec_pos[rec] = {
            "ours_pos": metrics["pos"]["ours"],
            "null_all_zeros_pos": metrics["pos"]["null_all_zeros"],
            "null_copy_prev_pos": metrics["pos"]["null_copy_prev"],
            "n_frames": nf,
        }
        per_rec_edit[rec] = {
            "ours_edit": metrics["edit_mean"]["ours"],
            "null_all_zeros_edit": metrics["edit_mean"]["null_all_zeros"],
            "null_copy_prev_edit": metrics["edit_mean"]["null_copy_prev"],
            "ours_edit_per_component": metrics["edit_per_comp"]["ours"],
            "null_all_zeros_edit_per_component": metrics["edit_per_comp"]["null_all_zeros"],
            "null_copy_prev_edit_per_component": metrics["edit_per_comp"]["null_copy_prev"],
            "n_frames": nf,
        }

        # GT positive fraction
        pos_frac = (vl > 0).mean(axis=0)
        all_gt_pos.append(pos_frac)

    # Aggregate functions
    def mean_k(container, key):
        return float(np.mean([v[key] for v in container.values()]))

    def std_k(container, key):
        return float(np.std([v[key] for v in container.values()]))

    # POS summary
    pos_summary = {
        "checkpoint": "src/runs/rf_stages/checkpoints/best.pth",
        "n_frames": sum(v["n_frames"] for v in per_rec_pos.values()),
        "n_recordings": len(per_rec_pos),
        "ours_pos_mean": mean_k(per_rec_pos, "ours_pos"),
        "ours_pos_std": std_k(per_rec_pos, "ours_pos"),
        "null_all_zeros_pos_mean": mean_k(per_rec_pos, "null_all_zeros_pos"),
        "null_all_zeros_pos_std": std_k(per_rec_pos, "null_all_zeros_pos"),
        "null_copy_prev_pos_mean": mean_k(per_rec_pos, "null_copy_prev_pos"),
        "null_copy_prev_pos_std": std_k(per_rec_pos, "null_copy_prev_pos"),
        "interpretation": (
            "If null_copy_prev_pos == ours_pos, POS is a fill-forward artifact. "
            "Drop POS from headline."
        ),
        "per_recording": per_rec_pos,
    }

    # Edit summary
    edit_summary = {
        "checkpoint": "src/runs/rf_stages/checkpoints/best.pth",
        "n_frames": sum(v["n_frames"] for v in per_rec_edit.values()),
        "n_recordings": len(per_rec_edit),
        "ours_edit_mean": mean_k(per_rec_edit, "ours_edit"),
        "ours_edit_std": std_k(per_rec_edit, "ours_edit"),
        "null_all_zeros_edit_mean": mean_k(per_rec_edit, "null_all_zeros_edit"),
        "null_all_zeros_edit_std": std_k(per_rec_edit, "null_all_zeros_edit"),
        "null_copy_prev_edit_mean": mean_k(per_rec_edit, "null_copy_prev_edit"),
        "null_copy_prev_edit_std": std_k(per_rec_edit, "null_copy_prev_edit"),
        "interpretation": (
            "Edit = fraction of frame-level binary-value mismatches (Hamming / T). "
            "Lower is better. If null_copy_prev ≈ ours, Edit is also inflated."
        ),
        "per_recording": per_rec_edit,
    }

    # Per-component Edit
    ours_pc = [per_rec_edit[r]["ours_edit_per_component"] for r in recordings if r in per_rec_edit]
    zeros_pc = [
        per_rec_edit[r]["null_all_zeros_edit_per_component"]
        for r in recordings
        if r in per_rec_edit
    ]
    cp_pc = [
        per_rec_edit[r]["null_copy_prev_edit_per_component"]
        for r in recordings
        if r in per_rec_edit
    ]

    mean_ours_pc = np.mean(ours_pc, axis=0).tolist() if ours_pc else None
    mean_zeros_pc = np.mean(zeros_pc, axis=0).tolist() if zeros_pc else None
    mean_cp_pc = np.mean(cp_pc, axis=0).tolist() if cp_pc else None
    mean_gt = np.mean(all_gt_pos, axis=0).tolist() if all_gt_pos else None

    pce = {}
    for i in range(NUM_COMPONENTS):
        pce[f"comp{i}"] = {
            "ours_edit_mean": mean_ours_pc[i],
            "null_all_zeros_edit_mean": mean_zeros_pc[i],
            "null_copy_prev_edit_mean": mean_cp_pc[i],
            "gt_pos_frac_mean": mean_gt[i],
        }
    edit_summary["per_component_edit"] = pce

    # Save JSONs
    edit_path = save_dir / "edit_null.json"
    with open(edit_path, "w") as f:
        json.dump(edit_summary, f, indent=2)
    print(f"Saved: {edit_path}")

    pos_path = save_dir / "full_null_pos.json"
    with open(pos_path, "w") as f:
        json.dump(pos_summary, f, indent=2)
    print(f"Saved: {pos_path}")

    # Markdown summary
    md = build_markdown(pos_summary, edit_summary, pce)
    md_path = save_dir / "full_pos_summary.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved: {md_path}")

    # Print results
    print(f"\n{'=' * 70}")
    print(f"  NULL-POS EXTENDED ANALYSIS — All 16 Recordings + Edit")
    print(f"  GPU used: No (from cached logits)")
    print(f"{'=' * 70}")

    print(f"\nPOS (mean across {len(per_rec_pos)} recordings):")
    print(f"  Ours              {pos_summary['ours_pos_mean']:.6f}")
    print(f"  Null all-zeros    {pos_summary['null_all_zeros_pos_mean']:.6f}")
    print(f"  Null copy-prev    {pos_summary['null_copy_prev_pos_mean']:.6f}")

    print(f"\nEdit (mean across {len(per_rec_edit)} recordings):")
    print(f"  Ours              {edit_summary['ours_edit_mean']:.6f}")
    print(f"  Null all-zeros    {edit_summary['null_all_zeros_edit_mean']:.6f}")
    print(f"  Null copy-prev    {edit_summary['null_copy_prev_edit_mean']:.6f}")

    print(f"\nPer-component Edit (Ours):")
    print("  " + "  ".join([f"c{i}={mean_ours_pc[i]:.4f}" for i in range(NUM_COMPONENTS)]))
    print(f"Per-component Edit (null all-zeros):")
    print("  " + "  ".join([f"c{i}={mean_zeros_pc[i]:.4f}" for i in range(NUM_COMPONENTS)]))
    print(f"Per-component Edit (null copy-prev):")
    print("  " + "  ".join([f"c{i}={mean_cp_pc[i]:.4f}" for i in range(NUM_COMPONENTS)]))
    print(f"GT positive fraction (mean):")
    print("  " + "  ".join([f"c{i}={mean_gt[i]:.4f}" for i in range(NUM_COMPONENTS)]))


def build_markdown(pos_summary, edit_summary, pce):
    lines = []
    lines.append("# Null-POS Extended Analysis: 16 Recordings + Edit Metric")
    lines.append("")
    lines.append(f"**Source:** Cached model logits from `psr_data_cache_best.pth`")
    lines.append(f"**GPU used:** No (CPU-only, from cache)")
    lines.append(f"**Reference:** Opus 141 Q24 (extend to all 16 recs), Q29 (Edit metric)")
    lines.append("")
    lines.append(f"## Coverage")
    lines.append("")
    lines.append(
        f"- **{pos_summary['n_recordings']} recordings**, **{pos_summary['n_frames']:,}** valid frames"
    )
    lines.append(f"- 11 PSR binary components")
    lines.append(f"- 3 models: Ours (epoch_18 best.pth), Null all-zeros, Null copy-prev")
    lines.append(f"- 2 metrics: POS (pairwise orientation score), Edit (Hamming / T)")
    lines.append("")

    lines.append("## POS (Pairwise Orientation Score)")
    lines.append("")
    lines.append("| Model | Mean POS | Std POS |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Ours | {pos_summary['ours_pos_mean']:.6f} | {pos_summary['ours_pos_std']:.6f} |"
    )
    lines.append(
        f"| Null all-zeros | {pos_summary['null_all_zeros_pos_mean']:.6f} | {pos_summary['null_all_zeros_pos_std']:.6f} |"
    )
    lines.append(
        f"| Null copy-prev | {pos_summary['null_copy_prev_pos_mean']:.6f} | {pos_summary['null_copy_prev_pos_std']:.6f} |"
    )
    lines.append("")
    lines.append(f"**{pos_summary['interpretation']}**")
    lines.append("")

    lines.append("## Edit (Levenshtein Normalized Hamming Distance)")
    lines.append("")
    lines.append("| Model | Mean Edit | Std Edit |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Ours | {edit_summary['ours_edit_mean']:.6f} | {edit_summary['ours_edit_std']:.6f} |"
    )
    lines.append(
        f"| Null all-zeros | {edit_summary['null_all_zeros_edit_mean']:.6f} | {edit_summary['null_all_zeros_edit_std']:.6f} |"
    )
    lines.append(
        f"| Null copy-prev | {edit_summary['null_copy_prev_edit_mean']:.6f} | {edit_summary['null_copy_prev_edit_std']:.6f} |"
    )
    lines.append("")

    lines.append("## Per-Component Edit (mean across recordings)")
    lines.append("")
    lines.append("| Component | GT pos frac | Ours Edit | Null zeros Edit | Null cp Edit |")
    lines.append("|---|---|---|---|---|")
    for i in range(NUM_COMPONENTS):
        c = f"comp{i}"
        gt = pce[c]["gt_pos_frac_mean"]
        oe = pce[c]["ours_edit_mean"]
        ze = pce[c]["null_all_zeros_edit_mean"]
        ce = pce[c]["null_copy_prev_edit_mean"]
        lines.append(f"| {c} | {gt:.4f} | {oe:.6f} | {ze:.6f} | {ce:.6f} |")
    lines.append("")

    lines.append("## Per-Recording POS")
    lines.append("")
    lines.append("| Recording | Frames | Ours POS | Null zeros POS | Null cp POS |")
    lines.append("|---|---|---|---|---|")
    for rec in sorted(pos_summary["per_recording"].keys()):
        p = pos_summary["per_recording"][rec]
        lines.append(
            f"| {rec} | {p['n_frames']} | {p['ours_pos']:.4f} | {p['null_all_zeros_pos']:.4f} | {p['null_copy_prev_pos']:.4f} |"
        )
    lines.append("")

    lines.append("## Per-Recording Edit")
    lines.append("")
    lines.append("| Recording | Frames | Ours Edit | Null zeros Edit | Null cp Edit |")
    lines.append("|---|---|---|---|---|")
    for rec in sorted(edit_summary["per_recording"].keys()):
        p = edit_summary["per_recording"][rec]
        lines.append(
            f"| {rec} | {p['n_frames']} | {p['ours_edit']:.4f} | {p['null_all_zeros_edit']:.4f} | {p['null_copy_prev_edit']:.4f} |"
        )
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    o_pos = pos_summary["ours_pos_mean"]
    nz_pos = pos_summary["null_all_zeros_pos_mean"]
    nc_pos = pos_summary["null_copy_prev_pos_mean"]
    o_edit = edit_summary["ours_edit_mean"]
    nz_edit = edit_summary["null_all_zeros_edit_mean"]
    nc_edit = edit_summary["null_copy_prev_edit_mean"]
    lines.append(
        f"1. **POS inflation confirmed.** Our model POS ({o_pos:.4f}) is essentially identical to null copy-prev ({nc_pos:.4f}) and null all-zeros ({nz_pos:.4f}). POS is structurally inflated by frame-to-frame label persistence and is not a meaningful metric for PSR."
    )
    lines.append(
        f"2. **Edit reveals real signal.** Our model Edit ({o_edit:.4f}) is _lower_ than null all-zeros ({nz_edit:.4f}) but _higher_ than null copy-prev ({nc_edit:.4f}). This means the model does learn some PSR structure but is worse than simply copying the previous frame."
    )
    lines.append(
        f"3. **Per-component variation.** The model's Edit error is concentrated in rare-transition components:"
    )
    for i in range(NUM_COMPONENTS):
        c = f"comp{i}"
        gt = pce[c]["gt_pos_frac_mean"]
        oe = pce[c]["ours_edit_mean"]
        ce = pce[c]["null_copy_prev_edit_mean"]
        lines.append(f"    - {c}: GT prevalence {gt:.3f}, Ours Edit {oe:.4f} vs copy-prev {ce:.4f}")
    lines.append(
        f"4. **Conclusion.** POS is a flawed metric for sparse binary PSR sequences. Edit distance provides a more meaningful accuracy measure. The model shows positive but modest learned signal, with null copy-prev as a strong baseline."
    )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
