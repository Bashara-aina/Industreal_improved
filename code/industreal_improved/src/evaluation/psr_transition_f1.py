"""PSR per-component F1 vs transition F1 side-by-side.

For the same predictions:
- per-frame F1: each frame's per-component prediction vs GT
- transition F1: event matching within tolerance (B3/STORM protocol)
- per-component transition F1: per-component breakdown (Opus 141 Q45)

Output: json with both metrics per recording, summary stats,
and per_comp_f1.json/md with per-component transition F1.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

COMPONENT_NAMES = [
    "comp0", "comp1", "comp2", "comp3", "comp4",
    "comp5", "comp6", "comp7", "comp8", "comp9", "comp10",
]


def event_f1(pred_tr, gt_tr, tol=3):
    """Aggregate event F1 with tolerance. B3/STORM protocol."""
    if not pred_tr.any() and not gt_tr.any():
        return 1.0
    if not pred_tr.any() or not gt_tr.any():
        return 0.0
    n_comp = pred_tr.shape[1]
    tp, fp, fn_tot = 0, 0, 0
    for c in range(n_comp):
        p_frames = np.where(pred_tr[:, c])[0]
        g_frames = np.where(gt_tr[:, c])[0]
        matched = set()
        for pf in p_frames:
            for gi, gf in enumerate(g_frames):
                if gi not in matched and abs(pf - gf) <= tol:
                    matched.add(gi)
                    tp += 1
                    break
            else:
                fp += 1
        fn_tot += len(g_frames) - len(matched)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn_tot, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def per_component_event_f1(pred_tr, gt_tr, tol=3):
    """Per-component event F1. Returns list of F1 per comp, micro F1, macro F1."""
    n_comp = pred_tr.shape[1]
    comp_f1s = []
    micro_tp, micro_fp, micro_fn = 0, 0, 0
    for c in range(n_comp):
        p_frames = np.where(pred_tr[:, c])[0]
        g_frames = np.where(gt_tr[:, c])[0]
        if not p_frames.any() and not g_frames.any():
            comp_f1s.append(1.0)
            continue
        if not p_frames.any() or not g_frames.any():
            comp_f1s.append(0.0)
            micro_fn += len(g_frames)
            continue
        tp, fp, fn_c = 0, 0, 0
        matched = set()
        for pf in p_frames:
            for gi, gf in enumerate(g_frames):
                if gi not in matched and abs(pf - gf) <= tol:
                    matched.add(gi)
                    tp += 1
                    break
            else:
                fp += 1
        fn_c = len(g_frames) - len(matched)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn_c, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        comp_f1s.append(f1)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn_c

    # Macro = simple mean of per-component F1
    macro = float(np.mean(comp_f1s))
    # Micro = F1 on pooled TP/FP/FN
    mprec = micro_tp / max(micro_tp + micro_fp, 1)
    mrec = micro_tp / max(micro_tp + micro_fn, 1)
    micro = 2 * mprec * mrec / max(mprec + mrec, 1e-9)
    return comp_f1s, macro, micro


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="src/runs/rf_stages/checkpoints/best.pth")
    parser.add_argument("--max-batches", type=int, default=5000)
    parser.add_argument("--save-dir", default="src/runs/rf_stages/checkpoints/psr_per_vs_trans")
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--thresholds", default=None,
                        help="Path to optimal_thresholds.json for per-component thresholds")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load per-component thresholds if provided
    per_comp_thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thr_data = json.load(f)
        per_comp_thresholds = np.array(thr_data["optimal_thresholds"], dtype=np.float32)
        print(f"Loaded per-component thresholds: {per_comp_thresholds.tolist()}")
    else:
        print("Using global threshold 0.5 for all components")

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Epoch: {ckpt.get('epoch')}")

    from src.models.model import POPWMultiTaskModel
    from src.models.psr_transition import MonotonicDecoder
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    model = POPWMultiTaskModel(
        pretrained=True, backbone_type='convnext_tiny',
        use_hand_film=True, use_headpose_film=True,
        use_videomae=False, train_pose=False,
    )
    state_dict = {k: v for k, v in ckpt["model"].items()
                  if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()

    decoder = MonotonicDecoder(num_components=11)
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False,
    )

    # Collect per-recording per-frame predictions and labels
    rec_preds = defaultdict(list)
    rec_labels = defaultdict(list)
    rec_frame_nums = defaultdict(list)

    n = 0
    for i, batch in enumerate(val_loader):
        if i >= args.max_batches:
            break
        images, targets = batch
        if images.shape[0] == 0:
            continue
        images_f = images.cuda().float()
        if images_f.max() > 1.0:
            images_f = images_f.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images_f.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images_f.device).view(1, 3, 1, 1)
        images_n = (images_f - mean) / std

        with torch.no_grad():
            outputs = model(images_n)

        pl = outputs.get("psr_logits")
        pl_lbl = targets.get("psr_labels")
        if pl is None or pl_lbl is None:
            continue

        # Per-component binary prediction (sigmoid > threshold)
        sig = torch.sigmoid(pl[0]).cpu().numpy()
        if per_comp_thresholds is not None:
            pred_bin = (sig > per_comp_thresholds).astype(np.int32)
        else:
            pred_bin = (sig > 0.5).astype(np.int32)
        lbl = pl_lbl[0].cpu().numpy()

        meta = targets.get("metadata", [])
        rec_id = meta[0].get("recording_id", "unknown")
        frame_num = meta[0].get("frame_num", 0)

        rec_preds[rec_id].append(pred_bin)
        rec_labels[rec_id].append(lbl)
        rec_frame_nums[rec_id].append(frame_num)
        n += 1
        if n % 2000 == 0:
            print(f"  processed {n} frames...")

    print(f"\nCollected {n} frames across {len(rec_preds)} recordings")

    # Compute both metrics per recording
    from src import config as C
    remap_fn = getattr(C, 'remap_activity_label', None)

    per_frame_f1_per_rec = {}
    transition_f1_per_rec = {}
    per_comp_trans_f1_per_rec = {}  # Opus 141 Q45
    pos_per_rec = {}
    edit_per_rec = {}

    for rec, preds in rec_preds.items():
        gt = rec_labels[rec]
        frames = np.array(rec_frame_nums[rec])
        sort_idx = frames.argsort()
        preds_sorted = np.array(preds)[sort_idx]
        gt_sorted = np.array(gt)[sort_idx]

        # Per-frame F1 (only valid labels)
        valid_mask = gt_sorted.max(axis=1) >= 0
        if valid_mask.sum() == 0:
            continue
        vp = preds_sorted[valid_mask]
        vl = gt_sorted[valid_mask]
        f1_per_comp = []
        for c in range(11):
            tp = ((vp[:, c] == 1) & (vl[:, c] == 1)).sum()
            fp = ((vp[:, c] == 1) & (vl[:, c] == 0)).sum()
            fn = ((vp[:, c] == 0) & (vl[:, c] == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec_v = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec_v / max(prec + rec_v, 1e-9)
            f1_per_comp.append(f1)
        per_frame_f1_per_rec[rec] = float(np.mean(f1_per_comp))

        # Transition F1
        # Use events = (frame+1 - frame) > 0 (transitions)
        pred_tr = np.clip(preds_sorted[1:] - preds_sorted[:-1], a_min=0, a_max=None)
        gt_tr = np.clip(gt_sorted[1:] - gt_sorted[:-1], a_min=0, a_max=None)
        # Filter to valid GT frames
        valid_tr = gt_sorted[1:].max(axis=1) >= 0
        trans_f1 = event_f1(pred_tr[valid_tr], gt_tr[valid_tr], tol=args.tolerance)
        transition_f1_per_rec[rec] = trans_f1

        # Per-component transition F1 (Opus 141 Q45)
        comp_f1s, comp_macro, comp_micro = per_component_event_f1(
            pred_tr[valid_tr], gt_tr[valid_tr], tol=args.tolerance,
        )
        per_comp_trans_f1_per_rec[rec] = {
            "per_component": {COMPONENT_NAMES[i]: float(comp_f1s[i]) for i in range(11)},
            "macro": float(comp_macro),
            "micro": float(comp_micro),
        }

        # POS (ordered-pair fraction)
        pred_pairs = preds_sorted[1:] - preds_sorted[:-1]
        gt_pairs = gt_sorted[1:] - gt_sorted[:-1]
        pos = float((np.sign(pred_pairs) == np.sign(gt_pairs)).mean())
        pos_per_rec[rec] = pos

        # Edit score
        pred_events = "".join(
            str(int(b)) for b in (preds_sorted[1:] != preds_sorted[:-1]).any(axis=1)
        )
        gt_events = "".join(
            str(int(b)) for b in (gt_sorted[1:] != gt_sorted[:-1]).any(axis=1)
        )
        if gt_events:
            m, k = len(pred_events), len(gt_events)
            dp = np.zeros((m + 1, k + 1))
            for ii in range(m + 1):
                dp[ii, 0] = ii
            for jj in range(k + 1):
                dp[0, jj] = jj
            for ii in range(1, m + 1):
                for jj in range(1, k + 1):
                    dp[ii, jj] = min(
                        dp[ii - 1, jj] + 1,
                        dp[ii, jj - 1] + 1,
                        dp[ii - 1, jj - 1] + (pred_events[ii - 1] != gt_events[jj - 1]),
                    )
            edit_per_rec[rec] = 1.0 - dp[m, k] / max(m, k, 1)
        else:
            edit_per_rec[rec] = 1.0

    summary = {
        "checkpoint": args.checkpoint,
        "n_frames": n,
        "n_recordings": len(per_frame_f1_per_rec),
        "tolerance": args.tolerance,
        "per_frame_macro_f1": float(np.mean(list(per_frame_f1_per_rec.values()))) if per_frame_f1_per_rec else 0.0,
        "transition_macro_f1": float(np.mean(list(transition_f1_per_rec.values()))) if transition_f1_per_rec else 0.0,
        "macro_pos": float(np.mean(list(pos_per_rec.values()))) if pos_per_rec else 0.0,
        "macro_edit": float(np.mean(list(edit_per_rec.values()))) if edit_per_rec else 0.0,
        "per_recording": {
            rec: {
                "per_frame_f1": per_frame_f1_per_rec[rec],
                "transition_f1": transition_f1_per_rec[rec],
                "per_comp_trans_f1": per_comp_trans_f1_per_rec[rec],
                "pos": pos_per_rec[rec],
                "edit": edit_per_rec[rec],
            }
            for rec in per_frame_f1_per_rec
        },
    }

    # --- Per-component transition F1 aggregation (Opus 141 Q45) ---
    # Gather per-component F1 arrays across all recordings
    comp_f1_by_comp = defaultdict(list)  # comp_name -> list of per-recording F1
    macro_list = []
    micro_list = []
    for rec, data in per_comp_trans_f1_per_rec.items():
        for cname, f1val in data["per_component"].items():
            comp_f1_by_comp[cname].append(f1val)
        macro_list.append(data["macro"])
        micro_list.append(data["micro"])

    per_comp_summary = {}
    for cname in COMPONENT_NAMES:
        vals = comp_f1_by_comp[cname]
        per_comp_summary[cname] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_recordings": len(vals),
        }

    per_comp_cross_rec = {
        "checkpoint": args.checkpoint,
        "tolerance": args.tolerance,
        "n_frames": n,
        "n_recordings": len(per_comp_trans_f1_per_rec),
        "macro_across_recordings": {
            "mean": float(np.mean(macro_list)),
            "std": float(np.std(macro_list)),
        },
        "micro_across_recordings": {
            "mean": float(np.mean(micro_list)),
            "std": float(np.std(micro_list)),
        },
        "per_component": per_comp_summary,
        "per_recording": {
            rec: per_comp_trans_f1_per_rec[rec]
            for rec in per_comp_trans_f1_per_rec
        },
    }

    # Save JSON
    out = Path(args.save_dir) / "per_frame_vs_transition.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}")

    # Save per-component JSON
    per_comp_dir = Path(args.save_dir)
    per_comp_json = per_comp_dir / "per_comp_f1.json"
    with open(per_comp_json, "w") as f:
        json.dump(per_comp_cross_rec, f, indent=2)
    print(f"Per-component results saved to {per_comp_json}")

    # Save markdown report
    per_comp_md = Path(args.save_dir) / "per_comp_f1.md"
    lines_md = [
        "# Per-Component Transition F1\n",
        f"**Checkpoint:** `{args.checkpoint}`  \n",
        f"**Tolerance:** {args.tolerance} frames  \n",
        f"**Frames:** {n}  \n",
        f"**Recordings:** {len(per_comp_trans_f1_per_rec)}\n\n",
        "| Component | Mean F1 | Std | Min | Max | Recordings |\n",
        "|-----------|---------|-----|-----|-----|------------|\n",
    ]
    for cname in COMPONENT_NAMES:
        s = per_comp_summary[cname]
        lines_md.append(
            f"| {cname} | {s['mean']:.4f} | {s['std']:.4f} "
            f"| {s['min']:.4f} | {s['max']:.4f} | {s['n_recordings']} |\n"
        )
    lines_md.append(f"\n**Macro average:** {per_comp_cross_rec['macro_across_recordings']['mean']:.4f}  \n")
    lines_md.append(f"**Micro average:** {per_comp_cross_rec['micro_across_recordings']['mean']:.4f}  \n")
    with open(per_comp_md, "w") as f:
        f.writelines(lines_md)
    print(f"Markdown report saved to {per_comp_md}")

    print(f"Per-frame macro F1: {summary['per_frame_macro_f1']:.4f}")
    print(f"Transition macro F1 (tol={args.tolerance}): {summary['transition_macro_f1']:.4f}")
    print(f"Macro POS: {summary['macro_pos']:.4f}")
    print(f"Macro Edit: {summary['macro_edit']:.4f}")

    # Print per-component table
    print(f"\n{'='*60}")
    print("PER-COMPONENT TRANSITION F1 (Opus 141 Q45)")
    print(f"{'='*60}")
    print(f"{'Component':<12} {'Mean F1':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Recs':<6}")
    print(f"{'-'*58}")
    for cname in COMPONENT_NAMES:
        s = per_comp_summary[cname]
        print(f"{cname:<12} {s['mean']:.4f}     {s['std']:.4f}     {s['min']:.4f}     {s['max']:.4f}     {s['n_recordings']:<6}")
    print(f"{'-'*58}")
    print(f"{'Macro':<12} {per_comp_cross_rec['macro_across_recordings']['mean']:.4f}")
    print(f"{'Micro':<12} {per_comp_cross_rec['micro_across_recordings']['mean']:.4f}")


if __name__ == "__main__":
    main()