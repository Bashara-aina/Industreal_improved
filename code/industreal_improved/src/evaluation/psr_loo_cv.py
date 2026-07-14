"""PSR per-comp threshold LOO-CV: tests if 0.7499 is real or val overfit.

For each of 16 recordings: compute optimal thresholds on the other 15,
evaluate on the held-out one. If the held-out improvement matches the
in-sample improvement, threshold is real.
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="src/runs/rf_stages/checkpoints/best.pth")
    parser.add_argument("--max-batches", type=int, default=5000)
    parser.add_argument("--save-dir", default="src/runs/rf_stages/checkpoints/psr_loo_cv")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    from src.models.model import POPWMultiTaskModel
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type="convnext_tiny",
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=False,
    )
    state_dict = {
        k: v for k, v in ckpt["model"].items() if "total_ops" not in k and "total_params" not in k
    }
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()

    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Collect per-recording per-component sigmoid scores and labels
    rec_sig = defaultdict(list)
    rec_lbl = defaultdict(list)

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

        sig = torch.sigmoid(pl[0]).cpu().numpy()
        lbl = pl_lbl[0].cpu().numpy()
        meta = targets.get("metadata", [])
        rec_id = meta[0].get("recording_id", "unknown")

        rec_sig[rec_id].append(sig)
        rec_lbl[rec_id].append(lbl)
        n += 1
        if n % 2000 == 0:
            print(f"  processed {n} frames...")

    print(f"\nCollected {n} frames across {len(rec_sig)} recordings")

    # Find optimal threshold for each component on each (N-1) subset
    thresholds = np.arange(0.05, 1.0, 0.05)
    recordings = sorted(rec_sig.keys())
    n_rec = len(recordings)
    print(f"Running LOO-CV across {n_rec} recordings...")

    # First compute per-component optimal thresholds on full data (for reference)
    full_opt = {}
    full_global_f1 = []
    for c in range(11):
        all_sig = np.concatenate([np.array(rec_sig[r])[:, c] for r in recordings])
        all_lbl = np.concatenate([np.array(rec_lbl[r])[:, c] for r in recordings])
        valid = all_lbl != -1
        sigs = all_sig[valid]
        lbls = all_lbl[valid]
        # Global 0.10 F1
        binary = (sigs > 0.10).astype(np.int32)
        tp = ((binary == 1) & (lbls == 1)).sum()
        fp = ((binary == 1) & (lbls == 0)).sum()
        fn = ((binary == 0) & (lbls == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        global_f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        full_global_f1.append(global_f1)
        # Optimal
        best_f1, best_t = 0, 0
        for t in thresholds:
            binary = (sigs > t).astype(np.int32)
            tp = ((binary == 1) & (lbls == 1)).sum()
            fp = ((binary == 1) & (lbls == 0)).sum()
            fn = ((binary == 0) & (lbls == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        full_opt[c] = float(best_t)

    # LOO-CV: for each held-out rec, fit on (N-1) others, evaluate on held-out
    loo_f1s_global = []
    loo_f1s_optimal = []
    per_rec_results = {}

    if n_rec < 2:
        print(f"ERROR: need >=2 recordings for LOO, got {n_rec}. Increase --max-batches.")
        return

    for held in recordings:
        train_recs = [r for r in recordings if r != held]

        # Fit optimal thresholds on training subset
        opt_thresh = {}
        for c in range(11):
            all_sig = np.concatenate([np.array(rec_sig[r])[:, c] for r in train_recs])
            all_lbl = np.concatenate([np.array(rec_lbl[r])[:, c] for r in train_recs])
            valid = all_lbl != -1
            if valid.sum() == 0:
                opt_thresh[c] = 0.5
                continue
            sigs = all_sig[valid]
            lbls = all_lbl[valid]
            best_f1, best_t = 0, 0.5
            for t in thresholds:
                binary = (sigs > t).astype(np.int32)
                tp = ((binary == 1) & (lbls == 1)).sum()
                fp = ((binary == 1) & (lbls == 0)).sum()
                fn = ((binary == 0) & (lbls == 1)).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            opt_thresh[c] = float(best_t)

        # Evaluate on held-out with global and optimal
        f1s_global_c = []
        f1s_optimal_c = []
        for c in range(11):
            sigs = np.array(rec_sig[held])[:, c]
            lbls = np.array(rec_lbl[held])[:, c]
            valid = lbls != -1
            if valid.sum() == 0:
                continue
            sigs_v = sigs[valid]
            lbls_v = lbls[valid]
            # Global 0.10
            binary = (sigs_v > 0.10).astype(np.int32)
            tp = ((binary == 1) & (lbls_v == 1)).sum()
            fp = ((binary == 1) & (lbls_v == 0)).sum()
            fn = ((binary == 0) & (lbls_v == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1g = 2 * prec * rec / max(prec + rec, 1e-9)
            f1s_global_c.append(f1g)
            # Optimal
            binary = (sigs_v > opt_thresh[c]).astype(np.int32)
            tp = ((binary == 1) & (lbls_v == 1)).sum()
            fp = ((binary == 1) & (lbls_v == 0)).sum()
            fn = ((binary == 0) & (lbls_v == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1o = 2 * prec * rec / max(prec + rec, 1e-9)
            f1s_optimal_c.append(f1o)

        if f1s_global_c:
            loo_f1s_global.append(np.mean(f1s_global_c))
            loo_f1s_optimal.append(np.mean(f1s_optimal_c))
            per_rec_results[held] = {
                "global_f1": float(np.mean(f1s_global_c)),
                "optimal_f1": float(np.mean(f1s_optimal_c)),
                "improvement": float(np.mean(f1s_optimal_c) - np.mean(f1s_global_c)),
                "opt_thresholds": opt_thresh,
            }

    summary = {
        "checkpoint": args.checkpoint,
        "n_frames": n,
        "n_recordings": n_rec,
        "full_data_global_f1": float(np.mean(full_global_f1)),
        "loo_global_f1_mean": float(np.mean(loo_f1s_global)),
        "loo_optimal_f1_mean": float(np.mean(loo_f1s_optimal)),
        "loo_improvement_mean": float(np.mean(loo_f1s_optimal) - np.mean(loo_f1s_global)),
        "loo_improvement_std": float(np.std([v["improvement"] for v in per_rec_results.values()])),
        "per_recording": per_rec_results,
    }

    # Compute full-data per-comp optimal F1
    full_opt_f1s = []
    for c in range(11):
        all_sig = np.concatenate([np.array(rec_sig[r])[:, c] for r in recordings])
        all_lbl = np.concatenate([np.array(rec_lbl[r])[:, c] for r in recordings])
        valid = all_lbl != -1
        sigs = all_sig[valid]
        lbls = all_lbl[valid]
        binary = (sigs > full_opt[c]).astype(np.int32)
        tp = ((binary == 1) & (lbls == 1)).sum()
        fp = ((binary == 1) & (lbls == 0)).sum()
        fn = ((binary == 0) & (lbls == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        full_opt_f1s.append(f1)
    summary["full_data_optimal_f1_macro"] = float(np.mean(full_opt_f1s))
    summary["full_data_optimal_thresholds"] = full_opt

    out = Path(args.save_dir) / "loo_cv_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"Full-data global F1: {summary['full_data_global_f1']:.4f}")
    print(f"Full-data optimal F1: {summary['full_data_optimal_f1_macro']:.4f}")
    print(
        f"Full-data improvement: {summary['full_data_optimal_f1_macro'] - summary['full_data_global_f1']:.4f}"
    )
    print(f"LOO-CV global F1: {summary['loo_global_f1_mean']:.4f}")
    print(f"LOO-CV optimal F1: {summary['loo_optimal_f1_mean']:.4f}")
    print(f"LOO-CV improvement (mean): {summary['loo_improvement_mean']:.4f}")
    print(f"LOO-CV improvement (std): {summary['loo_improvement_std']:.4f}")
    if summary["loo_improvement_mean"] < 0.005:
        print(">>> WARNING: LOO-CV improvement < 0.005 — threshold is likely val overfit")
    else:
        print(">>> LOO-CV improvement persists — threshold is real")


if __name__ == "__main__":
    main()
