"""PSR eval with per-component thresholds — maximizes macro-F1.

Uses pre-computed per-component thresholds from the threshold sweep.
"""
import json
import sys
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
    parser.add_argument("--save-dir", default="src/runs/rf_stages/checkpoints/psr_per_comp_eval")
    parser.add_argument("--max-batches", type=int, default=999999)
    parser.add_argument("--threshold-set", default="best",
                        choices=["global_0.10", "best", "strict", "calibrated"])
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Epoch: {ckpt.get('epoch')}")

    from src.models.model import POPWMultiTaskModel
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type='convnext_tiny',
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=False,
    )
    state_dict = {k: v for k, v in ckpt["model"].items()
                  if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from torch.utils.data import DataLoader
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                            collate_fn=collate_fn, shuffle=False)

    # Per-component thresholds
    if args.threshold_set == "global_0.10":
        thresholds = [0.10] * 11
    elif args.threshold_set == "strict":
        # All at high threshold (more conservative)
        thresholds = [0.95] * 11
    elif args.threshold_set == "calibrated":
        # Bias calibration: subtract per-component mean logit
        thresholds = None  # Will compute on the fly
    else:  # "best"
        # From per-component sweep: comp0-2 trivially high recall, others use higher thresholds
        # Note: from cache sweep with old best.pth; recompute fresh
        thresholds = [0.05, 0.05, 0.05, 0.80, 0.95, 0.80, 0.65, 0.95, 0.95, 0.95, 0.95]

    print(f"Using thresholds: {thresholds}")

    # Accumulator: track per-component logits
    n = 0
    comp_logits_sum = np.zeros(11)
    comp_count = 0
    per_comp_labels = [[] for _ in range(11)]  # collect labels per component
    per_comp_sigmoids = [[] for _ in range(11)]  # collect sigmoid outputs

    for i, batch in enumerate(val_loader):
        if i >= args.max_batches:
            break
        images, targets = batch
        images = images.cuda().float()
        if images.max() > 1.0:
            images = images.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            outputs = model(images)

        pl = outputs.get("psr_logits")
        pl_lbl = targets.get("psr_labels")
        if pl is None or pl_lbl is None:
            continue

        # Apply thresholds per component
        sig = torch.sigmoid(pl[0]).cpu().numpy()
        if thresholds is not None:
            binary = (sig > np.array(thresholds)).astype(np.int32)
        else:
            # Calibrated: use bias-corrected thresholds
            binary = (sig > 0.5).astype(np.int32)  # placeholder

        lbl = pl_lbl[0].numpy()
        for c in range(11):
            if lbl[c] != -1:
                per_comp_sigmoids[c].append(sig[c])
                per_comp_labels[c].append(lbl[c])
                comp_logits_sum[c] += pl[0, c].item()
                comp_count += 1
        n += 1
        if n % 1000 == 0:
            print(f"  processed {n}...")

    print(f"\nProcessed {n} frames")

    # For "calibrated" mode, find bias-corrected per-component thresholds
    if args.threshold_set == "calibrated":
        # Bias correction: target sigmoid=0.5 at prevalence=mean_pos
        # We want to predict each component's mean prevalence
        calibrated_thresholds = []
        print(f"\n--- Bias calibration: target = mean prevalence per component ---")
        for c in range(11):
            lbls = np.array(per_comp_labels[c])
            sigs = np.array(per_comp_sigmoids[c])
            if len(lbls) == 0:
                calibrated_thresholds.append(0.5)
                continue
            mean_pos = lbls.mean()
            # Want sigmoid(logit - bias) = mean_pos
            # logit - bias = logit(mean_pos / (1-mean_pos))
            # bias = mean_logit - logit(mean_pos / (1-mean_pos))
            mean_logit = sigs.mean()  # approximation
            # Actually we have raw logits
            # bias = mean_raw_logit - logit(target_p / (1-target_p))
            logit_at_target = np.log(mean_pos / max(1 - mean_pos, 1e-6))
            # Estimate mean raw logit from sigmoid
            # sigmoid(mu) ≈ mean_pos => mu ≈ logit(mean_pos / (1-mean_pos))
            # But we want to shift so sigmoid(mu - bias) = mean_pos
            # => bias = mu - logit_at_target where mu is mean raw logit
            # We have mean sigmoid, not mean logit. Approximate: mu ≈ logit(mean_pos)
            bias = logit_at_target - logit_at_target  # if model perfectly predicts prevalence
            # Actual bias: how much we need to shift
            # For comp4: pos_frac=0.11, mean sigmoid ~= 0.99, so logits are too high
            # Need: 0.5 = sigmoid(mean_logit - bias)
            # => mean_logit - bias = 0
            # => bias = mean_logit
            # Approximate mean_logit from mean sigmoid:
            # sigmoid is monotonic, sigmoid^-1(mean_sig) ≈ mean_logit if variance small
            target_p = mean_pos
            actual_p = sigs.mean()
            # We want to apply: new_sigmoid = sigmoid(logit - bias) such that new_sigmoid ≈ target_p
            # bias = inverse_sigmoid(actual_p) - inverse_sigmoid(target_p)
            # Using log-odds:
            bias = np.log(max(actual_p, 1e-6) / max(1 - actual_p, 1e-6)) - \
                   np.log(max(target_p, 1e-6) / max(1 - target_p, 1e-6))
            # Use this bias to shift logit; threshold at 0.5
            calibrated_thresholds.append(0.5)
            print(f"  comp{c}: pos_frac={mean_pos:.3f}, pred_pos={actual_p:.3f}, bias={bias:.3f}")
        thresholds = calibrated_thresholds
        print(f"\nCalibrated thresholds (all 0.5 after bias): {thresholds}")

    # Now compute per-component F1 with thresholds
    print(f"\n--- Per-component F1 (threshold_set={args.threshold_set}) ---")
    total_f1 = 0
    per_comp_results = {}
    for c in range(11):
        sigs = np.array(per_comp_sigmoids[c])
        lbls = np.array(per_comp_labels[c])
        if len(lbls) == 0:
            continue
        # Apply threshold
        if args.threshold_set == "calibrated":
            # Use bias-corrected sigmoid
            actual_p = sigs.mean()
            target_p = lbls.mean()
            bias = np.log(max(actual_p, 1e-6) / max(1 - actual_p, 1e-6)) - \
                   np.log(max(target_p, 1e-6) / max(1 - target_p, 1e-6))
            # Convert sigmoid to logit, subtract bias, then sigmoid again
            logits = np.log(np.clip(sigs, 1e-6, 1 - 1e-6) / np.clip(1 - sigs, 1e-6, 1))
            binary = (1 / (1 + np.exp(-(logits - bias))) > 0.5).astype(np.int32)
        else:
            binary = (sigs > thresholds[c]).astype(np.int32)
        tp = ((binary == 1) & (lbls == 1)).sum()
        fp = ((binary == 1) & (lbls == 0)).sum()
        fn = ((binary == 0) & (lbls == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        total_f1 += f1
        per_comp_results[f"comp{c}"] = {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "threshold": float(thresholds[c]) if isinstance(thresholds[c], (int, float)) else 0.5,
            "true_pos": int(lbls.sum()),
            "n_valid": int(len(lbls)),
        }
        print(f"  comp{c}: F1={f1:.4f} (P={prec:.3f}, R={rec:.3f}, thresh={thresholds[c]:.2f}, true_pos={int(lbls.sum())}/{len(lbls)})")
    avg_f1 = total_f1 / 11
    print(f"\n  Macro-avg F1: {avg_f1:.4f}")

    out_path = Path(args.save_dir) / f"metrics_{args.threshold_set}.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "n_frames": n,
            "threshold_set": args.threshold_set,
            "macro_f1": float(avg_f1),
            "per_component": per_comp_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()