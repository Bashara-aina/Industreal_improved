"""Optimal per-component threshold sweep on epoch_18 checkpoint."""

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
    parser.add_argument("--save-dir", default="/tmp/psr_optimal_threshold")
    parser.add_argument("--max-batches", type=int, default=20000)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Epoch: {ckpt.get('epoch')}")

    from src.models.model import POPWMultiTaskModel

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

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from torch.utils.data import DataLoader

    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False
    )

    # Collect per-component sigmas and labels
    per_comp_sig = [[] for _ in range(11)]
    per_comp_lbl = [[] for _ in range(11)]
    n = 0
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
        sig = torch.sigmoid(pl[0]).cpu().numpy()
        lbl = pl_lbl[0].numpy()
        for c in range(11):
            if lbl[c] != -1:
                per_comp_sig[c].append(sig[c])
                per_comp_lbl[c].append(lbl[c])
        n += 1
        if n % 2000 == 0:
            print(f"  processed {n}...")

    # Sweep per-component thresholds
    print(f"\nProcessed {n} frames")
    print("\n--- Optimal per-component thresholds ---")
    optimal_thresholds = []
    total_f1 = 0
    for c in range(11):
        sigs = np.array(per_comp_sig[c])
        lbls = np.array(per_comp_lbl[c])
        if len(lbls) == 0:
            optimal_thresholds.append(0.5)
            continue
        gt_pos = lbls.mean()
        best_f1, best_t = 0, 0
        for t in np.arange(0.05, 1.0, 0.05):
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
        optimal_thresholds.append(float(best_t))
        total_f1 += best_f1
        print(f"  comp{c}: gt_pos={gt_pos:.3f}, best_thresh={best_t:.2f}, F1={best_f1:.4f}")

    macro_f1 = total_f1 / 11
    print(f"\nOptimal macro F1 (with per-comp thresholds): {macro_f1:.4f}")

    # Compare with global 0.10
    print("\n--- Comparison ---")
    total_f1_g = 0
    for c in range(11):
        sigs = np.array(per_comp_sig[c])
        lbls = np.array(per_comp_lbl[c])
        binary = (sigs > 0.10).astype(np.int32)
        tp = ((binary == 1) & (lbls == 1)).sum()
        fp = ((binary == 1) & (lbls == 0)).sum()
        fn = ((binary == 0) & (lbls == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        total_f1_g += f1
    macro_g = total_f1_g / 11
    print(f"Global 0.10 macro F1: {macro_g:.4f}")

    # Save
    out = {
        "checkpoint": args.checkpoint,
        "n_frames": n,
        "n_per_comp": [len(per_comp_lbl[c]) for c in range(11)],
        "optimal_thresholds": optimal_thresholds,
        "optimal_macro_f1": float(macro_f1),
        "global_0.10_macro_f1": float(macro_g),
    }
    out_path = Path(args.save_dir) / "optimal_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
