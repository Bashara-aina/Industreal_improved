"""Memory-efficient streaming full eval — computes running stats without storing all logits."""

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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--max-batches", type=int, default=999999)
    args = parser.parse_args()

    save_dir = args.save_dir or f"/tmp/eval_{Path(args.checkpoint).stem}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

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

    # Running accumulators
    # PSR: store per-component stats
    n = 0
    # PSR
    psr_tp = np.zeros(11)
    psr_fp = np.zeros(11)
    psr_fn = np.zeros(11)
    psr_pos_pred = np.zeros(11)
    psr_pos_true = np.zeros(11)
    psr_valid = np.zeros(11)
    # Activity
    act_correct = 0
    act_total = 0
    act_correct_valid = 0
    act_total_valid = 0
    # Pose
    pose_fwd_mae_sum = 0.0
    pose_fwd_n = 0
    pose_up_mae_sum = 0.0
    pose_up_n = 0

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

        # PSR
        if outputs.get("psr_logits") is not None and targets.get("psr_labels") is not None:
            pl = outputs["psr_logits"].cpu()  # [1, 11]
            pl_lbl = targets["psr_labels"]  # [1, 11]
            valid_mask = pl_lbl[0] != -1
            # Use best threshold 0.10 for epoch_18 (from sweep)
            binary = (torch.sigmoid(pl[0]) > 0.10).int()
            for c in range(11):
                if valid_mask[c]:
                    psr_valid[c] += 1
                    if binary[c] == 1 and pl_lbl[0, c] == 1:
                        psr_tp[c] += 1
                    elif binary[c] == 1 and pl_lbl[0, c] == 0:
                        psr_fp[c] += 1
                    elif binary[c] == 0 and pl_lbl[0, c] == 1:
                        psr_fn[c] += 1
                    psr_pos_pred[c] += binary[c].item()
                    psr_pos_true[c] += pl_lbl[0, c].item()

        # Activity
        if outputs.get("act_logits") is not None and targets.get("activity") is not None:
            al = outputs["act_logits"].cpu()
            al_lbl = targets["activity"].cpu()
            pred = al.argmax(dim=-1)
            act_correct += (pred == al_lbl).sum().item()
            act_total += al.shape[0]
            valid_mask = al_lbl >= 0
            act_correct_valid += (pred[valid_mask] == al_lbl[valid_mask]).sum().item()
            act_total_valid += valid_mask.sum().item()

        # Pose
        if outputs.get("head_pose") is not None and targets.get("head_pose") is not None:
            hp_p = outputs["head_pose"].cpu()
            hp_l = targets["head_pose"].cpu()
            # Forward (cols 0-2), position (3-5), Up (6-8)
            fwd_p = hp_p[:, :3]
            fwd_l = hp_l[:, :3]
            up_p = hp_p[:, 6:9]
            up_l = hp_l[:, 6:9]
            fwd_pn = fwd_p / (fwd_p.norm(dim=-1, keepdim=True) + 1e-8)
            fwd_ln = fwd_l / (fwd_l.norm(dim=-1, keepdim=True) + 1e-8)
            up_pn = up_p / (up_p.norm(dim=-1, keepdim=True) + 1e-8)
            up_ln = up_l / (up_l.norm(dim=-1, keepdim=True) + 1e-8)
            fwd_cos = (fwd_pn * fwd_ln).sum(dim=-1).clamp(-1, 1)
            up_cos = (up_pn * up_ln).sum(dim=-1).clamp(-1, 1)
            pose_fwd_mae_sum += torch.rad2deg(torch.acos(fwd_cos)).sum().item()
            pose_fwd_n += fwd_p.shape[0]
            pose_up_mae_sum += torch.rad2deg(torch.acos(up_cos)).sum().item()
            pose_up_n += up_p.shape[0]

        n += 1
        if n % 500 == 0:
            print(f"  processed {n}...")

    # Compute metrics
    results = {
        "checkpoint": args.checkpoint,
        "n_batches": n,
        "psr": {},
        "activity": {},
        "head_pose": {},
    }

    # PSR per-component
    print("\n--- PSR per-component (at threshold 0.10) ---")
    total_f1 = 0
    for c in range(11):
        prec = psr_tp[c] / max(psr_tp[c] + psr_fp[c], 1)
        rec = psr_tp[c] / max(psr_tp[c] + psr_fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if c < 5:
            print(
                f"  comp{c}: F1={f1:.4f} (P={prec:.3f}, R={rec:.3f}, valid={int(psr_valid[c])}, pred_pos={int(psr_pos_pred[c])}, true_pos={int(psr_pos_true[c])})"
            )
        total_f1 += f1
        results["psr"][f"comp{c}"] = {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "valid": int(psr_valid[c]),
            "pred_pos": int(psr_pos_pred[c]),
            "true_pos": int(psr_pos_true[c]),
        }
    avg_f1 = total_f1 / 11
    print(f"  Macro-avg F1: {avg_f1:.4f}")
    results["psr"]["macro_f1"] = float(avg_f1)

    # Activity
    if act_total > 0:
        act_top1 = act_correct / act_total
        act_top1_valid = act_correct_valid / max(act_total_valid, 1)
        results["activity"] = {
            "top1": float(act_top1),
            "top1_valid": float(act_top1_valid),
            "n_total": act_total,
            "n_valid": act_total_valid,
        }
        print(
            f"\nActivity: top1={act_top1:.4f}, top1_valid={act_top1_valid:.4f} ({act_total_valid} valid / {act_total} total)"
        )

    # Pose
    if pose_fwd_n > 0:
        fwd_mae = pose_fwd_mae_sum / pose_fwd_n
        up_mae = pose_up_mae_sum / pose_up_n
        results["head_pose"] = {
            "forward_angular_MAE_deg": float(fwd_mae),
            "up_angular_MAE_deg": float(up_mae),
            "n": pose_fwd_n,
        }
        print(f"\nHead Pose: fwd_angular_MAE={fwd_mae:.2f}°, up_angular_MAE={up_mae:.2f}°")

    # Save
    out_path = Path(save_dir) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
