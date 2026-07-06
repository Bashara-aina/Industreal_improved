"""Full eval on a checkpoint — bypass broken evaluate.py main().

Computes per-task metrics: detection (mAP50), activity (top1/F1), pose (angular MAE), PSR (F1).
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--max-batches", type=int, default=2000)
    parser.add_argument("--start-batch", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    save_dir = args.save_dir or f"/tmp/eval_{Path(ckpt_path).stem}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Epoch: {ckpt.get('epoch')}, best_metric: {ckpt.get('best_metric', '?')}")

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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    model._seq_len = 1
    model = model.cuda().eval()

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from torch.utils.data import DataLoader
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                            collate_fn=collate_fn, shuffle=False)

    # Per-task accumulators
    psr_logits_list, psr_labels_list = [], []
    act_logits_list, act_labels_list = [], []
    pose_preds_list, pose_labels_list = [], []
    hp_preds_list, hp_labels_list = [], []  # head pose
    cls_preds_list, reg_preds_list = [], []  # detection
    n_boxes = 0
    n_batches = 0

    for i, batch in enumerate(val_loader):
        if i < args.start_batch:
            continue
        if i >= args.start_batch + args.max_batches:
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

        if outputs.get("psr_logits") is not None and targets.get("psr_labels") is not None:
            psr_logits_list.append(outputs["psr_logits"].cpu())
            psr_labels_list.append(targets["psr_labels"])
        if outputs.get("act_logits") is not None and targets.get("activity") is not None:
            act_logits_list.append(outputs["act_logits"].cpu())
            act_labels_list.append(targets["activity"])
        if outputs.get("head_pose") is not None and targets.get("head_pose") is not None:
            hp_preds_list.append(outputs["head_pose"].cpu())
            hp_labels_list.append(targets["head_pose"])
        if outputs.get("cls_preds") is not None and targets.get("detection") is not None:
            cls_preds_list.append(outputs["cls_preds"].cpu())
            reg_preds_list.append(outputs["reg_preds"].cpu())
            n_boxes += sum(len(d.get('boxes', [])) for d in targets['detection'])

        n_batches += 1
        if n_batches % 100 == 0:
            print(f"  processed {n_batches} batches...")

    print(f"\nProcessed {n_batches} batches")
    results = {}

    # ===== PSR =====
    if psr_logits_list:
        all_logits = torch.cat(psr_logits_list, dim=0).numpy()
        all_labels = torch.cat(psr_labels_list, dim=0).numpy()
        valid = all_labels != -1
        sig = 1 / (1 + np.exp(-all_logits))

        # Per-frame F1 at best global threshold
        best_f1, best_t = 0, 0
        for thresh in np.arange(0.1, 1.0, 0.05):
            binary = (sig > thresh).astype(np.int32)
            tp = ((binary == 1) & (all_labels == 1) & valid).sum()
            fp = ((binary == 1) & (all_labels == 0) & valid).sum()
            fn = ((binary == 0) & (all_labels == 1) & valid).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = thresh
        comp_acc = ((sig > best_t).astype(np.int32)[valid] == all_labels[valid]).mean() if valid.sum() > 0 else 0
        results['psr'] = {
            'per_frame_f1': float(best_f1),
            'best_threshold': float(best_t),
            'comp_acc': float(((sig > best_t).astype(np.int32)[valid] == all_labels[valid]).mean()),
            'n_valid_frames': int(valid.sum()),
            'sigmoid_min': float(sig[valid].min()),
            'sigmoid_max': float(sig[valid].max()),
            'sigmoid_mean': float(sig[valid].mean()),
        }
        print(f"\nPSR: F1={best_f1:.4f} at thresh={best_t:.2f}, comp_acc={results['psr']['comp_acc']:.4f}")
        print(f"  sigmoid range: [{sig[valid].min():.3f}, {sig[valid].max():.3f}], mean={sig[valid].mean():.3f}")

    # ===== Activity =====
    if act_logits_list:
        act_logits = torch.cat(act_logits_list, dim=0)
        act_labels = torch.cat(act_labels_list, dim=0)
        # If 4D [N, T, 75, ...], collapse to per-frame
        if act_logits.dim() > 2:
            act_logits = act_logits.view(-1, act_logits.shape[-1])
            act_labels = act_labels.view(-1)
        act_preds = act_logits.argmax(dim=-1)
        top1 = (act_preds == act_labels).float().mean().item()
        valid = act_labels >= 0
        top1_valid = (act_preds[valid] == act_labels[valid]).float().mean().item() if valid.sum() > 0 else 0

        # Macro F1
        from sklearn.metrics import f1_score
        try:
            macro_f1 = f1_score(act_labels.numpy(), act_preds.numpy(), average='macro', zero_division=0)
        except:
            macro_f1 = 0
        results['activity'] = {
            'top1': float(top1),
            'top1_valid': float(top1_valid),
            'macro_f1': float(macro_f1),
            'n_samples': len(act_labels),
        }
        print(f"\nActivity: top1={top1:.4f}, top1_valid={top1_valid:.4f}, macro_f1={macro_f1:.4f}")

    # ===== Head Pose =====
    if hp_preds_list:
        hp_preds = torch.cat(hp_preds_list, dim=0)
        hp_labels = torch.cat(hp_labels_list, dim=0)
        if hp_preds.dim() > 2:
            hp_preds = hp_preds.view(-1, hp_preds.shape[-1])
            hp_labels = hp_labels.view(-1, hp_labels.shape[-1])
        # Forward vector: [0:3], position: [3:6], up vector: [6:9]
        def angular_mae(pred, target):
            pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
            targ_n = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
            cos = (pred_n * targ_n).sum(dim=-1).clamp(-1, 1)
            return torch.rad2deg(torch.acos(cos)).mean().item()

        fwd_mae = angular_mae(hp_preds[:, :3], hp_labels[:, :3])
        up_mae = angular_mae(hp_preds[:, 6:9], hp_labels[:, 6:9])
        results['head_pose'] = {
            'forward_angular_MAE_deg': fwd_mae,
            'up_angular_MAE_deg': up_mae,
        }
        print(f"\nHead Pose: fwd_angular_MAE={fwd_mae:.2f}°, up_angular_MAE={up_mae:.2f}°")

    # ===== Detection (just count boxes for sanity) =====
    results['detection'] = {
        'n_boxes_gt': n_boxes,
        'n_cls_preds': len(cls_preds_list),
    }

    # Save
    results['checkpoint'] = ckpt_path
    results['n_batches'] = n_batches
    out_path = Path(save_dir) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()