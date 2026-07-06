"""Null-model POS table: prove POS=0.968 is fill-forward artifact.

Three models:
- Ours (epoch_18 best.pth)
- Null 1: predict all zeros
- Null 2: copy previous frame's state

Output: per-recording POS for all three, summary table for §5.2.1 disclosure.
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
    parser.add_argument("--save-dir", default="src/runs/rf_stages/checkpoints/null_model_pos")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    from src.models.model import POPWMultiTaskModel
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

    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False,
    )

    # Collect per-recording per-frame predictions and labels, in frame order
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
        sig = torch.sigmoid(pl[0]).cpu().numpy()
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

    def pos_metric(preds_sorted, gt_sorted):
        pred_pairs = preds_sorted[1:] - preds_sorted[:-1]
        gt_pairs = gt_sorted[1:] - gt_sorted[:-1]
        return float((np.sign(pred_pairs) == np.sign(gt_pairs)).mean())

    per_rec = {}
    for rec, preds in rec_preds.items():
        gt = rec_labels[rec]
        frames = np.array(rec_frame_nums[rec])
        sort_idx = frames.argsort()
        preds_sorted = np.array(preds)[sort_idx]
        gt_sorted = np.array(gt)[sort_idx]

        # Filter to valid GT frames
        valid = gt_sorted.max(axis=1) >= 0
        if valid.sum() < 2:
            continue
        vp = preds_sorted[valid]
        vl = gt_sorted[valid]

        # 1. Ours
        ours_pos = pos_metric(vp, vl)

        # 2. Null 1: predict all zeros
        null1 = np.zeros_like(vp)
        null1_pos = pos_metric(null1, vl)

        # 3. Null 2: copy previous frame's state
        null2 = np.zeros_like(vp)
        null2[1:] = vp[:-1]
        null2_pos = pos_metric(null2, vl)

        per_rec[rec] = {
            "ours_pos": ours_pos,
            "null_all_zeros_pos": null1_pos,
            "null_copy_prev_pos": null2_pos,
            "n_frames": int(valid.sum()),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "n_frames": n,
        "n_recordings": len(per_rec),
        "ours_pos_mean": float(np.mean([v["ours_pos"] for v in per_rec.values()])),
        "null_all_zeros_pos_mean": float(np.mean([v["null_all_zeros_pos"] for v in per_rec.values()])),
        "null_copy_prev_pos_mean": float(np.mean([v["null_copy_prev_pos"] for v in per_rec.values()])),
        "interpretation": (
            "If null_copy_prev_pos is similar to ours_pos, POS is a fill-forward artifact, not a real metric. "
            "Drop POS from headline. Use per-component F1 (and transition F1 if available) as primary PSR metric."
        ),
        "per_recording": per_rec,
    }
    out = Path(args.save_dir) / "null_model_pos.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"Ours POS (mean):                  {summary['ours_pos_mean']:.4f}")
    print(f"Null all-zeros POS (mean):        {summary['null_all_zeros_pos_mean']:.4f}")
    print(f"Null copy-prev-frame POS (mean):  {summary['null_copy_prev_pos_mean']:.4f}")


if __name__ == "__main__":
    main()