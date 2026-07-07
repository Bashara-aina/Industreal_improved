"""Null-model POS + Edit table: prove POS=0.968 / Edit are fill-forward artifacts.

Three models:
- Ours (epoch_18 best.pth)
- Null 1: predict all zeros
- Null 2: copy previous frame's state

Now extended to ALL 16 recordings (Opus 141 Q24) and also computes Edit
metric per-component Levenshtein distance normalized by sequence length
(Opus 141 Q29).

Outputs:
  full_null_pos.json   — per-recording POS for all three models
  full_null_edit.json  — per-recording Edit for all three models
"""
import json
import gc
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def log(msg):
    """Print with explicit flush so nohup redirects get it immediately."""
    print(msg, flush=True)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def edit_metric(preds, gt):
    """Per-component frame-level error rate (normalised Hamming / T).

    For each of the C binary components, compute the fraction of frames where
    prediction differs from ground truth, then average across components.
    This is the substitution-only edit distance for equal-length sequences,
    computed in O(T*C).  A perfect match gives 0.0; random guessing gives ≈0.5.
    """
    T, C = preds.shape
    if T == 0:
        return float("nan")
    errors = (preds != gt).sum(axis=0)  # shape (C,) — total mismatches per comp
    return float((errors / T).mean())


def pos_metric(preds_sorted, gt_sorted):
    """Pairwise orientation score: fraction of frame-pairs where the
    sign of the predicted transition matches the sign of the GT transition."""
    pred_pairs = preds_sorted[1:] - preds_sorted[:-1]
    gt_pairs = gt_sorted[1:] - gt_sorted[:-1]
    return float((np.sign(pred_pairs) == np.sign(gt_pairs)).mean())


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="src/runs/rf_stages/checkpoints/best.pth")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Max batches to process (0 = all)")
    parser.add_argument("--save-dir", default="src/runs/rf_stages/checkpoints/null_model_pos_extended")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    log(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from src.models.model import POPWMultiTaskModel

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
    max_batches = args.max_batches if args.max_batches > 0 else None
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
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
        if n % 500 == 0:
            log(f"  processed {n} frames...")
            if n % 4000 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    log(f"\nCollected {n} frames across {len(rec_preds)} recordings")

    per_rec_pos = {}
    per_rec_edit = {}
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

        n_frames = int(valid.sum())

        # --- POS ---
        ours_pos = pos_metric(vp, vl)
        # Null 1: predict all zeros
        null1 = np.zeros_like(vp)
        null1_pos = pos_metric(null1, vl)
        # Null 2: copy previous frame's state
        null2 = np.zeros_like(vp)
        null2[1:] = vp[:-1]
        null2_pos = pos_metric(null2, vl)

        per_rec_pos[rec] = {
            "ours": ours_pos,
            "null_all_zeros": null1_pos,
            "null_copy_prev": null2_pos,
            "n_frames": n_frames,
        }

        # --- Edit ---
        ours_edit = edit_metric(vp, vl)
        null1_edit = edit_metric(null1, vl)
        null2_edit = edit_metric(null2, vl)

        per_rec_edit[rec] = {
            "ours": ours_edit,
            "null_all_zeros": null1_edit,
            "null_copy_prev": null2_edit,
            "n_frames": n_frames,
        }

    def mean_k(container, key):
        return float(np.mean([v[key] for v in container.values()]))

    def std_k(container, key):
        return float(np.std([v[key] for v in container.values()]))

    # --- POS summary ---
    pos_summary = {
        "checkpoint": args.checkpoint,
        "n_frames": n,
        "n_recordings": len(per_rec_pos),
        "ours_mean": mean_k(per_rec_pos, "ours"),
        "ours_std": std_k(per_rec_pos, "ours"),
        "null_all_zeros_mean": mean_k(per_rec_pos, "null_all_zeros"),
        "null_all_zeros_std": std_k(per_rec_pos, "null_all_zeros"),
        "null_copy_prev_mean": mean_k(per_rec_pos, "null_copy_prev"),
        "null_copy_prev_std": std_k(per_rec_pos, "null_copy_prev"),
        "interpretation": (
            "If null_copy_prev ≈ ours, POS is a fill-forward artifact. "
            "Drop POS from headline."
        ),
        "per_recording": per_rec_pos,
    }
    pos_out = Path(args.save_dir) / "full_null_pos.json"
    with open(pos_out, "w") as f:
        json.dump(pos_summary, f, indent=2)

    # --- Edit summary ---
    edit_summary = {
        "checkpoint": args.checkpoint,
        "n_frames": n,
        "n_recordings": len(per_rec_edit),
        "ours_mean": mean_k(per_rec_edit, "ours"),
        "ours_std": std_k(per_rec_edit, "ours"),
        "null_all_zeros_mean": mean_k(per_rec_edit, "null_all_zeros"),
        "null_all_zeros_std": std_k(per_rec_edit, "null_all_zeros"),
        "null_copy_prev_mean": mean_k(per_rec_edit, "null_copy_prev"),
        "null_copy_prev_std": std_k(per_rec_edit, "null_copy_prev"),
        "interpretation": (
            "Per-component Levenshtein distance / T. "
            "Lower is better. If null_copy_prev ≈ ours, Edit is also inflated."
        ),
        "per_recording": per_rec_edit,
    }
    edit_out = Path(args.save_dir) / "full_null_edit.json"
    with open(edit_out, "w") as f:
        json.dump(edit_summary, f, indent=2)

    log(f"\nPOS results saved to  {pos_out}")
    log(f"Edit results saved to {edit_out}")
    log("")
    log("=== POS (mean +/- std per recording) ===")
    log(f"  Ours:                {pos_summary['ours_mean']:.4f} +/- {pos_summary['ours_std']:.4f}")
    log(f"  Null all-zeros:      {pos_summary['null_all_zeros_mean']:.4f} +/- {pos_summary['null_all_zeros_std']:.4f}")
    log(f"  Null copy-prev:      {pos_summary['null_copy_prev_mean']:.4f} +/- {pos_summary['null_copy_prev_std']:.4f}")
    log("")
    log("=== Edit (mean +/- std per recording) ===")
    log(f"  Ours:                {edit_summary['ours_mean']:.4f} +/- {edit_summary['ours_std']:.4f}")
    log(f"  Null all-zeros:      {edit_summary['null_all_zeros_mean']:.4f} +/- {edit_summary['null_all_zeros_std']:.4f}")
    log(f"  Null copy-prev:      {edit_summary['null_copy_prev_mean']:.4f} +/- {edit_summary['null_copy_prev_std']:.4f}")


if __name__ == "__main__":
    main()