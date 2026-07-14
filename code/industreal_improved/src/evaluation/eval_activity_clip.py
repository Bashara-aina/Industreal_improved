"""Clip-level activity eval — groups frames by recording, builds 16-frame windows."""

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
    parser.add_argument("--save-dir", default="/tmp/activity_clip_eval")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-recordings", type=int, default=16)
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
    model._seq_len = args.clip_length
    model = model.cuda().eval()

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from torch.utils.data import DataLoader

    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False
    )

    rec_preds = defaultdict(list)
    rec_labels = defaultdict(list)
    rec_frame_nums = defaultdict(list)

    n = 0
    save_interval = 5000
    for i, batch in enumerate(val_loader):
        try:
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

            al = outputs.get("act_logits")
            al_lbl = targets.get("activity")
            if al is None or al_lbl is None:
                continue
            pred = al.argmax(dim=-1).cpu().item()
            lbl = al_lbl.cpu().item()
            meta = targets.get("metadata", [])
            rec_id = meta[0].get("recording_id", "unknown")
            frame_num = meta[0].get("frame_num", 0)
            rec_preds[rec_id].append(pred)
            rec_labels[rec_id].append(lbl)
            rec_frame_nums[rec_id].append(frame_num)
            n += 1
        except Exception as e:
            print(f"[SKIP] batch {i}: {e}", flush=True)
            continue
        if n % 2000 == 0:
            print(f"  processed {n} frames (batch {i})...", flush=True)
        if n % save_interval == 0:
            # Save intermediate checkpoint
            ckpt_path = Path(args.save_dir) / f"checkpoint_{n}frames.pkl"
            import pickle

            with open(ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "rec_preds": dict(rec_preds),
                        "rec_labels": dict(rec_labels),
                        "rec_frame_nums": dict(rec_frame_nums),
                        "n": n,
                    },
                    f,
                )
            print(f"  [CHECKPOINT] saved {ckpt_path} ({n} frames)", flush=True)

    print(f"\nCollected {n} frames across {len(rec_preds)} recordings")
    print(f"Frames per recording: {[(k, len(v)) for k, v in rec_preds.items()]}")

    # For each recording, sort by frame_num and create clips
    clip_preds = []
    clip_labels = []
    for rec_id in rec_preds:
        frames = np.array(rec_frame_nums[rec_id])
        sort_idx = frames.argsort()
        preds_sorted = np.array(rec_preds[rec_id])[sort_idx]
        labels_sorted = np.array(rec_labels[rec_id])[sort_idx]

        # Apply verb-group remap if applicable
        from src import config as C

        remap_fn = getattr(C, "remap_activity_label", None)
        if remap_fn is not None and str(getattr(C, "ACT_CLASS_GROUPING", "none")).lower() in (
            "verb",
            "hybrid",
        ):
            labels_sorted = np.array([remap_fn(int(l)) if l >= 0 else l for l in labels_sorted])

        # Build clips with stride
        n_frames = len(preds_sorted)
        clip_count = 0
        for start in range(0, n_frames - args.clip_length + 1, args.stride):
            end = start + args.clip_length
            clip_p = preds_sorted[start:end]
            clip_l = labels_sorted[start:end]
            # Majority vote over labeled frames only
            valid_l = clip_l[clip_l >= 0]
            if len(valid_l) == 0:
                continue
            from collections import Counter

            label_majority = Counter(valid_l.tolist()).most_common(1)[0][0]
            pred_majority = Counter(clip_p.tolist()).most_common(1)[0][0]
            clip_preds.append(pred_majority)
            clip_labels.append(label_majority)
            clip_count += 1

        print(f"  {rec_id}: {n_frames} frames, {clip_count} clips")

    clip_preds = np.array(clip_preds)
    clip_labels = np.array(clip_labels)

    # Filter to valid labels
    valid = clip_labels >= 0
    if valid.sum() == 0:
        print("No valid clips!")
        return
    vp = clip_preds[valid]
    vl = clip_labels[valid]

    clip_acc = (vp == vl).mean()
    print(f"\nClip-level Top-1 (majority vote, valid clips): {clip_acc:.4f}")
    print(f"  Total clips: {len(clip_preds)}, valid: {valid.sum()}")

    # Per-class breakdown
    by_class = defaultdict(list)
    for p, l in zip(vp, vl):
        by_class[int(l)].append(p == l)
    print("\nPer-class:")
    for label in sorted(by_class.keys()):
        acc = np.mean(by_class[label])
        n = len(by_class[label])
        print(f"  class {label}: top1={acc:.3f} (n={n})")

    # Save
    out_path = Path(args.save_dir) / "activity_clip.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "n_clips": int(len(clip_preds)),
                "n_valid_clips": int(valid.sum()),
                "clip_top1": float(clip_acc),
                "clip_length": args.clip_length,
                "stride": args.stride,
                "per_class": {
                    str(k): {"acc": float(np.mean(v)), "n": len(v)} for k, v in by_class.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
