"""Sequence-mode activity eval — uses 16-frame clips with proper TCN+ViT temporal context."""
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
    parser.add_argument("--save-dir", default="/tmp/eval_activity_seq")
    parser.add_argument("--max-batches", type=int, default=2000)
    parser.add_argument("--seq-length", type=int, default=16)
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
    model._seq_len = args.seq_length
    model = model.cuda().eval()

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
    from torch.utils.data import DataLoader
    val_ds = IndustRealMultiTaskDataset(
        split="val", sequence_mode=True, sequence_length=args.seq_length
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                            collate_fn=collate_fn_sequences, shuffle=False)

    # Per-clip: get act_logits per frame, aggregate via majority vote
    clip_preds = []
    clip_labels = []
    per_frame_preds = []
    per_frame_labels = []

    n_batches = 0
    for batch in val_loader:
        if n_batches >= args.max_batches:
            break
        images, targets = batch  # images [B, T, 3, H, W]
        B, T = images.shape[:2]
        images = images.cuda().float()
        if images.max() > 1.0:
            images = images.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            outputs = model(images)

        act_logits = outputs.get("act_logits")  # [B*T, num_classes]
        act_labels = targets.get("activity")    # [B]
        if act_logits is None or act_labels is None:
            continue
        if n_batches == 0:
            print(f"  DEBUG: act_labels shape={act_labels.shape if hasattr(act_labels, 'shape') else type(act_labels)}, value={act_labels.tolist() if hasattr(act_labels, 'tolist') else act_labels}")
        act_logits = act_logits.view(B, T, -1)  # [B, T, C]
        act_preds = act_logits.argmax(dim=-1)    # [B, T]

        # Clip-level majority vote using per-frame labels from the dataset's annotation cache
        from src.data.industreal_dataset import IndustRealMultiTaskDataset
        metadata_list = targets.get("metadata", [])
        for b in range(B):
            preds_b = act_preds[b]
            # Get per-frame activity labels via the dataset cache
            if b < len(metadata_list):
                meta = metadata_list[b]
                rec_id = meta.get('recording_id', 'unknown')
                frame_nums = meta.get('frame_nums', [])
                if b == 0 and n_batches == 0:
                    print(f"  DEBUG meta keys: {list(meta.keys())}, rec={rec_id}, frame_nums={frame_nums[:5] if len(frame_nums)>0 else 'EMPTY'}")
                cache = val_ds._anno_cache.get(rec_id)
                if cache is not None and hasattr(cache, '_ar_per_frame') and cache._ar_per_frame is not None and len(frame_nums) > 0:
                    per_frame_actions = cache._ar_per_frame[frame_nums]
                    # Check how many frames have labels
                    n_pos = (per_frame_actions >= 0).sum()
                    if n_pos == 0:
                        # Try shifting to find a labeled region
                        ar_total = cache._ar_per_frame
                        labeled_mask = ar_total >= 0
                        if labeled_mask.any():
                            labeled_idx = np.where(labeled_mask)[0]
                            # Pick first labeled window of size T
                            start = labeled_idx[0]
                            frame_nums = list(range(start, min(start + T, len(ar_total))))
                            per_frame_actions = ar_total[frame_nums]
                            meta['frame_nums'] = frame_nums  # update for completeness
                    n_pos = (per_frame_actions >= 0).sum()
                    if n_pos == 0 and b == 0 and n_batches == 0:
                        print(f"  DEBUG {rec_id}: no labeled frames even after skip")
                    # majority over valid (non -1) frames
                    valid_actions = per_frame_actions[per_frame_actions >= 0]
                    if len(valid_actions) > 0:
                        from collections import Counter
                        # Apply the same verb-group remap as training
                        from src import config as C
                        remap_fn = getattr(C, 'remap_activity_label', None)
                        grouped = valid_actions.tolist()
                        if remap_fn is not None and str(getattr(C, 'ACT_CLASS_GROUPING', 'none')).lower() in ('verb', 'hybrid'):
                            grouped = [remap_fn(a) for a in grouped]
                        majority_label = Counter(grouped).most_common(1)[0][0]
                    else:
                        majority_label = -1
                else:
                    majority_label = act_labels[b].item() if act_labels is not None else -1
            else:
                majority_label = act_labels[b].item() if act_labels is not None else -1
            counts = torch.bincount(preds_b, minlength=act_logits.shape[-1])
            clip_pred = counts.argmax().item()
            clip_preds.append(clip_pred)
            clip_labels.append(majority_label)
            per_frame_preds.extend(preds_b.tolist())
            per_frame_labels.extend([majority_label] * T)

        n_batches += 1
        if n_batches % 50 == 0:
            print(f"  processed {n_batches} clips...")

    print(f"\nProcessed {n_batches} clips ({len(clip_preds)} valid clip predictions)")

    # Filter to clips with valid labels
    valid_mask = np.array(clip_labels) >= 0
    if valid_mask.sum() == 0:
        print("No valid clips!")
        return
    valid_clip_preds = np.array(clip_preds)[valid_mask]
    valid_clip_labels = np.array(clip_labels)[valid_mask]

    # Per-clip accuracy
    clip_acc = (valid_clip_preds == valid_clip_labels).mean()
    print(f"\nClip-level Top-1 (majority vote, valid clips): {clip_acc:.4f} ({valid_mask.sum()}/{len(valid_mask)})")

    # Per-frame accuracy
    frame_acc_all = (np.array(per_frame_preds) == np.array(per_frame_labels)).mean()
    per_frame_labels_arr = np.array(per_frame_labels)
    per_frame_preds_arr = np.array(per_frame_preds)
    valid_frame_mask = per_frame_labels_arr >= 0
    frame_acc_valid = (per_frame_preds_arr[valid_frame_mask] == per_frame_labels_arr[valid_frame_mask]).mean() if valid_frame_mask.sum() > 0 else 0
    print(f"Per-frame Top-1 (all): {frame_acc_all:.4f}")
    print(f"Per-frame Top-1 (valid only): {frame_acc_valid:.4f}")

    # Per-class breakdown (clip-level)
    from collections import Counter
    by_class = defaultdict(list)
    for pred, label in zip(clip_preds, clip_labels):
        by_class[label].append(pred == label)
    print(f"\nPer-class Top-1 (n_clips):")
    for label in sorted(by_class.keys()):
        acc = np.mean(by_class[label])
        n = len(by_class[label])
        print(f"  class {label}: acc={acc:.3f} (n={n})")

    out_path = Path(args.save_dir) / "activity_seq.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "n_clips": len(clip_preds),
            "clip_top1": float(clip_acc),
            "per_frame_top1": float(frame_acc),
            "seq_length": args.seq_length,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()