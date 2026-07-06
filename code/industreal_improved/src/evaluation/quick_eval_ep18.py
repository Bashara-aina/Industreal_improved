"""Quick eval on epoch_18.pth to confirm good PSR.

Bypasses the broken evaluate.py main() and runs inference directly.
Reports per-task metrics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

CKPT = "src/runs/rf_stages/checkpoints/epoch_18.pth"


def main():
    print(f"Loading {CKPT}...")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
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
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from torch.utils.data import DataLoader
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                            collate_fn=collate_fn, shuffle=False)

    # Load cached data if available, else collect
    cache = Path("src/runs/rf_stages/checkpoints/psr_data_cache.pt")
    if cache.exists():
        print(f"Loading cached data from {cache}...")
        data = torch.load(cache, weights_only=False)
        all_logits = np.concatenate([data['rec_logits'][r].numpy() for r in data['rec_logits']], axis=0)
        all_labels = np.concatenate([data['rec_labels'][r].numpy() for r in data['rec_labels']], axis=0)
        print(f"Loaded {len(all_logits)} cached frames")
    else:
        # Collect fresh
        from collections import defaultdict
        rec_logits = defaultdict(list)
        rec_labels = defaultdict(list)

        n = 0
        for batch in val_loader:
            images, targets = batch
            images = images.cuda().float()
            if images.max() > 1.0:
                images = images.div_(255.0)
            mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
            images = (images - mean) / std
            with torch.no_grad():
                outputs = model(images)
            psr_logits = outputs.get("psr_logits")
            psr_labels = targets.get("psr_labels")
            if psr_logits is None or psr_labels is None:
                continue
            rec_logits[targets['metadata'][0].get('recording_id', 'unknown')].append(psr_logits.cpu())
            rec_labels[targets['metadata'][0].get('recording_id', 'unknown')].append(psr_labels.cpu())
            n += 1
            if n % 1000 == 0:
                print(f"  processed {n}/{len(val_ds)}")
        all_logits = np.concatenate([torch.cat(r, 0).numpy() for r in rec_logits.values()], axis=0)
        all_labels = np.concatenate([torch.cat(r, 0).numpy() for r in rec_labels.values()], axis=0)
        print(f"Collected {len(all_logits)} fresh frames")

    # Compute per-task metrics
    valid = all_labels != -1
    sig = 1 / (1 + np.exp(-all_logits))

    # Per-frame F1 at various thresholds
    print("\n--- Per-frame F1 at thresholds ---")
    print(f"{'thresh':<8} {'F1':<9} {'pos_frac':<9} {'comp_acc':<9}")
    for thresh in [0.1, 0.3, 0.5, 0.65, 0.8, 0.9, 0.95]:
        binary = (sig > thresh).astype(np.int32)
        tp = ((binary == 1) & (all_labels == 1) & valid).sum()
        fp = ((binary == 1) & (all_labels == 0) & valid).sum()
        fn = ((binary == 0) & (all_labels == 1) & valid).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        comp_acc = (binary[valid] == all_labels[valid]).mean()
        pos_frac = binary[valid].mean()
        print(f"{thresh:<8.2f} {f1:<9.4f} {pos_frac:<9.4f} {comp_acc:<9.4f}")

    # Per-component
    print("\n--- Per-component best F1 (per-frame) ---")
    for c in range(11):
        valid_c = all_labels[:, c] != -1
        c_sig = sig[:, c]
        c_labels = all_labels[:, c]
        gt_pos = c_labels[valid_c].mean() if valid_c.sum() > 0 else 0
        best_f1, best_t = 0, 0
        for thresh in np.arange(0.05, 1.0, 0.05):
            binary = (c_sig > thresh).astype(np.int32)
            tp = ((binary == 1) & (c_labels == 1) & valid_c).sum()
            fp = ((binary == 1) & (c_labels == 0) & valid_c).sum()
            fn = ((binary == 0) & (c_labels == 1) & valid_c).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = thresh
        print(f"  comp={c}: best_thresh={best_t:.2f}, F1={best_f1:.4f}, gt_pos={gt_pos:.3f}")


if __name__ == "__main__":
    main()