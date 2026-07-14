"""
Activity Temporal Probe (CPU, OOM-Safe).

Forms 16-frame clips (stride=8) from GAP-pooled C5 features,
mean-pools features per clip, and trains a linear classifier.

Per Opus Q3/141 ACT-ARCH-2: determines whether temporal aggregation
amplifies the per-frame signal (0.2169) above the majority baseline (0.2217).

OOM safeguard:
  - CPU-only (no GPU needed)
  - batch_size=4 during feature extraction to keep memory low
  - Features are 768-dim vectors (tiny memory for clip formation + training)

Usage:
    python -u src/evaluation/activity_temporal_probe_cpu.py \
        --save-dir src/runs/rf_stages/checkpoints/activity_temporal_probe_cpu
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import ConvNeXtBackbone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("activity_temporal_probe_cpu")
logger.setLevel(logging.INFO)

for _log_name in ["src.data.industreal_dataset", "src"]:
    logging.getLogger(_log_name).setLevel(logging.WARNING)


def extract_backbone_features(backbone: ConvNeXtBackbone, images: torch.Tensor) -> torch.Tensor:
    """Run images through frozen backbone, return [B, 768] GAP-pooled C5.

    Optimized for CPU: resize 720x1280 -> 224x224 first, then normalize.
    """
    # float -> resize with area mode (best for downsampling) -> normalize
    if images.dtype == torch.uint8:
        images = images.float()
    if images.shape[-1] != 224 or images.shape[-2] != 224:
        images = F.interpolate(images, size=(224, 224), mode="area")
    images = images.div_(255.0)
    mean = torch.tensor(C.IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
    images = (images - mean) / std
    c2, c3, c4, c5 = backbone(images)
    features = F.adaptive_avg_pool2d(c5, 1).flatten(1)
    return features


def compute_majority_baseline(all_labels: np.ndarray) -> tuple:
    """Compute majority-class baseline from label array."""
    if len(all_labels) == 0:
        return 0.0, -1
    counts = np.bincount(all_labels.astype(int))
    majority_class = int(counts.argmax())
    baseline = counts[majority_class] / len(all_labels)
    return baseline, majority_class


def extract_features_and_metadata(
    backbone: ConvNeXtBackbone,
    loader: DataLoader,
    desc: str = "",
    max_batches: int = 0,
) -> dict:
    """
    One-pass extraction of GAP-pooled C5 features + metadata.

    Returns dict with keys:
      rec_ids, frame_nums, features, labels
    """
    all_rec_ids = []
    all_frame_nums = []
    all_features = []
    all_labels = []
    total_skipped = 0
    num_batches = len(loader)
    effective_max = max_batches if max_batches > 0 else num_batches

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= effective_max:
            logger.info(f"  {desc}: stopping at batch {batch_idx}/{effective_max} (max_batches)")
            break

        images = batch[0]
        targets = batch[1]
        labels = targets["activity"]
        metadata_list = targets["metadata"]

        valid = labels >= 0
        n_invalid = (~valid).sum().item()
        total_skipped += n_invalid

        if not valid.any():
            continue

        with torch.no_grad():
            features = extract_backbone_features(backbone, images)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

        for m in metadata_list:
            all_rec_ids.append(m.get("recording_id", ""))
            all_frame_nums.append(m.get("frame_num", 0))

        if batch_idx % 50 == 0:
            elapsed = time.time() - _extract_t0
            rate = batch_idx / max(elapsed, 1e-6)
            remaining = (effective_max - batch_idx) / max(rate, 1e-6)
            if batch_idx == 0:
                logger.info(f"  {desc}: starting feature extraction...")
                import sys

                sys.stdout.flush()  # ensure unbuffered
            else:
                logger.info(
                    f"  {desc} batch {batch_idx}/{effective_max} "
                    f"({total_skipped} skipped) "
                    f"[{elapsed:.0f}s, {rate:.1f} batch/s, ~{remaining:.0f}s remaining]"
                )

    if not all_features:
        logger.error(f"  {desc}: no valid features extracted")
        return {
            "rec_ids": [],
            "frame_nums": [],
            "features": torch.empty(0, 768),
            "labels": torch.empty(0, dtype=torch.long),
        }

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    logger.info(
        f"  {desc}: {features.shape[0]} frames, {total_skipped} skipped (-1), "
        f"{len(set(all_rec_ids))} recordings"
    )

    return {
        "rec_ids": all_rec_ids,
        "frame_nums": all_frame_nums,
        "features": features,
        "labels": labels,
    }


_extract_t0 = 0.0  # module-level timer for progress logging


def form_clips(
    rec_ids: list,
    frame_nums: list,
    features: torch.Tensor,
    labels: torch.Tensor,
    clip_size: int = 16,
    stride: int = 8,
) -> tuple:
    """
    Group frames by recording, sort by frame_num, slide window.

    Mean-pool features per clip, mode-label from valid frame labels.
    Returns (clip_features [N_clips, 768], clip_labels [N_clips]).
    """
    # Group by recording
    rec_groups = defaultdict(list)
    for i, rec_id in enumerate(rec_ids):
        rec_groups[rec_id].append((frame_nums[i], features[i].clone(), labels[i].item()))

    clip_feat_list = []
    clip_label_list = []
    per_rec_counts = {}

    for rec_id, frames in rec_groups.items():
        frames.sort(key=lambda x: x[0])
        n_frames = len(frames)

        rec_features = torch.stack([f[1] for f in frames])  # [N, 768]
        rec_labels = torch.tensor([f[2] for f in frames], dtype=torch.long)

        num_clips = 0
        for start in range(0, n_frames - clip_size + 1, stride):
            end = start + clip_size
            clip_feat = rec_features[start:end].mean(dim=0, keepdim=True)
            clip_lbls = rec_labels[start:end]

            valid_mask = clip_lbls >= 0
            if not valid_mask.any():
                continue

            valid_lbls = clip_lbls[valid_mask]
            counts = torch.bincount(valid_lbls)
            clip_label = counts.argmax().item()

            clip_feat_list.append(clip_feat)
            clip_label_list.append(clip_label)
            num_clips += 1

        per_rec_counts[rec_id] = num_clips

    if not clip_feat_list:
        logger.warning("No valid clips formed!")
        return torch.empty(0, 768), torch.empty(0, dtype=torch.long), per_rec_counts

    clip_features = torch.cat(clip_feat_list, dim=0)
    clip_labels = torch.tensor(clip_label_list, dtype=torch.long)
    return clip_features, clip_labels, per_rec_counts


def compute_per_class_metrics(
    all_labels: np.ndarray, all_preds: np.ndarray, num_classes: int
) -> dict:
    """Per-class precision, recall, F1, support."""
    per_class = {}
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        support = (all_labels == c).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        per_class[str(c)] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support),
        }
    return per_class


class LinearProbeClip(nn.Module):
    """Linear classifier on clip-level mean-pooled features."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def main():
    parser = argparse.ArgumentParser(description="Activity temporal probe (CPU, OOM-safe)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="src/runs/rf_stages/checkpoints/activity_temporal_probe_cpu",
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for backbone feature extraction (CPU)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Max batches for extraction (0 = all; use e.g. 2000 for OOM-safe GPU)",
    )
    parser.add_argument(
        "--clip-size",
        type=int,
        default=16,
        help="Frames per clip",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride between clips",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for linear classifier",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    args = parser.parse_args()

    # Use all CPU cores but avoid thread oversubscription
    torch.set_num_threads(min(16, torch.get_num_threads()))

    device = torch.device("cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {args}")
    logger.info(f"Torch threads: {torch.get_num_threads()}")

    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_root = project_root / "src" / "runs"
    checkpoint_dir = runs_root / "rf_stages" / "checkpoints"
    checkpoint_path = checkpoint_dir / "best.pth"
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = project_root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # --- Load backbone ---
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model"]

    num_classes = int(getattr(C, "NUM_ACT_OUTPUTS", C.NUM_CLASSES_ACT))
    backbone_dim = 768
    logger.info(f"Backbone dim: {backbone_dim}, Num classes: {num_classes}")

    backbone_state = {}
    for k, v in model_state.items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone.") :]] = v

    backbone = ConvNeXtBackbone(pretrained=False)
    backbone.load_state_dict(backbone_state, strict=False)
    backbone = backbone.to(device)
    # channels_last memory format gives ~10% CPU speedup on ConvNeXt
    backbone = backbone.to(memory_format=torch.channels_last)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    logger.info(
        f"Backbone loaded with {sum(p.numel() for p in backbone.parameters()):,} "
        f"parameters (frozen)"
    )

    # --- Build datasets ---
    logger.info("Loading datasets...")
    # With num_workers=0, RAM cache directly accelerates data loading.
    # 2000 slots ~ 700MB covers ~25% of max-batches frames for cache-friendly
    # sequential access (val) and some overlap (train with sampler).
    C.RAM_CACHE_MAX_IMAGES = 2000
    train_dataset = IndustRealMultiTaskDataset(
        split="train",
        augment=False,
        subset_ratio=1.0,
    )
    val_dataset = IndustRealMultiTaskDataset(
        split="val",
        augment=False,
        subset_ratio=1.0,
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Use train sampler for class balance
    train_sampler = train_dataset.get_sampler()

    # num_workers=0 avoids DataLoader-multiprocessing hangs on this system.
    # The forward pass (200ms per batch at batch_size=4) is the bottleneck anyway,
    # so parallel data loading provides marginal benefit.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # --- Per-frame majority baseline (O(1) from cached class_counts) ---
    logger.info("Computing per-frame majority baseline...")
    counts = val_dataset.class_counts  # np.bincount of valid activity_ids
    if len(counts) > 0:
        majority_class = int(counts.argmax())
        total_valid = int(counts.sum())
        majority_baseline = float(counts[majority_class] / max(total_valid, 1))
    else:
        majority_baseline, majority_class = 0.0, -1
    logger.info(
        f"Per-frame majority baseline: {majority_baseline:.4f} "
        f"(class {majority_class}, {int(counts.sum())} valid labels)"
    )

    # --- Step 1: Extract features + metadata ---
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting backbone features (CPU)")
    logger.info("=" * 60)

    global _extract_t0
    _extract_t0 = time.time()

    logger.info("Extracting training features...")
    train_data = extract_features_and_metadata(
        backbone,
        train_loader,
        desc="Train",
        max_batches=args.max_batches,
    )
    logger.info(f"Train done: {time.time() - _extract_t0:.0f}s elapsed")

    logger.info("Extracting validation features...")
    val_data = extract_features_and_metadata(
        backbone,
        val_loader,
        desc="Val",
        max_batches=args.max_batches,
    )
    logger.info(f"Val done: {time.time() - _extract_t0:.0f}s elapsed")

    train_n = train_data["features"].shape[0]
    val_n = val_data["features"].shape[0]
    logger.info(f"Extracted: train={train_n}, val={val_n} frames")

    if train_n == 0 or val_n == 0:
        logger.error("No valid features extracted. Exiting.")
        sys.exit(1)

    # --- Step 2: Form clips ---
    logger.info("=" * 60)
    logger.info(f"STEP 2: Forming clips (size={args.clip_size}, stride={args.stride})")
    logger.info("=" * 60)

    train_clip_feat, train_clip_lbl, train_rec_counts = form_clips(
        train_data["rec_ids"],
        train_data["frame_nums"],
        train_data["features"],
        train_data["labels"],
        clip_size=args.clip_size,
        stride=args.stride,
    )
    val_clip_feat, val_clip_lbl, val_rec_counts = form_clips(
        val_data["rec_ids"],
        val_data["frame_nums"],
        val_data["features"],
        val_data["labels"],
        clip_size=args.clip_size,
        stride=args.stride,
    )

    logger.info(f"Train clips: {train_clip_feat.shape[0]}, Val clips: {val_clip_feat.shape[0]}")
    logger.info(f"Val clips/recording: {dict(sorted(val_rec_counts.items()))}")

    if train_clip_feat.shape[0] == 0 or val_clip_feat.shape[0] == 0:
        logger.error("No clips formed. Exiting.")
        sys.exit(1)

    # Clip-level majority baseline
    clip_majority_baseline, clip_majority_class = compute_majority_baseline(val_clip_lbl.numpy())
    logger.info(
        f"Clip-level majority baseline: {clip_majority_baseline:.4f} (class {clip_majority_class})"
    )

    # --- Step 3: Train linear probe on clip features ---
    logger.info("=" * 60)
    logger.info("STEP 3: Training linear probe on clip-level features")
    logger.info("=" * 60)

    train_ds = TensorDataset(train_clip_feat, train_clip_lbl)
    train_loader_clip = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )
    val_ds = TensorDataset(val_clip_feat, val_clip_lbl)
    val_loader_clip = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    probe = LinearProbeClip(backbone_dim, num_classes)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_top1 = 0.0
    best_epoch = -1
    best_val_preds = None
    best_val_labels = None

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        probe.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for feat, lbl in train_loader_clip:
            logits = probe(feat)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == lbl).sum().item()
            train_total += lbl.size(0)

        train_acc = train_correct / max(train_total, 1)
        train_loss = np.mean(train_losses).item()

        # Val
        probe.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for feat, lbl in val_loader_clip:
                logits = probe(feat)
                loss = criterion(logits, lbl)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                val_correct += (preds == lbl).sum().item()
                val_total += lbl.size(0)
                all_preds.extend(preds.numpy().tolist())
                all_labels.extend(lbl.numpy().tolist())

        val_acc = val_correct / max(val_total, 1)
        val_loss = np.mean(val_losses).item()

        if val_acc > best_val_top1:
            best_val_top1 = val_acc
            best_epoch = epoch
            best_val_preds = np.array(all_preds)
            best_val_labels = np.array(all_labels)

        scheduler.step()

        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train L: {train_loss:.4f} A: {train_acc:.4f} | "
            f"Val L: {val_loss:.4f} A: {val_acc:.4f} | "
            f"{time.time() - epoch_start:.0f}s"
        )

    # --- Per-class metrics ---
    logger.info("=" * 60)
    logger.info("FINAL METRICS")
    logger.info("=" * 60)

    per_class_metrics = compute_per_class_metrics(
        best_val_labels,
        best_val_preds,
        num_classes,
    )
    f1_scores = [m["f1"] for m in per_class_metrics.values() if m["support"] > 0]
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    weights = [m["support"] for m in per_class_metrics.values() if m["support"] > 0]
    weighted_f1 = np.average(f1_scores, weights=weights) if f1_scores and sum(weights) > 0 else 0.0

    logger.info(f"Per-frame majority baseline (ref):      {majority_baseline:.4f}")
    logger.info(f"Per-frame linear probe (ref):           0.2169")
    logger.info(f"Clip majority baseline:                 {clip_majority_baseline:.4f}")
    logger.info(f"Best clip mean-pool top-1 (epoch {best_epoch}): {best_val_top1:.4f}")
    logger.info(f"Macro F1:                                {macro_f1:.4f}")
    logger.info(f"Weighted F1:                             {weighted_f1:.4f}")

    # Gate decision per 141 Q3 / ACT-ARCH-2
    if best_val_top1 > 0.27:
        gate = "PASS"
        gn = "Temporal aggregation amplifies signal above 0.27 threshold"
    elif best_val_top1 > 0.22:
        gate = "GRAY"
        gn = "Gray zone (0.22-0.27); attention pooling or TCN indicated"
    else:
        gate = "FAIL"
        gn = "At or below baseline; no temporal signal detected"
    logger.info(f"Gate: {gate} ({best_val_top1:.4f}) — {gn}")

    # --- Save results.json ---
    results = {
        "experiment": "activity_temporal_probe_cpu",
        "reference": "Opus 141 ACT-ARCH-2, Q3",
        "clip_size": args.clip_size,
        "stride": args.stride,
        "backbone_dim": backbone_dim,
        "num_classes": num_classes,
        "num_train_frames": train_n,
        "num_val_frames": val_n,
        "num_train_clips": int(train_clip_feat.shape[0]),
        "num_val_clips": int(val_clip_feat.shape[0]),
        "val_clips_per_recording": {k: int(v) for k, v in sorted(val_rec_counts.items())},
        "train_clips_per_recording": {k: int(v) for k, v in sorted(train_rec_counts.items())},
        "per_frame_majority_baseline": round(majority_baseline, 6),
        "per_frame_linear_probe_reference": 0.216869,
        "clip_level_majority_baseline": round(clip_majority_baseline, 6),
        "best_epoch": best_epoch,
        "best_val_clip_top1": round(best_val_top1, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "comparison_to_per_frame": {
            "per_frame": 0.216869,
            "clip_mean_pool": round(best_val_top1, 6),
            "delta": round(best_val_top1 - 0.216869, 6),
        },
        "comparison_to_majority": {
            "majority_baseline": round(majority_baseline, 6),
            "clip_mean_pool": round(best_val_top1, 6),
            "delta": round(best_val_top1 - majority_baseline, 6),
        },
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "max_batches": args.max_batches if args.max_batches > 0 else "all",
        "per_class_metrics": per_class_metrics,
        "gate": gate,
        "gate_note": gn,
        "note": (
            "Temporal probe on frozen ConvNeXt-Tiny C5 GAP-pooled 768-dim features. "
            f"Clips of {args.clip_size} frames (stride={args.stride}), "
            "features mean-pooled per clip, label = mode of valid frame labels. "
            "Linear classifier on clip features. CPU-only, OOM-safe."
        ),
    }

    results_path = save_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # --- Save per_clip_f1.md ---
    delta_vs_maj = best_val_top1 - majority_baseline
    delta_vs_pf = best_val_top1 - 0.2169
    md_lines = [
        f"# Activity Temporal Probe - Per-Clip Mean-Pool Results\n\n",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"- **Reference:** Opus 141 ACT-ARCH-2\n",
        f"- **Clip size:** {args.clip_size} frames, stride={args.stride}\n",
        f"- **Pooling:** Mean\n\n",
        f"## Summary\n\n",
        f"| Metric | Value |\n",
        f"|--------|-------|\n",
        f"| Per-frame linear probe (reference) | 0.2169 |\n",
        f"| Per-frame majority baseline | {majority_baseline:.4f} |\n",
        f"| Clip majority baseline | {clip_majority_baseline:.4f} |\n",
        f"| **Clip mean-pool top-1** | **{best_val_top1:.4f}** |\n",
        f"| Macro F1 | {macro_f1:.4f} |\n",
        f"| Weighted F1 | {weighted_f1:.4f} |\n",
        f"| Delta vs majority | {delta_vs_maj:.4f} |\n",
        f"| Delta vs per-frame | {delta_vs_pf:.4f} |\n",
        f"| Gating | **{gate}** |\n\n",
        f"## Data\n\n",
        f"- Train: {train_n} frames -> {train_clip_feat.shape[0]} clips\n",
        f"- Val: {val_n} frames -> {val_clip_feat.shape[0]} clips\n",
        f"- Val recordings: {len(val_rec_counts)}\n\n",
        f"## Per-Class F1 (sorted by support)\n\n",
        f"| Class | F1 | Precision | Recall | Support |\n",
        f"|-------|----|-----------|--------|--------|\n",
    ]

    sorted_classes = sorted(
        per_class_metrics.items(),
        key=lambda x: x[1]["support"],
        reverse=True,
    )
    for cls_id, m in sorted_classes:
        if m["support"] > 0:
            try:
                cn = C.ACT_CLASS_NAMES[int(cls_id)]
            except (IndexError, ValueError, AttributeError):
                cn = f"cls_{cls_id}"
            md_lines.append(
                f"| {cn} (id={cls_id}) | {m['f1']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['support']} |\n"
            )

    nonzero = sum(1 for m in per_class_metrics.values() if m["support"] > 0)
    zerof1 = sum(1 for m in per_class_metrics.values() if m["support"] > 0 and m["f1"] == 0)
    md_lines.append(f"\nNon-zero classes: {nonzero}/{num_classes}\n")
    md_lines.append(f"Non-predicted (zero F1 with support): {zerof1}\n")

    md_path = save_dir / "per_clip_f1.md"
    with open(md_path, "w") as f:
        f.writelines(md_lines)
    logger.info(f"Per-class F1 saved to {md_path}")

    # Final one-liner for parsing
    print(
        f"\nVERDICT: val_clip_top1={best_val_top1:.4f} "
        f"maj_baseline={majority_baseline:.4f} "
        f"pf_ref=0.2169 "
        f"delta_maj={delta_vs_maj:.4f} "
        f"delta_pf={delta_vs_pf:.4f} "
        f"macro_f1={macro_f1:.4f} "
        f"gate={gate}"
    )

    return best_val_top1


if __name__ == "__main__":
    main()
