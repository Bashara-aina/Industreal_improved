"""
Activity Linear Probe on Frozen ConvNeXt-Tiny Features.

Trains a single Linear(backbone_dim -> num_classes) layer on GAP-pooled C5
features with the backbone frozen. Answers Opus Q4: does the backbone encode
any action-discriminative signal?

If probe top-1 < 0.05, the backbone is the bottleneck and P1.4/P5.1 are dead
on arrival.

Usage:
    python -m src.evaluation.activity_linear_probe
"""

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
from torch.utils.data import DataLoader

# Project imports (assumes running from project root with PYTHONPATH set)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import ConvNeXtBackbone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("activity_linear_probe")
logger.setLevel(logging.INFO)

# Suppress noisy loggers from imports
for _log_name in ["src.data.industreal_dataset", "src"]:
    logging.getLogger(_log_name).setLevel(logging.WARNING)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize uint8 images to float [0,1] then ImageNet mean/std.
    Matches _prepare_images in training/train.py.
    """
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
        mean = torch.tensor(C.IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(C.IMAGENET_STD, device=images.device, dtype=images.dtype).view(
            1, 3, 1, 1
        )
        images = (images - mean) / std
    return images


def extract_backbone_features(backbone: ConvNeXtBackbone, images: torch.Tensor) -> torch.Tensor:
    """
    Run images through the backbone and return GAP-pooled C5 features.

    Args:
        backbone: ConvNeXtBackbone (eval mode, frozen).
        images: [B, 3, H, W] normalized RGB tensor.

    Returns:
        features: [B, 768] GAP-pooled ConvNeXt-Tiny C5 features.
    """
    images = normalize_images(images)
    c2, c3, c4, c5 = backbone(images)
    # GAP pool C5 to [B, 768]
    features = F.adaptive_avg_pool2d(c5, 1).flatten(1)
    return features


class LinearProbeHead(nn.Module):
    """Single linear layer on top of frozen backbone features."""

    def __init__(self, backbone_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def compute_clip_top1(
    all_rec_ids: list,
    all_frame_nums: list,
    all_labels: list,
    all_preds: list,
    clip_size: int = 16,
) -> float:
    """
    Compute clip-level top-1 accuracy by grouping frames into non-overlapping
    T-frame clips (by recording_id) and taking the majority prediction per clip.

    Args:
        all_rec_ids: List of recording IDs per sample.
        all_frame_nums: List of frame numbers per sample.
        all_labels: List of ground-truth action labels.
        all_preds: List of predicted action labels.
        clip_size: Number of frames per clip (default 16).

    Returns:
        Clip-level top-1 accuracy as a float.
    """
    # Group frames by recording
    rec_groups = defaultdict(list)
    for rec_id, fn, label, pred in zip(all_rec_ids, all_frame_nums, all_labels, all_preds):
        rec_groups[rec_id].append((fn, label, pred))

    correct = 0
    total = 0

    for rec_id, frames in rec_groups.items():
        # Sort by frame number
        frames.sort(key=lambda x: x[0])
        labels = [f[1] for f in frames]
        preds = [f[2] for f in frames]

        # Split into non-overlapping T-frame clips
        for i in range(0, len(labels), clip_size):
            clip_labels = labels[i : i + clip_size]
            clip_preds = preds[i : i + clip_size]

            # Skip clips with no valid labels (-1 sentinel)
            valid_mask = [lbl >= 0 for lbl in clip_labels]
            if not any(valid_mask):
                continue

            # Majority vote of predictions within this clip
            clip_pred_majority = max(set(clip_preds), key=clip_preds.count)
            # Get the most common valid ground-truth in this clip
            valid_labels = [lbl for lbl, m in zip(clip_labels, valid_mask) if m]
            if not valid_labels:
                continue
            # Use mode of valid labels as clip ground truth
            clip_label_mode = max(set(valid_labels), key=valid_labels.count)

            if clip_pred_majority == clip_label_mode:
                correct += 1
            total += 1

    return correct / max(total, 1)


def compute_majority_class_baseline(all_labels: list) -> tuple:
    """
    Compute the majority-class baseline: accuracy of always predicting the
    most frequent class in the labeled set.
    """
    valid_labels = [lbl for lbl in all_labels if lbl >= 0]
    if not valid_labels:
        return 0.0, -1
    counts = np.bincount(valid_labels)
    majority_class = int(counts.argmax())
    majority_count = counts[majority_class]
    baseline = majority_count / len(valid_labels)
    return baseline, majority_class


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    # Checkpoint lives under src/runs/ (config.py sets OUTPUT_ROOT = Path(__file__).parent / 'runs')
    runs_root = project_root / "src" / "runs"
    checkpoint_dir = runs_root / "rf_stages" / "checkpoints"
    checkpoint_path = checkpoint_dir / "best.pth"
    output_json = checkpoint_dir / "activity_linear_probe.json"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # --- Load model and extract backbone ---
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model"]  # OrderedDict of full model state

    # Build the model architecture and load backbone weights
    num_classes = int(getattr(C, "NUM_ACT_OUTPUTS", C.NUM_CLASSES_ACT))
    backbone_dim = 768  # ConvNeXt-Tiny C5 channels
    logger.info(f"Backbone dim: {backbone_dim}, Num classes: {num_classes}")

    # Extract backbone state dict from the full model state
    backbone_state = {}
    for k, v in model_state.items():
        if k.startswith("backbone."):
            # Remove 'backbone.' prefix to match ConvNeXtBackbone keys
            backbone_state[k[len("backbone.") :]] = v

    # Build backbone
    backbone = ConvNeXtBackbone(pretrained=False)
    backbone.load_state_dict(backbone_state, strict=False)
    backbone = backbone.to(device)
    backbone.eval()

    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    logger.info(
        f"Backbone loaded with {sum(p.numel() for p in backbone.parameters()):,} parameters (frozen)"
    )

    # --- Build datasets ---
    logger.info("Loading datasets...")
    # Limit RAM cache to prevent OOM: 500 images (~350KB each = ~175 MB)
    C.RAM_CACHE_MAX_IMAGES = 500
    train_dataset = IndustRealMultiTaskDataset(
        split="train",
        augment=False,  # No augmentation for linear probe
        subset_ratio=1.0,
    )
    val_dataset = IndustRealMultiTaskDataset(
        split="val",
        augment=False,
        subset_ratio=1.0,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Use class-balanced sampler for training
    train_sampler = train_dataset.get_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # --- Build linear probe head ---
    probe = LinearProbeHead(backbone_dim, num_classes)
    probe = probe.to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # --- Compute majority-class baseline on validation set ---
    all_val_labels = []
    for _, targets in val_loader:
        labels = targets["activity"].cpu().numpy()
        all_val_labels.extend(labels.tolist())
    majority_baseline, majority_class = compute_majority_class_baseline(all_val_labels)
    logger.info(f"Majority-class baseline: {majority_baseline:.4f} (class {majority_class})")

    # --- Training loop ---
    num_epochs = 10
    best_val_top1 = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        probe.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            labels = targets["activity"].to(device)

            # Forward through frozen backbone
            with torch.no_grad():
                features = extract_backbone_features(backbone, images)

            # Forward through probe
            logits = probe(features)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Per-frame accuracy (excluding -1 sentinel)
            preds = logits.argmax(dim=1)
            valid_mask = labels >= 0
            if valid_mask.any():
                train_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                train_total += valid_mask.sum().item()

        train_acc = train_correct / max(train_total, 1)
        train_loss = np.mean(train_losses).item()

        # Validation
        probe.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels_epoch = []
        all_val_rec_ids = []
        all_val_frame_nums = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                labels = targets["activity"].to(device)
                metadata = targets["metadata"]

                features = extract_backbone_features(backbone, images)
                logits = probe(features)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                preds = logits.argmax(dim=1)
                valid_mask = labels >= 0
                if valid_mask.any():
                    val_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                    val_total += valid_mask.sum().item()

                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_labels_epoch.extend(labels.cpu().numpy().tolist())
                for m in metadata:
                    all_val_rec_ids.append(m.get("recording_id", ""))
                    all_val_frame_nums.append(m.get("frame_num", 0))

        val_acc = val_correct / max(val_total, 1)
        val_loss = np.mean(val_losses).item()

        # Clip-level top-1
        clip_top1 = compute_clip_top1(
            all_val_rec_ids,
            all_val_frame_nums,
            all_val_labels_epoch,
            all_val_preds,
            clip_size=16,
        )

        logger.info(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Clip Top-1: {clip_top1:.4f} | "
            f"Time: {time.time() - epoch_start:.0f}s"
        )

        if val_acc > best_val_top1:
            best_val_top1 = val_acc
            best_val_clip = clip_top1

        scheduler.step()

    # --- Final results ---
    logger.info("=" * 60)
    logger.info("LINEAR PROBE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Majority-class baseline:     {majority_baseline:.4f}")
    logger.info(f"Best validation per-frame top-1:  {best_val_top1:.4f}")
    logger.info(f"Best validation clip-level top-1: {best_val_clip:.4f}")

    verdict = "BOTTLENECK" if best_val_top1 < 0.05 else "BACKBONE HAS SIGNAL"
    logger.info(f"Verdict: {verdict}")

    # Save results
    results = {
        "model": "ConvNeXt-Tiny linear probe",
        "checkpoint": str(checkpoint_path),
        "backbone_dim": backbone_dim,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "batch_size": 128,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "majority_class_baseline": round(majority_baseline, 6),
        "majority_class": majority_class,
        "best_val_per_frame_top1": round(best_val_top1, 6),
        "best_val_clip_top1": round(best_val_clip, 6),
        "verdict": verdict,
        "note": (
            "Linear probe on frozen ConvNeXt-Tiny C5 (768-dim) GAP-pooled features. "
            "If top-1 < 0.05, backbone is the bottleneck and P1.4/P5.1 dead on arrival."
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_json}")

    return best_val_top1


if __name__ == "__main__":
    main()
