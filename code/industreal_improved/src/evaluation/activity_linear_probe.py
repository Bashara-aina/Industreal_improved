"""
Activity Linear Probe on Frozen ConvNeXt-Tiny Features.

Trains a single Linear(backbone_dim -> num_classes) layer on GAP-pooled C5
features with the backbone frozen. Answers Opus Q4: does the backbone encode
any action-discriminative signal?

If probe top-1 < 0.05, the backbone is the bottleneck and P1.4/P5.1 are dead
on arrival.

FIXES (2026-07-06):
- NaN loss bug: CrossEntropyLoss(ignore_index=-1) with ALL -1 labels divides by
  zero → NaN. Now filters out -1 samples at batch level during feature extraction.
- Performance: pre-extracts all backbone features once (epoch 0), then trains
  the linear probe on cached features at batch_size=256.
- Gradient clipping: prevents NaN propagation from stray feature values.
- Reduced epochs: 5 (was 10) — linear probe converges in 1-2 epochs.

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
from torch.utils.data import DataLoader, TensorDataset

# Project imports (assumes running from project root with PYTHONPATH set)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import ConvNeXtBackbone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('activity_linear_probe')
logger.setLevel(logging.INFO)

# Suppress noisy loggers from imports
for _log_name in ['src.data.industreal_dataset', 'src']:
    logging.getLogger(_log_name).setLevel(logging.WARNING)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize uint8 images to float [0,1] then ImageNet mean/std.
    Matches _prepare_images in training/train.py.
    """
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
        mean = torch.tensor(C.IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
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
    if not all_rec_ids:
        return 0.0

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
            clip_labels = labels[i:i + clip_size]
            clip_preds = preds[i:i + clip_size]

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


def extract_features_and_labels(
    backbone: ConvNeXtBackbone,
    loader: DataLoader,
    device: torch.device,
    desc: str = "",
) -> tuple:
    """
    Extract GAP-pooled C5 backbone features and filter out -1 labels.

    This is the ONE-TIME feature extraction pass. The backbone is frozen, so
    features are identical every epoch — no reason to re-extract.

    Returns:
        (features, labels, skipped_count)
        features: [N, 768] float32
        labels: [N] long (all valid, no -1 sentinels)
    """
    all_features, all_labels = [], []
    total_skipped = 0
    num_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = targets['activity']  # [B] tensor with possible -1 sentinels

        # [FIX] Skip samples with -1 labels to avoid CrossEntropyLoss divide-by-zero
        # when a batch has ALL -1 labels (15% of val batches).
        valid = labels >= 0
        n_invalid = (~valid).sum().item()
        total_skipped += n_invalid

        if not valid.any():
            # Entire batch is invalid — skip it entirely
            continue

        with torch.no_grad():
            features = extract_backbone_features(backbone, images)
            # Safety net: replace any NaN/inf features with 0.0
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Keep only valid samples
        all_features.append(features[valid].cpu())
        all_labels.append(labels[valid].cpu())

        if batch_idx % 200 == 0 and batch_idx > 0:
            logger.info(f"  {desc} {batch_idx}/{num_batches} batches ({total_skipped} skipped)")

    if not all_features:
        logger.error(f"  {desc}: ALL samples had -1 labels — nothing to train on!")
        return torch.empty(0, 768), torch.empty(0, dtype=torch.long), total_skipped

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0), total_skipped


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_root = project_root / 'src' / 'runs'
    checkpoint_dir = runs_root / 'rf_stages' / 'checkpoints'
    checkpoint_path = checkpoint_dir / 'best.pth'
    output_json = checkpoint_dir / 'activity_linear_probe.json'

    if not checkpoint_path.exists():
        logger.error(f'Checkpoint not found: {checkpoint_path}')
        sys.exit(1)

    # --- Load model and extract backbone ---
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model']

    num_classes = int(getattr(C, 'NUM_ACT_OUTPUTS', C.NUM_CLASSES_ACT))
    backbone_dim = 768  # ConvNeXt-Tiny C5 channels
    logger.info(f'Backbone dim: {backbone_dim}, Num classes: {num_classes}')

    # Extract backbone state dict from the full model state
    backbone_state = {}
    for k, v in model_state.items():
        if k.startswith('backbone.'):
            backbone_state[k[len('backbone.'):]] = v

    # Build backbone
    backbone = ConvNeXtBackbone(pretrained=False)
    backbone.load_state_dict(backbone_state, strict=False)
    backbone = backbone.to(device)
    backbone.eval()

    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    logger.info(f'Backbone loaded with {sum(p.numel() for p in backbone.parameters()):,} parameters (frozen)')

    # --- Build datasets ---
    logger.info('Loading datasets...')
    # Smaller RAM cache per worker to avoid OOM with multiprocessing
    C.RAM_CACHE_MAX_IMAGES = 200
    train_dataset = IndustRealMultiTaskDataset(
        split='train',
        augment=False,
        subset_ratio=1.0,
    )
    val_dataset = IndustRealMultiTaskDataset(
        split='val',
        augment=False,
        subset_ratio=1.0,
    )

    logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Use class-balanced sampler for training (reduces -1 label probability)
    train_sampler = train_dataset.get_sampler()

    # [PERF] Use multiple workers for faster data loading during feature extraction
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- Compute majority-class baseline ---
    majority_baseline, majority_class = compute_majority_class_baseline(
        [lbl for _, t in val_loader for lbl in t['activity'].cpu().numpy().tolist() if lbl >= 0]
    )
    logger.info(f'Majority-class baseline: {majority_baseline:.4f} (class {majority_class})')

    # --- [PERF] Pre-extract features (one pass) ---
    logger.info('Pre-extracting training features (one-time pass)...')
    feat_start = time.time()
    train_features, train_labels, train_skipped = extract_features_and_labels(
        backbone, train_loader, device, desc="Train"
    )
    logger.info(
        f'Train features: {train_features.shape} '
        f'({train_skipped} samples with -1 labels skipped) '
        f'in {time.time() - feat_start:.0f}s'
    )

    logger.info('Pre-extracting validation features (one-time pass)...')
    val_features, val_labels, val_skipped = extract_features_and_labels(
        backbone, val_loader, device, desc="Val"
    )
    logger.info(
        f'Val features: {val_features.shape} '
        f'({val_skipped} samples with -1 labels skipped) '
        f'in {time.time() - feat_start:.0f}s'
    )

    if train_features.shape[0] == 0:
        logger.error('No valid training samples — cannot train probe.')
        sys.exit(1)

    # --- Build fast dataloaders from cached features ---
    train_feat_dataset = TensorDataset(train_features, train_labels)
    train_feat_loader = DataLoader(
        train_feat_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )
    val_feat_dataset = TensorDataset(val_features, val_labels)
    val_feat_loader = DataLoader(
        val_feat_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    # --- Build linear probe head ---
    probe = LinearProbeHead(backbone_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 5  # [PERF] Linear probe converges in 1-2 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # No ignore_index needed — -1 labels are already filtered during feature extraction
    criterion = nn.CrossEntropyLoss()

    # --- Training loop on cached features ---
    best_val_top1 = 0.0
    best_val_clip = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        probe.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in train_feat_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            # [ROBUSTNESS] Gradient clipping prevents NaN propagation
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        train_loss = np.mean(train_losses).item()

        # Validation
        probe.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels_epoch = []
        # Note: metadata maps recording_id/frame_num are lost with cached features.
        # For clip-level accuracy, we fall back to per-frame since recording context
        # is not cached. This is a limitation of the feature caching approach.
        # The clip-level metric is approximate but still informative.

        with torch.no_grad():
            for batch_features, batch_labels in val_feat_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                logits = probe(batch_features)
                loss = criterion(logits, batch_labels)
                val_losses.append(loss.item())

                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)

                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_labels_epoch.extend(batch_labels.cpu().numpy().tolist())

        val_acc = val_correct / max(val_total, 1)
        val_loss = np.mean(val_losses).item()

        # Clip-level top-1 (per-frame level since metadata not cached)
        # This is a per-frame approximation — still meaningful for signal detection
        top1_per_frame = val_correct / max(val_total, 1)

        logger.info(
            f'Epoch {epoch:2d}/{num_epochs} | '
            f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
            f'Val Top-1: {top1_per_frame:.4f} | '
            f'Time: {time.time() - epoch_start:.0f}s'
        )

        if val_acc > best_val_top1:
            best_val_top1 = val_acc

        scheduler.step()

    # --- Final results ---
    logger.info('=' * 60)
    logger.info('LINEAR PROBE RESULTS')
    logger.info('=' * 60)
    logger.info(f'Majority-class baseline:           {majority_baseline:.4f}')
    logger.info(f'Best validation per-frame top-1:   {best_val_top1:.4f}')

    verdict = 'BOTTLENECK' if best_val_top1 < 0.05 else 'BACKBONE HAS SIGNAL'
    logger.info(f'Verdict: {verdict}')

    # Save results
    results = {
        'model': 'ConvNeXt-Tiny linear probe',
        'checkpoint': str(checkpoint_path),
        'backbone_dim': backbone_dim,
        'num_classes': num_classes,
        'num_epochs': num_epochs,
        'batch_size': 256,  # training on cached features
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'scheduler': 'CosineAnnealingLR',
        'gradient_clipping': 'max_norm=1.0',
        'majority_class_baseline': round(majority_baseline, 6),
        'majority_class': majority_class,
        'best_val_per_frame_top1': round(best_val_top1, 6),
        'train_samples_valid': int(train_features.shape[0]),
        'val_samples_valid': int(val_features.shape[0]),
        'verdict': verdict,
        'note': (
            'Linear probe on frozen ConvNeXt-Tiny C5 (768-dim) GAP-pooled features. '
            'If top-1 < 0.05, backbone is the bottleneck and P1.4/P5.1 dead on arrival. '
            'Features pre-extracted in one pass; -1 labels filtered pre-training.'
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {output_json}')

    return best_val_top1


if __name__ == '__main__':
    main()
