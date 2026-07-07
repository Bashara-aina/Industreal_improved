"""Train ActivityTCN on frozen ConvNeXt features (Opus 141 ACT-ARCH-4 Phase 1).

Pre-extracts C5 features from frozen ConvNeXt-Tiny backbone, then trains
the minimal TCN head on top of clip features. This is the gating experiment
to decide if TCN+ViT is worth the full 2-3 day run.

Usage:
    python -m src.training.train_activity_tcn
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_activity_tcn')


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(images):
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


class FeatureClipDataset(Dataset):
    """Pre-extracted ConvNeXt features arranged as clips."""
    def __init__(self, features, labels, recording_ids, clip_len=16, stride=8):
        self.features = features  # [N, 768]
        self.labels = labels  # [N]
        self.recording_ids = recording_ids  # [N]
        self.clip_len = clip_len
        self.stride = stride
        # Group by recording
        from collections import defaultdict
        rec_to_indices = defaultdict(list)
        for i, rid in enumerate(recording_ids):
            rec_to_indices[rid].append(i)
        # Build clips
        self.clips = []
        for rid, idxs in rec_to_indices.items():
            for start in range(0, len(idxs) - clip_len + 1, stride):
                self.clips.append(idxs[start:start + clip_len])
        logger.info(f"Built {len(self.clips)} clips of length {clip_len}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_idx = self.clips[idx]
        feats = self.features[clip_idx]  # [T, 768]
        labels = self.labels[clip_idx]  # [T]
        return feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--save-dir', default='src/runs/rf_stages/checkpoints/activity_tcn')
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')

    # Load backbone
    logger.info("Loading ConvNeXt-Tiny backbone (frozen)...")
    from src.models.model import ConvNeXtBackbone
    backbone = ConvNeXtBackbone(pretrained=True)
    ckpt = torch.load('src/runs/rf_stages/checkpoints/best.pth', map_location='cpu', weights_only=False)
    bb_state = {k.replace('backbone.', ''): v for k, v in ckpt['model'].items() if k.startswith('backbone.')}
    backbone.load_state_dict(bb_state, strict=False)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Load datasets
    train_ds = IndustRealMultiTaskDataset(split='train', sequence_mode=False)
    val_ds = IndustRealMultiTaskDataset(split='val', sequence_mode=False)

    # Pre-extract features
    logger.info("Pre-extracting train features...")
    train_feats, train_labels, train_recs = extract_features(train_ds, backbone, device)
    logger.info(f"Train features: {train_feats.shape}")

    logger.info("Pre-extracting val features...")
    val_feats, val_labels, val_recs = extract_features(val_ds, backbone, device)
    logger.info(f"Val features: {val_feats.shape}")

    # Build clip datasets
    train_clips = FeatureClipDataset(train_feats, train_labels, train_recs, args.clip_len, args.stride)
    val_clips = FeatureClipDataset(val_feats, val_labels, val_recs, args.clip_len, args.stride)

    train_loader = DataLoader(train_clips, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_clips, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Import TCN
    from src.models.activity_tcn import ActivityTCN
    model = ActivityTCN(in_dim=768, num_classes=69, hidden=args.hidden, levels=args.levels).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Majority baseline
    majority_class = int(np.bincount(train_labels.numpy()).argmax())
    majority_acc = (val_labels == majority_class).float().mean().item()
    logger.info(f"Majority-class baseline: {majority_acc:.4f} (class {majority_class})")

    best_val = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)  # [B, T, num_classes]
            # Per-frame loss
            loss = F.cross_entropy(logits.reshape(-1, 69), labels.reshape(-1), ignore_index=-1)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * feats.shape[0]
            # Per-clip accuracy (majority vote)
            preds = logits.argmax(dim=-1)  # [B, T]
            for b in range(feats.shape[0]):
                # Filter valid
                valid = labels[b] >= 0
                if valid.sum() > 0:
                    clip_pred = preds[b][valid].mode().values.item()
                    clip_true = labels[b][valid].mode().values.item()
                    correct += int(clip_pred == clip_true)
                    total += 1
        train_acc = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)
        scheduler.step()

        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)
                preds = logits.argmax(dim=-1)
                for b in range(feats.shape[0]):
                    valid = labels[b] >= 0
                    if valid.sum() > 0:
                        clip_pred = preds[b][valid].mode().values.item()
                        clip_true = labels[b][valid].mode().values.item()
                        val_correct += int(clip_pred == clip_true)
                        val_total += 1
        val_acc = val_correct / max(val_total, 1)

        logger.info(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_acc={val_acc:.4f} (majority={majority_acc:.4f})")
        if val_acc > best_val:
            best_val = val_acc

    logger.info(f"\n{'='*60}")
    logger.info(f"TCN Probe Results")
    logger.info(f"  Majority baseline: {majority_acc:.4f}")
    logger.info(f"  Best val clip acc: {best_val:.4f}")
    logger.info(f"  vs baseline:       {best_val - majority_acc:+.4f}")
    if best_val > 0.27:
        logger.info(f"  GATING DECISION: PASS — TCN+ViT justified")
    elif best_val > majority_acc:
        logger.info(f"  GATING DECISION: GRAY ZONE — minimal TCN+ViT attempt")
    else:
        logger.info(f"  GATING DECISION: FAIL — cut TCN+ViT")
    logger.info(f"{'='*60}")

    # Save results
    import json
    out = Path(args.save_dir) / 'results.json'
    with open(out, 'w') as f:
        json.dump({
            'clip_len': args.clip_len,
            'stride': args.stride,
            'hidden': args.hidden,
            'levels': args.levels,
            'majority_baseline': majority_acc,
            'best_val_clip_acc': best_val,
            'gating_decision': 'PASS' if best_val > 0.27 else ('GRAY' if best_val > majority_acc else 'FAIL')
        }, f, indent=2)
    logger.info(f"Saved {out}")


def extract_features(dataset, backbone, device):
    """Pre-extract features for all frames."""
    feats_list = []
    labels_list = []
    recs_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            if i % 5000 == 0:
                logger.info(f"  {i}/{len(dataset)}")
            sample = dataset[i]
            image = sample['images']['rgb'].unsqueeze(0).to(device).float()
            image = normalize(image)
            feature = backbone(image)
            if feature.dim() == 4:
                feature = feature.mean(dim=(-2, -1))
            feats_list.append(feature.cpu().squeeze(0))
            label = sample['activity']
            if hasattr(label, 'item'):
                label = label.item()
            labels_list.append(int(label))
            meta = sample.get('metadata', {})
            recs_list.append(meta.get('recording_id', 'unknown'))
    return torch.stack(feats_list), torch.tensor(labels_list), recs_list


if __name__ == '__main__':
    main()