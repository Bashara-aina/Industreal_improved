"""Activity temporal probe: aggregate frozen ConvNeXt features over clips.

Tests whether temporal aggregation of frozen backbone features improves
over the per-frame linear probe (val top-1 = 0.2169).

Per Opus ACT-1 / ACT-6: if probe < 0.05, backbone is the bottleneck.
Our per-frame probe = 0.2169 (vs 0.2217 baseline) - barely above chance.
This script tests if temporal aggregation (mean/max/attention pool over
clips of N frames) can amplify the weak signal.

Pool modes tested:
  - mean: average features across N frames
  - max: element-wise max across N frames
  - concat: concatenate N frames (16 * 768 = 12288-D, with linear projection)

If temporal aggregation helps significantly (>+0.05), TCN+ViT is justified.
If not, backbone is truly the bottleneck.

Usage:
    python -m src.evaluation.activity_temporal_probe
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('activity_temporal_probe')


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


class ClipDataset(Dataset):
    """Stride-based clip dataset for temporal probe."""
    def __init__(self, base_ds, clip_len=16, stride=8):
        self.base = base_ds
        self.clip_len = clip_len
        self.stride = stride
        # Build index of (recording_id, start_frame) pairs
        # The base dataset returns single frames with metadata
        # We use it directly but collect features for groups of clip_len frames
        # Simpler: iterate base dataset, group consecutive frames by recording
        self._build_index()

    def _build_index(self):
        # Walk through base dataset, group frames by recording_id
        # Then form clips of clip_len from each recording
        from collections import defaultdict
        recs = defaultdict(list)
        for i in range(len(self.base)):
            sample = self.base[i]  # returns dict from IndustRealMultiTaskDataset.__getitem__
            meta = sample.get('metadata', {})
            rec_id = meta.get('recording_id', f'unknown_{i}')
            frame_num = meta.get('frame_num', i)
            recs[rec_id].append((i, frame_num))

        self.clips = []
        for rec_id, frames in recs.items():
            frames.sort(key=lambda x: x[1])
            for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                clip = frames[start:start + self.clip_len]
                if len(clip) == self.clip_len:
                    self.clips.append([idx for idx, _ in clip])
        logger.info(f"Built {len(self.clips)} clips of length {self.clip_len}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frames = [self.base[i] for i in self.clips[idx]]
        images = torch.stack([f['images']['rgb'] for f in frames])  # [T, 3, H, W]
        labels = torch.tensor([f['activity'].item() if hasattr(f['activity'], 'item') else int(f['activity'])
                                for f in frames])  # [T]
        return images, labels


def main():
    from src.models.model import ConvNeXtBackbone
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--pool', default='mean', choices=['mean', 'max'])
    parser.add_argument('--save-dir', default='src/runs/rf_stages/checkpoints/activity_temporal_probe')
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load backbone
    logger.info("Loading backbone...")
    backbone = ConvNeXtBackbone(pretrained=True)
    ckpt = torch.load('src/runs/rf_stages/checkpoints/best.pth', map_location='cpu', weights_only=False)
    # Extract backbone weights from the multi-task state dict
    bb_state = {k.replace('backbone.', ''): v for k, v in ckpt['model'].items() if k.startswith('backbone.')}
    backbone.load_state_dict(bb_state, strict=False)
    backbone = backbone.cuda().eval()
    for p in backbone.parameters():
        p.requires_grad = False
    logger.info("Backbone frozen")

    # Load datasets
    train_ds = IndustRealMultiTaskDataset(split='train', sequence_mode=False)
    val_ds = IndustRealMultiTaskDataset(split='val', sequence_mode=False)

    # Build clip datasets
    train_clips = ClipDataset(train_ds, clip_len=args.clip_len, stride=args.stride)
    val_clips = ClipDataset(val_ds, clip_len=args.clip_len, stride=args.stride)

    train_loader = DataLoader(train_clips, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_clips, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Pre-extract features
    logger.info("Pre-extracting train features...")
    t0 = time.time()
    train_feats, train_labels = extract_features(train_loader, backbone)
    logger.info(f"Train features: {train_feats.shape}, {time.time()-t0:.0f}s")

    logger.info("Pre-extracting val features...")
    t0 = time.time()
    val_feats, val_labels = extract_features(val_loader, backbone)
    logger.info(f"Val features: {val_feats.shape}, {time.time()-t0:.0f}s")

    feat_dim = train_feats.shape[-1]
    num_classes = 69

    # Aggregate features over clip dimension
    if args.pool == 'mean':
        train_pool = train_feats.mean(dim=1)  # [N, feat_dim]
        val_pool = val_feats.mean(dim=1)
    elif args.pool == 'max':
        train_pool = train_feats.max(dim=1).values
        val_pool = val_feats.max(dim=1).values
    # Get clip-level labels (majority vote within each clip)
    train_clip_labels = majority_vote(train_labels)
    val_clip_labels = majority_vote(val_labels)

    # Filter -1 labels
    valid_train = train_clip_labels >= 0
    valid_val = val_clip_labels >= 0
    train_pool = train_pool[valid_train]
    train_clip_labels = train_clip_labels[valid_train]
    val_pool = val_pool[valid_val]
    val_clip_labels = val_clip_labels[valid_val]

    logger.info(f"After -1 filter: train={len(train_pool)}, val={len(val_pool)}")

    # Majority baseline
    majority_class = int(np.bincount(train_clip_labels.numpy()).argmax())
    majority_acc = (val_clip_labels == majority_class).float().mean().item()
    logger.info(f"Majority-class baseline: {majority_acc:.4f} (class {majority_class})")

    # Train linear classifier on pooled features
    logger.info(f"Training linear classifier (pool={args.pool}, dim={train_pool.shape[-1]})...")
    classifier = nn.Linear(train_pool.shape[-1], num_classes).cuda()
    optim = torch.optim.Adam(classifier.parameters(), lr=1e-2, weight_decay=1e-4)

    bs = 256
    best_val = 0.0
    for epoch in range(args.epochs):
        # Train
        classifier.train()
        perm = torch.randperm(len(train_pool))
        total_loss = 0
        correct = 0
        for i in range(0, len(train_pool), bs):
            idx = perm[i:i+bs]
            x = train_pool[idx].cuda()
            y = train_clip_labels[idx].cuda()
            logits = classifier(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(x)
            correct += (logits.argmax(-1) == y).sum().item()
        train_acc = correct / len(train_pool)
        train_loss = total_loss / len(train_pool)

        # Val
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_pool.cuda())
            val_loss = F.cross_entropy(val_logits, val_clip_labels.cuda()).item()
            val_acc = (val_logits.argmax(-1) == val_clip_labels.cuda()).float().mean().item()

        logger.info(f"  Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        best_val = max(best_val, val_acc)

    logger.info(f"\n{'='*60}")
    logger.info(f"TEMPORAL PROBE RESULTS (clip_len={args.clip_len}, pool={args.pool})")
    logger.info(f"  Majority baseline: {majority_acc:.4f}")
    logger.info(f"  Best val top-1:    {best_val:.4f}")
    logger.info(f"  Per-frame probe:   0.2169 (from previous run)")
    logger.info(f"  Improvement:       {best_val - 0.2169:+.4f}")
    logger.info(f"  Verdict:           {'TEMPORAL HELPS — TCN+ViT justified' if best_val > 0.27 else 'Backbone is the bottleneck'}")
    logger.info(f"{'='*60}")

    # Save results
    results = {
        'clip_len': args.clip_len,
        'stride': args.stride,
        'pool': args.pool,
        'majority_baseline': majority_acc,
        'val_top1': best_val,
        'per_frame_probe': 0.2169,
        'improvement': best_val - 0.2169,
        'verdict': 'TEMPORAL HELPS — TCN+ViT justified' if best_val > 0.27 else 'Backbone is the bottleneck',
    }
    out = Path(args.save_dir) / f'temporal_probe_{args.pool}_T{args.clip_len}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out}")


def extract_features(loader, backbone):
    """Pre-extract features for all clips."""
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # images: [B, T, 3, H, W]
            B, T = images.shape[:2]
            images = images.cuda().float()
            images = normalize(images)
            images_flat = images.view(B * T, *images.shape[2:])
            features = backbone(images_flat)  # backbone returns [BT, 768]
            # GAP if feature map
            if features.dim() == 4:
                features = features.mean(dim=(-2, -1))
            features = features.view(B, T, -1).cpu()  # [B, T, 768]
            all_feats.append(features)
            all_labels.append(labels)
            if (i + 1) % 50 == 0:
                logger.info(f"  processed {i+1}/{len(loader)} batches")
    return torch.cat(all_feats), torch.cat(all_labels)


def majority_vote(labels):
    """For each clip, take majority class label. labels: [N, T]"""
    from scipy import stats
    return torch.tensor([stats.mode(l.numpy(), keepdims=False).mode for l in labels], dtype=torch.long)


if __name__ == '__main__':
    main()