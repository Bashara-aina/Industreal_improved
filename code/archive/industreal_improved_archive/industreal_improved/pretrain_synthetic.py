"""
Pretrain detection head on synthetic data.
This is a critical step before main multi-task training per Doc 01 §A.
Run: 20 epochs, detection-only loss, lr=5e-4, save best by val mAP@0.5.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from config import C
from model import POPWModel
from losses import FocalLoss, GIoULoss


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain detection on synthetic data")
    parser.add_argument("--epochs", type=int, default=C.PRETRAIN_DET_EPOCHS)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_detection_loss(cls_preds, reg_preds, targets, focal_loss, giou_loss):
    """Compute detection loss from anchor-matched predictions."""
    total_cls_loss = 0
    total_reg_loss = 0
    n_samples = 0

    for cls_pred, reg_pred, tgt in zip(cls_preds, reg_preds, targets):
        if tgt['boxes'].size(0) == 0:
            continue

        # Simplified: use focal loss on classification
        # In real impl, would do anchor matching first
        scores, labels = cls_pred[0].max(dim=-1)
        valid_mask = scores > C.DET_CONF_THRESH

        if valid_mask.sum() > 0:
            valid_labels = labels[valid_mask]
            valid_targets = torch.zeros_like(valid_labels)  # placeholder
            cls_loss = focal_loss(
                cls_pred[0][valid_mask].unsqueeze(0),
                valid_targets.unsqueeze(0)
            )
            total_cls_loss += cls_loss

        n_samples += 1

    if n_samples == 0:
        return torch.tensor(0.0, device=cls_preds[0][0].device)

    return (total_cls_loss + total_reg_loss) / n_samples


def train_epoch(model, loader, optimizer, focal_loss, giou_loss, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        images = batch['images'].to(device)
        targets = batch['det_labels']

        # Forward
        outputs = model(images)
        cls_preds = outputs['cls_preds']
        reg_preds = outputs['reg_preds']

        # Loss
        loss = compute_detection_loss(cls_preds, reg_preds, targets, focal_loss, giou_loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0


def validate(model, loader, device):
    """Validate and compute mAP@0.5."""
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            targets = batch['det_labels']

            outputs = model(images)
            cls_preds = outputs['cls_preds']
            reg_preds = outputs['reg_preds']

            # Collect predictions and targets for mAP computation
            # (simplified - would need proper anchor matching and NMS)
            all_predictions.append({
                'cls_preds': cls_preds,
                'reg_preds': reg_preds
            })
            all_ground_truths.append(targets)

    # Simplified mAP estimate
    # In practice, would compute full COCO-style mAP
    return 0.0  # Placeholder


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = POPWModel(config=C)
    model = model.to(device)

    # Freeze all except detection head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.detection_head.parameters():
        param.requires_grad = True
    for param in model.fpn.parameters():
        param.requires_grad = True
    for param in model.backbone.parameters():
        param.requires_grad = True  # Unfreeze backbone for pretrain

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=C.WEIGHT_DECAY
    )

    # Loss
    focal_loss = FocalLoss(alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA)
    giou_loss = GIoULoss()

    # Load data
    # (In practice, would load synthetic IndustReal data)
    print("Loading synthetic data...")

    # Placeholder - would use actual synthetic dataset
    print("WARNING: Using placeholder data. Implement actual synthetic data loading.")
    print("Per Doc 01, use assembly_state_detection_synthetic_data from dataset.")

    # Training loop
    best_val_map = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, None, optimizer, focal_loss, giou_loss, device)
        print(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_map = validate(model, None, device)  # Placeholder
        print(f"  Val mAP@0.5: {val_map:.4f}")

        # Save checkpoint
        checkpoint_path = Path(C.CHECKPOINT_DIR) / "pretrain_synthetic" / f"epoch_{epoch}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'val_map': val_map,
        }, checkpoint_path)

        # Best
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            best_path = Path(C.CHECKPOINT_DIR) / "pretrain_synthetic" / "best.pth"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_map': val_map,
            }, best_path)
            print(f"  New best! Saved to {best_path}")

    print(f"\nPretraining complete. Best val mAP@0.5: {best_val_map:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {Path(C.CHECKPOINT_DIR) / 'pretrain_synthetic' / 'best.pth'}")


if __name__ == "__main__":
    main()