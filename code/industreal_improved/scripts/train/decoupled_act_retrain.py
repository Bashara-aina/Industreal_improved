#!/usr/bin/env python3
"""decoupled_act_retrain.py — Decoupled activity classifier retrain (doc 207 §4 item 4).

Kang et al. ICLR 2020: after backbone training, freeze the backbone and retrain
only the classifier head with class-balanced sampling. +2-10% on long-tail
datasets, ~2-5 epochs needed, minimal compute.

Usage:
    python scripts/decoupled_act_retrain.py \
        --checkpoint src/runs/st_act/best.pt \
        --epochs 5 \
        --output-dir src/runs/st_act_decoupled

Reference: doc 207 §6 item 4, Kang et al. "Decoupling Representation and
Classifier for Long-Tailed Recognition" (ICLR 2020).
"""

# DEPRECATED: This script uses the legacy MTLMViTModel. Use POPWMultiTaskModel from src/models/model.py instead.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C

C.NUM_ACT_OUTPUTS = 75
C.ACT_CLASS_GROUPING = "none"

from src.data.industreal_dataset import IndustRealMultiTaskDataset
from src.models.mvit_mtl_model import MTLMViTModel


def compute_class_counts(ds, num_classes=75):
    """Compute per-class sample counts from dataset labels."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(ds)):
        try:
            sample = ds[i]
            act = sample.get("action_label")
            if act is not None and act.item() >= 0:
                counts[act.item()] += 1
        except Exception:
            continue
    return counts


def evaluate_activity(model, loader, device):
    """Quick activity accuracy evaluation."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in loader:
            if images.shape[0] == 0:
                continue
            images = images.to(device).float() / 255.0
            mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
            images = (images - mean) / std
            images = images.permute(0, 2, 1, 3, 4)

            act_labels = targets.get("activity")
            if act_labels is None:
                continue
            if isinstance(act_labels, torch.Tensor):
                act_labels = act_labels.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                preds = outputs["activity"].argmax(dim=-1)

            correct += (preds == act_labels).sum().item()
            total += act_labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Decoupled activity classifier retrain")
    parser.add_argument("--checkpoint", required=True, help="Path to trained ST-act checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Retrain epochs (2-5 typically)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Classifier-only LR")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")

    # ── Load model ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MTLMViTModel(num_act_classes=75).to(device)

    ckpt_sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    filtered = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded {len(filtered)}/{len(ckpt_sd)} tensors from checkpoint (epoch {ckpt.get('epoch', '?')})"
    )

    # ── Freeze backbone + all non-activity heads ──────────────────────────
    for name, param in model.named_parameters():
        if "act_head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M total (act_head only)")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = IndustRealMultiTaskDataset(
        split="train",
        img_size=(224, 224),
        augment=False,
        sequence_mode=True,
        sequence_length=16,
    )
    val_ds = IndustRealMultiTaskDataset(
        split="val",
        img_size=(224, 224),
        augment=False,
        sequence_mode=True,
        sequence_length=16,
    )

    # Class-balanced sampling: compute sample weights
    class_counts = compute_class_counts(train_ds)
    sample_weights = np.zeros(len(train_ds), dtype=np.float32)
    for i in range(len(train_ds)):
        try:
            sample = train_ds[i]
            act = sample.get("action_label")
            if act is not None and act.item() >= 0:
                cls = act.item()
                sample_weights[i] = 1.0 / max(class_counts[cls], 1)
        except Exception:
            sample_weights[i] = 1.0 / len(train_ds)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=8000 * args.batch_size,
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")
    print(f"Class-balanced sampling enabled (inverse frequency)")

    # ── Optimizer (classifier only) ──────────────────────────────────────
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Retrain loop ─────────────────────────────────────────────────────
    best_acc = 0.0
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= 4000:  # 4000 batches per epoch
                break
            if images.shape[0] == 0:
                continue

            images = images.to(device).float() / 255.0
            mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
            images = (images - mean) / std
            images = images.permute(0, 2, 1, 3, 4)

            act_labels = targets.get("activity")
            if act_labels is None:
                continue
            if isinstance(act_labels, torch.Tensor):
                act_labels = act_labels.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = F.cross_entropy(outputs["activity"], act_labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx % 500 == 0:
                print(f"  Epoch {epoch}/{args.epochs} batch {batch_idx}: loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()

        # Eval
        acc = evaluate_activity(model, val_loader, device)
        print(f"  Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "method": "decoupled_retrain",
                    "val_acc": acc,
                },
                output_dir / "best.pt",
            )
            print(f"  New best: {acc:.4f}")

    print(f"\nDecoupled retrain complete. Best acc: {best_acc:.4f}")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"best_acc": best_acc, "epochs": args.epochs, "lr": args.lr}, f, indent=2)


if __name__ == "__main__":
    main()
