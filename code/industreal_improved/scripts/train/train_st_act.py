#!/usr/bin/env python3
"""
train_st_act.py — ST-Act MViTv2-S 75-class single-task baseline (175 §6 row 2).

The CRITICAL control per 175 §6 row 2. Same MViTv2-S backbone as MTL — if MTL beats
this on test split, the gain is attributable to MTL itself (not bigger model).

Architecture:
  - Backbone: MViTv2-S (Kinetics-400 pretrained, 34.5M, frozen then unfrozen)
  - Head: LayerNorm(768) → Linear(768, 75)
  - Loss: CrossEntropyLoss (label_smoothing=0.1)
  - Data: single-frame, expanded to T=16 inside the backbone for compatibility

Usage:
    python scripts/train_st_act.py --epochs 30
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

# Path
_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C

# Force 75-class for SOTA comparison (175 §7.2)
C.ACT_CLASS_GROUPING = "none"
C.NUM_ACT_OUTPUTS = 75
# RAM cache from env (avoids fork OOM with workers)
C.RAM_CACHE_MAX_IMAGES = int(os.environ.get("RAM_CACHE_MAX_IMAGES", 8000))

from src.data.industreal_dataset import IndustRealMultiTaskDataset

logger = logging.getLogger("train_st_act")

OUTPUT_ROOT = _CODE_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "st_act_run"
Kinetics_MEAN = torch.tensor([0.45, 0.45, 0.45])
Kinetics_STD = torch.tensor([0.225, 0.225, 0.225])


class STActMViT(nn.Module):
    """Single-task MViTv2-S 75-class activity model.

    Same backbone as MTL-MViT (175 §6 row 2). MViTv2-S requires T=16 frames;
    we expand a single frame to T=16 inside forward (matches the original code in
    src/models/model.py:177-179).
    """

    def __init__(self, num_classes: int = 75, freeze_backbone: bool = False):
        super().__init__()
        backbone = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        feat_dim = backbone.head[1].in_features  # 768
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.norm = nn.LayerNorm(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone frozen (ablation).")
        else:
            logger.info("Backbone trainable (end-to-end fine-tune).")
        self._init_head()

    def _init_head(self):
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] (single frame). Expand to T=16 for MViTv2-S.
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, 3, 1, H, W]
            x = x.expand(-1, -1, 16, -1, -1).contiguous()  # [B, 3, 16, H, W]
        features = self.backbone(x)  # [B, 768]
        return self.classifier(self.norm(features))


def collate_act(batch):
    """Stacks single-frame images + activity labels. [B, 3, 224, 224]."""
    images = torch.stack([b["images"]["rgb"] for b in batch], dim=0)  # [B, 3, 224, 224]
    labels = torch.tensor(
        [
            int(b["action_label"])
            if hasattr(b["action_label"], "item")
            else int(b.get("action_label", -1))
            for b in batch
        ],
        dtype=torch.long,
    )
    return images, labels


def normalize_x(x_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 [B, 3, H, W] → Kinetics-normalized float32."""
    x = x_uint8.float().div_(255.0)  # [0, 1]
    mean = Kinetics_MEAN.to(device).view(1, 3, 1, 1)
    std = Kinetics_STD.to(device).view(1, 3, 1, 1)
    return (x - mean) / std


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, max_batches):
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    n_batches = 0
    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x = normalize_x(images, device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_correct += (logits.argmax(-1) == labels).sum().item()
        n_total += labels.size(0)
        n_batches += 1

        if batch_idx % 200 == 0:
            acc = 100.0 * n_correct / max(n_total, 1)
            logger.info(
                "  [ep %d batch %d/%d] loss=%.4f acc=%.2f%% (%.2fs/batch)",
                epoch,
                batch_idx,
                len(loader),
                loss.item(),
                acc,
                (time.time() - t0) / max(batch_idx + 1, 1),
            )

    return total_loss / max(n_batches, 1), n_correct / max(n_total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x = normalize_x(images, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    if not all_logits:
        return {"top1": 0.0, "top5": 0.0, "n": 0}
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    valid = labels >= 0
    p1 = (logits[valid].argmax(-1) == labels[valid]).float().mean().item()
    # top-5
    _, top5 = logits[valid].topk(5, dim=-1)
    p5 = (top5 == labels[valid].unsqueeze(1)).any(dim=1).float().mean().item()
    return {"top1": p1, "top5": p5, "n": int(valid.sum().item())}


def main():
    parser = argparse.ArgumentParser(
        description="ST-Act MViTv2-S 75-class baseline (175 §6 row 2 — same backbone as MTL)"
    )
    parser.add_argument("--plumbing", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Args: %s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(
        "Device: %s, GPU: %s",
        device,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data — single-frame mode for speed
    logger.info("Building ST-Act dataset (single-frame mode for fast I/O)...")
    train_ds = IndustRealMultiTaskDataset(
        split="train",
        img_size=(224, 224),
        augment=True,
        sequence_mode=False,  # SINGLE FRAME per sample — 16x less I/O than sequence mode
    )
    val_ds = IndustRealMultiTaskDataset(
        split="val",
        img_size=(224, 224),
        augment=False,
        sequence_mode=False,
    )
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_act,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_act,
    )

    # Model
    logger.info("Building ST-Act MViTv2-S (Kinetics pretrained)...")
    model = STActMViT(num_classes=75, freeze_backbone=False).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Params: %.1fM total, %.1fM trainable", n_params, n_train)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=-1)

    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
        ]
    )
    warmup_epochs = min(3, args.epochs // 5)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=True)

    # Train
    history = []
    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch <= warmup_epochs and warmup_epochs > 0:
            warmup_sched.step()
        else:
            cosine_sched.step()

        t0 = time.time()
        loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args.max_batches_per_epoch,
        )

        val_metrics = evaluate(model, val_loader, device)
        val_top1 = val_metrics["top1"]
        val_top5 = val_metrics["top5"]
        dt = time.time() - t0

        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_acc": train_acc,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "val_n": val_metrics["n"],
                "time": dt,
            }
        )
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f train_acc=%.4f | val_top1=%.4f val_top5=%.4f (n=%d) | %.1fs",
            epoch,
            args.epochs,
            loss,
            train_acc,
            val_top1,
            val_top5,
            val_metrics["n"],
            dt,
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "config": vars(args),
                },
                output_dir / "best.pth",
            )
            logger.info("  New best: val_top1=%.4f, top5=%.4f", val_top1, val_top5)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "config": vars(args),
            },
            output_dir / "latest.pth",
        )

        if args.plumbing and epoch >= 1:
            break

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {"history": history, "best_top1": best_top1, "config": vars(args)},
            f,
            indent=2,
            default=str,
        )
    logger.info("Training complete. Best val top-1: %.4f (vs WACV MViTv2-S: 0.6525)", best_top1)


if __name__ == "__main__":
    main()
