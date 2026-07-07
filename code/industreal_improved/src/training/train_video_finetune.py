#!/usr/bin/env python3
"""
MViTv2-S fine-tuning for activity recognition (Opus 144).

Frozen MViTv2-S probe achieved 0.3810 on 69 classes (above the 0.30 threshold),
justifying full fine-tuning. This script trains the video backbone end-to-end
on the IndustReal activity recognition task with a single MLP head.

Architecture
============
  Input:  [B, 3, T, 224, 224]  (T=16 frames, Kinetics-400 normalization)
  Backbone: MViTv2-S (Kinetics-400 pretrained, 34.5M params)
  Head:    MLP(768 -> 512 -> ReLU -> Dropout -> NUM_ACT_OUTPUTS)
  Loss:    CrossEntropyLoss

Training Stages
===============
  Stage 1 (epochs 1-3):  Backbone frozen, head only
  Stage 2 (epochs 4-20): Backbone unfrozen (last stage), head + backbone train

Memory Budget (batch=2, T=16, FP16 with gradient checkpointing)
========================================
  Backbone forward:   ~2.8 GB
  Head + overhead:    ~0.2 GB
  Total (checkpoint): ~3.5 GB
  Fits RTX 3060 12 GB

Usage
=====
  python -m src.training.train_video_finetune \
      --backbone mvit_v2_s \
      --batch-size 2 \
      --epochs 20 \
      --lr 5e-5 \
      --backbone-lr 1e-5

Author: Opus 144
"""

import argparse
import gc
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_video_finetune")

# =========================================================================
# MViTv2-S transforms (Kinetics-400 preprocessing)
# =========================================================================
_MVIT_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
_MVIT_STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
_MVIT_RESIZE = 256
_MVIT_CROP = 224


def _mvit_transform(img: Image.Image) -> torch.Tensor:
    """Apply MViTv2-S preprocessing: resize short side->256, center crop 224, normalize."""
    img = TF.resize(img, [_MVIT_RESIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
    img = TF.center_crop(img, [_MVIT_CROP, _MVIT_CROP])
    img = TF.to_tensor(img)
    img = (img - _MVIT_MEAN) / _MVIT_STD
    return img


# =========================================================================
# Remap 75 -> 69 class groups
# =========================================================================
_REMAP_PATH = _PROJECT_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "act_remap_75_to_69.json"


def _load_remap() -> dict:
    """Load 75-to-69 class remap from rf_stages checkpoint."""
    if _REMAP_PATH.exists():
        return json.loads(_REMAP_PATH.read_text())
    logger.warning("Remap table not found at %s; defaulting to identity mapping", _REMAP_PATH)
    return {"id_to_group": list(range(75)), "group_names": [str(i) for i in range(69)]}


# =========================================================================
# Clip dataset (adapted from activity_mvit_probe.py)
# =========================================================================
class MViTFinetuneDataset(Dataset):
    """16-frame clip dataset for MViTv2-S fine-tuning.

    Generates overlapping clips from action segments in the IndustReal dataset.
    Uses on-demand JPEG loading (kernel page cache handles the I/O).
    """

    def __init__(
        self,
        split: str,
        recordings_root: str | Path,
        clip_len: int = 16,
        stride: int = 8,
    ):
        self.recordings_root = Path(recordings_root)
        self.split = split
        self.clip_len = clip_len
        self.stride = stride
        self.id_to_group = _load_remap()["id_to_group"]

        self.clips: list[tuple[str, int, int]] = []
        self._build_index()

        logger.info(
            "[MViTFinetuneDataset] split=%s, %d clips, clip_len=%d, stride=%d",
            split, len(self.clips), clip_len, stride,
        )

    def _build_index(self) -> None:
        """Scan AR_labels.csv and build clip index."""
        split_dir = self.recordings_root / self.split
        if not split_dir.exists():
            logger.warning("Split directory does not exist: %s", split_dir)
            return

        for rec_dir in sorted(split_dir.iterdir()):
            if not rec_dir.is_dir():
                continue

            ar_file = rec_dir / "AR_labels.csv"
            if not ar_file.exists():
                continue

            rec_id = rec_dir.name
            try:
                lines = ar_file.read_text().strip().split("\n")
            except OSError as e:
                logger.warning("Cannot read %s: %s", ar_file, e)
                continue

            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                try:
                    action_id = int(parts[1])
                    if action_id < 0:
                        continue
                    start = int(Path(parts[3]).stem)
                    end = int(Path(parts[4]).stem)
                except (ValueError, IndexError):
                    continue

                seg_len = end - start + 1
                if seg_len < self.clip_len:
                    continue
                for clip_start in range(start, end - self.clip_len + 2, self.stride):
                    self.clips.append((rec_id, clip_start, action_id))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec_id, clip_start, action_id_raw = self.clips[idx]
        rgb_dir = self.recordings_root / self.split / rec_id / "rgb"

        frames: list[torch.Tensor] = []
        for i in range(clip_start, clip_start + self.clip_len):
            img_path = rgb_dir / f"{i:06d}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
                frames.append(_mvit_transform(img))
            except Exception:
                frames.append(torch.zeros(3, _MVIT_CROP, _MVIT_CROP))

        clip = torch.stack(frames, dim=0)  # [T, 3, H, W]

        # Remap raw action_id to grouped output index
        gid = self.id_to_group[action_id_raw] if action_id_raw < len(self.id_to_group) else 0
        return clip, gid


def mvit_collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate clips into batched tensor [B, 3, T, H, W] with labels [B]."""
    clips, labels = zip(*batch)
    return torch.stack(clips, dim=0), torch.tensor(labels, dtype=torch.long)


# =========================================================================
# MViTv2-S Fine-Tuning Model
# =========================================================================
class MViTFinetuneModel(nn.Module):
    """MViTv2-S backbone + single MLP head for activity classification.

    The backbone is loaded from torchvision (Kinetics-400 pretrained) with
    the classification head removed. A lightweight MLP head is added for
    the IndustReal activity task.
    """

    def __init__(
        self,
        num_classes: int = 69,
        freeze_backbone: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # -- Backbone --
        self.backbone = self._build_backbone(pretrained=True)
        self.backbone.head = nn.Identity()
        self.hidden_size = 768

        # -- Freeze control --
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("MViTFinetuneModel: backbone frozen")
        else:
            logger.info("MViTFinetuneModel: backbone trainable")

        # -- MLP Head --
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes),
        )

        # Initialize head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "MViTFinetuneModel: total=%.1fM, trainable=%.1fM, num_classes=%d",
            total_params / 1e6, trainable_params / 1e6, num_classes,
        )

    @staticmethod
    def _build_backbone(pretrained: bool = True) -> nn.Module:
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = mvit_v2_s(weights=weights)
        return model

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            clip: [B, 3, T, H, W] video clip (Kinetics-400 normalized).

        Returns:
            logits: [B, num_classes] activity logits.
        """
        # Handle [B, T, C, H, W] -> [B, C, T, H, W]
        if clip.dim() == 5 and clip.shape[1] != 3:
            clip = clip.permute(0, 2, 1, 3, 4).contiguous()

        # Backbone forward with optional gradient checkpointing
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            features = checkpoint(self.backbone, clip, use_reentrant=False)
        else:
            features = self.backbone(clip)  # [B, 768]

        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.head(features)
        return logits

    def unfreeze_last_stage(self, num_blocks: int = 4) -> None:
        """Unfreeze the last N transformer blocks for stage-2 training."""
        blocks = self.backbone.blocks
        total = len(blocks)
        start = max(0, total - num_blocks)
        for i in range(start, total):
            for p in blocks[i].parameters():
                p.requires_grad = True
        logger.info("Unfroze backbone blocks %d-%d (last %d blocks)", start, total - 1, num_blocks)

    def get_trainable_param_groups(
        self, backbone_lr: float, head_lr: float,
    ) -> list[dict]:
        """Return parameter groups with separate LRs for backbone and head."""
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": head_lr, "name": "head"},
        ]


# =========================================================================
# Training helpers
# =========================================================================
def train_one_epoch(
    model: MViTFinetuneModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    """Train for one epoch. Returns metrics dict."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": correct / max(total, 1),
        "num_samples": total,
    }


@torch.no_grad()
def validate(
    model: MViTFinetuneModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate model on validation set. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    per_class_correct = torch.zeros(num_classes, device=device)
    per_class_total = torch.zeros(num_classes, device=device)

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        for c in range(num_classes):
            mask = labels == c
            per_class_total[c] += mask.sum()
            per_class_correct[c] += (preds[mask] == c).sum()

    all_top1 = correct / max(total, 1)
    per_class_accuracy = (per_class_correct / per_class_total.clamp(min=1)).cpu().tolist()

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": all_top1,
        "per_class_accuracy": per_class_accuracy,
        "num_samples": total,
    }


# =========================================================================
# Main
# =========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MViTv2-S Fine-Tuning for IndustReal Activity Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, default="mvit_v2_s",
                        help="Video backbone name")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Backbone learning rate (stage 2)")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Epochs with frozen backbone (stage 1)")
    parser.add_argument("--clip-len", type=int, default=16,
                        help="Frames per clip")
    parser.add_argument("--stride", type=int, default=8,
                        help="Clip stride (overlap)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--save-dir", type=str, default="src/runs/mvit_finetune",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")
    parser.add_argument("--no-gradient-checkpointing", action="store_false",
                        dest="gradient_checkpointing")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                        help="Use FP16 mixed precision")
    parser.add_argument("--no-mixed-precision", action="store_false",
                        dest="mixed_precision")
    parser.add_argument("--unfreeze-blocks", type=int, default=4,
                        help="Number of last blocks to unfreeze in stage 2")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Seed ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Paths ----
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler for logging
    fh = logging.FileHandler(log_dir / "train.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name())
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("VRAM: %.1f GB", vram)

    # ---- Dataset ----
    recordings_root = Path(C.RECORDINGS_ROOT)
    logger.info("Recordings root: %s", recordings_root)

    train_dataset = MViTFinetuneDataset(
        split="train",
        recordings_root=recordings_root,
        clip_len=args.clip_len,
        stride=args.stride,
    )
    val_dataset = MViTFinetuneDataset(
        split="val",
        recordings_root=recordings_root,
        clip_len=args.clip_len,
        stride=args.stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mvit_collate,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mvit_collate,
        prefetch_factor=2,
        pin_memory=True,
    )

    logger.info(
        "Dataset: train=%d clips, val=%d clips, classes=%d",
        len(train_dataset), len(val_dataset), len(set(train_dataset.id_to_group)),
    )

    # ---- Model ----
    num_classes = max(train_dataset.id_to_group) + 1
    model = MViTFinetuneModel(
        num_classes=num_classes,
        freeze_backbone=True,
        use_checkpoint=args.gradient_checkpointing,
    ).to(device)

    if args.gradient_checkpointing:
        model.backbone.head = nn.Identity()  # ensure head removed after .to(device)

    # ---- Loss & Optimizer ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Stage 1: head only (backbone frozen)
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- Mixed Precision ----
    use_amp = args.mixed_precision and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info("Mixed precision: %s", "FP16" if use_amp else "disabled")

    # ---- Training Loop ----
    best_val_acc = 0.0
    best_epoch = -1
    stage2_started = False

    logger.info("=" * 60)
    logger.info("MViTv2-S Fine-Tuning")
    logger.info("  Epochs: %d (stage 1: %d frozen, stage 2: unfreeze last %d blocks)",
                args.epochs, args.warmup_epochs, args.unfreeze_blocks)
    logger.info("  Head LR: %.1e, Backbone LR: %.1e", args.lr, args.backbone_lr)
    logger.info("  Batch size: %d, Clip len: %d", args.batch_size, args.clip_len)
    logger.info("  Classes: %d, Save dir: %s", num_classes, save_dir)
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Stage transition: unfreeze backbone ----
        if epoch == args.warmup_epochs + 1 and not stage2_started:
            stage2_started = True
            model.unfreeze_last_stage(num_blocks=args.unfreeze_blocks)
            # Rebuild optimizer with separate LR groups
            optimizer = torch.optim.AdamW(
                model.get_trainable_param_groups(
                    backbone_lr=args.backbone_lr,
                    head_lr=args.lr,
                ),
                weight_decay=args.weight_decay,
            )
            logger.info("Stage 2: backbone unfrozen, optimizer rebuilt with backbone_lr=%.1e", args.backbone_lr)

        # ---- Train ----
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler=scaler,
        )
        stage_label = "frozen" if epoch <= args.warmup_epochs else "finetune"

        # ---- Validate ----
        val_metrics = validate(
            model, val_loader, criterion, device, num_classes=num_classes,
        )
        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d [%s] | Train Loss: %.4f Acc: %.4f | "
            "Val Loss: %.4f Acc: %.4f | Time: %.0fs",
            epoch, args.epochs, stage_label,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            elapsed,
        )

        # ---- Checkpoint ----
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / "best.pth")
            logger.info("  -> New best model saved (val_acc=%.4f)", best_val_acc)

        # Periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }, ckpt_dir / f"epoch_{epoch}.pth")

        # Force GC at epoch boundary
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Final Summary ----
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Best epoch: %d (val_acc=%.4f)", best_epoch, best_val_acc)
    logger.info("Checkpoints: %s", ckpt_dir)
    logger.info("Log: %s", log_dir / "train.log")
    logger.info("=" * 60)

    # Save final results metadata
    results = {
        "best_val_acc": round(best_val_acc, 6),
        "best_epoch": best_epoch,
        "num_epochs": args.epochs,
        "num_classes": num_classes,
        "backbone": args.backbone,
        "clip_len": args.clip_len,
        "batch_size": args.batch_size,
        "head_lr": args.lr,
        "backbone_lr": args.backbone_lr,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
