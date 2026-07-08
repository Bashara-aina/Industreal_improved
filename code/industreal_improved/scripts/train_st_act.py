#!/usr/bin/env python3
"""
train_st_act.py — Single-task MViTv2-S activity baseline (175 §6 row 2).

Fine-tunes MViTv2-S end-to-end on 75-class clip-level activity recognition.

Architecture:
  - Backbone: MViTv2-S (Kinetics-400 pretrained, torchvision), trainable
  - Head: Linear(768, 75)
  - No other heads (single-task only)

Data:
  - 16-frame clips from AR_labels.csv segments, stride 8
  - Train split for training, val for model selection

Loss:
  - Cross-entropy with label smoothing 0.1
  - Class-balanced weights (inverse-frequency)

Optimizer:
  - AdamW, lr 1e-4 backbone / 1e-3 head, weight_decay 1e-4
  - Cosine schedule with 3-epoch linear warmup

After training, evaluates on val split and writes metrics.json.

Usage:
    python scripts/train_st_act.py --epochs 30
    python scripts/train_st_act.py --epochs 1 --max-train-clips 200 --max-val-clips 100  # plumbing test
"""

import argparse
import gc
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CODE_ROOT = _PROJECT_ROOT / "code" / "industreal_improved"
for _p in [
    str(_CODE_ROOT),
    str(_CODE_ROOT / "src"),
    str(_CODE_ROOT / "src" / "models"),
    str(_CODE_ROOT / "src" / "training"),
    str(_CODE_ROOT / "src" / "evaluation"),
    str(_CODE_ROOT / "src" / "data"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_st_act")

from src.models.video_backbones import load_mvit_v2_s
from src.split_config import require_split
import src.config as C

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 75
CLIP_LEN = 16
CLIP_STRIDE = 8

# MViTv2-S Kinetics normalization
_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
_STD = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)

OUTPUT_DIR = _CODE_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "st_act_run"


# ===========================================================================
# Model
# ===========================================================================
class MViTv2STAct(nn.Module):
    """Single-task MViTv2-S activity model with 75-class clip head."""

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()
        backbone = load_mvit_v2_s(pretrained=True)
        feat_dim = backbone.head[1].in_features  # 768
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone frozen.")
        else:
            logger.info("Backbone trainable (end-to-end fine-tune).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C, H, W] — normalized float32 clips.

        Returns:
            logits: [B, num_classes]
        """
        # MViTv2 expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        features = self.backbone(x)  # [B, 768]
        return self.classifier(features)  # [B, 75]


# ===========================================================================
# Dataset
# ===========================================================================
class ClipDataset(torch.utils.data.Dataset):
    """16-frame clip dataset from AR_labels.csv segments.

    Spans all recordings in a split, builds non-overlapping clips within
    each action segment at a given stride.
    """

    def __init__(
        self,
        split: str = "train",
        clip_len: int = CLIP_LEN,
        stride: int = CLIP_STRIDE,
        max_clips: int = None,
    ):
        self.split = split
        self.clip_len = clip_len
        self.stride = stride
        self.clips: list[tuple[str, int, int]] = []  # (recording, clip_start, action_id)
        self._build(max_clips)
        logger.info(
            "ClipDataset(split=%s): %d clips from %d recordings",
            split, len(self.clips), len(set(c[0] for c in self.clips)),
        )

    def _build(self, max_clips: int = None):
        """Walk recordings/{split}/ and build clip list from AR_labels.csv."""
        split_dir = C.RECORDINGS_ROOT / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        rec_dirs = sorted(split_dir.iterdir())
        count = 0
        for rec_dir in rec_dirs:
            if not rec_dir.is_dir():
                continue
            ar_file = rec_dir / "AR_labels.csv"
            if not ar_file.exists():
                continue
            rec_id = rec_dir.name

            for line in ar_file.read_text().strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                try:
                    action_id = int(parts[1])
                except (ValueError, IndexError):
                    continue
                if action_id < 0 or action_id >= NUM_CLASSES:
                    continue
                # Parse frame numbers from filename stems
                try:
                    start = int(Path(parts[3]).stem)
                    end = int(Path(parts[4]).stem)
                except (ValueError, IndexError):
                    continue
                seg_len = end - start + 1
                if seg_len < self.clip_len:
                    continue
                # Create clips at stride intervals
                for clip_start in range(start, end - self.clip_len + 1, self.stride):
                    self.clips.append((rec_id, clip_start, action_id))
                    count += 1
                    if max_clips is not None and count >= max_clips:
                        return

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec_id, clip_start, action_id = self.clips[idx]
        clip = self._load_clip(rec_id, clip_start)
        return clip, action_id

    def _load_clip(self, rec_id: str, clip_start: int) -> torch.Tensor:
        """Load 16 consecutive frames, normalize, return [16, 3, 224, 224]."""
        split_dir = C.RECORDINGS_ROOT / self.split
        rgb_dir = split_dir / rec_id / "rgb"
        if not rgb_dir.exists():
            # Some recordings have frames directly in rec dir under different structure
            rgb_dir = split_dir / rec_id
        frames = []
        for i in range(self.clip_len):
            frame_idx = clip_start + i
            img_path = rgb_dir / f"{frame_idx:06d}.jpg"
            try:
                from PIL import Image
                from torchvision.transforms import functional as TF

                img = Image.open(img_path).convert("RGB")
                img = TF.resize(img, [256], antialias=True)
                img = TF.center_crop(img, [224, 224])
                img = TF.to_tensor(img)  # [3, 224, 224], [0, 1]
                # MViTv2 normalization (Kinetics)
                img = (img - 0.45) / 0.225
                frames.append(img)
            except Exception:
                frames.append(torch.zeros(3, 224, 224))
        return torch.stack(frames, dim=0)  # [16, 3, 224, 224]


def compute_class_weights(dataset: ClipDataset, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Compute inverse-frequency class weights from dataset labels."""
    counter = Counter()
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        counter[int(label)] += 1

    weights = torch.zeros(num_classes, dtype=torch.float)
    for c in range(num_classes):
        n = counter.get(c, 0)
        weights[c] = 1.0 / max(n, 1)
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    logger.info("Class weights computed: min=%.3f, max=%.3f, n_nonzero=%d",
                weights[weights > 0].min().item(), weights.max().item(),
                (weights > 0).sum().item())
    return weights


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader,
             num_classes: int = NUM_CLASSES, device: torch.device = None) -> dict:
    """Compute clip-level top-1, top-5 on val split."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    for clips, labels in val_loader:
        clips = clips.to(device, non_blocking=True)
        logits = model(clips)  # [B, 75]
        preds = logits.argmax(dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_logits.append(logits.cpu().numpy())

    if len(all_preds) == 0:
        return {"error": "empty predictions", "clip_count": 0}

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)

    valid = all_labels >= 0
    labels_v = all_labels[valid]
    preds_v = all_preds[valid]
    logits_v = all_logits[valid]

    top1 = (preds_v == labels_v).mean()
    # Top-5
    top5_indices = np.argsort(logits_v, axis=1)[:, -5:]
    top5 = np.any(top5_indices == labels_v[:, None], axis=1).mean()

    # Per-class accuracy
    per_class = {}
    for c in range(num_classes):
        mask = labels_v == c
        n_c = mask.sum()
        if n_c > 0:
            correct_c = (preds_v[mask] == c).sum()
            per_class[str(c)] = {
                "count": int(n_c),
                "top1": float(correct_c) / float(n_c),
            }

    metrics = {
        "clip_count": int(len(all_labels)),
        "valid_clip_count": int(valid.sum()),
        "top1": float(top1),
        "top5": float(top5),
        "per_class": per_class,
    }
    logger.info("Eval: top1=%.4f top5=%.4f (n=%d)", top1, top5, int(valid.sum()))
    return metrics


# ===========================================================================
# Training
# ===========================================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch: int):
    """Run one training epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_scheduler: torch.optim.lr_scheduler.LinearLR,
    num_epochs: int,
    warmup_epochs: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Run full training loop with per-epoch eval."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_top1 = 0.0
    history = {"train_loss": [], "train_acc": [], "val_top1": [], "val_top5": [], "lr": []}

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
        )

        # Step scheduler AFTER optimizer (PyTorch convention)
        if epoch <= warmup_epochs and warmup_epochs > 0:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Evaluate on val
        val_metrics = evaluate(model, val_loader, device=device)
        val_top1 = val_metrics.get("top1", 0.0)
        val_top5 = val_metrics.get("top5", 0.0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)
        history["lr"].append(current_lr)

        epoch_time = time.time() - t0

        logger.info(
            "Epoch %3d/%d | loss=%.4f acc=%.4f | val top1=%.4f top5=%.4f | lr=%.2e | %.1fs",
            epoch, num_epochs, train_loss, train_acc, val_top1, val_top5, current_lr, epoch_time,
        )

        # Save best checkpoint
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "config": {
                    "backbone": "mvit_v2_s",
                    "num_classes": NUM_CLASSES,
                    "clip_len": CLIP_LEN,
                    "stride": CLIP_STRIDE,
                },
            }
            ckpt_path = output_dir / "best.pth"
            torch.save(ckpt, ckpt_path)
            logger.info("  New best model saved to %s (top1=%.4f)", ckpt_path, val_top1)

    # Final checkpoint
    final_ckpt = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_top1": val_top1,
        "val_top5": val_top5,
    }
    torch.save(final_ckpt, output_dir / "final.pth")

    # Final eval
    logger.info("Running final evaluation...")
    final_metrics = evaluate(model, val_loader, device=device)
    final_metrics["history"] = history
    final_metrics["best_val_top1"] = best_val_top1
    final_metrics["config"] = {
        "backbone": "mvit_v2_s",
        "num_classes": NUM_CLASSES,
        "clip_len": CLIP_LEN,
        "stride": CLIP_STRIDE,
        "num_epochs": num_epochs,
        "warmup_epochs": warmup_epochs,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    return final_metrics


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Single-task MViTv2-S activity fine-tune (175 §6 row 2)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head classifier learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone (for ablation)")
    parser.add_argument("--max-train-clips", type=int, default=None, help="Cap train clips (plumbing)")
    parser.add_argument("--max-val-clips", type=int, default=None, help="Cap val clips (plumbing)")
    parser.add_argument("--eval-split", type=str, default="val",
                        choices=["val", "test"], help="Split for evaluation")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # Validate eval split
    require_split(args.eval_split, allow_test_only=False)

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    logger.info("Device: %s", device)
    logger.info("Args: %s", vars(args))

    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Building train dataset...")
    train_ds = ClipDataset(split="train", clip_len=CLIP_LEN, stride=CLIP_STRIDE,
                           max_clips=args.max_train_clips)
    logger.info("Building val dataset...")
    val_ds = ClipDataset(split=args.eval_split, clip_len=CLIP_LEN, stride=CLIP_STRIDE,
                         max_clips=args.max_val_clips)

    if len(train_ds) == 0:
        logger.error("Empty train dataset. Check RECORDINGS_ROOT path: %s", C.RECORDINGS_ROOT)
        sys.exit(1)
    if len(val_ds) == 0:
        logger.error("Empty val dataset.")
        sys.exit(1)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Initializing MViTv2-S + 75-class head...")
    model = MViTv2STAct(num_classes=NUM_CLASSES, freeze_backbone=args.freeze_backbone)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %.1fM total, %.1fM trainable", total_params / 1e6, trainable_params / 1e6)

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    logger.info("Computing inverse-frequency class weights...")
    class_weights = compute_class_weights(train_ds, num_classes=NUM_CLASSES).to(device)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
    ])
    logger.info(
        "Optimizer: backbone params=%d (lr=%.2e), head params=%d (lr=%.2e)",
        len(backbone_params), args.lr_backbone, len(head_params), args.lr_head,
    )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    # Cosine schedule after warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
    )
    # Linear warmup from 0.1x to 1.0x over warmup_epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting training (%d epochs, %d warmup)...", args.epochs, args.warmup_epochs)
    metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=cosine_scheduler,
        warmup_scheduler=warmup_scheduler,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        device=device,
        output_dir=output_dir,
    )

    logger.info("Training complete. Best val top1: %.4f", metrics.get("best_val_top1", 0.0))
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
