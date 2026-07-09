#!/usr/bin/env python3
"""Single-task training script (4-task MTL baselines).

[OPUS 192 §5 step 7] Mandatory baselines for the paper. Per-head ceiling
numbers; what the MTL model's performance is judged against.

Trains ONE task (det, act, psr, pose) on the chosen backbone, no MTL.
Each run produces:
  - A specialist checkpoint
  - Per-head val metrics (the ceiling)
  - The backbone weights (for model soup — see build_soup.py)

This is a unified script; pass --task to select which head to train.

Usage:
    python scripts/train_st.py --task det --epochs 30 --output_dir runs/st_det
    python scripts/train_st.py --task act --epochs 30 --output_dir runs/st_act
    python scripts/train_st.py --task psr --epochs 30 --output_dir runs/st_psr
    python scripts/train_st.py --task pose --epochs 30 --output_dir runs/st_pose
"""
import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Path setup
_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.NUM_ACT_OUTPUTS = 75
C.ACT_CLASS_GROUPING = "none"

from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
from src.models.mvit_mtl_model import MTLMViTModel
from scripts.train_mtl_mvit import (
    detection_loss, activity_loss, psr_loss, pose_loss, evaluate
)

logger = logging.getLogger("train_st")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def to_device_targets(targets, device):
    """Move all tensors in targets dict to device."""
    for k, v in list(targets.items()):
        if isinstance(v, torch.Tensor):
            targets[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    for sk, sv in item.items():
                        if isinstance(sv, torch.Tensor):
                            item[sk] = sv.to(device, non_blocking=True)
    return targets


def normalize_images(images, device):
    """Normalize images: [B, T, 3, H, W] in [0,255] → [B, 3, T, H, W] normalized."""
    images = images.float() / 255.0
    mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
    images = (images - mean) / std
    images = images.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, W]
    return images


def train_one_task(task: str, args):
    """Train a single task end-to-end."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    train_ds = IndustRealMultiTaskDataset(
        split="train", img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    val_ds = IndustRealMultiTaskDataset(
        split="val", img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_sequences, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_sequences, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    # Build model
    model = MTLMViTModel(num_act_classes=75).to(device)
    logger.info(f"Built MTL model (will use only --task {task} head)")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_metric = 0.0 if task != "pose" else float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            targets = to_device_targets(targets, device)
            images = normalize_images(images, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                # Single-task loss: only the selected head
                if task == "det":
                    loss = detection_loss(outputs["detection"], targets.get("detection", []))
                elif task == "act":
                    loss = activity_loss(outputs["activity"], targets["activity"])
                elif task == "psr":
                    loss = psr_loss(outputs["psr_logits"], targets.get("psr_labels",
                                          torch.zeros(images.size(0), 16, 11, device=device)))
                elif task == "pose":
                    if "head_pose" in targets:
                        hp = targets["head_pose"]
                        hp_6d = hp[:, hp.size(1) // 2, :6]
                        loss = pose_loss(outputs["pose_6d"], hp_6d)
                    else:
                        continue
                else:
                    raise ValueError(f"Unknown task: {task}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0
        logger.info(f"Epoch {epoch}/{args.epochs}: {task} loss = {avg_loss:.4f} ({dt:.0f}s)")

        # Eval
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device, epoch=epoch)
            # Pick the task-specific metric
            if task == "det":
                key = "det_mAP50"
                val = metrics.get(key, 0.0)
                is_better = val > best_metric
            elif task == "act":
                key = "act_top1"
                val = metrics.get(key, 0.0)
                is_better = val > best_metric
            elif task == "psr":
                key = "psr_event_f1_at_3"
                val = metrics.get(key, 0.0)
                is_better = val > best_metric
            elif task == "pose":
                key = "pose_fwd_mae"
                val = metrics.get(key, float("inf"))
                is_better = val < best_metric
            else:
                key, val, is_better = None, 0.0, False
            logger.info(f"  Eval {key}: {val:.4f}")

            # Save best
            if is_better:
                best_metric = val
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "task": task,
                    "metric": val,
                    "metric_key": key,
                }, output_dir / "best.pt")
                logger.info(f"  New best {key}: {val:.4f} (epoch {epoch})")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "task": task,
        }, output_dir / "latest.pt")

    # Final save
    metrics_log = {"task": task, "best_metric": best_metric,
                   "best_metric_key" if 'key' in dir() else "metric_key": key,
                   "best_epoch": best_epoch, "epochs": args.epochs}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2, default=str)
    logger.info(f"Done. Best {key} = {best_metric:.4f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Single-task training for MTL baselines")
    parser.add_argument("--task", choices=["det", "act", "psr", "pose"], required=True,
                        help="Which task to train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-batches-per-epoch", type=int, default=4000)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    train_one_task(args.task, args)


if __name__ == "__main__":
    main()
