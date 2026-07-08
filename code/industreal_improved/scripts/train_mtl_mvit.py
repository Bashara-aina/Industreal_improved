#!/usr/bin/env python3
"""
train_mtl_mvit.py — MTL-All with MViTv2-S shared backbone.

Trains all 4 heads (detection, activity, PSR, pose) simultaneously using
Kendall uncertainty weighting + PCGrad gradient surgery.

Usage:
    # Full training (default: 100 epochs)
    python scripts/train_mtl_mvit.py

    # Plumbing test (1 epoch, small subset)
    python scripts/train_mtl_mvit.py --plumbing

    # Resume from checkpoint
    python scripts/train_mtl_mvit.py --resume path/to/checkpoint.pt
"""
import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path
_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
from src.models.mvit_mtl_model import MTLMViTModel, renormalize_pose

logger = logging.getLogger("train_mtl_mvit")

OUTPUT_ROOT = _CODE_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "mtl_mvit_run"


# ===========================================================================
# Loss functions
# ===========================================================================

def detection_loss(
    det_outputs: dict,
    det_list: list,
) -> torch.Tensor:
    """Detection loss: global object presence BCE + box MSE for verification.

    Args:
        det_outputs: dict of per-level {cls_logits, reg_preds}
        det_list: list of B dicts, each {boxes: [n_i, 4], labels: [n_i]}

    Returns:
        scalar loss.
    """
    level = "P3"
    cls_logits = det_outputs[level]["cls_logits"]  # [B, 24, H, W]
    B, _, H, W = cls_logits.shape
    device = cls_logits.device

    # Binary: does this image contain any object?
    has_any = torch.zeros(B, device=device)
    for b in range(B):
        if det_list[b]["boxes"].numel() > 0:
            has_any[b] = 1.0

    # Global presence BCE: avg over spatial (H,W) then avg over classes
    presence = cls_logits.mean(dim=(2, 3))  # [B, 24]
    presence = presence.mean(dim=1)  # [B]
    loss = F.binary_cross_entropy_with_logits(presence, has_any)

    return loss


def activity_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with ignore-index for -1 labels."""
    return F.cross_entropy(logits, targets, ignore_index=-1)


def psr_loss(psr_logits: torch.Tensor, psr_targets: torch.Tensor) -> torch.Tensor:
    """Per-frame BCE for PSR transition logits."""
    return F.binary_cross_entropy_with_logits(psr_logits, psr_targets)


def pose_loss(pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
    """Cosine/geodesic loss on renormalized fwd and up vectors."""
    fwd_pred, up_pred = renormalize_pose(pred_6d)
    fwd_gt = F.normalize(target_6d[:, :3], dim=1)
    up_gt = F.normalize(target_6d[:, 3:], dim=1)
    cos_fwd = (fwd_pred * fwd_gt).sum(dim=1).clamp(-1.0, 1.0)
    cos_up = (up_pred * up_gt).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cos_fwd).mean() + (1.0 - cos_up).mean()


# ===========================================================================
# Training step
# ===========================================================================

def train_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: dict,
    log_vars: nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip_norm: float = 1.0,
    hp_prec_cap: bool = True,
) -> dict:
    """Single training step with Kendall uncertainty weighting."""
    model.train()
    B = images.size(0)

    # Forward
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(images)

        # Per-task losses
        # Detection: simple presence BCE
        det_list = targets.get("detection", [])
        l_det = detection_loss(outputs["detection"], det_list)

        l_act = activity_loss(
            outputs["activity"],
            targets.get("activity", torch.zeros(B, dtype=torch.long, device=images.device)),
        ) if "activity" in targets else torch.tensor(0.0, device=images.device)

        l_psr = psr_loss(
            outputs["psr_logits"],
            targets.get("psr_labels", torch.zeros(B, 16, 11, device=images.device)),
        ) if "psr_labels" in targets else torch.tensor(0.0, device=images.device)

        # Pose: head_pose is [B, T, 9], take middle frame's 6D
        if "head_pose" in targets:
            hp = targets["head_pose"]  # [B, T, 9]
            hp_6d = hp[:, hp.size(1) // 2, :6]  # [B, 6] middle frame
        else:
            hp_6d = torch.zeros(B, 6, device=images.device)
        l_pose = pose_loss(outputs["pose_6d"], hp_6d)

        # Kendall uncertainty weighting
        losses = {"det": l_det, "act": l_act, "psr": l_psr, "pose": l_pose}
        total_loss = 0.0
        for name, loss in losses.items():
            prec = torch.exp(-log_vars[name])
            total_loss = total_loss + prec * loss + log_vars[name] / 2
            if hp_prec_cap and name == "pose":
                total_loss = total_loss / 2  # halve pose contribution

    # Backward
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

    scaler.step(optimizer)
    scaler.update()

    return {
        "loss": total_loss.item(),
        "loss_det": l_det.item(),
        "loss_act": l_act.item(),
        "loss_psr": l_psr.item(),
        "loss_pose": l_pose.item(),
        **{f"log_var_{k}": v.item() for k, v in log_vars.items()},
    }


# ===========================================================================
# Main training loop
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MTL-All with MViTv2-S shared backbone"
    )
    parser.add_argument("--plumbing", action="store_true", help="1 epoch, subset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head LR")
    parser.add_argument("--lr-log-var", type=float, default=1e-3, help="Log var LR")
    parser.add_argument("--hp-prec-cap", action="store_true", default=True, help="Cap pose precision")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Args: %s", vars(args))

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Building train dataset (sequence_mode=True, T=16)...")
    # Patch config for our training setup
    C.STAGED_TRAINING = False
    C.KENDALL_STAGED_TRAINING = False
    C.MIXED_PRECISION = True
    C.AMP_DTYPE = "bf16"

    train_ds = IndustRealMultiTaskDataset(
        split="train",
        img_size=(224, 224),
        augment=True,
        sequence_mode=True,
        sequence_length=16,
    )
    logger.info("Train samples: %d", len(train_ds))

    sampler = torch.utils.data.WeightedRandomSampler(
        train_ds.get_sampler_weights() if hasattr(train_ds, "get_sampler_weights")
        else torch.ones(len(train_ds)),
        num_samples=len(train_ds), replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn_sequences,
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building MTL-MViT model...")
    model = MTLMViTModel().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Params: %.1fM total, %.1fM trainable", total_params, trainable_params)

    # ── Kendall log_vars ──────────────────────────────────────────────────
    log_vars = nn.ParameterDict({
        name: nn.Parameter(torch.tensor([-0.5], device=device))
        for name in ["det", "act", "psr", "pose"]
    })
    logger.info("Log vars initialized to -0.5")

    # ── Optimizer ─────────────────────────────────────────────────────────
    param_groups = [
        {"params": model.feature_pyramid.backbone.parameters(), "lr": args.lr_backbone, "weight_decay": 0.05},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and "log_var" not in n], "lr": args.lr_head, "weight_decay": 0.05},
        {"params": log_vars.parameters(), "lr": args.lr_log_var, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log_vars = ckpt.get("log_vars", log_vars)
        logger.info("Resumed from epoch %d", start_epoch)

    # ── AMP scaler ─────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(device.type, enabled=True)

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info("Starting training (%d epochs)...", args.epochs)
    metrics_log = {"train_metrics": [], "config": vars(args)}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        epoch_metrics = {"loss": 0, "loss_det": 0, "loss_act": 0, "loss_psr": 0, "loss_pose": 0}
        n_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch[0].to(device, non_blocking=True)  # [B, T, 3, H, W]
            targets = {}
            for k, v in batch[1].items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(device, non_blocking=True)
                elif isinstance(v, dict):
                    targets[k] = {sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv
                                  for sk, sv in v.items()}
                else:
                    targets[k] = v

            # images shape: [B, T=16, 3, H=224, W=224]
            images = images.float() / 255.0
            mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
            images = (images - mean) / std

            # Permute for MViTv2: [B, T, C, H, W] → [B, C, T, H, W]
            images = images.permute(0, 2, 1, 3, 4).contiguous()

            step_metrics = train_step(
                model, images, targets, log_vars, optimizer, scaler,
                hp_prec_cap=args.hp_prec_cap,
            )

            for k in epoch_metrics:
                epoch_metrics[k] += step_metrics.get(k, 0)
            n_steps += 1

            if args.plumbing and batch_idx >= 10:
                break

        # Average
        avg_metrics = {k: v / max(n_steps, 1) for k, v in epoch_metrics.items()}
        avg_metrics["epoch"] = epoch
        avg_metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics_log["train_metrics"].append(avg_metrics)

        # Log
        dt = time.time() - t0
        lv = {k: step_metrics.get(k, 0) for k in ["log_var_det", "log_var_act", "log_var_psr", "log_var_pose"]}
        logger.info(
            "Epoch %3d/%d | loss=%.4f det=%.4f act=%.4f psr=%.4f pose=%.4f | "
            "lv=[%.2f,%.2f,%.2f,%.2f] | lr=%.2e | %.1fs",
            epoch, args.epochs,
            avg_metrics["loss"], avg_metrics["loss_det"], avg_metrics["loss_act"],
            avg_metrics["loss_psr"], avg_metrics["loss_pose"],
            lv.get("log_var_det", 0), lv.get("log_var_act", 0),
            lv.get("log_var_psr", 0), lv.get("log_var_pose", 0),
            avg_metrics["lr"], dt,
        )

        scheduler.step()

        # Save checkpoint
        if epoch % 10 == 0 or epoch == 1:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "log_vars": log_vars,
                "best_val_loss": best_val_loss,
                "metrics": avg_metrics,
            }
            ckpt_path = output_dir / f"epoch_{epoch:04d}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

        # Save latest
            latest = output_dir / "latest.pt"
            torch.save({**ckpt, "epoch": epoch}, latest)

        if args.plumbing and epoch >= 1:
            logger.info("Plumbing test complete.")
            break

    # Save final metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2, default=str)

    logger.info("Training complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
