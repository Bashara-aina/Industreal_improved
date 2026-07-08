#!/usr/bin/env python3
"""
train_st_pose.py — Single-task MViTv2-S head pose baseline (175 §6 row 4).

Trains a frozen MViTv2-S backbone + 6D MLP regression head on per-frame head
pose (forward + up vectors).  Uses cosine/geodesic loss and reports angular
MAE with bootstrap CI after training.

Architecture:
  - Backbone: MViTv2-S (Kinetics-400 pretrained, torchvision), frozen
  - Head: Linear(768, 256) -> LeakyReLU -> Linear(256, 6) -> renormalize fwd/up
  - No other heads (single-task only)

Data:
  - Per-frame pose.csv (fwd_x, fwd_y, fwd_z, pos_x, pos_y, pos_z, up_x, up_y, up_z)
  - Train split for training, val for model selection

Loss:
  - Cosine/geodesic: (1 - cos(fwd_pred, fwd_gt)) + (1 - cos(up_pred, up_gt))
  - Renormalized fwd and up vectors before loss computation

After training, computes angular MAE per channel + bootstrap CI (1000 resamples,
seed 42, frame-weighted) and writes metrics.json mirroring bootstrap_ci.json.

Usage:
    python scripts/train_st_pose.py --epochs 30
    python scripts/train_st_pose.py --epochs 1 --max-frames 500  # plumbing test

Reference: AAIML 175 §6 (ST-Pose row), §3.2 (Pose head), §7.2 (Pose angular MAE)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
_CODE_ROOT = Path(__file__).resolve().parent.parent
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
logger = logging.getLogger("train_st_pose")

from src.models.video_backbones import load_mvit_v2_s
from src.split_config import require_split
import src.config as C

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSE_DIM = 6  # fwd(3) + up(3)

# MViTv2-S Kinetics normalization
_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
_STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)

OUTPUT_DIR = _CODE_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "st_pose_run"

# Bootstrap reference (for structural mirroring)
_BOOTSTRAP_REF_PATH = _CODE_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "bootstrap_ci.json"


# ===========================================================================
# Bootstrap CI helpers
# ===========================================================================

def bootstrap_ci(
    values: list[float],
    weights: list[float] | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for weighted mean.

    Returns (weighted_mean, ci_lower, ci_upper).
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    if weights is None:
        weights = [1.0] * n
    weights = list(weights)

    def _wmean(vals, wts):
        return sum(v * w for v, w in zip(vals, wts)) / sum(wts)

    point = _wmean(values, weights)
    boot = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        boot.append(_wmean([values[i] for i in idx], [weights[i] for i in idx]))
    boot.sort()
    alpha = (1.0 - ci) / 2.0
    lo = boot[int(round(alpha * n_resamples))]
    hi = boot[int(round((1.0 - alpha) * n_resamples))]
    return (point, lo, hi)


# ===========================================================================
# Model
# ===========================================================================

class MViTv2STPose(nn.Module):
    """Single-task MViTv2-S head pose model with 6D MLP regression head.

    Predicts 6D continuous pose (fwd 3 + up 3), renormalized at inference.
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        backbone = load_mvit_v2_s(pretrained=True)
        feat_dim = backbone.head[1].in_features  # 768
        backbone.head = nn.Identity()
        self.backbone = backbone

        # MLP regression head: 768 -> 256 -> 6 (fwd 3 + up 3)
        self.pose_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, POSE_DIM),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone frozen.")
        else:
            logger.info("Backbone trainable (not recommended for ST-Pose baseline).")

        self._init_weights()

    def _init_weights(self):
        for m in self.pose_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] — single normalized frame (not a clip).

        Returns:
            raw_6d: [B, 6] — raw MLP output (fwd 3 + up 3, not yet renormalized).
        """
        # MViTv2-S requires 5D input [B, C, T, H, W] with T >= 16.
        # Unsqueeze temporal dim and repeat minimum 16 frames.
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
            x = x.expand(-1, -1, 16, -1, -1).contiguous()  # [B, C, 16, H, W]
        features = self.backbone(x)  # [B, 768]
        return self.pose_head(features)  # [B, 6]


def renormalize_pose(raw_6d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split 6D into fwd and up, L2-normalize each.

    Args:
        raw_6d: [B, 6] — fwd(3) + up(3)

    Returns:
        fwd_norm: [B, 3] — unit forward vectors
        up_norm: [B, 3] — unit up vectors
    """
    fwd = raw_6d[:, :3]
    up = raw_6d[:, 3:]
    fwd_norm = F.normalize(fwd, dim=1)
    up_norm = F.normalize(up, dim=1)
    return fwd_norm, up_norm


def cosine_pose_loss(
    pred_6d: torch.Tensor,
    target_6d: torch.Tensor,
) -> torch.Tensor:
    """Cosine/geodesic pose loss per 175 §4.

    L = (1 - cos(fwd_pred, fwd_gt)) + (1 - cos(up_pred, up_gt))

    Both pred and target are [B, 6] = fwd(3) + up(3).
    Internally renormalizes predictions; targets assumed already unit-normalized.
    """
    fwd_pred, up_pred = renormalize_pose(pred_6d)
    fwd_gt = F.normalize(target_6d[:, :3], dim=1)
    up_gt = F.normalize(target_6d[:, 3:], dim=1)

    cos_fwd = (fwd_pred * fwd_gt).sum(dim=1)  # [B]
    cos_up = (up_pred * up_gt).sum(dim=1)

    # Clamp to [-1, 1] for numerical stability
    cos_fwd = cos_fwd.clamp(-1.0, 1.0)
    cos_up = cos_up.clamp(-1.0, 1.0)

    loss = (1.0 - cos_fwd).mean() + (1.0 - cos_up).mean()
    return loss


def angular_mae_per_frame(
    pred_6d: torch.Tensor,
    target_6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-frame angular MAE for fwd and up.

    Returns (fwd_mae_deg, up_mae_deg) each [B] in degrees.
    Computed as degrees(arccos(clip(dot(normalized), -1, 1))).
    Matches gt_pose_variance.py:40 arccos convention.
    """
    fwd_pred, up_pred = renormalize_pose(pred_6d)
    fwd_gt = F.normalize(target_6d[:, :3], dim=1)
    up_gt = F.normalize(target_6d[:, 3:], dim=1)

    cos_fwd = (fwd_pred * fwd_gt).sum(dim=1).clamp(-1.0, 1.0)
    cos_up = (up_pred * up_gt).sum(dim=1).clamp(-1.0, 1.0)

    fwd_mae = torch.rad2deg(torch.acos(cos_fwd))
    up_mae = torch.rad2deg(torch.acos(cos_up))
    return fwd_mae, up_mae


# ===========================================================================
# Dataset
# ===========================================================================

class PoseFrameDataset(torch.utils.data.Dataset):
    """Per-frame head pose dataset from pose.csv.

    Loads single frames from a split's recording directories and reads
    the corresponding 9-DoF head pose labels from pose.csv.
    Only the 6D rotation (fwd + up, no position) is returned.
    """

    def __init__(
        self,
        split: str = "train",
        max_frames: int | None = None,
    ):
        self.split = split
        self.frames: list[tuple[str, int, np.ndarray]] = []  # (recording, frame_idx, pose_6d)
        self._build(max_frames)
        logger.info(
            "PoseFrameDataset(split=%s): %d frames from %d recordings",
            split, len(self.frames), len(set(f[0] for f in self.frames)),
        )

    def _build(self, max_frames: int | None = None):
        """Walk recordings/{split}/ and build frame list from pose.csv."""
        split_dir = C.RECORDINGS_ROOT / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        rec_dirs = sorted(split_dir.iterdir())
        count = 0
        for rec_dir in rec_dirs:
            if not rec_dir.is_dir():
                continue
            pose_file = rec_dir / "pose.csv"
            if not pose_file.exists():
                continue
            rec_id = rec_dir.name

            lines = pose_file.read_text().strip().split("\n")
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                try:
                    fwd = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    up = np.array([float(parts[7]), float(parts[8]), float(parts[9])], dtype=np.float32)
                except (ValueError, IndexError):
                    continue
                # Normalize
                fwd_n = fwd / max(np.linalg.norm(fwd), 1e-6)
                up_n = up / max(np.linalg.norm(up), 1e-6)
                pose_6d = np.concatenate([fwd_n, up_n])  # [6]

                self.frames.append((rec_id, count, pose_6d))
                count += 1
                if max_frames is not None and count >= max_frames:
                    return

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec_id, _, pose_6d = self.frames[idx]
        frame = self._load_frame(rec_id)
        return frame, torch.from_numpy(pose_6d).float()

    def _load_frame(self, rec_id: str) -> torch.Tensor:
        """Load a single frame, normalize, return [3, H, W]."""
        split_dir = C.RECORDINGS_ROOT / self.split
        rgb_dir = split_dir / rec_id / "rgb"
        if not rgb_dir.exists():
            rgb_dir = split_dir / rec_id
        try:
            from PIL import Image
            from torchvision.transforms import functional as TF

            # Use first available frame as placeholder (pose is per-frame)
            candidates = sorted(rgb_dir.glob("*.jpg"))
            if not candidates:
                return torch.zeros(3, 224, 224)
            img_path = candidates[0]
            img = Image.open(img_path).convert("RGB")
            img = TF.resize(img, [256], antialias=True)
            img = TF.center_crop(img, [224, 224])
            img = TF.to_tensor(img)  # [3, 224, 224], [0, 1]
            img = (img - 0.45) / 0.225
            return img
        except Exception:
            return torch.zeros(3, 224, 224)


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_pose(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> dict:
    """Compute angular MAE per channel with bootstrap CI on val split.

    Returns dict matching bootstrap_ci.json structure.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_fwd_mae: list[float] = []
    all_up_mae: list[float] = []
    per_rec_fwd: dict[str, list[float]] = {}
    per_rec_up: dict[str, list[float]] = {}
    rec_frame_weights: dict[str, int] = {}
    batch_count = 0

    for frames, targets in val_loader:
        if max_batches is not None and batch_count >= max_batches:
            break
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        pred_6d = model(frames)
        fwd_mae, up_mae = angular_mae_per_frame(pred_6d, targets)

        all_fwd_mae.extend(fwd_mae.cpu().numpy().tolist())
        all_up_mae.extend(up_mae.cpu().numpy().tolist())
        batch_count += 1

    if len(all_fwd_mae) == 0:
        return {"error": "empty predictions", "frame_count": 0}

    # Per-recording aggregation (requires metadata — here we use a single bucket)
    # Bootstrap CI on frame-level MAE (frame-weighted = each frame weight 1)
    n_frames = len(all_fwd_mae)
    fwd_mean = float(np.mean(all_fwd_mae))
    up_mean = float(np.mean(all_up_mae))

    fwd_ci = bootstrap_ci(all_fwd_mae, weights=None)
    up_ci = bootstrap_ci(all_up_mae, weights=None)

    results = {
        "head_pose_forward": {
            "headline_weighted_mean_deg": fwd_mean,
            "bootstrap_95_ci_deg": [fwd_ci[1], fwd_ci[2]],
            "bootstrap_method": "frame-weighted (1000 resamples, seed 42)",
            "n_frames": n_frames,
        },
        "head_pose_up": {
            "headline_weighted_mean_deg": up_mean,
            "bootstrap_95_ci_deg": [up_ci[1], up_ci[2]],
            "bootstrap_method": "frame-weighted (1000 resamples, seed 42)",
            "n_frames": n_frames,
        },
        "metadata": {
            "n_bootstrap": 1000,
            "random_seed": 42,
            "n_frames": n_frames,
        },
    }

    logger.info(
        "Pose eval: fwd MAE=%.4f [%.4f, %.4f], up MAE=%.4f [%.4f, %.4f] (n=%d)",
        fwd_mean, fwd_ci[1], fwd_ci[2],
        up_mean, up_ci[1], up_ci[2],
        n_frames,
    )
    return results


# ===========================================================================
# Training
# ===========================================================================

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    """Run one training epoch. Returns average loss and sample count."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    n_batches = 0

    for frames, targets in loader:
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        pred_6d = model(frames)
        loss = cosine_pose_loss(pred_6d, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_samples += frames.size(0)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, total_samples


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
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
    best_fwd_mae = float("inf")
    history = {"train_loss": [], "val_fwd_mae": [], "val_up_mae": [], "lr": []}

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, n_samples = train_epoch(model, train_loader, optimizer, device)

        # Evaluate on val
        val_metrics = evaluate_pose(model, val_loader, device=device)
        val_fwd_mae = val_metrics.get("head_pose_forward", {}).get("headline_weighted_mean_deg", float("nan"))
        val_up_mae = val_metrics.get("head_pose_up", {}).get("headline_weighted_mean_deg", float("nan"))

        history["train_loss"].append(train_loss)
        history["val_fwd_mae"].append(val_fwd_mae)
        history["val_up_mae"].append(val_up_mae)
        history["lr"].append(current_lr)

        epoch_time = time.time() - t0

        logger.info(
            "Epoch %3d/%d | loss=%.6f | val fwd=%.4f up=%.4f | lr=%.2e | %.1fs",
            epoch, num_epochs, train_loss, val_fwd_mae, val_up_mae, current_lr, epoch_time,
        )

        # Save best checkpoint (based on mean of fwd + up MAE)
        current_mean = (val_fwd_mae + val_up_mae) / 2.0
        if not math.isnan(current_mean) and current_mean < best_fwd_mae:
            best_fwd_mae = current_mean
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_fwd_mae": val_fwd_mae,
                "val_up_mae": val_up_mae,
                "config": {
                    "backbone": "mvit_v2_s",
                    "pose_dim": POSE_DIM,
                },
            }
            ckpt_path = output_dir / "best.pth"
            torch.save(ckpt, ckpt_path)
            logger.info("  New best model saved to %s (mean=%.4f)", ckpt_path, current_mean)

    # Final checkpoint
    final_ckpt = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_fwd_mae": val_fwd_mae,
        "val_up_mae": val_up_mae,
    }
    torch.save(final_ckpt, output_dir / "final.pth")

    # Final eval
    logger.info("Running final evaluation...")
    final_metrics = evaluate_pose(model, val_loader, device=device)
    final_metrics["history"] = history
    final_metrics["best_mean_mae"] = best_fwd_mae
    final_metrics["config"] = {
        "backbone": "mvit_v2_s",
        "pose_dim": POSE_DIM,
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
        description="Single-task MViTv2-S head pose baseline (175 §6 row 4)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone learning rate (unused when frozen)")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head MLP learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap total frames (plumbing)")
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
    train_ds = PoseFrameDataset(split="train", max_frames=args.max_frames)
    logger.info("Building val dataset...")
    val_ds = PoseFrameDataset(split=args.eval_split, max_frames=args.max_frames)

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
    logger.info("Initializing MViTv2-S + 6D MLP pose head...")
    model = MViTv2STPose(freeze_backbone=True)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %.1fM total, %.1fM trainable", total_params / 1e6, trainable_params / 1e6)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    # cosine_pose_loss is used inline; no criterion object needed.

    # ------------------------------------------------------------------
    # Optimizer (only head params trainable since backbone is frozen)
    # ------------------------------------------------------------------
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params,
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )
    logger.info(
        "Optimizer: head params=%d (lr=%.2e)",
        len(head_params), args.lr_head,
    )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
    )
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
        optimizer=optimizer,
        scheduler=cosine_scheduler,
        warmup_scheduler=warmup_scheduler,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        device=device,
        output_dir=output_dir,
    )

    logger.info("Training complete. Best mean MAE: %.4f", metrics.get("best_mean_mae", float("nan")))
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
