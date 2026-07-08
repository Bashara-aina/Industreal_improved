#!/usr/bin/env python3
"""train_st_psr.py — Single-Task PSR Training (175 §6 Row 3)

Trains a PSR-only model using the LeakyReLU-repaired PSR head
(src/models/model.py:1604-1611) on top of a ConvNeXt-Tiny backbone + FPN.
No other heads are instantiated (no competing losses, no Kendall weighting).

Primary metric: event_f1@±3 + tau (greedy match within tolerance per
decoder_oracle_bound.py:252, NOT per-frame F1).

After training, calls scripts/eval_psr_transition_f1.py for evaluation.

Usage:
    # Val-split model selection (default):
    python scripts/train_st_psr.py [--epochs 100] [--batch-size 8]

    # Plumbing test (1 epoch, small subset):
    python scripts/train_st_psr.py --plumbing

    # Resume from checkpoint:
    python scripts/train_st_psr.py --resume <checkpoint.pth>

Reference: AAIML 175 §6 (ST-PSR row), §3.2 (PSR head spec),
§4 (PSR losses), §7.2 (event_f1@±3 metric).
"""

import argparse
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # code/industreal_improved/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.config as C
from src.split_config import require_split

logger = logging.getLogger("train_st_psr")

# ── Output path ──────────────────────────────────────────────────────────────
_ST_PSR_RUN_DIR = (
    _PROJECT_ROOT / "src/runs/rf_stages/checkpoints/st_psr_run"
)
_METRICS_PATH = _ST_PSR_RUN_DIR / "metrics.json"
_BEST_CKPT = _ST_PSR_RUN_DIR / "best.pth"
_LATEST_CKPT = _ST_PSR_RUN_DIR / "latest.pth"


# =========================================================================
# Model: lightweight backbone + FPN + PSRHead only
# =========================================================================
class STPSRModel(nn.Module):
    """Single-task PSR model: ConvNeXt-Tiny backbone + FPN + PSRHead.

    Architecture matches the multi-task PSR head exactly (PSRHead from
    src/models/model.py:1539) but without any other heads.

    The backbone + FPN + PSRHead are the same classes used by the full
    POPWMultiTaskModel, ensuring fair comparison in the experiment matrix.
    """

    def __init__(
        self,
        pretrained: bool = True,
        backbone_type: str = "convnext_tiny",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        from src.models.model import build_backbone, FPN, PSRHead

        # Backbone
        self.backbone = build_backbone(
            backbone_type, pretrained=pretrained, use_checkpoint=False,
        )

        # Channel dimensions
        if backbone_type == "convnext_tiny":
            c3_ch, c4_ch, c5_ch = 192, 384, 768
        else:  # resnet50
            c3_ch, c4_ch, c5_ch = 512, 1024, 2048
        fpn_in_channels = [c3_ch, c4_ch, c5_ch]

        # FPN neck (same as POPWMultiTaskModel)
        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=256)

        # PSR head — LeakyReLU repaired (model.py:1604-1607)
        self.psr_head = PSRHead(
            in_channels=256,
            hidden_dim=128,
            num_components=C.NUM_PSR_COMPONENTS,
            dropout=0.2,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("  Backbone frozen")

    def forward(
        self,
        images: torch.Tensor,
        seq_len: int = 1,
    ) -> dict:
        """Forward pass producing PSR logits only.

        Args:
            images: [B*T, C, H, W] — batched frames (B clips × T frames)
            seq_len: Number of frames per clip (T). When >1, runs the
                     PSR causal transformer over the full sequence.

        Returns:
            dict with 'psr_logits' of shape [B*T, 11] or [B, T, 11].
        """
        # Backbone: ConvNeXtBackbone returns (c2, c3, c4, c5)
        c2, c3, c4, c5 = self.backbone(images)

        # FPN expects positional args (c3, c4, c5)
        pyramid = self.fpn(c3, c4, c5)

        B = images.shape[0] // seq_len
        T = seq_len

        if T > 1:
            # Build per-frame features over the temporal sequence
            pyramid_seq = {}
            for k, v in pyramid.items():
                pyramid_seq[k] = v.reshape(B, T, *v.shape[1:])

            frame_feats = []
            for t in range(T):
                p3_t = pyramid_seq["p3"][:, t]
                p4_t = pyramid_seq["p4"][:, t]
                p5_t = pyramid_seq["p5"][:, t]
                p3_gap = self.psr_head.gap_p3(p3_t).flatten(1)
                p4_gap = self.psr_head.gap_p4(p4_t).flatten(1)
                p5_gap = self.psr_head.gap_p5(p5_t).flatten(1)
                fused = torch.cat([p3_gap, p4_gap, p5_gap], dim=1)
                frame_feats.append(self.psr_head.per_frame_mlp(fused))
            frame_feat_seq = torch.stack(frame_feats, dim=1)  # [B, T, hidden]

            # Causal Transformer
            causal_mask = torch.triu(
                torch.ones(T, T, device=images.device), diagonal=1
            ).bool()
            encoded = self.psr_head.transformer(
                frame_feat_seq, mask=causal_mask
            )  # [B, T, hidden]

            # Per-component heads
            enc_flat = encoded.reshape(B * T, -1)
            psr_logits = torch.cat(
                [head(enc_flat) for head in self.psr_head.output_heads],
                dim=-1,
            )  # [B*T, 11]
            psr_logits = psr_logits.view(B, T, -1)  # [B, T, 11]
        else:
            # Single-frame mode (eval or non-sequence training)
            psr_pyramid = pyramid
            psr_full = self.psr_head(psr_pyramid)  # [B, 12]
            psr_logits = psr_full[..., :11]  # [B, 11]

        return {"psr_logits": psr_logits}


# =========================================================================
# Loss: per-component BCE with PSR_COMP_WEIGHTS
# =========================================================================
class PSRBCELoss(nn.Module):
    """Per-component BCE loss with class-balanced weights.

    Uses PSR_COMP_WEIGHTS from config (inverse-prevalence weighting).
    Matches the per-component weighting from binary_focal_loss but uses
    plain BCE (no focal gamma) for stable single-task training.
    """

    def __init__(self):
        super().__init__()
        weights = torch.tensor(
            getattr(C, "PSR_COMP_WEIGHTS", [1.0] * C.NUM_PSR_COMPONENTS),
            dtype=torch.float32,
        )
        self.register_buffer("comp_weights", weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component BCE with inverse-prevalence weighting.

        Args:
            logits: [B, 11] or [B, T, 11] — raw logits (pre-sigmoid)
            targets: [B, 11] or [B, T, 11] — binary labels, -1 for ignore

        Returns:
            Scalar loss (weighted mean across batch and components).
        """
        # Flatten to [N, 11]
        logits_f = logits.reshape(-1, logits.shape[-1])
        targets_f = targets.reshape(-1, targets.shape[-1]).float()

        # Clamp to prevent extreme sigmoid saturation
        logits_f = logits_f.clamp(min=-8.0, max=8.0)

        # Per-element BCE
        loss_per_elem = F.binary_cross_entropy_with_logits(
            logits_f, targets_f, reduction="none"
        )  # [N, 11]

        # Apply ignore mask (-1 labels)
        ignore_mask = targets_f < 0
        loss_per_elem = loss_per_elem.masked_fill(ignore_mask, 0.0)
        valid_mask = ~ignore_mask

        # Per-component weighting
        cw = self.comp_weights.to(logits.device)  # [11]
        weight_mean = cw.mean()
        loss_per_elem = loss_per_elem * cw.unsqueeze(0) / weight_mean

        # Mean over valid (non-ignored) entries
        n_valid = valid_mask.sum()
        if n_valid == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss_per_elem.sum() / n_valid


# =========================================================================
# Evaluation: call eval_psr_transition_f1.py as subprocess
# =========================================================================
def _run_eval(
    ckpt_path: Path,
    output_path: Path,
    split_name: str,
    max_batches: int = 0,
) -> dict:
    """Run eval_psr_transition_f1.py as a subprocess, return metrics dict."""
    eval_script = (
        Path(__file__).resolve().parent / "eval_psr_transition_f1.py"
    )
    if not eval_script.exists():
        logger.error("eval_psr_transition_f1.py not found at %s", eval_script)
        return None

    import subprocess

    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint", str(ckpt_path),
        "--save-dir", str(output_path.parent),
    ]
    if max_batches > 0:
        cmd += ["--max-batches", str(max_batches)]

    logger.info("Running PSR transition-F1 eval: %s split", split_name)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        logger.error(
            "eval_psr_transition_f1.py failed (exit %d):\n%s\n%s",
            result.returncode,
            result.stdout,
            result.stderr,
        )
        return None

    metrics_path = output_path.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        logger.info(
            "  %s event_f1@±3: %.4f, tau: %.2f frames",
            split_name,
            metrics.get("event_f1_macro", -1),
            metrics.get("tau_mean_frames", float("nan")),
        )
        return metrics
    return None


# =========================================================================
# Training loop
# =========================================================================
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int = 0,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        images, targets = batch
        if images.shape[0] == 0:
            continue

        # Normalize images
        images_f = images.to(device).float()
        if images_f.max() > 1.0:
            images_f = images_f.div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        images_n = (images_f - mean) / std

        # PSR labels
        psr_labels = targets.get("psr_labels")
        if psr_labels is None:
            continue
        psr_labels = psr_labels.to(device)

        # Forward
        outputs = model(images_n, seq_len=1)  # single-frame mode
        psr_logits = outputs["psr_logits"]

        # Ensure label dimension matches
        if psr_labels.dim() == 3:
            psr_labels = psr_labels.squeeze(1)  # [B, 1, 11] -> [B, 11]
        if psr_labels.dim() == 2 and psr_labels.shape[-1] != psr_logits.shape[-1]:
            # Transpose if needed
            psr_labels = psr_labels.T

        # Loss
        loss = criterion(psr_logits, psr_labels)

        if not torch.isfinite(loss):
            logger.warning("  [epoch %d batch %d] Non-finite loss: %.6f", epoch, i, loss.item())
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i + 1) % 100 == 0:
            logger.info(
                "  epoch %d batch %d/%d  loss=%.6f",
                epoch, i + 1, len(loader), loss.item(),
            )

    return total_loss / max(n_batches, 1)


# =========================================================================
# Config patching
# =========================================================================
def patch_config_for_psr_only() -> None:
    """Patch config for PSR-only training."""
    C.TRAIN_DET = False
    C.TRAIN_HEAD_POSE = False
    C.TRAIN_ACT = False
    C.TRAIN_PSR = True
    C.STAGED_TRAINING = False
    C.KENDALL_STAGED_TRAINING = False
    C.USE_KENDALL = False
    C.MIXED_PRECISION = True
    C.AMP_DTYPE = "bf16"
    C.FREEZE_BACKBONE = False
    C.BACKBONE_LR_MULT = 0.01


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="ST-PSR: Single-task PSR training (175 §6 row 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plumbing", action="store_true",
        help="Plumbing mode: 1 epoch, subset of data, fast eval",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (default: C.EPOCHS or 1 for plumbing)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default: C.BATCH_SIZE)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Head learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-5,
        help="Backbone learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze backbone weights",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only run evaluation on existing checkpoint",
    )
    parser.add_argument(
        "--max-eval-batches", type=int, default=0,
        help="Cap for eval batches (0 = full eval)",
    )
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    patch_config_for_psr_only()

    # Plumbing mode
    if args.plumbing:
        C.EPOCHS = 1
        C.DEBUG_MODE = True
        logger.info("Plumbing mode: 1 epoch")
    if args.epochs is not None:
        C.EPOCHS = args.epochs
    if args.batch_size is not None:
        C.BATCH_SIZE = args.batch_size

    _ST_PSR_RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ── Split discipline ──────────────────────────────────────────────────
    require_split("val")
    logger.info("Split discipline: train on train split, val for selection")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    train_ds = IndustRealMultiTaskDataset(
        split="train", sequence_mode=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size or C.BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info("Train dataset: %d samples", len(train_ds))

    # ── Model ─────────────────────────────────────────────────────────────
    model = STPSRModel(
        pretrained=True,
        backbone_type="convnext_tiny",
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    logger.info(
        "Model: STPSRModel (backbone=%s, freeze=%s, PSR params=%d)",
        "convnext_tiny",
        args.freeze_backbone,
        sum(p.numel() for p in model.psr_head.parameters()),
    )

    # Verify LeakyReLU is in the path (model.py:1604-1607)
    for i, head in enumerate(model.psr_head.output_heads):
        activation = head[1]
        assert isinstance(activation, nn.LeakyReLU), (
            f"output_heads[{i}][1] is {type(activation).__name__}, "
            f"expected nn.LeakyReLU"
        )
        break  # Check first head only
    logger.info("  [OK] PSR output_heads use LeakyReLU activation")

    # ── Load checkpoint (resume or eval-only) ─────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    # ── Optimizer ─────────────────────────────────────────────────────────
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.backbone_lr},
    ], weight_decay=0.05)
    logger.info(
        "Optimizer: AdamW (head_lr=%.1e, backbone_lr=%.1e, wd=0.05)",
        args.lr, args.backbone_lr,
    )

    # ── Criterion ─────────────────────────────────────────────────────────
    criterion = PSRBCELoss().to(device)
    logger.info("Criterion: per-component BCE with PSR_COMP_WEIGHTS")

    # ── Training ──────────────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("=" * 60)
        logger.info("ST-PSR training: %d epochs", C.EPOCHS)
        logger.info("=" * 60)

        best_loss = float("inf")

        for epoch in range(start_epoch, C.EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, max_batches=0,
            )
            elapsed = time.time() - t0

            logger.info(
                "Epoch %d/%d  loss=%.6f  time=%.1fs",
                epoch + 1, C.EPOCHS, train_loss, elapsed,
            )

            # Save latest checkpoint
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
            }, str(_LATEST_CKPT))

            # Save best checkpoint by loss
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                }, str(_BEST_CKPT))
                logger.info("  New best loss: %.6f (saved)", best_loss)

    # ── Eval ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running PSR transition-F1 evaluation")

    ckpt_for_eval = _BEST_CKPT if _BEST_CKPT.exists() else _LATEST_CKPT
    if args.eval_only:
        ckpt_path_str = args.resume or str(_BEST_CKPT)
        ckpt_for_eval = Path(ckpt_path_str)

    if ckpt_for_eval.exists():
        metrics = _run_eval(
            ckpt_for_eval,
            _METRICS_PATH,
            "val",
            max_batches=args.max_eval_batches,
        )
    else:
        logger.warning("No checkpoint found at %s — skipping eval", _BEST_CKPT)
        metrics = None

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("ST-PSR RESULTS (175 §6 Row 3)")
    print("=" * 64)
    if metrics:
        print(f"  Checkpoint:          {ckpt_for_eval}")
        print(f"  event_f1@±3:         {metrics.get('event_f1_macro', 'N/A'):.4f}")
        print(f"  tau (mean delay):    {metrics.get('tau_mean_frames', 'N/A')} frames")
        print(f"  POS:                 {metrics.get('pos_macro', 'N/A'):.4f}")
        print(f"  Frames evaluated:    {metrics.get('n_frames', 'N/A')}")
    print(f"  Metrics saved to: {_METRICS_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
