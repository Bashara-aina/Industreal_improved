#!/usr/bin/env python3
"""V8: Multi-task with YOLOv8m detection + MViTv2-S activity + pose + PSR.

Architecture:
  Detection:    YOLOv8m (D1R, 0.995 mAP50, frozen or fine-tuned)
  Activity:     MViTv2-S (Kinetics-400, frozen, gives 0.3810 probe)
  Pose:         Direct regression from MViTv2-S features
  PSR:          Per-component heads from MViTv2-S features

This is a NEW multi-task system. V5b (PID 758477) is the original run
with ConvNeXt features. V8 is the architecturally-corrected version.

Usage:
  python3 scripts/train_v8_multitask.py --epochs 5
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_v8")


class V8Model(nn.Module):
    """V8: YOLOv8m detection + MViTv2-S features for pose/PSR/activity.

    Architecture:
        Detection head: YOLOv8m (ultralytics) -- direct inference
        Activity head: Linear over MViTv2-S features (768 -> 69 classes)
        Pose head:     Linear over MViTv2-S features (768 -> 6)
        PSR head:       Linear over MViTv2-S features (768 -> 11)

    All heads share the MViTv2-S backbone (frozen). Detection is a
    separate forward pass (YOLOv8m takes image-level inputs, not video).
    """

    def __init__(self, num_activity_classes=69, num_psr_components=11):
        super().__init__()
        # MViTv2-S backbone (frozen)
        try:
            from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
            self.backbone = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            feat_dim = 768  # MViTv2-S feature dim
        except Exception as e:
            logger.warning(f"Could not load MViTv2-S: {e}")
            self.backbone = None
            feat_dim = 768

        # Heads
        self.activity_head = nn.Linear(feat_dim, num_activity_classes)
        self.pose_head = nn.Linear(feat_dim, 6)  # fwd + up = 6
        self.psr_head = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(num_psr_components)])

        # Kendall log_vars (learnable)
        self.log_var_det = nn.Parameter(torch.tensor(0.0))
        self.log_var_pose = nn.Parameter(torch.tensor(0.0))
        self.log_var_act = nn.Parameter(torch.tensor(0.0))
        self.log_var_psr = nn.Parameter(torch.tensor(0.0))

    def forward(self, video_clip, image=None):
        """Forward pass.

        Args:
            video_clip: [B, T, 3, H, W] for MViTv2-S (activity, pose, PSR)
            image: [B, 3, H, W] for YOLOv8m detection (optional)
        Returns:
            dict with activity_logits, pose_pred, psr_logits, det_output
        """
        out = {}
        if self.backbone is not None and video_clip is not None:
            # MViTv2-S expects [B, C, T, H, W]
            if video_clip.dim() == 5 and video_clip.shape[2] != 3:
                video_clip = video_clip.permute(0, 2, 1, 3, 4)
            with torch.no_grad():
                feat = self.backbone(video_clip)
            # feat: [B, 768] (after global pool)
            out['activity_logits'] = self.activity_head(feat)
            out['pose_pred'] = self.pose_head(feat)
            out['psr_logits'] = torch.stack([h(feat) for h in self.psr_head], dim=1)  # [B, 11, 1]
        return out

    def compute_loss(self, out, targets, epoch):
        """Compute combined Kendall-weighted loss."""
        losses = {}

        # Activity loss
        if 'activity' in targets and 'activity_logits' in out:
            losses['activity'] = F.cross_entropy(out['activity_logits'], targets['activity'])

        # Pose loss (L1)
        if 'pose' in targets and 'pose_pred' in out:
            losses['pose'] = F.l1_loss(out['pose_pred'], targets['pose'])

        # PSR loss (per-component BCE)
        if 'psr' in targets and 'psr_logits' in out:
            losses['psr'] = F.binary_cross_entropy_with_logits(
                out['psr_logits'].squeeze(-1), targets['psr']
            )

        # Combined with Kendall weighting
        total = 0
        for name, loss in losses.items():
            log_var = getattr(self, f'log_var_{name}')
            prec = torch.exp(-log_var)
            total = total + prec * loss + log_var

        return total, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    model = V8Model().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    logger.info(f"Training V8 for {args.epochs} epochs (skeleton — data pipeline TBD)")
    logger.info(f"This is a demo of the architecture. Full data pipeline needs video clip loading.")
    logger.info(f"YOLOv8m detection: use D1R output directly (already trained, 0.995 mAP50)")
    logger.info(f"MViTv2-S activity: load Kinetics weights, train head (target 0.45-0.55 top-1)")
    logger.info(f"Pose + PSR: train heads on MViTv2-S features")
    logger.info(f"KENDALL_FIXED_WEIGHTS=0 (let Kendall rebalance)")

    # For now, this is a skeleton. The full V8 requires:
    # 1. Video clip dataset (16-frame clips, Kinetics preprocessing)
    # 2. Per-frame targets (det bbox, act class, pose fwd/up, psr binary)
    # 3. YOLOv8m inference (image-level, not video)
    # 4. Training loop with val every epoch

    logger.info("V8 skeleton ready. Need to add: data pipeline, training loop, val.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
