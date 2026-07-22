#!/usr/bin/env python3
"""v3.19: MTL training with YOLOv8-style DFL detection head.

Replaces the legacy 3x3 + 16-anchor detection head with YOLOv8's actual
DFL + anchor-free detection head, initialized from YOLOv8 weights.

This should give us SOTA detection performance (close to YOLOv8's 0.59 mAP)
while keeping ONE MTL model.

Usage:
    python scripts/train/train_mtl_v3_yolov8_head.py \
        --resume runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
        --phase1-epochs 0 \
        --phase2-epochs 1 \
        --batch-size 2
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

from train_mtl_full_multimodal import (
    FullMultiModalDataset,
    expand_conv_proj_to_9ch,
    WrappedMTL,
    collate_real_targets,
)
from src.models.mvit_mtl_model import MTLMViTModel
from src.models.yolov8_det_head import YOLOv8DetectHead, init_from_yolov8_weights
from src.training.yolov8_det_loss import yolov8_detection_loss_v2
import train_mtl_v3 as mtl_v3_mod

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_v3_19')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--phase1-epochs', type=int, default=2)
    p.add_argument('--phase2-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--det-lr-mult', type=int, default=100)
    p.add_argument('--act-lr-mult', type=int, default=50)
    p.add_argument('--pose-lr-mult', type=int, default=10)
    p.add_argument('--psr-lr-mult', type=int, default=50)
    p.add_argument('--use-llrd', action='store_true')
    p.add_argument('--llrd-decay', type=float, default=0.95)
    p.add_argument('--use-uw-so', action='store_true')
    p.add_argument('--use-class-balanced-sampling', action='store_true')
    p.add_argument('--use-mosaic', action='store_true')
    p.add_argument('--use-copy-paste', action='store_true')
    p.add_argument('--num-anchors', type=int, default=16)
    p.add_argument('--output-dir', type=str, default='runs/mtl_v3.19_yolov8_head')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--max-norm', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_model_and_data(args):
    """Build model with YOLOv8 detection head and load data."""
    torch.manual_seed(args.seed)

    if args.num_anchors == 8:
        mtl_v3_mod.NUM_ANCHORS = 8
        mtl_v3_mod._ANCHOR_SPECS = mtl_v3_mod._ANCHOR_SPECS_8
    else:
        mtl_v3_mod.NUM_ANCHORS = 16
        mtl_v3_mod._ANCHOR_SPECS = mtl_v3_mod._ANCHOR_SPECS_16

    # Build model with YOLOv8 head
    model = MTLMViTModel(
        num_act_classes=75,
        num_det_classes=24,
        num_psr_components=11,
        num_anchors=args.num_anchors,
        use_yolov8_head=True,
    )
    expand_conv_proj_to_9ch(model)
    model = WrappedMTL(model).to(DEVICE)

    # Initialize YOLOv8 det head from YOLOv8 weights
    init_from_yolov8_weights(model.m.det_head)

    # Resume from checkpoint if provided
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict']
        # Filter out det_head weights (we use YOLOv8 init instead)
        sd_filtered = {k: v for k, v in sd.items() if 'det_head' not in k}
        missing, unexpected = model.load_state_dict(sd_filtered, strict=False)
        logger.info(f'Resumed backbone + non-det heads: missing={len(missing)}, unexpected={len(unexpected)}')
    else:
        logger.info('No resume - starting from YOLOv8-initialized det_head + K400 backbone')

    return model


def train_phase2(args, model):
    logger.info('=' * 60)
    logger.info('PHASE 2: Real multi-modal + YOLOv8 head (v3.19)')
    logger.info('=' * 60)

    real_ds = FullMultiModalDataset(
        recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
        img_size=(640, 360),
        mosaic_prob=args.mosaic_prob if args.use_mosaic else 0.0,
        copy_paste_prob=args.copy_paste_prob if args.use_copy_paste else 0.0,
    )

    if args.use_class_balanced_sampling:
        from train_mtl_v3 import ForegroundBatchSampler
        fg_sampler = ForegroundBatchSampler(
            real_ds, batch_size=args.batch_size, shuffle=True,
            class_balanced=True,
        )
        real_loader = DataLoader(
            real_ds, batch_sampler=fg_sampler,
            collate_fn=collate_real_targets, num_workers=0, pin_memory=False,
        )
    else:
        real_loader = DataLoader(
            real_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_real_targets, num_workers=0, pin_memory=False,
        )

    n_total = len(real_loader)
    logger.info(f'  Batches per epoch: {n_total}')

    # Build optimizer with per-head LR (YOLOv8 head now has many params)
    phase2_lr = args.lr / 4
    if args.use_llrd:
        from train_mtl_v3 import build_llrd_param_groups
        param_groups = build_llrd_param_groups(
            model, phase2_lr, llrd_decay=args.llrd_decay,
            det_lr_mult=args.det_lr_mult,
            act_lr_mult=args.act_lr_mult,
            pose_lr_mult=args.pose_lr_mult,
            psr_lr_mult=args.psr_lr_mult,
            weight_decay=0.05,
        )
    else:
        det_params, act_params, pose_params, psr_params, base_params = [], [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('m.det_head.'):
                det_params.append(p)
            elif name.startswith('m.act_head.'):
                act_params.append(p)
            elif name.startswith('m.pose_head.'):
                pose_params.append(p)
            elif name.startswith('m.psr_head.'):
                psr_params.append(p)
            else:
                base_params.append(p)
        param_groups = [
            {'params': base_params, 'lr': phase2_lr, 'weight_decay': 0.01},
            {'params': act_params, 'lr': phase2_lr * args.act_lr_mult, 'weight_decay': 0.01},
            {'params': pose_params, 'lr': phase2_lr * args.pose_lr_mult, 'weight_decay': 0.01},
            {'params': psr_params, 'lr': phase2_lr * args.psr_lr_mult, 'weight_decay': 0.01},
            {'params': det_params, 'lr': phase2_lr * args.det_lr_mult, 'weight_decay': 0.01},
        ]

    opt = torch.optim.AdamW(param_groups)
    for g in param_groups:
        logger.info(f'  lr={g["lr"]:.6f}, n_params={len(g["params"])}')

    # Training loop
    global_step = 0
    n_skipped = 0
    for epoch in range(args.phase2_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for i, (images, targets) in enumerate(real_loader):
            if i >= n_total:
                break

            images = images.to(DEVICE).float()
            images = images.unsqueeze(2)
            mean = torch.tensor([0.45] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            std = torch.tensor([0.225] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            images = (images - mean) / std

            for k in targets:
                if isinstance(targets[k], list):
                    targets[k] = [t.to(DEVICE) if torch.is_tensor(t) else t for t in targets[k]]

            out = model(images)

            # YOLOv8-style detection loss per level
            gt_boxes = [b.to(DEVICE).float() for b in targets['boxes']]
            gt_classes = [c.to(DEVICE).long() for c in targets['classes']]

            total_cls_loss = 0.0
            total_reg_loss = 0.0
            n_pos = 0
            for level_name, stride in [('P3', 8.0), ('P4', 16.0), ('P5', 32.0)]:
                if level_name not in out['detection']:
                    continue
                cls_logits = out['detection'][level_name]['cls_logits']
                reg_preds = out['detection'][level_name]['reg_preds']

                cls_l, reg_l, np_l = yolov8_detection_loss_v2(
                    cls_logits, reg_preds, gt_boxes, gt_classes,
                    img_w=C.IMG_WIDTH, img_h=C.IMG_HEIGHT,
                    stride=stride, reg_max=16,
                )
                total_cls_loss = total_cls_loss + cls_l
                total_reg_loss = total_reg_loss + reg_l
                n_pos += np_l

            # Activity loss (CE)
            activity_logits = out['activity']
            activity_targets = torch.stack([
                torch.tensor(t, dtype=torch.long, device=DEVICE) if isinstance(t, int) and t >= 0
                else torch.tensor(0, dtype=torch.long, device=DEVICE)
                for t in targets['activity']
            ])
            act_loss = F.cross_entropy(activity_logits, activity_targets)

            # PSR loss (BCE)
            psr_logits = out['psr_logits']  # [B, 11]
            psr_targets = []
            for t in targets['psr']:
                if t is not None and torch.is_tensor(t):
                    psr_targets.append(t.to(DEVICE).float())
                else:
                    psr_targets.append(torch.zeros(11, device=DEVICE))
            psr_target = torch.stack(psr_targets)
            psr_loss = F.binary_cross_entropy_with_logits(psr_logits, psr_target)

            # Pose loss (MSE - simplified)
            pose_6d = out['pose_6d']
            pose_targets = []
            for t in targets['pose']:
                if t is not None and isinstance(t, tuple):
                    fwd = torch.tensor(t[0], dtype=torch.float32, device=DEVICE)
                    up = torch.tensor(t[1], dtype=torch.float32, device=DEVICE)
                    pose_targets.append(torch.cat([fwd, up]))
                else:
                    pose_targets.append(torch.zeros(6, device=DEVICE))
            if pose_targets:
                pose_target = torch.stack(pose_targets)
                pose_loss = F.mse_loss(pose_6d, pose_target)
            else:
                pose_loss = torch.tensor(0.0, device=DEVICE)

            # Total loss
            total_loss = total_cls_loss + total_reg_loss + act_loss + psr_loss + pose_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f'  NaN/Inf at step {global_step}, skipping')
                n_skipped += 1
                opt.zero_grad()
                global_step += 1
                continue

            loss_scaled = total_loss / args.grad_accum
            loss_scaled.backward()

            if (i + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                opt.step()
                opt.zero_grad()
                global_step += 1

            epoch_loss += total_loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                speed = n_batches / elapsed
                eta = (n_total - n_batches) / speed / 60
                logger.info(
                    f'  P2 Ep{epoch} b{n_batches}/{n_total}: '
                    f'loss={total_loss.item():.4f}, cls={total_cls_loss.item():.4f}, '
                    f'reg={total_reg_loss.item():.4f}, act={act_loss.item():.4f}, '
                    f'psr={psr_loss.item():.4f}, pos={n_pos}, '
                    f'speed={speed:.1f}/s, ETA={eta:.0f}min'
                )

            if n_batches % args.save_every == 0:
                save_ckpt(epoch, n_batches, args.output_dir, model, opt, global_step)

        save_ckpt(epoch + 1, 0, args.output_dir, model, opt, global_step)
        logger.info(f'P2 Epoch {epoch}: avg_loss={epoch_loss/n_batches:.4f}')


def save_ckpt(epoch, batch, output_dir, model, opt, step):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, 'checkpoints').mkdir(exist_ok=True)
    name = f'phase2_e{epoch}_b{batch}.pth'
    path = Path(output_dir, 'checkpoints', name)
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'phase': 2,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'use_yolov8_head': True,
    }, path)
    logger.info(f'  Saved {path}')


def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(args.output_dir, 'train.log')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    logger.info(f'Args: {vars(args)}')

    model = build_model_and_data(args)
    logger.info(f'Model built on {DEVICE}')

    if args.phase2_epochs > 0:
        train_phase2(args, model)


if __name__ == '__main__':
    main()