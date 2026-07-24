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
import csv
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
from src.losses.uw_so import UWSOLoss
from src.losses.psr_balanced_loss import PSRBalancedLoss, compute_psr_priors
import train_mtl_v3 as mtl_v3_mod

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_v3_19')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradNormBalancer:
    """Closed-form GradNorm task-weight balancing (Chen et al. 2018).

    Every K steps, computes per-task gradient norms w.r.t. the last shared
    backbone layer and adjusts weights to balance gradient magnitudes.

    Reference: https://arxiv.org/abs/1711.02257
    """

    def __init__(self, shared_params, task_names, alpha=1.5, update_every=200):
        self.shared_params = list(shared_params)
        self.task_names = task_names
        self.alpha = alpha
        self.update_every = update_every
        self.weights = {t: 1.0 for t in task_names}
        self._initial_losses = {}
        self._counter = 0
        self._initialized = False

    def _get_grad_norms(self, losses):
        norms = {}
        for name in self.task_names:
            if name not in losses:
                norms[name] = 1.0
                continue
            grads = torch.autograd.grad(
                losses[name], self.shared_params,
                retain_graph=True, create_graph=False, allow_unused=True,
            )
            sq = 0.0
            for g in grads:
                if g is not None:
                    sq += g.detach().norm(2).item() ** 2
            norms[name] = math.sqrt(sq) + 1e-8
        return norms

    def update(self, losses):
        if not self._initialized:
            for t in self.task_names:
                if t in losses:
                    self._initial_losses[t] = losses[t].detach().item()
            self._initialized = True
            return

        self._counter += 1
        if self._counter % self.update_every != 0:
            return

        G = self._get_grad_norms(losses)
        avg_G = sum(G.values()) / len(G)

        ratios = {}
        for t in self.task_names:
            if t in losses and t in self._initial_losses:
                cur = max(losses[t].detach().item(), 1e-8)  # clamp to non-negative
                init_val = max(self._initial_losses[t], 1e-8)
                ratios[t] = cur / init_val
            else:
                ratios[t] = 1.0
        avg_r = sum(ratios.values()) / len(ratios)
        r = {t: ratios[t] / max(avg_r, 1e-8) for t in self.task_names}

        targets = {t: avg_G * (r[t] ** self.alpha) for t in self.task_names}

        for t in self.task_names:
            self.weights[t] = max(0.01, min(100.0, targets[t] / G[t]))

        mw = sum(self.weights.values()) / len(self.weights)
        if mw > 1e-8:
            for t in self.task_names:
                self.weights[t] /= mw


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
    p.add_argument('--mosaic-prob', type=float, default=0.5)
    p.add_argument('--copy-paste-prob', type=float, default=0.5)
    p.add_argument('--use-mosaic', action='store_true')
    p.add_argument('--use-copy-paste', action='store_true')
    p.add_argument('--num-anchors', type=int, default=16)
    p.add_argument('--output-dir', type=str, default='runs/mtl_v3.25_loss_balance')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--max-norm', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    # Loss weighting (YOLOv8 DFL reg loss is ~10x larger than old anchor head)
    p.add_argument('--det-cls-weight', type=float, default=1.0,
                   help='Weight for detection cls loss')
    p.add_argument('--det-reg-weight', type=float, default=0.1,
                   help='Weight for detection reg loss (default 0.1 because YOLOv8 DFL reg loss is ~10-40x act loss)')
    p.add_argument('--act-weight', type=float, default=1.0,
                   help='Weight for activity CE loss')
    p.add_argument('--pose-weight', type=float, default=5.0,
                   help='Weight for pose MSE loss (default 5.0 because pose loss is typically 0.003-0.36)')
    p.add_argument('--psr-weight', type=float, default=2.0,
                   help='Weight for PSR BCE loss (default 2.0 because psr loss is typically 0.1-0.5)')
    p.add_argument('--use-psr-logit-adjust', action='store_true',
                   help='Enable LogitAdjust + class-balanced weights for PSR loss (fixes dead components)')
    p.add_argument('--psr-logit-adjust-tau', type=float, default=1.0,
                   help='LogitAdjust temperature (0=none, 1=full, default 1.0)')
    p.add_argument('--psr-class-balanced-beta', type=float, default=0.999,
                   help='Effective-number beta for PSR per-component weights (0=disable, default 0.999)')
    p.add_argument('--loss-balancing', type=str, default='manual',
                   choices=['manual', 'uw_so', 'gradnorm'],
                   help='Loss balancing method (default: manual)')
    p.add_argument('--save-uw-so-history', action='store_true',
                   help='Record UW-SO log_sigma every 50 batches to CSV')
    args = p.parse_args()
    # Backward compat: --use-uw-so activates uw_so balancing
    if args.use_uw_so:
        args.loss_balancing = 'uw_so'
    return args


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

    # Initialize YOLOv8 det head from YOLOv8 weights (only when NOT resuming)
    if not (args.resume and Path(args.resume).exists()):
        init_from_yolov8_weights(model.m.det_head)

    # Build loss balancing objects
    uw_so_loss = None
    if args.loss_balancing == 'uw_so':
        uw_so_loss = UWSOLoss().to(DEVICE)
        logger.info(f'UW-SO enabled: {len(UWSOLoss.TASK_NAMES)} learnable log_sigma params')

    gradnorm_balancer = None
    if args.loss_balancing == 'gradnorm':
        last_shared = model.m.feature_pyramid.backbone.blocks[-1]
        gradnorm_balancer = GradNormBalancer(
            shared_params=last_shared.parameters(),
            task_names=['det', 'act', 'pose', 'psr'],
            alpha=1.5, update_every=200,
        )
        logger.info(f'GradNorm enabled: {len(gradnorm_balancer.task_names)} tasks, alpha=1.5, update_every=200')

    # Resume from checkpoint if provided
    ckpt = None
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict']
        # Load ALL weights including det_head (don't re-initialize from YOLOv8)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_det_loaded = sum(1 for k in sd if 'det_head' in k)
        logger.info(f'Resumed full model (incl. det_head): '
                    f'missing={len(missing)}, unexpected={len(unexpected)}, '
                    f'det_head_keys_loaded={n_det_loaded}')
        # Resume UW-SO state if available
        if uw_so_loss is not None and 'uw_so_state_dict' in ckpt:
            uw_so_loss.load_state_dict(ckpt['uw_so_state_dict'])
            logger.info(f'  UW-SO state resumed: sigmas={uw_so_loss.sigma.tolist()}')
    else:
        logger.info('No resume - starting from YOLOv8-initialized det_head + K400 backbone')

    # PSR balanced loss (LogitAdjust + class-balanced weights) — needs training data priors
    psr_loss_fn = None
    if args.use_psr_logit_adjust:
        logger.info('Computing PSR priors from training data...')
        probe_ds = FullMultiModalDataset(
            recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
            img_size=(640, 360), mosaic_prob=0.0, copy_paste_prob=0.0,
        )
        probe_loader = DataLoader(probe_ds, batch_size=8, shuffle=False,
                                  collate_fn=collate_real_targets, num_workers=0)
        priors = compute_psr_priors(probe_loader, num_components=11, max_batches=80)
        del probe_ds, probe_loader
        psr_loss_fn = PSRBalancedLoss(
            priors=priors, tau=args.psr_logit_adjust_tau,
            beta=args.psr_class_balanced_beta, num_components=11,
        ).to(DEVICE)
        logger.info(f'PSR LogitAdjust ENABLED: tau={args.psr_logit_adjust_tau}, '
                    f'beta={args.psr_class_balanced_beta}')
        logger.info(f'  Per-bit positive rates: {priors.round(3).tolist()}')
        logger.info(f'  Logit biases: {psr_loss_fn.logit_bias.cpu().numpy().round(3).tolist()}')

    return model, uw_so_loss, gradnorm_balancer, psr_loss_fn, ckpt


def train_phase2(args, model, uw_so_loss=None, gradnorm_balancer=None, psr_loss_fn=None, resume_ckpt=None):
    logger.info('=' * 60)
    logger.info('PHASE 2: Real multi-modal + YOLOv8 head (v3.25)')
    logger.info('=' * 60)
    logger.info(f'  Loss balancing: {args.loss_balancing}')
    if args.loss_balancing == 'uw_so' and uw_so_loss is not None:
        logger.info(f'  UW-SO init sigmas={uw_so_loss.sigma.tolist()}')
    elif args.loss_balancing == 'gradnorm' and gradnorm_balancer is not None:
        logger.info(f'  GradNorm: tasks={gradnorm_balancer.task_names}, alpha={gradnorm_balancer.alpha}')
    else:
        logger.info(f'  Manual weights: det_cls={args.det_cls_weight}, det_reg={args.det_reg_weight}, '
                     f'act={args.act_weight}, pose={args.pose_weight}, psr={args.psr_weight}')

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

    # UW-SO uses SGD (not AdamW) because AdamW's adaptive LR normalization
    # suppresses the large, consistent gradients from UW-SO, making convergence
    # pathologically slow (~250h estimated at 5e-6 with AdamW).
    # SGD with lr=1e-3 converges UW-SO in minutes.
    uwso_opt = None
    if uw_so_loss is not None:
        uwso_opt = torch.optim.SGD(
            uw_so_loss.parameters(),
            lr=1e-3,  # High LR needed for fast UW-SO convergence
            momentum=0.0,
            weight_decay=0.0,
        )
        logger.info(f'  UW-SO SGD optimizer: lr=1e-3, momentum=0.0, wd=0.0')

    # Resume UW-SO optimizer state from checkpoint if available
    if uwso_opt is not None and resume_ckpt is not None and 'uwso_opt_state_dict' in resume_ckpt:
        uwso_opt.load_state_dict(resume_ckpt['uwso_opt_state_dict'])
        logger.info('  UW-SO optimizer state resumed from checkpoint')

    for g in param_groups:
        logger.info(f'  lr={g["lr"]:.6f}, n_params={len(g["params"])}')

    # UW-SO history CSV logger
    uwso_csv_writer = None
    uwso_csv_file = None
    if args.save_uw_so_history and args.loss_balancing == 'uw_so' and uw_so_loss is not None:
        csv_path = Path(args.output_dir, 'uw_so_history.csv')
        uwso_csv_file = open(csv_path, 'w', newline='')
        uwso_csv_writer = csv.writer(uwso_csv_file)
        uwso_csv_writer.writerow([
            'step', 'batch', 'epoch',
            'loss_det', 'loss_act', 'loss_pose', 'loss_psr',
            'log_sigma_det', 'log_sigma_act', 'log_sigma_pose', 'log_sigma_psr',
            'weight_det', 'weight_act', 'weight_pose', 'weight_psr',
        ])
        logger.info(f'  UW-SO history CSV: {csv_path}')

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

            # PSR loss (BCE — optionally with LogitAdjust + class-balanced weights)
            psr_logits = out['psr_logits']  # [B, 11]
            psr_targets = []
            for t in targets['psr']:
                if t is not None and torch.is_tensor(t):
                    psr_targets.append(t.to(DEVICE).float())
                else:
                    psr_targets.append(torch.zeros(11, device=DEVICE))
            psr_target = torch.stack(psr_targets)
            if psr_loss_fn is not None:
                psr_loss = psr_loss_fn(psr_logits, psr_target)
            else:
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

            # Total loss — weighted sum
            raw_losses = {
                'det': total_cls_loss + total_reg_loss,
                'act': act_loss,
                'pose': pose_loss,
                'psr': psr_loss,
            }

            if args.loss_balancing == 'uw_so' and uw_so_loss is not None:
                total_loss = uw_so_loss(raw_losses)
            elif args.loss_balancing == 'gradnorm' and gradnorm_balancer is not None:
                gradnorm_balancer.update(raw_losses)
                total_loss = sum(gradnorm_balancer.weights[t] * raw_losses[t]
                                 for t in gradnorm_balancer.task_names)
            else:
                total_loss = (args.det_cls_weight * total_cls_loss
                              + args.det_reg_weight * total_reg_loss
                              + args.act_weight * act_loss
                              + args.pose_weight * pose_loss
                              + args.psr_weight * psr_loss)

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
                if uwso_opt is not None:
                    uwso_opt.step()
                    uw_so_loss.project()
                opt.zero_grad()
                if uwso_opt is not None:
                    uwso_opt.zero_grad()
                global_step += 1

            epoch_loss += total_loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                speed = n_batches / elapsed
                eta = (n_total - n_batches) / speed / 60
                balancing_str = ''
                if args.loss_balancing == 'uw_so' and uw_so_loss is not None:
                    sigmas = uw_so_loss.sigma.tolist()
                    balancing_str = f' uw_so=[{sigmas[0]:.3f},{sigmas[1]:.3f},{sigmas[2]:.3f},{sigmas[3]:.3f}]'
                elif args.loss_balancing == 'gradnorm' and gradnorm_balancer is not None:
                    w = gradnorm_balancer.weights
                    balancing_str = f' gn_w=[{w["det"]:.3f},{w["act"]:.3f},{w["pose"]:.3f},{w["psr"]:.3f}]'
                logger.info(
                    f'  P2 Ep{epoch} b{n_batches}/{n_total}: '
                    f'loss={total_loss.item():.4f}, cls={total_cls_loss.item():.4f}, '
                    f'reg={total_reg_loss.item():.4f}, act={act_loss.item():.4f}, '
                    f'pose={pose_loss.item():.4f}, psr={psr_loss.item():.4f}, pos={n_pos},'
                    f'{balancing_str} '
                    f'speed={speed:.1f}/s, ETA={eta:.0f}min'
                )

                if uwso_csv_writer is not None:
                    ls = uw_so_loss.log_sigma.detach().tolist()
                    wts = [math.exp(-2.0 * l) for l in ls]
                    uwso_csv_writer.writerow([
                        global_step, i, epoch,
                        total_cls_loss.item() + total_reg_loss.item(),
                        act_loss.item(), pose_loss.item(), psr_loss.item(),
                        ls[0], ls[1], ls[2], ls[3],
                        wts[0], wts[1], wts[2], wts[3],
                    ])
                    uwso_csv_file.flush()

            if n_batches % args.save_every == 0:
                save_ckpt(epoch, n_batches, args.output_dir, model, opt, global_step,
                          uw_so_loss=uw_so_loss, uwso_opt=uwso_opt)

        save_ckpt(epoch + 1, 0, args.output_dir, model, opt, global_step,
                  uw_so_loss=uw_so_loss, uwso_opt=uwso_opt)
        logger.info(f'P2 Epoch {epoch}: avg_loss={epoch_loss/n_batches:.4f}')

    if uwso_csv_file is not None:
        uwso_csv_file.close()


def save_ckpt(epoch, batch, output_dir, model, opt, step, uw_so_loss=None, uwso_opt=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, 'checkpoints').mkdir(exist_ok=True)
    name = f'phase2_e{epoch}_b{batch}.pth'
    path = Path(output_dir, 'checkpoints', name)
    ckpt = {
        'epoch': epoch,
        'batch': batch,
        'phase': 2,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'use_yolov8_head': True,
    }
    if uw_so_loss is not None:
        ckpt['uw_so_state_dict'] = uw_so_loss.state_dict()
    if uwso_opt is not None:
        ckpt['uwso_opt_state_dict'] = uwso_opt.state_dict()
    torch.save(ckpt, path)
    logger.info(f'  Saved {path}')


def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(args.output_dir, 'train.log')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    logger.info(f'Args: {vars(args)}')

    model, uw_so_loss, gradnorm_balancer, psr_loss_fn, resume_ckpt = build_model_and_data(args)
    logger.info(f'Model built on {DEVICE}')

    if args.phase2_epochs > 0:
        train_phase2(args, model, uw_so_loss=uw_so_loss,
                     gradnorm_balancer=gradnorm_balancer,
                     psr_loss_fn=psr_loss_fn, resume_ckpt=resume_ckpt)


if __name__ == '__main__':
    main()