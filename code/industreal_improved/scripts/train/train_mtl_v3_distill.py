#!/usr/bin/env python3
"""MTL training with YOLOv8 detection distillation.

Key differences from train_mtl_v3.py:
- Detection head uses YOLOv8 as teacher (soft labels)
- Distillation loss added to detection loss
- Activity/pose/PSR train with hard labels (no distillation)
- All 4 heads receive balanced LR (per-head multipliers)

Usage:
    python scripts/train/train_mtl_v3_distill.py \
        --resume runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
        --phase1-epochs 0 \
        --phase2-epochs 1 \
        --batch-size 2 \
        --output-dir runs/mtl_v3.12_distill
"""
import argparse, json, logging, math, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-use the v3 trainer's building blocks
sys.path.insert(0, str(_CODE_ROOT))
import train_mtl_v3 as mtl_v3_mod
from train_mtl_v3 import (
    build_llrd_param_groups,
    match_anchors_to_gt,
    sigmoid_focal_loss,
    ciou_loss,
    decode_deltas_to_xyxy,
    ForegroundBatchSampler,
    multi_task_loss_v3,
    _ANCHOR_SPECS_16, _ANCHOR_SPECS_8,
    NUM_DET_CLASSES,
)
from train_mtl_full_multimodal import (
    FullMultiModalDataset,
    FullSyntheticDataset,
    expand_conv_proj_to_9ch,
    WrappedMTL,
    collate_real_targets,
    collate_synth_targets,
)
from src.models.mvit_mtl_model import MTLMViTModel
from src.training.yolov8_distill import YOLOv8Distiller, distill_loss

import src.config as C
C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_distill')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--phase1-epochs', type=int, default=2)
    p.add_argument('--phase2-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--det-lr-mult', type=float, default=1000)
    p.add_argument('--act-lr-mult', type=float, default=50)
    p.add_argument('--pose-lr-mult', type=float, default=50)
    p.add_argument('--psr-lr-mult', type=float, default=50)
    p.add_argument('--use-llrd', action='store_true')
    p.add_argument('--llrd-decay', type=float, default=0.95)
    p.add_argument('--use-uw-so', action='store_true')
    p.add_argument('--use-mosaic', action='store_true')
    p.add_argument('--use-copy-paste', action='store_true')
    p.add_argument('--use-class-balanced-sampling', action='store_true')
    p.add_argument('--num-anchors', type=int, default=16, choices=[8, 16])
    p.add_argument('--distill-weight', type=float, default=2.0,
                   help='Weight for distillation loss (relative to hard label cls loss)')
    p.add_argument('--distill-score-thresh', type=float, default=0.3,
                   help='Min YOLOv8 score to use as soft label')
    p.add_argument('--yolov8-weights', type=str,
                   default='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/assembly_state_detection_model_weights/asd_best_IndustRealandSynthetic.pt')
    p.add_argument('--output-dir', type=str, default='runs/mtl_v3.12_distill')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--max-norm', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def setup_model_and_data(args):
    torch.manual_seed(args.seed)

    # Apply anchor config
    global NUM_ANCHORS
    if args.num_anchors == 8:
        mtl_v3_mod.NUM_ANCHORS = 8
        mtl_v3_mod._ANCHOR_SPECS = mtl_v3_mod._ANCHOR_SPECS_8
    else:
        mtl_v3_mod.NUM_ANCHORS = 16
        mtl_v3_mod._ANCHOR_SPECS = mtl_v3_mod._ANCHOR_SPECS_16
    NUM_ANCHORS = mtl_v3_mod.NUM_ANCHORS

    # Build model
    model = MTLMViTModel(num_act_classes=75, num_det_classes=24,
                          num_psr_components=11, num_anchors=args.num_anchors)
    expand_conv_proj_to_9ch(model)
    model = WrappedMTL(model).to(DEVICE)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = {k[2:] if k.startswith('m.') else k: v for k, v in ckpt['model_state_dict'].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f'Resumed: missing={len(missing)}, unexpected={len(unexpected)}')
        if 'batch' in ckpt:
            start_epoch = ckpt.get('epoch', 0)

    # Load YOLOv8 distiller (only in Phase 2)
    yolo_distiller = None
    if args.phase2_epochs > 0:
        yolo_distiller = YOLOv8Distiller(
            weights_path=args.yolov8_weights,
            device=str(DEVICE),
            conf_threshold=0.05,
        )

    return model, yolo_distiller, start_epoch


def train_phase2(args, model, yolo_distiller, start_epoch):
    """Phase 2: real data with YOLOv8 distillation for detection."""
    logger.info('=' * 60)
    logger.info(f'PHASE 2: Real multi-modal + YOLOv8 distillation')
    logger.info('=' * 60)

    real_ds = FullMultiModalDataset(
        recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
        img_size=(640, 360),
        mosaic_prob=args.mosaic_prob if args.use_mosaic else 0.0,
        copy_paste_prob=args.copy_paste_prob if args.use_copy_paste else 0.0,
    )

    if args.use_class_balanced_sampling:
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

    # Build optimizer with per-head LR
    phase2_lr = args.lr / 4
    if args.use_llrd:
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
    logger.info(f'Optimizer: {len(param_groups)} groups')

    # Per-head LR is already in place
    for g in param_groups:
        logger.info(f'  lr={g["lr"]:.6f}, n_params={len(g["params"])}')

    # Training loop
    global_step = start_epoch * n_total
    n_skipped = 0

    for epoch in range(args.phase2_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_det = 0.0
        epoch_distill = 0.0
        epoch_act = 0.0
        epoch_pose = 0.0
        epoch_psr = 0.0
        n_batches = 0
        t0 = time.time()

        for i, (images, targets) in enumerate(real_loader):
            if i >= n_total:
                break

            images = images.to(DEVICE).float()  # [B, 9, H, W]
            images = images.unsqueeze(2)  # [B, 9, 1, H, W]

            # Normalize
            mean = torch.tensor([0.45]*9, device=DEVICE).view(1, 9, 1, 1, 1)
            std = torch.tensor([0.225]*9, device=DEVICE).view(1, 9, 1, 1, 1)
            images_norm = (images - mean) / std

            # Move targets
            for k in targets:
                if isinstance(targets[k], list):
                    targets[k] = [t.to(DEVICE) if torch.is_tensor(t) else t for t in targets[k]]

            # Forward
            out = model(images_norm)

            # Build anchors
            anchors_per_level = {}
            for level in ['P3', 'P4', 'P5']:
                if level in out['detection']:
                    cls_l = out['detection'][level]['cls_logits']
                    H, W = cls_l.shape[2], cls_l.shape[3]
                    anchors_per_level[level] = mtl_v3_mod.generate_anchors(H, W, DEVICE)

            # Multi-task loss (no distill yet)
            loss, lc = multi_task_loss_v3(
                out, targets, anchors_per_level,
                use_supcon=False, uw_so=None,
                loss_type='focal', matcher_type='iou',
                use_tal=False, tal_alpha=2.0,
                use_class_balanced_sampling=args.use_class_balanced_sampling,
            )

            # Distillation loss for detection
            distill_total = torch.tensor(0.0, device=DEVICE)
            if yolo_distiller is not None:
                # Get YOLOv8 soft labels
                # Use only RGB channels (first 3 of 9). The distiller expects
                # [0, 1] float range (post TF.to_tensor, pre-MTL normalization).
                # Un-normalize from MTL format: x_norm = (x_orig - 0.45) / 0.225
                # so x_orig = x_norm * 0.225 + 0.45
                rgb = images[:, :3].squeeze(2)  # [B, 3, H, W] (normalized)
                rgb_unnorm = (rgb * 0.225 + 0.45).clamp(0, 1)  # [B, 3, H, W] in [0,1]
                soft_labels = yolo_distiller.get_soft_labels(rgb_unnorm, C.IMG_WIDTH, C.IMG_HEIGHT)

                # CRITICAL FIX-2026-07-22: Detach backbone gradient from distill.
                # Distillation was destroying activity/pose/PSR (Activity 37%->13%).
                # By detaching cls_logits, the distill loss only updates the
                # detection head weights, NOT the backbone. This keeps multi-task
                # features intact while still teaching the detection head from
                # YOLOv8's predictions.
                for level in ['P3', 'P4', 'P5']:
                    if level not in out['detection']:
                        continue
                    cls_logits = out['detection'][level]['cls_logits']
                    reg_preds = out['detection'][level]['reg_preds']
                    H, W = cls_logits.shape[2], cls_logits.shape[3]
                    anchors = anchors_per_level[level]

                    # Detach: stop gradient to backbone, only train det_head
                    cls_logits_detached = cls_logits.detach()
                    reg_preds_detached = reg_preds.detach()

                    d_loss = distill_loss(
                        cls_logits_detached, reg_preds_detached, anchors, soft_labels,
                        img_w=C.IMG_WIDTH, img_h=C.IMG_HEIGHT,
                        distill_weight=args.distill_weight,
                        score_thresh=args.distill_score_thresh,
                    )
                    distill_total = distill_total + d_loss

            total_loss = loss + distill_total

            # Check for NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f'  NaN/Inf at step {global_step}, skipping')
                n_skipped += 1
                opt.zero_grad()
                global_step += 1
                continue

            # Backward
            loss_scaled = total_loss / args.grad_accum
            loss_scaled.backward()

            if (i + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                opt.step()
                opt.zero_grad()
                global_step += 1

            epoch_loss += total_loss.item()
            epoch_det += lc['det_cls']
            epoch_distill += distill_total.item()
            epoch_act += lc.get('act', 0)
            epoch_pose += lc.get('pose', 0)
            epoch_psr += lc.get('psr', 0)
            n_batches += 1

            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                speed = n_batches / elapsed
                eta = (n_total - n_batches) / speed / 60
                logger.info(
                    f'  P2 Ep{epoch} b{n_batches}/{n_total}: '
                    f'loss={total_loss.item():.4f}, det={lc["det_cls"]:.4f}, '
                    f'distill={distill_total.item():.4f}, '
                    f'act={lc.get("act", 0):.4f}, psr={lc.get("psr", 0):.4f}, '
                    f'speed={speed:.1f}/s, ETA={eta:.0f}min'
                )

            if n_batches % args.save_every == 0:
                save_ckpt(epoch, n_batches, args.output_dir, model, opt, global_step)

        save_ckpt(epoch + 1, 0, args.output_dir, model, opt, global_step)
        logger.info(f'P2 Epoch {epoch}: loss={epoch_loss/n_batches:.4f}, '
                    f'det={epoch_det/n_batches:.4f}, distill={epoch_distill/n_batches:.4f}')


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

    model, yolo_distiller, start_epoch = setup_model_and_data(args)
    logger.info(f'Model loaded on {DEVICE}')

    if args.phase2_epochs > 0:
        train_phase2(args, model, yolo_distiller, start_epoch)


if __name__ == '__main__':
    main()