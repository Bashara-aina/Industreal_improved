#!/usr/bin/env python3
"""
50-image cls-only overfit experiment (Opus v9 §R4).

Tests whether the detection classifier CAN overfit a tiny sample:
  - 50 images with ≥1 GT box each (so pos_mask is non-empty every step)
  - Detection-only (no pose/activity/PSR)
  - Same config as RF2 (DET_POS_IOU_THRESH=0.4, DET_POS_IOU_TOP_K=9, etc.)
  - Outputs: POS_ANCHOR_PROBE scores, cls_weight.norm(), loss curve, step mAP

If cls scores → 1.0 on positives and loss → 0 within 200-500 steps:
  → The classifier CAN learn, so the RF2 issue is gradient starvation (data supply),
    not an architectural problem.
If cls scores stay ~0.02-0.07 and loss plateaus above 0.1:
  → There's an architectural or label-noise problem in the cls head itself.

Usage:
  python scripts/overfit_50img_cls.py [--device cuda] [--lr 1e-4] [--epochs 50]
"""
import sys, os, time, math, argparse, json

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, os.pardir))
SRC_DIR = os.path.join(WORK_DIR, 'src')
sys.path.insert(0, WORK_DIR)
sys.path.insert(1, os.path.join(SRC_DIR, 'models'))
sys.path.insert(2, os.path.join(SRC_DIR, 'training'))
sys.path.insert(3, os.path.join(SRC_DIR, 'evaluation'))
sys.path.insert(4, SRC_DIR)

import config as C
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.ops import box_iou

C.TRAIN_DET = True
C.TRAIN_HEAD_POSE = False
C.TRAIN_ACT = False
C.TRAIN_PSR = False
C.USE_KENDALL = True
C.KENDALL_FIXED_WEIGHTS = True
C.KENDALL_HP_PREC_CAP = True
C.KENDALL_STAGED_TRAINING = False
C.DET_OHEM_ENABLED = True
C.USE_LDAM_DRW = False
C.STAGED_TRAINING = False
C.BATCH_SIZE = 4
C.GRAD_ACCUM_STEPS = 1
C.NUM_WORKERS = 0
C.PIN_MEMORY = False
C.IMG_WIDTH = 1280
C.IMG_HEIGHT = 720
C.IMG_SIZE = (C.IMG_WIDTH, C.IMG_HEIGHT)

from models import model as model_module
from training import losses as losses_module
from data.industreal_dataset import IndustRealMultiTaskDataset


def parse_args():
    p = argparse.ArgumentParser(description='50-image cls-only overfit experiment')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--n-images', type=int, default=50, help='Number of training images')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=10)
    return p.parse_args()


def build_model(device):
    model = model_module.POPWMultiTaskModel(
        backbone_type=C.BACKBONE,
        pretrained=False,
        use_videomae=False,
    ).to(device)
    model.train()

    # Override all non-detection params to eval mode (no grad)
    for name, param in model.named_parameters():
        if not any(det_key in name for det_key in
                   ['det_head', 'detection_head', 'fpn', 'backbone']):
            param.requires_grad_(False)

    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Model: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable (det+FPN+backbone)')
    return model


def filter_detection_targets(batch, device):
    """Extract only detection targets from a dataset batch."""
    frames = batch['images']['rgb'].to(device).float().div(255.0)
    targets = []
    B = frames.size(0)
    for i in range(B):
        boxes = batch['gt_boxes']['rgb'][i].to(device)
        labels = batch['gt_classes']['rgb'][i].to(device)
        targets.append({'boxes': boxes, 'labels': labels})
    return frames, targets


def run_overfit(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 60)
    print('50-IMAGE CLS-ONLY OVERFIT EXPERIMENT (Opus v9 §R4)')
    print('=' * 60)
    print(f'  device={args.device}  lr={args.lr}  epochs={args.epochs}  n_images={args.n_images}')
    print(f'  DET_POS_IOU_THRESH={C.DET_POS_IOU_THRESH}  DET_POS_IOU_TOP_K={C.DET_POS_IOU_TOP_K}')
    print(f'  DET_POS_IOU_IOU_FLOOR={C.DET_POS_IOU_IOU_FLOOR}')
    print(f'  DET_OHEM_ENABLED={C.DET_OHEM_ENABLED}  KENDALL_FIXED_WEIGHTS={C.KENDALL_FIXED_WEIGHTS}')

    # Load dataset (all recordings to find images with GT)
    print('\n[1] Loading full training dataset...')
    ds_train = IndustRealMultiTaskDataset(split='train', augment=False)
    ds_val = IndustRealMultiTaskDataset(split='val', augment=False)
    print(f'  Full train: {len(ds_train)} samples, val: {len(ds_val)} samples')

    # Filter to images with ≥1 detection GT box
    print(f'\n[2] Selecting {args.n_images} images with GT boxes...')
    indices_with_gt = []
    for idx in range(len(ds_train)):
        sample = ds_train[idx]
        gt_boxes = sample['gt_boxes']['rgb']
        if gt_boxes is not None and len(gt_boxes) > 0:
            indices_with_gt.append(idx)
        if len(indices_with_gt) >= args.n_images:
            break

    if len(indices_with_gt) < args.n_images:
        print(f'  WARNING: only found {len(indices_with_gt)} images with GT, using all')
        indices_with_gt = indices_with_gt

    train_indices = indices_with_gt[:args.n_images]
    print(f'  Using {len(train_indices)} training images (indices: {train_indices[:5]}...), '
          f'val: {min(len(ds_val), 10)} images')

    # Override dataset to subset
    class SubsetDataset:
        def __init__(self, ds, indices):
            self.ds = ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.ds[self.indices[i]]

    train_ds = SubsetDataset(ds_train, train_indices)
    val_ds = SubsetDataset(ds_val, list(range(min(len(ds_val), 10))))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=lambda x: x[0] if len(x) == 1 else x,
    )

    # Build model, loss, optimizer
    print('\n[3] Building model...')
    model = build_model(args.device)
    model.train()

    # Find cls_score for direct weight-norm logging
    cls_score_module = None
    for name, module in model.named_modules():
        if name.endswith('cls_score'):
            cls_score_module = module
            break
    if cls_score_module is not None:
        init_norm = cls_score_module.weight.detach().norm().item()
        print(f'  cls_score.weight.norm() at init: {init_norm:.4f}')

    print('\n[4] Building loss...')
    loss_fn = losses_module.MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=True,
        train_pose=False,
        train_act=False,
        train_psr=False,
        use_kendall=True,
    ).to(args.device)

    # Set Kendall fixed weights for det-only (no head_pose)
    loss_fn.train_det = True
    loss_fn.train_pose = False
    loss_fn.train_act = False
    loss_fn.train_psr = False
    loss_fn.use_kendall = True

    print('\n[5] Building optimizer (AdamW, det params only)...')
    det_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and any(dk in n for dk in ['det_head', 'detection_head', 'fpn'])]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'backbone' in n]
    optimizer = optim.AdamW([
        {'params': det_params, 'lr': args.lr, 'weight_decay': 1e-4},
        {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': 1e-4},
    ])
    print(f'  Det params: {sum(p.numel() for p in det_params)/1e3:.1f}K, '
          f'Backbone params: {sum(p.numel() for p in backbone_params)/1e6:.2f}M')

    # Training loop
    print(f'\n[6] Training for {args.epochs} epochs...')
    print(f'  {"Epoch":>5} | {"Loss":>8} | {"ClsL":>8} | {"RegL":>8} | '
          f'{"PosN":>5} | {"PosScore":>20} | {"ClsWN":>7} | {"Time":>6}')
    print('-' * 80)

    history = {'loss': [], 'cls_loss': [], 'reg_loss': [], 'cls_w_norm': [],
               'pos_score_mean': [], 'pos_score_max': [], 'pos_n': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_reg = 0.0
        epoch_pos_score_mean = 0.0
        epoch_pos_score_max = 0.0
        epoch_pos_n = 0
        n_batches = 0
        cls_w_norm = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, list):
                batch = batch[0]
            frames, det_targets = filter_detection_targets(batch, args.device)

            # Skip batches with no GT boxes in any image
            total_gt = sum(t['boxes'].shape[0] for t in det_targets)
            if total_gt == 0:
                continue

            optimizer.zero_grad()
            outputs = model(frames)

            # Ensure float
            for key in ['cls_preds', 'reg_preds']:
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].float()

            # Build targets dict for MultiTaskLoss
            targets = {
                'detection': det_targets,
                'keypoints': torch.zeros(frames.size(0), 17, 2, device=args.device),
                'pose_confidence': torch.zeros(frames.size(0), 17, device=args.device),
                'head_pose': torch.zeros(frames.size(0), 9, device=args.device),
                'activity': torch.zeros(frames.size(0), dtype=torch.long, device=args.device),
                'psr_labels': torch.zeros(frames.size(0), 11, device=args.device),
            }

            total_loss, loss_dict = loss_fn(outputs, targets)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            optimizer.step()

            # Log probe: positive-anchor scores from FocalLoss internal probe
            # (already logged via POS_ANCHOR_PROBE in losses.py)
            cls_loss_i = loss_dict.get('det_cls', 0)
            reg_loss_i = loss_dict.get('det_reg', 0)

            # cls_score weight norm
            if cls_score_module is not None:
                cls_w_norm = cls_score_module.weight.detach().norm().item()

            # Count positives via anchor matching (re-run on last image in batch)
            with torch.no_grad():
                anchors = outputs.get('anchors', None)
                if anchors is not None and det_targets[0]['boxes'].shape[0] > 0:
                    focal_loss = None
                    for m in model.modules():
                        if isinstance(m, losses_module.FocalLoss):
                            focal_loss = m
                            break
                    if focal_loss is not None:
                        _pos_n = 0
                        _pos_scores = []
                        for bi in range(min(1, frames.size(0))):
                            _boxes = det_targets[bi]['boxes']
                            _labels = det_targets[bi]['labels']
                            if _boxes.shape[0] > 0:
                                _ml, _ = focal_loss._match_anchors(anchors, _boxes, _labels)
                                _pm = _ml >= 0
                                _pos_n += _pm.sum().item()
                                if _pm.sum() > 0:
                                    _valid = _pm | (_ml == -2)
                                    _cp = outputs['cls_preds'][bi][_valid]
                                    _piv = _pm[_valid]
                                    _p_scores = torch.sigmoid(_cp[_piv])
                                    _gt_scores = _p_scores.gather(
                                        1, _ml[_valid][_piv].unsqueeze(1)).squeeze(1)
                                    _pos_scores.append(_gt_scores)
                        if _pos_scores:
                            _all_scores = torch.cat(_pos_scores)
                            epoch_pos_score_mean += _all_scores.mean().item()
                            epoch_pos_score_max += _all_scores.max().item()
                            epoch_pos_n += 1

            epoch_loss += total_loss.item()
            epoch_cls += cls_loss_i if isinstance(cls_loss_i, (int, float)) else cls_loss_i.item()
            epoch_reg += reg_loss_i if isinstance(reg_loss_i, (int, float)) else reg_loss_i.item()
            n_batches += 1

        if n_batches == 0:
            continue

        epoch_loss /= n_batches
        epoch_cls /= n_batches
        epoch_reg /= n_batches
        ps_mean = epoch_pos_score_mean / max(epoch_pos_n, 1)
        ps_max = epoch_pos_score_max / max(epoch_pos_n, 1)
        dt = time.time() - t0

        history['loss'].append(epoch_loss)
        history['cls_loss'].append(epoch_cls)
        history['reg_loss'].append(epoch_reg)
        history['cls_w_norm'].append(cls_w_norm)
        history['pos_score_mean'].append(ps_mean)
        history['pos_score_max'].append(ps_max)
        history['pos_n'].append(epoch_pos_n)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'  {epoch:>5d} | {epoch_loss:>8.4f} | {epoch_cls:>8.4f} | {epoch_reg:>8.4f} | '
                  f'{epoch_pos_n:>5d} | {ps_mean:>.4f}/{ps_max:>.4f} | {cls_w_norm:>7.4f} | {dt:>5.1f}s')

            # Early success: cls_loss < 0.01 and pos_score > 0.95
            if epoch_cls < 0.01 and ps_mean > 0.95:
                print(f'\n  ✅ EARLY SUCCESS at epoch {epoch}: cls_loss={epoch_cls:.4f}, '
                      f'pos_score_mean={ps_mean:.4f}')
                print(f'     The detection classifier CAN overfit 50 images.')
                break

    # Final results
    print('\n' + '=' * 60)
    last_loss = history['loss'][-1]
    last_cls = history['cls_loss'][-1]
    last_ps_mean = history['pos_score_mean'][-1]
    last_cls_w = history['cls_w_norm'][-1]
    print(f'RESULTS:')
    print(f'  Final loss:       {last_loss:.4f}')
    print(f'  Final cls loss:   {last_cls:.4f}')
    print(f'  Final pos score:  {last_ps_mean:.4f} (mean on positives)')
    print(f'  cls_w_norm:       {last_cls_w:.4f}')
    print(f'  Epochs:           {len(history["loss"])}')

    if last_cls < 0.05 and last_ps_mean > 0.90:
        verdict = 'PASS — cls head CAN overfit. RF2 issue is data supply (gradient starvation).'
    elif last_cls < 0.5 and last_ps_mean > 0.5:
        verdict = 'WEAK PASS — cls head learns but slowly. Check LR / OHEM / IoU floor settings.'
    else:
        verdict = 'FAIL — cls head cannot overfit. Architectural or label-noise problem.'

    print(f'\n  VERDICT: {verdict}')

    # Save results
    out = {
        'config': {
            'DET_POS_IOU_THRESH': C.DET_POS_IOU_THRESH,
            'DET_POS_IOU_TOP_K': C.DET_POS_IOU_TOP_K,
            'DET_POS_IOU_IOU_FLOOR': C.DET_POS_IOU_IOU_FLOOR,
            'DET_OHEM_ENABLED': C.DET_OHEM_ENABLED,
            'lr': args.lr,
            'n_images': args.n_images,
            'epochs': args.epochs,
        },
        'verdict': verdict,
        'final_cls_loss': last_cls,
        'final_pos_score_mean': last_ps_mean,
        'final_cls_w_norm': last_cls_w,
        'history': {k: (v if isinstance(v, list) else list(v)) for k, v in history.items()},
    }
    out_path = os.path.join(WORK_DIR, 'src', 'runs', 'overfit_50img_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n  Results saved to: {out_path}')

    return last_cls < 0.5


if __name__ == '__main__':
    args = parse_args()
    success = run_overfit(args)
    sys.exit(0 if success else 1)
