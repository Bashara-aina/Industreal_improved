#!/usr/bin/env python3
"""Pre-flight validation for YOLOv8 distillation training.

Validates:
1. YOLOv8 model loads and runs inference correctly
2. Soft label generation works (boxes, classes, scores)
3. Distillation loss computation works
4. MTL model + distillation loss combines correctly
5. All 4 heads still get gradients
6. Per-head LR multipliers are correct
"""
import argparse
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
from PIL import Image

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('preflight_distill')


class Results:
    def __init__(self):
        self.checks = []
    def add(self, cat, name, status, details=''):
        self.checks.append({'category': cat, 'name': name, 'status': status, 'details': details})
    def any_failed(self):
        return any(c['status'] == 'FAIL' for c in self.checks)
    def summary(self):
        n_pass = sum(1 for c in self.checks if c['status'] == 'PASS')
        n_fail = sum(1 for c in self.checks if c['status'] == 'FAIL')
        n_warn = sum(1 for c in self.checks if c['status'] == 'WARN')
        n_info = sum(1 for c in self.checks if c['status'] == 'INFO')
        return f'{n_pass} PASS, {n_warn} WARN, {n_fail} FAIL, {n_info} INFO'


def check_yolov8_setup(R, weights_path):
    """YOLOv8 loads and runs inference."""
    log.info('=' * 60)
    log.info('Category 1: YOLOv8 Teacher Setup')
    log.info('=' * 60)

    # 1.1: Weights file exists
    wp = Path(weights_path)
    R.add('YOLO', 'Weights file exists',
          'PASS' if wp.exists() else 'FAIL',
          f'{wp} ({wp.stat().st_size // 1024 // 1024}MB)' if wp.exists() else 'missing')

    # 1.2: YOLOv8 loads
    try:
        from ultralytics import YOLO
        yolo = YOLO(weights_path)
        R.add('YOLO', 'YOLOv8 loads', 'PASS',
              f'{len(yolo.names)} classes')
    except Exception as e:
        R.add('YOLO', 'YOLOv8 loads', 'FAIL', str(e))
        return None

    # 1.3: YOLOv8 has 24 classes (matching our 24-class detection head)
    n_classes = len(yolo.names)
    R.add('YOLO', 'YOLOv8 has 24 classes',
          'PASS' if n_classes == 24 else 'WARN',
          f'{n_classes} classes')

    # 1.4: YOLOv8 runs inference on real image
    try:
        img_path = '/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/val/14_main_2_2/rgb/000000.jpg'
        img = Image.open(img_path).convert('RGB')
        result = yolo(img, verbose=False, conf=0.01)
        n_boxes = len(result[0].boxes) if result[0].boxes else 0
        R.add('YOLO', 'YOLOv8 inference on real image', 'PASS' if n_boxes >= 0 else 'FAIL',
              f'{n_boxes} boxes (>=0 expected)')
    except Exception as e:
        R.add('YOLO', 'YOLOv8 inference on real image', 'FAIL', str(e))

    return yolo


def check_distiller(R, weights_path):
    """Distiller generates correct soft labels."""
    log.info('=' * 60)
    log.info('Category 2: Distiller Soft Labels')
    log.info('=' * 60)

    try:
        from src.training.yolov8_distill import YOLOv8Distiller
        distiller = YOLOv8Distiller(weights_path=weights_path)
        R.add('DIST', 'YOLOv8Distiller instantiates', 'PASS', 'OK')
    except Exception as e:
        R.add('DIST', 'YOLOv8Distiller instantiates', 'FAIL', str(e))
        return None

    # 2.2: Soft labels from real image
    try:
        img_path = '/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/val/14_main_2_2/rgb/000000.jpg'
        img = Image.open(img_path).convert('RGB').resize((640, 360))
        import numpy as np
        img_arr = np.array(img)
        img_tensor = torch.from_numpy(img_arr).float().permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor - 0.45) / 0.225
        img_tensor = img_tensor.unsqueeze(0).cuda()

        labels = distiller.get_soft_labels(img_tensor, 640, 360)
        n_boxes = len(labels[0]['boxes'])
        R.add('DIST', 'Soft labels generated', 'PASS' if n_boxes > 0 else 'INFO',
              f'{n_boxes} boxes from real image')
    except Exception as e:
        R.add('DIST', 'Soft labels generated', 'FAIL', str(e))
        return None

    return distiller


def check_distill_loss(R):
    """Distillation loss computation is correct."""
    log.info('=' * 60)
    log.info('Category 3: Distillation Loss')
    log.info('=' * 60)

    from src.training.yolov8_distill import distill_loss

    # Mock data - cls_logits requires_grad so backward can flow
    B, H, W, A = 1, 8, 8, 4
    cls_logits = torch.randn(B, 24, H, W, requires_grad=True).cuda()  # [1, 24, 8, 8]
    reg_preds = torch.randn(B, 4*A, H, W).cuda()

    # Anchors at H=8, W=8 (use larger anchors so they overlap with boxes)
    ys = (torch.arange(H).float() + 0.5) / H
    xs = (torch.arange(W).float() + 0.5) / W
    anchors_list = []
    for y in range(H):
        for x in range(W):
            for a in range(A):
                anchors_list.append([xs[x].item(), ys[y].item(), 0.3, 0.3])
    anchors = torch.tensor(anchors_list).reshape(H, W, A, 4).cuda()

    # Mock soft labels - 2 boxes (large enough to overlap with anchors)
    soft_labels = [
        {
            'boxes': np.array([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]], dtype=np.float32),
            'classes': np.array([5, 10], dtype=np.int64),
            'scores': np.array([0.9, 0.8], dtype=np.float32),
        }
    ]

    try:
        loss = distill_loss(cls_logits, reg_preds, anchors, soft_labels,
                            img_w=640, img_h=360, distill_weight=1.0, score_thresh=0.3)
        R.add('DIST', 'Distill loss computation works', 'PASS' if torch.isfinite(loss) else 'FAIL',
              f'loss={loss.item():.4f}, requires_grad={loss.requires_grad}')
    except Exception as e:
        R.add('DIST', 'Distill loss computation works', 'FAIL', str(e))

    # 3.2: Backward works
    try:
        loss.backward()
        R.add('DIST', 'Distill loss backward works', 'PASS', 'OK')
    except Exception as e:
        R.add('DIST', 'Distill loss backward works', 'FAIL', str(e))


def check_combined_training(R, weights_path):
    """MTL + distillation loss: all 4 heads still train, no head crashes."""
    log.info('=' * 60)
    log.info('Category 4: Combined Training (MTL + Distillation)')
    log.info('=' * 60)

    # Build model
    import train_mtl_v3 as mtl_v3_mod
    mtl_v3_mod.NUM_ANCHORS = 16
    mtl_v3_mod._ANCHOR_SPECS = mtl_v3_mod._ANCHOR_SPECS_16
    from train_mtl_full_multimodal import expand_conv_proj_to_9ch, WrappedMTL
    from src.models.mvit_mtl_model import MTLMViTModel
    from src.training.yolov8_distill import YOLOv8Distiller, distill_loss

    model = MTLMViTModel(num_act_classes=75, num_det_classes=24, num_psr_components=11, num_anchors=16)
    expand_conv_proj_to_9ch(model)
    model = WrappedMTL(model).cuda()

    # Build distiller
    distiller = YOLOv8Distiller(weights_path=weights_path)

    # Build optimizer with per-head LR
    from train_mtl_v3 import build_llrd_param_groups, multi_task_loss_v3
    param_groups = build_llrd_param_groups(
        model, base_lr=2e-5, det_lr_mult=1000,
        act_lr_mult=50, pose_lr_mult=50, psr_lr_mult=50,
    )
    opt = torch.optim.AdamW(param_groups)

    # Mock batch with real data
    from train_mtl_full_multimodal import FullMultiModalDataset
    from torch.utils.data import DataLoader, Subset
    train_ds = FullMultiModalDataset(
        recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
        img_size=(640, 360),
        mosaic_prob=0.0,
        copy_paste_prob=0.0,
    )
    subset = Subset(train_ds, list(range(4)))
    loader = DataLoader(subset, batch_size=2, shuffle=False,
                        collate_fn=lambda b: (
                            torch.stack([x[0] for x in b]),
                            {
                                'boxes': [x[1]['boxes'] for x in b],
                                'classes': [x[1]['classes'] for x in b],
                                'activity': [x[1].get('activity', -1) for x in b],
                                'pose': [x[1].get('pose', None) for x in b],
                                'psr': [x[1].get('psr', None) for x in b],
                            }
                        ),
                        num_workers=0)

    n_steps = 0
    for images, targets in loader:
        n_steps += 1
        if n_steps > 3:
            break

        x = images.float().cuda().unsqueeze(2)  # [B, 9, 1, H, W]
        mean = torch.tensor([0.45]*9).cuda().view(1, 9, 1, 1, 1)
        std = torch.tensor([0.225]*9).cuda().view(1, 9, 1, 1, 1)
        x = (x - mean) / std

        # Move targets
        for k in targets:
            if isinstance(targets[k], list):
                targets[k] = [t.cuda() if torch.is_tensor(t) else t for t in targets[k]]

        # Forward
        out = model(x)

        # Anchors
        anchors_per_level = {}
        for level in ['P3', 'P4', 'P5']:
            if level in out['detection']:
                cls_l = out['detection'][level]['cls_logits']
                H, W = cls_l.shape[2], cls_l.shape[3]
                anchors_per_level[level] = mtl_v3_mod.generate_anchors(H, W, x.device)

        # Multi-task loss
        loss_mtl, lc = multi_task_loss_v3(
            out, targets, anchors_per_level,
            use_supcon=False, uw_so=None,
            loss_type='focal', matcher_type='iou',
            use_tal=False, tal_alpha=2.0,
        )

        # Distillation loss
        rgb = x[:, :3].squeeze(2)  # [B, 3, H, W]
        rgb_unnorm = (rgb * 0.225 + 0.45).clamp(0, 1)
        soft_labels = distiller.get_soft_labels(rgb_unnorm, 640, 360)

        distill_total = torch.tensor(0.0, device=x.device)
        n_distill = 0
        for level in ['P3', 'P4', 'P5']:
            if level not in out['detection']:
                continue
            cls_logits = out['detection'][level]['cls_logits']
            reg_preds = out['detection'][level]['reg_preds']
            H, W = cls_logits.shape[2], cls_logits.shape[3]
            anchors = anchors_per_level[level]
            d_loss = distill_loss(
                cls_logits, reg_preds, anchors, soft_labels,
                img_w=640, img_h=360,
                distill_weight=2.0, score_thresh=0.3,
            )
            distill_total = distill_total + d_loss
            n_distill += 1

        total_loss = loss_mtl + distill_total

        # Check losses are finite
        if not torch.isfinite(loss_mtl):
            R.add('COMBO', f'MTL loss finite (step {n_steps})', 'FAIL', f'{loss_mtl.item()}')
        if not torch.isfinite(distill_total):
            R.add('COMBO', f'Distill loss finite (step {n_steps})', 'FAIL', f'{distill_total.item()}')
        if not torch.isfinite(total_loss):
            R.add('COMBO', f'Total loss finite (step {n_steps})', 'FAIL', f'{total_loss.item()}')

        # Backward
        opt.zero_grad()
        total_loss.backward()

        # Check all 4 heads get gradients
        grad_heads = set()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                for h in ['act_head', 'pose_head', 'psr_head', 'det_head']:
                    if h in name:
                        grad_heads.add(h)
                        break

        missing_heads = {'act_head', 'pose_head', 'psr_head', 'det_head'} - grad_heads
        if missing_heads:
            R.add('COMBO', f'All 4 heads get grad (step {n_steps})', 'FAIL',
                  f'Missing: {missing_heads}')
        else:
            R.add('COMBO', f'All 4 heads get grad (step {n_steps})', 'PASS',
                  f'act/pose/psr/det all OK')

        opt.step()

    if n_steps == 3:
        R.add('COMBO', '3-batch training loop completes', 'PASS',
              '3 batches ran end-to-end without errors')


def check_per_head_lr(R):
    """Per-head LR multipliers are correct in distillation training."""
    log.info('=' * 60)
    log.info('Category 5: Per-head LR (Distillation)')
    log.info('=' * 60)

    from train_mtl_full_multimodal import expand_conv_proj_to_9ch, WrappedMTL
    from src.models.mvit_mtl_model import MTLMViTModel
    from train_mtl_v3 import build_llrd_param_groups

    model = MTLMViTModel(num_act_classes=75, num_det_classes=24, num_psr_components=11, num_anchors=16)
    expand_conv_proj_to_9ch(model)
    wrapped = WrappedMTL(model)

    groups = build_llrd_param_groups(
        wrapped, base_lr=2e-5, det_lr_mult=1000,
        act_lr_mult=50, pose_lr_mult=50, psr_lr_mult=50,
    )

    head_lrs = {g['name']: g['lr'] for g in groups if g['name'] in ('act_head', 'pose_head', 'psr_head', 'det_head', 'fpn')}

    expected = {
        'act_head': 2e-5 * 50,
        'pose_head': 2e-5 * 50,
        'psr_head': 2e-5 * 50,
        'det_head': 2e-5 * 1000,
        'fpn': 2e-5,
    }

    for name, expected_lr in expected.items():
        actual_lr = head_lrs.get(name)
        ok = actual_lr is not None and abs(actual_lr - expected_lr) < 1e-8
        R.add('LR', f'{name} LR correct', 'PASS' if ok else 'FAIL',
              f'expected={expected_lr:.2e}, actual={actual_lr:.2e}' if actual_lr else f'missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/assembly_state_detection_model_weights/asd_best_IndustRealandSynthetic.pt')
    parser.add_argument('--output', type=str, default='/tmp/preflight_distill.json')
    parser.add_argument('--skip-combined', action='store_true')
    args = parser.parse_args()

    R = Results()

    check_yolov8_setup(R, args.weights)
    distiller = check_distiller(R, args.weights)
    check_distill_loss(R)
    check_per_head_lr(R)
    if not args.skip_combined and distiller is not None:
        check_combined_training(R, args.weights)

    log.info('=' * 60)
    log.info('PRE-FLIGHT SUMMARY (Distillation)')
    log.info('=' * 60)
    log.info(f'Total checks: {R.summary()}')

    if R.any_failed():
        log.error('=' * 60)
        log.error('VERDICT: ❌ PRE-FLIGHT FAILED — DO NOT LAUNCH')
        log.error('=' * 60)
        for c in R.checks:
            if c['status'] == 'FAIL':
                log.error(f'  [{c["category"]}] {c["name"]}: {c["details"]}')
        verdict = 'FAIL'
    else:
        log.info('=' * 60)
        log.info('VERDICT: ✅ PRE-FLIGHT PASSED — SAFE TO LAUNCH')
        log.info('=' * 60)
        verdict = 'PASS'

    summary = {
        'verdict': verdict,
        'summary': R.summary(),
        'checks': R.checks,
    }
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f'Results saved: {args.output}')

    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()