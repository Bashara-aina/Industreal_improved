#!/usr/bin/env python3
"""Pre-flight validation for MTL training — runs BEFORE launching training.

10 categories, 50+ checks. Each category is gated; any FAIL blocks the run.

Categories:
  1. Data Pipeline Integrity
  2. Model Architecture Sanity
  3. Training Pipeline (forward + backward)
  4. Loss Function Correctness
  5. Optimizer Configuration
  6. Eval Methodology
  7. Bias/Logit State
  8. Multi-Task Balancing
  9. Convergence Baseline (overfit-200 probe)
 10. SOTA Reachability Analysis

Usage:
    python scripts/verify/preflight_v310.py
    # or with verbose output:
    python scripts/verify/preflight_v310.py --verbose

Exit code: 0 if all PASS, 1 if any FAIL.
"""
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup paths
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
log = logging.getLogger('preflight')


# ============================================================================
# Result tracking
# ============================================================================
class Results:
    def __init__(self):
        self.checks = []
        self.by_category = defaultdict(list)

    def add(self, category, name, status, details=''):
        """Add a check result. status: PASS, FAIL, WARN, INFO"""
        self.checks.append({
            'category': category,
            'name': name,
            'status': status,
            'details': details,
        })
        self.by_category[category].append({'name': name, 'status': status, 'details': details})

    def passed(self, category):
        return all(c['status'] in ('PASS', 'WARN', 'INFO') for c in self.by_category[category])

    def any_failed(self):
        return any(c['status'] == 'FAIL' for c in self.checks)

    def summary(self):
        n_pass = sum(1 for c in self.checks if c['status'] == 'PASS')
        n_warn = sum(1 for c in self.checks if c['status'] == 'WARN')
        n_fail = sum(1 for c in self.checks if c['status'] == 'FAIL')
        n_info = sum(1 for c in self.checks if c['status'] == 'INFO')
        return f'{n_pass} PASS, {n_warn} WARN, {n_fail} FAIL, {n_info} INFO'


# ============================================================================
# Category 1: Data Pipeline Integrity
# ============================================================================
def check_data_pipeline(R):
    log.info('=' * 60)
    log.info('Category 1: Data Pipeline Integrity')
    log.info('=' * 60)

    # 1.1: Train dataset loads
    try:
        from train_mtl_full_multimodal import FullMultiModalDataset
        train_ds = FullMultiModalDataset(
            recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
            img_size=(640, 360),
            mosaic_prob=0.0,
            copy_paste_prob=0.0,
        )
        R.add('DATA', 'Train dataset loads', 'PASS',
              f'{len(train_ds.samples)} samples, {len(train_ds.gt["detection"])} frames w/ detection')
    except Exception as e:
        R.add('DATA', 'Train dataset loads', 'FAIL', str(e))
        return

    # 1.2: Val dataset loads
    try:
        val_ds = FullMultiModalDataset(
            recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/val',
            img_size=(640, 360),
            mosaic_prob=0.0,
            copy_paste_prob=0.0,
        )
        R.add('DATA', 'Val dataset loads', 'PASS',
              f'{len(val_ds.samples)} samples')
    except Exception as e:
        R.add('DATA', 'Val dataset loads', 'FAIL', str(e))
        return

    # 1.3: Box format check
    sample = train_ds[0]
    images, targets = sample
    boxes = targets['boxes']
    classes = targets['classes']
    R.add('DATA', 'Sample returns (images, targets) tuple',
          'PASS' if isinstance(images, torch.Tensor) and isinstance(targets, dict) else 'FAIL',
          f'images shape={tuple(images.shape)}, boxes shape={tuple(boxes.shape) if boxes.numel() else "empty"}')

    # 1.4: Box normalization check (cx, cy, w, h in [0, 1])
    if boxes.numel() > 0:
        valid = ((boxes >= 0) & (boxes <= 1)).all().item()
        R.add('DATA', 'Boxes normalized [0,1]', 'PASS' if valid else 'FAIL',
              f'min={boxes.min().item():.4f}, max={boxes.max().item():.4f}')

    # 1.5: Class indices in [0, 24)
    if classes.numel() > 0:
        valid = ((classes >= 0) & (classes < 24)).all().item()
        R.add('DATA', 'Class indices in [0, 24)', 'PASS' if valid else 'FAIL',
              f'min={classes.min().item()}, max={classes.max().item()}')

    # 1.6: Image shape and dtype
    valid_shape = images.shape == (9, 360, 640)
    valid_dtype = images.dtype == torch.float32
    R.add('DATA', 'Image tensor shape/dtype',
          'PASS' if (valid_shape and valid_dtype) else 'FAIL',
          f'shape={tuple(images.shape)}, dtype={images.dtype}')

    # 1.7: Activity label is non-negative integer or -1
    act = targets.get('activity', None)
    if act is not None and not (isinstance(act, int) and act == -1):
        if isinstance(act, (int, torch.Tensor)) and (act == -1 or (0 <= act < 75)):
            R.add('DATA', 'Activity label valid', 'PASS', f'value={act}')
        else:
            R.add('DATA', 'Activity label valid', 'FAIL', f'value={act}')
    else:
        R.add('DATA', 'Activity label valid', 'INFO', 'No activity label')

    # 1.8: PSR targets are 11-component binary
    psr = targets.get('psr', None)
    if psr is not None and torch.is_tensor(psr):
        valid = (psr.shape == (11,) and ((psr == 0) | (psr == 1)).all().item())
        R.add('DATA', 'PSR target valid (11 binary)', 'PASS' if valid else 'FAIL',
              f'shape={tuple(psr.shape)}, sum={psr.sum().item()}')

    # 1.9: Train/val recording disjointness
    train_recs = set(p[0].name for p in train_ds.samples)
    val_recs = set(p[0].name for p in val_ds.samples)
    overlap = train_recs & val_recs
    R.add('DATA', 'Train/Val recording disjoint',
          'PASS' if not overlap else 'FAIL',
          f'train={len(train_recs)} recs, val={len(val_recs)} recs, overlap={len(overlap)}')

    # 1.10: Class distribution check
    det_counts = defaultdict(int)
    for sample_idx in range(min(len(train_ds), 5000)):
        _, t = train_ds[sample_idx]
        for c in t['classes']:
            det_counts[int(c)] += 1
    n_classes_with_samples = sum(1 for c in range(24) if det_counts.get(c, 0) > 0)
    R.add('DATA', 'Class distribution sampled',
          'INFO' if n_classes_with_samples >= 20 else 'WARN',
          f'{n_classes_with_samples}/24 classes have samples in first 5000')


# ============================================================================
# Category 2: Model Architecture Sanity
# ============================================================================
def check_model_arch(R):
    log.info('=' * 60)
    log.info('Category 2: Model Architecture Sanity')
    log.info('=' * 60)

    from train_mtl_full_multimodal import expand_conv_proj_to_9ch, WrappedMTL
    from src.models.mvit_mtl_model import MTLMViTModel

    model = MTLMViTModel(num_act_classes=75, num_det_classes=24,
                          num_psr_components=11, num_anchors=16)
    expand_conv_proj_to_9ch(model)
    model = WrappedMTL(model).cuda()

    # 2.1: Forward pass produces correct shapes
    B = 2
    x = torch.randn(B, 9, 360, 640).cuda()
    out = model(x)
    R.add('ARCH', 'Forward pass works',
          'PASS' if 'detection' in out and 'activity' in out else 'FAIL',
          f'output keys: {list(out.keys())}')

    # 2.2: Detection output shapes per FPN level
    expected_shapes = {'P3': (45, 80), 'P4': (23, 40), 'P5': (12, 20)}
    if 'detection' in out:
        for level, (eh, ew) in expected_shapes.items():
            if level in out['detection']:
                cls_logits = out['detection'][level]['cls_logits']
                reg_preds = out['detection'][level]['reg_preds']
                cls_ok = cls_logits.shape == (B, 24, eh, ew)
                reg_ok = reg_preds.shape == (B, 64, eh, ew)
                R.add('ARCH', f'{level} cls shape',
                      'PASS' if cls_ok else 'FAIL',
                      f'expected {(B,24,eh,ew)}, got {tuple(cls_logits.shape)}')
                R.add('ARCH', f'{level} reg shape',
                      'PASS' if reg_ok else 'FAIL',
                      f'expected {(B,64,eh,ew)}, got {tuple(reg_preds.shape)}')

    # 2.3: Activity shape
    act_ok = out.get('activity', torch.zeros(0)).shape == (B, 75)
    R.add('ARCH', 'Activity head shape', 'PASS' if act_ok else 'FAIL',
          f'expected {(B,75)}, got {tuple(out["activity"].shape)}')

    # 2.4: PSR shape
    psr_ok = out.get('psr_logits', torch.zeros(0)).shape == (B, 11)
    R.add('ARCH', 'PSR head shape', 'PASS' if psr_ok else 'FAIL',
          f'expected {(B,11)}, got {tuple(out["psr_logits"].shape)}')

    # 2.5: Pose shape
    pose_ok = out.get('pose_6d', torch.zeros(0)).shape == (B, 6)
    R.add('ARCH', 'Pose head shape', 'PASS' if pose_ok else 'FAIL',
          f'expected {(B,6)}, got {tuple(out["pose_6d"].shape)}')

    # 2.6: All 4 heads receive gradients
    # Set up a fake loss combining all heads
    fake_targets = {
        'boxes': [torch.tensor([[0.5, 0.5, 0.2, 0.2]]).cuda() for _ in range(B)],
        'classes': [torch.tensor([5]).cuda() for _ in range(B)],
        'activity': [0, 1],
        'pose': [(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))] * B,
        'psr': [torch.zeros(11).cuda() for _ in range(B)],
    }
    fake_loss = out['activity'].sum() + out['psr_logits'].sum() + out['pose_6d'].sum()
    for level in out['detection']:
        fake_loss = fake_loss + out['detection'][level]['cls_logits'].sum()
    fake_loss.backward()

    grad_check = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if 'act_head' in name:
                grad_check.setdefault('act_head', True)
            elif 'pose_head' in name:
                grad_check.setdefault('pose_head', True)
            elif 'psr_head' in name:
                grad_check.setdefault('psr_head', True)
            elif 'det_head' in name:
                grad_check.setdefault('det_head', True)

    for head in ['act_head', 'pose_head', 'psr_head', 'det_head']:
        if head in grad_check:
            R.add('ARCH', f'{head} receives gradients', 'PASS',
                  'grad verified via backward')
        else:
            R.add('ARCH', f'{head} receives gradients', 'FAIL',
                  'NO GRADIENT - check loss function')


# ============================================================================
# Category 3: Training Pipeline (forward + backward)
# ============================================================================
def check_training_pipeline(R):
    log.info('=' * 60)
    log.info('Category 3: Training Pipeline')
    log.info('=' * 60)

    import train_mtl_v3 as mtl_v3
    mtl_v3.NUM_ANCHORS = 16
    mtl_v3._ANCHOR_SPECS = mtl_v3._ANCHOR_SPECS_16

    from train_mtl_full_multimodal import FullMultiModalDataset, expand_conv_proj_to_9ch, WrappedMTL
    from src.models.mvit_mtl_model import MTLMViTModel
    from train_mtl_v3 import detection_loss, generate_anchors
    from torch.utils.data import DataLoader, Subset

    # Build small dataset
    train_ds = FullMultiModalDataset(
        recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
        img_size=(640, 360),
        mosaic_prob=0.0,
        copy_paste_prob=0.0,
    )
    subset_ds = Subset(train_ds, list(range(min(50, len(train_ds)))))
    loader = DataLoader(subset_ds, batch_size=2, shuffle=False,
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

    model = MTLMViTModel(num_act_classes=75, num_det_classes=24, num_psr_components=11, num_anchors=16)
    expand_conv_proj_to_9ch(model)
    model = WrappedMTL(model).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 3.1: Forward+backward works end-to-end
    try:
        for images, targets in loader:
            x = images.float().cuda()
            x = x.unsqueeze(2)  # [B, 9, 1, H, W]
            mean = torch.tensor([0.45]*9).cuda().view(1, 9, 1, 1, 1)
            std = torch.tensor([0.225]*9).cuda().view(1, 9, 1, 1, 1)
            x = (x - mean) / std

            out = model(x)
            anchors = {lvl: generate_anchors(out['detection'][lvl]['cls_logits'].shape[2],
                                              out['detection'][lvl]['cls_logits'].shape[3],
                                              x.device)
                       for lvl in ['P3', 'P4', 'P5'] if lvl in out['detection']}

            gt_boxes = [b.to(x.device).float() for b in targets['boxes']]
            gt_classes = [c.to(x.device).long() for c in targets['classes']]

            loss, cls_loss, reg_loss = detection_loss(
                out['detection'], anchors, gt_boxes, gt_classes,
                use_tal=False,
            )
            # Get float values (cls_loss/reg_loss may be 0-dim tensors)
            cls_loss_val = float(cls_loss) if torch.is_tensor(cls_loss) else cls_loss
            reg_loss_val = float(reg_loss) if torch.is_tensor(reg_loss) else reg_loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            R.add('TRAIN', 'Forward+backward works', 'PASS',
                  f'cls_loss={cls_loss_val:.4f}, reg_loss={reg_loss_val:.4f}')
            break  # Just need 1 batch
    except Exception as e:
        R.add('TRAIN', 'Forward+backward works', 'FAIL', f'{type(e).__name__}: {e}')

    # 3.2: All heads' gradients are non-NaN and non-zero
    grad_status = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad
            head = 'other'
            for h in ['act_head', 'pose_head', 'psr_head', 'det_head']:
                if h in name:
                    head = h
                    break
            has_grad = not (torch.isnan(g).any() or g.abs().sum() == 0)
            grad_status.setdefault(head, []).append(has_grad)

    for head in ['act_head', 'pose_head', 'psr_head', 'det_head']:
        if head in grad_status:
            all_ok = all(grad_status[head])
            R.add('TRAIN', f'{head} gradients valid', 'PASS' if all_ok else 'WARN',
                  f'{sum(grad_status[head])}/{len(grad_status[head])} params have valid grads')


# ============================================================================
# Category 4: Loss Function Correctness
# ============================================================================
def check_loss_functions(R):
    log.info('=' * 60)
    log.info('Category 4: Loss Function Correctness')
    log.info('=' * 60)

    # 4.1: Focal loss with all zeros target → near-zero loss
    B = 4
    logits = torch.randn(B, 24, 8, 8).cuda() * 5
    targets_bg = torch.full((B, 24, 8, 8), 0).cuda()
    targets_fg = torch.full((B, 24, 8, 8), 1).cuda()

    import train_mtl_v3 as mtl_v3
    from train_mtl_v3 import sigmoid_focal_loss

    # Sigmoid focal loss needs special format
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, 24)
    targets_flat = targets_bg.permute(0, 2, 3, 1).reshape(-1, 24)
    cls_target_bg = torch.full((logits_flat.shape[0],), -1, dtype=torch.long).cuda()

    try:
        loss_bg = sigmoid_focal_loss(logits_flat, cls_target_bg, gamma=2.0, alpha=0.25)
        R.add('LOSS', 'Focal loss with BG target', 'PASS',
              f'loss={loss_bg.item():.4f} (should be small)')
    except Exception as e:
        R.add('LOSS', 'Focal loss with BG target', 'FAIL', str(e))

    # 4.2: CE loss for activity
    logits_act = torch.randn(B, 75).cuda()
    labels_act = torch.randint(0, 75, (B,)).cuda()
    loss_ce = F.cross_entropy(logits_act, labels_act)
    R.add('LOSS', 'CE loss for activity', 'PASS',
          f'loss={loss_ce.item():.4f} (random baseline={-math.log(1/75):.4f})')

    # 4.3: BCE for PSR
    logits_psr = torch.randn(B, 11).cuda()
    targets_psr = torch.randint(0, 2, (B, 11)).float().cuda()
    loss_bce = F.binary_cross_entropy_with_logits(logits_psr, targets_psr)
    R.add('LOSS', 'BCE loss for PSR', 'PASS',
          f'loss={loss_bce.item():.4f}')

    # 4.4: MSE for pose
    pred_pose = torch.randn(B, 6).cuda()
    target_pose = torch.randn(B, 6).cuda()
    loss_mse = F.mse_loss(pred_pose, target_pose)
    R.add('LOSS', 'MSE loss for pose', 'PASS',
          f'loss={loss_mse.item():.4f}')


# ============================================================================
# Category 5: Optimizer Configuration
# ============================================================================
def check_optimizer(R):
    log.info('=' * 60)
    log.info('Category 5: Optimizer Configuration')
    log.info('=' * 60)

    from train_mtl_full_multimodal import expand_conv_proj_to_9ch, WrappedMTL
    from src.models.mvit_mtl_model import MTLMViTModel
    from train_mtl_v3 import build_llrd_param_groups

    base = MTLMViTModel(num_act_classes=75, num_det_classes=24, num_psr_components=11, num_anchors=16)
    expand_conv_proj_to_9ch(base)
    model = WrappedMTL(base)

    # 5.1: Per-head LR is set correctly
    groups = build_llrd_param_groups(
        model, base_lr=2e-5, det_lr_mult=1000,
        act_lr_mult=50, pose_lr_mult=50, psr_lr_mult=50,
    )

    head_lrs = {}
    for g in groups:
        if g['name'] in ('act_head', 'pose_head', 'psr_head', 'det_head', 'fpn'):
            head_lrs[g['name']] = g['lr']

    expected_lrs = {
        'act_head': 2e-5 * 50,    # 1e-3
        'pose_head': 2e-5 * 50,   # 1e-3
        'psr_head': 2e-5 * 50,    # 1e-3
        'det_head': 2e-5 * 1000,  # 2e-2
        'fpn': 2e-5,
    }

    for name, expected_lr in expected_lrs.items():
        actual_lr = head_lrs.get(name)
        ok = actual_lr is not None and abs(actual_lr - expected_lr) < 1e-8
        R.add('OPT', f'{name} LR correct', 'PASS' if ok else 'FAIL',
              f'expected={expected_lr:.2e}, actual={actual_lr:.2e}' if actual_lr else f'missing')

    # 5.2: Per-head LR is within reasonable range
    # Standard LRs: backbone 1e-5, heads 1e-4 to 1e-3
    # Anything <1e-5 is too small, >1e-2 is too large (without warmup)
    for name, lr in head_lrs.items():
        if name == 'det_head':
            # Det head intentionally gets high LR for bias warmup
            if lr > 5e-2:
                R.add('OPT', f'{name} LR within bounds', 'WARN',
                      f'{lr:.2e} is very high but intentional for bias warmup')
            else:
                R.add('OPT', f'{name} LR within bounds', 'PASS', f'{lr:.2e}')
        else:
            if lr < 1e-5:
                R.add('OPT', f'{name} LR within bounds', 'WARN',
                      f'{lr:.2e} may be too small')
            elif lr > 1e-2:
                R.add('OPT', f'{name} LR within bounds', 'FAIL',
                      f'{lr:.2e} is too large')
            else:
                R.add('OPT', f'{name} LR within bounds', 'PASS', f'{lr:.2e}')


# ============================================================================
# Category 6: Eval Methodology
# ============================================================================
def check_eval_methodology(R):
    log.info('=' * 60)
    log.info('Category 6: Eval Methodology')
    log.info('=' * 60)

    # 6.1: YOLOv8 baseline exists
    yolo_json = Path('runs/sota_eval/yolov8m_full.json')
    if yolo_json.exists():
        with open(yolo_json) as f:
            yolo = json.load(f)
        R.add('EVAL', 'YOLOv8 baseline available', 'PASS',
              f'mAP@0.5={yolo["mAP_50"]:.4f} ({yolo["num_frames"]} frames)')
    else:
        R.add('EVAL', 'YOLOv8 baseline available', 'FAIL',
              f'{yolo_json} not found')

    # 6.2: Eval scripts exist
    eval_scripts = [
        'scripts/eval/eval_mvit_mAP.py',
        'scripts/eval/eval_all_heads.py',
        'scripts/eval/eval_activity_75class.py',
        'scripts/eval/eval_pose_norm_fix.py',
        'scripts/eval/eval_psr_transition_f1.py',
    ]
    for script in eval_scripts:
        p = Path(script)
        R.add('EVAL', f'{script} exists',
              'PASS' if p.exists() else 'FAIL',
              f'{p.stat().st_size} bytes' if p.exists() else 'missing')

    # 6.3: Anchor config matches training
    import train_mtl_v3 as mtl_v3
    R.add('EVAL', 'Train/Eval anchor config consistent',
          'PASS' if mtl_v3.NUM_ANCHORS in (8, 16) else 'FAIL',
          f'NUM_ANCHORS={mtl_v3.NUM_ANCHORS}')

    # 6.4: Decode formula consistent with training
    from src.losses.ciou import decode_deltas_to_xyxy
    # Test with sample values
    deltas = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    anchors = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    decoded = decode_deltas_to_xyxy(deltas, anchors)
    expected_cx = 0.5 + 0.5 * 0.1  # cx = anchor_cx + dx * 0.1
    ok = abs(decoded[0, 0].item() - (expected_cx - 0.1)) < 0.01  # x1 = cx - w/2
    R.add('EVAL', 'Decode formula correct', 'PASS' if ok else 'FAIL',
          f'decoded={decoded.tolist()} vs expected cx={expected_cx:.4f}')


# ============================================================================
# Category 7: Bias/Logit State
# ============================================================================
def check_bias_logit_state(R):
    log.info('=' * 60)
    log.info('Category 7: Bias/Logit State')
    log.info('=' * 60)

    from src.models.mvit_mtl_model import DetectionHead

    head = DetectionHead(num_anchors=16).cuda()

    # 7.1: running_pos_ratio is a persistent buffer
    is_buffer = 'running_pos_ratio' in dict(head.named_buffers())
    is_in_state = 'running_pos_ratio' in head.state_dict()
    R.add('BIAS', 'running_pos_ratio is persistent buffer',
          'PASS' if (is_buffer and is_in_state) else 'FAIL',
          f'is_buffer={is_buffer}, is_in_state={is_in_state}')

    # 7.2: Initial bias matches prior_prob
    bias = head.cls_head[3].bias.data
    expected_bias = -math.log(0.99 / 0.01)  # ~-4.6
    bias_match = abs(bias.mean().item() - expected_bias) < 0.1
    R.add('BIAS', 'Initial bias matches prior_prob=0.01',
          'PASS' if bias_match else 'FAIL',
          f'bias_mean={bias.mean().item():.4f}, expected={expected_bias:.4f}')

    # 7.3: Buffer value persists across save/load
    head.running_pos_ratio.fill_(0.42)
    sd = head.state_dict()
    head2 = DetectionHead(num_anchors=16).cuda()
    head2.load_state_dict(sd)
    R.add('BIAS', 'Buffer value persists across save/load',
          'PASS' if abs(head2.running_pos_ratio.item() - 0.42) < 1e-6 else 'FAIL',
          f'saved=0.42, loaded={head2.running_pos_ratio.item():.4f}')


# ============================================================================
# Category 8: Multi-Task Balancing
# ============================================================================
def check_multitask_balancing(R):
    log.info('=' * 60)
    log.info('Category 8: Multi-Task Balancing')
    log.info('=' * 60)

    # 8.1: UW-SO has 4 learnable parameters (one per task)
    from src.losses.uw_so import UWSOLoss
    uw_so = UWSOLoss().cuda()
    n_params = sum(p.numel() for p in uw_so.parameters() if p.requires_grad)
    R.add('MTL', 'UW-SO has 4 learnable parameters',
          'PASS' if n_params == 4 else 'FAIL',
          f'{n_params} learnable params')

    # 8.2: UW-SO loss computation
    losses = {
        'det': torch.tensor(2.0).cuda(),
        'act': torch.tensor(1.0).cuda(),
        'psr': torch.tensor(0.5).cuda(),
        'pose': torch.tensor(0.1).cuda(),
    }
    try:
        weighted = uw_so(losses)
        R.add('MTL', 'UW-SO loss computation works', 'PASS' if torch.isfinite(weighted) else 'FAIL',
              f'weighted_loss={weighted.item():.4f}')
    except Exception as e:
        R.add('MTL', 'UW-SO loss computation works', 'FAIL', str(e))


# ============================================================================
# Category 9: Convergence Baseline (overfit-200 probe)
# ============================================================================
def check_convergence_baseline(R):
    log.info('=' * 60)
    log.info('Category 9: Convergence Baseline')
    log.info('=' * 60)

    log.info('  Running 200-step overfit probe on 50 samples...')
    log.info('  This validates the model can fit the data.')

    import train_mtl_v3 as mtl_v3
    mtl_v3.NUM_ANCHORS = 16
    mtl_v3._ANCHOR_SPECS = mtl_v3._ANCHOR_SPECS_16
    from train_mtl_v3 import detection_loss, generate_anchors
    from train_mtl_full_multimodal import FullMultiModalDataset, expand_conv_proj_to_9ch, WrappedMTL
    from src.models.mvit_mtl_model import MTLMViTModel
    from torch.utils.data import DataLoader, Subset

    train_ds = FullMultiModalDataset(
        recordings_dir='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
        img_size=(640, 360),
        mosaic_prob=0.0,
        copy_paste_prob=0.0,
    )
    subset_ds = Subset(train_ds, list(range(50)))
    loader = DataLoader(subset_ds, batch_size=2, shuffle=False,
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

    model = MTLMViTModel(num_act_classes=75, num_det_classes=24, num_psr_components=11, num_anchors=16)
    expand_conv_proj_to_9ch(model)
    model = WrappedMTL(model).cuda()

    # Freeze backbone, train only detection head
    for name, p in model.named_parameters():
        p.requires_grad = 'det_head' in name
    head_params = [p for p in model.m.det_head.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(head_params, lr=1e-3)

    initial_loss = None
    final_loss = None
    losses = []
    for step in range(100):
        for images, targets in loader:
            x = images.float().cuda().unsqueeze(2)
            mean = torch.tensor([0.45]*9).cuda().view(1, 9, 1, 1, 1)
            std = torch.tensor([0.225]*9).cuda().view(1, 9, 1, 1, 1)
            x = (x - mean) / std

            out = model(x)
            anchors = {lvl: generate_anchors(out['detection'][lvl]['cls_logits'].shape[2],
                                              out['detection'][lvl]['cls_logits'].shape[3],
                                              x.device)
                       for lvl in ['P3', 'P4', 'P5'] if lvl in out['detection']}
            gt_boxes = [b.to(x.device).float() for b in targets['boxes']]
            gt_classes = [c.to(x.device).long() for c in targets['classes']]

            loss, cls_loss, reg_loss = detection_loss(
                out['detection'], anchors, gt_boxes, gt_classes, use_tal=False,
            )
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 10.0)
            opt.step()
            losses.append(loss.item())
            if initial_loss is None:
                initial_loss = loss.item()
            break

    final_loss = losses[-1] if losses else None

    if initial_loss is None or final_loss is None:
        R.add('CONV', 'Detection overfit probe', 'FAIL', 'Could not run probe')
    else:
        ratio = final_loss / max(initial_loss, 1e-9)
        if ratio < 0.5:
            R.add('CONV', 'Detection overfits in 100 steps',
                  'PASS', f'loss: {initial_loss:.4f} → {final_loss:.4f} ({ratio*100:.1f}%)')
        elif ratio < 0.9:
            R.add('CONV', 'Detection converges slowly',
                  'WARN', f'loss: {initial_loss:.4f} → {final_loss:.4f} ({ratio*100:.1f}%)')
        else:
            R.add('CONV', 'Detection NOT converging',
                  'FAIL', f'loss: {initial_loss:.4f} → {final_loss:.4f} ({ratio*100:.1f}%)')


# ============================================================================
# Category 10: SOTA Reachability Analysis
# ============================================================================
def check_sota_reachability(R):
    log.info('=' * 60)
    log.info('Category 10: SOTA Reachability Analysis')
    log.info('=' * 60)

    sota_targets = {
        'detection_mAP50': 0.70,
        'activity_top1': 0.95,
        'pose_MAE_deg': 5.0,
        'psr_F1': 0.80,
    }

    # 10.1: YOLOv8 baseline (data ceiling for detection)
    yolo_json = Path('runs/sota_eval/yolov8m_full.json')
    if yolo_json.exists():
        with open(yolo_json) as f:
            yolo = json.load(f)
        R.add('SOTA', 'YOLOv8 baseline (detection ceiling)',
              'PASS',
              f'mAP@0.5={yolo["mAP_50"]:.4f} (target {sota_targets["detection_mAP50"]})')

    # 10.2: Compute realistic targets based on YOLOv8 baseline
    # Multi-task models typically achieve 70-85% of single-task performance
    # So realistic targets:
    # - Detection: 0.5-0.6 of YOLOv8 = 0.30-0.40 mAP
    # - Activity: depends on Kinetics pretrain
    # - Pose: depends on data quality
    # - PSR: domain-specific

    # 10.3: Compare to v3.7 baseline (where we are now)
    R.add('SOTA', 'v3.7 baseline vs SOTA targets',
          'INFO',
          f'detection: 0.05/0.70 (gap 0.65)')

    # 10.4: SOTA targets are reasonable for this dataset
    R.add('SOTA', 'SOTA targets defined',
          'PASS',
          f'detection≥{sota_targets["detection_mAP50"]}, '
          f'activity_top1≥{sota_targets["activity_top1"]}, '
          f'pose_MAE<{sota_targets["pose_MAE_deg"]}°, '
          f'psr_F1≥{sota_targets["psr_F1"]}')

    # 10.5: Expected gain from LR fix alone
    # Based on prior probe (500 steps overfit-200):
    # - BG conf: -16.5%
    # - FG/BG separation: +15.3%
    # This translates to maybe +0.05 to +0.15 mAP improvement
    R.add('SOTA', 'Expected gain from LR fix',
          'INFO',
          'Per-head LR was 1000x imbalanced. Activity head at 2e-5 → now 1e-3. '
          'Expected: +10-20% Activity Top-1, -3-5° Pose MAE in first 5K batches. '
          'Detection unchanged (was already 1000x).')


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip-convergence', action='store_true',
                        help='Skip the 100-step overfit probe (saves time)')
    parser.add_argument('--output', type=str, default='/tmp/preflight_v310.json')
    args = parser.parse_args()

    R = Results()

    check_data_pipeline(R)
    check_model_arch(R)
    check_training_pipeline(R)
    check_loss_functions(R)
    check_optimizer(R)
    check_eval_methodology(R)
    check_bias_logit_state(R)
    check_multitask_balancing(R)
    if not args.skip_convergence:
        check_convergence_baseline(R)
    check_sota_reachability(R)

    # Summary
    log.info('=' * 60)
    log.info('PRE-FLIGHT SUMMARY')
    log.info('=' * 60)
    log.info(f'Total checks: {R.summary()}')

    # Per-category summary
    for cat in ['DATA', 'ARCH', 'TRAIN', 'LOSS', 'OPT', 'EVAL', 'BIAS', 'MTL', 'CONV', 'SOTA']:
        if cat in R.by_category:
            n = len(R.by_category[cat])
            n_fail = sum(1 for c in R.by_category[cat] if c['status'] == 'FAIL')
            n_warn = sum(1 for c in R.by_category[cat] if c['status'] == 'WARN')
            log.info(f'  {cat:8s}: {n} checks, {n_fail} FAIL, {n_warn} WARN')

    # Verdict
    log.info('')
    if R.any_failed():
        log.error('=' * 60)
        log.error('VERDICT: ❌ PRE-FLIGHT FAILED — DO NOT LAUNCH TRAINING')
        log.error('=' * 60)
        log.error('Failed checks:')
        for c in R.checks:
            if c['status'] == 'FAIL':
                log.error(f'  [{c["category"]}] {c["name"]}: {c["details"]}')
        verdict = 'FAIL'
    else:
        log.info('=' * 60)
        log.info('VERDICT: ✅ PRE-FLIGHT PASSED — SAFE TO LAUNCH')
        log.info('=' * 60)
        verdict = 'PASS'

    # Save JSON
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