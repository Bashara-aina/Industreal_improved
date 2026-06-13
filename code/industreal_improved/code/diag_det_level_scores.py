#!/usr/bin/env python3
"""D3 [opus RC-22] — Per-FPN-level detection score diagnostic.

For each FPN level p3..p7, compute per-level cls scores by running det_head's
internal cls_subnet+cls_score on each level's FPN feature. Also compute per-level
anchor best-IoU against GT boxes.

Verdict:
  - confident preds concentrated on p3/p4 with low best-IoU → cls head fires on wrong levels.
  - confident preds on all 5 levels with reasonable best-IoU → head alive.
"""
import os, sys
from pathlib import Path
import numpy as np

PROJ = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '4')

import torch
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
from models import model as _popw_model_module

CKPT = os.environ.get('CHECKPOINT', str(PROJ / 'src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth'))
MAX_BATCHES = int(os.environ.get('MAX_BATCHES', '10'))
BS = int(os.environ.get('EVAL_BS', '2'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FPN_LEVELS = ['p3', 'p4', 'p5', 'p6', 'p7']


def _box_iou(b1, b2):
    """IoU matrix: b1 [N,4] vs b2 [M,4] in xyxy → [N, M]."""
    a1, a2 = b1[:, None, :2], b1[:, None, 2:]
    b1_, b2_ = b2[None, :, :2], b2[None, :, 2:]
    inter_wh = (torch.minimum(a2, b2_) - torch.maximum(a1, b1_)).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (a2 - a1).prod(-1)
    area2 = (b2_ - b1_).prod(-1)
    return inter / (area1 + area2 - inter).clamp(min=1e-9)


def main():
    print(f'[D3] loading {CKPT}')
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'[D3] epoch={ckpt.get("epoch")} step={ckpt.get("step")} best={ckpt.get("best_metric", 0.0):.4f}')

    model = _popw_model_module.POPWMultiTaskModel(
        pretrained=False,
        backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
        use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),
        use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
        train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
    ).to(DEVICE)
    state = {k.replace('ema.', ''): v for k, v in ckpt['model'].items() if not k.startswith('ema.')}
    res = model.load_state_dict(state, strict=False)
    print(f'[D3] load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()

    val_ds = _ds_module.IndustRealMultiTaskDataset(
        split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BS, shuffle=False, num_workers=0,
        collate_fn=_ds_module.collate_fn, pin_memory=False, drop_last=False,
    )

    # Per-level accumulators
    level_score_count = {l: 0 for l in FPN_LEVELS}
    level_total       = {l: 0 for l in FPN_LEVELS}
    level_best_iou    = {l: [] for l in FPN_LEVELS}
    n_batches_with_gt = 0

    backbone_fn = model.backbone
    fpn_fn = model.fpn
    det_head = model.detection_head
    anchor_gen = model.anchor_gen

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            img = batch[0].to(DEVICE).float().div_(255.0)
            gt_list = [d['boxes'].to(DEVICE).float() for d in batch[1]['detection']]
            if all(g.numel() == 0 for g in gt_list):
                continue

            # Forward backbone + FPN
            c2, c3, c4, c5 = backbone_fn(img)
            features = fpn_fn(c3, c4, c5)

            # Get all anchors for anchor-IoU analysis
            anchors = anchor_gen(features).detach()  # [A, 4] xyxy pixel

            # Per-level anchor counts from AnchorGenerator internals
            # anchors are generated in FPN order: p3, p4, p5, p6, p7
            # Each level has H*W*9 anchors
            per_level_anchors = {}
            offset = 0
            for lvl in FPN_LEVELS:
                feat = features[lvl]
                _, _, H_l, W_l = feat.shape
                n_a = H_l * W_l * 9
                per_level_anchors[lvl] = anchors[offset:offset + n_a]
                offset += n_a

            B = img.shape[0]

            # Per-level cls: run det_head's internal subnet + score on each FPN level
            for lvl in FPN_LEVELS:
                feat = features[lvl]  # [B, 256, H_l, W_l]
                _, _, H_l, W_l = feat.shape

                # Run cls_subnet + cls_score on this level's feature
                cls_out = det_head.cls_score(det_head.cls_subnet(feat))  # [B, 9*C, H_l, W_l]
                # Reshape: [B, 9, C, H_l, W_l] → max over anchors and classes
                cls_out = cls_out.view(B, 9, -1, H_l, W_l)  # [B, 9, C, H_l, W_l]
                cls_max = torch.sigmoid(cls_out).flatten(2).max(dim=2)[0].max(dim=1)[0]  # [B, H_l*W_l]

                level_total[lvl] += int(cls_max.numel())
                level_score_count[lvl] += int((cls_max > 0.5).sum().item())

                # Anchor best-IoU for this level
                lvl_anchors = per_level_anchors[lvl]
                for b in range(B):
                    gt = gt_list[b]
                    if gt.numel() == 0:
                        continue
                    iou = _box_iou(gt, lvl_anchors)
                    best = iou.max(dim=1).values
                    level_best_iou[lvl].extend(best.cpu().tolist())

            n_batches_with_gt += 1
            if n_batches_with_gt >= MAX_BATCHES:
                break

    print(f'\n=== D3: Per-FPN-level detection on {n_batches_with_gt} GT-bearing batches ===\n')
    print(f'  {"level":<5} {"#locations":>12} {"#score>0.5":>12} {"frac":>8}  {"p50 IoU":>8} {"p90 IoU":>8} {"max IoU":>8}')
    print('  ' + '-' * 75)
    for lvl in FPN_LEVELS:
        arr = np.asarray(level_best_iou[lvl], dtype=np.float32) if level_best_iou[lvl] else np.array([0.0])
        n_total = max(1, level_total[lvl])
        n_pos = level_score_count[lvl]
        pct = n_pos / n_total * 100
        print(f'  {lvl:<5} {n_total:>12d} {n_pos:>12d} {pct:>7.2f}%  '
              f'{float(np.percentile(arr, 50)):>8.4f} {float(np.percentile(arr, 90)):>8.4f} {float(arr.max()):>8.4f}')

    # Verdict
    small = sum(level_score_count[l] for l in ('p3', 'p4'))
    big = sum(level_score_count[l] for l in ('p6', 'p7'))
    p67 = level_best_iou['p6'] + level_best_iou['p7']
    p50_big = float(np.percentile(np.asarray(p67, dtype=np.float32), 50)) if p67 else 0.0

    print()
    if sum(level_score_count.values()) == 0:
        print('  ⚠️  NO SCORES > 0.5 ANYWHERE — head is dead/collapsed. Re-eval after retrain.')
    elif small > 5 * max(1, big) and p50_big < 0.3:
        print(f'  ❌  CLS HEAD FIRES ON WRONG LEVELS (small={small}, big={big}, p50_p67={p50_big:.3f}).')
    else:
        print(f'  ✅  Scores distributed across levels; head is alive (small={small}, big={big}).')


if __name__ == '__main__':
    main()
