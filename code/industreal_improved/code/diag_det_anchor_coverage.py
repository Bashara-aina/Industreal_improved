#!/usr/bin/env python3
"""D2 [opus RC-22] — Anchor coverage diagnostic.

For every val GT box, compute the MAXIMUM IoU against any anchor in the
AnchorGenerator output. This is the "identity regression upper bound": the
highest IoU any GT can ever achieve with a single anchor, before training.

Input : val split GT boxes
Output: p50/p90/p99 of best-anchor IoU + a histogram
Verdict:
    p50 < 0.5  ⇒ anchors are a binding constraint (act on RC-22:
                 change ANCHOR_SIZES to k-means centers).
    p50 > 0.6  ⇒ anchors fine, the problem is purely training/contamination.
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
from models.model import AnchorGenerator


def _box_iou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """b1: [N,4], b2: [M,4] in xyxy → [N,M] IoU."""
    a1 = b1[:, None, :2]
    a2 = b1[:, None, 2:]
    b1_ = b2[None, :, :2]
    b2_ = b2[None, :, 2:]
    inter_lt = torch.maximum(a1, b1_)
    inter_rb = torch.minimum(a2, b2_)
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (a2 - a1).prod(-1)
    area2 = (b2_ - b1_).prod(-1)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-9)


def main():
    ds = _ds_module.IndustRealMultiTaskDataset(
        split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    )
    print(f'[D2] val size: {len(ds)}')

    # Build a synthetic feature map dict sized to the model's FPN levels.
    # Use the actual model strides (8,16,32,64,128) and IMG_SIZE.
    # FPN heights: IMG_HEIGHT/stride, widths: IMG_WIDTH/stride.
    H, W = C.IMG_HEIGHT, C.IMG_WIDTH
    feats = {
        'p3': torch.zeros(1, 256, H // 8,  W // 8),
        'p4': torch.zeros(1, 256, H // 16, W // 16),
        'p5': torch.zeros(1, 256, H // 32, W // 32),
        'p6': torch.zeros(1, 256, H // 64, W // 64),
        'p7': torch.zeros(1, 256, H // 128, W // 128),
    }
    print('[D2] building anchors (≈ {} total)...'.format(
        sum(f.shape[2] * f.shape[3] * 9 for f in feats.values())))
    gen = AnchorGenerator()
    anchors = gen(feats)               # [A, 4] xyxy in pixel space
    print(f'[D2] anchors shape: {tuple(anchors.shape)}  (A=anchors per FPN level × 9 ratios/scales)')

    # Walk val, collect GT boxes.
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_ds_module.collate_fn)
    max_ious = []
    n_gt = 0
    n_with_pos_anchor = 0
    for i, batch in enumerate(loader):
        # batch is (images, targets_dict); targets_dict['detection'] is a list[dict] of len B
        targets = batch[1]
        for sample in targets['detection']:
            boxes = sample['boxes']
            if boxes.numel() == 0:
                continue
            boxes = boxes.float()
            n_gt += boxes.shape[0]
            iou = _box_iou(boxes, anchors)   # [N, A]
            best, _ = iou.max(dim=1)
            max_ious.extend(best.tolist())
            n_with_pos_anchor += int((best >= 0.5).sum().item())
        if i >= 500:                        # cap walk to 500 batches
            break

    arr = np.asarray(max_ious, dtype=np.float32)
    print(f'\n=== D2: Anchor coverage on {n_gt} val GT boxes (cap 500 batches) ===')
    print(f'  N GT            : {n_gt}')
    print(f'  p25 best-IoU    : {np.percentile(arr, 25):.4f}')
    print(f'  p50 best-IoU    : {np.percentile(arr, 50):.4f}')
    print(f'  p75 best-IoU    : {np.percentile(arr, 75):.4f}')
    print(f'  p90 best-IoU    : {np.percentile(arr, 90):.4f}')
    print(f'  p99 best-IoU    : {np.percentile(arr, 99):.4f}')
    print(f'  max best-IoU    : {arr.max():.4f}')
    print(f'  GT w/ IoU>=0.5  : {n_with_pos_anchor}/{n_gt}  ({100*n_with_pos_anchor/max(1,n_gt):.1f}%)')
    print(f'  GT w/ IoU>=0.3  : {int((arr >= 0.3).sum())}/{n_gt}  ({100*(arr >= 0.3).mean():.1f}%)')

    # Crude ASCII histogram
    print('\n  Histogram (best-IoU per GT):')
    bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist, edges = np.histogram(arr, bins=bins)
    width = 50
    max_h = max(hist.max(), 1)
    for j, h in enumerate(hist):
        bar = '█' * int(width * h / max_h)
        print(f'    [{edges[j]:.2f}-{edges[j+1]:.2f}) {h:6d}  {bar}')

    p50 = float(np.percentile(arr, 50))
    print()
    if p50 < 0.5:
        print(f'  ❌  ANCHOR BOTTLENECK CONFIRMED (p50={p50:.3f} < 0.5).')
        print('     ANCHOR_SIZES=(24,48,96,192,384) cannot reach IoU>=0.5 on the')
        print('     median GT box.  Consider k-means centers (≈64,128,192,288,416)')
        print('     — but only AFTER the heads are alive (RC-13/14/19).')
    else:
        print(f'  ✅  ANCHORS ADEQUATE (p50={p50:.3f} >= 0.5).')
        print('     The problem is not anchor sizes — focus on training/contamination.')


if __name__ == '__main__':
    main()
