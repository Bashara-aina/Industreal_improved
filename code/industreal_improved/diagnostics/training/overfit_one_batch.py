#!/usr/bin/env python3
"""overfit_one_batch.py — Minimum Viable Experiment for the detection head.

WHAT IT PROVES
--------------
Before spending GPU-days on staged training, answer the only question that
matters: *can the model+loss localise objects on a SINGLE batch of GT-bearing
frames?* A correct architecture overfits one batch to near-zero loss in a few
hundred steps. If it can't, the bug is in the head / loss / anchor-matching —
NOT in the sampler, and no amount of GT oversampling will help.

This deliberately bypasses ALL the orchestration (stage_manager, supervisor,
EMA, Kendall, cron) and trains detection-only on one fixed batch, so the result
is unambiguous.

HOW TO READ THE OUTPUT
----------------------
  * det_cls and det_reg fall toward 0, and max(+logit) climbs well past +2:
        -> ARCHITECTURE OK. The death spiral is purely a data/sampling problem.
           Apply the DET_GT_FRAME_FRACTION fix (already wired into RF1-RF10)
           and re-run RF1.
  * Loss plateaus / max logit stays small / reg stuck at its floor:
        -> ARCHITECTURE BUG. Inspect, in order: anchor<->GT IoU matching
           (losses.py _match_anchors, the [0,1] normalisation), box decode
           (_decode_boxes), and the cls bias init. Use detection_collapse_probe.py
           to see best-IoU-vs-GT on this same batch.

USAGE
-----
  python overfit_one_batch.py                  # 300 steps, bs=4, lr=1e-3
  python overfit_one_batch.py --steps 600 --lr 2e-3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
for p in (str(PROJ), str(PROJ / 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('OMP_NUM_THREADS', '4')

import torch  # noqa: E402

import config as C  # noqa: E402
from models.model import POPWMultiTaskModel  # noqa: E402
from data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn  # noqa: E402
from training.losses import MultiTaskLoss  # noqa: E402
from training.train import _prepare_images  # noqa: E402


def build_gt_batch(ds, bs):
    """Build one batch of `bs` frames that all carry GT boxes."""
    gt_idx = [i for i, s in enumerate(ds.samples) if s.get('num_dets', 0) > 0]
    if not gt_idx:
        raise SystemExit(
            'No GT-bearing frames in this subset. Run diag_gt_coverage.py — the '
            'problem is upstream of training (recording-level OD sparsity).'
        )
    chosen = gt_idx[:bs]
    images, targets = collate_fn([ds[i] for i in chosen])
    return images, targets, chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='stage_rf1')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-recordings', type=int, default=8,
                    help='Scan only this many recordings to find a GT batch fast.')
    args = ap.parse_args()

    C.apply_preset(args.preset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  preset: {args.preset}  steps: {args.steps}  lr: {args.lr}')

    ds = IndustRealMultiTaskDataset(
        split='train', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
        max_recordings=args.max_recordings,
    )
    images, targets, chosen = build_gt_batch(ds, args.bs)
    n_boxes = sum(int(t['boxes'].shape[0]) for t in targets['detection'])
    print(f'Overfit batch: {args.bs} frames, {n_boxes} total GT boxes (sample idx {chosen})')

    images = _prepare_images(images, device, training=False)
    det_targets = []
    for t in targets['detection']:
        det_targets.append({'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)})

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
        use_hand_film=False, use_headpose_film=False, use_videomae=False,
        train_pose=False,
    ).to(device)
    model.train()

    crit = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT, num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=True, train_pose=False, train_act=False, train_psr=False,
        use_kendall=False,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print(f'\n{"step":>5} {"det_cls":>10} {"det_reg":>10} {"max+logit":>10} {"min logit":>10}')
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        outputs = model(images)
        # Report cls/reg separately via the verified FocalLoss API.
        cls_loss, reg_loss = crit.det_loss_fn(
            outputs['cls_preds'], outputs['reg_preds'],
            outputs['anchors'], det_targets,
        )
        loss = cls_loss + float(getattr(C, 'GIOU_WEIGHT', 2.0)) * reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            with torch.no_grad():
                cp = outputs['cls_preds']
                print(f'{step:5d} {cls_loss.item():10.4f} {reg_loss.item():10.4f} '
                      f'{cp.max().item():10.3f} {cp.min().item():10.3f}')

    print('\n' + '=' * 64)
    print('  det_cls -> ~0 and max(+logit) climbing past +2  => ARCHITECTURE OK')
    print('  (the death spiral is data/sampling; apply DET_GT_FRAME_FRACTION).')
    print('  Loss plateaus / logits stuck  => bug in head/loss/anchor-matching;')
    print('  inspect losses.py _match_anchors + _decode_boxes next.')
    print('=' * 64)


if __name__ == '__main__':
    main()
