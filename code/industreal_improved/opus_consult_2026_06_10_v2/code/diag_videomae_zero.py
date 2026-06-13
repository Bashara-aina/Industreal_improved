#!/usr/bin/env python3
"""D5 [opus RC-17] — VideoMAE-half-zeroed diagnostic.

Eval the same checkpoint twice on the same N val batches:
  Pass A: clip_rgb as-is (whatever the dataset returns; usually real clips)
  Pass B: clip_rgb forced to zeros (mimics eval_post_reinit.py using
          collate_fn_sequences, which omits clip_rgb entirely → model.py:1347-1348
          does cat([feat, zeros_like(feat)]), effectively zeroing the
          VideoMAE half of the activity head input).

Compare the activity metrics. Any non-zero delta ⇒ RC-17 magnitude.

Note: the model is identical in both passes; the difference is the second
half of the activity-head input. Det/PSR/pose should NOT change.
"""
import os, sys, json
from pathlib import Path
import numpy as np

PROJ = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

import torch
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
from models import model as _popw_model_module
from training.losses import MultiTaskLoss
from evaluation.evaluate import evaluate_all

CKPT = os.environ.get('CHECKPOINT', str(PROJ / 'src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth'))
MAX_BATCHES = int(os.environ.get('MAX_BATCHES', '50'))
BS = int(os.environ.get('EVAL_BS', '4'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_model_and_criterion():
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = _popw_model_module.POPWMultiTaskModel(
        pretrained=False,
        backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
        use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),
        use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
        train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
    ).to(DEVICE)
    state = {k.replace('ema.', ''): v for k, v in ckpt['model'].items() if not k.startswith('ema.')}
    model.load_state_dict(state, strict=False)
    model.eval()
    crit = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
    ).to(DEVICE)
    return model, crit


def _wrap_model_with_forced_zeros(model, force_zeros: bool):
    """If force_zeros is True, monkey-patch the model's `forward` to swap
    any non-None clip_rgb for a zero tensor of the same shape/dtype/device.
    """
    if not force_zeros:
        return lambda: None  # no-op

    def _patch():
        original = model.forward

        def patched(*args, **kwargs):
            clip_rgb = kwargs.get('clip_rgb', None)
            if clip_rgb is not None:
                kwargs['clip_rgb'] = torch.zeros_like(clip_rgb)
            return original(*args, **kwargs)

        model.forward = patched

    return _patch


def _strip_for_json(o):
    import math
    if isinstance(o, dict):
        return {k: _strip_for_json(v) for k, v in o.items()
                if not isinstance(v, (torch.Tensor, np.ndarray))}
    if isinstance(o, (list, tuple)):
        return [_strip_for_json(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o


def main():
    print(f'[D5] loading {CKPT}')

    val_ds = _ds_module.IndustRealMultiTaskDataset(
        split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BS, shuffle=False, num_workers=0,
        collate_fn=_ds_module.collate_fn, pin_memory=False, drop_last=False,
    )

    results = {}
    for tag, force_zeros in (('clip_rgb_real', False), ('clip_rgb_zeroed', True)):
        print(f'\n[D5] === Pass {tag} (force_zeros={force_zeros}) ===')
        model, crit = _load_model_and_criterion()
        if force_zeros:
            _wrap_model_with_forced_zeros(model, True)()
        results[tag] = evaluate_all(
            model, crit, val_loader, DEVICE,
            max_batches=MAX_BATCHES, save_dir=None,
            use_flip_tta=False, use_crop_tta=False,
        )
        # Free GPU before loading again
        del model, crit
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print('\n=== D5: VideoMAE-zeroing metric comparison ===\n')
    keys_of_interest = [
        'act_top1', 'act_top5', 'act_f1_macro', 'act_f1_micro', 'act_f1_weighted',
        'det_mAP50', 'det_mAP50_95', 'psr_overall_f1', 'psr_comp0_f1',
        'head_pose_mae', 'head_pose_angular_mae',
    ]
    print(f'  {"metric":<28s} {"clip_real":>12s} {"clip_zero":>12s} {"delta":>12s}')
    print('  ' + '-' * 70)
    for k in keys_of_interest:
        a = results['clip_rgb_real'].get(k, None)
        b = results['clip_rgb_zeroed'].get(k, None)
        if a is None or b is None:
            continue
        try:
            delta = float(b) - float(a)
        except Exception:
            continue
        print(f'  {k:<28s} {float(a):>12.4f} {float(b):>12.4f} {delta:>+12.4f}')

    # Verdict
    a5 = float(results['clip_rgb_real'].get('act_top5', 0.0))
    b5 = float(results['clip_rgb_zeroed'].get('act_top5', 0.0))
    a1 = float(results['clip_rgb_real'].get('act_top1', 0.0))
    b1 = float(results['clip_rgb_zeroed'].get('act_top1', 0.0))
    det = abs(float(results['clip_rgb_zeroed'].get('det_mAP50', 0.0)) -
              float(results['clip_rgb_real'].get('det_mAP50', 0.0)))
    print()
    if abs(a1 - b1) > 0.01 or abs(a5 - b5) > 0.01:
        print(f'  ❌  RC-17 CONFIRMED — activity_top1 delta = {a1-b1:+.4f}, top5 delta = {a5-b5:+.4f}.')
        print('     The VideoMAE half of the activity input matters at eval time.')
    elif det > 0.01:
        print(f'  ⚠️  Det metric changed (delta={det:+.4f}); activity did not.')
        print('     The model is in a regime where the zero-fill doesn\'t change activity outputs.')
    else:
        print('  ✅  No meaningful delta — RC-17 is dormant in this checkpoint.')
        print('     Either the activity head ignores the VideoMAE half, or both halves are constant.')


if __name__ == '__main__':
    main()
