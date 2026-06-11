#!/usr/bin/env python3
"""D9 [opus RC-25] — Per-layer weight-norm ratios.

Computes weight-norm ratios (current / fresh-init) for every Conv2d layer
in backbone, FPN, and detection head. High ratios indicate weights that have
diverged significantly from their initialization scale during collapse.

RC-25 predicts that backbone/FPN weight norms are O(10^1-10^2)× larger than
fresh init, causing the feature-magnitude explosion.

Reports:
  - Per-layer: name, current_norm, fresh_norm, ratio, flag
  - Summary: layers with ratio > 5x (candidates for reinit)

Usage:
  python code/diag_weight_norms.py
  CHECKPOINT=path/to/latest.pth python code/diag_weight_norms.py
"""
import os, sys
from pathlib import Path
import numpy as np

PROJ = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '4')

import torch
import torch.nn as nn

import config as C
from models.model import POPWMultiTaskModel

CKPT = os.environ.get('CHECKPOINT', str(PROJ / 'src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = float(os.environ.get('RATIO_THRESHOLD', '5.0'))


def weight_norm(param):
    """Frobenius norm of weight tensor."""
    return param.data.float().norm().item()


def collect_conv_weights(model):
    """Return {name: weight_tensor} for all Conv2d layers in model."""
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'weight'):
            weights[name] = module.weight.data.clone().cpu()
    return weights


def main():
    print('D9: Per-Layer Weight-Norm Ratios')
    print(f'  Checkpoint: {CKPT}')
    print(f'  Threshold: {THRESHOLD}x')

    # 1) Fresh model weights
    print('\n[1/3] Collecting fresh-init weights...')
    C.ZERO_DET_CONF_FOR_RECOVERY = False
    model_fresh = POPWMultiTaskModel(
        num_det_classes=C.NUM_DET_CLASSES if hasattr(C, 'NUM_DET_CLASSES') else 24,
        num_act_classes=C.NUM_ACT_CLASSES,
        num_psr_classes=C.NUM_PSR_CLASSES if hasattr(C, 'NUM_PSR_CLASSES') else 11,
        backbone_type=C.BACKBONE_TYPE if hasattr(C, 'BACKBONE_TYPE') else 'convnext_tiny',
        pretrained=False,
    )
    fresh_weights = collect_conv_weights(model_fresh)
    print(f'  Fresh model: {len(fresh_weights)} Conv2d layers')
    del model_fresh

    # 2) Checkpoint weights
    print('\n[2/3] Loading checkpoint model...')
    model_ckpt = POPWMultiTaskModel(
        num_det_classes=C.NUM_DET_CLASSES if hasattr(C, 'NUM_DET_CLASSES') else 24,
        num_act_classes=C.NUM_ACT_CLASSES,
        num_psr_classes=C.NUM_PSR_CLASSES if hasattr(C, 'NUM_PSR_CLASSES') else 11,
        backbone_type=C.BACKBONE_TYPE if hasattr(C, 'BACKBONE_TYPE') else 'convnext_tiny',
        pretrained=False,
    )
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt.get('model')))
    model_ckpt.load_state_dict(state, strict=False)
    ckpt_weights = collect_conv_weights(model_ckpt)
    del model_ckpt

    # 3) Compare
    print('\n[3/3] Computing weight-norm ratios...\n')

    # Group by component
    groups = {
        'backbone': [],
        'fpn': [],
        'detection_head': [],
        'other': [],
    }

    common = set(fresh_weights) & set(ckpt_weights)
    for name in sorted(common):
        f_norm = weight_norm(fresh_weights[name])
        c_norm = weight_norm(ckpt_weights[name])
        ratio = c_norm / f_norm if f_norm > 1e-8 else float('inf')

        if 'backbone' in name:
            grp = 'backbone'
        elif 'fpn' in name:
            grp = 'fpn'
        elif 'detection_head' in name or 'det_head' in name:
            grp = 'detection_head'
        else:
            grp = 'other'

        flag = ' *** EXPLODED' if ratio > THRESHOLD else ''
        groups[grp].append((name, f_norm, c_norm, ratio, flag))

    for grp_name, entries in groups.items():
        if not entries:
            continue
        print(f'  ── {grp_name.upper()} ({len(entries)} layers) ──')
        for name, fn, cn, ratio, flag in entries:
            print(f'    {name:60s}  fresh={fn:8.3f}  ckpt={cn:8.3f}  ratio={ratio:5.1f}x{flag}')
        # Summary for group
        ratios = [e[3] for e in entries if e[3] != float('inf')]
        if ratios:
            print(f'    → median={np.median(ratios):.1f}x  max={np.max(ratios):.1f}x  '
                  f'layers>{THRESHOLD}x: {sum(1 for r in ratios if r > THRESHOLD)}/{len(ratios)}')
        print()

    # Overall summary
    all_ratios = []
    for entries in groups.values():
        for _, _, _, ratio, _ in entries:
            if ratio != float('inf'):
                all_ratios.append(ratio)

    print(f'{"="*60}')
    print(f'  OVERALL: {len(all_ratios)} layers compared')
    print(f'  median ratio: {np.median(all_ratios):.1f}x')
    print(f'  max ratio:    {np.max(all_ratios):.1f}x')
    n_exploded = sum(1 for r in all_ratios if r > THRESHOLD)
    print(f'  layers > {THRESHOLD}x: {n_exploded}/{len(all_ratios)}')
    print(f'{"="*60}')

    if n_exploded > 0:
        print(f'\nVerdict: {n_exploded} layers have weight norms {THRESHOLD}x+ larger than fresh init.')
        print('RC-25 weight explosion is CONFIRMED. FPN (and possibly backbone) reinit needed.')
    else:
        print('\nVerdict: No weight explosion detected. RC-25 may not be the dominant factor.')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
