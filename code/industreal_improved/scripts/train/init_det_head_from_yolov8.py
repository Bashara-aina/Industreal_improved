#!/usr/bin/env python3
"""Initialize MTL detection head from YOLOv8 priors and save as new checkpoint.

Usage:
    python scripts/train/init_det_head_from_yolov8.py \
        --src runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
        --priors runs/yolov8_priors.json \
        --out runs/mtl_v3.18/checkpoints/init_from_yolov8.pth
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.training.yolov8_head_init import (
    extract_yolov8_priors,
    init_mtl_det_head_from_yolov8,
    load_priors,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('init_det_head')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True,
                        help='Source checkpoint (e.g., v3.7 b18000)')
    parser.add_argument('--priors', type=str, required=True,
                        help='Priors JSON file from yolov8_head_init.py')
    parser.add_argument('--out', type=str, required=True,
                        help='Output checkpoint path with new init')
    parser.add_argument('--extract-priors', action='store_true',
                        help='Re-extract priors from YOLOv8 (overrides --priors)')
    parser.add_argument('--yolov8-weights', type=str,
                        default='/home/newadmin/swarm-bot/master/POPW/datasets/industreal/assembly_state_detection_model_weights/asd_best_IndustRealandSynthetic.pt')
    args = parser.parse_args()

    # Load or extract priors
    if args.extract_priors:
        log.info('Extracting priors from YOLOv8...')
        priors = extract_yolov8_priors(yolov8_weights=args.yolov8_weights)
    else:
        priors = load_priors(args.priors)
        log.info(f'Loaded priors from {args.priors}')

    log.info(f'Priors:')
    log.info(f'  GT class freqs: {[f"{x:.4f}" for x in priors["class_frequencies_gt"][:10]]}...')

    # Load checkpoint
    log.info(f'Loading checkpoint: {args.src}')
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    log.info(f'  Checkpoint epoch={ckpt.get("epoch", "?")}, batch={ckpt.get("batch", "?")}')

    # Find cls_head[3].bias and reg_head[3] in state dict
    # These are inside WrappedMTL -> m.det_head.cls_head[3]
    cls_bias_key = 'm.det_head.cls_head.3.bias'
    cls_weight_key = 'm.det_head.cls_head.3.weight'
    reg_bias_key = 'm.det_head.reg_head.3.bias'
    reg_weight_key = 'm.det_head.reg_head.3.weight'

    log.info(f'Before init:')
    if cls_bias_key in sd:
        old_bias = sd[cls_bias_key].clone()
        log.info(f'  {cls_bias_key}: min={old_bias.min().item():.3f}, max={old_bias.max().item():.3f}')
        log.info(f'    Sigmoid range: [{torch.sigmoid(old_bias).min().item():.3f}, {torch.sigmoid(old_bias).max().item():.3f}]')

    # Apply YOLOv8 init in-place
    # We need a temporary nn.Module to use init function
    import torch.nn as nn
    num_classes = sd[cls_bias_key].shape[0] if cls_bias_key in sd else 24
    num_anchors = sd[reg_weight_key].shape[0] // 4 if reg_weight_key in sd else 16

    # Create a temporary DetectionHead-like structure for the init function
    class TempHead(nn.Module):
        def __init__(self):
            super().__init__()
            in_ch = sd[cls_weight_key].shape[1] if cls_weight_key in sd else 256
            # Mimic MTLMViTModel's det_head.cls_head and det_head.reg_head
            self.cls_head = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, num_classes, 1),
            )
            self.reg_head = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, 4 * num_anchors, 1),
            )

    temp = TempHead()
    # Load current weights into temp
    with torch.no_grad():
        for key in ['cls_head.3.weight', 'cls_head.3.bias', 'reg_head.3.weight', 'reg_head.3.bias']:
            parts = key.split('.')
            mod = temp
            for p in parts[:-1]:
                mod = getattr(mod, p)
            tensor_name = parts[-1]
            full_key = f'm.det_head.{key}'
            if full_key in sd:
                getattr(mod, tensor_name).data.copy_(sd[full_key])

    # Apply YOLOv8 init
    log.info('Applying YOLOv8 priors...')
    init_mtl_det_head_from_yolov8(
        temp, priors,
        num_classes=num_classes,
        num_anchors=num_anchors,
    )

    # Update state dict with new weights
    new_sd = dict(sd)
    with torch.no_grad():
        new_sd['m.det_head.cls_head.3.weight'] = temp.cls_head[3].weight.data.clone()
        new_sd['m.det_head.cls_head.3.bias'] = temp.cls_head[3].bias.data.clone()
        new_sd['m.det_head.reg_head.3.weight'] = temp.reg_head[3].weight.data.clone()
        new_sd['m.det_head.reg_head.3.bias'] = temp.reg_head[3].bias.data.clone()

    # Save updated checkpoint
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = dict(ckpt)
    new_ckpt['model_state_dict'] = new_sd
    new_ckpt['epoch'] = 0
    new_ckpt['batch'] = 0
    new_ckpt['init_from_yolov8'] = True
    torch.save(new_ckpt, out_path)

    log.info(f'After init:')
    if cls_bias_key in new_sd:
        new_bias = new_sd[cls_bias_key]
        log.info(f'  {cls_bias_key}: min={new_bias.min().item():.3f}, max={new_bias.max().item():.3f}')
        log.info(f'    Sigmoid range: [{torch.sigmoid(new_bias).min().item():.3f}, {torch.sigmoid(new_bias).max().item():.3f}]')

    log.info(f'Saved init checkpoint: {out_path}')


if __name__ == '__main__':
    main()