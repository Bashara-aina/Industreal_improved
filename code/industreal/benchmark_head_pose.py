import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


"""
IndustReal Head Pose Benchmark (9-DoF)
=====================================
Standalone head pose evaluation against raw ground truth.
No competing baseline exists for this task — POPW head pose MAE is
the first reported number.

The 9 DoFs are ordered as:
    0-2: forward_vector (forward_x, forward_y, forward_z)
    3-5: position     (pos_x, pos_y, pos_z)
    6-8: up_vector    (up_x, up_y, up_z)

Usage:
    python benchmark_head_pose.py
    python benchmark_head_pose.py --checkpoint runs/checkpoints/best.pth
    python benchmark_head_pose.py --split test

Author: Bashara
Date: April 2026
"""

import argparse
import logging
from typing import Dict

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

import config as C
from dataset import collate_fn, IndustRealDataset
from evaluate import compute_head_pose_metrics
from model import MultiTaskIndustReal

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_head_pose_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run head pose inference and compute 9-DoF MAE metrics."""
    model.eval()
    all_preds = []
    all_gts = []

    for batch in dataloader:
        images = batch['images'].to(device, non_blocking=True)
        head_pose_gt = batch['head_pose'].numpy()
        all_gts.append(head_pose_gt)

        with autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(images)
            head_pose_pred = outputs['head_pose'].cpu().numpy()

        all_preds.append(head_pose_pred)

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    return compute_head_pose_metrics(all_preds, all_gts)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        logger.error('CUDA not available. Head pose benchmark requires GPU.')
        return

    logger.info(f'GPU: {torch.cuda.get_device_name()}')

    split = args.split
    batch_size = args.batch_size

    logger.info(f'Loading IndustReal dataset (split={split})')
    dataset = IndustRealDataset(
        split=split,
        img_size=(C.IMG_HEIGHT, C.IMG_WIDTH),
        dataset_mode=C.DATASET_MODE,
        detection_mode=C.DETECTION_MODE,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=C.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logger.info(f'Dataset: {len(dataset)} frames, {len(loader)} batches')

    if args.checkpoint and torch.load.__name__ == 'torch.load':
        logger.info(f'Loading checkpoint: {args.checkpoint}')
        model = MultiTaskIndustReal(pretrained=False).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        logger.info(f'  From epoch {ckpt.get("epoch", "?")}')
    else:
        logger.info('No checkpoint — using ImageNet-pretrained initialization.')
        model = MultiTaskIndustReal(pretrained=True).to(device)

    model.eval()

    logger.info('Running head pose inference...')
    metrics = run_head_pose_evaluation(model, loader, device)

    print('\n' + '=' * 60)
    print('INDUSTREAL HEAD POSE BENCHMARK (9-DoF)')
    print('=' * 60)
    print(f'Split: {split}  |  Samples: {len(dataset)}')
    model_label = "checkpoint" if args.checkpoint else "ImageNet-pretrained"
    print(f'Model: {model_label}')
    print('-' * 60)
    print(f'  Overall MAE            : {metrics["head_pose_MAE"]:.4f}')
    print(f'  MAE Std                : {metrics["head_pose_MAE_std"]:.4f}')
    print(f'  forward_x MAE          : {metrics["forward_x_MAE"]:.4f}')
    print(f'  forward_y MAE          : {metrics["forward_y_MAE"]:.4f}')
    print(f'  forward_z MAE          : {metrics["forward_z_MAE"]:.4f}')
    print(f'  pos_x MAE              : {metrics["pos_x_MAE"]:.4f}')
    print(f'  pos_y MAE              : {metrics["pos_y_MAE"]:.4f}')
    print(f'  pos_z MAE              : {metrics["pos_z_MAE"]:.4f}')
    print(f'  up_x MAE               : {metrics["up_x_MAE"]:.4f}')
    print(f'  up_y MAE               : {metrics["up_y_MAE"]:.4f}')
    print(f'  up_z MAE               : {metrics["up_z_MAE"]:.4f}')
    print('=' * 60)
    print('NOTE: No competing baseline exists. This is the first reported')
    print('9-DoF head pose evaluation on IndustReal.')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IndustReal Head Pose Benchmark')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (optional)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split (default: val)')
    parser.add_argument('--batch_size', type=int, default=C.BATCH_SIZE,
                        help='Batch size (default: from config.BATCH_SIZE)')
    main(parser.parse_args())
