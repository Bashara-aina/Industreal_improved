#!/usr/bin/env python3
"""Standalone checkpoint evaluation for mid-training metrics.

Usage:
  python scripts/eval_checkpoint.py runs/paper_run_r25_fix_20260615/epoch_48.pth
  python scripts/eval_checkpoint.py --checkpoint best.pth --split val --max-batches 200
  python scripts/eval_checkpoint.py --checkpoint best.pth --save-dir /tmp/eval_results

Validates checkpoint compatibility and runs evaluate_all().
Saves structured results to JSON alongside print output.
"""

import sys, os, json, argparse, logging
from pathlib import Path
from datetime import datetime

_PROJ = Path(__file__).resolve().parent.parent  # project root
_SRC = _PROJ / 'src'
sys.path.insert(0, str(_SRC))
# model.py does `from src import config as C` — needs project root in path
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
# Add src/ subdirectories so `from models.model import ...` resolves
for _sub in ['models', 'training', 'evaluation', 'data']:
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from torch.utils.data import DataLoader

import config as C
from models.model import POPWMultiTaskModel
from training.losses import MultiTaskLoss
from data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from evaluation.evaluate import evaluate_all

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('eval_checkpoint')


def load_checkpoint_compat(path: str, model, device: torch.device):
    """Load a training checkpoint handling multiple key formats and shape mismatches."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    epoch = ckpt.get('epoch', -1)
    step = ckpt.get('step', -1)
    # Handle all known checkpoint key formats
    model_state = ckpt.get('model_state_dict',
                   ckpt.get('model_state',
                   ckpt.get('model', ckpt)))
    if 'model' not in ckpt and 'model_state_dict' not in ckpt and 'model_state' not in ckpt:
        # Full state dict directly
        pass
    # Filter size mismatches before loading (strict=False still raises on shape mismatch)
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in model_state.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            skipped.append(f'{k}: ckpt={list(v.shape)} model={list(model_sd[k].shape)}')
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning(f'  Missing keys: {len(missing)} (expected for partial checkpoint)')
    if unexpected:
        logger.warning(f'  Unexpected keys: {len(unexpected)}')
    if skipped:
        logger.warning(f'  Skipped {len(skipped)} keys (shape mismatch, re-initialized):')
        for s in skipped[:10]:
            logger.warning(f'    {s}')
        if len(skipped) > 10:
            logger.warning(f'    ... and {len(skipped)-10} more')
    return ckpt, epoch, step


def main():
    parser = argparse.ArgumentParser(description='Evaluate any mid-training checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint .pth file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max-batches', type=int, default=0,
                        help='0 = full eval, else limit batches')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Output directory (default: checkpoint dir)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Checkpoint: {args.checkpoint}')

    # Build model
    model = POPWMultiTaskModel(
        pretrained=False,
        backbone_type=str(getattr(C, 'BACKBONE', 'resnet50')),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),
        use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
        use_backbone_checkpoint=bool(getattr(C, 'USE_BACKBONE_CHECKPOINT', False)),
    ).to(device)
    model.eval()

    ckpt, epoch, step = load_checkpoint_compat(args.checkpoint, model, device)
    logger.info(f'Loaded checkpoint: epoch={epoch}, step={step}')

    # Build criterion
    criterion = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
    ).to(device)

    # Load criterion state if available
    if 'criterion' in ckpt:
        cs = ckpt['criterion']
        for k, v in cs.items():
            if hasattr(criterion, k):
                getattr(criterion, k).data.copy_(v.to(device))
        logger.info(f'Loaded criterion state: {list(cs.keys())}')

    # Build dataset and loader
    ds = IndustRealMultiTaskDataset(
        split=args.split, img_size=C.IMG_SIZE, augment=False, seed=args.seed,
    )
    loader = DataLoader(
        ds, batch_size=int(getattr(C, 'VAL_BATCH_SIZE', C.BATCH_SIZE)),
        shuffle=False, num_workers=int(getattr(C, 'VAL_NUM_WORKERS', 4)),
        collate_fn=collate_fn,
    )
    criterion.set_class_counts(ds.class_counts)
    logger.info(f'Dataset: {args.split} split, {len(ds)} samples, {len(loader)} batches')

    # Run evaluation
    save_dir = args.save_dir or str(Path(args.checkpoint).parent)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    max_batches = args.max_batches if args.max_batches > 0 else 999999

    logger.info(f'Starting evaluation (max_batches={max_batches})...')
    results = evaluate_all(model, criterion, loader, device,
                           max_batches=max_batches, save_dir=save_dir)
    logger.info('Evaluation complete.')

    # Print key results
    print('\n' + '=' * 60)
    print(f'Mid-Training Eval Results — epoch={epoch} step={step}')
    print('=' * 60)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f'  {k}: {v:.6f}')
        elif isinstance(v, (int, np.integer)):
            print(f'  {k}: {v}')
    print('=' * 60)

    # Save JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(save_dir) / f'eval_epoch{epoch}_step{step}_{timestamp}.json'
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (float, int, str, bool, np.integer, np.floating)):
            serializable[k] = float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {sk: float(sv) if isinstance(sv, (np.floating, float)) else sv
                               for sk, sv in v.items()}
    serializable['_meta'] = {'checkpoint': args.checkpoint, 'epoch': int(epoch), 'step': int(step),
                              'split': args.split, 'seed': args.seed}
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f'Results saved to {save_path}')


if __name__ == '__main__':
    main()
