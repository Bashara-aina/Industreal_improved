#!/usr/bin/env python3
"""
Multi-Seed Training & Evaluation Orchestration (Doc 03 C)

Runs train.py with multiple seeds, then evaluate.py on each checkpoint,
producing a summary table with mean ± std across seeds.

Usage:
    # Train 3 seeds (recommended: 42, 123, 7)
    python run_multi_seed.py --seeds 42,123,7 --epochs 60 --eval-only

    # Full pipeline (train + evaluate)
    python run_multi_seed.py --seeds 42,123,7 --epochs 60 --train-only

    # Train + evaluate with TTA
    python run_multi_seed.py --seeds 42,123,7 --epochs 60 --flip-tta --crop-tta

    # Just evaluate existing checkpoints
    python run_multi_seed.py --seeds 42,123,7 --eval-only --flip-tta
"""
import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Multi-seed training & evaluation orchestration (Doc 03 C)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--seeds', type=str, default='42,123,7',
        help='Comma-separated seed values. Default: 42,123,7',
    )
    parser.add_argument(
        '--epochs', type=int, default=60,
        help='Number of training epochs. Default: 60',
    )
    parser.add_argument(
        '--output-root', type=str, default='runs/multi_seed',
        help='Root directory for all seed outputs. Default: runs/multi_seed',
    )
    parser.add_argument(
        '--data-root', type=str,
        default='/home/newadmin/swarm-bot/project/popw/working/data/datasets/industreal',
        help='Path to IndustReal dataset',
    )
    parser.add_argument(
        '--train-only', action='store_true',
        help='Only run training, skip evaluation',
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Only run evaluation (assume checkpoints exist)',
    )
    parser.add_argument(
        '--flip-tta', action='store_true',
        help='Enable horizontal-flip TTA during evaluation (Doc 02 F.1)',
    )
    parser.add_argument(
        '--crop-tta', action='store_true',
        help='Enable 5-crop TTA during evaluation (Doc 02 F.2)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Override batch size',
    )
    parser.add_argument(
        '--max-batches', type=int, default=9999,
        help='Max batches per evaluation (for quick eval). Default: 9999 (full)',
    )
    return parser.parse_args()


def get_best_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find best.pth in checkpoint directory."""
    if ckpt_dir.exists():
        for p in [ckpt_dir / 'best.pth', ckpt_dir / 'best_val.pth']:
            if p.exists():
                return p
        candidates = list(ckpt_dir.glob('*.pth'))
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    return None


def run_training(seed: int, output_dir: Path, args: argparse.Namespace) -> bool:
    """Run train.py for one seed. Returns True on success."""
    logger.info(f'=== Training seed={seed} ===')
    t0 = time.time()

    env = os.environ.copy()
    env['C_SEED'] = str(seed)

    train_cmd = [
        sys.executable, str(CODE_DIR / 'train.py'),
        '--seed', str(seed),
    ]
    if args.output_root:
        output_dir.mkdir(parents=True, exist_ok=True)
    train_cmd.extend(['--max-epochs', str(args.epochs)])
    if args.batch_size:
        train_cmd.extend(['--batch-size', str(args.batch_size)])

    result = subprocess.run(
        train_cmd,
        cwd=CODE_DIR,
        env=env,
        capture_output=False,
    )
    elapsed = time.time() - t0
    logger.info(f'  Seed {seed} training done in {elapsed:.0f}s (exit={result.returncode})')
    return result.returncode == 0


def run_evaluation(
    seed: int,
    checkpoint: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run evaluate.py on a checkpoint. Returns results dict."""
    logger.info(f'  Evaluating seed={seed} with checkpoint={checkpoint.name}...')
    t0 = time.time()

    eval_cmd = [
        sys.executable, str(CODE_DIR / 'evaluate.py'),
        '--checkpoint', str(checkpoint),
        '--split', 'val',
        '--seeds', str(seed),
    ]
    if args.flip_tta:
        eval_cmd.append('--flip-tta')
    if args.crop_tta:
        eval_cmd.append('--crop-tta')
    if args.max_batches < 9999:
        eval_cmd.extend(['--max-batches', str(args.max_batches)])

    result = subprocess.run(
        eval_cmd,
        cwd=CODE_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.warning(f'  Evaluation failed for seed={seed}: {result.stderr[-500:]}')
        return {'seed': seed, 'failed': True, 'elapsed_s': elapsed}

    logger.info(f'  Evaluation done in {elapsed:.0f}s')

    try:
        output_json = checkpoint.parent / 'eval_results.json'
        if output_json.exists():
            with open(output_json) as f:
                data = json.load(f)
                data['_seed'] = seed
                data['_elapsed_s'] = elapsed
                return data
    except Exception:
        pass

    return {'seed': seed, 'failed': False, 'elapsed_s': elapsed}


def print_summary_table(all_results: List[Dict[str, Any]]) -> None:
    """Print summary table with mean ± std across seeds."""
    headline_metrics = [
        ('act_accuracy', 'Activity Top-1'),
        ('act_macro_f1', 'Activity Macro-F1'),
        ('act_clip_accuracy', 'Clip-level Acc'),
        ('det_mAP50', 'ASD mAP@0.5'),
        ('det_mAP_50_95', 'ASD mAP@0.5:0.95'),
        ('psr_overall_f1', 'PSR Overall F1'),
        ('psr_f1_at_t5', 'PSR F1@±5frames'),
        ('psr_precision_at_t5', 'PSR P@±5frames'),
        ('psr_recall_at_t5', 'PSR R@±5frames'),
        ('psr_f1_at_t3', 'PSR F1@±3frames'),
        ('psr_precision_at_t3', 'PSR P@±3frames'),
        ('psr_recall_at_t3', 'PSR R@±3frames'),
        ('psr_edit_score', 'PSR Edit Score'),
        ('psr_pos', 'PSR POS'),
        ('head_pose_MAE', 'Head Pose MAE (deg)'),
        ('forward_angular_MAE_deg', 'Forward Angular MAE (deg)'),
        ('up_angular_MAE_deg', 'Up Angular MAE (deg)'),
        ('position_MAE_mm', 'Position MAE (mm)'),
        ('as_f1', 'Assembly State F1'),
        ('as_top1_accuracy', 'Assembly State Top-1 Acc'),
        ('as_map_at_r', 'Assembly State MAP@R(+)'),
        ('ev_ap', 'Error Verification AP'),
        ('ev_f1', 'Error Verification F1'),
    ]

    print('\n' + '=' * 80)
    print(f'  Multi-Seed Summary ({len(all_results)} seeds)')
    print('=' * 80)
    header = f'  {"Metric":<28} ' + ''.join(f'  Seed {r["seed"]:<8}' for r in all_results) + '  Mean±Std'
    print(header)
    print('  ' + '-' * 76)

    for key, label in headline_metrics:
        values = [r.get(key, float('nan')) for r in all_results]
        clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        val_strs = []
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                val_strs.append('   N/A   ')
            else:
                val_strs.append(f'  {v:.4f}  ')
        if len(clean) >= 2:
            mean, std = float(np.mean(clean)), float(np.std(clean))
            summary = f'  {mean:.4f} ± {std:.4f}'
        elif len(clean) == 1:
            summary = f'  {clean[0]:.4f}  (single run)'
        else:
            summary = '  N/A'
        print(f'  {label:<28} ' + ''.join(val_strs) + summary)

    print('=' * 80 + '\n')


def save_results_csv(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save all results to CSV."""
    if not all_results:
        return
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    exclude = {'det_per_class_ap', 'act_per_class_report', 'act_confusion_matrix'}
    cols = [k for k in sorted(all_keys) if k not in exclude and not k.startswith('_')]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    logger.info(f'Results saved to {output_path}')


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    output_root = Path(args.output_root)

    logger.info(f'Starting multi-seed run: seeds={seeds}, epochs={args.epochs}')
    logger.info(f'Output root: {output_root}')

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = output_root / f'seed_{seed}'
        ckpt_dir = seed_dir / 'checkpoints'
        checkpoint = get_best_checkpoint(ckpt_dir)

        if args.eval_only:
            if checkpoint is None:
                logger.warning(f'No checkpoint found for seed={seed}, skipping eval')
                continue
            result = run_evaluation(seed, checkpoint, args)
            all_results.append(result)
        elif args.train_only:
            success = run_training(seed, seed_dir, args)
            if success and checkpoint is None:
                checkpoint = get_best_checkpoint(ckpt_dir)
            if not success:
                logger.error(f'Training failed for seed={seed}')
        else:
            success = run_training(seed, seed_dir, args)
            if success:
                checkpoint = get_best_checkpoint(ckpt_dir)
            if success and checkpoint is not None:
                result = run_evaluation(seed, checkpoint, args)
                all_results.append(result)
            else:
                logger.error(f'Training failed for seed={seed}, skipping eval')

    if all_results:
        print_summary_table(all_results)
        save_results_csv(all_results, output_root / 'multi_seed_results.csv')
        logger.info(f'\nAll results saved to {output_root / "multi_seed_results.csv"}')
    else:
        logger.info('No results to summarize (--train-only mode or all evals failed)')


if __name__ == '__main__':
    main()
