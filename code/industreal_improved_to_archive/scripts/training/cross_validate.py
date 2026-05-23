#!/usr/bin/env python3
"""
5-Fold Cross-Validation Script (Doc 03 C.1)

Performs 5-fold CV at the recording level (not frame level) to get unbiased
generalization estimates. Records per-fold metrics and computes mean ± std
across folds.

Usage:
    python cross_validate.py --folds 5 --epochs 20

Each fold:
    Fold i is validation
    All other folds are training
    Metrics: det_mAP50, act_top1, act_macro_f1, head_pose_MAE, psr_macro_f1
"""
import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config as C


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_root', type=str,
                        default='/home/newadmin/swarm-bot/project/popw/working/data/datasets/industreal')
    parser.add_argument('--output', type=str, default='runs/cv_results.json')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_recordings(split: str = 'train') -> List[Tuple[str, str]]:
    """
    Load recording IDs and their dominant action class from a split CSV.

    Returns:
        List of (recording_id, dominant_action_class)
    """
    csv_path = C.TRAIN_CSV if split == 'train' else C.VAL_CSV
    recordings = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                rec_id = row[0].strip()
                action = row[1].strip() if len(row) > 1 else '0'
                recordings.append((rec_id, action))
    return recordings


def stratified_kfold_split(
    recordings: List[Tuple[str, str]],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    Stratified K-Fold split at the recording level.

    Groups recordings by dominant action class and distributes them
    across folds proportionally.

    Returns:
        List of (train_rec_ids, val_rec_ids) tuples, one per fold.
    """
    rng = np.random.RandomState(seed)

    rec_ids = [r[0] for r in recordings]
    actions = [r[1] for r in recordings]

    unique_actions = sorted(set(actions))
    action_to_recs: Dict[str, List[str]] = {a: [] for a in unique_actions}
    for rec_id, action in recordings:
        action_to_recs[action].append(rec_id)

    for a in action_to_recs:
        rng.shuffle(action_to_recs[a])

    fold_recs: List[List[str]] = [[] for _ in range(n_folds)]
    for action, recs in action_to_recs.items():
        n = len(recs)
        indices = np.array_split(range(n), n_folds)
        for fold_idx, idx in enumerate(indices):
            fold_recs[fold_idx].extend([recs[i] for i in idx])

    folds = []
    for val_fold in range(n_folds):
        val_recs = fold_recs[val_fold]
        train_recs = []
        for f in range(n_folds):
            if f != val_fold:
                train_recs.extend(fold_recs[f])
        folds.append((train_recs, val_recs))

    return folds


def write_split_csv(rec_ids: List[str], path: Path, split_type: str) -> None:
    """Write a temporary CSV with the given recording IDs."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['recording_id', 'action_class_id'])
        for rec_id in sorted(rec_ids):
            writer.writerow([rec_id, 0])


def run_fold(
    fold_idx: int,
    train_recs: List[str],
    val_recs: List[str],
    epochs: int,
    data_root: str,
    output_dir: Path,
) -> Dict:
    """Run one fold of cross-validation."""
    logger.info(f'=== Fold {fold_idx + 1}/5 ===')
    logger.info(f'  Train: {len(train_recs)} recordings')
    logger.info(f'  Val:   {len(val_recs)} recordings')

    fold_dir = output_dir / f'fold_{fold_idx + 1}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        train_csv = tmp_path / 'train_fold.csv'
        val_csv = tmp_path / 'val_fold.csv'
        write_split_csv(train_recs, train_csv, 'train')
        write_split_csv(val_recs, val_csv, 'val')

        cmd = [
            sys.executable, 'train.py',
            '--max-epochs', str(epochs),
            '--data-root', data_root,
        ]

        env = os.environ.copy()
        env['C_TRAIN_CSV'] = str(train_csv)
        env['C_VAL_CSV'] = str(val_csv)
        env['C_OUTPUT_DIR'] = str(fold_dir)
        env['C_SEED'] = str(42 + fold_idx)

        t0 = time.time()
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.warning(f'  Fold {fold_idx + 1} failed: {result.stderr[-500:]}')
            return {'fold': fold_idx + 1, 'failed': True, 'elapsed_s': elapsed}

    best_ckpt = fold_dir / 'best.pth'
    if not best_ckpt.exists():
        logger.warning(f'  No best.pth for fold {fold_idx + 1}')
        return {'fold': fold_idx + 1, 'failed': True, 'elapsed_s': elapsed}

    logger.info(f'  Fold {fold_idx + 1} complete in {elapsed:.0f}s')

    metrics = {}
    log_file = fold_dir / 'train_log.jsonl'
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                metrics = last_record.get('val', {})

    return {
        'fold': fold_idx + 1,
        'train_recs': len(train_recs),
        'val_recs': len(val_recs),
        'elapsed_s': elapsed,
        'det_mAP50': metrics.get('det_mAP50', float('nan')),
        'act_top1': metrics.get('act_accuracy', float('nan')),
        'act_macro_f1': metrics.get('act_macro_f1', float('nan')),
        'head_pose_MAE': metrics.get('head_pose_MAE', float('nan')),
        'psr_macro_f1': metrics.get('psr_macro_f1', float('nan')),
    }


def print_summary(fold_results: List[Dict]) -> None:
    """Print a formatted CV summary table."""
    print('\n' + '=' * 70)
    print('5-FOLD CROSS-VALIDATION SUMMARY')
    print('=' * 70)
    print(f'{"Fold":>4}  {"mAP@0.5":>8}  {"Act Top-1":>9}  {"Act F1":>8}  '
          f'{"Pose MAE":>9}  {"PSR F1":>8}  {"Time":>7}')
    print('-' * 70)

    metrics_keys = ['det_mAP50', 'act_top1', 'act_macro_f1', 'head_pose_MAE', 'psr_macro_f1']
    metric_names = ['det_mAP50', 'Act Top-1', 'Act F1', 'Pose MAE', 'PSR F1']

    fold_rows = []
    for r in fold_results:
        if r.get('failed'):
            print(f'{r["fold"]:>4}  FAILED')
            continue
        vals = [r.get(k, float('nan')) for k in metrics_keys]
        times = f'{r["elapsed_s"]/60:.1f}m'
        print(f'{r["fold"]:>4}  {vals[0]:>8.4f}  {vals[1]:>9.4f}  '
              f'{vals[2]:>8.4f}  {vals[3]:>9.4f}  {vals[4]:>8.4f}  {times:>7}')
        fold_rows.append(vals)

    if not fold_rows:
        print('No successful folds')
        return

    fold_arr = np.array(fold_rows)
    means = fold_arr.mean(axis=0)
    stds = fold_arr.std(axis=0)

    print('-' * 70)
    print(f'{"MEAN":>4}  {means[0]:>8.4f}  {means[1]:>9.4f}  '
          f'{means[2]:>8.4f}  {means[3]:>9.4f}  {means[4]:>8.4f}')
    print(f'{"STD":>4}  {stds[0]:>8.4f}  {stds[1]:>9.4f}  '
          f'{stds[2]:>8.4f}  {stds[3]:>9.4f}  {stds[4]:>8.4f}')
    print('=' * 70)


def main() -> None:
    args = parse_args()
    logger.info(f'Starting {args.folds}-Fold CV at {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    recordings = load_recordings('train')
    logger.info(f'Loaded {len(recordings)} recordings from train split')

    folds = stratified_kfold_split(recordings, n_folds=args.folds, seed=args.seed)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for fold_idx, (train_recs, val_recs) in enumerate(folds):
        result = run_fold(
            fold_idx,
            train_recs,
            val_recs,
            epochs=args.epochs,
            data_root=args.data_root,
            output_dir=output_dir,
        )
        results.append(result)

    print_summary(results)

    results_path = Path(args.output)
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'n_folds': args.folds,
                'epochs': args.epochs,
                'seed': args.seed,
            },
            'folds': results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)

    logger.info(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
