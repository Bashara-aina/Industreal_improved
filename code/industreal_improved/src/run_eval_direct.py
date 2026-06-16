"""Standalone eval script — bypasses the broken __main__ block in evaluate.py
which fails with ModuleNotFoundError for `config` when run as `python -m`.

Loads the model + val loader, calls evaluate_all directly, prints metrics.
"""
import sys
import os
import time
import json
from pathlib import Path

# Ensure src/ is on path so `import config` works
_SRC = Path('/media/newadmin/master/POPW/working/code/industreal_improved/src')
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / 'evaluation'))

import torch
import config as C  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Imports from evaluate.py (only what's at module scope)
from evaluation.evaluate import (  # noqa: E402
    evaluate_all,
    _print_single_run_results,
)
# Dataset is imported inside evaluate.main() at L3846 — re-import here at top level
from industreal_dataset import (  # noqa: E402
    IndustRealMultiTaskDataset,
    collate_fn,
)
from model import POPWMultiTaskModel  # noqa: E402
from losses import MultiTaskLoss  # noqa: E402


def main():
    # Allow env-var override so the same script can eval any checkpoint
    # without editing. Falls back to original hard-coded defaults.
    ckpt_path = os.environ.get(
        'EVAL_CKPT',
        '/media/newadmin/master/POPW/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth',
    )
    save_dir = os.environ.get(
        'EVAL_SAVE_DIR',
        '/media/newadmin/master/POPW/working/code/industreal_improved/src/runs/eval_post_subset_fix',
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    max_batches = int(os.environ.get('EVAL_BATCHES', '200'))
    split = 'val'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Max batches: {max_batches}')
    print(f'Split: {split}')
    print('=' * 60)

    # Build model
    model = POPWMultiTaskModel(
        pretrained=False,
        backbone_type=str(getattr(C, 'BACKBONE', 'resnet50')),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),
        use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
    ).to(device)

    # Build criterion
    criterion = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
    ).to(device)

    # Build val loader
    ds = IndustRealMultiTaskDataset(
        split=split,
        img_size=C.IMG_SIZE,
        augment=False,
        seed=42,
    )
    criterion.set_class_counts(ds.class_counts)
    loader = DataLoader(
        ds,
        batch_size=C.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=C.VAL_NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Load checkpoint
    print(f'Loading checkpoint: {ckpt_path}')
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f'  loaded in {time.time() - t0:.1f}s')
    if 'model' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        print(f'  model loaded — missing={len(missing)}, unexpected={len(unexpected)}')
        if missing:
            print(f'  missing keys (first 5): {missing[:5]}')
        if unexpected:
            print(f'  unexpected keys (first 5): {unexpected[:5]}')
    else:
        model.load_state_dict(ckpt, strict=False)
        print('  model loaded (raw state_dict)')

    print('=' * 60)
    print(f'Running evaluate_all on {split} split, max_batches={max_batches}')
    print('=' * 60)

    # Run eval
    t0 = time.time()
    results = evaluate_all(
        model=model,
        criterion=criterion,
        loader=loader,
        device=device,
        max_batches=max_batches,
        save_dir=save_dir,
    )
    elapsed = time.time() - t0
    print(f'evaluate_all completed in {elapsed:.1f}s')
    print()

    # Save raw results dict to JSON BEFORE the print helper (which has KeyError issues)
    out_json = Path(save_dir) / f'metrics_max{max_batches}.json'
    # Convert tensors/paths to JSON-serializable
    safe = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            try:
                safe[k] = v.detach().cpu().tolist()
            except Exception:
                safe[k] = str(v)
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            safe[k] = v
        else:
            safe[k] = str(v)
    safe['_elapsed_s'] = elapsed
    safe['_n_batches'] = max_batches
    safe['_split'] = split
    safe['_checkpoint'] = ckpt_path
    with open(out_json, 'w') as f:
        json.dump(safe, f, indent=2, default=str)
    print(f'Saved raw metrics to: {out_json}')

    # Print formatted results (wrapped to tolerate missing keys)
    try:
        _print_single_run_results(results, split)
    except KeyError as e:
        print(f'\n[NOTE] _print_single_run_results raised KeyError: {e}')
        print('  (results JSON saved above; raw dict has all keys)')
        # Print the metrics keys we DID get
        print(f'  Total result keys: {len(results)}')
        print(f'  Sample keys: {list(results.keys())[:20]}')


if __name__ == '__main__':
    main()
