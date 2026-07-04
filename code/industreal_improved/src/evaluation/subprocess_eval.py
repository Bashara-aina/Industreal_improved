"""
Subprocess Evaluation — SIGKILL-safe validation on isolated CUDA context.

Spawns a ``spawn`` child process (CUDA context isolated from parent) on
GPU 0 (the idle RTX 3060), loads the latest checkpoint, runs evaluate_all,
and writes results to a JSON file. The parent can SIGKILL the child on
timeout without corrupting the training CUDA context.

Usage::

    from evaluation.subprocess_eval import run_val_subprocess

    metrics = run_val_subprocess(
        ckpt_path='/path/to/latest.pth',
        out_path='/tmp/val_results.json',
        overrides={'EVAL_MAX_BATCHES': 200, 'VAL_BATCH_SIZE': 4},
        timeout=900,
    )

Reference: Opus Decision 5 (69_OPUS_RESPONSE_FINAL.md)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger('subprocess_eval')

_CTX = mp.get_context('spawn')


def _val_worker(
    ckpt_path: str,
    out_path: str,
    overrides: dict[str, Any],
    predictions_path: str | None = None,
) -> None:
    """Load checkpoint, run evaluation, write results to JSON.

    Runs inside a ``spawn`` child process with ``CUDA_VISIBLE_DEVICES=0``
    so that all GPU work lands on the idle RTX 3060, completely isolated
    from the training CUDA context on GPU 1.
    """
    # Route this subprocess to the idle GPU (RTX 3060).
    # CUDA reorders GPUs by compute capability: 0=5060 Ti (faster), 1=3060 (slower).
    # Isolate to GPU 1 so the 3060 runs eval while the 5060 Ti trains uninterrupted.
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Ensure src/ is on sys.path (same pattern as train.py / evaluate.py)
    _src = Path(__file__).resolve().parent.parent  # src/
    for _sub in ['models', 'training', 'evaluation', 'data', str(_src)]:
        _p = _src / _sub if _sub != str(_src) else _src
        _p = str(_p)
        if _p not in sys.path:
            sys.path.insert(0, _p)
    if str(_src.parent) not in sys.path:
        sys.path.insert(0, str(_src.parent))

    import torch

    from src import config as C
    for k, v in overrides.items():
        setattr(C, k, v)

    from src.data.industreal_dataset import IndustRealMultiTaskDataset as IndustRealDataset
    from src.evaluation.evaluate import evaluate_all
    from src.models.model import POPWMultiTaskModel

    logger.info('[SUB] Loading checkpoint %s', ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = state.get('config', {})

    # Build model matching train.py constructor API
    backbone_type = cfg.get('BACKBONE_TYPE', getattr(C, 'BACKBONE_TYPE', 'convnext_tiny'))
    use_hand_film = bool(cfg.get('USE_HAND_FILM', getattr(C, 'USE_HAND_FILM', True)))
    use_headpose_film = bool(cfg.get('USE_HEADPOSE_FILM', getattr(C, 'USE_HEADPOSE_FILM', False)))
    use_videomae = bool(cfg.get('USE_VIDEOMAE', getattr(C, 'USE_VIDEOMAE', False)))
    train_pose = bool(cfg.get('TRAIN_HEAD_POSE', getattr(C, 'TRAIN_HEAD_POSE', True)))
    use_backbone_checkpoint = bool(cfg.get('USE_BACKBONE_CHECKPOINT', getattr(C, 'USE_BACKBONE_CHECKPOINT', False)))

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type=backbone_type,
        use_hand_film=use_hand_film,
        use_headpose_film=use_headpose_film,
        use_videomae=use_videomae,
        train_pose=train_pose,
        use_backbone_checkpoint=use_backbone_checkpoint,
    ).to('cuda').eval()
    model._seq_len = cfg.get(
        'PSR_SEQUENCE_LENGTH',
        getattr(C, 'PSR_SEQUENCE_LENGTH', 1),
    ) if cfg.get('USE_PSR_SEQUENCE_MODE', getattr(C, 'USE_PSR_SEQUENCE_MODE', False)) else 1

    result = model.load_state_dict(state['model'], strict=False)
    if result.missing_keys:
        logger.warning('[SUB] Missing keys: %s', result.missing_keys)
    if result.unexpected_keys:
        logger.warning('[SUB] Unexpected keys: %s', result.unexpected_keys)

    # Build val dataset and loader
    val_ds = IndustRealDataset(
        split='val',
        img_size=(C.IMG_HEIGHT, C.IMG_WIDTH),
    )

    val_batch_size = int(overrides.get('VAL_BATCH_SIZE', getattr(C, 'VAL_BATCH_SIZE', C.BATCH_SIZE)))
    _ds_module = sys.modules.get('src.data.industreal_dataset') or __import__('src.data.industreal_dataset', fromlist=['collate_fn'])
    _collate_fn = getattr(_ds_module, 'collate_fn', None)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2 if int(getattr(C, 'VAL_NUM_WORKERS', 0)) > 0 else None,
        collate_fn=_collate_fn,
    )

    _raw_max_batches = overrides.get('EVAL_MAX_BATCHES')
    max_batches = int(_raw_max_batches) if _raw_max_batches is not None else 0

    logger.info('[SUB] Starting evaluate_all (max_batches=%s, batch_size=%d) ...', max_batches or 'unlimited', val_batch_size)
    with torch.no_grad():
        metrics = evaluate_all(
            model,
            criterion=None,  # loss not needed for inference-only eval
            loader=val_loader,
            device='cuda',
            max_batches=max_batches if max_batches > 0 else None,
            epoch=int(overrides.get('epoch', 0)),
            predictions_path=predictions_path,
        )

    # Convert any non-serialisable values
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            json.dumps(v)
            clean[k] = v
        except (TypeError, OverflowError):
            clean[k] = float(v) if isinstance(v, (int, float)) else str(v)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(clean, open(out_path, 'w'), indent=2, default=str)
    logger.info('[SUB] Results written to %s', out_path)


def run_val_subprocess(
    ckpt_path: str | Path,
    out_path: str | Path,
    overrides: dict[str, Any] | None = None,
    timeout: int = 900,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run validation in a subprocess on GPU 0, killable via SIGKILL.

    Parameters
    ----------
    ckpt_path:
        Path to the checkpoint (``latest.pth`` or ``best.pth``).
    out_path:
        Path where the subprocess writes JSON results.
    overrides:
        Config overrides passed to the child (e.g.
        ``{'EVAL_MAX_BATCHES': 200, 'VAL_BATCH_SIZE': 4}``).
    timeout:
        Maximum wall-clock seconds before SIGKILL.
    predictions_path:
        If set, save per-frame predictions to this JSON file.

    Returns
    -------
    dict
        The validation metrics dict, or an empty dict on timeout / error.
    """
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)
    predictions_path = Path(predictions_path) if predictions_path else None

    if not ckpt_path.exists():
        logger.error('[SUB] Checkpoint not found: %s', ckpt_path)
        return {}

    overrides = overrides or {}

    p = _CTX.Process(
        target=_val_worker,
        args=(str(ckpt_path), str(out_path), overrides, str(predictions_path) if predictions_path else None),
    )
    p.start()

    _log_timeout_warning = True
    _elapsed = 0
    _check_interval = 15

    while _elapsed < timeout:
        p.join(timeout=_check_interval)
        _elapsed += _check_interval
        if not p.is_alive():
            break
        if _elapsed >= timeout // 2 and _log_timeout_warning:
            logger.warning(
                '[SUB] Validation subprocess still running after %d s '
                '(PID=%d, will kill at %d s)',
                _elapsed, p.pid, timeout,
            )
            _log_timeout_warning = False

    if p.is_alive():
        logger.error(
            '[SUB] Validation subprocess (PID=%d) timed out after %d s — sending SIGKILL',
            p.pid, timeout,
        )
        p.kill()
        p.join(timeout=30)
        if p.is_alive():
            logger.error('[SUB] Subprocess refused to die — zombie PID=%d', p.pid)
        return {}

    if out_path.exists():
        try:
            with open(out_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error('[SUB] Failed to read results from %s: %s', out_path, exc)
            return {}

    logger.warning('[SUB] Subprocess finished but no output at %s', out_path)
    return {}


def main() -> None:
    """CLI entry point for standalone evaluation with per-frame persistence."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run subprocess evaluation (SIGKILL-safe, isolated CUDA context).'
    )
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint (.pth)')
    parser.add_argument('--out_path', required=True, help='Path to write metrics JSON')
    parser.add_argument(
        '--predictions_path',
        default=None,
        help='If set, save per-frame predictions to this JSON file',
    )
    parser.add_argument(
        '--persist_predictions',
        action='store_true',
        default=False,
        help='Persist per-frame predictions to --predictions_path',
    )
    parser.add_argument(
        '--EVAL_MAX_BATCHES',
        type=int,
        default=0,
        help='Max eval batches (0 = full dataset)',
    )
    parser.add_argument(
        '--VAL_BATCH_SIZE',
        type=int,
        default=None,
        help='Validation batch size (overrides config default)',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=7200,
        help='Subprocess timeout in seconds (default 7200 = 2h)',
    )

    args = parser.parse_args()

    # Build overrides from CLI args
    overrides: dict[str, Any] = {}
    if args.EVAL_MAX_BATCHES is not None:
        overrides['EVAL_MAX_BATCHES'] = args.EVAL_MAX_BATCHES
    if args.VAL_BATCH_SIZE is not None:
        overrides['VAL_BATCH_SIZE'] = args.VAL_BATCH_SIZE

    # If --persist_predictions is set but no --predictions_path, derive from --out_path
    predictions_path: str | None = args.predictions_path
    if args.persist_predictions and predictions_path is None:
        out_dir = Path(args.out_path).parent
        predictions_path = str(out_dir / 'per_frame_predictions.json')
        logger.info('[SUB] --persist_predictions set, writing to %s', predictions_path)

    logger.info(
        '[SUB] Starting subprocess eval: ckpt=%s out=%s predictions=%s',
        args.ckpt, args.out_path, predictions_path,
    )

    result = run_val_subprocess(
        ckpt_path=args.ckpt,
        out_path=args.out_path,
        overrides=overrides,
        timeout=args.timeout,
        predictions_path=predictions_path,
    )

    if result:
        logger.info('[SUB] Evaluation complete: %d metrics written', len(result))
    else:
        logger.error('[SUB] Evaluation returned empty results (timeout or error)')


if __name__ == '__main__':
    main()
