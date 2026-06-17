#!/usr/bin/env python3
"""Test if AMP (mixed precision) is the cause of backward NaN.

Hypothesis: PSR logits overflow fp16 → GradScaler produces inf scale →
unscale produces NaN gradients → PSR_NAN replacement triggers.

Tests:
1. Load crash_recovery.pth + reinit heads
2. Build a single train-mode step
3. Compare forward+backward with AMP ON vs OFF
4. Report whether the gradient is finite in both cases
"""
import sys, os
from pathlib import Path

_SRC = Path(__file__).resolve().parent / 'src'
for _sub in ['models', 'training', 'evaluation', 'data', str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

os.environ.setdefault('SUBSET_RATIO', '0.05')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
import model as _popw_model_module
from training.train import seed_everything, _reinit_dead_heads
from training.losses import MultiTaskLoss

CKPT = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth'

ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = _ds_module.IndustRealMultiTaskDataset(
    split='train', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
)
collate_fn = _ds_module.collate_fn_sequences if C.USE_PSR_SEQUENCE_MODE else _ds_module.collate_fn
train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=collate_fn, pin_memory=False, drop_last=True)

# Get one batch
for images, targets in train_loader:
    if images.ndim == 5:
        B, T, C_, H, W = images.shape
        images_in = images.view(B * T, C_, H, W).to(device).float() / 255.0
    else:
        images_in = images.to(device).float() / 255.0
        B = images_in.shape[0]
        T = 1
    targets_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
    break

print(f'[amp-test] batch: B={B} T={T} images_in={tuple(images_in.shape)}', flush=True)

for amp_setting in [False, True]:
    print(f'\n{"="*60}\nAMP={amp_setting}\n{"="*60}', flush=True)

    # Fresh model
    model = _popw_model_module.POPWMultiTaskModel(
        pretrained=True, backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
        use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
        use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FIM', False)),
        use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
        train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
    ).to(device)
    model._seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 4) if C.USE_PSR_SEQUENCE_MODE else 1
    state = {k.replace('ema.', ''): v for k, v in ckpt['model'].items() if not k.startswith('ema.')}
    res = model.load_state_dict(state, strict=False)
    _reinit_dead_heads(model)
    model.train()

    criterion = MultiTaskLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-4,
    )
    scaler = amp.GradScaler('cuda', enabled=amp_setting)

    # Forward + backward
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast('cuda', enabled=amp_setting):
        out = model(images_in)
        for k in ('cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits'):
            if k in out and isinstance(out[k], torch.Tensor):
                out[k] = out[k].float()
        loss, loss_dict = criterion(out, targets_dev)

    print(f'[amp-test]   loss dtype={loss.dtype} value={loss.item():.4f} finite={torch.isfinite(loss).item()}', flush=True)
    psr = loss_dict.get('psr', None)
    if isinstance(psr, torch.Tensor):
        print(f'[amp-test]   psr loss: value={psr.item():.4f} finite={torch.isfinite(psr).item()}', flush=True)
    pl = out['psr_logits']
    print(f'[amp-test]   psr_logits: dtype={pl.dtype} min={pl.min().item():.4f} max={pl.max().item():.4f} std={pl.std().item():.4f}', flush=True)

    # Backward
    try:
        scaler.scale(loss).backward()
        print(f'[amp-test]   backward OK', flush=True)
    except Exception as e:
        print(f'[amp-test]   backward FAILED: {e}', flush=True)
        continue

    # Check gradients
    n_inf = 0
    n_nan = 0
    n_finite = 0
    inf_names = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad
            if torch.isnan(g).any():
                n_nan += 1
                if len(inf_names) < 3:
                    inf_names.append(f'NaN:{n}')
            elif torch.isinf(g).any():
                n_inf += 1
                if len(inf_names) < 3:
                    inf_names.append(f'Inf:{n}')
            else:
                n_finite += 1
    print(f'[amp-test]   grads: {n_finite} finite, {n_nan} NaN, {n_inf} Inf', flush=True)
    if inf_names:
        print(f'[amp-test]   bad grad examples: {inf_names}', flush=True)

    # Try scaler.step
    try:
        scaler.unscale_(optimizer)
        print(f'[amp-test]   unscale OK', flush=True)
    except Exception as e:
        print(f'[amp-test]   unscale FAILED: {e}', flush=True)

    # Clean up
    del model, criterion, optimizer, scaler, loss, out
    import gc; gc.collect(); torch.cuda.empty_cache()

print(f'\n{"="*60}\nAMP-NAAN-DIAG COMPLETE\n{"="*60}', flush=True)
