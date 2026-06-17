#!/usr/bin/env python3
"""Test bf16 vs fp16 vs no-AMP for training stability.

AMP=True (fp16) produces 85 Inf gradients in backbone first layers.
Test: does bf16 (better range, same speed on Ampere) fix it?
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

# Get two batches
batches = []
for images, targets in train_loader:
    if images.ndim == 5:
        B, T, C_, H, W = images.shape
        images_in = images.view(B * T, C_, H, W).to(device).float() / 255.0
    else:
        images_in = images.to(device).float() / 255.0
        B = images_in.shape[0]
        T = 1
    targets_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
    batches.append((images_in, targets_dev))
    if len(batches) >= 2:
        break

print(f'[amp-test-2step] got {len(batches)} batches B={B} T={T}', flush=True)

for label, amp_setting, dtype_setting in [
    ('FP32 (AMP off)', False, torch.float32),
    ('FP16 (AMP on)', True, torch.float16),
    ('BF16 (AMP on, bf16)', True, torch.bfloat16),
]:
    print(f'\n{"="*60}\n{label}  dtype={dtype_setting}\n{"="*60}', flush=True)

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

    for step, (images_in, targets_dev) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast('cuda', enabled=amp_setting, dtype=dtype_setting):
            out = model(images_in)
            for k in ('cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits'):
                if k in out and isinstance(out[k], torch.Tensor):
                    out[k] = out[k].float()
            loss, loss_dict = criterion(out, targets_dev)

        psr = loss_dict.get('psr', None)
        psr_v = psr.item() if isinstance(psr, torch.Tensor) else 'N/A'
        nan_flag = loss_dict.get('__nan_detected__', False)
        print(f'[amp-test]   step {step}: loss={loss.item():.4f} psr={psr_v} finite={torch.isfinite(loss).item()} nan_flag={nan_flag}', flush=True)

        try:
            scaler.scale(loss).backward()
        except Exception as e:
            print(f'[amp-test]   step {step} backward FAILED: {e}', flush=True)
            break

        n_inf = 0
        n_nan = 0
        n_finite = 0
        first_bad = None
        for n, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad
                if torch.isnan(g).any():
                    n_nan += 1
                    if first_bad is None: first_bad = ('NaN', n)
                elif torch.isinf(g).any():
                    n_inf += 1
                    if first_bad is None: first_bad = ('Inf', n)
                else:
                    n_finite += 1
        print(f'[amp-test]   step {step} grads: {n_finite} finite, {n_nan} NaN, {n_inf} Inf  first_bad={first_bad}', flush=True)

        # Try to step the optimizer (this is where AMP skips bad grads)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()),
            1.0,
        )
        grad_norm_v = grad_norm.item() if torch.isfinite(grad_norm).all() else float('inf')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(f'[amp-test]   step {step} grad_norm={grad_norm_v:.4f} optimizer_step_OK', flush=True)

        # Check params after step
        n_bad_params = sum(1 for p in model.parameters() if torch.isnan(p).any() or torch.isinf(p).any())
        print(f'[amp-test]   step {step} bad_params={n_bad_params}', flush=True)

    del model, criterion, optimizer, scaler, loss, out
    import gc; gc.collect(); torch.cuda.empty_cache()

print(f'\n{"="*60}\nAMP-NAAN-2STEP-DIAG COMPLETE\n{"="*60}', flush=True)
