#!/usr/bin/env python
"""3-head collapse diagnostic: probe raw logits/activations for detection,
activity, and PSR heads. Run from crash_recovery.pth.

Outputs go to stdout; redirect to log file.
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
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
os.environ.setdefault('MALLOC_ARENA_MAX', '4')
os.environ.setdefault('PYTHONHASHSEED', '42')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', '4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
import model as _popw_model_module
from training.train import _build_loader, seed_everything

CKPT = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth'
print(f'[diag] loading {CKPT}', flush=True)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f'[diag] ckpt epoch={ckpt.get("epoch")} tag={ckpt.get("tag")} best={ckpt.get("best_metric"):.4f}', flush=True)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[diag] device={device}', flush=True)

subset_ratio = float(os.environ.get('SUBSET_RATIO', '0.05'))
val_ds = _ds_module.IndustRealMultiTaskDataset(
    split='val', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    max_recordings=max(4, int(40 * subset_ratio)),
)
print(f'[diag] val_ds size: {len(val_ds):,}', flush=True)

collate_fn = (
    _ds_module.collate_fn_sequences
    if C.USE_PSR_SEQUENCE_MODE
    else _ds_module.collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=C.VAL_BATCH_SIZE, shuffle=False, num_workers=0,
    collate_fn=collate_fn, pin_memory=C.PIN_MEMORY, drop_last=False,
    persistent_workers=False, prefetch_factor=None,
)
print(f'[diag] val_loader batches: {len(val_loader)}', flush=True)

model = _popw_model_module.POPWMultiTaskModel(
    pretrained=True, backbone_type=str(getattr(C, 'BACKBONE', 'convnext_tiny')),
    use_hand_film=bool(getattr(C, 'USE_HAND_FILM', True)),
    use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FIM', False)),
    use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
    train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
).to(device)
model._seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 4) if C.USE_PSR_SEQUENCE_MODE else 1

# Load weights. Filter EMA keys if present.
state_dict = ckpt['model']
if any(k.startswith('ema.') for k in state_dict):
    state_dict = {k.replace('ema.', ''): v for k, v in state_dict.items() if not k.startswith('ema.')}
res = model.load_state_dict(state_dict, strict=False)
print(f'[diag] model load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}', flush=True)
if res.missing_keys[:3]:
    print(f'[diag]   missing sample: {res.missing_keys[:3]}', flush=True)

# Also load criterion state (log_var etc) from checkpoint
print(f'[diag] ckpt["criterion"] keys: {list(ckpt.get("criterion", {}).keys())[:5]}', flush=True)

print()
print('=' * 70)
print('PROBE 1: DETECTION RAW LOGITS (cls_preds before sigmoid)')
print('=' * 70)

model.train(False)
all_cls_logits = []
all_cls_sig = []
all_max_scores = []
anchors_np = None
N_PROBE = 8

with torch.no_grad():
    for bi, (images, targets) in enumerate(val_loader):
        if bi >= N_PROBE: break
        images = images.to(device).float() / 255.0
        if images.ndim == 5:
            B, T, C_, H, W = images.shape
            images = images.view(B * T, C_, H, W)
        out = model(images)
        cls_logits = out['cls_preds']  # [B, N, 24] raw
        cls_sig = torch.sigmoid(cls_logits)
        if anchors_np is None:
            anchors_np = out['anchors'].cpu().numpy()
        for i in range(cls_logits.shape[0]):
            l = cls_logits[i].cpu().numpy()
            s = cls_sig[i].cpu().numpy()
            all_cls_logits.append(l.flatten())
            all_cls_sig.append(s.flatten())
            all_max_scores.append(s.max(axis=1))

L = np.concatenate(all_cls_logits)
S = np.concatenate(all_cls_sig)
MS = np.concatenate(all_max_scores)
print(f'[diag] raw logit stats: min={L.min():.4f} max={L.max():.4f} mean={L.mean():.4f} std={L.std():.4f}', flush=True)
print(f'[diag] raw sigmoid stats: min={S.min():.4f} max={S.max():.4f} mean={S.mean():.4f} std={S.std():.4f}', flush=True)
print(f'[diag] per-image max score stats: min={MS.min():.4f} p50={np.percentile(MS,50):.4f} p90={np.percentile(MS,90):.4f} p99={np.percentile(MS,99):.4f} max={MS.max():.4f}', flush=True)
print(f'[diag] preds>0.01: {(MS>0.01).sum()}/{len(MS)} ({100.0*(MS>0.01).sum()/len(MS):.1f}%)', flush=True)
print(f'[diag] preds>0.05: {(MS>0.05).sum()}/{len(MS)} ({100.0*(MS>0.05).sum()/len(MS):.1f}%)', flush=True)
print(f'[diag] preds>0.15: {(MS>0.15).sum()}/{len(MS)} ({100.0*(MS>0.15).sum()/len(MS):.1f}%)', flush=True)
print(f'[diag] preds>0.30: {(MS>0.30).sum()}/{len(MS)} ({100.0*(MS>0.30).sum()/len(MS):.1f}%)', flush=True)
print(f'[diag] preds>0.50: {(MS>0.50).sum()}/{len(MS)} ({100.0*(MS>0.50).sum()/len(MS):.1f}%)', flush=True)
del L, S, MS, all_cls_logits, all_cls_sig, all_max_scores
import gc; gc.collect(); torch.cuda.empty_cache()

print()
print('=' * 70)
print('PROBE 2: ACTIVITY CLASS DISTRIBUTION (75 classes)')
print('=' * 70)

act_pred_dist = Counter()
act_pred_argmax = []
all_act_logits = []
GT_act_dist = Counter()

with torch.no_grad():
    for bi, (images, targets) in enumerate(val_loader):
        if bi >= N_PROBE: break
        images = images.to(device).float() / 255.0
        if images.ndim == 5:
            B, T, C_, H, W = images.shape
            images = images.view(B * T, C_, H, W)
        out = model(images)
        # Activity logits — find shape
        if 'activity_logits' in out:
            act_logits = out['activity_logits']
        elif 'act_logits' in out:
            act_logits = out['act_logits']
        else:
            # search for key with 'act' or 'activity' (excluding videomae)
            act_keys = [k for k in out.keys() if 'act' in k.lower() and 'videomae' not in k.lower()]
            act_logits = out[act_keys[0]] if act_keys else None
            print(f'[diag] activity keys found: {act_keys}', flush=True)
        if act_logits is None:
            break
        # [B, 75]
        am = act_logits.argmax(dim=-1).cpu().numpy()
        for a in am:
            act_pred_dist[int(a)] += 1
            act_pred_argmax.append(int(a))
        all_act_logits.append(act_logits.cpu().numpy())
        # GT
        for t in targets.get('activity', []):
            if isinstance(t, torch.Tensor):
                lbl = int(t.item()) if t.numel() == 1 else int(t.argmax().item())
            else:
                lbl = int(t)
            GT_act_dist[lbl] += 1

if all_act_logits:
    A = np.concatenate(all_act_logits, axis=0)
    print(f'[diag] activity logit stats: shape={A.shape} min={A.min():.4f} max={A.max():.4f} mean={A.mean():.4f} std={A.std():.4f}', flush=True)
    print(f'[diag] per-class mean logits (top 5 highest):', flush=True)
    cls_mean = A.mean(axis=0)
    top5 = np.argsort(-cls_mean)[:5]
    for c in top5:
        print(f'[diag]   class {c:3d}: mean_logit={cls_mean[c]:.4f} pred_count={act_pred_dist[c]}', flush=True)
    print(f'[diag] predicted class (top 10): {act_pred_dist.most_common(10)}', flush=True)
    print(f'[diag] total preds: {sum(act_pred_dist.values())}; unique classes predicted: {len(act_pred_dist)}/75', flush=True)
    if act_pred_argmax:
        unique, counts = np.unique(act_pred_argmax, return_counts=True)
        print(f'[diag] top-1 unique-class: class {unique[np.argmax(counts)]} (count={counts.max()})', flush=True)
    if GT_act_dist:
        print(f'[diag] GT activity histogram (top 10): {GT_act_dist.most_common(10)}', flush=True)
else:
    print(f'[diag] ! NO activity logits captured', flush=True)
del all_act_logits
gc.collect(); torch.cuda.empty_cache()

print()
print('=' * 70)
print('PROBE 3: PSR PATTERN DISTRIBUTION (11 components)')
print('=' * 70)

PSR_PATTERNS = Counter()
PSR_unique = set()
all_psr_logits = []

with torch.no_grad():
    for bi, (images, targets) in enumerate(val_loader):
        if bi >= N_PROBE: break
        images = images.to(device).float() / 255.0
        if images.ndim == 5:
            B, T, C_, H, W = images.shape
            images = images.view(B * T, C_, H, W)
        out = model(images)
        if 'psr_logits' in out:
            psr = out['psr_logits']
        elif 'psr_preds' in out:
            psr = out['psr_preds']
        else:
            psr_keys = [k for k in out.keys() if 'psr' in k.lower()]
            print(f'[diag] psr keys found: {psr_keys}', flush=True)
            psr = out[psr_keys[0]] if psr_keys else None
        if psr is None:
            break
        # Use sigmoid to get binary
        psr_sig = torch.sigmoid(psr).cpu().numpy()
        all_psr_logits.append(psr_logits_arr := psr.cpu().numpy())
        for row in psr_sig:
            pat = tuple((row > 0.5).astype(int).tolist())
            PSR_PATTERNS[pat] += 1
            PSR_unique.add(pat)

if all_psr_logits:
    P = np.concatenate(all_psr_logits, axis=0)
    print(f'[diag] psr logit stats: shape={P.shape} min={P.min():.4f} max={P.max():.4f} mean={P.mean():.4f} std={P.std():.4f}', flush=True)
    print(f'[diag] unique binary patterns: {len(PSR_unique)} (across {P.shape[0]} samples)', flush=True)
    for pat, count in PSR_PATTERNS.most_common(10):
        print(f'[diag]   pattern {pat}: count={count}', flush=True)
    print(f'[diag] per-component mean of sigmoid (should vary if not collapsed):', flush=True)
    PS = np.concatenate([torch.sigmoid(torch.from_numpy(p)).numpy() for p in all_psr_logits], axis=0)
    for c in range(min(11, PS.shape[-1])):
        print(f'[diag]   comp{c}: sigmoid mean={PS[:,c].mean():.4f} std={PS[:,c].std():.4f} frac_pos={(PS[:,c]>0.5).mean():.4f}', flush=True)
else:
    print(f'[diag] ! NO psr logits captured', flush=True)

print()
print('=' * 70)
print('PROBE 4: CRITERION log_var (Kendall uncertainty weights)')
print('=' * 70)
if ckpt.get('criterion'):
    crit = ckpt['criterion']
    for k, v in crit.items():
        if 'log_var' in k or 'weight' in k:
            if hasattr(v, 'cpu'):
                arr = v.cpu().numpy()
                print(f'[diag] {k}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}', flush=True)
            else:
                print(f'[diag] {k}: {v}', flush=True)
else:
    print('[diag] ! No criterion in ckpt', flush=True)

print()
print('=' * 70)
print('DIAG COMPLETE')
print('=' * 70)
