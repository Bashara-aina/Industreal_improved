#!/usr/bin/env python
"""One-shot eval driver for audit task. Mirrors quick_eval.py sys.path setup.
Uses the newest available checkpoint (ikea_multitask_improved4/latest.pth) and
runs evaluate_all with max_batches=3 to get a quick but non-trivial set of
real numbers out of the model.
"""
import sys, os
from pathlib import Path

_SRC = Path(__file__).resolve().parent
for _sub in ['models', 'training', 'evaluation', 'data', str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import evaluate as _eval_check
assert hasattr(_eval_check, 'evaluate_all'), (
    f"Wrong evaluate module: {_eval_check.__file__}"
)
print(f"[audit_eval] evaluate module: {_eval_check.__file__}", flush=True)

import torch
from torch.utils.data import DataLoader
import config as C
import data as _ds_module
import model as _popw_model_module
import losses as _losses_module
import evaluate as _evaluate_module

CKPT = "/media/newadmin/master/POPW/working/code/industreal_improved/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth"
print(f"[audit_eval] loading {CKPT}", flush=True)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f"[audit_eval] ckpt keys: {list(ckpt.keys())[:10]}", flush=True)
print(f"[audit_eval] ckpt epoch={ckpt.get('epoch', '?')}", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[audit_eval] device={device}", flush=True)

val_ds = _ds_module.IndustRealMultiTaskDataset(
    split='val',
    img_size=C.IMG_SIZE,
    augment=False,
    seed=C.SEED,
    max_recordings=4,
)
print(f"[audit_eval] val_ds size: {len(val_ds):,}", flush=True)

collate_fn = (
    _ds_module.collate_fn_sequences
    if C.USE_PSR_SEQUENCE_MODE
    else _ds_module.collate_fn
)
val_loader = DataLoader(
    val_ds,
    batch_size=C.VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=False,
    drop_last=False,
)
print(f"[audit_eval] val_loader batches: {len(val_loader)}", flush=True)

backbone_type = str(getattr(C, 'BACKBONE', 'resnet50'))
use_hand_film = bool(getattr(C, 'USE_HAND_FILM', True))
use_headpose_film = bool(getattr(C, 'USE_HEADPOSE_FILM', False))
use_videomae = bool(getattr(C, 'USE_VIDEOMAE', False))
print(
    f"[audit_eval] model config: backbone={backbone_type} "
    f"hand_film={use_hand_film} head_film={use_headpose_film} videomae={use_videomae}",
    flush=True,
)

model = _popw_model_module.POPWMultiTaskModel(
    pretrained=True,
    backbone_type=backbone_type,
    use_hand_film=use_hand_film,
    use_headpose_film=use_headpose_film,
    use_videomae=use_videomae,
    train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
).to(device)
model._seq_len = getattr(C, 'PSR_SEQUENCE_LENGTH', 4) if C.USE_PSR_SEQUENCE_MODE else 1

state = ckpt.get('model', ckpt)
state = {k.replace('ema.', ''): v for k, v in state.items() if not k.startswith('ema.')}
res = model.load_state_dict(state, strict=False)
print(
    f"[audit_eval] model load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
    flush=True,
)
if res.missing_keys:
    print(f"[audit_eval]   missing sample: {res.missing_keys[:3]}", flush=True)
if res.unexpected_keys:
    print(f"[audit_eval]   unexpected sample: {res.unexpected_keys[:3]}", flush=True)

criterion = _losses_module.MultiTaskLoss(
    num_classes_act=C.NUM_CLASSES_ACT,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=bool(getattr(C, 'TRAIN_DET', True)),
    train_pose=bool(getattr(C, 'TRAIN_HEAD_POSE', True)),
    train_act=bool(getattr(C, 'TRAIN_ACT', True)),
    train_psr=bool(getattr(C, 'TRAIN_PSR', True)),
    use_kendall=bool(getattr(C, 'USE_KENDALL', True)),
).to(device)
if hasattr(val_ds, 'class_counts'):
    criterion.set_class_counts(val_ds.class_counts)

print(f"[audit_eval] running evaluate_all with max_batches=3 ...", flush=True)
metrics = _evaluate_module.evaluate_all(
    model=model,
    criterion=criterion,
    loader=val_loader,
    device=device,
    max_batches=3,
    epoch=0,
    save_dir='/tmp/eval_output',
    use_flip_tta=False,
    use_crop_tta=False,
)

print(f"\n{'='*60}")
print(f"[audit_eval] REAL METRICS:")
print(f"{'='*60}")
for k in sorted(metrics.keys()):
    v = metrics[k]
    if isinstance(v, float):
        import math
        if math.isnan(v):
            print(f"  {k:40s} = NaN")
        elif math.isinf(v):
            print(f"  {k:40s} = Inf")
        else:
            print(f"  {k:40s} = {v:.6f}")
    elif isinstance(v, (int, bool)):
        print(f"  {k:40s} = {v}")
    elif isinstance(v, dict):
        print(f"  {k}: <dict with {len(v)} keys>")
    elif isinstance(v, (list, tuple)):
        s = str(v)
        if len(s) > 120:
            s = s[:117] + '...'
        print(f"  {k:40s} = {s}")
    else:
        print(f"  {k:40s} = {v}")
