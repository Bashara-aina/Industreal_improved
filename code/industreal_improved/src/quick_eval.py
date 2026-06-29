#!/usr/bin/env python
"""Quick eval: load crash_recovery.pth, run evaluate_all on a few val batches,
print all key metrics, and exit. Confirms the agent's eval fixes show real numbers.

Mirrors the build pattern in training/train.py main() but skips the optimizer/
scheduler/EMA-update/ckpt-save machinery. Mirrors train.py's exact sys.path
setup so `import evaluate` resolves to src/evaluation/evaluate.py (NOT
huggingface's evaluate package).
"""
import sys, os
from pathlib import Path

# --- MIRROR training/train.py sys.path setup exactly ---
# src/ lives at the parent of this file's directory.
_SRC = Path(__file__).resolve().parent
for _sub in ['models', 'training', 'evaluation', 'data', str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Add project root so `from src import config` resolves correctly.
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

# Environment: small subset + thread caps + CUDA debug = same as training
os.environ.setdefault('SUBSET_RATIO', '0.05')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
os.environ.setdefault('MALLOC_ARENA_MAX', '4')
os.environ.setdefault('PYTHONHASHSEED', '42')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Sanity: prove we picked the right `evaluate` (src/evaluation/evaluate.py),
# not the huggingface one. If this prints "evaluate: .../huggingface/..."
# then sys.path setup is wrong and the script will silently evaluate against
# the wrong module.
import evaluate as _eval_check
assert hasattr(_eval_check, 'evaluate_all'), (
    f"Wrong evaluate module resolved: {_eval_check.__file__}. "
    "Expected src/evaluation/evaluate.py — check sys.path setup."
)
print(f"[quick_eval] evaluate module: {_eval_check.__file__}", flush=True)

import torch
from torch.utils.data import DataLoader

import config as C
import data as _ds_module
import model as _popw_model_module
import losses as _losses_module
import evaluate as _evaluate_module
from training.train import _build_loader, seed_everything

CKPT = "/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
print(f"[quick_eval] loading {CKPT}", flush=True)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f"[quick_eval] ckpt epoch={ckpt.get('epoch')} best={ckpt.get('best_metric'):.4f}", flush=True)
print(f"[quick_eval] SUBSET_RATIO={os.environ.get('SUBSET_RATIO')}", flush=True)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[quick_eval] device={device}", flush=True)

# Build val dataset (no augment) with a tiny subset.
subset_ratio = float(os.environ.get('SUBSET_RATIO', '0.05'))
val_ds = _ds_module.IndustRealMultiTaskDataset(
    split='val',
    img_size=C.IMG_SIZE,
    augment=False,
    seed=C.SEED,
    max_recordings=max(4, int(40 * subset_ratio)),
)
print(f"[quick_eval] val_ds size: {len(val_ds):,}", flush=True)

# Pick the right collate fn for PSR sequence mode (matches train.py logic).
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
    pin_memory=C.PIN_MEMORY,
    drop_last=False,
    persistent_workers=False,
    prefetch_factor=None,
)
print(f"[quick_eval] val_loader batches: {len(val_loader)}", flush=True)

# Build model + load weights
backbone_type = str(getattr(C, 'BACKBONE', 'resnet50'))
use_hand_film = bool(getattr(C, 'USE_HAND_FILM', True))
use_headpose_film = bool(getattr(C, 'USE_HEADPOSE_FILM', False))
use_videomae = bool(getattr(C, 'USE_VIDEOMAE', False))
print(
    f"[quick_eval] model config: backbone={backbone_type} "
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

# Filter EMA-prefixed keys the way the train.py resume code does.
state = {k.replace('ema.', ''): v for k, v in ckpt['model'].items() if not k.startswith('ema.')}
res = model.load_state_dict(state, strict=False)
print(
    f"[quick_eval] model load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
    flush=True,
)
if res.missing_keys:
    print(f"[quick_eval]   missing sample: {res.missing_keys[:3]}", flush=True)
if res.unexpected_keys:
    print(f"[quick_eval]   unexpected sample: {res.unexpected_keys[:3]}", flush=True)

# Build criterion (mirrors train.py main()).
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

# Run evaluate_all with tiny max_batches.
print(f"[quick_eval] running evaluate_all with max_batches=4 ...", flush=True)
metrics = _evaluate_module.evaluate_all(
    model=model,
    criterion=criterion,
    loader=val_loader,
    device=device,
    max_batches=4,
    epoch=int(ckpt.get('epoch', 99)) + 1,
    save_dir=None,
    use_flip_tta=False,
    use_crop_tta=False,
)

# --- DET CLASS DISTRIBUTION PROBE ---
# The probe is class-agnostic (151 preds at IoU>0.5) but AP is class-aware.
# If the model collapses to predicting a few classes, AP=0 for the others.
# Reproduce the post-NMS class distribution to see what the AP function actually sees.
import numpy as _np
import torch as _torch
from collections import Counter as _Counter
from evaluation.evaluate import decode_boxes as _decode_boxes
from evaluation.evaluate import nms_numpy as _nms_numpy
from evaluation.evaluate import _prepare_images as _prepare_images_fn

print(f"\n[quick_eval] --- DET CLASS-DIST PROBE (post-threshold, pre-AP) ---", flush=True)
model.train(False)
_cls_dist = _Counter()
_score_dist = []
total_gt_per_class = _Counter()
score_thresh_probe = float(getattr(C, 'DET_EVAL_SCORE_THRESH', 0.02))
print(f"[quick_eval]   using score_thresh={score_thresh_probe}", flush=True)

with _torch.no_grad():
    for bi, (images, targets) in enumerate(val_loader):
        if bi >= 4: break
        # val_loader yields uint8; eval pipeline uses _prepare_images to cast+normalize
        images = _prepare_images_fn(images, device)
        out = model(images)
        anchors = out['anchors'].cpu().numpy()
        cls_sig = _torch.sigmoid(out['cls_preds']).cpu().numpy()
        reg = out['reg_preds'].cpu().numpy()
        for i in range(images.shape[0]):
            max_scores = cls_sig[i].max(axis=1)
            keep = max_scores > score_thresh_probe
            if keep.sum() == 0: continue
            kept_cls = cls_sig[i][keep]
            kept_reg = reg[i][keep]
            kept_anc = anchors[keep]
            ms = kept_cls.max(axis=1)
            ml = kept_cls.argmax(axis=1)
            pb = _decode_boxes(kept_anc, kept_reg)
            pb[:, 0] = _np.clip(pb[:, 0], 0, C.IMG_WIDTH)
            pb[:, 1] = _np.clip(pb[:, 1], 0, C.IMG_HEIGHT)
            pb[:, 2] = _np.clip(pb[:, 2], 0, C.IMG_WIDTH)
            pb[:, 3] = _np.clip(pb[:, 3], 0, C.IMG_HEIGHT)
            for c in range(C.NUM_DET_CLASSES):
                cm = ml == c
                if cm.sum() == 0: continue
                nk = _nms_numpy(pb[cm], ms[cm], C.DET_EVAL_NMS_IOU_THRESH)
                _cls_dist[int(c)] += int(len(nk))
                for s in ms[cm][nk]:
                    _score_dist.append(float(s))
            gt_lbls = targets['detection'][i]['labels'].cpu().numpy()
            for g in gt_lbls:
                total_gt_per_class[int(g)] += 1

print(f"[quick_eval]   GT class histogram (top 10): {total_gt_per_class.most_common(10)}", flush=True)
print(f"[quick_eval]   Pred class histogram (top 10): {_cls_dist.most_common(10)}", flush=True)
if _score_dist:
    print(f"[quick_eval]   Post-NMS score stats: n={len(_score_dist)} min={min(_score_dist):.4f} max={max(_score_dist):.4f} mean={sum(_score_dist)/len(_score_dist):.4f}", flush=True)
gt_set = set(total_gt_per_class.keys())
pred_set = set(_cls_dist.keys())
overlap = gt_set & pred_set
print(f"[quick_eval]   GT classes in batch: {len(gt_set)}; Pred classes: {len(pred_set)}; overlap: {len(overlap)}", flush=True)
if len(overlap) < 5:
    print(f"[quick_eval]   GT classes: {sorted(gt_set)}", flush=True)
    print(f"[quick_eval]   Pred classes: {sorted(pred_set)}", flush=True)
print(f"[quick_eval]   --- END DET CLASS-DIST PROBE ---\n", flush=True)
del _cls_dist, _score_dist, total_gt_per_class
import gc; gc.collect(); _torch.cuda.empty_cache()

# Print key metrics
print(f"\n{'='*60}")
print(f"[quick_eval] REAL METRICS (post-agent-fix):")
print(f"{'='*60}")
for k in sorted(metrics.keys()):
    v = metrics[k]
    if isinstance(v, float):
        print(f"  {k:40s} = {v:.6f}")
    elif isinstance(v, dict):
        print(f"  {k}: <dict with {len(v)} keys>")
        for sk, sv in sorted(v.items())[:6]:
            if isinstance(sv, float):
                print(f"    .{str(sk):36s} = {sv:.6f}")
            elif isinstance(sv, (int, bool)):
                print(f"    .{str(sk):36s} = {sv}")
            else:
                s = str(sv)
                if len(s) > 80:
                    s = s[:77] + '...'
                print(f"    .{str(sk):36s} = {s}")
        if len(v) > 6:
            print(f"    ... and {len(v) - 6} more sub-keys")
    elif isinstance(v, (list, tuple)):
        s = str(v)
        if len(s) > 120:
            s = s[:117] + '...'
        print(f"  {k:40s} = {s}")
    else:
        print(f"  {k:40s} = {v}")
