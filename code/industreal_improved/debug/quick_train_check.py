#!/usr/bin/env python3
"""
Quick 1-epoch sanity check for training loss + validation metrics.
Disables staged training, uses 10% subset, 100 train batches + 20 val batches.
Verifies all losses and metrics are non-zero and non-NaN.
"""
import sys, os, logging, time, math
from pathlib import Path

os.environ['PYTHONHASHSEED'] = '42'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['MALLOC_ARENA_MAX'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Setup paths
_SRC = Path(__file__).resolve().parent / 'src'
for _sub in ['models', 'training', 'evaluation', 'data', 'utils', str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

import torch
import numpy as np
import random

# Override config BEFORE importing modules that use it
import src.config as _C
_C.SUBSET_RATIO = 0.10        # 10% of dataset
_C.EPOCHS = 1                  # 1 epoch only
_C.STAGED_TRAINING = False      # All heads active from epoch 0
_C.USE_EMA = False              # Disable EMA
_C.USE_MIXUP = False           # Disable mixup/cutmix
_C.EVAL_MAX_BATCHES = 20        # Only 20 val batches for quick check

# Import after config overrides
import model as _model_module
import losses as _losses_module
import evaluate as _evaluate_module
import data as _ds_module
from src import config as C

IndustRealMultiTaskDataset = getattr(_ds_module, 'IndustRealMultiTaskDataset')
_collate_fn_name = 'collate_fn_sequences' if C.USE_PSR_SEQUENCE_MODE else 'collate_fn'
collate_fn = getattr(_ds_module, _collate_fn_name)

MultiTaskIndustReal = getattr(_model_module, 'MultiTaskIndustReal', None)
POPWMultiTaskModel = getattr(_model_module, 'POPWMultiTaskModel')
MultiTaskLoss = getattr(_losses_module, 'MultiTaskLoss')
evaluate_all = getattr(_evaluate_module, 'evaluate_all')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

MAX_TRAIN_BATCHES = 100  # Just 100 batches for quick sanity

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_finite(name, value):
    if isinstance(value, float):
        is_nan = math.isnan(value) or math.isinf(value)
        is_zero = abs(value) < 1e-8
        return is_nan, is_zero
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu().item()
        return check_finite(name, v)
    return False, False

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Config: SUBSET_RATIO={C.SUBSET_RATIO}, STAGED_TRAINING={C.STAGED_TRAINING}, USE_KENDALL={C.USE_KENDALL}")

    seed_everything(C.SEED)

    # Build datasets
    logger.info("Loading datasets...")
    try:
        train_ds = IndustRealMultiTaskDataset(split='train')
        val_ds = IndustRealMultiTaskDataset(split='val')
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

    logger.info(f"Train ds: {len(train_ds)} frames, Val ds: {len(val_ds)} frames")

    # Build model
    logger.info("Building model...")
    try:
        model = POPWMultiTaskModel(
            pretrained=True,
            backbone_type=C.BACKBONE,
            use_headpose_film=C.USE_HEADPOSE_FILM,
            use_hand_film=C.USE_HAND_FILM,
            use_videomae=C.USE_VIDEOMAE,
            train_pose=C.TRAIN_HEAD_POSE,
        )
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        return False

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params/1e6:.1f}M")

    # Build criterion
    criterion = MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT - 1,
        num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=C.TRAIN_DET,
        train_pose=C.TRAIN_HEAD_POSE,
        train_act=C.TRAIN_ACT,
        train_psr=C.TRAIN_PSR,
        use_kendall=C.USE_KENDALL,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=C.BASE_LR, weight_decay=C.WEIGHT_DECAY)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=C.PIN_MEMORY, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=C.VAL_BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=C.PIN_MEMORY
    )

    logger.info(f"Batch size: {C.BATCH_SIZE}, Grad accum: {C.GRAD_ACCUM_STEPS}")

    # ====================== TRAIN ======================
    model.train()
    running_losses = {k: 0.0 for k in ['total', 'det', 'det_cls', 'det_reg', 'head_pose', 'activity', 'psr']}
    num_batches = 0
    nan_skips = 0
    scaler = torch.amp.GradScaler('cuda', enabled=C.MIXED_PRECISION)
    accum_steps = C.GRAD_ACCUM_STEPS
    optimizer.zero_grad(set_to_none=True)

    from tqdm import tqdm
    pbar = tqdm(train_loader, desc='Training', leave=True)

    for step, (images, targets) in enumerate(pbar):
        if step >= MAX_TRAIN_BATCHES:
            logger.info(f"\n  Reached MAX_TRAIN_BATCHES={MAX_TRAIN_BATCHES}, stopping training.")
            break

        images = images.to(device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
            mean = torch.tensor(C.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            std = torch.tensor(C.IMAGENET_STD, device=device).view(1, 3, 1, 1)
            images = (images - mean) / std

        for i in range(len(targets['detection'])):
            targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].to(device)
            targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to(device)
        targets['head_pose'] = targets['head_pose'].to(device)
        targets['psr_labels'] = targets['psr_labels'].to(device)
        targets['activity'] = targets['activity'].to(device)

        with torch.amp.autocast('cuda', enabled=C.MIXED_PRECISION):
            outputs = model(images)
            for _k in ['cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits']:
                if _k in outputs and isinstance(outputs[_k], torch.Tensor):
                    outputs[_k] = outputs[_k].float()

            criterion.set_epoch(0)
            loss, loss_dict = criterion(outputs, targets)

            # Ensure loss stays as proper gradient tensor
            if not loss.requires_grad or not loss.grad_fn:
                logger.error(f"  loss.requires_grad={loss.requires_grad}, loss.grad_fn={loss.grad_fn}")
                logger.error(f"  loss_dict types: {[(k, type(v)) for k,v in loss_dict.items()]}")
                raise RuntimeError("Loss tensor lost grad_fn!")

            loss = loss / float(accum_steps)

        if not torch.isfinite(loss):
            nan_skips += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix_str(f'NaN skip #{nan_skips}', refresh=True)
            continue

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), C.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k in running_losses:
            if k in loss_dict:
                running_losses[k] += loss_dict[k]
        num_batches += 1

        pbar.set_postfix_str(
            f"loss={loss_dict.get('total', 0):.3f} "
            f"det={loss_dict.get('det', 0):.3f} "
            f"hp={loss_dict.get('head_pose', 0):.3f} "
            f"act={loss_dict.get('activity', 0):.3f} "
            f"psr={loss_dict.get('psr', 0):.3f}",
            refresh=True
        )

    avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
    logger.info(f"\n=== TRAINING SUMMARY (batches={num_batches}, NaN skips={nan_skips}) ===")
    all_ok = True
    for k, v in avg_losses.items():
        is_nan, is_zero = check_finite(k, v)
        status = "FAIL: NaN" if is_nan else ("FAIL: ZERO" if is_zero else "PASS")
        logger.info(f"  {k:12s}: {v:.6f} [{status}]")
        if is_nan or is_zero:
            all_ok = False

    # ====================== VALIDATION (quick) ======================
    logger.info(f"\n=== VALIDATION (first {C.EVAL_MAX_BATCHES} batches) ===")
    model.eval()
    val_metrics = {'det_mAP50': [], 'act_accuracy': [], 'psr_f1': [], 'head_pose_MAE': [], 'assembly_f1': [], 'error_detection_f1': []}

    with torch.no_grad():
        for val_step, (images, targets) in enumerate(tqdm(val_loader, desc='Validation', leave=True)):
            if val_step >= C.EVAL_MAX_BATCHES:
                break

            images = images.to(device, non_blocking=True)
            if images.dtype == torch.uint8:
                images = images.float().div_(255.0)
                mean = torch.tensor(C.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                std = torch.tensor(C.IMAGENET_STD, device=device).view(1, 3, 1, 1)
                images = (images - mean) / std

            for i in range(len(targets['detection'])):
                targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].to(device)
                targets['detection'][i]['labels'] = targets['detection'][i]['labels'].to(device)
            targets['head_pose'] = targets['head_pose'].to(device)
            targets['psr_labels'] = targets['psr_labels'].to(device)
            targets['activity'] = targets['activity'].to(device)

            with torch.amp.autocast('cuda', enabled=C.MIXED_PRECISION):
                outputs = model(images)
                for _k in ['cls_preds', 'reg_preds', 'head_pose', 'psr_logits', 'act_logits']:
                    if _k in outputs and isinstance(outputs[_k], torch.Tensor):
                        outputs[_k] = outputs[_k].float()

            # Quick compute per-task metrics from outputs
            # detection mAP50 approximation
            if 'cls_preds' in outputs and 'det_mAP50' not in val_metrics:
                val_metrics['det_mAP50'].append(0.0)
            val_metrics['det_mAP50'].append(outputs.get('det_mAP50', 0.0))
            val_metrics['act_accuracy'].append(outputs.get('act_accuracy', outputs.get('act_clip_accuracy', 0.0)))
            val_metrics['psr_f1'].append(outputs.get('psr_f1', outputs.get('psr_overall_f1', 0.0)))
            val_metrics['head_pose_MAE'].append(outputs.get('head_pose_MAE', 0.0))
            val_metrics['assembly_f1'].append(outputs.get('assembly_f1', outputs.get('assembly_state_f1', 0.0)))
            val_metrics['error_detection_f1'].append(outputs.get('error_detection_f1', 0.0))

    logger.info(f"\n=== VALIDATION METRICS ===")
    for k, vals in val_metrics.items():
        if vals:
            v = vals[0].item() if hasattr(vals[0], 'item') else vals[0]
            is_nan, is_zero = check_finite(k, v)
            status = "FAIL: NaN" if is_nan else ("FAIL: ZERO" if is_zero else "PASS")
            logger.info(f"  {k:20s}: {v:.6f} [{status}]")
            if is_nan or is_zero:
                all_ok = False

    # ====================== FINAL ======================
    logger.info(f"\n{'='*50}")
    if all_ok:
        logger.info("✅ ALL losses and metrics: non-zero and non-NaN")
    else:
        logger.info("❌ SOME values are NaN or zero - see FAILs above")
    logger.info("="*50)
    return all_ok

if __name__ == '__main__':
    success = run()
    sys.exit(0 if success else 1)