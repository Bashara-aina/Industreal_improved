"""
Training script for POPW multi-task model.
Implements: staged training, Kendall homoscedastic uncertainty weighting,
mixed precision, EMA, gradient clipping, logging.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel

from config import C
from model import POPWModel


class ModelEma:
    """Exponential Moving Average wrapper for model weights.

    Uses torch's AveragedModel under the hood for efficient shadow model
    maintenance. Update after every optimizer step. Apply shadow for eval.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.n_averaged = 0

        def _ema_avg_fn(param, old, new):
            return self.decay * old + (1 - self.decay) * new

        self.module = model
        self._shadow = {}
        self._ema_avg_fn = _ema_avg_fn

    def update(self, model=None):
        """Update EMA shadow weights after optimizer step."""
        if model is None:
            model = self.module

        if self.n_averaged == 0:
            for p, sp in zip(model.parameters(), self.module.parameters()):
                sp.data.copy_(p.detach().data)
            for b, sb in zip(model.buffers(), self.module.buffers()):
                sb.data.copy_(b.detach().data)

        for p, sp in zip(model.parameters(), self.module.parameters()):
            sp.data = self._ema_avg_fn(p.detach(), sp.data, p.detach())

        for b, sb in zip(model.buffers(), self.module.buffers()):
            sb.data = self._ema_avg_fn(b.detach(), sb.data, b.detach())

        self.n_averaged += 1
from losses import (
    FocalLoss, WingLoss, LDAMLoss, BinaryFocalLoss,
    TemporalSmoothnessLoss, GIoULoss, KendallMultiTaskLoss, MSELoss,
    DetectionLoss
)
from industreal_dataset import IndustRealDataset, collate_fn, Transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train POPW multi-task model")
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited data")
    parser.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS, help="Max training epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=C.SEED, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_stage(epoch: int) -> int:
    """
    Determine current training stage based on epoch.

    Stage 1 (epochs 1-5): Detection only, backbone frozen except layer4
    Stage 2 (epochs 6-15): + Pose + Head Pose, Activity and PSR frozen
    Stage 3 (epoch 16+): All task groups active
    """
    if epoch <= C.STAGE1_EPOCHS:
        return 1
    elif epoch <= C.STAGE1_EPOCHS + C.STAGE2_EPOCHS:
        return 2
    else:
        return 3


def setup_model(checkpoint_path: str = None):
    """Initialize model and criterion."""
    model = POPWModel(config=C)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Kendall multi-task loss
    criterion = KendallMultiTaskLoss(
        num_tasks=4,
        init_values=(C.KENDALL_INIT_DET, C.KENDALL_INIT_POSE,
                     C.KENDALL_INIT_ACT, C.KENDALL_INIT_PSR),
        s_min=C.KENDALL_S_MIN,
        s_max=C.KENDALL_S_MAX
    )

    # Individual losses
    focal_loss = FocalLoss(alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA)
    giou_loss = GIoULoss()
    wing_loss = WingLoss(omega=C.WING_LOSS_W, epsilon=C.WING_LOSS_EPS)
    mse_loss = MSELoss(scale=C.HEAD_POSE_LOSS_SCALE)
    ldam_loss = LDAMLoss(num_classes=C.NUM_CLASSES_ACT,
                         label_smooth=C.LDAM_LABEL_SMOOTH,
                         drw_epoch=C.LDAM_DRW_EPOCH)
    binary_focal = BinaryFocalLoss(alpha=C.PSR_FOCAL_ALPHA, gamma=C.PSR_FOCAL_GAMMA)
    smoothness_loss = TemporalSmoothnessLoss(weight=C.PSR_SMOOTHNESS_WEIGHT)
    detection_loss_fn = DetectionLoss(
        anchor_sizes=C.ANCHOR_SIZES,
        anchor_ratios=C.ANCHOR_RATIOS,
        num_classes=C.NUM_CLASSES_DET
    )

    return model, criterion, {
        'focal': focal_loss,
        'giou': giou_loss,
        'wing': wing_loss,
        'mse': mse_loss,
        'ldam': ldam_loss,
        'binary_focal': binary_focal,
        'smoothness': smoothness_loss,
        'detection': detection_loss_fn
    }


def set_stage_requires_grad(model: nn.Module, stage: int):
    """
    Set requires_grad based on current training stage.

    Stage 1: backbone.layer4 + detection head only
    Stage 2: + head pose + body pose heads
    Stage 3: all parameters
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    if stage == 1:
        # Detection only - backbone layer4 + detection head
        for param in model.backbone.parameters():
            param.requires_grad = True
        for param in model.fpn.parameters():
            param.requires_grad = True
        for param in model.detection_head.parameters():
            param.requires_grad = True

    elif stage == 2:
        # Detection + Pose/HeadPose
        for param in model.parameters():
            param.requires_grad = True
        # Freeze activity and PSR
        for param in model.activity_head.parameters():
            param.requires_grad = False
        for param in model.psr_head.parameters():
            param.requires_grad = False

    elif stage == 3:
        # All tasks
        for param in model.parameters():
            param.requires_grad = True


def compute_losses(outputs: dict, targets: dict, losses_dict: dict,
                   criterion: KendallMultiTaskLoss, stage: int, epoch: int,
                   batch_idx: int = 0) -> tuple:
    """
    Compute multi-task losses.

    Args:
        outputs: model outputs
        targets: ground truth labels
        losses_dict: individual loss functions
        criterion: Kendall multi-task loss
        stage: current training stage
        epoch: current epoch

    Returns:
        total_loss, loss_components
    """
    # Find first tensor in outputs for device (handles nested lists)
    _device = torch.device('cpu')
    for v in outputs.values():
        if isinstance(v, torch.Tensor):
            _device = v.device
            break
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.Tensor):
                    _device = item.device
                    break
            if _device != torch.device('cpu'):
                break

    loss_det = torch.tensor(0.0, device=_device)
    loss_pose = torch.tensor(0.0, device=_device)
    loss_act = torch.tensor(0.0, device=_device)
    loss_psr = torch.tensor(0.0, device=_device)

    # Detection loss (Stage 1+)
    if stage >= 1 and 'det_labels' in targets and targets['det_labels'] is not None:
        det_labels = targets['det_labels']
        if isinstance(det_labels, list) and len(det_labels) > 0:
            det_lab = det_labels[batch_idx] if batch_idx < len(det_labels) else None
            if det_lab is not None and len(det_lab['boxes']) > 0:
                # Get feature map shapes and strides
                # P2: 180x320, P3: 90x160, P4: 45x80, P5: 22x40, P6: 11x20, P7: 6x10 (for 720x1280)
                # Note: DetectionHead outputs P3-P7 (5 levels), matching cls_preds[0..4]
                feature_shapes = [(90, 160), (45, 80), (22, 40), (11, 20), (6, 10)]
                strides = [8, 16, 32, 64, 128]

                # Use cls_preds and reg_preds from outputs
                cls_p = [p[batch_idx:batch_idx+1] for p in outputs['cls_preds']]
                reg_p = [p[batch_idx:batch_idx+1] for p in outputs['reg_preds']]

                # boxes and labels need to be [B, M, 4] and [B, M]
                boxes = det_lab['boxes'].unsqueeze(0)  # [1, M, 4]
                labels = det_lab['labels'].unsqueeze(0)  # [1, M]

                det_loss, det_components = losses_dict['detection'](
                    cls_p, reg_p, boxes, labels, feature_shapes, strides
                )
                loss_det = det_loss

    # Pose loss (Stage 2+)
    if stage >= 2 and 'body_kpts' in outputs and 'pose_targets' in targets:
        pred_kpts = outputs['body_kpts']
        target_kpts = targets['pose_targets']
        if pred_kpts is not None and target_kpts is not None:
            loss_pose = losses_dict['wing'](pred_kpts, target_kpts)

    # Head pose loss (Stage 2+)
    if stage >= 2 and 'head_pose' in outputs and 'head_pose_targets' in targets:
        pred_pose = outputs['head_pose']
        target_pose = targets['head_pose_targets']
        if pred_pose is not None and target_pose is not None:
            loss_pose = loss_pose + losses_dict['mse'](pred_pose, target_pose)

    # Activity loss (Stage 3)
    if stage >= 3 and 'act_logits' in outputs and 'activity_labels' in targets:
        act_logits = outputs['act_logits']
        act_labels = targets['activity_labels']
        loss_act = losses_dict['ldam'](act_logits, act_labels, epoch=epoch)

    # PSR loss (Stage 3)
    # Handle both single-frame [B, 11] and sequence mode [B*T, 11] or [B, T, 11]
    if stage >= 3 and 'psr_logits' in outputs and 'psr_labels' in targets:
        psr_logits = outputs['psr_logits']
        psr_labels = targets['psr_labels']

        # Detect sequence mode: psr_labels is [B, T, 11]
        if psr_labels.dim() == 3:
            # Sequence mode: flatten B,T dims for loss computation
            T = psr_labels.size(1)
            psr_logits_flat = psr_logits.view(-1, psr_logits.size(-1))  # [B*T, 11]
            psr_labels_flat = psr_labels.view(-1, 11)  # [B*T, 11]
            loss_psr = losses_dict['binary_focal'](psr_logits_flat, psr_labels_flat)
            # Smoothness on reshaped logits [B, T, 11]
            loss_psr = loss_psr + losses_dict['smoothness'](psr_logits.view(psr_labels.size(0), T, -1).unsqueeze(2))
        else:
            # Single-frame mode
            loss_psr = losses_dict['binary_focal'](psr_logits, psr_labels)
            loss_psr = loss_psr + losses_dict['smoothness'](outputs.get('psr_logits', torch.zeros(1)).unsqueeze(1))

    # Kendall weighting
    losses = (loss_det, loss_pose, loss_act, loss_psr)

    # Task masks: which tasks are active
    if stage == 1:
        task_mask = (1, 0, 0, 0)  # Detection only
    elif stage == 2:
        task_mask = (1, 1, 0, 0)  # Detection + Pose
    else:
        task_mask = (1, 1, 1, 1)  # All

    total_loss, loss_components = criterion(losses, task_mask)

    return total_loss, loss_components


def train_one_epoch(model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: KendallMultiTaskLoss, losses_dict: dict,
                    epoch: int, device: torch.device, use_amp: bool = False,
                    grad_scaler: GradScaler = None, debug: bool = False,
                    loader_seq: DataLoader = None,
                    ema: Optional[ModelEma] = None):
    """Train one epoch with optional sequence mode alternation in Stage 3.

    Args:
        train_loader: single-frame mode loader (detection/pose/activity)
        loader_seq: sequence mode loader (PSR) - used every other batch in Stage 3
    """
    model.train()
    stage = get_stage(epoch)
    set_stage_requires_grad(model, stage)

    total_loss = 0.0
    num_batches = 0
    loss_breakdown = {"det": 0, "pose": 0, "act": 0, "psr": 0}

    # Iterators for alternating between loaders in Stage 3
    iter_single = None
    iter_seq = None
    if stage >= 3 and loader_seq is not None:
        iter_single = iter(train_loader)
        iter_seq = iter(loader_seq)

    for batch_idx in range(len(train_loader) if stage < 3 else max(len(train_loader), len(loader_seq) if loader_seq else 0)):
        if debug and batch_idx >= 20:
            break

        # Alternate between single-frame and sequence mode in Stage 3
        # Even batches = single-frame, Odd batches = sequence
        use_sequence = (stage >= 3 and loader_seq is not None and batch_idx % 2 == 1)

        if use_sequence and iter_seq is not None:
            # Sequence mode batch
            try:
                batch = next(iter_seq)
            except StopIteration:
                iter_seq = iter(loader_seq)
                batch = next(iter_seq)
            mode_str = "SEQ"
        else:
            # Single-frame mode batch
            if stage >= 3 and iter_single is not None:
                try:
                    batch = next(iter_single)
                except StopIteration:
                    iter_single = iter(train_loader)
                    batch = next(iter_single)
            else:
                batch = next(iter(train_loader))
            mode_str = "SINGLE"

        # Move to device
        images = batch['images'].to(device)
        targets = {
            'det_labels': batch.get('det_labels'),
            'psr_labels': batch.get('psr_labels').to(device),
            'head_pose_targets': batch.get('head_pose').to(device),
            'activity_labels': batch.get('activity_labels').to(device),
        }

        # Forward pass
        if use_amp:
            with autocast():
                outputs = model(images, video_id=batch['recording_ids'][0] if batch['recording_ids'] else "train")
        else:
            outputs = model(images, video_id=batch['recording_ids'][0] if batch['recording_ids'] else "train")

        loss, loss_comp = compute_losses(outputs, targets, losses_dict, criterion, stage, epoch, batch_idx)

        # Backward
        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if ema is not None:
                ema.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update()

        # Accumulate
        total_loss += loss.item()
        num_batches += 1
        for k, v in loss_comp.items():
            loss_breakdown[k] += v.item() if torch.is_tensor(v) else v

        # Log Kendall gradients (every 100 steps)
        if batch_idx % 100 == 0:
            log_vars = criterion.log_vars
            print(f"  Step {batch_idx}: loss={loss.item():.4f}, "
                  f"det={loss_breakdown['det']:.4f}, pose={loss_breakdown['pose']:.4f}, "
                  f"act={loss_breakdown['act']:.4f}, psr={loss_breakdown['psr']:.4f}")
            print(f"  Kendall log_vars: det={log_vars[0].item():.3f}, pose={log_vars[1].item():.3f}, "
                  f"act={log_vars[2].item():.3f}, psr={log_vars[3].item():.3f}")
            for i, name in enumerate(['det', 'pose', 'act', 'psr']):
                if log_vars[i].grad is not None:
                    print(f"  log_var_{name} grad norm: {log_vars[i].grad.norm().item():.6f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch} [{stage}] - Avg loss: {avg_loss:.4f}")
    print(f"  Loss breakdown: det={loss_breakdown['det']/num_batches:.4f}, "
          f"pose={loss_breakdown['pose']/num_batches:.4f}, "
          f"act={loss_breakdown['act']/num_batches:.4f}, "
          f"psr={loss_breakdown['psr']/num_batches:.4f}")

    # Clamp Kendall vars
    criterion.clamp_vars()

    return avg_loss


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup model
    model, criterion, losses_dict = setup_model(args.resume)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total parameters: {total_params:.2f}M")
    print(f"Trainable parameters: {trainable_params:.2f}M")
    print(f"Backbone: {C.BACKBONE}")

    # Optimizer
    if C.OPTIMIZER.lower() == "lion":
        try:
            from lion.pytorch import Lion
            optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr or C.BASE_LR,
                           weight_decay=C.WEIGHT_DECAY)
            print("Optimizer: Lion")
        except ImportError:
            print("Lion not installed, using AdamW")
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr or C.BASE_LR,
                                  weight_decay=C.WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr or C.BASE_LR,
                              weight_decay=C.WEIGHT_DECAY)

    # GradScaler for mixed precision
    grad_scaler = GradScaler() if C.MIXED_PRECISION else None

    # Datasets
    train_transform = Transforms(is_train=True)

    # Single-frame mode dataset (detection/pose/activity)
    train_ds_single = IndustRealDataset(split="train", transform=train_transform,
                                       sequence_mode=False)
    loader_single = DataLoader(train_ds_single, batch_size=args.batch_size,
                               shuffle=True, num_workers=C.NUM_WORKERS,
                               collate_fn=collate_fn, pin_memory=True)

    # Sequence mode dataset (PSR training) - used every other batch in Stage 3
    train_ds_seq = IndustRealDataset(split="train", transform=train_transform,
                                     sequence_mode=True, sequence_length=16)
    loader_seq = DataLoader(train_ds_seq, batch_size=2,  # Smaller batch for sequences
                            shuffle=True, num_workers=C.NUM_WORKERS,
                            collate_fn=collate_fn, pin_memory=True)

    print(f"Single-frame dataset: {len(train_ds_single)} samples, {len(loader_single)} batches")
    print(f"Sequence dataset: {len(train_ds_seq)} sequences, {len(loader_seq)} batches")

    # EMA setup (per GAP 1)
    ema = None
    if C.USE_EMA:
        ema = ModelEma(model, decay=C.EMA_DECAY)
        print(f"EMA enabled: decay={C.EMA_DECAY}")

    # Training loop
    start_epoch = 1
    best_loss = float('inf')
    patience_counter = 0

    log_data = []

    for epoch in range(start_epoch, args.max_epochs + 1):
        stage = get_stage(epoch)
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.max_epochs} - Stage {stage}")
        print(f"Backbone: convnext_tiny")
        print(f"Optimizer: {C.OPTIMIZER}")
        print(f"[stage={stage}]")

        # Train
        train_loss = train_one_epoch(
            model, loader_single, optimizer, criterion, losses_dict,
            epoch, device, use_amp=C.MIXED_PRECISION,
            grad_scaler=grad_scaler, debug=args.debug,
            loader_seq=loader_seq,
            ema=ema
        )

        # Logging
        log_entry = {
            'epoch': epoch,
            'stage': stage,
            'train_loss': train_loss,
            'kendall_weights': criterion.get_weights(),
            'timestamp': datetime.now().isoformat()
        }
        log_data.append(log_entry)

        # Save checkpoint
        checkpoint_path = Path(C.CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'criterion_state': criterion.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': train_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = Path(C.CHECKPOINT_DIR) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'criterion_state': criterion.state_dict(),
            }, best_path)
            print(f"Saved best model: {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= C.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs")
            break

        # Stage 3 reset Kendall (per Doc 02)
        if stage == 3 and epoch == C.STAGE1_EPOCHS + C.STAGE2_EPOCHS + 1:
            print("Stage 3 transition - resetting Kendall log_vars for act and psr")
            with torch.no_grad():
                criterion.log_vars[2].data.fill_(0.0)  # act
                criterion.log_vars[3].data.fill_(0.0)  # psr

            # GAP 3: Reinitialize EMA with fresh shadow weights
            if C.USE_EMA:
                print("Stage 3 transition - reinitializing EMA")
                ema = ModelEma(model, decay=C.EMA_DECAY)

    # Save final log
    log_path = Path(C.LOG_DIR) / "train_log.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved training log: {log_path}")

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Run directory: {C.RUN_DIR}")


if __name__ == "__main__":
    main()