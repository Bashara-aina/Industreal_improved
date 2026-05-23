#!/usr/bin/env python3
"""
Debug script to investigate:
1. Activity NaN - what produces NaN in activity logits/loss
2. PSR near-zero loss - what are actual PSR logits
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Setup paths
sys.path.insert(0, os.path.dirname(__file__))
import config as C
from src.models.model import UnifiedModel

def load_checkpoint(path):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state' in ckpt:
        return ckpt['model_state'], ckpt.get('epoch', 0), ckpt.get('batch', 0)
    return ckpt, 0, 0

def inspect_model(checkpoint_path):
    """Inspect model outputs and losses."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict, epoch, batch = load_checkpoint(checkpoint_path)
    print(f"Checkpoint: epoch={epoch}, batch={batch}")

    # Build model
    model = POPWMultiTaskModel(
        backbone=C.BACKBONE,
        num_classes_det=C.NUM_DET_CLASSES,
        num_classes_act=C.NUM_CLASSES_ACT - 1,  # 74 not 75
        num_head_pose_dof=C.NUM_HEAD_POSE_DOF,
        num_psr_components=C.NUM_PSR_COMPONENTS,
        use_kendall=C.USE_KENDALL,
        train_head_pose=True,
        train_psr=True,
        train_act=True,
        train_det=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully")

    # Create fake input
    B, T, C_, H, W = 2, 16, 3, 720, 1280
    images = torch.randn(B, T, C_, H, W)
    keypoints = torch.randn(B, T, 26, 2)
    targets = {
        'cls_preds': torch.randn(B, 172440, 24),
        'reg_preds': torch.randn(B, 172440, 4),
        'act_logits': torch.randn(B, 74),  # Activity logits
        'psr_logits': torch.randn(B, 11),   # PSR logits
        'head_pose': torch.randn(B, 9),    # Head pose
    }
    targets['activity'] = torch.randint(0, 74, (B,))  # integer class labels
    targets['psr_labels'] = torch.rand(B, 11) > 0.5    # binary multi-label
    targets['head_pose'] = torch.randn(B, 9)

    print(f"\nInput shape: {images.shape}")
    print(f"Activity targets: {targets['activity']}")
    print(f"PSR labels shape: {targets['psr_labels'].shape}")

    # Forward pass with hooks to capture intermediate values
    with torch.no_grad():
        outputs = model(images, keypoints)

    print(f"\n=== MODEL OUTPUTS ===")
    for k, v in outputs.items():
        if v is not None:
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min().item():.4f}, max={v.max().item():.4f}, "
                  f"mean={v.mean().item():.4f}, has_nan={torch.isnan(v).any().item()}")

    # Inspect activity logits specifically
    act_logits = outputs.get('act_logits', None)
    if act_logits is not None:
        print(f"\n=== ACTIVITY LOGITS ANALYSIS ===")
        print(f"  act_logits shape: {act_logits.shape}")
        print(f"  act_logits min/max: {act_logits.min():.4f} / {act_logits.max():.4f}")
        print(f"  act_logits has NaN: {torch.isnan(act_logits).any()}")
        print(f"  act_logits has Inf: {torch.isinf(act_logits).any()}")
        print(f"  act_logits[0] sample: {act_logits[0][:10].tolist()}")

        # Check cross entropy manually
        targets_int = targets['activity']
        try:
            ce_raw = F.cross_entropy(act_logits, targets_int, reduction='none')
            print(f"  cross_entropy raw: {ce_raw}")
            print(f"  cross_entropy has NaN: {torch.isnan(ce_raw).any()}")
            print(f"  cross_entropy mean: {ce_raw.mean().item()}")
        except Exception as e:
            print(f"  cross_entropy ERROR: {e}")

        # Check LDAM if used
        if C.USE_LDAM_DRW:
            print(f"  LDAM-DRW is ENABLED (s={getattr(C, 'LDAM_S', 30)}, drw_epoch={getattr(C, 'LDAM_DRW_EPOCH', 60)})")

    # Inspect PSR logits
    psr_logits = outputs.get('psr_logits', None)
    if psr_logits is not None:
        print(f"\n=== PSR LOGITS ANALYSIS ===")
        print(f"  psr_logits shape: {psr_logits.shape}")
        print(f"  psr_logits min/max: {psr_logits.min():.4f} / {psr_logits.max():.4f}")
        print(f"  psr_logits has NaN: {torch.isnan(psr_logits).any()}")
        print(f"  psr_logits[0]: {psr_logits[0].tolist()}")

        # Manual BCE loss
        psr_labels = targets['psr_labels']
        bce = F.binary_cross_entropy_with_logits(psr_logits, psr_labels, reduction='mean')
        print(f"  BCE loss (manual): {bce.item():.6f}")

        # Sigmoid predictions
        psr_preds = torch.sigmoid(psr_logits)
        print(f"  PSR predictions[0]: {psr_preds[0].tolist()}")
        print(f"  PSR labels[0]: {psr_labels[0].tolist()}")

        # Per-component loss
        for c in range(11):
            comp_loss = F.binary_cross_entropy_with_logits(psr_logits[:, c], psr_labels[:, c])
            print(f"    component {c}: pred={psr_preds[0,c].item():.3f}, label={psr_labels[0,c].item():.1f}, loss={comp_loss.item():.6f}")

    # Check head pose
    head_pose = outputs.get('head_pose', None)
    if head_pose is not None:
        print(f"\n=== HEAD POSE ANALYSIS ===")
        print(f"  head_pose shape: {head_pose.shape}")
        print(f"  head_pose min/max: {head_pose.min():.4f} / {head_pose.max():.4f}")
        print(f"  head_pose has NaN: {torch.isnan(head_pose).any()}")
        mse = F.mse_loss(head_pose, targets['head_pose'])
        print(f"  MSE loss (manual): {mse.item():.4f}")

    # Check detection outputs
    cls_preds = outputs.get('cls_preds', None)
    if cls_preds is not None:
        print(f"\n=== DETECTION LOGITS ANALYSIS ===")
        print(f"  cls_preds shape: {cls_preds.shape}")
        print(f"  cls_preds has NaN: {torch.isnan(cls_preds).any()}")
        print(f"  cls_preds has Inf: {torch.isinf(cls_preds).any()}")

if __name__ == '__main__':
    import glob

    # Find latest checkpoint
    ckpt_dir = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints'
    checkpoints = sorted(glob.glob(f'{ckpt_dir}/epoch_0_batch_*.pth'))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Using latest checkpoint: {latest}")
        inspect_model(latest)
    else:
        print("No checkpoints found!")