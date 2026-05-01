#!/usr/bin/env python3
"""
Quick validation test to verify evaluate_all exports all benchmark metrics.
Uses the pretrain_synthetic checkpoint as a sanity check.
"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import POPWMultiTaskModel
from losses import MultiTaskLoss
from industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from torch.utils.data import DataLoader
import config as C
from evaluate import evaluate_all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Build model
model = POPWMultiTaskModel(
    pretrained=True,
    backbone_type=C.BACKBONE,
    use_headpose_film=C.USE_HEADPOSE_FILM,
    use_videomae=C.USE_VIDEOMAE,
).to(device)

criterion = MultiTaskLoss(
    num_classes_act=C.NUM_CLASSES_ACT,
    num_psr_components=C.NUM_PSR_COMPONENTS,
    train_det=C.TRAIN_DET,
    train_pose=C.TRAIN_HEAD_POSE,
    train_act=C.TRAIN_ACT,
    train_psr=C.TRAIN_PSR,
    use_kendall=C.USE_KENDALL,
).to(device)

# Load pretrain_synthetic checkpoint (most recent usable checkpoint)
ckpt_path = 'runs/manual_only_tma_tbank_benchmark/checkpoints/best.pth'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
else:
    print(f"WARNING: No checkpoint at {ckpt_path}, using random weights")

# Val dataset
val_ds = IndustRealMultiTaskDataset(
    split='val',
    img_size=(C.IMG_WIDTH, C.IMG_HEIGHT),
    augment=False,
)
val_loader = DataLoader(val_ds, batch_size=4, num_workers=0, collate_fn=collate_fn, pin_memory=False)
print(f"Val dataset: {len(val_ds)} clips, {len(val_loader)} batches")

# Run evaluation on 5 batches
print("\nRunning evaluate_all on 5 batches...")
results = evaluate_all(model, criterion, val_loader, device, max_batches=5)

print("\n" + "="*60)
print("BENCHMARK METRICS VERIFICATION")
print("="*60)

benchmark_keys = {
    'det_mAP50': 'ASD mAP@0.5',
    'det_mAP_50_95': 'ASD mAP@[0.5:0.95]',
    'act_accuracy': 'Activity Top-1',
    'act_top5_accuracy': 'Activity Top-5',
    'psr_overall_f1': 'PSR Overall F1',
    'psr_f1_at_t5': 'PSR F1@±5',
    'psr_precision_at_t5': 'PSR P@±5',
    'psr_recall_at_t5': 'PSR R@±5',
    'psr_f1_at_t3': 'PSR F1@±3',
    'psr_precision_at_t3': 'PSR P@±3',
    'psr_recall_at_t3': 'PSR R@±3',
    'psr_pos': 'PSR POS',
    'psr_edit_score': 'PSR Edit Score',
    'as_f1': 'AS F1@1',
    'as_top1_accuracy': 'AS Top-1 Acc',
    'as_map_at_r': 'AS MAP@R(+)',
    'ev_ap': 'EV AP',
    'ev_f1': 'EV F1',
    'forward_angular_MAE_deg': 'Forward Angular MAE (deg)',
    'up_angular_MAE_deg': 'Up Angular MAE (deg)',
    'position_MAE_mm': 'Position MAE (mm)',
    'head_pose_MAE': 'Head Pose MAE',
}

all_ok = True
for key, label in benchmark_keys.items():
    if key in results:
        val = results[key]
        if isinstance(val, float):
            print(f"  {label:30s}: {val:.4f}  [OK]")
        else:
            print(f"  {label:30s}: {val}  [OK]")
    else:
        print(f"  {label:30s}: MISSING  [FAIL]")
        all_ok = False

print()
if all_ok:
    print("ALL BENCHMARK METRICS PRESENT ✓")
else:
    print("SOME METRICS MISSING - FIX REQUIRED")

# Also check total keys
print(f"\nTotal scalar metric keys returned: {len([k for k,v in results.items() if isinstance(v,(int,float)) and not isinstance(v,bool)])}")
