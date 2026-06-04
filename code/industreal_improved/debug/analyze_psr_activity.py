#!/usr/bin/env python3
"""
Investigate PSR near-zero loss and Activity NaN.
Add to training loop temporarily to capture logits during training.
"""
import sys
import os
from pathlib import Path

# Setup
SRC = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive')
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / 'src'))
sys.path.insert(0, str(SRC / 'src' / 'training'))

import config as C
import torch
import torch.nn.functional as F
import numpy as np

def analyze_psr_loss():
    """
    Simulate PSR loss computation with the data from train.log.
    The log shows:
    - PSR prevalence: [1.0, 0.683, 0.801, 0.709, 0.064, 0.769, 0.714, 0.376, 0.376, 0.271, 0.152]
    - PSR loss = 0.0003
    - psr_logits sample from model: [8.09, 4.20, 1.76, -0.87, -0.59, -2.12, -1.28, -2.05, -1.98, -1.65, -0.82]
    - psr_labels sample: [1.0, 0.683..., 0.064...]
    """
    print("="*60)
    print("PSR LOSS ANALYSIS")
    print("="*60)

    # From train.log: prevalence
    prevalence = torch.tensor([1.0, 0.683, 0.801, 0.709, 0.064, 0.769, 0.714, 0.376, 0.376, 0.271, 0.152])

    # Per-component alpha = 2 * (1 - prevalence)
    alpha_per_component = 2.0 * (1.0 - prevalence)
    print(f"\nPer-component alpha (from prevalence):")
    for c in range(11):
        print(f"  component {c}: prevalence={prevalence[c]:.3f}, alpha={alpha_per_component[c]:.4f}")

    # Compute alpha_c
    prev = prevalence.float().clamp(0.01, 0.99)
    alpha_c = 2.0 * (1.0 - prev)
    print(f"\nalpha_c (same as above): {alpha_c.numpy().round(3).tolist()}")

    # Simulated psr_logits (from model output - typical values)
    # These are sigmoid inputs - high values mean model predicts 1
    psr_logits = torch.tensor([
        [8.09, 4.20, 1.76, -0.87, -0.59, -2.12, -1.28, -2.05, -1.98, -1.65, -0.82],  # B=0
        [7.50, 3.80, 1.50, -0.50, -0.30, -1.80, -1.00, -1.90, -1.80, -1.40, -0.60],  # B=1
    ])
    psr_labels = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # B=0
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # B=1
    ])

    print(f"\n=== Simulated PSR logits and labels ===")
    preds = torch.sigmoid(psr_logits)
    for b in range(2):
        print(f"\n  Batch {b}:")
        for c in range(11):
            print(f"    component {c}: logit={psr_logits[b,c]:.3f}, pred={preds[b,c]:.4f}, label={psr_labels[b,c]:.1f}")

    # Compute BCE per component
    print(f"\n=== Per-component BCE loss ===")
    loss_per_comp = []
    for c in range(11):
        loss_c = F.binary_cross_entropy_with_logits(psr_logits[:, c], psr_labels[:, c])
        loss_per_comp.append(loss_c.item())
        print(f"  component {c}: BCE={loss_c.item():.6f}, alpha={alpha_c[c]:.4f}, weighted={loss_c.item() * alpha_c[c]:.6f}")

    total_bce = sum(loss_per_comp) / 11  # mean across components
    print(f"\nTotal BCE (mean of per-component): {total_bce:.6f}")

    # With focal alpha weighting
    total_focal = sum(loss_per_comp[i] * alpha_c[i] for i in range(11)) / 11
    print(f"Total focal-weighted BCE: {total_focal:.6f}")

    # Temporal smooth loss
    print(f"\n=== Temporal Smooth Loss ===")
    smooth_weight = 0.05
    psr_preds = torch.sigmoid(psr_logits)
    smooth_loss = torch.tensor(0.0)
    bs = psr_preds.shape[0]
    for i in range(bs):
        p_i = psr_preds[i]
        l_i = psr_labels[i]
        diff_p = (p_i[1:] - p_i[:-1]).abs().mean()
        diff_l = (l_i[1:] - l_i[:-1]).abs().mean()
        pred_change = torch.sigmoid(diff_p)  # sigmoid on diff (small values)
        label_change = diff_l  # already in [0, 1]
        smooth_loss = smooth_loss + ((pred_change - label_change) ** 2)
    smooth_loss = smooth_loss / max(bs, 1)
    print(f"Temporal smooth loss: {smooth_loss.item():.6f}")
    print(f"With weight ({smooth_weight}): {smooth_weight * smooth_loss.item():.6f}")

    # Check: what if temporal smooth is near zero?
    # diff_p is small because sigmoid(small) ≈ small value
    # diff_l for component 4 (label changes 0→0→0) = 0
    # But some components DO change: label[0]=1 and label[1]=1, all same
    # The diff_l is 0 for all, but diff_p is small positive
    # smooth_loss is dominated by pred_change - label_change ≈ pred_change

    print(f"\n=== KEY INSIGHT ===")
    print(f"The model predicts component 0 (prevalence=1.0, alpha=0) as 0.9997.")
    print(f"BCE for component 0: {F.binary_cross_entropy_with_logits(psr_logits[0,0:1], psr_labels[0,0:1]).item():.6f}")
    print(f"With alpha=0: {F.binary_cross_entropy_with_logits(psr_logits[0,0:1], psr_labels[0,0:1]).item() * 0:.6f}")
    print(f"")
    print(f"Component 4 (label=0, rare, alpha=1.872): pred={preds[0,4]:.4f}")
    comp4_bce = F.binary_cross_entropy_with_logits(psr_logits[0,4:5], psr_labels[0,4:5])
    print(f"BCE for component 4: {comp4_bce:.6f}")
    print(f"With focal alpha: {comp4_bce * alpha_c[4]:.6f}")
    print(f"")
    print(f"Total PSR loss with temporal smooth: {total_focal + smooth_weight * smooth_loss:.6f}")
    print(f"Expected range: ~1-5 like detection (4.3369)")
    print(f"Actual observed: 0.0003")
    print(f"")
    print(f"WHY PSR IS NEAR-ZERO:")
    print(f"1. Component 0 has prevalence=1.0 and alpha=0 → model learns to predict it perfectly")
    print(f"2. With perfect prediction (pred=0.9997, label=1.0), BCE ≈ 0.0027")
    print(f"3. Most other common components also have high prevalence (0.68-0.80) → alpha low → low loss")
    print(f"4. Rare components (4, 10, 9, 8) have low prevalence but may not dominate the average")

    # Check: maybe the loss is being zeroed somewhere?
    print(f"\n=== Check temporal smooth effect ===")
    print(f"Temporal smooth with label changes: For batch 0, labels are mostly 1→1→1 (no change)")
    print(f"diff_l = 0 for all transitions → label_change = 0")
    print(f"diff_p = (pred[1:] - pred[:-1]).abs().mean() — small because predictions stable")
    print(f"pred_change = sigmoid(diff_p) ≈ diff_p (small)")
    print(f"Loss = (pred_change - 0)^2 = pred_change^2 ≈ small^2 ≈ very small")
    print(f"This explains why temporal smooth contributes almost nothing")


def analyze_activity_nan():
    """
    Investigate activity NaN.
    From train.log: act=nan at batch 3110, epoch 0.
    """
    print("\n" + "="*60)
    print("ACTIVITY NaN ANALYSIS")
    print("="*60)

    print(f"\nPossible causes:")
    print(f"1. LDAM-DRW cross_entropy producing NaN from extreme logits")
    print(f"2. Logits are all zeros or extreme values")
    print(f"3. NaN from label_smoothing when logits contain inf")
    print(f"4. Activity ramp at epoch 0 = (0+1)/5 = 0.2 (not 0)")

    # Check LDAM forward
    print(f"\nLDAM s=30, x_m.clamp(-10, 10)")
    print(f"If logits are large (e.g., 50), s*x_m = 30*40 = 1200")
    print(f"softmax(1200) is numerically unstable → inf → log(inf) = inf → NaN")

    print(f"\nChecking code at losses.py line 397:")
    print(f"F.cross_entropy(self.s * x_m, hard_targets, reduction='none', label_smoothing=0.1)")
    print(f"s=30, x_m.clamp(-10, 10)")
    print(f"If logits are e.g., [100, 90, 80...] - margin subtracted → x_m can be large")
    print(f"30 * large_x_m → overflow → inf → cross_entropy → NaN")

    print(f"\nGuard at line 389: x_m = x_m.clamp(-10.0, 10.0)")
    print(f"But this only handles extreme negative values. Large positive values remain.")
    print(f"30 * 10 = 300, softmax(300) is still extreme but not inf")

    print(f"\nAlso check: the NaN guard at line 789-797 catches NaN after act_loss_fn")
    print(f"But the guard at line 818-823 uses torch.where which passes gradients through")
    print(f"If loss_act is NaN, torch.where returns NaN (not zero)")

    print(f"\nCONCLUSION: Activity NaN likely caused by:")
    print(f"- LDAM s=30 amplifying logits → cross_entropy overflow → NaN")
    print(f"- The NaN guard should catch this but loss_dict shows NaN persisted")


if __name__ == '__main__':
    analyze_psr_loss()
    analyze_activity_nan()