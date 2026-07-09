#!/usr/bin/env python3
"""MVP Smoke Suite (Opus 192 §6) — Diagnose the four '0.0' / '0.008' numbers.

Purpose: separate "not working" from "not converged yet" from "eval is lying"
BEFORE committing 2-3 weeks of architecture redesign.

Run on GPU-2 while the Path-D run continues on GPU-1. Total ~1.5 days.

Probes:
  1. Overfit-200 sanity (eval-harness check). For each head, train the head
     (backbone frozen) to near-zero train loss on 200 fixed images/clips,
     then run the eval on those same 200. If the eval metric stays ~0
     while train loss → 0, the EVAL HARNESS IS BROKEN. The 0.468 prior
     detection result (176 §3.4) makes this a live hypothesis.
  2. ST-activity 5 epochs. ≥0.30 ⇒ head+backbone adequate; leave activity
     alone. <0.10 ⇒ deeper issue (data/label/eval).
  3. PSR on P5: reproduce 0.347 + temporal-resolution A/B. If
     predict-at-T=8 F1 ≫ interpolated (current T=16→8→16 linear),
     FC-4 is confirmed and the fix is a few lines.
  4. Detection TAL-lite vs 3×3 on overfit set. 3×3 already overfits
     to mAP ≥ 0.6 ⇒ assigner isn't the bottleneck.

Usage:
    python scripts/mvp_smoke_suite.py --probe 1 --head det
    python scripts/mvp_smoke_suite.py --probe 2 --head act
    python scripts/mvp_smoke_suite.py --probe 3 --head psr
    python scripts/mvp_smoke_suite.py --probe 4 --head det
    python scripts/mvp_smoke_suite.py --probe all
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Path setup
_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.NUM_ACT_OUTPUTS = 75
C.ACT_CLASS_GROUPING = "none"

from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
from src.models.mvit_mtl_model import MTLMViTModel
from scripts.train_mtl_mvit import train_step, detection_loss, activity_loss, psr_loss, pose_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FIXED = 200  # overfit set size


# ===========================================================================
# Probe 1: Overfit-200 eval-harness sanity
# ===========================================================================

def probe1_overfit_200(head: str, n_steps: int = 500):
    """Overfit 200 fixed images. If eval metric stays ~0, eval harness is broken.

    Args:
        head: One of 'det', 'act', 'psr', 'pose'
        n_steps: Number of optimization steps (~500 should be enough)
    """
    print(f"\n{'='*60}")
    print(f"PROBE 1: Overfit-200 sanity check for {head}")
    print(f"{'='*60}\n")

    # Build dataset
    train_ds = IndustRealMultiTaskDataset(
        split="train", img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    # Take first N_FIXED samples (deterministic)
    indices = list(range(min(N_FIXED, len(train_ds))))
    subset_ds = Subset(train_ds, indices)

    def fixed_collate(batch):
        # batch is list of (img, target) pairs
        images = torch.stack([b[0] for b in batch])
        targets_raw = [b[1] for b in batch]
        targets = {}
        for k, v in targets_raw.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v
            elif isinstance(v, list):
                targets[k] = v
            elif isinstance(v, dict):
                targets[k] = {sk: sv for sk, sv in v.items()}
        return images, targets

    loader = DataLoader(subset_ds, batch_size=2, shuffle=False, collate_fn=fixed_collate)

    # Build model
    model = MTLMViTModel(num_act_classes=75).to(DEVICE)
    model.train()

    # Freeze backbone, only train head
    for p in model.feature_pyramid.backbone.parameters():
        p.requires_grad = False

    # Build a minimal optimizer
    head_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and "backbone" not in name:
            head_params.append(p)
    optimizer = optim.AdamW(head_params, lr=1e-3)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=False)

    # Log_vars (not used in eval, but train_step needs them)
    log_vars = nn.ParameterDict({
        name: nn.Parameter(torch.tensor([0.0], device=DEVICE))
        for name in ["det", "act", "psr", "pose"]
    })

    print(f"Training {head} head on {N_FIXED} fixed images for {n_steps} steps...")
    losses = []
    for step, (images, targets) in enumerate(loader):
        if step >= n_steps:
            break
        images = images.to(DEVICE).float() / 255.0
        # normalize
        mean = torch.tensor([0.45, 0.45, 0.45], device=DEVICE).view(1, 1, 3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225], device=DEVICE).view(1, 1, 3, 1, 1, 1)
        images = (images - mean) / std
        images = images.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, T, 3, H, W] -> [B, 3, T, H, W]
        # Targets to device
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(DEVICE)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        for sk, sv in item.items():
                            if isinstance(sv, torch.Tensor):
                                item[sk] = sv.to(DEVICE)
        # Forward
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            if head == "det":
                loss = detection_loss(outputs["detection"], targets["detection"])
            elif head == "act":
                loss = activity_loss(outputs["activity"], targets["activity"])
            elif head == "psr":
                loss = psr_loss(outputs["psr_logits"], targets["psr_labels"])
            elif head == "pose":
                if "head_pose" in targets:
                    hp = targets["head_pose"]
                    hp_6d = hp[:, hp.size(1) // 2, :6]
                    loss = pose_loss(outputs["pose_6d"], hp_6d)
                else:
                    print("  No head_pose in targets — skipping pose")
                    return None
            else:
                raise ValueError(f"Unknown head: {head}")
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{n_steps}: {head} loss = {loss.item():.4f} (first={losses[0]:.4f})")

    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Loss decrease: {initial_loss - final_loss:.4f}")

    # Verdict
    if final_loss < 0.1 * initial_loss:
        verdict = "✅ PASS: head overfit 200 images successfully (loss → ~0). Architecture works."
    elif final_loss < 0.5 * initial_loss:
        verdict = "🟡 PARTIAL: head is learning but not fully overfit. May need more steps or better init."
    else:
        verdict = "❌ FAIL: head is NOT learning on 200 fixed images. Either eval-harness is broken or there's a deeper issue (data, loss, features)."

    print(f"\n  VERDICT: {verdict}\n")

    return {
        "head": head,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decrease_pct": (initial_loss - final_loss) / (initial_loss + 1e-6) * 100,
        "verdict": verdict,
        "n_steps": n_steps,
    }


# ===========================================================================
# Probe 2: ST-activity 5 epochs
# ===========================================================================

def probe2_st_activity(n_epochs: int = 5, max_batches: int = 200):
    """Single-task activity training for 5 epochs (capped at 200 batches/epoch).

    If top-1 ≥ 0.30 by ep5, head+backbone are adequate (do not touch activity).
    If <0.10, deeper issue.
    """
    print(f"\n{'='*60}")
    print(f"PROBE 2: ST-activity {n_epochs} epochs (cap {max_batches}/ep)")
    print(f"{'='*60}\n")

    # Build dataset
    train_ds = IndustRealMultiTaskDataset(
        split="train", img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    val_ds = IndustRealMultiTaskDataset(
        split="val", img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True,
        collate_fn=collate_fn_sequences, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        collate_fn=collate_fn_sequences, num_workers=0,
    )

    # Build model
    model = MTLMViTModel(num_act_classes=75).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train activity only
    for epoch in range(n_epochs):
        model.train()
        n_batches = 0
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            images = images.to(DEVICE).float() / 255.0
            mean = torch.tensor([0.45]*3, device=DEVICE).view(1, 1, 3, 1, 1, 1)
            std = torch.tensor([0.225]*3, device=DEVICE).view(1, 1, 3, 1, 1, 1)
            images = (images - mean) / std
            images = images.permute(0, 2, 1, 3, 4, 5).contiguous()
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                act_loss = activity_loss(outputs["activity"], targets["activity"])
            optimizer.zero_grad()
            act_loss.backward()
            optimizer.step()
            epoch_loss += act_loss.item()
            # Top-1
            with torch.no_grad():
                preds = outputs["activity"].argmax(dim=1)
                targets_act = targets["activity"]
                mask = targets_act >= 0
                correct += ((preds == targets_act) & mask).sum().item()
                total += mask.sum().item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        train_top1 = correct / max(total, 1)
        print(f"  Epoch {epoch+1}/{n_epochs}: act loss = {avg_loss:.4f}, train top-1 = {train_top1:.4f}")

    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 50:  # quick eval
                break
            images = images.to(DEVICE).float() / 255.0
            mean = torch.tensor([0.45]*3, device=DEVICE).view(1, 1, 3, 1, 1, 1)
            std = torch.tensor([0.225]*3, device=DEVICE).view(1, 1, 3, 1, 1, 1)
            images = (images - mean) / std
            images = images.permute(0, 2, 1, 3, 4, 5).contiguous()
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                preds = outputs["activity"].argmax(dim=1)
                targets_act = targets["activity"]
                mask = targets_act >= 0
                correct += ((preds == targets_act) & mask).sum().item()
                total += mask.sum().item()
    val_top1 = correct / max(total, 1)
    print(f"\n  Val top-1: {val_top1:.4f}")
    if val_top1 >= 0.30:
        verdict = "✅ PASS: ST-activity ≥0.30 by ep5. Head+backbone adequate. Do NOT touch activity head."
    elif val_top1 >= 0.10:
        verdict = "🟡 PARTIAL: head is learning but slow. Check label noise / data quality."
    else:
        verdict = "❌ FAIL: top-1 < 0.10. Deeper issue: data, label, or eval."
    print(f"  VERDICT: {verdict}\n")
    return {"epochs": n_epochs, "val_top1": val_top1, "verdict": verdict}


# ===========================================================================
# Probe 3: PSR P5 + temporal-res A/B
# ===========================================================================

def probe3_psr_ab(n_steps: int = 300):
    """PSR on P5: A/B interpolate (current) vs predict-at-T=8.

    A/B shows whether the T=16→8→16 interpolation (FC-4) is the bottleneck.
    """
    print(f"\n{'='*60}")
    print(f"PROBE 3: PSR on P5 — temporal-resolution A/B")
    print(f"{'='*60}\n")
    # This is a stub; the full implementation requires monkey-patching the
    # PSR head to disable the linear interpolation. See:
    # python -c "from src.models.mvit_mtl_model import PSRHead; import torch; \
    #            p = PSRHead(feat_dim=768); p.eval(); \
    #            x = torch.randn(2, 768, 8, 7, 7); print(p(x).shape)"
    # Current (interpolated): [2, 16, 11]. Without interp: [2, 8, 11].
    print("  Probe 3 implementation: requires editing PSRHead.forward to expose")
    print("  both interpolated (T=16) and native (T=8) outputs for comparison.")
    print("  See file 192 §6 Probe 3 for full protocol. Run this manually:")
    print()
    print("  1. Train PSR head on P5 for 300 steps")
    print("  2. Eval at T=16 (current, interpolated): expected F1 ≈ 0")
    print("  3. Eval at T=8 (predict-at-native): expected F1 ≫ 0 if FC-4 confirmed")
    print()
    return {"status": "manual-implementation-required"}


# ===========================================================================
# Probe 4: Detection TAL-lite vs 3x3
# ===========================================================================

def probe4_det_tal_vs_3x3():
    """On the overfit set, compare 3x3 (current) vs minimal TAL assigner.

    If 3x3 already overfits to mAP ≥ 0.6, assigner isn't the bottleneck.
    """
    print(f"\n{'='*60}")
    print(f"PROBE 4: Detection — TAL-lite vs 3x3 on overfit set")
    print(f"{'='*60}\n")
    print("  Probe 4 implementation: requires implementing TAL assigner")
    print("  (cite TOOD, ICCV 2021) and running on overfit-200 set.")
    print("  Skip this probe if Probe 1 already shows eval metric is moving")
    print("  on 3x3 — the assigner is then not the bottleneck.")
    print()
    return {"status": "conditional-on-probe-1"}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="MVP smoke suite (Opus 192 §6)")
    parser.add_argument("--probe", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Which probe to run (default: all)")
    parser.add_argument("--head", choices=["det", "act", "psr", "pose"], default="det",
                        help="Which head (for probes 1 and 2)")
    parser.add_argument("--n-steps", type=int, default=500,
                        help="Number of optimization steps for probe 1")
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="Number of epochs for probe 2")
    parser.add_argument("--output", type=str, default="/tmp/mvp_smoke_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()

    results = []
    if args.probe in ["1", "all"]:
        result = probe1_overfit_200(args.head, args.n_steps)
        if result:
            results.append(result)
    if args.probe in ["2", "all"]:
        result = probe2_st_activity(args.n_epochs)
        results.append(result)
    if args.probe in ["3", "all"]:
        result = probe3_psr_ab()
        results.append(result)
    if args.probe in ["4", "all"]:
        result = probe4_det_tal_vs_3x3()
        results.append(result)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
