#!/usr/bin/env python3
"""Overfit probe — the single most important diagnostic (Opus 201 Step 1).

For each head: freeze backbone, overfit 50-200 fixed clips to ~0 train loss,
then run the real eval on those same clips. If a head drives train loss → 0
but eval metric stays ~0 → the eval/target-encoding is broken, and no amount
of architecture fixes it. This probe is worth more than all of run11.

Usage:
    python scripts/overfit_probe.py --head det --n-clips 50 --steps 500
    python scripts/overfit_probe.py --head act --n-clips 50 --steps 200
    python scripts/overfit_probe.py --head psr --n-clips 50 --steps 200
    python scripts/overfit_probe.py --head pose --n-clips 50 --steps 100

[OPUS 201] This script was rewritten from the old ConvNeXt-based
overfit_50img_cls.py to use the current MViTv2-S MTL architecture.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.NUM_ACT_OUTPUTS = 75
C.ACT_CLASS_GROUPING = "none"

from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
from src.models.mvit_mtl_model import MTLMViTModel, renormalize_pose
from scripts.train_mtl_mvit import detection_loss, activity_loss, psr_loss, pose_loss


def parse_args():
    p = argparse.ArgumentParser(description="Overfit probe — verify eval harness per head")
    p.add_argument("--head", choices=["det", "act", "psr", "pose"], required=True)
    p.add_argument("--n-clips", type=int, default=50, help="Number of clips to overfit")
    p.add_argument("--steps", type=int, default=500, help="Training steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results JSON")
    return p.parse_args()


def normalize_clip(images, device):
    """[B, T, 3, H, W] in [0,255] → [B, 3, T, H, W] normalized."""
    images = images.float().to(device) / 255.0
    mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
    images = (images - mean) / std
    return images.permute(0, 2, 1, 3, 4).contiguous()


def to_device(targets, device):
    for k, v in list(targets.items()):
        if isinstance(v, torch.Tensor):
            targets[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for sk, sv in item.items():
                        if isinstance(sv, torch.Tensor):
                            item[sk] = sv.to(device, non_blocking=True)
    return targets


def run_overfit(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"{'='*60}")
    print(f"OVERFIT PROBE: --head {args.head} (Opus 201 Step 1)")
    print(f"{'='*60}")
    print(f"  device={device}  lr={args.lr}  steps={args.steps}  n_clips={args.n_clips}")
    print()

    # ── Build dataset (full, then subset) ──────────────────────────────
    print("[1] Loading dataset...")
    ds = IndustRealMultiTaskDataset(
        split="train", img_size=(224, 224), augment=False,
        sequence_mode=True, sequence_length=16,
    )
    # Find clips with valid labels for the target head
    # Dataset items are dicts with keys: images, gt_boxes, gt_classes, head_pose,
    # psr_labels, hand_joints, action_label, clip_rgb, metadata
    valid_indices = []
    for idx in range(len(ds)):
        try:
            sample = ds[idx]  # dict
        except Exception:
            continue
        if args.head == "det":
            gt_boxes = sample.get("gt_boxes", {}).get("rgb")
            if gt_boxes is not None and len(gt_boxes) > 0:
                valid_indices.append(idx)
        elif args.head == "act":
            act = sample.get("action_label")
            if act is not None and act.item() >= 0:
                valid_indices.append(idx)
        elif args.head == "psr":
            psr = sample.get("psr_labels")
            if psr is not None and psr.sum() > 0:  # at least one transition
                valid_indices.append(idx)
        elif args.head == "pose":
            hp = sample.get("head_pose")
            if hp is not None and hp.shape[0] > 0:
                valid_indices.append(idx)
        if len(valid_indices) >= args.n_clips:
            break

    if len(valid_indices) < args.n_clips:
        print(f"  WARNING: Only found {len(valid_indices)} valid clips, using all")
    used = valid_indices[:args.n_clips]
    print(f"  Using {len(used)} clips for {args.head} overfit")

    # Subset via list of samples
    subset_samples = [ds[i] for i in used]

    # ── Build model ────────────────────────────────────────────────────
    print("\n[2] Building model (freeze backbone)...")
    model = MTLMViTModel(num_act_classes=75).to(device)
    # Freeze backbone + FPN — only train the target head
    for name, param in model.named_parameters():
        if "feature_pyramid" in name or "fpn" in name:
            param.requires_grad = False
        elif args.head == "det" and "det_head" not in name:
            param.requires_grad = False
        elif args.head == "act" and "act_head" not in name:
            param.requires_grad = False
        elif args.head == "psr" and "psr_head" not in name:
            param.requires_grad = False
        elif args.head == "pose" and "pose_head" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n[3] Overfitting for {args.steps} steps...")
    print(f"  {'Step':>6} | {'Loss':>10} | {'Time':>6}")
    print(f"  {'-'*30}")

    history = {"loss": [], "step": []}
    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        # Cycle through subset samples (each is a dict)
        sample_idx = (step - 1) % len(subset_samples)
        sample = subset_samples[sample_idx]  # dict

        # Extract images [T, 3, H, W] and add batch dim
        images = sample["images"]["rgb"].clone()
        if images.dim() == 3:
            images = images.unsqueeze(0)  # [1, T, 3, H, W]
        elif images.dim() == 4:
            images = images.unsqueeze(0)
        images = normalize_clip(images, device)

        # Move targets to device
        targets = {}
        for k in ["head_pose", "psr_labels", "action_label"]:
            if k in sample:
                v = sample[k]
                if hasattr(v, 'clone'):
                    targets[k] = v.clone().to(device)
                else:
                    targets[k] = v
        # Detection targets
        gt_boxes = sample.get("gt_boxes", {}).get("rgb")
        gt_classes = sample.get("gt_classes", {}).get("rgb")
        if gt_boxes is not None and isinstance(gt_boxes, torch.Tensor):
            targets["det_boxes"] = gt_boxes.clone().to(device)
            targets["det_labels"] = gt_classes.clone().to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if args.head == "det":
            if "det_boxes" in targets and targets["det_boxes"].numel() > 0:
                det_list = [{"boxes": targets["det_boxes"], "labels": targets["det_labels"]}]
                loss = detection_loss(outputs["detection"], det_list)
            else:
                continue
        elif args.head == "act":
            act_target = targets.get("action_label")
            if act_target is None or act_target.numel() == 0:
                continue
            if act_target.dim() == 0:
                act_target = act_target.unsqueeze(0)
            if act_target.item() < 0:
                continue
            loss = activity_loss(outputs["activity"], act_target)
        elif args.head == "psr":
            psr_labels = targets.get("psr_labels")
            if psr_labels is None:
                psr_labels = torch.zeros(images.size(0), 16, 11, device=device)
            loss = psr_loss(outputs["psr_logits"], psr_labels, use_focal=True)
        elif args.head == "pose":
            hp = targets.get("head_pose")
            if hp is None:
                continue
            hp_6d = hp[:, hp.size(1) // 2, :6]
            loss = pose_loss(outputs["pose_6d"], hp_6d)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history["loss"].append(loss.item())
        history["step"].append(step)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            dt = time.time() - t0
            print(f"  {step:>6d} | {loss.item():>10.6f} | {dt:>5.1f}s")

        # Early success: loss very low
        if loss.item() < 0.01 and step > 50:
            print(f"\n  ✅ Loss < 0.01 at step {step} — head CAN overfit.")
            break

    # ── Eval on the subset ────────────────────────────────────────────
    print(f"\n[4] Running eval on the overfit subset ({len(used)} clips)...")
    model.eval()

    head_metrics = {}
    with torch.no_grad():
        for sample in subset_samples:
            images = sample["images"]["rgb"].clone()
            if images.dim() == 3:
                images = images.unsqueeze(0)
            elif images.dim() == 4:
                images = images.unsqueeze(0)
            images = normalize_clip(images, device)
            image_batch = images  # already [1, 3, T, H, W]
            outputs = model(image_batch)

            if args.head == "act":
                act = sample.get("action_label")
                if act is not None and act.numel() > 0:
                    if act.dim() == 0:
                        act = act.unsqueeze(0)
                    if act.item() >= 0:
                        act = act.to(device)
                        pred = outputs["activity"].argmax(dim=1).item()
                        head_metrics.setdefault("correct", 0)
                        head_metrics.setdefault("total", 0)
                        head_metrics["correct"] += int(pred == act.item())
                        head_metrics["total"] += 1
            elif args.head == "psr":
                # Check if predictions are non-trivial
                psr_pred = torch.sigmoid(outputs["psr_logits"])
                head_metrics.setdefault("mean_prob", [])
                head_metrics["mean_prob"].append(psr_pred.mean().item())
                head_metrics.setdefault("frac_positive", [])
                head_metrics["frac_positive"].append((psr_pred > 0.5).float().mean().item())
            elif args.head == "pose":
                hp = sample.get("head_pose")
                if hp is not None:
                    hp = hp.to(device).unsqueeze(0)
                    fwd_p, up_p = renormalize_pose(outputs["pose_6d"])
                    fwd_g = F.normalize(hp[:, hp.size(1)//2, :3], dim=1)
                    cos_f = (fwd_p * fwd_g).sum(dim=1).clamp(-1, 1)
                    head_metrics.setdefault("fwd_mae", [])
                    head_metrics["fwd_mae"].append(
                        torch.rad2deg(torch.acos(cos_f)).item()
                    )

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS for --head {args.head}")
    print(f"{'='*60}")
    final_loss = history["loss"][-1]
    print(f"  Final train loss: {final_loss:.6f}")

    if args.head == "act":
        acc = head_metrics.get("correct", 0) / max(head_metrics.get("total", 1), 1)
        print(f"  Eval top-1 on overfit: {acc:.4f}")
        if final_loss < 0.1 and acc > 0.8:
            verdict = "PASS — Activity head CAN overfit + eval works."
        elif final_loss < 0.1:
            verdict = "EVAL BUG — loss→0 but eval metric stays low. Check label space (75↔69 remap)."
        else:
            verdict = "FAIL — Activity head cannot overfit. Architecture or label-noise problem."
    elif args.head == "psr":
        mean_prob = np.mean(head_metrics.get("mean_prob", [0.5]))
        frac_pos = np.mean(head_metrics.get("frac_positive", [0]))
        print(f"  Mean sigmoid prob: {mean_prob:.4f}")
        print(f"  Fraction >0.5: {frac_pos:.4f}")
        if final_loss < 0.02 and (frac_pos > 0.01 or mean_prob > 0.6):
            verdict = "PASS — PSR head CAN overfit + produces non-trivial predictions."
        elif final_loss < 0.02:
            verdict = "FOCAL COLLAPSE — loss→0 but all predictions are negative. Focal-BCE on rare positives collapses. Adjust α/threshold."
        else:
            verdict = "FAIL — PSR head cannot overfit."
    elif args.head == "pose":
        fwd_mae = np.mean(head_metrics.get("fwd_mae", [90]))
        print(f"  Forward MAE on overfit: {fwd_mae:.1f}°")
        if final_loss < 0.05 and fwd_mae < 15:
            verdict = "PASS — Pose head CAN overfit."
        else:
            verdict = f"FAIL — Pose head cannot overfit (loss={final_loss:.4f}, MAE={fwd_mae:.1f}°)."
    elif args.head == "det":
        # For detection, the quick eval is harder — just check loss
        if final_loss < 0.05:
            verdict = "PASS — Detection head CAN overfit. If eval still shows 0.0 mAP, eval harness is broken."
        else:
            verdict = f"FAIL — Detection head cannot overfit (loss={final_loss:.4f})."

    print(f"\n  VERDICT: {verdict}")

    # Save results
    out = {
        "head": args.head,
        "n_clips": len(used),
        "steps": len(history["loss"]),
        "final_loss": final_loss,
        "verdict": verdict,
        "head_metrics": {k: (v if isinstance(v, (int, float)) else float(np.mean(v)))
                         for k, v in head_metrics.items()},
    }
    out_path = args.output or str(_CODE_ROOT / "src" / "runs" / f"overfit_{args.head}_results.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to: {out_path}")
    return "PASS" in verdict


if __name__ == "__main__":
    args = parse_args()
    success = run_overfit(args)
    sys.exit(0 if success else 1)
