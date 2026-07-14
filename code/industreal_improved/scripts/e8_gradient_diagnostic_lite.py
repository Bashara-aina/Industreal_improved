#!/usr/bin/env python3
"""E8 gradient-flow diagnostic — memory-efficient version.

[OPUS 181 §6, 186 §4 D-8] Computes per-task gradient norms on the shared
backbone and pairwise cosine similarity. This is the memory-efficient
version: uses bf16, small batch, and processes each task's gradient
one at a time (frees autograd graph between tasks).

Run on GPU 0 (training uses GPU 1). Total ~5-10 minutes for 100 batches.

Usage:
    python scripts/e8_gradient_diagnostic_lite.py --max-batches 100
"""
# DEPRECATED: This script uses the legacy MTLMViTModel. Use POPWMultiTaskModel from src/models/model.py instead.
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
from scripts.train_mtl_mvit import detection_loss, activity_loss, psr_loss, pose_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_images(images, device):
    images = images.float() / 255.0
    mean = torch.tensor([0.45]*3, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.225]*3, device=device).view(1, 1, 3, 1, 1)
    images = (images - mean) / std
    images = images.permute(0, 2, 1, 3, 4).contiguous()
    return images


def to_device_targets(targets, device):
    for k, v in list(targets.items()):
        if isinstance(v, torch.Tensor):
            targets[k] = v.to(device)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    for sk, sv in item.items():
                        if isinstance(sv, torch.Tensor):
                            item[sk] = sv.to(device)
    return targets


def get_task_grad_norm(model, images, targets, task_name, shared_params):
    """Compute the per-task gradient norm for one task. Frees the graph after."""
    model.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(images)
        if task_name == "det":
            loss = detection_loss(outputs["detection"], targets.get("detection", []))
        elif task_name == "act":
            loss = activity_loss(outputs["activity"], targets["activity"])
        elif task_name == "psr":
            loss = psr_loss(outputs["psr_logits"], targets["psr_labels"])
        elif task_name == "pose":
            hp_6d = targets["head_pose"][:, targets["head_pose"].size(1) // 2, :6]
            loss = pose_loss(outputs["pose_6d"], hp_6d)
    loss.backward()
    # Flatten the per-task gradient on the shared backbone
    flat = []
    for p in shared_params:
        if p.grad is not None:
            flat.append(p.grad.contiguous().view(-1).detach().cpu().float().numpy())
        else:
            flat.append(np.zeros(p.numel(), dtype=np.float32))
    grad_vec = np.concatenate(flat)
    norm = float(np.linalg.norm(grad_vec))
    # Free the autograd graph
    del outputs, loss
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return norm, grad_vec


def main():
    parser = argparse.ArgumentParser(description="E8 gradient-flow diagnostic (lite)")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="/tmp/e8_lite_results.json")
    args = parser.parse_args()

    # Build model in bf16 to save memory
    model = MTLMViTModel(num_act_classes=75).to(DEVICE)
    model.eval()  # disable dropout

    # Build dataset
    train_ds = IndustRealMultiTaskDataset(
        split="train", img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_sequences, num_workers=0,
    )

    shared_params = [p for p in model.feature_pyramid.backbone.parameters() if p.requires_grad]

    # Collect per-task gradient norms and vectors
    grad_norms = {t: [] for t in ("det", "act", "psr", "pose")}
    grad_vecs = {t: [] for t in ("det", "act", "psr", "pose")}

    print(f"Collecting per-task gradients over {args.max_batches} batches (lite)...")
    t0 = time.time()
    n_processed = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx >= args.max_batches:
            break
        images = images.to(DEVICE)
        targets = to_device_targets(targets, DEVICE)
        images = normalize_images(images, DEVICE)

        # Process each task separately (frees autograd graph between tasks)
        for task_name in ("det", "act", "psr", "pose"):
            try:
                norm, vec = get_task_grad_norm(model, images, targets, task_name, shared_params)
                grad_norms[task_name].append(norm)
                grad_vecs[task_name].append(vec)
            except Exception as e:
                print(f"  Warning: grad failed for {task_name} on batch {batch_idx}: {e}")
        n_processed += 1

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Batch {batch_idx+1}/{args.max_batches}: "
                  f"det_norm={grad_norms['det'][-1]:.2e}, "
                  f"act_norm={grad_norms['act'][-1]:.2e}, "
                  f"psr_norm={grad_norms['psr'][-1]:.2e}, "
                  f"pose_norm={grad_norms['pose'][-1]:.2e} "
                  f"({elapsed:.0f}s)")

    print(f"  Processed {n_processed} batches in {time.time() - t0:.1f}s")

    # Compute pairwise cosine similarities
    task_names = ("det", "act", "psr", "pose")
    cosine_matrix = np.zeros((4, 4))
    for i, ti in enumerate(task_names):
        for j, tj in enumerate(task_names):
            if i == j:
                cosine_matrix[i, j] = 1.0
            else:
                cos_vals = []
                for b in range(min(len(grad_vecs[ti]), len(grad_vecs[tj]))):
                    gi = grad_vecs[ti][b]
                    gj = grad_vecs[tj][b]
                    ni = np.linalg.norm(gi)
                    nj = np.linalg.norm(gj)
                    if ni > 1e-9 and nj > 1e-9:
                        cos_vals.append(float(np.dot(gi, gj) / (ni * nj)))
                cosine_matrix[i, j] = float(np.mean(cos_vals)) if cos_vals else 0.0

    # Conflict rate
    conflict_rate = {}
    for i, ti in enumerate(task_names):
        for j, tj in enumerate(task_names):
            if i >= j:
                continue
            n_conflict = n_total = 0
            for b in range(min(len(grad_vecs[ti]), len(grad_vecs[tj]))):
                gi = grad_vecs[ti][b]
                gj = grad_vecs[tj][b]
                ni = np.linalg.norm(gi)
                nj = np.linalg.norm(gj)
                if ni > 1e-9 and nj > 1e-9:
                    cos = float(np.dot(gi, gj) / (ni * nj))
                    n_total += 1
                    if cos < 0:
                        n_conflict += 1
            conflict_rate[f"{ti}-{tj}"] = {
                "n_conflict": n_conflict,
                "n_total": n_total,
                "rate": n_conflict / max(n_total, 1),
            }

    mean_grad_norms = {t: float(np.mean(grad_norms[t])) if grad_norms[t] else 0.0
                       for t in task_names}

    # Print heatmap
    print("\n" + "=" * 60)
    print("GRADIENT COSINE SIMILARITY HEATMAP")
    print("=" * 60)
    print(f"  {'':>6}  {'  det':>8}  {'   act':>8}  {'   psr':>8}  {'  pose':>8}")
    for i, ti in enumerate(task_names):
        row = f"  {ti:>6}  "
        for j, tj in enumerate(task_names):
            v = cosine_matrix[i, j]
            bar = "+" if v > 0.1 else (" " if v > -0.1 else "-")
            row += f"{v:>+7.3f}{bar}  "
        print(row)

    print("\nMEAN GRADIENT NORM:")
    for t in task_names:
        print(f"  {t}: {mean_grad_norms[t]:.4f}")

    print("\nCONFLICT RATE (% batch pairs with cosine < 0):")
    avg_conflict = np.mean([info["rate"] for info in conflict_rate.values()])
    for pair, info in conflict_rate.items():
        bar = "HIGH" if info["rate"] > 0.3 else "ok"
        print(f"  {pair}: {info['rate']:.1%} ({info['n_conflict']}/{info['n_total']}) [{bar}]")

    print(f"\nAVG CONFLICT RATE: {avg_conflict:.1%}")
    if avg_conflict < 0.05:
        print("  → PCGrad mostly no-op. Consider disabling for ~30% speedup.")
    elif avg_conflict > 0.30:
        print("  → PCGrad is essential; document this in the paper.")
    else:
        print("  → PCGrad is doing useful work; keep it on.")

    # Save
    results = {
        "n_batches": n_processed,
        "mean_grad_norms": mean_grad_norms,
        "cosine_matrix": {
            f"{ti}-{tj}": float(cosine_matrix[i, j])
            for i, ti in enumerate(task_names)
            for j, tj in enumerate(task_names)
        },
        "conflict_rate": conflict_rate,
        "avg_conflict_rate": float(avg_conflict),
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
