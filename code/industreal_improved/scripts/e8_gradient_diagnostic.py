#!/usr/bin/env python3
"""E8: Gradient-flow diagnostic for multi-task learning.

[OPUS 181 §6 K-5/P-3/P-4, 186 §4 D-8] The single most useful diagnostic for
MTL is the per-task gradient cosine-similarity heatmap. It tells you:
- Which task pairs ALIGN (positive transfer candidates)
- Which task pairs CONFLICT (cosine < 0, negative transfer)
- Whether PCGrad is doing useful work (conflict rate > 0)
- The relative gradient magnitudes per task

Output: JSON with per-task gradient norms and pairwise cosine similarities
+ a text heatmap you can read in the log.

Usage:
    python scripts/e8_gradient_diagnostic.py --max-batches 200 \
        --output /tmp/e8_results.json
"""

# DEPRECATED: This script uses the legacy MTLMViTModel. Use POPWMultiTaskModel from src/models/model.py instead.
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
    mean = torch.tensor([0.45] * 3, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.225] * 3, device=device).view(1, 1, 3, 1, 1)
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


def main():
    parser = argparse.ArgumentParser(description="E8 gradient-flow diagnostic for MTL")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Number of batches to accumulate gradients over",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=str, default="/tmp/e8_results.json")
    args = parser.parse_args()

    # Build model and dataset
    model = MTLMViTModel(num_act_classes=75).to(DEVICE)
    model.eval()  # disable dropout for cleaner gradients

    train_ds = IndustRealMultiTaskDataset(
        split="train",
        img_size=(224, 224),
        augment=False,
        sequence_mode=True,
        sequence_length=16,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_sequences,
        num_workers=0,
    )

    shared_params = [p for p in model.feature_pyramid.backbone.parameters() if p.requires_grad]

    # Collect per-task gradient norms and (flattened) gradient vectors
    grad_norms = {"det": [], "act": [], "psr": [], "pose": []}
    grad_vecs = {"det": [], "act": [], "psr": [], "pose": []}

    print(f"Collecting per-task gradients over {args.max_batches} batches...")
    t0 = time.time()
    n_processed = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx >= args.max_batches:
            break
        images = images.to(DEVICE)
        targets = to_device_targets(targets, DEVICE)
        images = normalize_images(images, DEVICE)

        # Per-task loss and per-task gradient on shared backbone
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            task_losses = {
                "det": detection_loss(outputs["detection"], targets.get("detection", [])),
                "act": activity_loss(outputs["activity"], targets["activity"]),
                "psr": psr_loss(outputs["psr_logits"], targets["psr_labels"]),
                "pose": pose_loss(
                    outputs["pose_6d"],
                    targets["head_pose"][:, targets["head_pose"].size(1) // 2, :6],
                ),
            }

        for task_name, loss in task_losses.items():
            try:
                grads = torch.autograd.grad(
                    loss, shared_params, retain_graph=True, allow_unused=True
                )
                # Filter to non-None gradients
                flat = []
                for g, p in zip(grads, shared_params):
                    if g is not None:
                        flat.append(g.contiguous().view(-1))
                    else:
                        flat.append(torch.zeros(p.numel(), device=DEVICE))
                flat = torch.cat(flat)
                norm = flat.norm().item()
                grad_norms[task_name].append(norm)
                grad_vecs[task_name].append(flat.detach().cpu().numpy())
            except Exception as e:
                print(f"  Warning: grad failed for {task_name} on batch {batch_idx}: {e}")
        n_processed += 1

    print(f"  Processed {n_processed} batches in {time.time() - t0:.1f}s")

    # Compute pairwise cosine similarities
    task_names = ["det", "act", "psr", "pose"]
    cosine_matrix = np.zeros((4, 4))
    for i, ti in enumerate(task_names):
        for j, tj in enumerate(task_names):
            if i == j:
                cosine_matrix[i, j] = 1.0
            else:
                # Average cosine over all batches where both tasks produced valid gradients
                cos_vals = []
                for b in range(min(len(grad_vecs[ti]), len(grad_vecs[tj]))):
                    gi = grad_vecs[ti][b]
                    gj = grad_vecs[tj][b]
                    norm_i = np.linalg.norm(gi)
                    norm_j = np.linalg.norm(gj)
                    if norm_i > 1e-9 and norm_j > 1e-9:
                        cos_vals.append(np.dot(gi, gj) / (norm_i * norm_j))
                cosine_matrix[i, j] = float(np.mean(cos_vals)) if cos_vals else 0.0

    # Conflict rate: % of batch pairs with cosine < 0
    conflict_rate = {}
    for i, ti in enumerate(task_names):
        for j, tj in enumerate(task_names):
            if i >= j:
                continue
            n_conflict = 0
            n_total = 0
            for b in range(min(len(grad_vecs[ti]), len(grad_vecs[tj]))):
                gi = grad_vecs[ti][b]
                gj = grad_vecs[tj][b]
                ni = np.linalg.norm(gi)
                nj = np.linalg.norm(gj)
                if ni > 1e-9 and nj > 1e-9:
                    cos = np.dot(gi, gj) / (ni * nj)
                    n_total += 1
                    if cos < 0:
                        n_conflict += 1
            conflict_rate[f"{ti}-{tj}"] = {
                "n_conflict": n_conflict,
                "n_total": n_total,
                "rate": n_conflict / max(n_total, 1),
            }

    # Mean gradient norm per task
    mean_grad_norms = {
        t: float(np.mean(grad_norms[t])) if grad_norms[t] else 0.0 for t in task_names
    }

    # Print heatmap
    print("\n" + "=" * 60)
    print("GRADIENT COSINE SIMILARITY HEATMAP (per-batch, averaged)")
    print("=" * 60)
    print(f"  {'':>6}  {'  det':>8}  {'   act':>8}  {'   psr':>8}  {'  pose':>8}")
    for i, ti in enumerate(task_names):
        row = f"  {ti:>6}  "
        for j, tj in enumerate(task_names):
            v = cosine_matrix[i, j]
            bar = "+" if v > 0.1 else (" " if v > -0.1 else "-")
            row += f"{v:>+7.3f}{bar}  "
        print(row)

    print("\nMEAN GRADIENT NORM (backbone):")
    for t in task_names:
        print(f"  {t}: {mean_grad_norms[t]:.4f}")

    print("\nCONFLICT RATE (% of batch pairs with cosine < 0):")
    for pair, info in conflict_rate.items():
        bar = "⚠️  HIGH" if info["rate"] > 0.3 else "OK"
        print(f"  {pair}: {info['rate']:.1%} ({info['n_conflict']}/{info['n_total']}) {bar}")

    # Verdicts
    print("\nVERDICTS:")
    avg_conflict = np.mean([info["rate"] for info in conflict_rate.values()])
    if avg_conflict < 0.05:
        print(
            f"  Conflict rate {avg_conflict:.1%} very low → PCGrad mostly no-op. Consider disabling for ~30% speedup."
        )
    elif avg_conflict > 0.30:
        print(
            f"  Conflict rate {avg_conflict:.1%} high → PCGrad is essential; document this in the paper."
        )
    else:
        print(
            f"  Conflict rate {avg_conflict:.1%} moderate → PCGrad is doing useful work; keep it on."
        )

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
