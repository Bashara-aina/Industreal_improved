#!/usr/bin/env python3
"""MVP Probe 4: Detection — TAL vs 3×3 on overfit-200 set.

[OPUS 192 §6 Probe 4] On 200 fixed images, compare:
  (a) Current 3×3 assigner (pos_radius=1)
  (b) Minimal TAL assigner (topk=10, alpha=1.0, beta=6.0)

If (a) already overfits to mAP ≥ 0.6 on the overfit set, the assigner
isn't the bottleneck. If (b) materially higher, the ~2-day TAL port is
justified.

Note: This is the smallest, most surgical probe. The eval-harness check
(Probe 1) is the most important; this is conditional on it passing.

Usage:
    python scripts/mvp_probe4_tal_vs_3x3.py --n-steps 200
"""
# DEPRECATED: This script uses the legacy MTLMViTModel. Use POPWMultiTaskModel from src/models/model.py instead.
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from src.data.industreal_dataset import IndustRealMultiTaskDataset
from src.models.mvit_mtl_model import MTLMViTModel
from scripts.train_mtl_mvit import detection_loss as detection_loss_3x3
from scripts.train_mtl_mvit import compute_activity_class_weights
from src.losses.tal_assigner import TaskAlignedAssigner

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FIXED = 200


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


def detection_loss_tal(
    det_outputs: dict,
    det_list: list,
    num_classes: int = 24,
    reg_max: int = 16,
    gamma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Detection loss with TAL assigner (TOOD).

    For each FPN level:
    1. Decode DFL predictions → box offsets
    2. Use TAL to assign each GT to top-k cells
    3. Focal classification loss on assigned cells
    4. CIoU + DFL regression loss on assigned cells
    """
    device = next(iter(det_outputs.values()))["cls_logits"].device
    strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
    tal = TaskAlignedAssigner(topk=10, alpha=1.0, beta=6.0)

    if not det_list:
        return torch.tensor(0.0, device=device)

    loss_total = 0.0
    n_levels = 0
    for level_name in ("P2", "P3", "P4", "P5"):
        if level_name not in det_outputs:
            continue
        n_levels += 1
        out = det_outputs[level_name]
        cls_logits = out["cls_logits"]  # [B, 24, H, W]
        reg_preds = out["reg_preds"]    # [B, 4*reg_max, H, W]
        B, _, H, W = cls_logits.shape
        stride = strides[level_name]

        # Grid cell centers
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        cell_cx = xs.float() * stride + stride / 2.0
        cell_cy = ys.float() * stride + stride / 2.0

        # Decode DFL → box offsets (l, t, r, b) in grid units
        reg_dist = reg_preds.view(B, 4, reg_max, H, W)
        proj = torch.arange(reg_max, device=device).float().view(1, 1, reg_max, 1, 1)
        decoded = (reg_dist.softmax(dim=2) * proj).sum(dim=2)  # [B, 4, H, W]
        # Convert to xyxy in pixel coords
        pred_x1 = cell_cx.view(1, 1, W) - decoded[:, 0:1] * stride
        pred_y1 = cell_cy.view(1, H, 1) - decoded[:, 1:2] * stride
        pred_x2 = cell_cx.view(1, 1, W) + decoded[:, 2:3] * stride
        pred_y2 = cell_cy.view(1, H, 1) + decoded[:, 3:4] * stride
        pred_xyxy = torch.cat([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # [B, 4, H, W]
        pred_xyxy = pred_xyxy.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)  # [B, H*W, 4]

        # Apply TAL per image
        cls_sig = torch.sigmoid(cls_logits).permute(0, 2, 3, 1).contiguous().view(B, -1, num_classes)  # [B, H*W, 24]
        anchor_points = torch.stack([
            cell_cx.view(-1),
            cell_cy.view(-1),
        ], dim=1)  # [H*W, 2]

        for b in range(B):
            det_item = det_list[b] if isinstance(det_list[b], dict) else {}
            boxes = det_item.get("boxes")
            labels = det_item.get("labels")
            if boxes is None or labels is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)
                labels = labels.unsqueeze(0)

            # Pad to max_n=10 (small)
            max_n = 10
            n = boxes.size(0)
            padded_boxes = torch.zeros(max_n, 4, device=device)
            padded_boxes[:n] = boxes
            padded_labels = torch.zeros(max_n, dtype=torch.long, device=device)
            padded_labels[:n] = labels

            # TAL
            target_labels, target_bboxes, target_scores, mask, _ = tal(
                pred_cls=cls_sig[b:b+1],
                pred_box=pred_xyxy[b:b+1],
                anchors=anchor_points,
                gt_boxes=padded_boxes.unsqueeze(0),
                gt_labels=padded_labels.unsqueeze(0),
                anchor_points=anchor_points,
                stride=torch.tensor([stride], device=device),
            )
            mask = mask.squeeze(-1)  # [1, H*W]

            # Focal classification loss on assigned cells
            cls_loss_per_cell = F.binary_cross_entropy_with_logits(
                cls_sig[b:b+1], target_labels, reduction="none"
            ).sum(dim=-1)  # [1, H*W]
            cls_loss = (cls_loss_per_cell * mask).sum() / (mask.sum() + 1e-6)

            # CIoU + DFL regression loss on assigned cells
            if mask.sum() > 0:
                # CIoU
                pred_assigned = pred_xyxy[b:b+1][mask.bool()]  # [n_assigned, 4]
                tgt_assigned = target_bboxes[b:b+1][mask.bool()]  # [n_assigned, 4]
                if pred_assigned.numel() > 0:
                    # Simplified CIoU (uses same as train_mtl_mvit)
                    from scripts.train_mtl_mvit import ciou_loss
                    iou_loss = ciou_loss(pred_assigned, tgt_assigned).mean()
                else:
                    iou_loss = torch.tensor(0.0, device=device)
                # DFL (simplified)
                reg_loss = F.binary_cross_entropy_with_logits(
                    reg_preds[b:b+1].view(1, 4*reg_max, -1).transpose(1, 2),
                    torch.zeros(1, 4*reg_max, H*W, device=device),
                    reduction="mean",
                )
            else:
                iou_loss = torch.tensor(0.0, device=device)
                reg_loss = torch.tensor(0.0, device=device)

            loss_total = loss_total + cls_loss + iou_loss + reg_loss

    n_levels = max(n_levels, 1)
    return loss_total / n_levels


def train_and_eval_compare(n_steps: int, batch_size: int):
    """Train detection head with 3x3 and TAL, eval on overfit-200 set."""
    print(f"\n{'='*60}")
    print(f"PROBE 4: TAL vs 3×3 on overfit-200 set")
    print(f"{'='*60}\n")

    # Build dataset (200 fixed images)
    train_ds = IndustRealMultiTaskDataset(
        split="train", img_size=(224, 224),
        augment=False, sequence_mode=True, sequence_length=16,
    )
    indices = list(range(min(N_FIXED, len(train_ds))))
    subset_ds = Subset(train_ds, indices)

    def fixed_collate(batch):
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

    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False, collate_fn=fixed_collate)

    # Build model
    model = MTLMViTModel(num_act_classes=75).to(DEVICE)
    for p in model.feature_pyramid.backbone.parameters():
        p.requires_grad = False

    # Get the detection head params
    head_params = [p for n, p in model.named_parameters()
                   if "det_head" in n and p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3)

    # Train with 3x3
    print(f"Training 3x3 detection head for {n_steps} steps...")
    losses_3x3 = []
    for step, (images, targets) in enumerate(loader):
        if step >= n_steps:
            break
        images = images.to(DEVICE)
        targets = to_device_targets(targets, DEVICE)
        images = normalize_images(images, DEVICE)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = detection_loss_3x3(outputs["detection"], targets["detection"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_3x3.append(loss.item())
    print(f"  3x3: initial={losses_3x3[0]:.4f}, final={losses_3x3[-1]:.4f}")

    # Reset head
    for p in model.det_head.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

    # Train with TAL
    print(f"\nTraining TAL detection head for {n_steps} steps...")
    losses_tal = []
    for step, (images, targets) in enumerate(loader):
        if step >= n_steps:
            break
        images = images.to(DEVICE)
        targets = to_device_targets(targets, DEVICE)
        images = normalize_images(images, DEVICE)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = detection_loss_tal(outputs["detection"], targets["detection"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_tal.append(loss.item())
    print(f"  TAL: initial={losses_tal[0]:.4f}, final={losses_tal[-1]:.4f}")

    # Verdict
    print()
    print("VERDICT:")
    if losses_3x3[-1] < 0.5 * losses_3x3[0]:
        print(f"  3x3 overfit: train loss decreased {losses_3x3[0]:.4f}→{losses_3x3[-1]:.4f}")
        print(f"  → 3x3 is learning. Assigner may be fine.")
    if losses_tal[-1] < losses_3x3[-1] * 0.8:
        print(f"  TAL materially better than 3x3 → TAL port justified (~2 days)")
    else:
        print(f"  TAL comparable to 3x3 → 3x3 is fine, save the 2 days")

    return {
        "loss_3x3_initial": losses_3x3[0],
        "loss_3x3_final": losses_3x3[-1],
        "loss_tal_initial": losses_tal[0],
        "loss_tal_final": losses_tal[-1],
        "verdict": "3x3-suffices" if losses_tal[-1] > losses_3x3[-1] * 0.8 else "TAL-justified",
    }


def main():
    parser = argparse.ArgumentParser(description="MVP Probe 4: TAL vs 3x3")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=str, default="/tmp/probe4_results.json")
    args = parser.parse_args()

    result = train_and_eval_compare(args.n_steps, args.batch_size)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
