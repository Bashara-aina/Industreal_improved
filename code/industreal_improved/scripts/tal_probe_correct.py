#!/usr/bin/env python3
"""Corrected TAL vs 3×3 probe — uses FullMultiModalDataset (matches training).

The original tal_probe_fixed.py used IndustRealMultiTaskDataset which
returns boxes in absolute pixel xyxy format, while FullMultiModalDataset
(used by training) returns boxes in normalized (cx, cy, w, h).

Without the correct dataset, the 3x3-suffices verdict from the prior
probe is meaningless.

Usage:
    python scripts/tal_probe_correct.py --n-steps 200
"""
import argparse, json, logging, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

from train_mtl_full_multimodal import (
    FullMultiModalDataset,
    expand_conv_proj_to_9ch,
    WrappedMTL,
    ensure_5d,
    collate_real_targets,
)
from src.models.mvit_mtl_model import MTLMViTModel
import train_mtl_v3 as mtl_v3_mod

# Match training config: v3.7 used 16 anchors
mtl_v3_mod.NUM_ANCHORS = 16
mtl_v3_mod._ANCHOR_SPECS = mtl_v3_mod._ANCHOR_SPECS_16
from train_mtl_v3 import (
    detection_loss,
    generate_anchors,
    NUM_DET_CLASSES,
    NUM_ANCHORS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FIXED = 200

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("tal_probe_correct")


def build_model():
    base = MTLMViTModel(
        num_act_classes=75, num_det_classes=24, num_psr_components=11, num_anchors=16,
    )
    expand_conv_proj_to_9ch(base)
    base = base.to(DEVICE)
    return WrappedMTL(base)


def freeze_backbone_keep_det(model):
    for name, p in model.named_parameters():
        if 'det_head' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return [p for p in model.m.det_head.parameters() if p.requires_grad]


def reset_det_head(model):
    """Reset detection head to Xavier + prior_prob-derived bias."""
    import math
    for m in model.m.det_head.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                if m.out_channels == 24:
                    nn.init.constant_(m.bias, -math.log(0.99 / 0.01))
                else:
                    nn.init.zeros_(m.bias)
    model.m.det_head.running_pos_ratio.fill_(0.01)


def train_compare(n_steps: int, batch_size: int = 2) -> dict:
    logger.info(f"{'='*60}")
    logger.info(f"TAL vs 3×3 on overfit-{N_FIXED} set ({n_steps} steps) [CORRECT dataset]")
    logger.info(f"{'='*60}")

    # Use the correct dataset (matches training)
    train_ds = FullMultiModalDataset(
        recordings_dir="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train",
        img_size=(640, 360),
        mosaic_prob=0.0,  # disable aug for overfitting
        copy_paste_prob=0.0,
    )
    indices = list(range(min(N_FIXED, len(train_ds))))
    subset_ds = Subset(train_ds, indices)
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_real_targets, num_workers=0)

    model = build_model()
    head_params = freeze_backbone_keep_det(model)

    # ── Train with 3×3 assigner (pos_radius=1) ──
    logger.info(f"\nTraining 3×3 assigner for {n_steps} steps...")
    reset_det_head(model)
    opt = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)
    losses_3x3 = []
    t0 = time.time()
    step = 0
    while step < n_steps:
        for images, targets in loader:
            if step >= n_steps:
                break
            images = ensure_5d(images).float().to(DEVICE)
            mean = torch.tensor([0.45] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            std = torch.tensor([0.225] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            images = (images - mean) / std
            gt_boxes = [b.to(DEVICE).float() for b in targets['boxes']]
            gt_classes = [c.to(DEVICE).long() for c in targets['classes']]

            opt.zero_grad()
            out = model(images)
            anchors = {lvl: generate_anchors(out['detection'][lvl]['cls_logits'].shape[2],
                                              out['detection'][lvl]['cls_logits'].shape[3],
                                              DEVICE)
                       for lvl in ['P3', 'P4', 'P5'] if lvl in out['detection']}
            loss, cls_loss, reg_loss = detection_loss(
                out['detection'], anchors, gt_boxes, gt_classes,
                use_tal=False,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 10.0)
            opt.step()
            losses_3x3.append(loss.item())
            if step % 50 == 0:
                logger.info(f"  3x3 step {step}/{n_steps}: loss={loss.item():.4f}")
            step += 1
    dt_3x3 = time.time() - t0
    logger.info(f"  3x3: {losses_3x3[0]:.4f} -> {losses_3x3[-1]:.4f}, time={dt_3x3:.1f}s")

    # ── Train with TAL ──
    logger.info(f"\nTraining TAL assigner for {n_steps} steps...")
    reset_det_head(model)
    opt = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)
    losses_tal = []
    t0 = time.time()
    step = 0
    while step < n_steps:
        for images, targets in loader:
            if step >= n_steps:
                break
            images = ensure_5d(images).float().to(DEVICE)
            mean = torch.tensor([0.45] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            std = torch.tensor([0.225] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            images = (images - mean) / std
            gt_boxes = [b.to(DEVICE).float() for b in targets['boxes']]
            gt_classes = [c.to(DEVICE).long() for c in targets['classes']]

            opt.zero_grad()
            out = model(images)
            anchors = {lvl: generate_anchors(out['detection'][lvl]['cls_logits'].shape[2],
                                              out['detection'][lvl]['cls_logits'].shape[3],
                                              DEVICE)
                       for lvl in ['P3', 'P4', 'P5'] if lvl in out['detection']}
            loss, cls_loss, reg_loss = detection_loss(
                out['detection'], anchors, gt_boxes, gt_classes,
                use_tal=True,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 10.0)
            opt.step()
            losses_tal.append(loss.item())
            if step % 50 == 0:
                logger.info(f"  TAL step {step}/{n_steps}: loss={loss.item():.4f}")
            step += 1
    dt_tal = time.time() - t0
    logger.info(f"  TAL: {losses_tal[0]:.4f} -> {losses_tal[-1]:.4f}, time={dt_tal:.1f}s")

    # Verdict
    final_3x3 = losses_3x3[-1]
    final_tal = losses_tal[-1]
    init_3x3 = losses_3x3[0]
    init_tal = losses_tal[0]
    logger.info(f"\n{'='*60}")
    logger.info("VERDICT:")
    logger.info(f"  3x3:  {init_3x3:.4f} -> {final_3x3:.4f}")
    logger.info(f"  TAL:  {init_tal:.4f} -> {final_tal:.4f}")

    verdict = "inconclusive"
    if final_3x3 < init_3x3 * 0.5:
        logger.info(f"  ✓ 3x3 overfits: loss dropped {init_3x3:.4f}->{final_3x3:.4f}")
        verdict = "3x3-learning"
    if final_tal < final_3x3 * 0.8:
        logger.info(f"  ✓ TAL materially better: {final_tal:.4f} < {final_3x3*0.8:.4f}")
        verdict = "TAL-justified"
    else:
        logger.info(f"  ~ TAL comparable or worse")
        verdict = "3x3-suffices"

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_samples": len(subset_ds),
        "dataset": "FullMultiModalDataset",
        "loss_3x3_initial": init_3x3,
        "loss_3x3_final": final_3x3,
        "loss_tal_initial": init_tal,
        "loss_tal_final": final_tal,
        "time_3x3_s": dt_3x3,
        "time_tal_s": dt_tal,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=str,
                        default="/tmp/tal_probe_results_correct.json")
    args = parser.parse_args()

    result = train_compare(args.n_steps, args.batch_size)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()