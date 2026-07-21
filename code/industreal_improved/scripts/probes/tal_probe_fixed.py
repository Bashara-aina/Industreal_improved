#!/usr/bin/env python3
"""Fixed TAL vs 3×3 probe — uses actual data pipeline from train_mtl_v3.py.

Compares detection_loss(use_tal=False) vs detection_loss(use_tal=True)
on an overfit-200 subset to determine whether TAL closes the detection gap.

Usage:
    python scripts/tal_probe_fixed.py --n-steps 200
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

from src.data.industreal_dataset import IndustRealMultiTaskDataset
from src.models.mvit_mtl_model import MTLMViTModel
from train_mtl_full_multimodal import expand_conv_proj_to_9ch, WrappedMTL
import train_mtl_v3 as mtl_v3_mod
# Override anchor config to match MTLMViTModel default (16 anchors)
mtl_v3_mod.NUM_ANCHORS = 16
mtl_v3_mod._ANCHOR_SPECS = mtl_v3_mod._ANCHOR_SPECS_16
from train_mtl_v3 import detection_loss, generate_anchors, NUM_DET_CLASSES, NUM_ANCHORS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FIXED = 200

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("tal_probe_fixed")


def normalize_images(images, device):
    """Convert to float, normalize to ~0 mean unit var, make [B,9,1,H,W]."""
    images = images.float().to(device)
    # images is [B, 3, H, W] from dataset
    mean = torch.tensor([0.45], device=device).view(1, 1, 1, 1)
    std = torch.tensor([0.225], device=device).view(1, 1, 1, 1)
    images = (images / 255.0 - mean) / std
    # Expand to 9 channels and add T=1 dim
    images = images.repeat(1, 3, 1, 1)  # [B, 9, H, W]
    images = images.unsqueeze(2)  # [B, 9, 1, H, W]
    return images


def collate_for_probe(batch):
    """Collate batch of dicts from IndustRealMultiTaskDataset into model inputs."""
    images = torch.stack([b['images']['rgb'] for b in batch])  # [B, 3, H, W]
    det_list = [b['detection'] for b in batch]  # list of {'boxes': [N,4], 'labels': [N]}
    return images, det_list


def build_anchors(det_out, device):
    """Build anchors for each FPN level."""
    anchors = {}
    for level, out in det_out.items():
        H, W = out['cls_logits'].shape[2], out['cls_logits'].shape[3]
        anchors[level] = generate_anchors(H, W, device)
    return anchors


def train_compare(n_steps: int, batch_size: int = 2):
    """Train detection head with 3×3 and TAL, compare convergence."""
    logger.info(f"{'='*60}")
    logger.info(f"TAL vs 3×3 on overfit-{N_FIXED} set ({n_steps} steps)")
    logger.info(f"{'='*60}")

    # Build dataset — 200 fixed images from training set
    train_ds = IndustRealMultiTaskDataset(
        split="train",
        img_size=(640, 360),
        augment=False,
        sequence_mode=False,
    )
    indices = list(range(min(N_FIXED, len(train_ds))))
    subset_ds = Subset(train_ds, indices)
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_for_probe)

    # Build model
    model = MTLMViTModel(num_act_classes=75, num_det_classes=24, num_psr_components=11)
    expand_conv_proj_to_9ch(model)
    model = model.to(DEVICE)

    # Freeze backbone (train only detection head)
    for name, p in model.named_parameters():
        if 'det_head' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    head_params = [p for p in model.det_head.parameters() if p.requires_grad]
    logger.info(f"Detection head params: {sum(p.numel() for p in head_params):,}")

    # ── Train with 3×3 assigner (pos_radius=1) ──
    logger.info(f"\nTraining 3×3 assigner for {n_steps} steps...")
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)
    losses_3x3 = []
    t0 = time.time()
    step = 0
    while step < n_steps:
        for images, det_list in loader:
            if step >= n_steps:
                break
            images = normalize_images(images, DEVICE)
            # Move targets to device
            for det in det_list:
                if 'boxes' in det and det['boxes'].numel() > 0:
                    det['boxes'] = det['boxes'].to(DEVICE).float()
                    det['labels'] = det['labels'].to(DEVICE).long()

            optimizer.zero_grad()
            out = model(images)
            anchors = build_anchors(out['detection'], DEVICE)
            # Extract boxes and classes separately
            gt_boxes = [d['boxes'] for d in det_list]
            gt_classes = [d['labels'] if 'labels' in d else d.get('classes', torch.zeros(0, dtype=torch.long)) for d in det_list]
            loss, _, _ = detection_loss(
                out['detection'], anchors, gt_boxes, gt_classes,
                use_tal=False,  # 3×3 assigner
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, 10.0)
                optimizer.step()
                losses_3x3.append(loss.item())

            if step % 50 == 0:
                logger.info(f"  3×3 step {step}/{n_steps}: loss={loss.item():.4f}")
            step += 1
    dt_3x3 = time.time() - t0
    logger.info(f"  3×3: initial={losses_3x3[0]:.4f}, final={losses_3x3[-1]:.4f}, "
                f"time={dt_3x3:.1f}s")

    # ── Reset head ──
    for m in model.det_head.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ── Train with TAL (topk=10) ──
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)
    logger.info(f"\nTraining TAL assigner for {n_steps} steps...")
    losses_tal = []
    t0 = time.time()
    step = 0
    while step < n_steps:
        for images, det_list in loader:
            if step >= n_steps:
                break
            images = normalize_images(images, DEVICE)
            for det in det_list:
                if 'boxes' in det and det['boxes'].numel() > 0:
                    det['boxes'] = det['boxes'].to(DEVICE).float()
                    det['labels'] = det['labels'].to(DEVICE).long()

            optimizer.zero_grad()
            out = model(images)
            anchors = build_anchors(out['detection'], DEVICE)
            gt_boxes = [d['boxes'] for d in det_list]
            gt_classes = [d['labels'] if 'labels' in d else d.get('classes', torch.zeros(0, dtype=torch.long)) for d in det_list]
            loss, _, _ = detection_loss(
                out['detection'], anchors, gt_boxes, gt_classes,
                use_tal=True,  # TAL assigner
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, 10.0)
                optimizer.step()
                losses_tal.append(loss.item())

            if step % 50 == 0:
                logger.info(f"  TAL step {step}/{n_steps}: loss={loss.item():.4f}")
            step += 1
    dt_tal = time.time() - t0
    logger.info(f"  TAL: initial={losses_tal[0]:.4f}, final={losses_tal[-1]:.4f}, "
                f"time={dt_tal:.1f}s")

    # ── Verdict ──
    final_3x3 = losses_3x3[-1]
    final_tal = losses_tal[-1]
    init_3x3 = losses_3x3[0]
    init_tal = losses_tal[0]

    logger.info(f"\n{'='*60}")
    logger.info("VERDICT:")
    logger.info(f"  3×3:  {init_3x3:.4f} → {final_3x3:.4f}")
    logger.info(f"  TAL:  {init_tal:.4f} → {final_tal:.4f}")

    verdict = "inconclusive"
    if final_3x3 < init_3x3 * 0.5:
        logger.info(f"  ✓ 3×3 overfits: loss dropped {init_3x3:.4f}→{final_3x3:.4f}")
        logger.info(f"    → 3×3 assigner IS learning. May not be the bottleneck.")
        verdict = "3x3-learning"
    if final_tal < final_3x3 * 0.8:
        logger.info(f"  ✓ TAL materially better: {final_tal:.4f} < {final_3x3*0.8:.4f} (80% of 3×3 final)")
        logger.info(f"    → TAL port justified (~2 days)")
        verdict = "TAL-justified"
    else:
        logger.info(f"  ∼ TAL comparable or worse than 3×3")
        logger.info(f"    → 3×3 is fine, save the 2 days")
        verdict = "3x3-suffices"

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_samples": len(subset_ds),
        "loss_3x3_initial": init_3x3,
        "loss_3x3_final": final_3x3,
        "loss_3x3_hist": losses_3x3,
        "loss_tal_initial": init_tal,
        "loss_tal_final": final_tal,
        "loss_tal_hist": losses_tal,
        "time_3x3_s": dt_3x3,
        "time_tal_s": dt_tal,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="TAL vs 3×3 probe (fixed)")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=str,
                        default="/tmp/tal_probe_results.json")
    args = parser.parse_args()

    result = train_compare(args.n_steps, args.batch_size)
    with open(args.output, "w") as f:
        json.dump({k: v for k, v in result.items() if "hist" not in k}, f, indent=2)
    logger.info(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
