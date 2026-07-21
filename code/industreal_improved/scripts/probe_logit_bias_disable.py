#!/usr/bin/env python3
"""Probe: does disabling update_logit_bias() help background suppression?

Hypothesis (from root-cause analysis):
- update_logit_bias() fills ALL 24 class biases with the SAME value computed
  from EMA of positive ratio. With ~3 positives out of ~75K anchors/batch,
  pos_ratio ≈ 4e-5, giving target_bias ≈ -10.
- But checkpoint shows class biases cluster at -0.22 to -0.49 (sigmoid ~0.42).
- This means the fill_() call is being constantly OVERWRITTEN by class-specific
  gradient updates. With uniform fill + per-class gradient divergence, the
  model cannot stably learn per-class background suppression.

This probe runs 500 batches with TWO configs:
  A) update_logit_bias() enabled (baseline - matches training behavior)
  B) update_logit_bias() DISABLED - bias left at initial prior_prob-derived value

We then evaluate:
- Mean background confidence (should DROP if suppression works)
- Foreground vs background confidence separation
- Positive anchor recall

Usage:
    python scripts/probe_logit_bias_disable.py --n-steps 500
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
logger = logging.getLogger("probe_logit_bias")


def build_model():
    """Build the multi-task model wrapped for 9ch input."""
    base = MTLMViTModel(
        num_act_classes=75, num_det_classes=24, num_psr_components=11, num_anchors=16,
    )
    expand_conv_proj_to_9ch(base)
    base = base.to(DEVICE)
    return WrappedMTL(base)


def freeze_backbone_keep_det(model):
    """Freeze all params except detection head."""
    for name, p in model.named_parameters():
        if 'det_head' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return [p for p in model.m.det_head.parameters() if p.requires_grad]


def reset_det_head(model):
    """Reset detection head weights (Xavier) and bias to initial prior_prob value."""
    import math
    for m in model.m.det_head.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                if m.out_channels == 24:  # classification final layer
                    # Reset to initial prior_prob=0.01 bias = -log(0.99/0.01) ≈ -4.6
                    nn.init.constant_(m.bias, -math.log(0.99 / 0.01))
                else:
                    nn.init.zeros_(m.bias)
    # Reset running_pos_ratio to prior_prob
    model.m.det_head.running_pos_ratio.fill_(0.01)


def train_run(
    n_steps: int,
    batch_size: int,
    enable_bias_update: bool,
    seed: int = 42,
) -> dict:
    """Run N training steps; return loss/conf metrics.

    Args:
        enable_bias_update: if False, override update_logit_bias to be a no-op.
    """
    torch.manual_seed(seed)

    train_ds = FullMultiModalDataset(
        recordings_dir="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train",
        img_size=(640, 360),
        mosaic_prob=0.0,  # disable aug for reproducibility
        copy_paste_prob=0.0,
    )
    indices = list(range(min(N_FIXED, len(train_ds))))
    subset_ds = Subset(train_ds, indices)
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_real_targets, num_workers=0)

    model = build_model()
    head_params = freeze_backbone_keep_det(model)

    # If disabling bias update, monkey-patch the method
    if not enable_bias_update:
        def _no_op_update(self, batch_pos_ratio, momentum=0.05):
            pass
        import types
        model.m.det_head.update_logit_bias = types.MethodType(_no_op_update, model.m.det_head)
        logger.info("  update_logit_bias() DISABLED")
    else:
        logger.info("  update_logit_bias() ENABLED (baseline)")

    opt = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)

    losses = []
    pos_counts = []
    bg_conf_means = []  # mean sigmoid confidence for background locations
    fg_conf_means = []  # mean sigmoid confidence for foreground (pos) locations
    bias_history = []  # track bias[0] (one representative class)

    reset_det_head(model)
    initial_bias = float(model.m.det_head.cls_head[3].bias.data[0].item())
    logger.info(f"  Initial bias[0]: {initial_bias:.4f} (sigmoid={1/(1+2.71828**-initial_bias):.4f})")

    step = 0
    t0 = time.time()
    while step < n_steps:
        for images, targets in loader:
            if step >= n_steps:
                break
            images = ensure_5d(images).float().to(DEVICE)
            # images: [B, 9, 1, H, W] from FullMultiModalDataset (already in [0,1])
            # Normalize all 9 channels with same mean/std
            mean = torch.tensor([0.45] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            std = torch.tensor([0.225] * 9, device=DEVICE).view(1, 9, 1, 1, 1)
            images = (images - mean) / std

            # Move targets to device
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
                logger.warning(f"  NaN/Inf at step {step}, skipping")
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 10.0)
            opt.step()
            losses.append(loss.item())

            # Compute background vs foreground confidence separation
            with torch.no_grad():
                all_bg_conf = []
                all_fg_conf = []
                for level in ['P3', 'P4', 'P5']:
                    if level not in out['detection']:
                        continue
                    cls_logits = out['detection'][level]['cls_logits']  # [B, 24, H, W]
                    scores = torch.sigmoid(cls_logits).max(dim=1)[0]  # [B, H, W]
                    # For each batch sample, identify positive cell locations via match
                    anchors_lvl = anchors[level]  # [H, W, A, 4]
                    H, W = scores.shape[1], scores.shape[2]
                    A = anchors_lvl.shape[2]
                    for b_idx in range(cls_logits.shape[0]):
                        gt_b = gt_boxes[b_idx]
                        scores_flat = scores[b_idx].reshape(-1).cpu()  # [H*W]
                        if gt_b.numel() == 0:
                            all_bg_conf.append(scores_flat)
                            continue
                        # Find positive anchor cells (any anchor within cell matches)
                        from train_mtl_v3 import match_anchors_to_gt
                        anc_flat = anchors_lvl.reshape(-1, 4)
                        _, _, pos_mask, _ = match_anchors_to_gt(
                            anc_flat, gt_b, gt_classes[b_idx], iou_threshold=0.5,
                        )
                        # Reduce anchor-level mask to cell-level mask
                        cell_pos = pos_mask.reshape(H, W, A).any(dim=-1).reshape(-1).cpu()
                        if cell_pos.sum() > 0:
                            all_fg_conf.append(scores_flat[cell_pos])
                        all_bg_conf.append(scores_flat[~cell_pos])
                if all_bg_conf:
                    bg_concat = torch.cat(all_bg_conf)
                    bg_mean = bg_concat.mean().item()
                else:
                    bg_mean = float('nan')
                if all_fg_conf:
                    fg_concat = torch.cat(all_fg_conf)
                    fg_mean = fg_concat.mean().item()
                else:
                    fg_mean = float('nan')
                bg_conf_means.append(bg_mean)
                fg_conf_means.append(fg_mean)

            # Bias update (only if enabled)
            if enable_bias_update:
                # Compute pos_ratio from this batch
                total_pos = sum(p.numel() for p in [b for b in []])
                # Recompute positive count
                n_pos = 0
                n_total = 0
                for level in ['P3', 'P4', 'P5']:
                    if level not in out['detection']:
                        continue
                    anchors_lvl = anchors[level]
                    for b_idx in range(out['detection'][level]['cls_logits'].shape[0]):
                        anc_flat = anchors_lvl.reshape(-1, 4)
                        n_total += anc_flat.shape[0]
                        gt_b = gt_boxes[b_idx]
                        if gt_b.numel() == 0:
                            continue
                        from train_mtl_v3 import match_anchors_to_gt
                        _, _, pos_mask, _ = match_anchors_to_gt(
                            anc_flat, gt_b, gt_classes[b_idx], iou_threshold=0.5,
                        )
                        n_pos += int(pos_mask.sum().item())
                pos_ratio = n_pos / max(n_total, 1)
                model.m.det_head.update_logit_bias(pos_ratio)

            if step % 50 == 0:
                bias0 = float(model.m.det_head.cls_head[3].bias.data[0].item())
                bias_history.append(bias0)
                logger.info(f"  step {step}/{n_steps}: loss={loss.item():.4f}, "
                            f"bg_conf={bg_mean:.4f}, fg_conf={fg_mean if fg_mean != fg_mean else 0:.4f}, "
                            f"bias[0]={bias0:.3f}")

            step += 1

    dt = time.time() - t0
    final_bias = float(model.m.det_head.cls_head[3].bias.data[0].item())
    return {
        "losses": losses,
        "bg_conf_means": bg_conf_means,
        "fg_conf_means": fg_conf_means,
        "initial_bias": initial_bias,
        "final_bias": final_bias,
        "time_s": dt,
    }


def summarize(name: str, res: dict) -> dict:
    """Summarize run results."""
    losses = res["losses"]
    bg = res["bg_conf_means"]
    fg = res["fg_conf_means"]
    final_bias = res["final_bias"]
    initial_bias = res["initial_bias"]

    # Drop first 10 steps (warmup noise) for stable measurement
    bg_stable = bg[10:] if len(bg) > 10 else bg
    fg_stable = [f for f in fg[10:] if f == f] if len(fg) > 10 else [f for f in fg if f == f]

    bg_initial = bg[0] if bg else float('nan')
    bg_final = sum(bg_stable) / len(bg_stable) if bg_stable else float('nan')
    fg_initial = fg[0] if fg else float('nan')
    fg_final = sum(fg_stable) / len(fg_stable) if fg_stable else float('nan')

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {name}")
    logger.info(f"  Initial bias[0]: {initial_bias:.4f}")
    logger.info(f"  Final bias[0]:   {final_bias:.4f}")
    logger.info(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    logger.info(f"  Background conf: {bg_initial:.4f} -> {bg_final:.4f} (mean over stable)")
    logger.info(f"  Foreground conf: {fg_initial:.4f} -> {fg_final:.4f}")
    logger.info(f"  Separation (FG - BG): {fg_final - bg_final:.4f}")
    logger.info(f"  Time: {res['time_s']:.1f}s")

    return {
        "name": name,
        "initial_bias": initial_bias,
        "final_bias": final_bias,
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "bg_conf_initial": bg_initial,
        "bg_conf_final": bg_final,
        "fg_conf_initial": fg_initial,
        "fg_conf_final": fg_final,
        "separation_final": fg_final - bg_final if fg_final == fg_final else None,
        "time_s": res["time_s"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=str,
                        default="/tmp/logit_bias_probe.json")
    args = parser.parse_args()

    logger.info(f"Probe: update_logit_bias enable vs disable ({args.n_steps} steps)")
    logger.info(f"Device: {DEVICE}")

    # Run A: update_logit_bias ENABLED (baseline)
    res_a = train_run(args.n_steps, args.batch_size, enable_bias_update=True, seed=42)
    summary_a = summarize("A) update_logit_bias ENABLED (baseline)", res_a)

    # Run B: update_logit_bias DISABLED
    res_b = train_run(args.n_steps, args.batch_size, enable_bias_update=False, seed=42)
    summary_b = summarize("B) update_logit_bias DISABLED", res_b)

    # Verdict
    logger.info(f"\n{'='*60}")
    logger.info("VERDICT:")
    if summary_b["bg_conf_final"] < summary_a["bg_conf_final"]:
        logger.info(f"  ✓ DISABLED achieves lower BG conf: "
                    f"{summary_b['bg_conf_final']:.4f} < {summary_a['bg_conf_final']:.4f}")
        logger.info(f"  → Disable update_logit_bias() in training")
        verdict = "disable-helps"
    elif summary_b["separation_final"] > summary_a["separation_final"]:
        logger.info(f"  ✓ DISABLED has better FG/BG separation: "
                    f"{summary_b['separation_final']:.4f} > {summary_a['separation_final']:.4f}")
        verdict = "disable-helps-separation"
    else:
        logger.info(f"  ∼ No clear benefit from disabling")
        verdict = "no-improvement"

    output = {
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "baseline_enabled": summary_a,
        "test_disabled": summary_b,
        "verdict": verdict,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()