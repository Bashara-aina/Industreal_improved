#!/usr/bin/env python3
"""
train_mtl_mvit.py — MTL-All with MViTv2-S shared backbone.

Trains all 4 heads (detection, activity, PSR, pose) simultaneously using
Kendall uncertainty weighting + PCGrad gradient surgery.

Usage:
    # Full training (default: 100 epochs, eval on val split)
    python scripts/train_mtl_mvit.py

    # Plumbing test (1 epoch, small subset)
    python scripts/train_mtl_mvit.py --plumbing

    # Resume from checkpoint
    python scripts/train_mtl_mvit.py --resume path/to/checkpoint.pt

    # Evaluate best checkpoint on test split only
    python scripts/train_mtl_mvit.py --test-only --resume output/best.pt

    # Eval split: val for model selection (default), test for final eval
    python scripts/train_mtl_mvit.py --eval-split val
"""
import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path
_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force 75-class activity for SOTA comparison (175 §7.2, §3.3)
# Also reduce RAM cache from env (allows fork() with DataLoader workers)
import os as _os
import src.config as C
C.ACT_CLASS_GROUPING = "none"
C.RAM_CACHE_MAX_IMAGES = int(_os.environ.get("RAM_CACHE_MAX_IMAGES", C.RAM_CACHE_MAX_IMAGES))

# Need to set before dataset and model init, since _act_grouping() runs at import
C.NUM_ACT_OUTPUTS = 75

from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
from src.models.mvit_mtl_model import MTLMViTModel, renormalize_pose
from src.split_config import require_split

logger = logging.getLogger("train_mtl_mvit")

OUTPUT_ROOT = _CODE_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "mtl_mvit_run"


# ===========================================================================
# Loss functions
# ===========================================================================

def ciou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """CIoU loss for box regression.

    Args:
        pred_boxes: [N, 4] xyxy predicted boxes.
        gt_boxes: [N, 4] xyxy ground truth boxes.

    Returns:
        [N] CIoU loss values.
    """
    # Intersection
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = pred_area + gt_area - inter

    iou = inter / (union + 1e-7)

    # Center distance
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    rho2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

    # Enclosing box diagonal
    en_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    en_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    en_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    en_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    c2 = (en_x2 - en_x1) ** 2 + (en_y2 - en_y1) ** 2 + 1e-7

    # Aspect ratio consistency
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    v = (4 / math.pi ** 2) * (torch.atan(gt_w / (gt_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7))) ** 2
    alpha = v / ((1 - iou).detach() + v + 1e-7)

    ciou = iou - rho2 / c2 - alpha * v
    return 1 - ciou


def dfl_loss_fn(
    pred_dist: torch.Tensor,
    target: torch.Tensor,
    reg_max: int = 16,
) -> torch.Tensor:
    """Distribution Focal Loss (DFL).

    Args:
        pred_dist: [N, 4, reg_max] predicted distribution logits.
        target: [N, 4] continuous target values in (l,t,r,b) grid units.
        reg_max: number of distribution bins per coordinate.

    Returns:
        [N] DFL loss values.
    """
    target = target.clamp(0, reg_max - 0.01).reshape(-1)  # [N*4]
    tl = target.long().clamp(0, reg_max - 2)
    tr = tl + 1
    wl = (tr - target).detach()
    wr = 1 - wl
    pred_flat = pred_dist.reshape(-1, reg_max)  # [N*4, reg_max]
    loss = (
        F.cross_entropy(pred_flat, tl, reduction="none") * wl
        + F.cross_entropy(pred_flat, tr, reduction="none") * wr
    )
    return loss.reshape(-1, 4).mean(dim=1)


def detection_loss(
    det_outputs: dict,
    det_list: list,
    num_classes: int = 24,
    reg_max: int = 16,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """YOLOv8-style detection loss over 4 FPN levels.

    Combines Focal loss (cls), CIoU (box), and DFL (box distribution) with
    center-based grid assignment per FPN level.

    Args:
        det_outputs: dict of per-level {cls_logits, reg_preds}.
        det_list: list of B dicts, each {boxes: [n_i, 4] xyxy, labels: [n_i]}.
        num_classes: number of detection classes.
        reg_max: DFL bins per coordinate.
        gamma: focal loss gamma.
        alpha: focal loss alpha.

    Returns:
        scalar loss.
    """
    device = next(iter(det_outputs.values()))["cls_logits"].device
    strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}

    if not det_list:
        return torch.tensor(0.0, device=device)

    loss_cls = 0.0
    loss_iou = 0.0
    loss_dfl = 0.0
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
        cell_cx = xs.float() * stride + stride / 2.0  # [W]
        cell_cy = ys.float() * stride + stride / 2.0  # [H]

        # Per-image GT assignment
        cls_target = torch.zeros(B, H, W, dtype=torch.long, device=device)
        pos_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        dfl_target = torch.zeros(B, H, W, 4, device=device)
        iou_target = torch.zeros(B, H, W, 4, device=device)

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

            # xyxy -> center
            gt_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            gt_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

            for n in range(boxes.shape[0]):
                gi = (gt_cx[n] / stride).long().clamp(0, W - 1)
                gj = (gt_cy[n] / stride).long().clamp(0, H - 1)

                if pos_mask[b, gj, gi]:
                    continue

                pos_mask[b, gj, gi] = True
                cls_target[b, gj, gi] = labels[n].long()

                # DFL target: (l,t,r,b) offsets from cell center in grid units
                dfl_target[b, gj, gi, 0] = (gt_cx[n] - boxes[n, 0]) / stride  # left
                dfl_target[b, gj, gi, 1] = (gt_cy[n] - boxes[n, 1]) / stride  # top
                dfl_target[b, gj, gi, 2] = (boxes[n, 2] - gt_cx[n]) / stride  # right
                dfl_target[b, gj, gi, 3] = (boxes[n, 3] - gt_cy[n]) / stride  # bottom

                iou_target[b, gj, gi] = boxes[n]

        # ---- Classification: Focal loss ----
        cls_logits_p = cls_logits.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 24]
        cls_onehot = F.one_hot(cls_target, num_classes).float()
        cls_prob = torch.sigmoid(cls_logits_p)
        pt = cls_onehot * cls_prob + (1 - cls_onehot) * (1 - cls_prob)
        focal_w = (1 - pt) ** gamma
        alpha_t = cls_onehot * alpha + (1 - cls_onehot) * (1 - alpha)
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits_p, cls_onehot, reduction="none")
        cls_loss = (alpha_t * focal_w * cls_loss).sum(dim=-1)  # [B, H, W]
        loss_cls = loss_cls + cls_loss.mean()

        # ---- Box losses (positive cells only) ----
        if pos_mask.any():
            # Reshape reg_preds for DFL: [B, 4*reg_max, H, W] -> [B, 4, reg_max, H, W]
            reg_dist = reg_preds.view(B, 4, reg_max, H, W)  # [B, 4, reg_max, H, W]

            # DFL loss
            pred_dist = reg_dist.permute(0, 3, 4, 1, 2)[pos_mask]  # [P, 4, reg_max]
            gt_dfl = dfl_target[pos_mask]  # [P, 4]
            loss_dfl = loss_dfl + dfl_loss_fn(pred_dist, gt_dfl, reg_max).mean()

            # Decode DFL distribution to continuous offsets for CIoU
            proj = torch.arange(reg_max, device=device).float().view(1, 1, reg_max, 1, 1)
            decoded = (reg_dist.softmax(dim=2) * proj).sum(dim=2)  # [B, 4, H, W]

            # Convert decoded (l,t,r,b) offsets to absolute xyxy
            pred_x1 = cell_cx.view(1, 1, W) - decoded[:, 0:1] * stride  # [B, 1, H, W]
            pred_y1 = cell_cy.view(1, H, 1) - decoded[:, 1:2] * stride  # [B, 1, H, W]
            pred_x2 = cell_cx.view(1, 1, W) + decoded[:, 2:3] * stride  # [B, 1, H, W]
            pred_y2 = cell_cy.view(1, H, 1) + decoded[:, 3:4] * stride  # [B, 1, H, W]
            pred_abs = torch.cat([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # [B, 4, H, W]
            pred_abs = pred_abs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 4]

            loss_iou = loss_iou + ciou_loss(
                pred_abs[pos_mask],
                iou_target[pos_mask],
            ).mean()

    n_levels = max(n_levels, 1)
    return loss_cls / n_levels + loss_iou / n_levels + loss_dfl / n_levels


def compute_activity_class_weights(
    dataset: "IndustRealMultiTaskDataset",
    num_classes: int = 75,
) -> torch.Tensor:
    """Inverse-frequency class weights for long-tailed activity labels.

    Args:
        dataset: Training dataset with ``class_counts`` attribute.
        num_classes: Number of activity classes (default 75).

    Returns:
        [num_classes] float tensor of inverse-frequency weights.
    """
    counts = dataset.class_counts.astype(np.float64)
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
    logger.info(
        "Class weights — min=%.4f  max=%.4f  mean=%.4f  num_nonzero=%d",
        weights.min(), weights.max(), weights.mean(),
        int((weights > 0).sum()),
    )
    return torch.from_numpy(weights).float()


def activity_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Class-balanced cross-entropy with label smoothing 0.1 and ignore-index -1.

    Args:
        logits: [B, 75] per-video activity logits.
        targets: [B] per-video activity labels (0..74; -1 = unlabeled / ignore).
        class_weights: [75] inverse-frequency weights or None for uniform.

    Returns:
        Scalar loss. Returns 0.0 (with no grad) if every label in the batch is
        unlabeled (P2 guard — non-fatal so training doesn't crash on rare
        all-unlabeled windows).
    """
    valid_mask = targets != -1
    if not valid_mask.any():
        # P2 guard (Doc 175 §2): never let ALL labels be ignored. This batch
        # has no supervision; contribute zero loss with no gradient. The
        # learnable log_var still gets its regularization term.
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits, targets,
        weight=class_weights,
        ignore_index=-1,
        label_smoothing=0.1,
    )


def psr_loss(
    psr_logits: torch.Tensor,
    psr_targets: torch.Tensor,
    comp_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-frame BCE for PSR transition logits with optional per-component inverse-prevalence weights.

    Args:
        psr_logits: [B, T, 11] per-frame transition logits.
        psr_targets: [B, T, 11] per-frame transition targets.
        comp_weights: [1, 1, 11] per-component inverse-prevalence weights or None.

    Returns:
        Scalar loss (mean over all elements).
    """
    loss = F.binary_cross_entropy_with_logits(psr_logits, psr_targets, reduction='none')
    if comp_weights is not None:
        loss = loss * comp_weights  # broadcast [B, T, 11] * [1, 1, 11]
    return loss.mean()


def pose_loss(pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
    """Cosine/geodesic loss on renormalized fwd and up vectors."""
    fwd_pred, up_pred = renormalize_pose(pred_6d)
    fwd_gt = F.normalize(target_6d[:, :3], dim=1)
    up_gt = F.normalize(target_6d[:, 3:], dim=1)
    cos_fwd = (fwd_pred * fwd_gt).sum(dim=1).clamp(-1.0, 1.0)
    cos_up = (up_pred * up_gt).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cos_fwd).mean() + (1.0 - cos_up).mean()


# ===========================================================================
# Efficiency measurement
# ===========================================================================

def measure_efficiency(args):
    """Measure FLOPs, FPS, peak VRAM, and parameter efficiency.

    Runs independently of training data pipeline — only needs model + device.
    Saves results to ``efficiency_metrics.json`` in the output directory.
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
    except ImportError:
        logger.error(
            "fvcore is required for efficiency measurement. "
            "Install: pip install fvcore"
        )
        sys.exit(1)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    logger.info("Device: %s", device)

    model = MTLMViTModel().to(device)
    model.eval()

    # ---- 1. Parameter count ----
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    logger.info("Params: %.1fM total, %.1fM trainable", total_params, trainable_params)
    param_table = parameter_count_table(model)
    logger.info("Parameter count table:\n%s", param_table)

    results = {
        "total_params_M": round(total_params, 2),
        "trainable_params_M": round(trainable_params, 2),
    }

    # ---- 2. FLOPs analysis ----
    logger.info("Measuring FLOPs...")

    # 2a. Temporal mode: [1, 3, 16, 224, 224]
    dummy_temporal = torch.randn(1, 3, 16, 224, 224, device=device)
    try:
        flops_temporal = FlopCountAnalysis(model, dummy_temporal)
        flops_temporal_g = flops_temporal.total() / 1e9
        results["flops_temporal_G"] = round(flops_temporal_g, 2)
        logger.info("Temporal mode (16x224x224): %.2f GFLOPs", flops_temporal_g)
    except Exception as e:
        logger.warning("FLOPs measurement (temporal) failed: %s", e)
        results["flops_temporal_G"] = None

    # 2b. Per-frame: [1, 3, 1, 224, 224]
    dummy_frame = torch.randn(1, 3, 1, 224, 224, device=device)
    try:
        flops_frame = FlopCountAnalysis(model, dummy_frame)
        flops_frame_g = flops_frame.total() / 1e9
        results["flops_per_frame_G"] = round(flops_frame_g, 2)
        logger.info("Per-frame (1x224x224): %.2f GFLOPs", flops_frame_g)
    except Exception as e:
        logger.warning("FLOPs measurement (per-frame) failed: %s", e)
        results["flops_per_frame_G"] = None

    # ---- 3. FPS measurement (batch=1, temporal mode) ----
    logger.info("Measuring FPS (batch=1, temporal, 100 forward passes)...")
    dummy = torch.randn(1, 3, 16, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    fps = 100.0 / elapsed
    results["fps_temporal"] = round(fps, 2)
    results["fps_measurement_seconds"] = round(elapsed, 3)
    logger.info("FPS: %.2f (%.3fs for 100 passes)", fps, elapsed)

    # ---- 4. Peak VRAM ----
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        results["peak_vram_GB"] = round(peak_memory_gb, 3)
        logger.info("Peak VRAM: %.3f GB", peak_memory_gb)
    else:
        results["peak_vram_GB"] = None
        logger.info("Peak VRAM: N/A (CPU mode)")

    # ---- 5. Comparison to single-task estimate ----
    estimated_single_task_sum_M = 100.0  # ~100M for 4x single-task sum (Doc 175 §8 Table C)
    results["estimated_single_task_sum_M"] = estimated_single_task_sum_M
    results["mtl_vs_single_ratio"] = round(total_params / estimated_single_task_sum_M, 3)
    logger.info(
        "MTL params (%.1fM) vs estimated 4x single task (%.0fM): %.1fx smaller",
        total_params, estimated_single_task_sum_M,
        estimated_single_task_sum_M / total_params,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "efficiency_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Efficiency metrics saved: %s", metrics_path)

    return results


# ===========================================================================
# PCGrad gradient surgery
# ===========================================================================

def pcgrad_fn(
    per_task_grads: list,
    shared_params: list,
) -> list:
    """PCGrad: project conflicting per-task gradients onto each other.

    For each task pair (i, j) with cosine similarity < 0, project g_i away
    from the conflict direction:
        g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j

    Args:
        per_task_grads: T tuples of gradient tensors (one per shared param).
        shared_params: list of shared backbone parameter tensors.

    Returns:
        Deconflicted summed gradients, one per shared parameter.
    """
    num_tasks = len(per_task_grads)
    device = shared_params[0].device

    # Flatten per-task gradients into single vectors
    flat_grads = []
    for task_grads in per_task_grads:
        pieces = []
        for g, p in zip(task_grads, shared_params):
            if g is None:
                pieces.append(torch.zeros(p.numel(), device=device, dtype=p.dtype))
            else:
                pieces.append(g.contiguous().view(-1))
        flat_grads.append(torch.cat(pieces))

    # Random task ordering (core PCGrad step)
    task_order = torch.randperm(num_tasks, device=device)

    for i_idx in range(num_tasks):
        i = task_order[i_idx]
        gi = flat_grads[i]
        for j_idx in range(i_idx):
            j = task_order[j_idx]
            gj = flat_grads[j]

            dot_ij = torch.dot(gi, gj)
            gj_norm_sq = torch.dot(gj, gj)
            if gj_norm_sq > 0 and dot_ij < 0:
                gi = gi - (dot_ij / gj_norm_sq) * gj
                flat_grads[i] = gi

    # Sum all deconflicted gradients
    sum_grad = torch.stack(flat_grads).sum(dim=0)

    # Unflatten back to parameter shapes
    result = []
    offset = 0
    for p in shared_params:
        numel = p.numel()
        result.append(sum_grad[offset:offset + numel].view_as(p).contiguous())
        offset += numel

    return result


# ===========================================================================
# Training step
# ===========================================================================

def train_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: dict,
    log_vars: nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip_norm: float = 1.0,
    hp_prec_cap: bool = True,
    pcgrad: bool = True,
    act_class_weights: Optional[torch.Tensor] = None,
    do_step: bool = True,
) -> dict:
    """Single training step with Kendall uncertainty weighting and optional PCGrad gradient surgery.

    Args:
        do_step: if True, calls optimizer.step() and zero_grad(). Set False for gradient
                 accumulation — caller must call optimizer.step() at the accumulation boundary.
    """
    model.train()
    B = images.size(0)

    # Forward
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(images)

        # Per-task losses
        # Detection: simple presence BCE
        det_list = targets.get("detection", [])
        l_det = detection_loss(outputs["detection"], det_list)

        l_act = activity_loss(
            outputs["activity"],
            targets.get("activity", torch.zeros(B, dtype=torch.long, device=images.device)),
            class_weights=act_class_weights,
        ) if "activity" in targets else torch.tensor(0.0, device=images.device)

        # PSR per-component inverse-prevalence weights (Doc 175 §4)
        _psr_comp_weights = None
        psr_comp_breakdown = {}
        if "psr_labels" in targets:
            if hasattr(C, 'PSR_COMP_WEIGHTS') and C.PSR_COMP_WEIGHTS:
                _psr_comp_weights = torch.tensor(
                    C.PSR_COMP_WEIGHTS, device=images.device, dtype=torch.float32
                ).view(1, 1, -1)
            else:
                # Fallback: compute inverse prevalence from current batch
                _pos = (targets["psr_labels"] > 0).float().sum(dim=(0, 1))
                _total = targets["psr_labels"].size(0) * targets["psr_labels"].size(1)
                _prevalence = _pos / (_total + 1e-6)
                _inv_weights = 1.0 / (_prevalence + 1e-6)
                _psr_comp_weights = (_inv_weights / _inv_weights[0].clamp(min=1e-6)).view(1, 1, -1)

        l_psr = psr_loss(
            outputs["psr_logits"],
            targets.get("psr_labels", torch.zeros(B, 16, 11, device=images.device)),
            comp_weights=_psr_comp_weights,
        ) if "psr_labels" in targets else torch.tensor(0.0, device=images.device)

        # Per-component PSR loss breakdown for logging
        if "psr_labels" in targets:
            with torch.no_grad():
                _pc = F.binary_cross_entropy_with_logits(
                    outputs["psr_logits"], targets["psr_labels"], reduction='none'
                ).mean(dim=(0, 1))
            for ci in range(_pc.size(0)):
                psr_comp_breakdown[f"loss_psr_c{ci}"] = _pc[ci].item()

        # Pose: head_pose is [B, T, 9], take middle frame's 6D
        if "head_pose" in targets:
            hp = targets["head_pose"]  # [B, T, 9]
            hp_6d = hp[:, hp.size(1) // 2, :6]  # [B, 6] middle frame
        else:
            hp_6d = torch.zeros(B, 6, device=images.device)
        l_pose = pose_loss(outputs["pose_6d"], hp_6d)

        # Kendall uncertainty weighting with precision capping
        losses = {"det": l_det, "act": l_act, "psr": l_psr, "pose": l_pose}
        total_loss = 0.0

        # Compute log_vars, with head-pose precision capped by detection (Doc 175 §5.2)
        # Pose precision (exp(-lv)) must never exceed detection precision. This prevents
        # the shared backbone from being optimized primarily for head_pose (loss ~0.01)
        # while neglecting detection (loss ~0.5), which has ~40x higher loss magnitude.
        # SAFETY: Clamp log_var to [-4, 4] to prevent exp(-lv) explosion (would cause
        # negative total_loss and gradient NaN). Per 175 §5.1: act_min=-4, psr/pose_max=2.
        LV_CLAMP_MIN, LV_CLAMP_MAX = -4.0, 4.0
        lv_values = {}
        for name in losses:
            lv_clamped = log_vars[name].clamp(LV_CLAMP_MIN, LV_CLAMP_MAX)
            lv_values[name] = lv_clamped
        if hp_prec_cap:
            lv_values["pose"] = torch.maximum(
                lv_values["pose"], lv_values["det"].detach()
            )

        # SAFETY: Clamp each per-task loss to [0, +inf) — BCE/CE/DFL/CIoU are all
        # bounded ≥ 0 in theory. Negative values come from numerical drift.
        losses_safe = {n: torch.clamp(v, min=0.0) if v.isfinite() else torch.zeros_like(v)
                       for n, v in losses.items()}

        for name, loss in losses_safe.items():
            lv = lv_values[name]
            prec = torch.exp(-lv)
            total_loss = total_loss + prec * loss + lv / 2

        # SAFETY: Skip step if total_loss is non-finite (log_var explosion caught here)
        if not torch.isfinite(total_loss):
            logger.warning(
                "Non-finite total_loss at step: det=%s act=%s psr=%s pose=%s | skipping optimizer step",
                losses["det"].item(), losses["act"].item(), losses["psr"].item(), losses["pose"].item(),
            )
            optimizer.zero_grad()
            return {
                "loss": float("nan"),
                "loss_det": losses["det"].item(), "loss_act": losses["act"].item(),
                "loss_psr": losses["psr"].item(), "loss_pose": losses["pose"].item(),
                **psr_comp_breakdown,
                **{f"log_var_{k}": v.item() for k, v in log_vars.items()},
            }

    # Backward
    optimizer.zero_grad()

    if pcgrad:
        shared_params = [p for p in model.feature_pyramid.backbone.parameters() if p.requires_grad]

        # Per-task gradients w.r.t. shared backbone
        # SAFETY: Use clamped log_var to match the forward pass (otherwise PCGrad
        # uses different precisions than the actual loss, causing gradient/total
        # loss mismatch). Also use losses_safe to prevent negative gradient.
        per_task_grads = []
        for name in losses:
            lv_safe = lv_values[name]  # already clamped
            prec = torch.exp(-lv_safe)
            weighted_loss = prec * losses_safe[name]
            g = torch.autograd.grad(
                weighted_loss, shared_params,
                retain_graph=True, allow_unused=True,
            )
            per_task_grads.append(g)

        # PCGrad: project conflicting gradients
        deconflicted = pcgrad_fn(per_task_grads, shared_params)

        # Backward for head params (also populates backbone grads, overridden below)
        # For bf16 AMP, the scaler is mostly a no-op; we skip explicit unscale_() to
        # avoid issues with gradient accumulation (unscale_ can only be called once
        # per scaler.update() cycle).
        scaler.scale(total_loss).backward()

        # Override backbone grads with PCGrad deconflicted grads
        for param, grad in zip(shared_params, deconflicted):
            param.grad = grad.to(param.dtype)
    else:
        scaler.scale(total_loss).backward()

    # Gradient clipping
    if do_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    # NOTE: With grad accumulation, scaler.unscale_() is only safe to call once
    # between scaler.update() calls. The first accumulated call here would
    # call scaler.scale(total_loss).backward(), accumulating gradients. Subsequent
    # calls in the same accumulation cycle would also try unscale_() — we skip
    # that since AMP bf16 doesn't need explicit unscale (no inf/nan checks).

    # For bf16, scaler is essentially a no-op (no inf check needed). We avoid
    # calling unscale_() in accumulation since it would raise.

    return {
        "loss": total_loss.item(),
        "loss_det": l_det.item(),
        "loss_act": l_act.item(),
        "loss_psr": l_psr.item(),
        "loss_pose": l_pose.item(),
        **psr_comp_breakdown,
        **{f"log_var_{k}": v.item() for k, v in log_vars.items()},
    }


# ===========================================================================
# Validation evaluation (Doc 175 section 7.2)
# ===========================================================================

def _compute_tau(pred_tr: np.ndarray, gt_tr: np.ndarray, tol: int = 3) -> float:
    """Mean signed delay between matched predicted and GT transition events.

    Positive = lag, negative = anticipation. NaN if no matches.
    """
    n_comp = pred_tr.shape[1]
    delays = []
    for c in range(n_comp):
        p_frames = np.where(pred_tr[:, c])[0]
        g_frames = np.where(gt_tr[:, c])[0]
        matched_gt = set()
        for pf in p_frames:
            best_delay = None
            best_gi = None
            for gi, gf in enumerate(g_frames):
                if gi not in matched_gt and abs(pf - gf) <= tol:
                    d = int(pf) - int(gf)
                    if best_delay is None or abs(d) < abs(best_delay):
                        best_delay = d
                        best_gi = gi
            if best_gi is not None:
                matched_gt.add(best_gi)
                delays.append(best_delay)
    if not delays:
        return float("nan")
    return float(np.mean(delays))


def _compute_pos(pred_tr: np.ndarray, gt_tr: np.ndarray) -> float:
    """Ordered-pair fraction: directional sign agreement of transitions."""
    return float((np.sign(pred_tr) == np.sign(gt_tr)).mean())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int | None = None,
) -> dict:
    """Full validation evaluation across all 4 tasks (Doc 175 section 7.2).

    Computes:
      - PSR: event_f1 at +/-3, POS, tau  (via decoder_oracle_bound.event_f1)
      - Activity: clip-level top-1 / top-5 on 75 classes
      - Detection: dual-protocol mAP@0.5 / mAP@0.5:0.95 + presence BCE
      - Pose: angular MAE with bootstrap CI (fwd + up in degrees)

    Args:
        model: MTL-MViT model.
        data_loader: DataLoader for the evaluation split.
        device: Torch device.
        epoch: Current epoch number (for logging only).

    Returns:
        dict of metric_name -> float.
    """
    from collections import defaultdict
    from src.evaluation.decoder_oracle_bound import event_f1
    from src.evaluation.evaluate import compute_det_metrics_extended, nms_numpy

    model.eval()
    _prefix = f" (epoch {epoch})" if epoch else ""
    logger.info("=" * 60)
    logger.info("Starting validation evaluation%s", _prefix)

    # Stream-through accumulators
    psr_pred_logits: list[np.ndarray] = []
    psr_labels_list: list[np.ndarray] = []
    psr_rec_ids: list[str] = []
    psr_frame_nums: list[list[int]] = []

    act_top1_correct = 0
    act_top5_correct = 0
    act_total = 0

    pose_fwd_maes: list[float] = []
    pose_up_maes: list[float] = []

    det_presence_preds: list[np.ndarray] = []
    det_presence_targets: list[np.ndarray] = []
    # mAP accumulators (DFL-decode + NMS per batch)
    det_map_pred_boxes: list[np.ndarray] = []
    det_map_pred_scores: list[np.ndarray] = []
    det_map_pred_labels: list[np.ndarray] = []
    det_map_gt_boxes: list[np.ndarray] = []
    det_map_gt_labels: list[np.ndarray] = []

    n_batches = 0
    t_start = time.time()

    for batch in data_loader:
        images = batch[0].to(device, non_blocking=True)
        targets_raw = batch[1]

        # Move tensors to device
        targets: dict = {}
        for k, v in targets_raw.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                targets[k] = {
                    sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv
                    for sk, sv in v.items()
                }
            else:
                targets[k] = v

        B = images.size(0)

        # Normalize and permute (same as training)
        images_f = images.float() / 255.0
        _mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
        _std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
        images_f = (images_f - _mean) / _std
        images_f = images_f.permute(0, 2, 1, 3, 4).contiguous()

        outputs = model(images_f)

        # -- PSR --
        psr_logits = outputs["psr_logits"]  # [B, 16, 11]
        psr_gt = targets["psr_labels"]      # [B, 16, 11]
        psr_pred_logits.append(psr_logits.cpu().numpy())
        psr_labels_list.append(psr_gt.cpu().numpy())
        for i in range(B):
            meta = targets["metadata"][i] if i < len(targets["metadata"]) else {}
            rid = meta.get("recording_id", f"b{n_batches}_{i}")
            fnums = meta.get("frame_nums", list(range(16)))
            psr_rec_ids.append(str(rid))
            psr_frame_nums.append(
                [int(f) for f in (fnums if isinstance(fnums, (list, tuple)) else [fnums])]
            )

        # -- Activity: top-1 and top-5 --
        act_logits = outputs["activity"]  # [B, 75]
        act_gt = targets["activity"]      # [B]
        act_mask = targets.get("activity_mask")
        if act_mask is not None:
            act_valid = act_mask.bool()
        else:
            act_valid = act_gt >= 0

        if act_valid.any():
            act_preds = act_logits[act_valid]
            act_gts = act_gt[act_valid]
            act_top1_correct += (act_preds.argmax(dim=1) == act_gts).sum().item()
            _, top5 = act_preds.topk(5, dim=1)
            act_top5_correct += top5.eq(act_gts.unsqueeze(1)).any(dim=1).sum().item()
            act_total += act_valid.sum().item()

        # -- Detection: presence BCE (lightweight) --
        det_outputs = outputs["detection"]
        det_list = targets.get("detection", [])

        level_presence = []
        for level_name in ("P2", "P3", "P4", "P5"):
            if level_name not in det_outputs:
                continue
            cls_logits = det_outputs[level_name]["cls_logits"]  # [B, 24, H, W]
            cls_sig = torch.sigmoid(cls_logits)
            level_presence.append(cls_sig.amax(dim=(2, 3)))     # [B, 24]

        if level_presence:
            pred_presence = torch.stack(level_presence, dim=0).amax(dim=0)
        else:
            pred_presence = torch.zeros(B, 24, device=device)

        gt_presence = torch.zeros(B, 24, device=device)
        for b in range(B):
            det_item = det_list[b] if isinstance(det_list[b], dict) else {}
            labels = det_item.get("labels")
            if labels is not None and labels.numel() > 0:
                for lbl_idx in range(labels.size(0)):
                    lbl = int(labels[lbl_idx].item())
                    if 0 <= lbl < 24:
                        gt_presence[b, lbl] = 1.0

        det_presence_preds.append(pred_presence.cpu().numpy())
        det_presence_targets.append(gt_presence.cpu().numpy())

        # -- Detection: mAP@0.5 / mAP@0.5:0.95 (DFL decode + NMS) --
        _DFL_REG_MAX = 16
        _strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
        level_boxes: list[torch.Tensor] = []
        level_scores: list[torch.Tensor] = []
        level_labels: list[torch.Tensor] = []

        for level_name in ("P2", "P3", "P4", "P5"):
            if level_name not in det_outputs:
                continue
            cls_logits_lvl = det_outputs[level_name]["cls_logits"]  # [B, 24, H, W]
            reg_preds_lvl = det_outputs[level_name]["reg_preds"]    # [B, 64, H, W]
            _B, _, H, W = cls_logits_lvl.shape
            stride = _strides[level_name]

            # Decode DFL: [B, 64, H, W] -> [B, 4, 16, H, W] -> softmax -> weighted sum
            reg_dist = reg_preds_lvl.view(_B, 4, _DFL_REG_MAX, H, W)
            proj = torch.arange(_DFL_REG_MAX, device=device).float().view(1, 1, _DFL_REG_MAX, 1, 1)
            decoded = (reg_dist.softmax(dim=2) * proj).sum(dim=2)  # [B, 4, H, W]

            # Grid cell centers
            ys = torch.arange(H, device=device)
            xs = torch.arange(W, device=device)
            cell_cx = xs.float() * stride + stride / 2.0
            cell_cy = ys.float() * stride + stride / 2.0

            # Deltas to absolute xyxy (matching detection_loss() decode)
            pred_x1 = cell_cx.view(1, 1, W) - decoded[:, 0:1] * stride  # [B, 1, H, W]
            pred_y1 = cell_cy.view(1, H, 1) - decoded[:, 1:2] * stride
            pred_x2 = cell_cx.view(1, 1, W) + decoded[:, 2:3] * stride
            pred_y2 = cell_cy.view(1, H, 1) + decoded[:, 3:4] * stride
            pred_abs = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # [B, 4, H, W]
            try:
                boxes_lvl = pred_abs.permute(0, 2, 3, 1).reshape(_B, -1, 4)         # [B, H*W, 4]
                scores_lvl = torch.sigmoid(cls_logits_lvl).permute(0, 2, 3, 1).reshape(_B, -1, C.NUM_DET_CLASSES)  # [B, H*W, 24]
                max_scores_lvl = scores_lvl.amax(dim=-1)  # [B, H*W]
                labels_lvl = scores_lvl.argmax(dim=-1)    # [B, H*W]
                level_boxes.append(boxes_lvl)
                level_scores.append(max_scores_lvl)
                level_labels.append(labels_lvl)
            except RuntimeError as e:
                # F.interpolate / 4D-5D permute edge cases — skip this level
                logger.debug("  Box decode skipped at %s: %s", level_name, e)
                continue

        if level_boxes:
            all_boxes = torch.cat(level_boxes, dim=1)    # [B, N_total, 4]
            all_scores = torch.cat(level_scores, dim=1)  # [B, N_total]
            all_labels = torch.cat(level_labels, dim=1)   # [B, N_total]
        else:
            all_boxes = torch.zeros(B, 0, 4, device=device)
            all_scores = torch.zeros(B, 0, device=device)
            all_labels = torch.zeros(B, 0, dtype=torch.long, device=device)

        for b in range(B):
            # Score filter
            keep = all_scores[b] > C.DET_EVAL_SCORE_THRESH
            boxes_np = all_boxes[b, keep].cpu().numpy()
            scores_np = all_scores[b, keep].cpu().numpy()
            labels_np = all_labels[b, keep].cpu().numpy()

            # Per-class NMS
            final_boxes, final_scores, final_labels = [], [], []
            for c in range(C.NUM_DET_CLASSES):
                c_mask = labels_np == c
                if not c_mask.any():
                    continue
                c_boxes = boxes_np[c_mask]
                c_scores = scores_np[c_mask]
                c_keep = nms_numpy(c_boxes, c_scores, C.DET_EVAL_NMS_IOU_THRESH)
                final_boxes.append(c_boxes[c_keep])
                final_scores.append(c_scores[c_keep])
                final_labels.append(np.full(len(c_keep), c, dtype=np.int64))

            if final_boxes:
                det_map_pred_boxes.append(np.concatenate(final_boxes, axis=0))
                det_map_pred_scores.append(np.concatenate(final_scores, axis=0))
                det_map_pred_labels.append(np.concatenate(final_labels, axis=0))
            else:
                det_map_pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
                det_map_pred_scores.append(np.zeros(0, dtype=np.float32))
                det_map_pred_labels.append(np.zeros(0, dtype=np.int64))

            # GT for this image
            det_item = det_list[b] if isinstance(det_list[b], dict) else {}
            gt_boxes_np = det_item.get("boxes", torch.zeros(0, 4, device=device)).cpu().numpy()
            gt_labels_np = det_item.get("labels", torch.zeros(0, dtype=torch.long, device=device)).cpu().numpy()
            det_map_gt_boxes.append(gt_boxes_np.reshape(-1, 4) if gt_boxes_np.size > 0 else np.zeros((0, 4), dtype=np.float32))
            det_map_gt_labels.append(gt_labels_np.ravel().astype(np.int64) if gt_labels_np.size > 0 else np.zeros(0, dtype=np.int64))

        # -- Pose: angular MAE --
        hp = targets.get("head_pose")  # [B, 16, 9]
        if hp is not None:
            hp_6d = hp[:, hp.size(1) // 2, :6]  # [B, 6] middle frame
            fwd_pred, up_pred = renormalize_pose(outputs["pose_6d"])
            fwd_gt = F.normalize(hp_6d[:, :3], dim=1)
            up_gt = F.normalize(hp_6d[:, 3:], dim=1)

            cos_fwd = (fwd_pred * fwd_gt).sum(dim=1).clamp(-1.0, 1.0)
            cos_up = (up_pred * up_gt).sum(dim=1).clamp(-1.0, 1.0)

            pose_fwd_maes.extend(torch.rad2deg(torch.acos(cos_fwd)).cpu().numpy().tolist())
            pose_up_maes.extend(torch.rad2deg(torch.acos(cos_up)).cpu().numpy().tolist())

        n_batches += 1
        if n_batches % 500 == 0:
            logger.info("  Eval batch %d", n_batches)

    # =====================================================================
    # Aggregate metrics
    # =====================================================================
    metrics: dict = {"eval_batches": n_batches, "eval_time_s": time.time() - t_start}

    # -- Activity --
    if act_total > 0:
        metrics["act_top1"] = act_top1_correct / act_total
        metrics["act_top5"] = act_top5_correct / act_total
        metrics["act_n_total"] = act_total
        logger.info("  Activity: top1=%.4f  top5=%.4f  (n=%d)",
                     metrics["act_top1"], metrics["act_top5"], act_total)
    else:
        metrics["act_top1"] = 0.0
        metrics["act_top5"] = 0.0

    # -- Detection presence BCE --
    if det_presence_preds:
        all_pred = np.concatenate(det_presence_preds, axis=0)   # [N, 24]
        all_gt = np.concatenate(det_presence_targets, axis=0)  # [N, 24]
        presence_bce = float(F.binary_cross_entropy(
            torch.from_numpy(all_pred), torch.from_numpy(all_gt), reduction="mean"
        ).item())
        pred_bin = (all_pred > 0.5).astype(np.float32)
        presence_acc = float((pred_bin == all_gt).mean())
        metrics["det_presence_bce"] = presence_bce
        metrics["det_presence_acc"] = presence_acc
        logger.info("  Detection: presence_bce=%.4f  presence_acc=%.4f",
                     presence_bce, presence_acc)

    # -- Detection mAP@0.5 / mAP@0.5:0.95 --
    if det_map_pred_boxes:
        det_map_result = compute_det_metrics_extended(
            det_map_pred_boxes, det_map_pred_scores, det_map_pred_labels,
            det_map_gt_boxes, det_map_gt_labels,
        )
        metrics["det_mAP50"] = det_map_result["det_mAP50"]
        metrics["det_mAP_50_95"] = det_map_result["det_mAP_50_95"]
        metrics["det_mAP50_pc"] = det_map_result["det_mAP50_pc"]
        logger.info("  Detection mAP: mAP50=%.4f  mAP50:95=%.4f  mAP50_pc=%.4f",
                     det_map_result["det_mAP50"], det_map_result["det_mAP_50_95"],
                     det_map_result["det_mAP50_pc"])

    # -- PSR transition eval (Doc 175 section 7.2) --
    if psr_pred_logits:
        all_logits = np.concatenate(psr_pred_logits, axis=0)  # [N, 16, 11]
        all_labels = np.concatenate(psr_labels_list, axis=0)  # [N, 16, 11]

        rec_data: dict = defaultdict(lambda: {"pred": [], "label": [], "frame": []})
        for i in range(len(psr_rec_ids)):
            rid = psr_rec_ids[i]
            fnums = psr_frame_nums[i]
            for t in range(all_logits.shape[1]):
                rec_data[rid]["pred"].append(all_logits[i, t])
                rec_data[rid]["label"].append(all_labels[i, t])
                rec_data[rid]["frame"].append(fnums[t] if t < len(fnums) else t)

        event_f1s: list[float] = []
        poss: list[float] = []
        taus: list[float] = []

        for rid, arrs in rec_data.items():
            _frames = np.array(arrs["frame"], dtype=np.int64)
            _sort = np.argsort(_frames)
            _vp_raw = np.array(arrs["pred"])[_sort]   # [T, 11]
            _vl_raw = np.array(arrs["label"])[_sort]  # [T, 11]

            # Sigmoid -> binary at 0.5 threshold
            _vp_bin = (1.0 / (1.0 + np.exp(-_vp_raw)) > 0.5).astype(np.int32)

            # Keep only valid frames
            _valid = _vl_raw.max(axis=1) >= 0
            _vp = _vp_bin[_valid]
            _vl = _vl_raw[_valid]
            if len(_vp) < 2:
                continue

            # Transition events: 0 -> 1
            _pred_tr = np.clip(_vp[1:] - _vp[:-1], a_min=0, a_max=None)
            _gt_tr = np.clip(_vl[1:] - _vl[:-1], a_min=0, a_max=None)
            _valid_tr = _vl[1:].max(axis=1) >= 0
            _pv = _pred_tr[_valid_tr]
            _gv = _gt_tr[_valid_tr]

            _ef1 = event_f1(_pv, _gv, tol=3)
            event_f1s.append(_ef1)
            poss.append(_compute_pos(_pv, _gv))
            _tau = _compute_tau(_pv, _gv, tol=3)
            if not np.isnan(_tau):
                taus.append(_tau)

        if event_f1s:
            metrics["psr_event_f1_at_3"] = float(np.mean(event_f1s))
            metrics["psr_pos"] = float(np.mean(poss)) if poss else 0.0
            metrics["psr_tau_frames"] = float(np.nanmean(taus)) if taus else float("nan")
            logger.info("  PSR: event_f1@+-3=%.4f  POS=%.4f  tau=%.2f  (n_recs=%d)",
                         metrics["psr_event_f1_at_3"],
                         metrics["psr_pos"],
                         metrics.get("psr_tau_frames", float("nan")),
                         len(event_f1s))

    # -- Pose angular MAE with bootstrap CI --
    if pose_fwd_maes:
        fwd_arr = np.array(pose_fwd_maes)
        up_arr = np.array(pose_up_maes)
        metrics["pose_fwd_mae"] = float(np.mean(fwd_arr))
        metrics["pose_up_mae"] = float(np.mean(up_arr))
        metrics["pose_n"] = len(fwd_arr)

        # Bootstrap 95% CI (1000 resamples)
        _rng = np.random.default_rng(42)
        _n_boot = 1000
        _n_samp = len(fwd_arr)
        _fwd_boot = np.array([np.mean(_rng.choice(fwd_arr, _n_samp)) for _ in range(_n_boot)])
        _up_boot = np.array([np.mean(_rng.choice(up_arr, _n_samp)) for _ in range(_n_boot)])
        metrics["pose_fwd_mae_ci95"] = [
            float(np.percentile(_fwd_boot, 2.5)),
            float(np.percentile(_fwd_boot, 97.5)),
        ]
        metrics["pose_up_mae_ci95"] = [
            float(np.percentile(_up_boot, 2.5)),
            float(np.percentile(_up_boot, 97.5)),
        ]
        logger.info("  Pose: fwd_MAE=%.2fdeg [%.2f, %.2f]  up_MAE=%.2fdeg [%.2f, %.2f]  (n=%d)",
                     metrics["pose_fwd_mae"], metrics["pose_fwd_mae_ci95"][0],
                     metrics["pose_fwd_mae_ci95"][1],
                     metrics["pose_up_mae"], metrics["pose_up_mae_ci95"][0],
                     metrics["pose_up_mae_ci95"][1], metrics["pose_n"])

    logger.info("Validation complete (%.1fs, %d batches)", metrics["eval_time_s"], n_batches)
    logger.info("=" * 60)

    model.train()
    return metrics


# ===========================================================================
# Main training loop
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MTL-All with MViTv2-S shared backbone"
    )
    parser.add_argument("--plumbing", action="store_true", help="1 epoch, subset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head LR")
    parser.add_argument("--lr-log-var", type=float, default=1e-3, help="Log var LR")
    parser.add_argument("--hp-prec-cap", action="store_true", default=True, help="Cap pose precision")
    parser.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=True, help="PCGrad gradient surgery (default: on; use --no-pcgrad to disable)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Run validation evaluation every N epochs (default: 5)")
    parser.add_argument("--eval-split", type=str, default="val",
                        help="Split for model selection (val) or final eval (test)")
    parser.add_argument("--test-only", action="store_true",
                        help="Load --resume checkpoint and evaluate on test split only")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--measure-efficiency", action="store_true",
        help="Run efficiency measurement (FLOPs, FPS, VRAM) and exit",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0,
                        help="Cap batches per epoch (0 = full epoch; use 200 for fast smoke)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for ~2x speedup (PyTorch 2.0+)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Args: %s", vars(args))

    # ── Split discipline (Doc 175 §7.1) ──
    if args.test_only:
        require_split("test", allow_test_only=True)
    else:
        assert args.eval_split == "val", \
            f"Model selection must use 'val' split (got '{args.eval_split}') per Doc 175 §7.1"

    # ── Standalone efficiency measurement ──
    if args.measure_efficiency:
        logger.info("Running standalone efficiency measurement...")
        measure_efficiency(args)
        return

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Building train dataset (sequence_mode=True, T=16)...")
    # Patch config for our training setup
    C.STAGED_TRAINING = False
    C.KENDALL_STAGED_TRAINING = False
    C.MIXED_PRECISION = True
    C.AMP_DTYPE = "bf16"

    train_ds = IndustRealMultiTaskDataset(
        split="train",
        img_size=(224, 224),
        augment=True,
        sequence_mode=True,
        sequence_length=16,
    )
    logger.info("Train samples: %d", len(train_ds))

    sampler = torch.utils.data.WeightedRandomSampler(
        train_ds.get_sampler_weights() if hasattr(train_ds, "get_sampler_weights")
        else torch.ones(len(train_ds)),
        num_samples=len(train_ds), replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn_sequences,
        drop_last=True,
    )

    # ── Eval split for model selection ────────────────────────────────────
    eval_ds = IndustRealMultiTaskDataset(
        split=args.eval_split,
        img_size=(224, 224),
        augment=False,
        sequence_mode=True,
        sequence_length=16,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_sequences,
        drop_last=False,
    )
    logger.info("Eval samples (%s): %d", args.eval_split, len(eval_ds))

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building MTL-MViT model...")
    model = MTLMViTModel(num_act_classes=getattr(C, "NUM_ACT_OUTPUTS", 75)).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Params: %.1fM total, %.1fM trainable", total_params, trainable_params)

    # ── torch.compile for ~2x speedup ──
    if args.compile:
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="default")
            logger.info("Model compiled.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e} — continuing without compile")

    # ── Test-only evaluation ──────────────────────────────────────────────
    if args.test_only:
        if args.resume is None:
            logger.error("--test-only requires --resume <checkpoint.pt>")
            sys.exit(1)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)",
                    args.resume, ckpt.get("epoch", "?"))

        test_ds = IndustRealMultiTaskDataset(
            split="test",
            img_size=(224, 224),
            augment=False,
            sequence_mode=True,
            sequence_length=16,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_sequences,
            drop_last=False,
        )
        test_metrics = evaluate(model, test_loader, device, epoch=None)
        logger.info("Test metrics: %s", test_metrics)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump({"test_metrics": test_metrics, "config": vars(args)}, f, indent=2, default=str)
        logger.info("Test-only evaluation complete. Output: %s", output_dir)
        return

    # ── Kendall log_vars ──────────────────────────────────────────────────
    log_vars = nn.ParameterDict({
        name: nn.Parameter(torch.tensor([-0.5], device=device))
        for name in ["det", "act", "psr", "pose"]
    })
    logger.info("Log vars initialized to -0.5")

    # ── Optimizer ─────────────────────────────────────────────────────────
    param_groups = [
        {"params": model.feature_pyramid.backbone.parameters(), "lr": args.lr_backbone, "weight_decay": 0.05},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and "log_var" not in n], "lr": args.lr_head, "weight_decay": 0.05},
        {"params": log_vars.parameters(), "lr": args.lr_log_var, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    best_act_top1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_act_top1 = ckpt.get("best_act_top1", 0.0)
        log_vars = ckpt.get("log_vars", log_vars)
        logger.info("Resumed from epoch %d", start_epoch)

    # ── AMP scaler ─────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(device.type, enabled=True)

    # ── Activity class weights (Doc 175 §4) ────────────────────────────────
    act_class_weights = compute_activity_class_weights(train_ds, num_classes=len(train_ds.class_counts))
    act_class_weights = act_class_weights.to(device)
    logger.info("Activity class weights computed — shape=%s, device=%s",
                act_class_weights.shape, act_class_weights.device)

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info("Starting training (%d epochs)...", args.epochs)
    metrics_log = {"train_metrics": [], "config": vars(args)}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        epoch_metrics = {
            "loss": 0, "loss_det": 0, "loss_act": 0, "loss_psr": 0, "loss_pose": 0,
            **{f"loss_psr_c{i}": 0 for i in range(11)},
        }
        n_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            # Cap batches per epoch (--max-batches-per-epoch) for fast smoke runs
            if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
                break

            images = batch[0].to(device, non_blocking=True)  # [B, T, 3, H, W]
            targets = {}
            for k, v in batch[1].items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(device, non_blocking=True)
                elif isinstance(v, dict):
                    targets[k] = {sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv
                                  for sk, sv in v.items()}
                else:
                    targets[k] = v

            # images shape: [B, T=16, 3, H=224, W=224]
            images = images.float() / 255.0
            mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
            images = (images - mean) / std

            # Permute for MViTv2: [B, T, C, H, W] → [B, C, T, H, W]
            images = images.permute(0, 2, 1, 3, 4).contiguous()

            # Gradient accumulation: only step optimizer on boundary
            is_accum_boundary = ((batch_idx + 1) % args.grad_accum_steps == 0) or \
                                (batch_idx + 1 == len(train_loader))
            do_step = is_accum_boundary

            step_metrics = train_step(
                model, images, targets, log_vars, optimizer, scaler,
                hp_prec_cap=args.hp_prec_cap,
                pcgrad=args.pcgrad,
                act_class_weights=act_class_weights,
                do_step=do_step,
            )

            for k in epoch_metrics:
                epoch_metrics[k] += step_metrics.get(k, 0)
            n_steps += 1

            if batch_idx % 100 == 0:
                logger.info(
                    "  [batch %5d/%d accum=%d/%d] loss=%.4f det=%.4f act=%.4f psr=%.4f pose=%.4f",
                    batch_idx, len(train_loader),
                    (batch_idx % args.grad_accum_steps) + 1, args.grad_accum_steps,
                    step_metrics.get("loss", 0),
                    step_metrics.get("loss_det", 0),
                    step_metrics.get("loss_act", 0),
                    step_metrics.get("loss_psr", 0),
                    step_metrics.get("loss_pose", 0),
                )
                # Force flush for nohup log visibility
                for handler in logging.getLogger().handlers:
                    handler.flush()

            if args.plumbing and batch_idx >= 10:
                break

        # Average
        avg_metrics = {k: v / max(n_steps, 1) for k, v in epoch_metrics.items()}
        avg_metrics["epoch"] = epoch
        avg_metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics_log["train_metrics"].append(avg_metrics)

        # Log
        dt = time.time() - t0
        lv = {k: step_metrics.get(k, 0) for k in ["log_var_det", "log_var_act", "log_var_psr", "log_var_pose"]}
        logger.info(
            "Epoch %3d/%d | loss=%.4f det=%.4f act=%.4f psr=%.4f pose=%.4f | "
            "lv=[%.2f,%.2f,%.2f,%.2f] | lr=%.2e | %.1fs",
            epoch, args.epochs,
            avg_metrics["loss"], avg_metrics["loss_det"], avg_metrics["loss_act"],
            avg_metrics["loss_psr"], avg_metrics["loss_pose"],
            lv.get("log_var_det", 0), lv.get("log_var_act", 0),
            lv.get("log_var_psr", 0), lv.get("log_var_pose", 0),
            avg_metrics["lr"], dt,
        )
        # Per-component PSR loss breakdown (Doc 175 §4)
        _pcs = [avg_metrics.get(f"loss_psr_c{i}", 0) for i in range(11)]
        logger.info("  PSR comp: [%s]", " ".join(f"{v:.4f}" for v in _pcs))

        scheduler.step()

        # ── Evaluate on eval split (Doc 175 section 7.2) ─────────────────
        if epoch % args.eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, device, epoch=epoch)
            eval_metrics["epoch"] = epoch
            metrics_log.setdefault("val_metrics", []).append(eval_metrics)
            act1 = eval_metrics.get("act_top1", 0.0)
            act5 = eval_metrics.get("act_top5", 0.0)
            psr = eval_metrics.get("psr_event_f1_at_3", 0.0)
            det = eval_metrics.get("det_presence_bce", 0.0)
            fwd = eval_metrics.get("pose_fwd_mae", 0.0)
            up = eval_metrics.get("pose_up_mae", 0.0)
            logger.info(
                "  Eval (%s): act_top1=%.4f act_top5=%.4f psr_f1=%.4f "
                "det_bce=%.4f pose_fwd=%.2fdeg pose_up=%.2fdeg",
                args.eval_split, act1, act5, psr, det, fwd, up,
            )

            # Best model based on activity top-1 (Doc 175 section 7.2)
            if act1 > best_act_top1:
                best_act_top1 = act1
                best_ckpt = output_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "log_vars": log_vars,
                    "best_act_top1": best_act_top1,
                    "best_val_loss": best_val_loss,
                    "val_metrics": eval_metrics,
                }, best_ckpt)
                logger.info("  New best activity top-1: %.4f (saved: %s)", best_act_top1, best_ckpt)

        # Save checkpoint
        if epoch % 10 == 0 or epoch == 1:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "log_vars": log_vars,
                "best_val_loss": best_val_loss,
                "metrics": avg_metrics,
            }
            ckpt_path = output_dir / f"epoch_{epoch:04d}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

        # Save latest
            latest = output_dir / "latest.pt"
            torch.save({**ckpt, "epoch": epoch}, latest)

        if args.plumbing and epoch >= 1:
            logger.info("Plumbing test complete.")
            break

    # ── Final evaluation on test split ────────────────────────────────────
    require_split("test", allow_test_only=True)
    logger.info("Running final evaluation on test split...")
    test_ds = IndustRealMultiTaskDataset(
        split="test",
        img_size=(224, 224),
        augment=False,
        sequence_mode=True,
        sequence_length=16,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_sequences,
        drop_last=False,
    )
    test_metrics = evaluate(model, test_loader, device, epoch=None)
    logger.info("Test metrics: %s", test_metrics)

    # Save final metrics
    metrics_log["test_metrics"] = test_metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2, default=str)

    logger.info("Training complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
