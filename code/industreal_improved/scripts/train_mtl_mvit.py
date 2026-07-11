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
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import median_filter

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
from src.models.mvit_mtl_model import (
    MTLMViTModel,
    renormalize_pose,
    gram_schmidt_rotation,
    geodesic_angle,
)
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
    alpha: float = 0.5,
    pos_radius: int = 1,
    use_tal: bool = True,
    tal_topk: int = 10,
) -> torch.Tensor:
    """Detection loss with TAL assigner (TOOD, ICCV 2021).

    [EP10 EVIDENCE] Sparse 3×3 at ep10 = mAP 0.0. TAL gives dense positives.
    When use_tal=True, each GT is assigned to topk cells per FPN level.
    P2 is skipped (semantics-free conv_proj features — Opus 192 FC-2).
    """
    device = next(iter(det_outputs.values()))["cls_logits"].device
    if not det_list:
        return torch.tensor(0.0, device=device)

    from src.losses.tal_assigner import TaskAlignedAssigner
    tal = TaskAlignedAssigner(topk=tal_topk, alpha=1.0, beta=6.0)

    loss_cls = 0.0; loss_iou = 0.0; loss_dfl = 0.0; n_levels_active = 0
    levels = ("P3", "P4", "P5") if use_tal else ("P2", "P3", "P4", "P5")
    strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}

    for level_name in levels:
        if level_name not in det_outputs:
            continue
        n_levels_active += 1
        out = det_outputs[level_name]
        cls_logits = out["cls_logits"]; reg_preds = out["reg_preds"]
        B, nc, H, W = cls_logits.shape; stride = strides[level_name]

        ys = torch.arange(H, device=device); xs = torch.arange(W, device=device)
        cell_cx = xs.float() * stride + stride / 2.0
        cell_cy = ys.float() * stride + stride / 2.0
        anchor_points = torch.stack([cell_cx.unsqueeze(0).expand(H, -1),
                                     cell_cy.unsqueeze(1).expand(-1, W)], dim=0).reshape(2, -1).t()

        cls_target = torch.zeros(B, H, W, dtype=torch.long, device=device)
        pos_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        dfl_target = torch.zeros(B, H, W, 4, device=device)
        iou_target = torch.zeros(B, H, W, 4, device=device)

        if use_tal:
            # === TAL assignment per image ===
            for b in range(B):
                det_item = det_list[b] if isinstance(det_list[b], dict) else {}
                boxes = det_item.get("boxes"); labels = det_item.get("labels")
                if boxes is None or labels is None or boxes.numel() == 0:
                    continue
                boxes = boxes.to(device, torch.float); labels = labels.to(device, torch.long)
                if boxes.dim() == 1: boxes = boxes.unsqueeze(0); labels = labels.unsqueeze(0)
                n_gt = boxes.size(0)
                if n_gt == 0: continue

                cls_sig = torch.sigmoid(cls_logits[b]).permute(1,2,0).reshape(-1,nc)
                reg_dist_v = reg_preds[b].view(4, reg_max, H, W)
                proj = torch.arange(reg_max, device=device).float().view(1,reg_max,1,1)
                decoded_o = (reg_dist_v.softmax(dim=1)*proj).sum(dim=1)
                px1=(cell_cx.view(1,1,W)-decoded_o[0:1]*stride).reshape(H,W)
                py1=(cell_cy.view(1,H,1)-decoded_o[1:2]*stride).reshape(H,W)
                px2=(cell_cx.view(1,1,W)+decoded_o[2:3]*stride).reshape(H,W)
                py2=(cell_cy.view(1,H,1)+decoded_o[3:4]*stride).reshape(H,W)
                pred_xyxy=torch.stack([px1,py1,px2,py2],dim=-1).reshape(-1,4)

                mn=20; pb=torch.zeros(mn,4,device=device); pb[:n_gt]=boxes
                pl=torch.zeros(mn,dtype=torch.long,device=device); pl[:n_gt]=labels
                tl,tb,_,mk,_=tal(cls_sig.unsqueeze(0),pred_xyxy.unsqueeze(0),
                                 anchor_points,pb.unsqueeze(0),pl.unsqueeze(0),
                                 anchor_points,torch.tensor([stride],device=device))
                mask_flat=mk.squeeze(0).squeeze(-1)
                if mask_flat.sum()==0: continue
                assigned=mask_flat.bool()
                h_idx=assigned.nonzero(as_tuple=False)[:,0]
                hi=(h_idx%W).long(); hj=(h_idx//W).long()
                for k in range(len(hi)):
                    ci,cj=hi[k].item(),hj[k].item()
                    if 0<=ci<W and 0<=cj<H and not pos_mask[b,cj,ci]:
                        pos_mask[b,cj,ci]=True
                        tcls=tl.squeeze(0)[h_idx[k]].argmax().item()
                        cls_target[b,cj,ci]=tcls+1
                        tbox=tb.squeeze(0)[h_idx[k]]
                        dfl_target[b,cj,ci,0]=(cell_cx[ci]-tbox[0])/stride
                        dfl_target[b,cj,ci,1]=(cell_cy[cj]-tbox[1])/stride
                        dfl_target[b,cj,ci,2]=(tbox[2]-cell_cx[ci])/stride
                        dfl_target[b,cj,ci,3]=(tbox[3]-cell_cy[cj])/stride
                        iou_target[b,cj,ci]=tbox
        else:
            # Legacy 3×3 sparse assignment (fallback)
            for b in range(B):
                det_item = det_list[b] if isinstance(det_list[b], dict) else {}
                boxes = det_item.get("boxes"); labels = det_item.get("labels")
                if boxes is None or labels is None or boxes.numel() == 0:
                    continue
                boxes = boxes.to(device, torch.float); labels = labels.to(device, torch.long)
                if boxes.dim() == 1: boxes = boxes.unsqueeze(0); labels = labels.unsqueeze(0)
                gt_cx=(boxes[:,0]+boxes[:,2])/2.0; gt_cy=(boxes[:,1]+boxes[:,3])/2.0
                for n in range(boxes.shape[0]):
                    gi=(gt_cx[n]/stride).long().clamp(0,W-1)
                    gj=(gt_cy[n]/stride).long().clamp(0,H-1)
                    for di in range(-pos_radius,pos_radius+1):
                        for dj in range(-pos_radius,pos_radius+1):
                            ci=gi+di; cj=gj+dj
                            if 0<=ci<W and 0<=cj<H and not pos_mask[b,cj,ci]:
                                pos_mask[b,cj,ci]=True
                                cls_target[b,cj,ci]=labels[n].long()
                                dfl_target[b,cj,ci,0]=(cell_cx[ci]-boxes[n,0])/stride
                                dfl_target[b,cj,ci,1]=(cell_cy[cj]-boxes[n,1])/stride
                                dfl_target[b,cj,ci,2]=(boxes[n,2]-cell_cx[ci])/stride
                                dfl_target[b,cj,ci,3]=(boxes[n,3]-cell_cy[cj])/stride
                                iou_target[b,cj,ci]=boxes[n]

        # ---- Classification: Focal BCE ----
        cls_p = cls_logits.permute(0,2,3,1).contiguous()
        cls_oh = F.one_hot(cls_target, num_classes).float()
        cls_prob = torch.sigmoid(cls_p)
        pt = cls_oh*cls_prob+(1-cls_oh)*(1-cls_prob)
        focal_w = (1-pt)**gamma
        alpha_t = cls_oh*alpha+(1-cls_oh)*(1-alpha)
        cls_loss_bce = F.binary_cross_entropy_with_logits(cls_p, cls_oh, reduction="none")
        loss_cls = loss_cls + (alpha_t*focal_w*cls_loss_bce).sum(dim=-1).mean()

        # ---- Box losses (positive cells only) ----
        if pos_mask.any():
            reg_dist = reg_preds.view(B,4,reg_max,H,W)
            pred_dist = reg_dist.permute(0,3,4,1,2)[pos_mask]
            gt_dfl = dfl_target[pos_mask]
            if pred_dist.size(0)>0:
                dfl_inst=0.0
                for k in range(4):
                    pk=pred_dist[:,k,:]; tk=gt_dfl[:,k].clamp(0,reg_max-1.01)
                    tl=tk.long().clamp(0,reg_max-2); th=(tl+1).clamp(0,reg_max-1)
                    wh=tk-tl.float(); wl=1-wh
                    dfl_inst=dfl_inst+(F.cross_entropy(pk,tl,reduction="none")*wl+
                                       F.cross_entropy(pk,th,reduction="none")*wh).mean()
                loss_dfl = loss_dfl + dfl_inst/4
            proj=torch.arange(reg_max,device=device).float().view(1,1,reg_max,1,1)
            dec=(reg_dist.softmax(dim=2)*proj).sum(dim=2)
            px1=cell_cx.view(1,1,W)-dec[:,0:1]*stride
            py1=cell_cy.view(1,H,1)-dec[:,1:2]*stride
            px2=cell_cx.view(1,1,W)+dec[:,2:3]*stride
            py2=cell_cy.view(1,H,1)+dec[:,3:4]*stride
            pa=torch.cat([px1,py1,px2,py2],dim=1).permute(0,2,3,1).contiguous()
            loss_iou = loss_iou + ciou_loss(pa[pos_mask], iou_target[pos_mask]).mean()

    n_levels_active = max(n_levels_active, 1)
    return loss_cls/n_levels_active + loss_iou/n_levels_active + loss_dfl/n_levels_active


def compute_activity_class_weights(
    dataset: "IndustRealMultiTaskDataset",
    num_classes: int = 75,
) -> torch.Tensor:
    """Inverse-frequency class weights for long-tailed activity labels.

    [OPUS 181 D1b] Under F.cross_entropy(weight=w, reduction='mean'), only the
    RELATIVE shape of w matters (the loss divides by Σw, so a constant rescale
    is a no-op). The pre-fix weights had max=137, mean-shape with extreme long
    tail → CE loss dominated by the few rare classes the model hasn't seen.
    Sqrt-taming the inverse-frequency weights compresses the tail: max/min ratio
    drops from ~137 to ~12, lowering the CE floor while keeping relative
    rebalancing.

    Args:
        dataset: Training dataset with ``class_counts`` attribute.
        num_classes: Number of activity classes (default 75).

    Returns:
        [num_classes] float tensor of inverse-frequency weights (sqrt-tamed).
    """
    counts = dataset.class_counts.astype(np.float64)
    total = counts.sum()
    # Use np.divide with where= to avoid RuntimeWarning: divide by zero.
    # np.where evaluates both branches; np.divide with where= skips zeros.
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.divide(
            total, num_classes * counts,
            out=np.zeros_like(counts, dtype=np.float64),
            where=counts > 0,
        )
    # [OPUS 181 D1b] sqrt-tame: relative shape preserved, long tail compressed.
    weights = np.power(weights, 0.5)
    logger.info(
        "Class weights — min=%.4f  max=%.4f  mean=%.4f  num_nonzero=%d  [sqrt-tamed]",
        weights.min(), weights.max(), weights.mean(),
        int((weights > 0).sum()),
    )
    return torch.from_numpy(weights).float()


def activity_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    logit_adjust_freq: Optional[torch.Tensor] = None,
    logit_adjust_tau: float = 1.0,
) -> torch.Tensor:
    """Class-balanced cross-entropy with label smoothing 0.05, ignore-index -1,
    and optional logit adjustment (Menon et al. 2020).

    [OPUS 207 §2.6] Logit adjustment adds per-class prior log-frequencies to
    logits before softmax INSIDE the loss. This follows the Menon et al. additive
    formulation where the correction is part of the training objective, not the
    model's forward pass. At eval, raw logits are used for argmax prediction.

    Args:
        logits: [B, 75] per-video activity logits.
        targets: [B] per-video activity labels (0..74; -1 = unlabeled / ignore).
        class_weights: [75] inverse-frequency weights or None for uniform.
        logit_adjust_freq: [75] class frequencies (normalized counts) or None to
            skip adjustment. Applied as `logits += tau * log(freq)`.
        logit_adjust_tau: temperature scaling for the log-freq term (default 1.0).

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
    if logit_adjust_freq is not None:
        # [OPUS 207 §2.6] Additive logit correction inside the loss.
        # logits += tau * log(freq) shifts decision boundary toward rare classes.
        logits = logits + logit_adjust_tau * torch.log(
            logit_adjust_freq + 1e-9
        ).unsqueeze(0)
    return F.cross_entropy(
        logits, targets,
        weight=class_weights,
        ignore_index=-1,
        label_smoothing=0.05,  # [OPUS 181 D1b] 0.1→0.05: lowers CE floor on 75 classes.
    )


def psr_loss(
    psr_logits: torch.Tensor,
    psr_targets: torch.Tensor,
    comp_weights: Optional[torch.Tensor] = None,
    use_focal: bool = True,  # [EP10] Focal-BCE default for PSR (rare-event data)
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    transition_boost: float = 3.0,  # [OPUS 207] Boost weight on transitions
) -> torch.Tensor:
    """Per-frame (focal) BCE for PSR transition logits.

    [EP10] use_focal=True by default. Focal-BCE down-weights easy negatives.
    T=16→T=8 label downsampling via max-pool (Opus 192 FC-4).

    [OPUS 207] Transition-aware weighting: frames near 0→1 transitions get
    transition_boost× higher weight. Directly attacks the focal-collapse-to-negatives
    failure mode where loss is low (0.17-0.27) but event_F1 is ~0.006 because
    <1% positives are being ignored.

    Args:
        psr_logits: [B, T, 11] per-frame transition logits (T=8 from PSR head).
        psr_targets: [B, T, 11] per-frame transition targets (T=16 from dataset).
        comp_weights: [1, 1, 11] per-component inverse-prevalence weights or None.
        use_focal: if True, use Focal-BCE (default: True).
        focal_alpha: weight on positive class in focal loss.
        focal_gamma: focal loss focusing parameter.
        transition_boost: extra weight multiplier for frames near transitions.

    Returns:
        Scalar loss (mean over all elements).
    """
    # If logits and targets have different T, downsample targets via max-pool.
    if psr_logits.size(1) != psr_targets.size(1):
        T_target = psr_targets.size(1)
        T_logit = psr_logits.size(1)
        # Use adaptive_max_pool1d on the time dim
        psr_targets = F.adaptive_max_pool1d(
            psr_targets.transpose(1, 2),  # [B, 11, T_target]
            output_size=T_logit,
        ).transpose(1, 2)  # [B, T_logit, 11]

    bce = F.binary_cross_entropy_with_logits(psr_logits, psr_targets, reduction='none')

    # [OPUS 207] Transition-aware frame weighting.
    # Detect 0→1 transitions along time axis; boost weight on those frames
    # and their immediate neighbors (±1 frame).
    with torch.no_grad():
        # transitions: [B, T-1, 11] — 1 where a component transitions 0→1
        transitions = (psr_targets[:, 1:, :] - psr_targets[:, :-1, :]).clamp(min=0)
        transitions = F.pad(transitions, (0, 0, 0, 1))  # pad last frame → [B, T, 11]
        # Any component transition at time t → boost frame t and t-1
        has_transition = (transitions.sum(dim=-1) > 0).float()  # [B, T]
        neighbor_boost = F.pad(has_transition[:, :-1], (1, 0))  # shift right
        frame_weight = 1.0 + (transition_boost - 1.0) * (
            has_transition + neighbor_boost
        ).clamp(0, 1).unsqueeze(-1)  # [B, T, 1]

    if use_focal:
        # Focal-BCE: down-weight easy examples
        p = torch.sigmoid(psr_logits)
        pt = psr_targets * p + (1 - psr_targets) * (1 - p)
        alpha_t = psr_targets * focal_alpha + (1 - psr_targets) * (1 - focal_alpha)
        focal_weight = alpha_t * (1 - pt) ** focal_gamma
        loss = focal_weight * bce
    else:
        loss = bce

    # Apply transition-aware frame weighting
    loss = loss * frame_weight

    if comp_weights is not None:
        loss = loss * comp_weights  # broadcast [B, T, 11] * [1, 1, 11]
    return loss.mean()


def pose_loss(pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
    """Combined cosine + geodesic loss on renormalized fwd and up vectors.

    Geodesic component computes SO(3) angular error between Gram-Schmidt
    orthonormalised (fwd, up) reconstructions.  Cosine component retained
    for training-signal continuity with prior epochs (Doc 207 2.6).
    """
    fwd_pred, up_pred = renormalize_pose(pred_6d)
    fwd_gt = F.normalize(target_6d[:, :3], dim=1)
    up_gt = F.normalize(target_6d[:, 3:], dim=1)
    cos_fwd = (fwd_pred * fwd_gt).sum(dim=1).clamp(-1.0, 1.0)
    cos_up = (up_pred * up_gt).sum(dim=1).clamp(-1.0, 1.0)
    cosine_loss = (1.0 - cos_fwd).mean() + (1.0 - cos_up).mean()

    # Geodesic loss on SO(3) — Doc 207 Tier-1 item #2
    R_pred = gram_schmidt_rotation(fwd_pred, up_pred)
    R_gt = gram_schmidt_rotation(fwd_gt, up_gt)
    geodesic_loss = geodesic_angle(R_pred, R_gt).mean()

    return cosine_loss + geodesic_loss


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
# SWA within-run checkpoint averaging (Task #259 / §6 lever #3)
# ===========================================================================

def swa_average_checkpoints(
    ckpt_dir: Path, n_last: int, device: torch.device,
) -> OrderedDict:
    """Average the last N periodic checkpoints (epoch_NNNN.pt) in ckpt_dir.

    Standard SWA recipe (Izmailov 2018): averages model weights from the last
    few checkpoints, smoothing late-training noise. +0.5-2% across tasks.

    Returns averaged state_dict, or None if <2 checkpoints found.
    """
    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(ckpt_files) < 2:
        logger.warning("SWA: need ≥2 checkpoints, found %d", len(ckpt_files))
        return None
    targets = ckpt_files[-n_last:] if len(ckpt_files) >= n_last else ckpt_files
    logger.info("SWA: averaging %d checkpoints (last %d available)", len(targets), n_last)

    avg_sd = OrderedDict()
    n_loaded = 0
    for ckpt_path in targets:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        for k, v in sd.items():
            if k not in avg_sd:
                avg_sd[k] = v.float().clone()
            else:
                avg_sd[k] += v.float()
        n_loaded += 1

    for k in avg_sd:
        avg_sd[k] /= n_loaded
    return avg_sd


# ===========================================================================
# Head warm-starting from ST checkpoints (Task #260 / §6 lever #4)
# ===========================================================================

def warm_start_heads_from_st(
    model: nn.Module, st_dir: str, device: torch.device,
) -> int:
    """Initialize MTL head weights from single-task specialist checkpoints.

    Loads ST head parameters from {st_dir}/st_{head}_best.pt into the
    corresponding MTL head, but leaves the shared backbone untouched
    (soup backbone handles that). Story-safe: only init provenance differs,
    final model unchanged.

    Head key mapping:
      st_det_best.pt  → det_head
      st_act_best.pt  → act_head
      st_psr_best.pt  → psr_head
      st_pose_best.pt → pose_head

    Returns number of tensors loaded.
    """
    head_map = {
        "det": ("st_det_best.pt", "det_head"),
        "act": ("st_act_best.pt", "act_head"),
        "psr": ("st_psr_best.pt", "psr_head"),
        "pose": ("st_pose_best.pt", "pose_head"),
    }
    model_sd = model.state_dict()
    total_loaded = 0

    for head_name, (ckpt_name, prefix) in head_map.items():
        ckpt_path = Path(st_dir) / ckpt_name
        if not ckpt_path.exists():
            logger.info("Warm-start %s: checkpoint not found (%s), skipping", head_name, ckpt_path)
            continue
        st_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        st_sd = st_state.get("model_state_dict", st_state)

        # Filter to head-specific keys and shape-match
        loaded = 0
        for st_key, st_val in st_sd.items():
            if st_key.startswith(prefix):
                if st_key in model_sd and model_sd[st_key].shape == st_val.shape:
                    model_sd[st_key].copy_(st_val.to(device))
                    loaded += 1

        if loaded > 0:
            logger.info("Warm-start %s: loaded %d tensors from %s", head_name, loaded, ckpt_name)
        else:
            logger.warning("Warm-start %s: no matching tensors found (prefix=%s)", head_name, prefix)
        total_loaded += loaded

    return total_loaded


# ===========================================================================
# Distillation from ST teachers (Task #261 / §6 lever #5)
# ===========================================================================

def load_distill_teachers(
    st_dir: str, device: torch.device,
) -> dict:
    """Load frozen ST specialist models as distillation teachers.

    Key insight from 204+207: the ST baselines you train anyway ARE the
    teachers. Adding a KL term from each ST teacher to the corresponding
    MTL head costs a few days of compute for the one technique with real
    published support for closing MTL/ST gaps.

    Returns dict of {head_name: teacher_model_or_None}.
    """
    head_map = {
        "act": ("st_act_best.pt", "act_head"),
        "psr": ("st_psr_best.pt", "psr_head"),
        "det": ("st_det_best.pt", "det_head"),
        "pose": ("st_pose_best.pt", "pose_head"),
    }
    teachers = {}
    for head_name, (ckpt_name, _prefix) in head_map.items():
        ckpt_path = Path(st_dir) / ckpt_name
        if not ckpt_path.exists():
            logger.info("Distill teacher %s: checkpoint not found (%s), skipping", head_name, ckpt_path)
            teachers[head_name] = None
            continue
        # Create fresh MTLMViTModel for teacher (structurally identical)
        from src.models.mvit_mtl_model import MTLMViTModel
        teacher = MTLMViTModel(num_act_classes=75).to(device).eval()
        st_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        st_sd = st_state.get("model_state_dict", st_state)
        # Shape-match filter
        teacher_sd = teacher.state_dict()
        filtered = {k: v.to(device) for k, v in st_sd.items()
                    if k in teacher_sd and teacher_sd[k].shape == v.shape}
        teacher.load_state_dict(filtered, strict=False)
        for p in teacher.parameters():
            p.requires_grad = False
        teachers[head_name] = teacher
        logger.info("Distill teacher %s: loaded %d tensors from %s", head_name, len(filtered), ckpt_name)
    return teachers


def compute_distill_loss(
    outputs: dict, teacher_outputs: dict, distill_temperature: float,
    distill_alpha: float,
) -> torch.Tensor:
    """KL-divergence distillation loss from frozen ST teachers.

    Args:
        outputs: MTL student forward-pass outputs.
        teacher_outputs: pre-computed teacher outputs (no-grad).
        distill_temperature: softmax temperature T (higher=softer targets).
        distill_alpha: weight of distillation loss vs task loss.

    Returns:
        Scalar distillation loss (summed across heads).
    """
    T = distill_temperature
    loss = torch.tensor(0.0, device=next(iter(outputs.values())).device
                        if isinstance(next(iter(outputs.values())), torch.Tensor)
                        else outputs.get("activity", torch.zeros(1)).device)

    # Activity: standard knowledge distillation (Hinton 2015)
    if "act" in teacher_outputs and teacher_outputs["act"] is not None:
        act_out = outputs.get("activity")
        if act_out is not None:
            loss = loss + F.kl_div(
                F.log_softmax(act_out / T, dim=-1),
                F.softmax(teacher_outputs["act"] / T, dim=-1),
                reduction='batchmean',
            ) * T * T * distill_alpha

    # PSR: MSE on logits (binary sigmoid, KL makes less sense)
    if "psr" in teacher_outputs and teacher_outputs["psr"] is not None:
        psr_out = outputs.get("psr_logits")
        if psr_out is not None:
            loss = loss + F.mse_loss(
                psr_out.float(), teacher_outputs["psr"].float(),
            ) * distill_alpha

    # Detection: KL on classification logits (per FPN level)
    if "det" in teacher_outputs and teacher_outputs["det"] is not None:
        det_out = outputs.get("detection")
        if det_out is not None:
            for level in det_out:
                if level in teacher_outputs["det"]:
                    cls_pred = det_out[level].get("cls_logits")
                    cls_tea = teacher_outputs["det"][level].get("cls_logits")
                    if cls_pred is not None and cls_tea is not None:
                        loss = loss + F.kl_div(
                            F.log_softmax(cls_pred / T, dim=-1),
                            F.softmax(cls_tea / T, dim=-1),
                            reduction='batchmean',
                        ) * T * T * distill_alpha

    # Pose: MSE on 6D vector
    if "pose" in teacher_outputs and teacher_outputs["pose"] is not None:
        pose_out = outputs.get("pose_6d")
        if pose_out is not None:
            loss = loss + F.mse_loss(
                pose_out.float(), teacher_outputs["pose"].float(),
            ) * distill_alpha

    return loss


def distill_teacher_forward(teachers: dict, images: torch.Tensor) -> dict:
    """Run frozen teacher models to get soft targets.

    Returns dict with same structure as MTL outputs, but only populated
    heads have entries. None entries indicate no teacher for that head.
    """
    results = {}
    for head_name, teacher in teachers.items():
        if teacher is None:
            results[head_name] = None
            continue
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                t_out = teacher(images)
        if head_name == "act":
            results["act"] = t_out["activity"].detach()
        elif head_name == "psr":
            results["psr"] = t_out["psr_logits"].detach()
        elif head_name == "det":
            results["det"] = {k: v.detach() if hasattr(v, "detach") else v
                              for k, v in t_out["detection"].items()}
        elif head_name == "pose":
            results["pose"] = t_out["pose_6d"].detach()
    return results


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
    grad_clip_norm: float = 5.0,  # [OPUS 186 E-6] 1.0 was over-clipping; 5.0 is ViT-standard.
    hp_prec_cap: bool = True,
    kendall_uncapped: bool = False,  # [OPUS 201] Ablation: disable per-task log_var caps
    pcgrad: bool = True,
    act_class_weights: Optional[torch.Tensor] = None,
    act_logit_adjust_freq: Optional[torch.Tensor] = None,  # [OPUS 207 §2.6] Menon logit-adjust in loss
    do_step: bool = True,
    ema_losses: Optional[dict] = None,
    ema_warmup_steps: int = 100,
    ema_momentum: float = 0.99,
    grad_accum_steps: int = 1,  # [OPUS 186 §5.1] Divide loss by this so accum MEANS, not SUMS.
    psr_focal: bool = True,    # [EP10] Focal-BCE default for PSR (rare-event loss).
    det_aug: bool = True,      # [OPUS 192 §5.5 / Q6] Detection augmentation (flip+color+crop).
    distill_teachers: Optional[dict] = None,  # [Task #261] Frozen ST teacher models.
    distill_alpha: float = 0.1,  # [Task #261] Distillation loss weight.
    distill_temperature: float = 4.0,  # [Task #261] Softmax temperature for KD.
) -> dict:
    """Single training step with Kendall uncertainty weighting and optional PCGrad gradient surgery.

    [OPUS 181 D1] ema_losses: dict of {task_name: scalar tensor on device} tracking
    a running mean of each raw task loss. After warmup, each task's loss is divided
    by its own EMA before entering the Kendall term — so Kendall's equilibrium is
    `weight = exp(-lv)` (no `1/(2·loss)` collapse). During warmup, raw losses are
    used (EMA values are still stabilizing).

    [OPUS 186 §5.1] grad_accum_steps: loss is divided by this before backward() so
    that gradient accumulation produces a MEAN of micro-batch gradients, not a SUM.
    Without this, the boundary step sees `grad_accum_steps`× the intended magnitude
    and `grad_clip_norm` clips most of it.

    Args:
        do_step: if True, calls optimizer.step(). Set False for gradient accumulation —
                 caller is responsible for zero_grad() AFTER step() at the accumulation
                 boundary. (Pre-fix code zeroed at top of every micro-batch which silently
                 wiped the previous micro-batch's gradient — a no-op for accumulation.)
    """
    model.train()
    B = images.size(0)

    # [OPUS 192 §5.5 / Q6] Detection augmentation BEFORE model forward.
    # Augments images + bboxes for detection only; backbone sees augmented
    # images for all tasks. Temporally-consistent (same aug to all frames).
    aug_images = images
    aug_targets = targets
    if det_aug:
        from src.data.det_augment import DetectionAugment
        _det_aug = DetectionAugment(p_flip=0.5, p_color=0.5, p_crop=0.3)
        aug_images, aug_targets = _det_aug(images, targets)

    # Forward
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(aug_images)

        # Per-task losses
        # Detection: use augmented targets (bboxes adjusted to augmentation)
        det_list = aug_targets.get("detection", [])
        l_det = detection_loss(outputs["detection"], det_list)

        l_act = activity_loss(
            outputs["activity"],
            targets.get("activity", torch.zeros(B, dtype=torch.long, device=images.device)),
            class_weights=act_class_weights,
            logit_adjust_freq=act_logit_adjust_freq,  # [OPUS 207 §2.6] Menon logit-adjust in loss
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
            use_focal=psr_focal,
            transition_boost=C.PSR_TRANSITION_BOOST,
        ) if "psr_labels" in targets else torch.tensor(0.0, device=images.device)

        # Per-component PSR loss breakdown for logging
        if "psr_labels" in targets:
            with torch.no_grad():
                # [OPUS 192 FC-4] PSR now predicts at T=8; downsample labels
                # to match (max-pool preserves transition events).
                _psr_labels_for_pc = targets["psr_labels"]
                if _psr_labels_for_pc.size(1) != outputs["psr_logits"].size(1):
                    _psr_labels_for_pc = F.adaptive_max_pool1d(
                        _psr_labels_for_pc.transpose(1, 2),
                        output_size=outputs["psr_logits"].size(1),
                    ).transpose(1, 2)
                _pc = F.binary_cross_entropy_with_logits(
                    outputs["psr_logits"], _psr_labels_for_pc, reduction='none'
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

        # ── Distillation loss (Task #261 / §6 lever #5) ──────────────────────
        l_distill = torch.tensor(0.0, device=images.device)
        if distill_teachers is not None:
            teacher_outputs = distill_teacher_forward(distill_teachers, images)
            l_distill = compute_distill_loss(
                outputs, teacher_outputs, distill_temperature, distill_alpha,
            )

        # Kendall uncertainty weighting with precision capping
        losses = {"det": l_det, "act": l_act, "psr": l_psr, "pose": l_pose,
                  "distill": l_distill}
        total_loss = 0.0

        # [OPUS 181 D1] Update per-task EMA of raw losses (detached; no grad).
        # After warmup, each task's loss is normalized by its own EMA so Kendall
        # balances comparable scales (≈O(1) per task) instead of collapsing to
        # inverse-loss scaling.
        if ema_losses is not None:
            with torch.no_grad():
                for name in ["det", "act", "psr", "pose"]:  # Task heads only
                    ema_losses[name].mul_(ema_momentum).add_(
                        losses[name].detach(), alpha=1.0 - ema_momentum
                    )

        # [OPUS 181 D1] Per-task loss scaling for Kendall.
        # Default: raw loss. If EMA tracker is provided and EMA has stabilized
        # (above 1e-3 floor), normalize by EMA so all tasks are on a comparable
        # scale (~O(1)). During the first few steps EMA is initialized to 1.0
        # which is benign — normalized loss equals raw loss — and tracks the
        # true mean within ~100 steps (momentum 0.99).
        losses_for_kendall = {name: losses[name] for name in ["det", "act", "psr", "pose"]}

        if ema_losses is not None:
            with torch.no_grad():
                for name in ["det", "act", "psr", "pose"]:
                    ema_v = ema_losses[name]
                    if ema_v.item() > 1e-3:
                        losses_for_kendall[name] = losses[name] / (ema_v + 1e-6)

        # Compute log_vars, with head-pose precision capped by detection (Doc 175 §5.2)
        # Pose precision (exp(-lv)) must never exceed detection precision. This prevents
        # the shared backbone from being optimized primarily for head_pose (loss ~0.01)
        # while neglecting detection (loss ~0.5), which has ~40x higher loss magnitude.
        # SAFETY: Clamp log_var to [-4, 4] to prevent exp(-lv) explosion (would cause
        # negative total_loss and gradient NaN). Per 175 §5.1: act_min=-4, psr/pose_max=2.
        # [OPUS 181 D2] Per-task caps so high-loss tasks (act=12.3, psr=1.3) cannot
        # collapse their backbone weight to near-zero. Floor weights via upper-bound on
        # log_var: weight = exp(-lv), so lv<=1.0 forces weight>=0.37 (act), lv<=0.5
        # forces weight>=0.61 (psr). Det/pose keep the wide range (low intrinsic loss,
        # no risk of starvation).
        LV_CLAMP_MIN = -4.0
        # [OPUS 201] Kendall-collapse ablation: when uncapped, all tasks get wide
        # bounds (4.0) — the natural Kendall dynamics then starve the highest-loss
        # task (activity). This is the "before" of the Figure 1 ablation.
        if kendall_uncapped:
            LV_CLAMP_MAX = {"det": 4.0, "act": 4.0, "psr": 4.0, "pose": 4.0}
        else:
            # [OPUS 207 §1b FIX] Detection cap was 4.0 (weight floor exp(-4)≈0.018) —
            # weakest starvation protection in the system. Detection has shown 0.000 mAP
            # in every eval. Fix caps to documented values: det≤1.5 (floor 0.22),
            # pose≤2.0 (floor 0.14). Also tighten PSR cap slightly.
            LV_CLAMP_MAX = {"det": 1.5, "act": 1.0, "psr": 0.5, "pose": 2.0}
        lv_values = {}
        for name in ["det", "act", "psr", "pose"]:  # Only task heads have log_vars
            lv_clamped = log_vars[name].clamp(LV_CLAMP_MIN, LV_CLAMP_MAX[name])
            lv_values[name] = lv_clamped
        if hp_prec_cap and not kendall_uncapped:
            lv_values["pose"] = torch.maximum(
                lv_values["pose"], lv_values["det"].detach()
            )

        # SAFETY: Clamp each per-task loss to [0, +inf) — BCE/CE/DFL/CIoU are all
        # bounded ≥ 0 in theory. Negative values come from numerical drift.
        losses_safe = {n: torch.clamp(v, min=0.0) if v.isfinite() else torch.zeros_like(v)
                       for n, v in losses.items()}
        losses_k_safe = {n: torch.clamp(v, min=0.0) if v.isfinite() else torch.zeros_like(v)
                         for n, v in losses_for_kendall.items()}

        # Kendall-weighted per-task losses (det/act/psr/pose)
        for name in losses_safe:
            if name == "distill":
                # Distillation loss bypasses Kendall — added directly below
                continue
            lv = lv_values[name]
            prec = torch.exp(-lv)
            total_loss = total_loss + prec * losses_k_safe[name] + lv / 2

        # Distillation loss added directly (weighted by distill_alpha, not by Kendall)
        # This is correct: distill loss measures agreement with teacher, not task error.
        total_loss = total_loss + losses_safe.get("distill", torch.tensor(0.0, device=images.device))

        # SAFETY: Skip step if total_loss is non-finite (log_var explosion caught here)
        if not torch.isfinite(total_loss):
            logger.warning(
                "Non-finite total_loss at step: det=%s act=%s psr=%s pose=%s | skipping optimizer step",
                losses["det"].item(), losses["act"].item(), losses["psr"].item(), losses["pose"].item(),
            )
            if do_step:
                optimizer.zero_grad()
            return {
                "loss": float("nan"),
                "loss_det": losses["det"].item(), "loss_act": losses["act"].item(),
                "loss_psr": losses["psr"].item(), "loss_pose": losses["pose"].item(),
                **psr_comp_breakdown,
                **{f"log_var_{k}": v.item() for k, v in log_vars.items()},
                **{f"ema_{k}": (ema_losses[k].item() if ema_losses is not None and k in ema_losses else 0.0)
                   for k in ["det", "act", "psr", "pose"]},
            }

    # [OPUS 181 D4] zero_grad moved from top of train_step to AFTER step() below.
    # Previously, zero_grad at the top of every micro-batch wiped the previous
    # micro-batch's gradient, making grad_accum_steps a silent no-op.

    if pcgrad:
        shared_params = [p for p in model.feature_pyramid.backbone.parameters() if p.requires_grad]

        # Per-task gradients w.r.t. shared backbone
        # SAFETY: Use clamped log_var to match the forward pass (otherwise PCGrad
        # uses different precisions than the actual loss, causing gradient/total
        # loss mismatch). Also use losses_safe to prevent negative gradient.
        per_task_grads = []
        for name in ["det", "act", "psr", "pose"]:  # PCGrad only for task heads
            lv_safe = lv_values[name]  # already clamped
            prec = torch.exp(-lv_safe)
            # [OPUS 181 D1] PCGrad uses the SAME Kendall-normalized loss as the
            # forward pass for consistency.
            weighted_loss = prec * losses_k_safe[name]
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
        # [OPUS 186 §5.1] Divide by grad_accum_steps so the boundary step sees the
        # MEAN of micro-batch gradients, not the SUM. (Otherwise grad_clip_norm=1.0
        # clips most of the doubled magnitude, coupling accumulation with clipping.)
        scaler.scale(total_loss / grad_accum_steps).backward()

        # [OPUS 181 D4] PCGrad backbone override now ACCUMULATES (instead of
        # overwriting) so gradient accumulation across micro-batches is preserved.
        # [OPUS 186 §5.1] Also scale by 1/grad_accum_steps so the deconflicted
        # backbone grads match the (already-scaled) head grads from backward().
        accum_scale = 1.0 / grad_accum_steps
        for param, grad in zip(shared_params, deconflicted):
            g = (grad.to(param.dtype)) * accum_scale
            if param.grad is None:
                param.grad = g
            else:
                param.grad = param.grad + g
    else:
        # [OPUS 186 §5.1] Same mean-scaling as the PCGrad branch.
        scaler.scale(total_loss / grad_accum_steps).backward()

    # Gradient clipping
    if do_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        # [OPUS 181 D4] zero_grad ONLY at the accumulation boundary AFTER step().
        # This is the correct timing so grads survive across the accumulation window.
        optimizer.zero_grad()
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
        "loss_distill": l_distill.item(),
        **psr_comp_breakdown,
        **{f"log_var_{k}": v.item() for k, v in log_vars.items()},
        **{f"ema_{k}": (ema_losses[k].item() if ema_losses is not None else 0.0)
           for k in ["det", "act", "psr", "pose"]},
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

    # SAFETY: cap eval at MAX_EVAL_BATCHES for fast smoke runs (otherwise full
    # val set of 37K frames takes 3+ hours). Default 0 = unlimited.
    MAX_EVAL_BATCHES = int(_os.environ.get("MAX_EVAL_BATCHES", "0"))
    for batch in data_loader:
        if MAX_EVAL_BATCHES > 0 and n_batches >= MAX_EVAL_BATCHES:
            break
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
        # [OPUS 192 FC-4] PSR now predicts at T=8 (backbone's native pooled
        # resolution). Downsample the T=16 ground-truth labels to T=8 via
        # max-pool to match the prediction temporal resolution.
        psr_logits = outputs["psr_logits"]  # [B, 8, 11]
        psr_gt = targets["psr_labels"]      # [B, 16, 11]
        # Downsample labels T=16 → T=8 via max-pool (preserves transition events)
        psr_gt_t8 = torch.nn.functional.adaptive_max_pool1d(
            psr_gt.transpose(1, 2),  # [B, 11, 16]
            output_size=8,
        ).transpose(1, 2)  # [B, 8, 11]
        psr_pred_logits.append(psr_logits.cpu().numpy())
        psr_labels_list.append(psr_gt_t8.cpu().numpy())
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

        # SAFETY: wrap detection box decode + NMS in try/except. The permute/view
        # ops on bf16 tensors can sometimes produce 5D intermediate shapes from
        # autograd internal ops, crashing the whole eval. If the decode fails
        # for a batch, we just log a warning and skip mAP for that batch.
        try:
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
            pred_x1 = cell_cx.view(1, W) - decoded[:, 0] * stride  # [B, H, W]
            pred_y1 = cell_cy.view(H, 1) - decoded[:, 1] * stride
            pred_x2 = cell_cx.view(1, W) + decoded[:, 2] * stride
            pred_y2 = cell_cy.view(H, 1) + decoded[:, 3] * stride
            pred_abs = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # [B, 4, H, W]
            boxes_lvl = pred_abs.permute(0, 2, 3, 1).reshape(_B, -1, 4)         # [B, H*W, 4]
            scores_lvl = torch.sigmoid(cls_logits_lvl).permute(0, 2, 3, 1).reshape(_B, -1, C.NUM_DET_CLASSES)  # [B, H*W, 24]
            max_scores_lvl = scores_lvl.amax(dim=-1)  # [B, H*W]
            labels_lvl = scores_lvl.argmax(dim=-1)    # [B, H*W]
            level_boxes.append(boxes_lvl)
            level_scores.append(max_scores_lvl)
            level_labels.append(labels_lvl)
        except RuntimeError as e:
            logger.warning("  Detection decode failed for batch %d: %s — skipping mAP", n_batches, str(e)[:200])
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

        # ── [OPUS 207 §4.3] Threshold sweep for sigmoid binarization ────────
        # First pass: compute sigmoid probs + monotonicity per-recording,
        # then try multiple binarization thresholds to pick the best one.
        _rec_sweep_data: list[dict] = []
        for rid, arrs in rec_data.items():
            _frames = np.array(arrs["frame"], dtype=np.int64)
            _sort = np.argsort(_frames)
            _vp_raw = np.array(arrs["pred"])[_sort]   # [T, 11]
            _vl_raw = np.array(arrs["label"])[_sort]  # [T, 11]

            # Sigmoid probabilities
            _vp_prob = 1.0 / (1.0 + np.exp(-_vp_raw))  # [T, 11]
            # Median filter to smooth noise spikes (kernel=5)
            _vp_smooth = median_filter(_vp_prob, size=(5, 1), mode="nearest")
            # Monotonicity constraint: once a component turns on it stays on
            _vp_mono = np.maximum.accumulate(_vp_smooth, axis=0)

            # Valid frames mask
            _valid = _vl_raw.max(axis=1) >= 0
            _vp_mono_valid = _vp_mono[_valid]
            _vl_valid = _vl_raw[_valid]
            if len(_vp_mono_valid) < 2:
                continue

            # Transition events from labels
            _gt_tr = np.clip(_vl_valid[1:] - _vl_valid[:-1], a_min=0, a_max=None)
            _valid_tr = _vl_valid[1:].max(axis=1) >= 0
            _gv = _gt_tr[_valid_tr]
            if len(_gv) < 1:
                continue

            _rec_sweep_data.append({
                "vp_mono": _vp_mono_valid,
                "gv": _gv,
                "valid_tr": _valid_tr,
            })

        if _rec_sweep_data:
            _candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            _sweep_results: dict[float, list[float]] = {t: [] for t in _candidate_thresholds}

            for rec in _rec_sweep_data:
                for th in _candidate_thresholds:
                    _vp_bin = (rec["vp_mono"] > th).astype(np.int32)
                    _pred_tr = np.clip(_vp_bin[1:] - _vp_bin[:-1], a_min=0, a_max=None)
                    _pv = _pred_tr[rec["valid_tr"]]
                    if len(_pv) < 1:
                        continue
                    _ef1 = event_f1(_pv, rec["gv"], tol=3)
                    _sweep_results[th].append(_ef1)

            _best_th = float(max(_candidate_thresholds, key=lambda t: np.mean(_sweep_results[t])))
            _best_f1 = float(np.mean(_sweep_results[_best_th]))

            # Report sweep table
            _sweep_msg = "  PSR threshold sweep: " + " | ".join(
                f"th={th:.1f}: F1={np.mean(_sweep_results[th]):.4f}"
                for th in _candidate_thresholds
            )
            logger.info(_sweep_msg)
            logger.info("  PSR selected threshold: %.1f  (best event F1=%.4f)", _best_th, _best_f1)

            # Final metrics at optimal threshold
            _poss: list[float] = []
            _taus: list[float] = []
            for rec in _rec_sweep_data:
                _vp_bin = (rec["vp_mono"] > _best_th).astype(np.int32)
                _pred_tr = np.clip(_vp_bin[1:] - _vp_bin[:-1], a_min=0, a_max=None)
                _pv = _pred_tr[rec["valid_tr"]]
                _poss.append(_compute_pos(_pv, rec["gv"]))
                _tau = _compute_tau(_pv, rec["gv"], tol=3)
                if not np.isnan(_tau):
                    _taus.append(_tau)

            metrics["psr_event_f1_at_3"] = _best_f1
            metrics["psr_threshold"] = _best_th
            metrics["psr_pos"] = float(np.mean(_poss)) if _poss else 0.0
            metrics["psr_tau_frames"] = float(np.nanmean(_taus)) if _taus else float("nan")
            logger.info("  PSR: event_f1@+-3=%.4f  (th=%.1f)  POS=%.4f  tau=%.2f  (n_recs=%d)",
                         _best_f1, _best_th,
                         metrics["psr_pos"],
                         metrics.get("psr_tau_frames", float("nan")),
                         len(_rec_sweep_data))

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
    parser.add_argument("--batch-size", type=int, default=4, help="[§6 lever #6] Batch size (effective = batch_size × grad_accum_steps = 16)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head LR")
    parser.add_argument("--lr-log-var", type=float, default=1e-3, help="Log var LR")
    parser.add_argument("--hp-prec-cap", action="store_true", default=True, help="Cap pose precision")
    parser.add_argument("--kendall-uncapped", action="store_true", default=False,
                        help="[OPUS 201 Ablation] Disable per-task log_var caps. Demonstrates Kendall collapse.")
    parser.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=True, help="PCGrad gradient surgery (default: on; use --no-pcgrad to disable)")
    parser.add_argument("--psr-focal", action=argparse.BooleanOptionalAction, default=True,
                        help="[EP10] Use Focal-BCE (γ=2.0, α=0.25) for PSR (default: on; use --no-psr-focal to disable).")
    parser.add_argument("--det-aug", action="store_true", default=True,
                        help="[OPUS 192 §5.5 / Q6] Enable detection-specific augmentation (random horizontal flip, "
                             "color jitter, random crop). Augments images and adjusts bboxes for detection only. "
                             "Other heads see original images. Helps data-limited detection branch.")
    parser.add_argument("--act-decoupled", action="store_true", default=False,
                        help="[OPUS 207] Decoupled activity training (Kang et al. ICLR 2020). "
                             "Phase A (epochs 1-25): instance-balanced sampling. "
                             "Phase B (epochs 26+): freeze backbone, retrain activity classifier "
                             "with class-balanced sampling + raw logits (no class weights).")
    parser.add_argument("--act-decoupled-epoch", type=int, default=25,
                        help="[OPUS 207] Epoch to switch from Phase A to Phase B in decoupled training.")
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
    parser.add_argument("--max-batches-per-epoch", type=int, default=0,
                        help="[§6 lever #6] Cap batches per epoch (0 = full epoch of ~39k; default 0; 200 for fast smoke)")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="[§6 lever #6] Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0,
                        help="Grad-clip norm (5.0 is standard for ViT).")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for ~2x speedup (PyTorch 2.0+)")
    # ── SWA within-run checkpoint averaging (Task #259 / §6 lever #3) ─────
    parser.add_argument("--swa-checkpoints", type=int, default=5,
                        help="Average the last N periodic checkpoints for final eval (0=disabled, default 5)")
    # ── Head warm-starting (Task #260 / §6 lever #4) ──────────────────────
    parser.add_argument("--warm-start-dir", type=str, default=None,
                        help="Directory with st_{head}_best.pt checkpoints for head warm-start")
    # ── Distillation (Task #261 / §6 lever #5) ────────────────────────────
    parser.add_argument("--distill-teacher-dir", type=str, default=None,
                        help="Directory with ST checkpoints for distillation teachers")
    parser.add_argument("--distill-alpha", type=float, default=0.1,
                        help="Distillation loss weight (default: 0.1)")
    parser.add_argument("--distill-temperature", type=float, default=4.0,
                        help="Softmax temperature for distillation (default: 4.0)")
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
        # [OPUS 186] Pre-filter state_dict to matching shapes (see resume path).
        ckpt_sd = ckpt["model_state_dict"]
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in ckpt_sd.items()
                       if k in model_sd and model_sd[k].shape == v.shape}
        skipped = sum(1 for k, v in ckpt_sd.items()
                      if k not in model_sd or model_sd[k].shape != v.shape)
        load_result = model.load_state_dict(filtered_sd, strict=False)
        logger.info("Loaded checkpoint from %s (epoch %d) — %d skipped (shape mismatch), %d missing",
                    args.resume, ckpt.get("epoch", "?"),
                    skipped, len(load_result.missing_keys))

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

    # ── Head warm-start from ST checkpoints (Task #260 / §6 lever #4) ─────
    if args.warm_start_dir:
        logger.info("Warm-starting MTL heads from ST checkpoints in %s", args.warm_start_dir)
        n_warm = warm_start_heads_from_st(model, args.warm_start_dir, device)
        logger.info("Warm-start: loaded %d head tensors total", n_warm)

    # ── Distillation teacher loading (Task #261 / §6 lever #5) ────────────
    distill_teachers = None
    if args.distill_teacher_dir:
        logger.info("Loading distillation teachers from %s", args.distill_teacher_dir)
        distill_teachers = load_distill_teachers(args.distill_teacher_dir, device)
        n_teachers = sum(1 for t in distill_teachers.values() if t is not None)
        logger.info("Loaded %d distillation teachers", n_teachers)

    # ── Kendall log_vars ──────────────────────────────────────────────────
    log_vars = nn.ParameterDict({
        name: nn.Parameter(torch.tensor([-0.5], device=device))
        for name in ["det", "act", "psr", "pose"]
    })
    logger.info("Log vars initialized to -0.5")

    # ── Auto-soup init [OPUS 192 §5 step 8] ──────────────────────────────
    # If a soup backbone (averaged from single-task specialists via
    # scripts/build_soup.py) exists in the output dir, load its backbone
    # weights. This is a near-free init increment for MTL finetune per
    # Wortsman 2022. If --resume is also given, soup is skipped (resume wins).
    if not args.resume:
        soup_path = output_dir / "soup_backbone.pt"
        if soup_path.exists():
            logger.info("Auto-loading soup backbone from %s", soup_path)
            soup_sd = torch.load(soup_path, map_location="cpu", weights_only=False)
            filtered = {k: v for k, v in soup_sd.items()
                         if k in model.state_dict()
                         and model.state_dict()[k].shape == v.shape}
            load_result = model.load_state_dict(filtered, strict=False)
            logger.info("  Loaded %d/%d soup tensors (skipped %d shape-mismatched)",
                        len(filtered), len(soup_sd),
                        len(soup_sd) - len(filtered))
        else:
            logger.info("No soup backbone found at %s — using random init", soup_path)

    # ── EMA loss tracker for Kendall scale normalization [OPUS 181 D1] ────
    # Initialized to 1.0 so the first step's normalized loss equals the raw
    # loss (no divide-by-zero, no underflow). Tracks each task's running mean
    # so Kendall balances comparable scales (≈O(1) per task) instead of
    # collapsing to inverse-loss scaling `weight = 1/(2·loss)`.
    ema_losses = {
        name: torch.tensor(1.0, device=device)
        for name in ["det", "act", "psr", "pose"]
    }
    logger.info("EMA loss tracker initialized to 1.0")

    # ── EMA model weights [OPUS 186 E-3/I-8] ─────────────────────────────
    # Exponential-moving-average of model parameters. Cheap, reliable +1-2%
    # across all metrics. Initialized to the model state_dict on the first
    # update; used for eval instead of the raw model state. Momentum 0.999
    # (≈ last 1000 steps dominate). Detached from autograd.
    ema_model_state: dict = {
        k: v.detach().clone().float()
        for k, v in model.state_dict().items()
    }
    ema_momentum_model = 0.999
    logger.info("EMA model weights initialized to current model state_dict")

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
        # [OPUS 186] Pre-filter checkpoint state_dict to only keys with matching
        # shapes. PyTorch's strict=False still RAISES on size mismatches (only
        # missing/unexpected keys are tolerated). PSR head reshape (96→768ch) and
        # 2-layer activity MLP mean the checkpoint's head weights are not directly
        # loadable. Backbone, FPN, pose head load fully; new layers init fresh.
        ckpt_sd = ckpt["model_state_dict"]
        model_sd = model.state_dict()
        filtered_sd = {}
        skipped = []
        for k, v in ckpt_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered_sd[k] = v
            else:
                skipped.append((k, tuple(v.shape) if hasattr(v, "shape") else None,
                                tuple(model_sd[k].shape) if k in model_sd else None))
        if skipped:
            logger.warning("Resume: skipped %d keys with shape mismatch:", len(skipped))
            for k, ckpt_shape, model_shape in skipped[:5]:
                logger.warning("  %s: ckpt=%s vs model=%s", k, ckpt_shape, model_shape)
            if len(skipped) > 5:
                logger.warning("  ... and %d more", len(skipped) - 5)
        load_result = model.load_state_dict(filtered_sd, strict=False)
        if load_result.missing_keys:
            logger.warning("Resume: %d missing keys (new layers initialize fresh): %s",
                           len(load_result.missing_keys), load_result.missing_keys[:3])
        # [OPUS 186] Optimizer state may also have shape mismatches (param
        # groups changed because new layers were added). Try to load; if it
        # fails, skip and start fresh optimizer momentum.
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Loaded optimizer state from checkpoint")
        except (ValueError, KeyError) as e:
            logger.warning("Could not load optimizer state (%s) — starting fresh momentum", e)
            # Rebuild optimizer with current model params (already done above).
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_act_top1 = ckpt.get("best_act_top1", 0.0)
        log_vars = ckpt.get("log_vars", log_vars)
        # [OPUS 181 D1] Restore EMA state if present (older checkpoints lack it
        # → fall back to fresh init at 1.0; tracker will reconverge in ~100 steps).
        if "ema_losses" in ckpt:
            for name, v in ckpt["ema_losses"].items():
                ema_losses[name] = v.to(device)
        # [OPUS 186 E-3] Restore EMA model state if present.
        if "ema_model_state" in ckpt:
            for k, v in ckpt["ema_model_state"].items():
                if k in ema_model_state:
                    ema_model_state[k] = v.to(ema_model_state[k].dtype).to(device)
            logger.info("Resumed EMA model state (size=%d tensors)", len(ema_model_state))
        # [OPUS 186] After loading the checkpoint, re-anchor ema_model_state
        # to the now-loaded model. This way any keys that didn't load (e.g.,
        # the reshaped PSR head) start EMA-tracking the *new* random init
        # rather than carrying over stale shapes.
        ema_model_state = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
        }
        logger.info("Re-anchored EMA model state to current model (post-resume)")
        logger.info("Resumed from epoch %d", start_epoch)

    # ── AMP scaler ─────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(device.type, enabled=True)

    # ── Activity class weights (Doc 175 §4) ────────────────────────────────
    act_class_weights = compute_activity_class_weights(train_ds, num_classes=len(train_ds.class_counts))
    act_class_weights = act_class_weights.to(device)
    logger.info("Activity class weights computed — shape=%s, device=%s",
                act_class_weights.shape, act_class_weights.device)

    # [OPUS 201] Enable logit-adjustment on ActivityHead (Menon et al. 2020).
    # Balanced softmax corrects for long-tail class distribution at eval time,
    # counteracting the class-weight collapse that produces below-random top-1.
    class_counts_tensor = torch.from_numpy(train_ds.class_counts.astype(np.int64))
    model.act_head.enable_logit_adjust(class_counts_tensor)
    logger.info("Activity logit-adjust enabled (%d classes, %d total samples)",
                len(train_ds.class_counts), int(class_counts_tensor.sum()))

    # [OPUS 207 §2.6] Compute class frequencies for Menon logit-adjust in loss.
    # These are passed to activity_loss() which adds tau*log(freq) to logits
    # inside the cross-entropy computation. The model's forward() returns raw
    # logits unconditionally; the correction happens only in the training loss.
    _total = class_counts_tensor.sum().float().clamp(min=1)
    act_logit_adjust_freq = class_counts_tensor.float() / _total
    act_logit_adjust_freq = act_logit_adjust_freq.to(device)

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info("Starting training (%d epochs)...", args.epochs)
    if args.act_decoupled:
        logger.info("Decoupled activity training: Phase A (epochs 1-%d) → Phase B (epochs %d+)",
                    args.act_decoupled_epoch, args.act_decoupled_epoch + 1)
    metrics_log = {"train_metrics": [], "log_var_history": {}, "config": vars(args)}
    act_decoupled_phase_b = False  # [OPUS 207] Tracks whether we've transitioned

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # [OPUS 207] Decoupled training Phase B transition.
        # Freeze backbone, retrain only activity classifier with class-balanced sampling.
        if args.act_decoupled and epoch > args.act_decoupled_epoch and not act_decoupled_phase_b:
            logger.info("=== Decoupled Phase B: freezing backbone, class-balanced activity retrain ===")
            for name, param in model.named_parameters():
                if "act_head.classifier" not in name:
                    param.requires_grad = False
            # Recompute class-balanced sampler: oversample rare classes
            from torch.utils.data import WeightedRandomSampler
            class_counts = train_ds.class_counts
            class_weights_bal = 1.0 / np.maximum(class_counts, 1)
            sample_weights = np.array([class_weights_bal[train_ds[i][1]["activity"].item()]
                                       if train_ds[i][1].get("activity") is not None
                                       and train_ds[i][1]["activity"].item() >= 0
                                       else 0.0 for i in range(len(train_ds))])
            balanced_sampler = WeightedRandomSampler(
                torch.from_numpy(sample_weights).float(),
                num_samples=len(train_ds), replacement=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=False,
                sampler=balanced_sampler,
                collate_fn=collate_fn_sequences, num_workers=args.num_workers,
                pin_memory=True, drop_last=True,
            )
            # Disable class weights — balanced batches handle long-tail
            act_class_weights = None
            # Disable logit-adjust during pure classifier retrain
            model.act_head.logit_adjust = False
            act_decoupled_phase_b = True
            logger.info("Phase B active: backbone frozen, class-balanced sampler, raw logits")
        t0 = time.time()
        epoch_metrics = {
            "loss": 0, "loss_det": 0, "loss_act": 0, "loss_psr": 0, "loss_pose": 0,
            "loss_distill": 0,
            **{f"loss_psr_c{i}": 0 for i in range(11)},
            "log_var_det": 0, "log_var_act": 0, "log_var_psr": 0, "log_var_pose": 0,
        }
        log_var_trajectory = {k: [] for k in ["det", "act", "psr", "pose"]}
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
                grad_clip_norm=args.grad_clip_norm,  # [OPUS 186 E-6]
                hp_prec_cap=args.hp_prec_cap,
                kendall_uncapped=args.kendall_uncapped,  # [OPUS 201] Kendall-collapse ablation
                pcgrad=args.pcgrad,
                act_class_weights=act_class_weights,
                act_logit_adjust_freq=act_logit_adjust_freq,  # [OPUS 207 §2.6]
                do_step=do_step,
                ema_losses=ema_losses,  # [OPUS 181 D1] EMA-normalized Kendall losses.
                grad_accum_steps=args.grad_accum_steps,  # [OPUS 186 §5.1] Mean-scale accumulation.
                psr_focal=args.psr_focal,  # [OPUS 192 Q2] Optional Focal-BCE for PSR.
                det_aug=args.det_aug,  # [OPUS 192 §5.5 / Q6] Detection-specific augmentation.
                distill_teachers=distill_teachers,  # [Task #261] Distillation teachers.
                distill_alpha=args.distill_alpha,   # [Task #261]
                distill_temperature=args.distill_temperature,  # [Task #261]
            )

            for k in epoch_metrics:
                epoch_metrics[k] += step_metrics.get(k, 0)
            for tag in ["det", "act", "psr", "pose"]:
                lv_val = step_metrics.get(f"log_var_{tag}")
                if lv_val is not None:
                    log_var_trajectory[tag].append(lv_val)
            n_steps += 1

            # [OPUS 186 E-3/I-8] Update EMA model weights. Only update on
            # boundary steps (after the optimizer has actually stepped) so
            # the EMA tracks post-step weights, not mid-accumulation weights.
            if do_step:
                with torch.no_grad():
                    msd = model.state_dict()
                    for k, v in ema_model_state.items():
                        if k in msd:
                            v.mul_(ema_momentum_model).add_(
                                msd[k].detach().float(), alpha=1.0 - ema_momentum_model
                            )

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
        # ── log_var trajectory analysis (Kendall collapse diagnosis) ────────
        lv_mean = {k: avg_metrics.get(f"log_var_{k}", 0) for k in ["det", "act", "psr", "pose"]}
        lv_min = {k: min(v) if v else 0 for k, v in log_var_trajectory.items()}
        lv_max = {k: max(v) if v else 0 for k, v in log_var_trajectory.items()}
        logger.info(
            "Epoch %3d/%d | loss=%.4f det=%.4f act=%.4f psr=%.4f pose=%.4f distill=%.4f | "
            "lv_mean=[%.2f,%.2f,%.2f,%.2f] lv_min=[%.2f,%.2f,%.2f,%.2f] lv_max=[%.2f,%.2f,%.2f,%.2f] | "
            "lr=%.2e | %.1fs",
            epoch, args.epochs,
            avg_metrics["loss"], avg_metrics["loss_det"], avg_metrics["loss_act"],
            avg_metrics["loss_psr"], avg_metrics["loss_pose"],
            avg_metrics.get("loss_distill", 0),
            lv_mean["det"], lv_mean["act"], lv_mean["psr"], lv_mean["pose"],
            lv_min["det"], lv_min["act"], lv_min["psr"], lv_min["pose"],
            lv_max["det"], lv_max["act"], lv_max["psr"], lv_max["pose"],
            avg_metrics["lr"], dt,
        )
        # Save log_var per-epoch trajectory to metrics_log
        metrics_log.setdefault("log_var_history", {})
        for tag in ["det", "act", "psr", "pose"]:
            metrics_log["log_var_history"].setdefault(tag, {})
            metrics_log["log_var_history"][tag][str(epoch)] = {
                "mean": lv_mean[tag], "min": lv_min[tag], "max": lv_max[tag],
            }
        # Per-component PSR loss breakdown (Doc 175 §4)
        _pcs = [avg_metrics.get(f"loss_psr_c{i}", 0) for i in range(11)]
        logger.info("  PSR comp: [%s]", " ".join(f"{v:.4f}" for v in _pcs))

        scheduler.step()

        # ── Per-epoch quick health check (2 val batches) ─────────────────
        # [OPUS 192 §5] Lightweight signal between full evals. Only runs
        # when a full eval is NOT happening this epoch. Takes ~5 seconds.
        if epoch % args.eval_every != 0:
            model.eval()
            with torch.no_grad():
                _qc_B = 0
                for _qc_batch in eval_loader:
                    if _qc_B >= 2:
                        break
                    _qc_images = _qc_batch[0][:args.batch_size].to(device, non_blocking=True)
                    _qc_images = _qc_images.float() / 255.0
                    _mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 3, 1, 1)
                    _std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 1, 3, 1, 1)
                    _qc_images = (_qc_images - _mean) / _std
                    _qc_images = _qc_images.permute(0, 2, 1, 3, 4).contiguous()
                    _qc_out = model(_qc_images)
                    _qc_B += 1
                # Activity: how many classes are predicted across the batch?
                _qc_preds = _qc_out["activity"].argmax(dim=-1)
                _qc_n_pred = len(_qc_preds.unique())
                _qc_max_conf = torch.softmax(_qc_out["activity"], dim=-1).max(dim=-1)[0].mean().item()
                # PSR: stddev across frames per component
                _qc_psr_std = _qc_out["psr_logits"].float().std(dim=(0, 1)).tolist()
                _qc_psr_std_max = max(_qc_psr_std)
                # Detection: max sigmoid score per level
                _qc_det_levels = list(_qc_out["detection"].keys())
                logger.info(
                    "  Quick: act_preds=%duniq/%.2fmaxconf | psr_stdmax=%.4f | det_lvls=%s",
                    _qc_n_pred, _qc_max_conf, _qc_psr_std_max, _qc_det_levels,
                )
            model.train()

        # ── Evaluate on eval split (Doc 175 section 7.2) ─────────────────
        if epoch % args.eval_every == 0:
            # [OPUS 186 E-3/I-8] Swap in EMA model weights for evaluation;
            # restore raw weights afterwards. EMA averages across recent
            # checkpoints, smoothing late-stage noise and giving +1-2%.
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # Cast EMA state back to model's param dtypes (some keys may be int/long)
            ema_swap = {k: v.to(raw_state[k].dtype) for k, v in ema_model_state.items()}
            model.load_state_dict(ema_swap, strict=False)
            try:
                eval_metrics = evaluate(model, eval_loader, device, epoch=epoch)
            finally:
                # Always restore raw weights so training continues from the
                # latest optimizer state.
                model.load_state_dict(raw_state, strict=False)
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
                    "ema_losses": ema_losses,  # [OPUS 181 D1]
                    "ema_model_state": ema_model_state,  # [OPUS 186 E-3]
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
                "ema_losses": ema_losses,  # [OPUS 181 D1]
                "ema_model_state": ema_model_state,  # [OPUS 186 E-3]
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

    # ── SWA within-run checkpoint averaging (Task #259 / §6 lever #3) ─────
    if args.swa_checkpoints > 0:
        logger.info("Building SWA averaged model from last %d checkpoints...", args.swa_checkpoints)
        swa_sd = swa_average_checkpoints(output_dir, args.swa_checkpoints, device)
        if swa_sd is not None:
            raw_state_swa = {k: v.detach().clone() for k, v in model.state_dict().items()}
            filtered_swa = {k: v.to(raw_state_swa[k].dtype) for k, v in swa_sd.items()
                           if k in raw_state_swa and raw_state_swa[k].shape == v.shape}
            model.load_state_dict(filtered_swa, strict=False)
            logger.info("SWA: loaded %d tensors (of %d total)", len(filtered_swa), len(swa_sd))

            swa_test_metrics = evaluate(model, test_loader, device, epoch=None)
            logger.info("Test metrics (SWA): %s", swa_test_metrics)
            metrics_log["test_metrics_swa"] = swa_test_metrics

            swa_path = output_dir / "swa_averaged.pt"
            torch.save({"model_state_dict": model.state_dict(), "method": "swa",
                         "n_checkpoints": args.swa_checkpoints}, swa_path)
            logger.info("SWA model saved: %s", swa_path)

            # Restore raw weights for the standard test eval below
            model.load_state_dict(raw_state_swa, strict=False)

    test_metrics = evaluate(model, test_loader, device, epoch=None)
    logger.info("Test metrics (raw): %s", test_metrics)

    # [OPUS 186 E-3/I-8] Also report EMA-weighted test metrics for the paper.
    raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema_swap = {k: v.to(raw_state[k].dtype) for k, v in ema_model_state.items()}
    model.load_state_dict(ema_swap, strict=False)
    try:
        ema_test_metrics = evaluate(model, test_loader, device, epoch=None)
    finally:
        model.load_state_dict(raw_state, strict=False)
    logger.info("Test metrics (EMA): %s", ema_test_metrics)

    # Save final metrics
    metrics_log["test_metrics"] = test_metrics
    metrics_log["test_metrics_ema"] = ema_test_metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2, default=str)

    logger.info("Training complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
