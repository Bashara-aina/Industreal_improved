"""Detection loss for YOLOv8-style DFL head.

Adapts YOLOv8's anchor-free output format to work with existing MTL training.
Instead of anchor matching, uses a grid-center based formulation where each
location predicts (l, t, r, b) distances via DFL.

Loss = focal (cls) + DFL (reg distribution) + CIoU (reg box)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mtl_v3 import sigmoid_focal_loss, ciou_loss


def dfl_decode(reg_preds: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """Decode DFL distribution to box offsets (l, t, r, b).

    Args:
        reg_preds: [B, 4*reg_max, H, W] distribution logits
        reg_max: number of discrete values

    Returns:
        [B, 4, H, W] expected values for each side
    """
    B, _, H, W = reg_preds.shape
    # Reshape to [B, 4, reg_max, H, W] then softmax over reg_max
    x = reg_preds.view(B, 4, reg_max, H, W).softmax(dim=2)
    # Compute expected value: dot product with values [0, 1, ..., reg_max-1]
    proj = torch.arange(reg_max, dtype=reg_preds.dtype, device=reg_preds.device).view(1, 1, reg_max, 1, 1)
    return (x * proj).sum(dim=2)  # [B, 4, H, W]


def grid_centers(H, W, stride, device, dtype=torch.float32):
    """Generate (grid_x, grid_y) centers for HxW grid."""
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * stride
    return ys, xs


def dfl_loss(
    pred_dist: torch.Tensor,  # [N, 4*reg_max]
    target_dist: torch.Tensor,  # [N, 4*reg_max] soft target
    reg_max: int = 16,
) -> torch.Tensor:
    """DFL loss: cross-entropy between predicted and target distributions.

    Pred and target are both soft distributions over [0, reg_max).
    """
    # [N, 4, reg_max]
    pred_log = F.log_softmax(pred_dist.view(-1, 4, reg_max), dim=2)
    target = target_dist.view(-1, 4, reg_max)
    loss = -(target * pred_log).sum(dim=2)  # [N, 4]
    return loss.sum(dim=1).mean()  # sum over 4 sides, mean over anchors


def yolov8_detection_loss(
    cls_logits: torch.Tensor,    # [B, 24, H, W]
    reg_preds: torch.Tensor,     # [B, 4*reg_max, H, W]
    gt_boxes: list,              # list of [N, 4] (cx, cy, w, h) normalized
    gt_classes: list,            # list of [N] class indices
    stride: float = 8.0,
    reg_max: int = 16,
    focal_gamma: float = 1.5,
    focal_alpha: float = 0.25,
    reg_weight: float = 5.0,
    pos_radius: float = 2.5,
) -> tuple:
    """YOLOv8-style detection loss.

    Args:
        cls_logits: classification logits
        reg_preds: regression distribution (DFL)
        gt_boxes: list per sample of (cx, cy, w, h) in [0, 1]
        gt_classes: list per sample of class indices
        stride: feature stride (8 for P3, 16 for P4, 32 for P5)

    Returns:
        cls_loss, reg_loss, n_pos
    """
    B, C, H, W = cls_logits.shape
    device = cls_logits.device

    # Build targets for this level
    # cls_target: [B, H, W] with -1 = ignore, 0..23 = class, C = background
    # reg_target: [B, 4, H, W] in (l, t, r, b) from grid center
    # reg_target_dist: [B, 4*reg_max, H, W] soft distribution
    cls_target = torch.full((B, H, W), C, dtype=torch.long, device=device)
    reg_target = torch.zeros(B, 4, H, W, device=device)
    reg_target_dist = torch.zeros(B, 4 * reg_max, H, W, device=device)
    pos_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

    # Grid centers (pixel coords)
    ys, xs = grid_centers(H, W, stride, device)
    # For each sample, assign GT to nearest grid cells within pos_radius
    for b in range(B):
        gt_b = gt_boxes[b]  # [N, 4] normalized (cx, cy, w, h)
        gc_b = gt_classes[b]  # [N]

        if gt_b.numel() == 0:
            continue

        gt_b_device = gt_b.to(device)
        gc_b_device = gc_b.to(device)

        # GT in pixel coords
        cx_pix = gt_b_device[:, 0] * (W * stride)
        cy_pix = gt_b_device[:, 1] * (H * stride)
        w_pix = gt_b_device[:, 2] * (W * stride)
        h_pix = gt_b_device[:, 3] * (H * stride)

        # Convert to (l, t, r, b) format
        # gt_l = cx - w/2; gt_t = cy - h/2; gt_r = cx + w/2; gt_b = cy + h/2
        gt_l = cx_pix - w_pix / 2
        gt_t = cy_pix - h_pix / 2
        gt_r = cx_pix + w_pix / 2
        gt_b = cy_pix + h_pix / 2

        for n in range(gt_b_device.shape[0]):
            gx, gy = cx_pix[n] / stride, cy_pix[n] / stride
            # Find grid cells within pos_radius of the GT center
            cell_x_min = max(0, int(gx - pos_radius))
            cell_x_max = min(W, int(gx + pos_radius) + 1)
            cell_y_min = max(0, int(gy - pos_radius))
            cell_y_max = min(H, int(gy + pos_radius) + 1)

            for cy in range(cell_y_min, cell_y_max):
                for cx in range(cell_x_min, cell_x_max):
                    # Compute (l, t, r, b) target relative to grid center
                    l = gx - cx  # grid units
                    t = gy - cy
                    r = (cx + 1) - gx
                    b = (cy + 1) - gy
                    # Clamp to reg_max
                    l = min(max(l, 0), reg_max - 1)
                    t = min(max(t, 0), reg_max - 1)
                    r = min(max(r, 0), reg_max - 1)
                    b = min(max(b, 0), reg_max - 1)

                    cls_target[b, cy, cx] = gc_b_device[n]
                    reg_target[b, :, cy, cx] = torch.tensor([l, t, r, b], device=device)
                    # Soft target distribution: one-hot with smoothing
                    l_int = int(l.item())
                    t_int = int(t.item())
                    r_int = int(r.item())
                    b_int = int(b.item())
                    reg_target_dist[b, l_int, cy, cx] += 1 - (l.item() - l_int)
                    reg_target_dist[b, reg_max + t_int, cy, cx] += 1 - (t.item() - t_int)
                    reg_target_dist[b, 2 * reg_max + r_int, cy, cx] += 1 - (r.item() - r_int)
                    reg_target_dist[b, 3 * reg_max + b_int, cy, cx] += 1 - (b.item() - b_int)
                    pos_mask[b, cy, cx] = True

    # Classification loss (focal on full grid)
    cls_target_flat = cls_target.view(-1)
    cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
    cls_loss = sigmoid_focal_loss(cls_flat, cls_target_flat, gamma=focal_gamma, alpha=focal_alpha)

    # Regression loss (DFL + CIoU on positive anchors only)
    pos_indices = pos_mask.nonzero(as_tuple=True)  # tuple of (b, y, x) indices

    if pos_indices[0].numel() > 0:
        # DFL loss
        pred_dist = reg_preds.permute(0, 2, 3, 1).reshape(-1, 4 * reg_max)  # [B*H*W, 4*reg_max]
        target_dist_flat = reg_target_dist.permute(0, 2, 3, 1).reshape(-1, 4 * reg_max)

        # Gather only positive positions
        flat_idx = pos_indices[0] * H * W + pos_indices[1] * W + pos_indices[2]
        pred_dist_pos = pred_dist[flat_idx]
        target_dist_pos = target_dist_flat[flat_idx]
        dfl = dfl_loss(pred_dist_pos, target_dist_pos, reg_max=reg_max)

        # CIoU loss: convert (l, t, r, b) predictions and targets to boxes
        # First DFL-decode predictions
        pred_ltrb = dfl_decode(reg_preds, reg_max=reg_max)  # [B, 4, H, W]
        pred_ltrb_pos = pred_ltrb[pos_indices[0], :, pos_indices[1], pos_indices[2]]  # [N, 4]

        target_ltrb = reg_target[pos_indices[0], :, pos_indices[1], pos_indices[2]]  # [N, 4]

        # Convert (l, t, r, b) to (x1, y1, x2, y2) in pixel coords
        # box = (cx*stride - l*stride, cy*stride - t*stride, cx*stride + r*stride, cy*stride + b*stride)
        cx_pix_pos = (pos_indices[2].float() + 0.5) * stride
        cy_pix_pos = (pos_indices[1].float() + 0.5) * stride
        pred_boxes = torch.stack([
            cx_pix_pos - pred_ltrb_pos[:, 0] * stride,
            cy_pix_pos - pred_ltrb_pos[:, 1] * stride,
            cx_pix_pos + pred_ltrb_pos[:, 2] * stride,
            cy_pix_pos + pred_ltrb_pos[:, 3] * stride,
        ], dim=1)

        target_boxes = torch.stack([
            cx_pix_pos - target_ltrb[:, 0] * stride,
            cy_pix_pos - target_ltrb[:, 1] * stride,
            cx_pix_pos + target_ltrb[:, 2] * stride,
            cy_pix_pos + target_ltrb[:, 3] * stride,
        ], dim=1)

        ciou = ciou_loss(pred_boxes, target_boxes).mean()
        reg_loss = dfl + reg_weight * ciou
    else:
        reg_loss = torch.tensor(0.0, device=device)

    n_pos = pos_mask.sum().item()
    return cls_loss, reg_loss, n_pos


def yolov8_detection_loss_v2(
    cls_logits: torch.Tensor,    # [B, 24, H, W]
    reg_preds: torch.Tensor,     # [B, 4*reg_max, H, W]
    gt_boxes: list,              # list of [N, 4] (cx, cy, w, h) normalized
    gt_classes: list,            # list of [N] class indices
    img_w: int = 640,
    img_h: int = 360,
    stride: float = 8.0,
    reg_max: int = 16,
    focal_gamma: float = 1.5,
    focal_alpha: float = 0.25,
    reg_weight: float = 5.0,
):
    """YOLOv8-style loss with CORRECT DFL target assignment.

    BUG FIX 2026-07-22 — The original (l, t, r, b) target computation was WRONG:
    it computed distances from cell EDGES to GT center, but YOLOv8 DFL expects
    distances from grid CENTER to box edges. This caused:
    - DFL loss to train wrong distribution targets
    - CIoU loss to compute IoU between incorrect box coordinates
    - Only ~1 positive per GT (the center-containing cell) due to wrong l>=0/r>=0 checks

    Correct DFL targets (grid units):
        grid_center = (cx+0.5, cy+0.5)
        l = grid_center_x - (gt_cx - gt_w/2)  = (cx+0.5) - x1_in_grid
        t = grid_center_y - (gt_cy - gt_h/2)  = (cy+0.5) - y1_in_grid
        r = (gt_cx + gt_w/2) - grid_center_x  = x2_in_grid - (cx+0.5)
        b = (gt_cy + gt_h/2) - grid_center_y  = y2_in_grid - (cy+0.5)

    Positive assignment: cells whose grid center falls within the GT box.
    For small objects (width/height < 1 grid unit), also include the nearest cell.
    """
    B, C, H, W = cls_logits.shape
    device = cls_logits.device

    # Initialize targets (-1 = background, matches sigmoid_focal_loss convention)
    cls_target = torch.full((B, H, W), -1, dtype=torch.long, device=device)
    reg_target = torch.zeros(B, 4, H, W, device=device)
    reg_target_dist = torch.zeros(B, 4 * reg_max, H, W, device=device)
    assigned_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

    # Grid centers in grid units: (grid_x+0.5, grid_y+0.5)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    grid_center_x = grid_x + 0.5  # [H, W]
    grid_center_y = grid_y + 0.5  # [H, W]

    for b in range(B):
        gt_b = gt_boxes[b]
        gc_b = gt_classes[b]

        if gt_b.numel() == 0:
            continue

        gt_b_dev = gt_b.to(device)
        gc_b_dev = gc_b.to(device)

        # GT in grid units
        cx_grid = gt_b_dev[:, 0] * W
        cy_grid = gt_b_dev[:, 1] * H
        w_grid = gt_b_dev[:, 2] * W
        h_grid = gt_b_dev[:, 3] * H

        cx_half = w_grid / 2
        cy_half = h_grid / 2
        x1_grid = cx_grid - cx_half  # left edge
        y1_grid = cy_grid - cy_half  # top edge
        x2_grid = cx_grid + cx_half  # right edge
        y2_grid = cy_grid + cy_half  # bottom edge

        n_in_level = 0
        for n in range(int(gt_b_dev.shape[0])):
            x1_n = float(x1_grid[n].item())
            y1_n = float(y1_grid[n].item())
            x2_n = float(x2_grid[n].item())
            y2_n = float(y2_grid[n].item())
            w_n = float(w_grid[n].item())
            h_n = float(h_grid[n].item())

            # Approach: find ALL cells whose grid center falls within the GT box.
            # Grid center (cx+0.5, cy+0.5) is in GT box iff:
            #   x1_n <= cx+0.5 < x2_n  AND  y1_n <= cy+0.5 < y2_n
            # This naturally covers ceil(w_n) * ceil(h_n) cells.
            #
            # For very small objects (< 1 grid unit), where center-containment
            # gives 0-1 cells, we also include the cell containing the GT center.

            # Cells whose grid centers are within the GT box
            cell_x_min = max(0, int(math.ceil(x1_n - 0.5)))
            cell_x_max = min(W - 1, int(math.floor(x2_n - 0.5)))
            cell_y_min = max(0, int(math.ceil(y1_n - 0.5)))
            cell_y_max = min(H - 1, int(math.floor(y2_n - 0.5)))

            # If object is too small for any cell center to be inside,
            # assign the cell containing the GT center
            if cell_x_min > cell_x_max or cell_y_min > cell_y_max:
                # GT center in grid units
                gx = (x1_n + x2_n) / 2
                gy = (y1_n + y2_n) / 2
                cell_x_min = max(0, int(gx))
                cell_x_max = min(W - 1, int(gx))
                cell_y_min = max(0, int(gy))
                cell_y_max = min(H - 1, int(gy))

            for cy in range(cell_y_min, cell_y_max + 1):
                for cx in range(cell_x_min, cell_x_max + 1):
                    # Skip if already assigned (first GT takes priority)
                    if bool(assigned_mask[b, cy, cx].item()):
                        continue

                    # CORRECT DFL targets: distance from grid CENTER to box edges
                    gc_x = grid_center_x[cy, cx].item()  # cx + 0.5
                    gc_y = grid_center_y[cy, cx].item()  # cy + 0.5

                    # DFL targets in grid units (must be in [0, reg_max))
                    # NOTE: dfl_l/dfl_t/dfl_r/dfl_b to avoid shadowing batch index b and temporal t
                    dfl_l = gc_x - x1_n
                    dfl_t = gc_y - y1_n
                    dfl_r = x2_n - gc_x
                    dfl_b = y2_n - gc_y

                    # Clamp to DFL range
                    dfl_l = max(0.0, min(dfl_l, reg_max - 1e-6))
                    dfl_t = max(0.0, min(dfl_t, reg_max - 1e-6))
                    dfl_r = max(0.0, min(dfl_r, reg_max - 1e-6))
                    dfl_b = max(0.0, min(dfl_b, reg_max - 1e-6))

                    cls_target[b, cy, cx] = int(gc_b_dev[n].item())
                    reg_target[b, 0, cy, cx] = dfl_l
                    reg_target[b, 1, cy, cx] = dfl_t
                    reg_target[b, 2, cy, cx] = dfl_r
                    reg_target[b, 3, cy, cx] = dfl_b

                    # Soft DFL target distribution with linear interpolation
                    for side, val in enumerate([dfl_l, dfl_t, dfl_r, dfl_b]):
                        idx = int(val)
                        frac = val - idx
                        channel = side * reg_max
                        reg_target_dist[b, channel + idx, cy, cx] += 1.0 - frac
                        if idx + 1 < reg_max:
                            reg_target_dist[b, channel + idx + 1, cy, cx] += frac

                    n_in_level += 1
                    assigned_mask[b, cy, cx] = True

    # Classification loss
    cls_target_flat = cls_target.view(-1)
    cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
    cls_loss = sigmoid_focal_loss(cls_flat, cls_target_flat, gamma=focal_gamma, alpha=focal_alpha)

    # Regression loss (only on positive cells)
    pos_mask = (cls_target >= 0)
    n_pos = pos_mask.sum().item()

    if n_pos > 0:
        # DFL loss
        pred_dist = reg_preds.permute(0, 2, 3, 1).reshape(-1, 4 * reg_max)
        target_dist_flat = reg_target_dist.permute(0, 2, 3, 1).reshape(-1, 4 * reg_max)
        dfl = dfl_loss(pred_dist[pos_mask.view(-1)], target_dist_flat[pos_mask.view(-1)], reg_max=reg_max)

        # CIoU loss
        pred_ltrb = dfl_decode(reg_preds, reg_max=reg_max)
        # Reshape: [B, 4, H, W] -> [B*H*W, 4]
        pred_ltrb_flat = pred_ltrb.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flat = reg_target.permute(0, 2, 3, 1).reshape(-1, 4)

        # Grid centers in pixel coords
        cx_grid_pix = (grid_x + 0.5) * stride
        cy_grid_pix = (grid_y + 0.5) * stride
        cx_grid_pix_flat = cx_grid_pix.view(-1).expand(B, -1).reshape(-1)
        cy_grid_pix_flat = cy_grid_pix.view(-1).expand(B, -1).reshape(-1)

        # Convert (l, t, r, b) in grid units to (x1, y1, x2, y2) in pixels
        # box_x1 = grid_center_pix - l * stride = (cx+0.5)*stride - l*stride
        pred_boxes = torch.stack([
            cx_grid_pix_flat - pred_ltrb_flat[:, 0] * stride,
            cy_grid_pix_flat - pred_ltrb_flat[:, 1] * stride,
            cx_grid_pix_flat + pred_ltrb_flat[:, 2] * stride,
            cy_grid_pix_flat + pred_ltrb_flat[:, 3] * stride,
        ], dim=1)

        target_boxes = torch.stack([
            cx_grid_pix_flat - reg_target_flat[:, 0] * stride,
            cy_grid_pix_flat - reg_target_flat[:, 1] * stride,
            cx_grid_pix_flat + reg_target_flat[:, 2] * stride,
            cy_grid_pix_flat + reg_target_flat[:, 3] * stride,
        ], dim=1)

        ciou = ciou_loss(pred_boxes[pos_mask.view(-1)], target_boxes[pos_mask.view(-1)]).mean()
        reg_loss = dfl + reg_weight * ciou
    else:
        reg_loss = torch.tensor(0.0, device=device)

    return cls_loss, reg_loss, n_pos