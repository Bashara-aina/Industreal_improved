"""TaskAlignedAssigner for YOLO-style object detection.

[OPUS 192] Citation: TOOD: Task-aligned One-stage Object Detection (Feng et al.,
ICCV 2021). NOT Ultralytics YOLOv8 (which is AGPL-3.0 and has no peer-reviewed
paper). The TAL algorithm itself is in TOOD; YOLOv8 uses a port of TOOD's
align_metric with a different topk. Our implementation follows TOOD with
topk=10 (YOLOv8 default).

Alignment score: s = cls_score^alpha * box_iou^beta
For each GT, pick top-k cells by alignment score.

Conditional: only used if MVP Probe 1 shows eval-harness works AND Probe 4
shows the assigner is the bottleneck.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAlignedAssigner(nn.Module):
    """TOOD's TaskAlignedAssigner — assigns each GT to top-k cells by alignment score.

    Args:
        topk: number of cells to assign per GT (YOLOv8 default: 10)
        alpha: weight on classification score in alignment metric (TOOD: 1.0)
        beta: weight on box IoU in alignment metric (TOOD: 6.0)
        eps: small constant for numerical stability
    """

    def __init__(self, topk: int = 10, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_cls: torch.Tensor,    # [B, H*W, nc] sigmoid scores (after sigmoid)
        pred_box: torch.Tensor,    # [B, H*W, 4] decoded xyxy (in pixel coords)
        anchors: torch.Tensor,      # [H*W, 2] cell centers (cx, cy)
        gt_boxes: torch.Tensor,    # [B, max_n, 4] xyxy (zero-padded)
        gt_labels: torch.Tensor,   # [B, max_n] class indices (zero-padded; 0 = ignore)
        anchor_points: torch.Tensor,  # [H*W, 2] (cx, cy) of each anchor cell
        stride: torch.Tensor,       # [1] or scalar; stride of this FPN level
    ) -> tuple:
        """Assign GTs to top-k cells per FPN level.

        Returns:
            target_labels: [B, H*W, nc] one-hot target cls (smoothed)
            target_bboxes: [B, H*W, 4] target box (zero where no GT)
            target_scores: [B, H*W, nc] target alignment score (alpha-blended)
            mask: [B, H*W, 1] 1 for assigned cells, 0 otherwise
            target_gt_idx: [B, H*W] index into gt_boxes of the assigned GT
        """
        B = pred_cls.size(0)
        n_anchors = pred_cls.size(1)
        nc = pred_cls.size(2)
        device = pred_cls.device

        # Mask of valid GTs (label != 0)
        # NOTE: in IndustReal, class 0 = background. So we treat gt_labels==0 as "no GT".
        # This is a convention choice — if IndustReal uses a different convention, change this.
        gt_mask = (gt_labels > 0).float()  # [B, max_n]
        gt_labels_pos = gt_labels.clamp(min=0)  # 0 = ignore in CE

        # === Alignment metric: s = cls_score[gt_class]^alpha * box_iou^beta ===
        # pred_cls: [B, H*W, nc]
        # For each (anchor, gt) pair, take pred_cls[anchor, gt_class].
        # Shape: [B, H*W, max_n]
        # We use gather along nc dim.

        # Box IoU: [B, H*W, max_n]
        # anchors are cell centers; pred_box is decoded offsets in cells
        # Convert pred_box to xyxy in pixel space (assuming stride)
        pred_box_xyxy = pred_box.clone()
        pred_box_xyxy[..., 0:2] = anchor_points[None, :, :] - pred_box[..., 0:2] * stride
        pred_box_xyxy[..., 2:4] = anchor_points[None, :, :] + pred_box[..., 2:4] * stride

        # For each (anchor, gt): box_iou
        # We use a vectorized IoU
        iou = self._box_iou(pred_box_xyxy, gt_boxes)  # [B, H*W, max_n]

        # Alignment metric
        # pred_cls: [B, H*W, nc] — gather class score for each GT
        # gt_labels: [B, max_n] — class index for each GT
        gt_labels_for_gather = gt_labels_pos.unsqueeze(1).expand(-1, n_anchors, -1)  # [B, H*W, max_n]
        pred_cls_for_gt = torch.gather(pred_cls, 2, gt_labels_for_gather.clamp(max=nc - 1))  # [B, H*W, max_n]
        # Set pred_cls for ignore (label 0) to 0
        pred_cls_for_gt = pred_cls_for_gt * (gt_labels_for_gather > 0).float()

        align_metric = (pred_cls_for_gt ** self.alpha) * (iou ** self.beta)  # [B, H*W, max_n]
        align_metric = align_metric * gt_mask.unsqueeze(1)  # mask out invalid GTs

        # === Top-k selection per GT ===
        # For each GT, pick top-k anchors by alignment metric
        # topk_metrics: [B, max_n, topk]
        # topk_idxs: [B, max_n, topk]
        topk_metrics, topk_idxs = align_metric.topk(self.topk, dim=1)  # [B, topk, max_n]

        # === Build target_labels, target_bboxes, target_scores, mask ===
        target_labels = torch.zeros(B, n_anchors, nc, device=device)
        target_bboxes = torch.zeros(B, n_anchors, 4, device=device)
        target_scores = torch.zeros(B, n_anchors, nc, device=device)
        target_gt_idx = torch.zeros(B, n_anchors, dtype=torch.long, device=device)
        mask = torch.zeros(B, n_anchors, 1, device=device)

        # For each GT, fill in the top-k anchors
        for b in range(B):
            for gt_idx in range(gt_boxes.size(1)):
                if gt_mask[b, gt_idx] < 0.5:
                    continue
                # topk_idxs[b, gt_idx, :] are the anchor indices for this GT
                k_idxs = topk_idxs[b, gt_idx, :]  # [topk]
                k_metrics = topk_metrics[b, gt_idx, :]  # [topk]

                target_labels[b, k_idxs, gt_labels_pos[b, gt_idx]] = 1.0
                target_bboxes[b, k_idxs] = gt_boxes[b, gt_idx].unsqueeze(0).expand(self.topk, -1)
                target_scores[b, k_idxs, gt_labels_pos[b, gt_idx]] = k_metrics
                target_gt_idx[b, k_idxs] = gt_idx
                mask[b, k_idxs, 0] = 1.0

        return target_labels, target_bboxes, target_scores, mask, target_gt_idx

    @staticmethod
    def _box_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Vectorized IoU: pred [B, N, 4], gt [B, M, 4] → iou [B, N, M]."""
        # Intersection
        px1, py1, px2, py2 = pred_boxes[..., 0:1], pred_boxes[..., 1:2], pred_boxes[..., 2:3], pred_boxes[..., 3:4]
        gx1, gy1, gx2, gy2 = gt_boxes[..., 0:1], gt_boxes[..., 1:2], gt_boxes[..., 2:3], gt_boxes[..., 3:4]

        inter_x1 = torch.maximum(px1, gx1.transpose(-1, -2))
        inter_y1 = torch.maximum(py1, gy1.transpose(-1, -2))
        inter_x2 = torch.minimum(px2, gx2.transpose(-1, -2))
        inter_y2 = torch.minimum(py2, gy2.transpose(-1, -2))

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h

        area_p = (px2 - px1) * (py2 - py1)
        area_g = (gx2 - gx1) * (gy2 - gy1)
        union = area_p + area_g.transpose(-1, -2) - inter

        return inter / (union + 1e-9)
