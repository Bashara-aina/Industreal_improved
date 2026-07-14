"""WIoU v3 (Tong et al. 2023) — dynamic non-monotonic IoU loss for detection box regression."""
import torch


def wiou_v3_loss(pred: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """WIoU v3 with dynamic non-monotonic focusing.
    pred/target: [N, 4] xyxy format. anchor: [N, 4] anchor boxes.
    """
    eps = 1e-7
    # Intersection
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    # Union
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_p + area_t - inter + eps
    iou = inter / union
    # Enclosing box
    ex1 = torch.min(pred[:, 0], target[:, 0])
    ey1 = torch.min(pred[:, 1], target[:, 1])
    ex2 = torch.max(pred[:, 2], target[:, 2])
    ey2 = torch.max(pred[:, 3], target[:, 3])
    ew, eh = (ex2 - ex1).clamp(min=eps), (ey2 - ey1).clamp(min=eps)
    # Distance term
    cx_p, cy_p = (pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2
    cx_t, cy_t = (target[:, 0] + target[:, 2]) / 2, (target[:, 1] + target[:, 3]) / 2
    dw = ((cx_p - cx_t) / ew).pow(2) + ((cy_p - cy_t) / eh).pow(2)
    # WIoU v1
    wiou_v1 = iou * torch.exp(-dw)
    # Dynamic non-monotonic focusing (v3)
    with torch.no_grad():
        if anchor is not None:
            ax1, ay1 = anchor[:, 0], anchor[:, 1]
            ax2, ay2 = anchor[:, 2], anchor[:, 3]
            a_inter = (torch.min(ax2, target[:, 2]) - torch.max(ax1, target[:, 0])).clamp(min=0) * \
                      (torch.min(ay2, target[:, 3]) - torch.max(ay1, target[:, 1])).clamp(min=0)
            a_union = (ax2 - ax1) * (ay2 - ay1) + area_t - a_inter + eps
            anchor_iou = a_inter / a_union
        else:
            anchor_iou = iou
        beta = (iou / anchor_iou.clamp(min=eps)).detach()
        # WIoU v3 dynamic non-monotonic focusing (Tong et al. 2023):
        # r = delta / (alpha^(beta - delta)) with delta=1.3, alpha=3.0
        # The original exp(beta-beta) was a bug that always produced r=1.0.
        delta = 1.3
        alpha = 3.0
        r = delta / (alpha ** (beta - delta))
    return ((1 - wiou_v1) * r).mean()
