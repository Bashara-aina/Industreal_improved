"""Varifocal Loss (Zhang et al. 2021 CVPR Oral) — IoU-aware classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: [N, C] logits, target: [N, C] one-hot with IoU values for positive class."""
        # target[:, c] = IoU for positive class, 0 for negative
        pred_score = pred.sigmoid()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
        return loss.sum() / max(target.sum(), 1.0)
