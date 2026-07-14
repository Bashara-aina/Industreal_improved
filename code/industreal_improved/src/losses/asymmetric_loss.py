"""Asymmetric Loss for extreme-imbalance binary classification (Ridnik et al. 2021 ICCV)."""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        prob = torch.clamp(prob, self.clip, 1 - self.clip)
        targets = targets.float()
        pos_loss = -targets * torch.log(prob) * torch.pow(1 - prob, self.gamma_pos)
        neg_loss = -(1 - targets) * torch.log(1 - prob) * torch.pow(prob, self.gamma_neg)
        return (pos_loss + neg_loss).mean()
