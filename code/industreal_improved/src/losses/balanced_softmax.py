"""Balanced Softmax (Ren et al. 2020 NeurIPS)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_priors: torch.Tensor):
        super().__init__()
        self.register_buffer('class_priors', class_priors)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_shifted = logits + torch.log(self.class_priors.unsqueeze(0))
        return F.cross_entropy(logits_shifted, targets)
