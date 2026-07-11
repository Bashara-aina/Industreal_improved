"""LDAM-DRW (Cao et al. 2019 NeurIPS)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list: List[int], max_m: float = 0.5, s: float = 30, reweight_epoch: int = 35):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float)))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer('m_list', m_list)
        self.cls_num_list = cls_num_list
        self.s = s
        self.reweight_epoch = reweight_epoch
        self.is_drw = False

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int) -> torch.Tensor:
        if epoch >= self.reweight_epoch:
            self.is_drw = True
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), 1)
        index_float = index.float()
        batch_m_list = self.m_list.unsqueeze(0) * index_float
        logits_m = logits - batch_m_list * self.s
        if self.is_drw:
            weights = 1.0 / torch.sqrt(torch.tensor(self.cls_num_list, dtype=torch.float))
            weights = weights / weights.sum() * len(self.cls_num_list)
            return F.cross_entropy(logits_m, targets, weight=weights.to(logits.device))
        return F.cross_entropy(logits_m, targets)
