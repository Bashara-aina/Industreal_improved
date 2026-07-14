"""IMTL-L: Loss-balance component of IMTL (Liu et al., ICLR 2021).

Standalone closed-form scalar weighting: w_k = softmax(-log(L_k)).
Single backward pass, O(1) complexity, stateless.
Inverse-log-loss weighting directly targets gradient magnitude imbalance.

Reference: "Towards Impartial Multi-Task Learning"
https://openreview.net/forum?id=IMPnRXEWpvr
"""

import torch
import torch.nn.functional as F
from typing import Dict


def imtl_l_loss(losses: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """IMTL-L weighting: w_k = softmax(-log(L_k)).

    Unlike UW-SO (softmax on raw losses), IMTL-L operates in log-space,
    making it robust to extreme loss ratios. A 312x loss ratio becomes
    ~5.7x in weight ratio instead of ~312x.

    Args:
        losses: dict of scalar task loss tensors.
        eps: small constant to prevent log(0).
    Returns:
        Weighted total loss (scalar).
    """
    loss_tensor = torch.stack(list(losses.values()))
    with torch.no_grad():
        log_losses = torch.log(loss_tensor.clamp(min=eps))
        weights = F.softmax(-log_losses, dim=0)
    return (loss_tensor * weights).sum()
