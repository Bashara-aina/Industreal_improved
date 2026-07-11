"""UW-SO: Uncertainty-Weighted Softmax Ordinal weighting (Kirchdorfer 2025 IJCV)."""
import torch
import torch.nn.functional as F
from typing import Dict


def uw_so_loss(losses: Dict[str, torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    """weights = softmax(-stop_gradient(losses) / temperature)."""
    loss_tensor = torch.stack(list(losses.values()))
    with torch.no_grad():
        weights = F.softmax(-loss_tensor / temperature, dim=0)
    return (loss_tensor * weights).sum()
