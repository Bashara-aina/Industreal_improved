"""UW-SO: Uncertainty-Weighted loss balancing for multi-task learning.

Contains two implementations:
  - uw_so_loss:  Static softmax-based weighting (Kirchdorfer 2025).
  - UWSOLoss:    Learnable homoscedastic uncertainty weighting (Kendall et al. 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def uw_so_loss(losses: dict[str, torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    """weights = softmax(-stop_gradient(losses) / temperature)."""
    loss_tensor = torch.stack(list(losses.values()))
    with torch.no_grad():
        weights = F.softmax(-loss_tensor / temperature, dim=0)
    return (loss_tensor * weights).sum()


class UWSOLoss(nn.Module):
    """Learnable uncertainty weighting for multi-task loss balancing.

    Implements the homoscedastic uncertainty formulation from
    Kendall et al. (2018, "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics"):
        L_total = sum_i (1/sigma_i^2 * L_i + log_sigma_i)

    Each task has a learnable log_sigma parameter. With init_log_sigma=0.0,
    the initial weight for each task is 1.0 (since exp(-2*0) = 1).

    Task order (4 tasks): det, act, pose, psr
    """

    TASK_NAMES = ("det", "act", "pose", "psr")

    def __init__(self, init_log_sigma: float = 0.0):
        super().__init__()
        self.log_sigma = nn.Parameter(
            torch.full((len(self.TASK_NAMES),), init_log_sigma)
        )

    @property
    def sigma(self) -> torch.Tensor:
        """Return standard deviations (detached, for logging)."""
        return torch.exp(self.log_sigma.detach())

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply UW-SO weighting to task losses.

        Only tasks present in *losses* are weighted; missing tasks are
        skipped (no gradient for that log_sigma on this batch).

        Args:
            losses: dict mapping task name -> raw loss scalar tensor.
                    Task names must be a subset of TASK_NAMES.

        Returns:
            Weighted total loss (scalar tensor).
        """
        device = next(iter(losses.values())).device
        total = torch.tensor(0.0, device=device)
        for i, name in enumerate(self.TASK_NAMES):
            if name in losses:
                ls = self.log_sigma[i]
                # weight = 1 / sigma^2 = exp(-2 * log_sigma)
                weight = torch.exp(-2.0 * ls)
                total = total + weight * losses[name] + ls
        return total
