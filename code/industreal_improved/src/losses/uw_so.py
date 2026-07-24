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

    log_sigma is clipped to [log_sigma_min, log_sigma_max] each forward()
    to prevent pathological divergence where one task gets completely
    suppressed (log_sigma → +inf → weight → 0). With the default bounds
    [-1.0, 2.0], the min weight is exp(-4) ≈ 0.018 and max is exp(2) ≈ 7.4.

    Task order (4 tasks): det, act, pose, psr
    """

    TASK_NAMES = ("det", "act", "pose", "psr")
    LOG_SIGMA_MIN = -1.0
    LOG_SIGMA_MAX = 2.0

    def __init__(self, init_log_sigma: float = 0.0):
        super().__init__()
        # Initialize within bounds
        init_val = max(self.LOG_SIGMA_MIN, min(self.LOG_SIGMA_MAX, init_log_sigma))
        self.log_sigma = nn.Parameter(
            torch.full((len(self.TASK_NAMES),), init_val)
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
                weight = torch.exp(-2.0 * ls)
                total = total + weight * losses[name] + ls
        return total

    def project(self) -> None:
        """Clip log_sigma parameters to [min, max] after optimizer step.

        Must be called after each uwso_opt.step() to prevent pathological
        divergence where log_sigma → +inf and a task gets completely
        suppressed.
        """
        with torch.no_grad():
            self.log_sigma.data.clamp_(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)
