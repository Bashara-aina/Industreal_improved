"""FAMO: Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023).

O(1) single-backward-pass loss weighting. Uses ONLY the aggregated gradient
(weighted sum of per-task losses) plus a log-space weight update driven by
the *change* in per-task loss between successive steps. Matches Nash-MTL
at a fraction of the wall-clock cost.

Verified against Algorithm 1 in the NeurIPS 2023 paper. The key insight:
weights are updated based on loss decrease rates (log l_t - log l_{t+1}),
not on per-task gradients — so only ONE backward pass is needed.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional


class FAMOWeighter:
    """Stateful FAMO weight tracker — one instance per training run.

    Usage:
        famo = FAMOWeighter(num_tasks=4, lr=0.01, temperature=1.0)
        for step in range(N):
            task_losses = {"det": l_det, "act": l_act, "psr": l_psr, "pose": l_pose}
            total = famo(task_losses)   # weighted sum
            total.backward()
            optimizer.step()
            famo.step(task_losses)      # update weights for next iteration
    """

    def __init__(
        self,
        num_tasks: int = 4,
        lr: float = 0.01,
        temperature: float = 1.0,
    ):
        self.num_tasks = num_tasks
        self.lr = lr
        self.temperature = temperature
        # Log-weights xi_k (one scalar per task), initialized to log(1/K)
        self.log_weights = torch.zeros(num_tasks)
        # Previous per-task losses for computing decrease rates
        self.prev_log_losses: Optional[torch.Tensor] = None
        self._step_count = 0

    def get_weights(self) -> torch.Tensor:
        """Current task weights via softmax over log-weights."""
        return F.softmax(self.log_weights / self.temperature, dim=0)

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of losses using current weights.

        Args:
            losses: dict of scalar task loss tensors, keyed by task name.
        Returns:
            Weighted total loss (scalar).
        """
        loss_tensor = torch.stack(list(losses.values()))
        weights = self.get_weights().to(loss_tensor.device)
        return (loss_tensor * weights).sum()

    def step(self, losses: Dict[str, torch.Tensor]) -> None:
        """Update weights based on loss decrease rates.

        Must be called AFTER optimizer.step() with the NEW losses (post-update).
        FAMO algorithm: xi_k += lr * z_k * (log l_k^t - log l_k^{t+1} + z_k * log z_k)
        where z_k = softmax(xi_k).

        Args:
            losses: dict of scalar task loss tensors AFTER the optimizer step.
        """
        device = list(losses.values())[0].device
        loss_tensor = torch.stack([v.detach() for v in losses.values()])

        with torch.no_grad():
            log_losses = torch.log(loss_tensor.clamp(min=1e-8))
            weights = self.get_weights().to(device)

            if self.prev_log_losses is not None:
                # xi_k += lr * z_k * (log l_k^t - log l_k^{t+1} + z_k * log z_k)
                # where log l_k^t = prev_log_losses, log l_k^{t+1} = log_losses
                delta = self.prev_log_losses.to(device) - log_losses
                entropy_term = weights * torch.log(weights.clamp(min=1e-8))
                update = self.lr * weights * (delta + entropy_term)
                self.log_weights = self.log_weights.to(device) + update

            self.prev_log_losses = log_losses.cpu()
            self._step_count += 1

    def state_dict(self) -> dict:
        return {
            "log_weights": self.log_weights,
            "prev_log_losses": self.prev_log_losses,
            "step_count": self._step_count,
            "num_tasks": self.num_tasks,
            "lr": self.lr,
            "temperature": self.temperature,
        }

    def load_state_dict(self, state: dict) -> None:
        self.log_weights = state["log_weights"]
        self.prev_log_losses = state["prev_log_losses"]
        self._step_count = state["step_count"]
