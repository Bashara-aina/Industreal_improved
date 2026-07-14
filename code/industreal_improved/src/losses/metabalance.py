"""MetaBalance: Gradient Magnitude Rescaling (He et al., WWW 2022).

Rescales auxiliary-task gradient magnitudes per parameter block to match
a target task's gradient norm. Directly attacks the 312× gradient magnitude
gap between activity (0.010) and PSR (3.18).

Unlike loss weighting (which operates on scalar losses), MetaBalance operates
on per-parameter gradients — rescaling STARVED head gradients UP and DOMINANT
head gradients DOWN at the parameter-block level.

Implementation: backward hooks on shared backbone parameters. After each
task's gradient is computed, rescale to match the target task's norm.

Usage:
    balancer = MetaBalance(target_task="pose", backbone_params=shared_params)
    # Register hooks
    balancer.register()
    # After each backward pass through all tasks, rescale
    balancer.rescale()
"""
import torch
import torch.nn as nn
from typing import List, Optional, Dict


class MetaBalance:
    """Gradient magnitude rescaling per parameter block.

    For each shared backbone parameter p, computes per-task gradient norms
    ||g_k(p)|| and rescales so that all tasks have the same gradient norm
    as the target task on that parameter block.

    Args:
        target_task: which task's gradient magnitude to match (default: "pose")
        alpha: EMA momentum for tracking per-block gradient norms (default: 0.9)
        eps: small constant for numerical stability
    """

    def __init__(
        self,
        target_task: str = "pose",
        alpha: float = 0.9,
        eps: float = 1e-8,
    ):
        self.target_task = target_task
        self.alpha = alpha
        self.eps = eps
        # Per-parameter, per-task running gradient norm estimates
        self._norms: Dict[int, Dict[str, float]] = {}
        self._params: List[nn.Parameter] = []
        self._task_names: List[str] = []

    def register(
        self,
        params: List[nn.Parameter],
        task_names: List[str],
    ) -> None:
        """Register shared backbone parameters for rescaling.

        Args:
            params: list of shared backbone parameters (requires_grad=True)
            task_names: ordered list of task names matching gradient order
        """
        self._params = params
        self._task_names = task_names
        for i, p in enumerate(params):
            self._norms[i] = {name: 0.0 for name in task_names}

    def record_grad(self, task_name: str, task_idx: int) -> None:
        """Record gradient norms for one task after its backward pass.

        Call after computing grads for task `task_name` on shared params.
        Uses EMA to track running norm.
        """
        alpha = self.alpha
        for i, p in enumerate(self._params):
            if p.grad is not None:
                gn = p.grad.norm().item()
                prev = self._norms[i][task_name]
                self._norms[i][task_name] = alpha * prev + (1 - alpha) * gn

    def rescale(
        self,
        per_task_grads: List[List[Optional[torch.Tensor]]],
    ) -> List[torch.Tensor]:
        """Rescale per-task gradients to match target task's magnitude.

        For each parameter p_i:
          target_norm = ||g_target(p_i)||
          For each task k:
            scale_k = target_norm / ||g_k(p_i)||   (capped at [0.1, 10.0])
            g_k(p_i) *= scale_k

        Args:
            per_task_grads: list of per-task gradient lists, aligned with self._params.
                           per_task_grads[k][i] = grad of task k w.r.t. param i

        Returns:
            Sum of rescaled gradients per parameter (for direct .grad assignment).
        """
        target_idx = self._task_names.index(self.target_task)
        n_tasks = len(self._task_names)
        n_params = len(self._params)

        summed_grads = [torch.zeros_like(p.data) for p in self._params]

        for i in range(n_params):
            # Use EMA-tracked gradient norms for stability, fall back to
            # instant norms if EMA hasn't accumulated (first few steps).
            norms = []
            for k in range(n_tasks):
                g = per_task_grads[k][i]
                if g is not None and self._norms[i][self._task_names[k]] > 0:
                    # Use EMA norm (smooth, across-batch average)
                    gn = self._norms[i][self._task_names[k]]
                elif g is not None:
                    gn = g.norm().item()
                else:
                    gn = 0.0
                norms.append(max(gn, self.eps))

            target_norm = norms[target_idx]

            for k in range(n_tasks):
                g = per_task_grads[k][i]
                if g is None:
                    continue
                scale = target_norm / max(norms[k], self.eps)
                # Cap scale to prevent extreme rescaling (stability)
                scale = max(0.1, min(10.0, scale))
                summed_grads[i] = summed_grads[i] + g * scale

        return summed_grads
