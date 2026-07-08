"""Multi-task loss balancer with PCGrad gradient surgery.

Implements PCGrad (Projecting Conflicting Gradients) per 175 §5.2.
PCGrad resolves gradient conflicts between tasks by projecting each
task's gradient onto the normal plane of any conflicting task gradient.

Algorithm (per step, for shared-backbone params only):
    for each task i:  g_i = ∇_shared (prec_i · loss_i)
    for each task i:
        g_i^PC = g_i
        for each task j != i (random order):
            if cos(g_i^PC, g_j) < 0:
                g_i^PC -= (g_i^PC · g_j / ||g_j||²) · g_j
    shared.grad = Σ_i g_i^PC
    # head params: normal per-head grads (no sharing -> no conflict)

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020).
"""

import random
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class MTLBalancer:
    """Multi-task loss balancer wrapping per-task losses with PCGrad.

    Wraps a list of per-task weighted losses and optionally applies PCGrad
    gradient surgery on shared backbone parameters to resolve conflicts.

    Modes:
        "none":    sum(task_losses) -- standard behavior, no surgery.
        "pcgrad":  PCGrad projection on shared params.

    Integration (training loop):
        balancer = MTLBalancer(model.backbone.parameters(), mode="pcgrad")
        ...
        weighted = [prec_det * loss_det + lv_det, ...]
        combined = balancer.compute_step(weighted)
        combined.backward()
        optimizer.step()

    In PCGrad mode, shared backbone params receive deconflicted gradients;
    non-shared params (task heads, log_vars) receive standard gradients from
    the backward pass.

    Args:
        shared_params: Iterable of shared backbone nn.Parameter tensors.
            If None, PCGrad degrades to sum-of-losses (no params to project).
        mode: ``"pcgrad"`` for gradient surgery, ``"none"`` for standard sum.
    """

    def __init__(
        self,
        shared_params: Optional[List[nn.Parameter]] = None,
        mode: str = "none",
    ):
        self.shared_params = list(shared_params) if shared_params is not None else []
        self.mode = mode
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_step(self, task_losses: List[torch.Tensor]) -> torch.Tensor:
        """Compute combined loss with optional PCGrad projection.

        In ``"pcgrad"`` mode:
            1. Computes per-task gradients w.r.t. shared backbone params
               using ``torch.autograd.grad(retain_graph=True)``.
            2. Applies PCGrad projection: for each task i, removes
               conflicting components from all other tasks j where
               ``cos(g_i, g_j) < 0``.
            3. Installs backward hooks on shared params to replace
               their gradients with the deconflicted result.
            4. Returns the summed loss so that ``.backward()`` flows
               normally for non-shared params (head weights, log_vars).

        In ``"none"`` mode: returns ``sum(task_losses)`` unchanged.

        Args:
            task_losses: List of per-task scalar loss tensors (already
                precision-weighted, e.g., ``prec_i * loss_i + lv_i``).

        Returns:
            Combined loss tensor. The caller should call ``.backward()``
            on it, then ``optimizer.step()``.
        """
        self._step_counter += 1
        self._remove_hooks()

        # Fast path: no surgery needed.
        if self.mode == "none" or len(task_losses) <= 1 or not self.shared_params:
            return sum(task_losses)

        # --------------------------------------------------------------
        # 1. Per-task gradients w.r.t. shared backbone params
        # --------------------------------------------------------------
        task_grads: List[Tuple[torch.Tensor, ...]] = []
        for i, loss in enumerate(task_losses):
            g = torch.autograd.grad(
                loss,
                self.shared_params,
                retain_graph=True,  # keep graph for subsequent tasks + backward
                create_graph=False,
                allow_unused=True,
            )
            # Replace None (param unreachable from this task's loss) with zero
            g = tuple(
                torch.zeros_like(p) if gi is None else gi
                for gi, p in zip(g, self.shared_params)
            )
            task_grads.append(g)

        # --------------------------------------------------------------
        # 2. PCGrad projection (flat-vector Gram-Schmidt)
        # --------------------------------------------------------------
        pcgrad_grads = self._project_pcgrad(task_grads)

        # --------------------------------------------------------------
        # 3. Install backward hooks that override shared-param grads
        # --------------------------------------------------------------
        for param, grad in zip(self.shared_params, pcgrad_grads):
            _grad = grad.clone()  # capture before it goes out of scope
            hook = param.register_hook(lambda g, pg=_grad: pg)
            self._hooks.append(hook)

        # --------------------------------------------------------------
        # 4. Return summed loss for backward on non-shared params
        # --------------------------------------------------------------
        return sum(task_losses)

    @torch.no_grad()
    def set_shared_params(self, params: List[nn.Parameter]) -> None:
        """Update the shared parameter list (e.g., after model surgery)."""
        self.shared_params = list(params)

    @property
    def has_hooks(self) -> bool:
        """Whether backward hooks are currently installed."""
        return len(self._hooks) > 0

    # ------------------------------------------------------------------
    # PCGrad core
    # ------------------------------------------------------------------

    def _project_pcgrad(
        self,
        task_grads: List[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, ...]:
        """Apply PCGrad projection to the per-task gradient list.

        For each task *i*, Gram-Schmidt projects out the component of
        every other task *j* (random order) whose cosine similarity is
        negative.  Returns per-parameter deconflicted summed gradients.

        Args:
            task_grads: ``task_grads[t]`` is a tuple of per-parameter
                gradients for task *t* (same length as ``shared_params``).

        Returns:
            Tuple of per-parameter summed deconflicted gradients (same
            structure as ``shared_params``).
        """
        n_tasks = len(task_grads)

        # Flatten each task's gradient into a single 1-D vector.
        flat_grads: List[torch.Tensor] = []
        for grads in task_grads:
            pieces = [g.contiguous().view(-1) for g in grads]
            flat_grads.append(torch.cat(pieces))

        # PCGrad: project out conflicting components.
        pc_grads = [g.clone() for g in flat_grads]
        for i in range(n_tasks):
            order = [j for j in range(n_tasks) if j != i]
            # Per-step random permutation for unbiased conflict resolution.
            seed = 42 + self._step_counter * n_tasks + i
            rng = random.Random(seed)
            rng.shuffle(order)

            for j in order:
                dot = torch.dot(pc_grads[i], flat_grads[j])
                if dot < 0:  # conflicting gradients
                    g_j_norm_sq = torch.dot(flat_grads[j], flat_grads[j])
                    if g_j_norm_sq > 1e-12:
                        coeff = dot / g_j_norm_sq
                        pc_grads[i] = pc_grads[i] - coeff * flat_grads[j]

        # Sum deconflicted gradients across tasks.
        combined_flat = sum(pc_grads)

        # Unflatten back to per-parameter structure.
        result: List[torch.Tensor] = []
        offset = 0
        for param in self.shared_params:
            numel = param.numel()
            result.append(combined_flat[offset : offset + numel].view(param.shape))
            offset += numel

        return tuple(result)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _remove_hooks(self) -> None:
        """Remove all installed backward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Utility: detect shared params from model
    # ------------------------------------------------------------------

    @staticmethod
    def detect_shared_params(
        model: nn.Module,
        task_losses: Optional[List[torch.Tensor]] = None,
    ) -> List[nn.Parameter]:
        """Detect parameters shared by all tasks.

        If ``task_losses`` is provided, only parameters that receive
        non-``None`` gradient from **every** task are considered shared.
        Otherwise returns all ``requires_grad`` parameters.

        Args:
            model: The multi-task model.
            task_losses: Optional list of per-task loss tensors to probe.

        Returns:
            List of shared ``nn.Parameter`` tensors.
        """
        all_params = [p for p in model.parameters() if p.requires_grad]
        if not task_losses:
            return all_params

        shared: List[nn.Parameter] = []
        for param in all_params:
            in_all = True
            for loss in task_losses:
                g = torch.autograd.grad(
                    loss, param, retain_graph=True, allow_unused=True
                )
                if g[0] is None:
                    in_all = False
                    break
            if in_all:
                shared.append(param)
        return shared
