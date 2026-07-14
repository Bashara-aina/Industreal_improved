"""Multi-task loss balancer with PCGrad gradient surgery and MetaBalance
gradient magnitude rescaling.

PCGrad (Projecting Conflicting Gradients) per 175 §5.2 resolves gradient
conflicts between tasks by projecting each task's gradient onto the normal
plane of any conflicting task gradient.

MetaBalance (He et al., WWW 2022) rescales auxiliary-task gradient
magnitudes per parameter block to match a target task's gradient norm.

Algorithm (per step, for shared-backbone params only) -- PCGrad:
    for each task i:  g_i = ∇_shared (prec_i · loss_i)
    for each task i:
        g_i^PC = g_i
        for each task j != i (random order):
            if cos(g_i^PC, g_j) < 0:
                g_i^PC -= (g_i^PC · g_j / ||g_j||²) · g_j
    shared.grad = Σ_i g_i^PC
    # head params: normal per-head grads (no sharing -> no conflict)

Algorithm (per step, for shared-backbone params only) -- MetaBalance:
    for each parameter block p_i:
        target_norm = EMA ||g_target(p_i)||
        for each task k:
            scale_k = target_norm / EMA ||g_k(p_i)||  (capped [0.1, 10.0])
            g_k(p_i) *= scale_k
    shared.grad = Σ_k g_k   (rescaled)

References:
    Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020).
    He et al., "MetaBalance: Gradient Magnitude Rescaling" (WWW 2022).
"""

import random
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class MTLBalancer:
    """Multi-task loss balancer wrapping per-task losses.

    Wraps a list of per-task weighted losses and optionally applies PCGrad
    gradient surgery or MetaBalance gradient rescaling on shared backbone
    parameters to improve multi-task learning.

    Modes:
        "none":         sum(task_losses) -- standard behavior, no surgery.
        "pcgrad":       PCGrad projection on shared params.
        "metabalance":  MetaBalance gradient magnitude rescaling on shared
                        params. Requires ``task_names`` and ``target_task``.

    Integration (training loop):
        balancer = MTLBalancer(model.backbone.parameters(), mode="pcgrad")
        ...
        weighted = [prec_det * loss_det + lv_det, ...]
        combined = balancer.compute_step(weighted)
        combined.backward()
        optimizer.step()

    In PCGrad/MetaBalance modes, shared backbone params receive deconflicted
    or rescaled gradients; non-shared params (task heads, log_vars) receive
    standard gradients from the backward pass.

    Args:
        shared_params: Iterable of shared backbone nn.Parameter tensors.
            If None, both PCGrad and MetaBalance degrade to sum-of-losses.
        mode: ``"pcgrad"``, ``"metabalance"``, or ``"none"``.
        task_names: List of task names matching the order of ``task_losses``
            passed to ``compute_step``. Required for ``"metabalance"`` mode.
        target_task: Task whose gradient norm MetaBalance targets
            (default: ``"head_pose"``). Must be in ``task_names``.
    """

    def __init__(
        self,
        shared_params: Optional[List[nn.Parameter]] = None,
        mode: str = "none",
        task_names: Optional[List[str]] = None,
        target_task: str = "head_pose",
    ):
        self.shared_params = list(shared_params) if shared_params is not None else []
        self.mode = mode
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._step_counter: int = 0

        # ---- MetaBalance-specific state ----
        if mode == "metabalance":
            if task_names is None or len(task_names) < 2:
                raise ValueError(
                    "MetaBalance requires task_names with at least 2 tasks"
                )
            if target_task not in task_names:
                raise ValueError(
                    f"target_task '{target_task}' not in task_names {task_names}"
                )
            self._mb_task_names: List[str] = list(task_names)
            self._mb_target_task: str = target_task
            self._mb_alpha: float = 0.9
            self._mb_eps: float = 1e-8
            # _mb_norms[param_idx][task_idx] = EMA of gradient norm
            self._mb_norms: List[List[float]] = [
                [0.0] * len(task_names) for _ in self.shared_params
            ]
        else:
            self._mb_task_names = []
            self._mb_target_task = ""
            self._mb_alpha = 0.0
            self._mb_eps = 0.0
            self._mb_norms = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_step(self, task_losses: List[torch.Tensor]) -> torch.Tensor:
        """Compute combined loss with optional PCGrad / MetaBalance.

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

        In ``"metabalance"`` mode:
            1. Computes per-task gradients via ``autograd.grad``.
            2. Updates EMA gradient norms per-parameter-block per-task.
            3. Rescales each task's gradient per block to match the
               target task's EMA norm (scale capped [0.1, 10.0]).
            4. Installs backward hooks with rescaled + summed grads.
            5. Returns the summed loss for non-shared params.

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
        # 2. Gradient surgery: PCGrad or MetaBalance
        # --------------------------------------------------------------
        if self.mode == "metabalance":
            projected_grads = self._project_metabalance(task_grads)
        else:
            projected_grads = self._project_pcgrad(task_grads)

        # --------------------------------------------------------------
        # 3. Install backward hooks that override shared-param grads
        # --------------------------------------------------------------
        for param, grad in zip(self.shared_params, projected_grads):
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
        if self.mode == "metabalance":
            self._mb_norms = [
                [0.0] * len(self._mb_task_names) for _ in self.shared_params
            ]

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
    # MetaBalance core
    # ------------------------------------------------------------------

    def _project_metabalance(
        self,
        task_grads: List[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, ...]:
        """Apply MetaBalance gradient magnitude rescaling.

        For each shared parameter block p_i:
            target_norm = EMA ||g_target(p_i)||
            For each task k:
                scale_k = target_norm / EMA ||g_k(p_i)||
                scale_k = clamp(scale_k, 0.1, 10.0)
                g_k(p_i) *= scale_k
        Returns summed rescaled gradients across all tasks.

        Uses EMA-smoothed gradient norms per parameter block per task,
        tracked across training steps with momentum ``_mb_alpha``.

        Args:
            task_grads: ``task_grads[t]`` is a tuple of per-parameter
                gradients for task *t* (same length as ``shared_params``).

        Returns:
            Tuple of per-parameter summed rescaled gradients (same
            structure as ``shared_params``).
        """
        n_tasks = len(task_grads)
        n_params = len(self.shared_params)
        target_idx = self._mb_task_names.index(self._mb_target_task)
        alpha = self._mb_alpha
        eps = self._mb_eps

        # 1. Update EMA gradient norms from current batch.
        for i in range(n_params):
            for k in range(n_tasks):
                g = task_grads[k][i]
                if g is not None:
                    gn = g.norm().item()
                    prev = self._mb_norms[i][k]
                    self._mb_norms[i][k] = alpha * prev + (1 - alpha) * gn

        # 2. Rescale per-parameter-block per-task, then sum.
        result: List[torch.Tensor] = []
        for i, param in enumerate(self.shared_params):
            target_norm = max(self._mb_norms[i][target_idx], eps)
            if target_norm <= 0 or not torch.isfinite(
                torch.tensor(target_norm)
            ):
                # Fallback: plain sum (before EMA warms up).
                pieces = [
                    task_grads[k][i]
                    for k in range(n_tasks)
                    if task_grads[k][i] is not None
                ]
                result.append(
                    sum(pieces) if pieces else torch.zeros_like(param.data)
                )
                continue

            accumulator = torch.zeros_like(param.data)
            for k in range(n_tasks):
                g = task_grads[k][i]
                if g is None:
                    continue
                norm_k = max(self._mb_norms[i][k], eps)
                scale = target_norm / norm_k
                scale = max(0.1, min(10.0, scale))
                accumulator += g * scale
            result.append(accumulator)

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
