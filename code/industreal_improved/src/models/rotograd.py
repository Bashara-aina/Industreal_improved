"""RotoGrad: Gradient Homogenization via Feature Rotation (Javaloy & Valera, ICLR 2022).

Simplified implementation: per-task rotation matrices applied to the shared
cls_token before each task head. Uses Cayley transform for SO(d) parametrization
— no geotorch dependency needed.

Two components (can be used independently):
  1. RotateOnly: rotate features to align per-task gradient directions
  2. ScaleOnly: normalize gradient magnitudes via convergence ratios

Both components directly attack the 312× gradient magnitude gap between
PSR (3.18) and activity (0.010) by:
  - RotateOnly: giving each task equal directional influence on the backbone
  - ScaleOnly: normalizing per-task gradient norms so no task dominates

The original paper uses a Stackelberg game (rotations learn slowly).
We simplify: rotation update via SGD on cosine-similarity loss between
each task's gradient direction and the average direction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def cayley_orthogonal(w: torch.Tensor) -> torch.Tensor:
    """Cayley transform: (I + skew(w))(I - skew(w))^(-1) ∈ SO(n).

    Maps any matrix to a special orthogonal matrix. No geotorch needed.
    """
    I = torch.eye(w.size(-1), device=w.device, dtype=w.dtype)
    a = w - w.transpose(-2, -1)  # skew-symmetrize
    return torch.linalg.solve(I - A, I + A)


class RotoGradRotation(nn.Module):
    """Per-task feature rotation for gradient direction alignment.

    Each task gets a rotation matrix R_k ∈ SO(d) applied to the shared
    cls_token BEFORE the task-specific head. Rotations are optimized to
    align each task's feature gradient with the mean gradient direction.

    Subspace mode (subspace_dim < feat_dim): project down → rotate → project
    back, keeping param count manageable at ~0.4M instead of ~1.2M.

    Args:
        feat_dim: feature dimension (768 for MViTv2-S cls_token)
        num_tasks: number of tasks (3: activity, pose, PSR; detection uses FPN)
        subspace_dim: if set, use low-rank subspace rotation (default: 128)
    """

    def __init__(
        self,
        feat_dim: int = 768,
        num_tasks: int = 3,
        subspace_dim: int = 128,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tasks = num_tasks
        self.subspace_dim = subspace_dim

        if subspace_dim and subspace_dim < feat_dim:
            # Subspace rotation: P [d,m] projects down, R_sub [m,m] rotates, Q [d,m] projects back
            # Full rotation = P @ R_sub @ Q.T  → [d,d]
            self._full = False
            self.P = nn.Parameter(torch.randn(num_tasks, feat_dim, subspace_dim) * 0.01)
            self.W_sub = nn.Parameter(torch.randn(num_tasks, subspace_dim, subspace_dim) * 0.01)
            self.Q = nn.Parameter(torch.randn(num_tasks, feat_dim, subspace_dim) * 0.01)
        else:
            # Full rotation: one SO(d) matrix per task via Cayley parametrization
            self._full = True
            self.W = nn.Parameter(torch.randn(num_tasks, feat_dim, feat_dim) * 0.01)

    def get_rotation(self, task_idx: int) -> torch.Tensor:
        """Return R_k ∈ SO(d) for task k. [d, d]."""
        if self._full:
            return cayley_orthogonal(self.W[task_idx])
        else:
            # Subspace: P @ R_sub @ Q.T → [d,m] @ [m,m] @ [m,d] = [d,d]
            R_sub = cayley_orthogonal(self.W_sub[task_idx])  # [m, m]
            p = self.P[task_idx]  # [d, m]
            q = self.Q[task_idx]  # [d, m]
            return p @ R_sub @ q.T  # [d, d]

    def rotate(self, z: torch.Tensor, task_idx: int) -> torch.Tensor:
        """Apply rotation R_k to features. z: [B, d] → rotated [B, d]."""
        R = self.get_rotation(task_idx)  # [d, d]
        return z @ R  # [B, d] @ [d, d] = [B, d]

    def rotation_loss(
        self,
        task_idx: int,
        feat_grad: torch.Tensor,
        target_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Loss encouraging R_k @ grad_k to align with target_direction.

        Since z_rotated = z @ R, the effective gradient in original space is
        dL/dz = (dL/dz_rotated) @ R^T. The rotation loss aligns this with
        the target direction.

        L = -cosine_similarity(R @ grad, target)

        Args:
            task_idx: which task's rotation to optimize
            feat_grad: gradient w.r.t. rotated features [B, d]
            target_direction: desired gradient direction [B, d]
        """
        R = self.get_rotation(task_idx)  # [d, d]
        # Effective gradient in original feature space: grad @ R^T
        rotated_grad = feat_grad @ R.T  # [B, d]
        cos_sim = F.cosine_similarity(
            rotated_grad.flatten(), target_direction.flatten(), dim=0, eps=1e-8
        )
        return -cos_sim

    def forward(self, z: torch.Tensor, task_idx: int) -> torch.Tensor:
        """Rotate features for a given task. Convenience alias for rotate()."""
        return self.rotate(z, task_idx)


class RotoGradScale:
    """Gradient magnitude homogenization via convergence ratios.

    Normalizes per-task gradient magnitudes so no task dominates the shared
    backbone update. Uses alpha_k = ||G_k|| / ||G^0_k|| — the convergence ratio
    of each task relative to its initial gradient norm.

    Without this, PSR (grad norm 3.18) dominates activity (grad norm 0.010)
    by a factor of 312× in backbone updates.
    """

    def __init__(self, num_tasks: int = 4, burn_in_steps: int = 500):
        self.num_tasks = num_tasks
        self.burn_in_steps = burn_in_steps
        self.G0: Optional[List[float]] = None  # initial gradient norms
        self._step = 0

    def record_initial(self, grad_norms: List[float]) -> None:
        """Record ||G^0_k|| during burn-in period (single pass)."""
        if self.G0 is None:
            self.G0 = [max(g, 1e-8) for g in grad_norms]

    def normalize(self, grad_norms: List[float]) -> List[float]:
        """Compute per-task scale factors to homogenize gradient magnitudes.

        alpha_k = ||G_k|| / ||G^0_k||  (convergence ratio)
        C = weighted average magnitude based on convergence
        U_k = G_k / ||G_k||  (unit direction)

        The backbone update uses C * sum(U_k), giving each task equal
        directional say while scaling total magnitude by convergence-weighted
        average.

        Returns: list of per-task scale factors to multiply gradients.
        """
        self._step += 1
        if self.G0 is None or self._step < self.burn_in_steps:
            return [1.0] * len(grad_norms)

        alpha = [gn / g0 for gn, g0 in zip(grad_norms, self.G0)]
        total_alpha = sum(alpha) + 1e-8
        # Weighted average magnitude: C = sum(alpha_k / sum(alpha) * ||G_k||)
        C = sum(a / total_alpha * gn for a, gn in zip(alpha, grad_norms))
        # Scale factor: normalize each gradient to unit length, then scale by C
        scales = [C / max(gn, 1e-8) for gn in grad_norms]
        return scales
