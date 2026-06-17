"""
Tier 3.11 — Geometry-Aware Head Pose Parameterization
======================================================
Replaces the naive 9-DoF MSE regression with a proper rotation representation.

Problem: 9 raw numbers with MSE produces 60-70° angular MAE (barely better than
chance for directions). The network doesn't know these are rotation vectors — it
treats them as independent scalars with no orthogonality constraints.

Fix: Predict rotation as a 6D continuous representation (Zhou et al., CVPR 2019),
then convert to a rotation matrix via Gram-Schmidt. Position remains a separate
normalized 3-vector. Use geodesic/cosine loss for rotation, MSE for position.

Expected improvement: angular MAE drops to 10-25° with no baseline to beat.

Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
           Networks," CVPR 2019. https://arxiv.org/abs/1812.07035
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ============================================================================
# 6D → Rotation Matrix conversion (Zhou et al. 2019)
# ============================================================================
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3×3 orthonormal rotation matrix.

    Args:
        d6: [B, 6] — first 3 = a1, last 3 = a2 (two unconstrained 3-vectors)

    Returns:
        R: [B, 3, 3] — orthonormal rotation matrix (SO(3), det = +1)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # Ensure det = +1 (flip b3 if needed)
    b3 = torch.where(
        torch.det(torch.stack([b1, b2, b3], dim=-1)).unsqueeze(-1) < 0,
        -b3, b3
    )
    return torch.stack([b1, b2, b3], dim=-1)  # [B, 3, 3]


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation (inverse of above)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)  # [B, 6]


# ============================================================================
# Geodesic angular loss
# ============================================================================
def geodesic_loss(R_pred: torch.Tensor, R_true: torch.Tensor) -> torch.Tensor:
    """Geodesic (angular) distance between two rotation matrices.

    d(R_pred, R_true) = acos((tr(R_pred^T R_true) - 1) / 2) in radians.

    Args:
        R_pred: [B, 3, 3] predicted rotation matrix
        R_true: [B, 3, 3] ground-truth rotation matrix

    Returns:
        loss: scalar — mean angular error in radians
    """
    # R_pred^T @ R_true
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_true)  # [B, 3, 3]
    trace = R_rel.diagonal(dim1=1, dim2=2).sum(dim=1)  # [B]
    # cos_theta = (trace - 1) / 2, clamp for numerical stability
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # angular error in radians
    return theta.mean()


def cosine_rotation_loss(R_pred: torch.Tensor, R_true: torch.Tensor) -> torch.Tensor:
    """Cosine-based rotation loss (simpler, differentiable at 0).

    L = 1 - (1/3) * sum_i |R_pred[:,i] · R_true[:,i]|
    """
    # Dot product between corresponding column vectors
    dots = (R_pred * R_true).sum(dim=1)  # [B, 3]
    # Each column should align (cos ~ 1 for correct rotation)
    loss = 1.0 - dots.abs().sum(dim=1) / 3.0
    return loss.mean()


# ============================================================================
# Geometry-Aware Head Pose Head
# ============================================================================
class GeometryAwareHeadPose(nn.Module):
    """Predicts head pose with proper 6D rotation + normalized position.

    Output: rotation_6d [B, 6] (for forward direction + up vector) +
            position [B, 3] (normalized displacement from head origin)

    Training: geodesic/cosine loss on rotation, MSE on position.
    """

    def __init__(self, in_channels_c4: int = 384, in_channels_c5: int = 768,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_channels_c4 + in_channels_c5  # 1152

        # GAP + MLP for rotation (6D continuous)
        self.rotation_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 6),  # 6D rotation
        )

        # GAP + MLP for position (3D normalized)
        self.position_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),  # 3D position
        )

        self._init_weights()

    def _init_weights(self):
        for net in [self.rotation_net, self.position_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Rotation: init bias toward identity (forward=[0,0,1], up=[0,1,0])
        # 6D representation of identity: a1=[1,0,0], a2=[0,1,0]
        nn.init.constant_(self.rotation_net[-1].bias[:3], 0.0)  # a1 ≈ [1,0,0]
        self.rotation_net[-1].bias.data[0] = 1.0
        nn.init.constant_(self.rotation_net[-1].bias[3:], 0.0)  # a2 ≈ [0,1,0]
        self.rotation_net[-1].bias.data[4] = 1.0

    def forward(self, c4: torch.Tensor, c5: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict head pose.

        Args:
            c4: [B, 384, H/16, W/16] — C4 backbone features
            c5: [B, 768, H/32, W/32] — C5 backbone features

        Returns:
            rotation_6d: [B, 6] — 6D rotation representation
            rotation_matrix: [B, 3, 3] — orthonormal rotation matrix
            position: [B, 3] — normalized 3D position
        """
        # GAP and concat
        c4_gap = F.adaptive_avg_pool2d(c4, 1).flatten(1)  # [B, 384]
        c5_gap = F.adaptive_avg_pool2d(c5, 1).flatten(1)  # [B, 768]
        feat = torch.cat([c4_gap, c5_gap], dim=1)  # [B, 1152]

        # Rotation (6D continuous)
        rotation_6d = self.rotation_net(feat)  # [B, 6]
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)  # [B, 3, 3]

        # Position (normalized)
        position_raw = self.position_net(feat)  # [B, 3]
        position = F.tanh(position_raw)  # [-1, 1] normalized

        return rotation_6d, rotation_matrix, position

    def compute_loss(self, rotation_6d: torch.Tensor, rotation_matrix: torch.Tensor,
                     position: torch.Tensor,
                     gt_rotation_6d: torch.Tensor, gt_position: torch.Tensor,
                     rotation_weight: float = 1.0, position_weight: float = 0.1
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute geometry-aware head pose loss.

        Args:
            rotation_6d, rotation_matrix, position: predictions from forward()
            gt_rotation_6d: [B, 6] — ground-truth 6D rotation
            gt_position: [B, 3] — ground-truth position

        Returns:
            total_loss, metrics dict
        """
        # Rotation loss: geodesic (radians) + cosine (0-1)
        gt_rot_matrix = rotation_6d_to_matrix(gt_rotation_6d)
        geo_loss = geodesic_loss(rotation_matrix, gt_rot_matrix)
        cos_loss = cosine_rotation_loss(rotation_matrix, gt_rot_matrix)

        # Position loss: MSE on normalized position
        pos_loss = F.mse_loss(position, gt_position)

        total = rotation_weight * (geo_loss + 0.5 * cos_loss) + position_weight * pos_loss

        metrics = {
            'head_geo_loss_rad': geo_loss.item(),
            'head_geo_loss_deg': geo_loss.item() * 180 / 3.14159,
            'head_cos_loss': cos_loss.item(),
            'head_pos_loss': pos_loss.item(),
            'head_total_loss': total.item(),
        }

        return total, metrics

    def to_legacy_9dof(self, rotation_6d: torch.Tensor, position: torch.Tensor
                       ) -> torch.Tensor:
        """Convert back to legacy 9-DoF format for backward compatibility.

        Output: forward[3] + position[3] + up[3] = [B, 9]
        """
        rot_mat = rotation_6d_to_matrix(rotation_6d)  # [B, 3, 3]
        forward = rot_mat[:, :, 2]  # third column → forward direction
        up = rot_mat[:, :, 1]       # second column → up direction
        return torch.cat([forward, position, up], dim=1)  # [B, 9]


# ============================================================================
# Conversion utilities for existing 9-DoF labels
# ============================================================================
def legacy_9dof_to_6d_rotation(legacy_9dof: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert legacy 9-DoF [forward(3), pos(3), up(3)] to 6D+pos format.

    Args:
        legacy_9dof: [B, 9]

    Returns:
        rot_6d: [B, 6], pos: [B, 3]
    """
    forward = legacy_9dof[:, :3]
    position = legacy_9dof[:, 3:6]
    up = legacy_9dof[:, 6:9]

    # Normalize forward and up
    forward = F.normalize(forward, dim=1)
    up = F.normalize(up - (up * forward).sum(dim=1, keepdim=True) * forward, dim=1)
    right = torch.cross(forward, up, dim=1)

    # Build rotation matrix: columns are [right, up, forward]
    R = torch.stack([right, up, forward], dim=2)  # [B, 3, 3]

    # Convert to 6D
    rot_6d = rotation_matrix_to_6d(R)

    return rot_6d, F.tanh(position)  # Normalize position to [-1, 1]
