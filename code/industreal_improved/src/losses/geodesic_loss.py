"""Huberised Geodesic Loss for 6D pose estimation (Geist et al., ICML 2024)."""
import torch
import torch.nn.functional as F


def _gram_schmidt_rotation(fwd: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt orthonormalisation to SO(3) rotation matrix. [B,3] → [B,3,3]."""
    r1 = F.normalize(fwd, dim=1)
    r2 = F.normalize(up - (up * r1).sum(dim=1, keepdim=True) * r1, dim=1)
    r3 = torch.cross(r1, r2, dim=1)
    return torch.stack([r1, r2, r3], dim=2)


def _geodesic_angle(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic angular distance on SO(3) in degrees. [B,3,3] → [B]."""
    rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_theta) * (180.0 / 3.141592653589793)


def huberised_geodesic_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 30.0) -> torch.Tensor:
    """Huber-capped geodesic error in degrees.

    Args:
        pred:  [B, 6] — 6D rotation representation (fwd3, up3) from model.
        target: [B, 6] — ground-truth 6D rotation.
        delta: Huber threshold in degrees (default 30°).

    Returns:
        Scalar loss (mean over batch).
    """
    R_pred = _gram_schmidt_rotation(pred[:, :3], pred[:, 3:])
    R_target = _gram_schmidt_rotation(target[:, :3], target[:, 3:])
    error = _geodesic_angle(R_pred, R_target)  # [B] in degrees

    # Huber: quadratic for |e| < delta, linear beyond
    mask = error < delta
    loss = torch.where(
        mask,
        0.5 * error.pow(2),
        delta * (error - 0.5 * delta),
    )
    return loss.mean()
