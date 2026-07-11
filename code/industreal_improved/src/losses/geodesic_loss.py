"""Huberised Geodesic Loss for 6D pose estimation."""
import torch


def huberised_geodesic_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 30.0) -> torch.Tensor:
    """Huber-capped geodesic error in degrees."""
    # pred/target: [B, 6] 6D rotation representation
    from src.models.mvit_mtl_model import gram_schmidt_rotation, geodesic_angle

    R_pred = gram_schmidt_rotation(pred[:, :3], pred[:, 3:])
    R_target = gram_schmidt_rotation(target[:, :3], target[:, 3:])
    error = geodesic_angle(R_pred, R_target)  # [B] in degrees
    mask = error < delta
    loss = torch.where(mask, 0.5 * error ** 2, delta * (error - 0.5 * delta))
    return loss.mean()
