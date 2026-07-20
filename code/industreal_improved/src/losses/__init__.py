"""Loss modules for MTL training."""

from .at_matcher import atss_match_anchors_to_gt
from .ciou import ciou_loss, decode_deltas_to_xyxy
from .qfl import quality_focal_loss_with_logits
from .supcon import SupConLoss, get_projection_head
from .uw_so import UWSOLoss, uw_so_loss

__all__ = [
    "SupConLoss",
    "UWSOLoss",
    "atss_match_anchors_to_gt",
    "ciou_loss",
    "decode_deltas_to_xyxy",
    "get_projection_head",
    "quality_focal_loss_with_logits",
    "uw_so_loss",
]
