"""Video backbones for activity recognition (Kinetics-400 pretrained)."""

import torch
import torch.nn as nn

try:
    from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

    HAS_TORCHVISION_MVIT = True
except ImportError:
    HAS_TORCHVISION_MVIT = False


def load_mvit_v2_s(pretrained: bool = True) -> nn.Module:
    """Load MViTv2-S pretrained on Kinetics-400 from torchvision."""
    if not HAS_TORCHVISION_MVIT:
        raise ImportError(
            "torchvision >= 0.15 with MViTv2-S required. Install: pip install torchvision --upgrade"
        )
    weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
    model = mvit_v2_s(weights=weights)
    return model


class VideoFeatureExtractor(nn.Module):
    """Wraps a video backbone for feature extraction.

    Input:  (B, T, C, H, W)  — batched video clips
    Output: (B, feat_dim)    — clip-level features

    Backbone is frozen (eval mode, no gradients).
    """

    def __init__(self, backbone: str = "mvit_v2_s", pretrained: bool = True):
        super().__init__()
        if backbone == "mvit_v2_s":
            self.backbone = load_mvit_v2_s(pretrained=pretrained)
            self.feat_dim = 768  # MViTv2-S final embedding
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove classification head
        self.backbone.head = nn.Identity()

        # Freeze and eval
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract clip-level features.

        Args:
            x: [B, T, C, H, W] — uint8 or float32 frames, normalized externally.

        Returns:
            features: [B, feat_dim] — 768-dim clip embedding.
        """
        # MViTv2 expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            features = self.backbone(x)
        return features  # [B, 768]
