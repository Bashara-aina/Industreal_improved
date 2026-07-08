"""
MViTv2-S Multi-Task Model — shared backbone for Detection + Activity + PSR + Pose.

Architecture:
  - Backbone: MViTv2-S (Kinetics-400 pretrained, 34.5M)
  - Detection: forward hooks at stages {conv_proj, blocks[1], blocks[3], blocks[14]}
               → FPN neck (lateral 1×1 + top-down 2× upsample + 3×3 conv, 256ch)
               → decoupled cls+box head (24 assembly-state classes)
  - Activity: class token → LayerNorm(768) → Linear(768, 75) → 75-class CE
  - PSR: conv_proj spatial-pooled [B, 96, T=8] → interpolate to T=16
         → causal TransformerEncoder (3 layers) → Linear(96, 11) → per-frame transition logits
  - Pose: class token → MLP(768→256→6) → Tanh → renormalized fwd+up

Total ~40M params (34.5M backbone + ~5.5M heads).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

logger = logging.getLogger("mvit_mtl_model")

NUM_ACT_CLASSES = 75
NUM_DET_CLASSES = 24
NUM_PSR_COMPONENTS = 11
POSE_DIM = 6
FPN_CHANNELS = 256


# ===========================================================================
# Helper: extract intermediate features from MViTv2-S for detection FPN
# ===========================================================================

class MViTFeaturePyramid(nn.Module):
    """MViTv2-S backbone with forward hooks for detection FPN features.

    Returns dict {P2, P3, P4, P5} with temporal-dimension collapsed.
    """

    def __init__(self):
        super().__init__()
        self.backbone = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        feat_dim = self.backbone.head[1].in_features  # 768
        self.backbone.head = nn.Identity()
        self.backbone.norm = nn.Identity()

        # Register hooks at stage boundaries
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def _make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    x, thw = output
                    # x: [B, N, C] sequence format, N = T*H*W + 1 (class token)
                    B, N, C = x.shape
                    T, H, W = thw
                    spatial = x[:, 1:, :]  # remove class token
                    spatial = spatial.reshape(B, T, H, W, C)
                    spatial = spatial.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]
                    self._features[name] = spatial
                else:
                    # conv_proj output: [B, C, T, H, W] directly
                    self._features[name] = output
            return hook

        # Hook conv_proj → P2 (96ch, 56×56)
        self._hooks.append(
            self.backbone.conv_proj.register_forward_hook(_make_hook("P2"))
        )
        # Hook blocks where spatial resolution halves
        # block[1] → 28×28 (192ch), block[3] → 14×14 (384ch), block[14] → 7×7 (768ch)
        for idx, name in [(1, "P3"), (3, "P4"), (14, "P5")]:
            self._hooks.append(
                self.backbone.blocks[idx].register_forward_hook(_make_hook(name))
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass, returns FPN feature dict.

        Args:
            x: [B, 3, T=16, 224, 224] normalized input.

        Returns:
            features: dict of {P2, P3, P4, P5} each [B, C, T=8, H, W]
            clip_feat: [B, 768] pooled clip feature (class token)
        """
        self._features.clear()

        # Full forward through backbone (triggers hooks)
        # We need the class token for activity/pose heads
        # Run forward without head (block 0..15)
        x = self.backbone.conv_proj(x)  # [B, 96, T=8, H=56, W=56]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.backbone.pos_encoding(x)
        thw = (
            self.backbone.pos_encoding.temporal_size,
            *self.backbone.pos_encoding.spatial_size,
        )  # (T=8, H=56, W=56)

        for i, block in enumerate(self.backbone.blocks):
            x, thw = block(x, thw)

        # Class token for activity/pose: [B, 768]
        cls_token = x[:, 0, :]

        # Temporal-pool detection features (mean over T)
        fpn_features = {}
        for name in ["P2", "P3", "P4", "P5"]:
            feat = self._features.get(name)
            if feat is not None:
                fpn_features[name] = feat  # [B, C, T=8, H, W]

        return fpn_features, cls_token


# ===========================================================================
# Lightweight FPN for detection
# ===========================================================================

class LightweightFPN(nn.Module):
    """Minimal FPN: lateral 1×1 convs + top-down 2× upsample.

    Input: dict {P2: 96ch, P3: 192ch, P4: 384ch, P5: 768ch} each [B, C, T, H, W]
    Output: dict {P2, P3, P4, P5} each [B, 256, T, H, W] with H,W halving per level.
    """

    def __init__(self, in_channels: Dict[str, int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.lateral = nn.ModuleDict({
            name: nn.Conv3d(ch, out_channels, kernel_size=1)
            for name, ch in in_channels.items()
        })
        self.smooth = nn.ModuleDict({
            name: nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            for name in in_channels
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build FPN pyramid via top-down pathway."""
        names = ["P5", "P4", "P3", "P2"]  # top-down order

        lat = {n: self.lateral[n](features[n]) for n in names}
        out = {}
        for i, name in enumerate(names):
            if i == 0:
                out[name] = self.smooth[name](lat[name])
            else:
                # Upsample previous top level 2×
                top = F.interpolate(
                    out[names[i - 1]],
                    size=lat[name].shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                out[name] = self.smooth[name](lat[name] + top)

        return out


# ===========================================================================
# Detection Head (lightweight YOLO-style decoupled head)
# ===========================================================================

class DetectionHead(nn.Module):
    """Lightweight detection head — decoupled cls + box regression with DFL.

    Operates on FPN features [B, 256, T, H, W] (temporal-pooled to [B, 256, H, W]).
    Box regression outputs a distribution over reg_max bins per coordinate for DFL.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 24, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * reg_max, 1),  # DFL: 4 coords x reg_max bins
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward per FPN level.

        Args:
            x: [B, 256, H, W] spatial feature map (temporal-pooled).

        Returns:
            cls_logits: [B, num_classes, H, W]
            reg_preds: [B, 4 * reg_max, H, W]
        """
        return {
            "cls_logits": self.cls_head(x),
            "reg_preds": self.reg_head(x),
        }


# ===========================================================================
# Activity Head
# ===========================================================================

class ActivityHead(nn.Module):
    """75-class activity recognition from MViTv2 class token."""

    def __init__(self, feat_dim: int = 768, num_classes: int = 75):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            cls_token: [B, 768]

        Returns:
            logits: [B, 75]
        """
        return self.classifier(self.norm(cls_token))


# ===========================================================================
# PSR Head (per-frame transition logits from temporal features)
# ===========================================================================

class PSRHead(nn.Module):
    """PSR head — causal Transformer on spatial-pooled temporal features.

    Extracts per-frame transition logits from conv_proj features [B, 96, T=8, H=56, W=56].
    """

    def __init__(
        self,
        feat_dim: int = 96,
        num_components: int = 11,
        nhead: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        # Spatial pooling → [B, 96, T=8] → interpolate to T=16
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # pool H,W

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=feat_dim * 4,
            activation=partial(F.leaky_relu, negative_slope=0.01),
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(feat_dim, num_components)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, conv_proj_feat: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            conv_proj_feat: [B, 96, T=8, H=56, W=56] from conv_proj forward hook.

        Returns:
            psr_logits: [B, T=16, 11] per-frame transition logits.
        """
        # Pool spatial dims → [B, 96, T=8, 1, 1] → [B, T=8, 96]
        x = self.spatial_pool(conv_proj_feat).squeeze(-1).squeeze(-1).transpose(1, 2)

        # Interpolate T=8 to T=16 for per-frame predictions
        x = x.permute(0, 2, 1)  # [B, 96, 8]
        x = F.interpolate(x, size=16, mode="linear", align_corners=False)  # [B, 96, 16]
        x = x.permute(0, 2, 1)  # [B, 16, 96]

        # Causal masking: each frame can only attend to past+present
        mask = torch.triu(
            torch.full((16, 16), float("-inf"), device=x.device),
            diagonal=1,
        )

        x = self.temporal_encoder(x, mask=mask)  # [B, 16, 96]
        return self.projection(x)  # [B, 16, 11]


# ===========================================================================
# Pose Head
# ===========================================================================

class PoseHead(nn.Module):
    """6D head pose MLP from MViTv2 class token."""

    def __init__(self, feat_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, POSE_DIM),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            cls_token: [B, 768]

        Returns:
            raw_6d: [B, 6] Tanh-bounded (fwd 3 + up 3)
        """
        return self.mlp(cls_token)


# ===========================================================================
# MTL-MViT: Full multi-task model
# ===========================================================================

class MTLMViTModel(nn.Module):
    """MViTv2-S shared backbone + 4 heads: Detection, Activity, PSR, Pose.

    Total ~40M params.
    """

    def __init__(
        self,
        num_act_classes: int = NUM_ACT_CLASSES,
        num_det_classes: int = NUM_DET_CLASSES,
        num_psr_components: int = NUM_PSR_COMPONENTS,
        fpn_channels: int = FPN_CHANNELS,
    ):
        super().__init__()
        self.num_act_classes = num_act_classes
        self.num_det_classes = num_det_classes
        self.num_psr_components = num_psr_components

        # Shared MViTv2-S backbone + feature pyramid
        self.feature_pyramid = MViTFeaturePyramid()
        backbone_dim = 768  # MViTv2-S final channel dim

        # Detection FPN + head
        self.fpn = LightweightFPN(
            in_channels={"P2": 96, "P3": 192, "P4": 384, "P5": 768},
            out_channels=fpn_channels,
        )
        self.det_head = DetectionHead(in_channels=fpn_channels, num_classes=num_det_classes)

        # Activity head
        self.act_head = ActivityHead(feat_dim=backbone_dim, num_classes=num_act_classes)

        # PSR head (uses conv_proj features from feature_pyramid hooks)
        self.psr_head = PSRHead(feat_dim=96, num_components=num_psr_components)

        # Pose head
        self.pose_head = PoseHead(feat_dim=backbone_dim)

        logger.info(
            "MTLMViTModel: feats={}, act={}-cls, det={}-cls, psr={}-comp, fpn={}ch".format(
                backbone_dim, num_act_classes, num_det_classes, num_psr_components, fpn_channels
            )
        )

    def forward(self, clip: torch.Tensor) -> Dict[str, Any]:
        """Forward pass.

        Args:
            clip: [B, 3, T=16, 224, 224] normalized video clip.

        Returns:
            dict with keys:
              - detection: dict of per-FPN-level {cls_logits, reg_preds}
              - activity: [B, 75] logits
              - psr_logits: [B, T=16, 11] per-frame transition logits
              - pose_6d: [B, 6] Tanh-bounded fwd+up
        """
        # Shared backbone forward
        fpn_feats, cls_token = self.feature_pyramid(clip)

        # Detection: FPN → decoupled head
        fpn_out = self.fpn(fpn_feats)
        det_outputs = {}
        for level_name, feat in fpn_out.items():
            # Temporal-pool T dimension for 2D detection
            pooled = feat.mean(dim=2)  # [B, 256, H, W]
            det_outputs[level_name] = self.det_head(pooled)

        # Activity
        act_logits = self.act_head(cls_token)

        # PSR (uses conv_proj features at P2 resolution)
        psr_input = fpn_feats.get("P2")  # [B, 96, T=8, 56, 56]
        if psr_input is not None:
            psr_logits = self.psr_head(psr_input)
        else:
            psr_logits = torch.zeros(
                clip.size(0), 16, self.num_psr_components, device=clip.device
            )

        # Pose
        pose_6d = self.pose_head(cls_token)

        return {
            "detection": det_outputs,
            "activity": act_logits,
            "psr_logits": psr_logits,
            "pose_6d": pose_6d,
            "cls_token": cls_token,
        }

    def load_pretrained_backbone(self):
        """Reload backbone weights (already loaded at init)."""
        logger.info("Backbone: MViTv2-S with Kinetics-400 pretrained weights.")
        return self


def renormalize_pose(raw_6d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Renormalize fwd and up vectors from raw 6D."""
    fwd = F.normalize(raw_6d[:, :3], dim=1)
    up = F.normalize(raw_6d[:, 3:], dim=1)
    return fwd, up
