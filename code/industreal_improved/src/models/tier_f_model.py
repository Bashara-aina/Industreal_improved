"""
Tier F -- Shared Hiera-B backbone with 4 heads: Detection, Activity, PSR, Pose.

Architecture (175 ULTIMATE GUIDE §3.1-3.2):
  Backbone: Hiera-B (~50.8M) via timm, features_only=True
  Two forward modes (weight-shared):
    - temporal: [B, T=16, 3, 224, 224] -> stage-4 pooled -> activity/PSR/pose
    - detection: [B, 3, H, W] -> multiscale P3/P4/P5 -> FPN -> detection
  4 heads:
    - Detection: FPN + decoupled cls/box (YOLOv8-style, anchor-free, DFL)
    - Activity: LayerNorm -> Linear(768, 75)  (~0.06M)
    - PSR: causal TransformerEncoder(3 layers) + 11 per-component MLPs (~2-3M)
    - Pose: MLP(768->256)->LeakyReLU->Linear(256, 6)  (~0.3M)
  Total: ~60M

Notes:
  - timm's Hiera is 2D-only; temporal mode processes each frame independently
    through the shared 2D backbone, then aggregates across time.
  - Pretrained weights may not be downloadable in all environments; the code
    degrades gracefully to random init with a clear warning.
  - The PSR per-component head follows the LeakyReLU(0.01) repair pattern
    from model.py:1604-1611 (GELU saturation fix).

References:
  - 175_ULTIMATE_GUIDE_TIER_F.md §3.1-3.2
  - model.py:1604-1611 (LeakyReLU PSR pattern)
"""

import logging
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ===========================================================================
# Backbone builder
# ===========================================================================


def build_backbone(pretrained: bool = True) -> nn.Module:
    """Build Hiera-B backbone via timm with ``features_only=True``.

    Returns a model whose forward returns a list of feature maps
    ``[stage0, stage1, stage2, stage3]`` with channels ``[96, 192, 384, 768]``
    and strides ``[4, 8, 16, 32]``.
    """
    import timm

    if pretrained:
        # Try best-available checkpoint first
        for tag in ("hiera_base_224.mae_in1k_ft_in1k", "hiera_base_224.mae"):
            try:
                backbone = timm.create_model(
                    tag, pretrained=True, features_only=True
                )
                logger.info("Loaded Hiera-B with %s pretrained weights", tag)
                return backbone
            except Exception:
                continue
        warnings.warn(
            "Could not load Hiera-B pretrained weights (may need HF login). "
            "Falling back to random initialization. "
            "The model architecture is correct; only the weights are uninitialized."
        )

    backbone = timm.create_model(
        "hiera_base_224", pretrained=False, features_only=True
    )
    logger.info(
        "Loaded Hiera-B architecture (random init, %d params)",
        sum(p.numel() for p in backbone.parameters()),
    )
    return backbone


# ===========================================================================
# FPN Neck
# ===========================================================================


class FPN(nn.Module):
    """Simple Feature Pyramid Network.

    Lateral 1x1 projections + top-down nearest-neighbour upsampling + 3x3
    smooth convolutions.  Operates on 3 input levels.

    Args:
        in_channels: List of 3 input channel counts [c3, c4, c5].
        fpn_dim: Output channel count for all FPN levels (default 256).
    """

    def __init__(self, in_channels: List[int], fpn_dim: int = 256):
        super().__init__()
        assert len(in_channels) == 3
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, fpn_dim, 1) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in range(3)]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """FPN forward.

        Args:
            features: [P3, P4, P5] at strides [8, 16, 32].

        Returns:
            [out3, out4, out5] all at ``fpn_dim`` channels.
        """
        laterals = [l(f) for l, f in zip(self.lateral, features)]

        out5 = self.smooth[2](laterals[2])
        out4 = laterals[1] + F.interpolate(
            out5, size=laterals[1].shape[-2:], mode="nearest"
        )
        out4 = self.smooth[1](out4)
        out3 = laterals[0] + F.interpolate(
            out4, size=laterals[0].shape[-2:], mode="nearest"
        )
        out3 = self.smooth[0](out3)
        return [out3, out4, out5]


# ===========================================================================
# Detection Head (YOLOv8-style decoupled)
# ===========================================================================


class DetectionHead(nn.Module):
    """Decoupled classification + DFL box regression head.

    Applies two independent conv branches per FPN level:
      - Cls: 2x Conv3×3-BN-SiLU -> Conv1×1(num_classes)
      - Box: 2x Conv3×3-BN-SiLU -> Conv1×1(4 * reg_max)

    This is a simplified single-level head applied identically across
    all FPN outputs.  A production-grade head would share weights across
    levels and include an implicit objectness branch; this version is
    sufficient for the ~60M total-param budget.
    """

    def __init__(
        self, num_classes: int = 24, fpn_dim: int = 256, reg_max: int = 16
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        def _make_branch(out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                nn.BatchNorm2d(fpn_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                nn.BatchNorm2d(fpn_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(fpn_dim, out_channels, 1),
            )

        self.cls_branch = _make_branch(num_classes)
        self.box_branch = _make_branch(4 * reg_max)

    def forward(
        self, features: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward on list of FPN outputs.

        Returns:
            det_cls_logits: list of [B, num_classes, H_i, W_i] per level.
            det_box_logits: list of [B, 4*reg_max, H_i, W_i] per level.
        """
        cls_outs = [self.cls_branch(f) for f in features]
        box_outs = [self.box_branch(f) for f in features]
        return {"det_cls_logits": cls_outs, "det_box_logits": box_outs}


# ===========================================================================
# Activity Head
# ===========================================================================


class ActivityHead(nn.Module):
    """Clip-level activity classifier.

    LayerNorm -> Linear(768, num_classes).

    Spec: ~0.06M params for 75 classes.
    """

    def __init__(self, feat_dim: int = 768, num_classes: int = 75):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, feat_dim] -> [B, num_classes] logits."""
        return self.fc(self.norm(x))


# ===========================================================================
# PSR Head
# ===========================================================================


class PSRHead(nn.Module):
    """Procedure-step-recognition head.

    Causal TransformerEncoder (3 layers) + 11 per-component transition
    logit heads (LeakyReLU pattern from model.py:1604-1611).

    Architecture:
      input (B, T, 768) -> Linear(768, 256) -> causal Transformer x3 ->
        per-component MLP(256->64->1) per channel -> (B, T, 11)
    """

    def __init__(
        self,
        feat_dim: int = 768,
        num_components: int = 11,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=nn.LeakyReLU(negative_slope=0.01),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._causal_mask: Optional[torch.Tensor] = None

        # Per-component output heads (model.py:1604-1611 pattern)
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(dropout * 0.3),
                    nn.Linear(64, 1),
                )
                for _ in range(num_components)
            ]
        )

        # Weight initialisation (model.py:1618-1624)
        for head in self.output_heads:
            for layer in (head[0], head[3]):
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)

    def _make_causal_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device), diagonal=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, feat_dim] -> [B, T, num_components] transition logits."""
        B, T, _ = x.shape
        x = self.input_proj(x)

        if (
            self._causal_mask is None
            or self._causal_mask.shape[0] != T
            or self._causal_mask.device != x.device
        ):
            self._causal_mask = self._make_causal_mask(T, x.device)

        x = self.encoder(x, mask=self._causal_mask)

        # Stack per-component heads: [B, T, 11]
        out = torch.stack([h(x) for h in self.output_heads], dim=-1)
        out = out.squeeze(-2)  # remove the singleton dim from Linear output
        return out  # (B, T, 11)


# ===========================================================================
# Pose Head
# ===========================================================================


class PoseHead(nn.Module):
    """Head-pose regressor -- 6D continuous (fwd3 + up3).

    MLP(768 -> 256) -> LeakyReLU -> Linear(256, 6).

    The output is raw 6D vectors; re-normalisation (ensuring forward and up
    are unit vectors) is done in the loss function, not here.
    """

    def __init__(
        self, feat_dim: int = 768, pose_dim: int = 6, hidden_dim: int = 256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, feat_dim] -> [B, pose_dim]."""
        return self.net(x)


# ===========================================================================
# TierFModel -- unified multi-task model
# ===========================================================================


class TierFModel(nn.Module):
    """Tier F multi-task model with Hiera-B backbone and 4 task heads.

    Two forward modes share the same backbone weights:
      - ``mode="temporal"``: 16-frame clip -> activity/PSR/pose heads
      - ``mode="detection"``: single frame -> FPN -> detection head

    .. note::
       **Resolution limitation.** timm's Hiera-B (2D variant) has fixed position
       embeddings and internal unroll/reroll state sized for 224x224 inputs.
       For now detection mode also uses 224x224 (same as temporal).  Production
       code targeting 448-640 detection resolution must adapt the backbone's
       ``pos_embed``, ``unroll.size``, ``reroll`` schedule, and
       ``tokens_spatial_shape`` dynamically per forward call.

    Args:
        num_classes_det: Number of detection classes (default 24).
        num_classes_act: Number of activity classes (default 75).
        num_components_psr: Number of PSR procedure-step components (default 11).
        pose_dim: Head-pose output dimension (default 6).
        pretrained: Attempt to load pretrained Hiera-B weights if ``True``.
    """

    def __init__(
        self,
        num_classes_det: int = 24,
        num_classes_act: int = 75,
        num_components_psr: int = 11,
        pose_dim: int = 6,
        pretrained: bool = True,
    ):
        super().__init__()

        # ---- Backbone ----
        self.backbone = build_backbone(pretrained=pretrained)
        # Hiera-B stages: ch=[96, 192, 384, 768], strides=[4, 8, 16, 32]
        feat_ch = self.backbone.feature_info.channels()
        self.stage4_dim = feat_ch[-1]  # 768
        self._detection_input_size_hint = 224  # native resolution for this Hiera variant

        # ---- Detection head (uses stages 1,2,3 = strides 8,16,32) ----
        self.fpn = FPN(
            in_channels=[feat_ch[1], feat_ch[2], feat_ch[3]]
        )
        self.detection_head = DetectionHead(num_classes=num_classes_det)

        # ---- Temporal heads ----
        self.activity_head = ActivityHead(
            feat_dim=self.stage4_dim, num_classes=num_classes_act
        )
        self.psr_head = PSRHead(
            feat_dim=self.stage4_dim, num_components=num_components_psr
        )
        self.pose_head = PoseHead(
            feat_dim=self.stage4_dim, pose_dim=pose_dim
        )

    def forward(
        self, clip: torch.Tensor, mode: str = "temporal"
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            clip:
                - ``mode="temporal"``: ``[B, T, 3, 224, 224]`` with ``T=16``.
                - ``mode="detection"``: ``[B, 3, H, W]`` (single high-res frame).
            mode: ``"temporal"`` or ``"detection"``.

        Returns:
            Temporal mode dict with keys ``act_logits``, ``psr_logits``,
            ``pose_6d``.
            Detection mode dict with keys ``det_cls_logits``,
            ``det_box_logits`` (each a list of 3 tensors, one per FPN
            level).
        """
        if mode == "temporal":
            return self._forward_temporal(clip)
        elif mode == "detection":
            return self._forward_detection(clip)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'temporal' or 'detection'."
            )

    def _forward_temporal(
        self, clip: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Temporal forward: ``[B, T, 3, 224, 224]``.

        Each frame is processed independently through the 2D Hiera backbone,
        then results are concatenated along the time axis.
        """
        B, T, C, H, W = clip.shape
        # Collapse batch-time dimension for 2D backbone
        frames = clip.view(B * T, C, H, W)
        features = self.backbone(frames)
        stage4 = features[-1]  # (B*T, 768, H/32, W/32)

        # Global average pool over spatial dimensions
        pooled = stage4.mean(dim=[-1, -2])  # (B*T, 768)
        clip_embed = pooled.view(B, T, -1)  # (B, T, 768)

        # Temporal mean pool for activity and pose
        temporal_pooled = clip_embed.mean(dim=1)  # (B, 768)

        return {
            "act_logits": self.activity_head(temporal_pooled),
            "psr_logits": self.psr_head(clip_embed),
            "pose_6d": self.pose_head(temporal_pooled),
        }

    def _forward_detection(
        self, clip: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Detection forward: ``[B, 3, H, W]`` single frame.

        If the spatial resolution differs from the backbone's native
        224x224, the input is resized (bilinear) to 224x224 first.
        This is a practical compromise -- see class docstring for the
        multi-resolution limitation.
        """
        if clip.shape[-1] != self._detection_input_size_hint or clip.shape[-2] != self._detection_input_size_hint:
            clip = F.interpolate(
                clip,
                size=(self._detection_input_size_hint, self._detection_input_size_hint),
                mode="bilinear",
                align_corners=False,
            )

        features = self.backbone(clip)
        # P3=stage1(stride8,192ch), P4=stage2(stride16,384ch),
        # P5=stage3(stride32,768ch)
        det_feats = [features[1], features[2], features[3]]

        fpn_outs = self.fpn(det_feats)
        return self.detection_head(fpn_outs)

    def get_param_counts(self) -> Dict[str, int]:
        """Return parameter counts per component for reporting."""
        counts: Dict[str, int] = {}
        for name, param in self.named_parameters():
            parts = name.split(".")
            component = parts[0] if parts else "other"
            counts[component] = counts.get(component, 0) + param.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
