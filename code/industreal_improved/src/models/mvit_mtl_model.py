"""
MViTv2-S Multi-Task Model — shared backbone for Detection + Activity + PSR + Pose.

[OPUS 201] PSR head on a diet: 70.9M → ~1.8M. Efficiency spine restored.
Total ~45M params (34.5M backbone + ~10M heads) vs ~100M specialists = ~2.2× win.

Architecture:
  - Backbone: MViTv2-S (Kinetics-400 pretrained, 34.5M)
  - Detection: P5/P4/P3 → FPN (256ch) → decoupled cls+reg head + TAL assigner
  - Activity: cls_token → 3-layer MLP (768→2048→1024→75) + optional logit-adjust
  - PSR: P5 features [B,768,T=8,7²] → spatial pool → Linear(768→256)
         → 2-layer causal Transformer (d=256, nhead=4, ff=1024)
         → Linear(256→11) per-frame transition logits. Total ≈1.8M.
  - Pose: cls_token → MLP(768→256→6) → Tanh → renormalized fwd+up
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
    """75-class activity recognition from MViTv2 class token.

    [EP10 EVIDENCE] 2-layer MLP (768→1024→75) at ep10 = 0.58% top-1 below random.
    The 1.1M-param head cannot discriminate 75 fine-grained long-tail assembly
    states from the pooled class token alone. Upgrade: 3-layer MLP with residual
    connection + balanced logit-adjustment for the long tail.

    Architecture: LayerNorm → Linear(768→2048) → GELU → Dropout →
                  Linear(2048→1024) → GELU → Dropout → Linear(1024→75)

    Also supports logit-adjustment (Menon et al. 2020): subtract per-class prior
    log-frequencies from the logits before softmax. This is a principled, margin-free
    alternative to ArcFace for long-tail classification. Activated via logit_adjust=True.
    """

    def __init__(
        self,
        feat_dim: int = 768,
        num_classes: int = 75,
        hidden1: int = 2048,
        hidden2: int = 1024,
        dropout: float = 0.2,
        logit_adjust: bool = False,
        class_freq: Optional[torch.Tensor] = None,
        logit_adjust_tau: float = 1.0,
    ):
        super().__init__()
        self.logit_adjust = logit_adjust
        self.logit_adjust_tau = logit_adjust_tau
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden2, num_classes)
        if logit_adjust and class_freq is not None:
            self.register_buffer("class_freq", class_freq.float())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Forward. Returns logit-adjusted logits if enabled.

        Args:
            cls_token: [B, 768]

        Returns:
            logits: [B, 75] (logit-adjusted if logit_adjust=True)
        """
        x = self.norm(cls_token)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        logits = self.classifier(x)
        if self.logit_adjust and hasattr(self, "class_freq") and self.training:
            # [OPUS 207 §2.6 FIX] Logit-adjust only during training.
            # At eval, predict from raw logits. This follows Menon et al. (2020)
            # protocol: adjustment inside loss, raw logits for argmax prediction.
            logits = logits + self.logit_adjust_tau * torch.log(
                self.class_freq + 1e-9
            ).unsqueeze(0)
        return logits

    def enable_logit_adjust(self, class_counts: torch.Tensor):
        """[OPUS 201] Enable logit-adjustment with dataset class frequencies.

        Args:
            class_counts: [num_classes] integer tensor of per-class sample counts.
        """
        total = class_counts.sum()
        freq = class_counts.float() / total.clamp(min=1)
        # Register buffer on the model's device so it matches logits at forward time
        device = next(self.parameters()).device
        self.register_buffer("class_freq", freq.to(device))
        self.logit_adjust = True
        self.logit_adjust_tau = 1.0


# ===========================================================================
# PSR Head (per-frame transition logits from temporal features)
# ===========================================================================

class PSRHead(nn.Module):
    """PSR head — causal Transformer on spatial-pooled temporal features.

    [OPUS 201 DIET] 70.9M → ~1.8M. The P5 feature-source fix (96-dim conv_proj
    → 768-dim semantic) was the load-bearing change — not head size. An 8-token
    sequence producing 8×11=88 outputs needs ≤15M, not 70.9M. The old 6-layer
    d=768 ff=8× head was 14-23× larger than the PSR specialist (3-5M), inverting
    the paper's efficiency claim.

    Architecture: Linear(768→256) input projection + 2-layer causal Transformer
    (d=256, nhead=4, ff=1024=4×) + Linear(256→11) output. Total ≈1.8M.

    Reads spatial-pooled features from the backbone's hook output (blocks[14] = P5).
    """

    def __init__(
        self,
        feat_dim: int = 256,      # [OPUS 201] internal transformer dim (was 768)
        input_dim: int = 768,     # [OPUS 201] P5 source feature dim
        num_components: int = 11,
        nhead: int = 4,
        num_layers: int = 2,      # [OPUS 201] 2 layers (was 6). 8 tokens needs ≤3.
    ):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # pool H,W
        # [OPUS 201] Project P5 features (768-dim) down to internal transformer dim
        self.input_proj = nn.Linear(input_dim, feat_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=feat_dim * 4,  # [OPUS 201] standard 4× (was 8×)
            dropout=0.1,
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
            conv_proj_feat: [B, 768, T=8, H=7, W=7] from P5 (blocks[14] hook).

        Returns:
            psr_logits: [B, T=8, 11] per-frame transition logits.

        [OPUS 192 FC-4] Predict at native T=8 (backbone's pooled resolution) instead
        of post-encoder linear interpolation 8→16. Linear interpolation blends
        adjacent frames, making sharp per-frame transitions unrepresentable.
        Labels are downsampled to T=8 in psr_loss() via max-pool to preserve
        transition events (any 1 in a 2-frame window → 1 in the downsampled
        label).
        """
        # Pool spatial dims → [B, 768, T=8, 1, 1] → [B, T=8, 768]
        x = self.spatial_pool(conv_proj_feat).squeeze(-1).squeeze(-1).transpose(1, 2)
        # [OPUS 201] Project from 768-dim P5 features to internal transformer dim
        x = self.input_proj(x)  # [B, T=8, feat_dim]

        # Predict at native T=8 — no interpolation. Causal mask is 8x8.
        T = x.size(1)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device),
            diagonal=1,
        )

        x = self.temporal_encoder(x, mask=mask)  # [B, 8, feat_dim]
        return self.projection(x)  # [B, 8, 11]

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

        # [OPUS 186 B-6] PSR head on `blocks[14]` features (768ch, post-all-attention),
        # NOT on `conv_proj` (96ch, layer 0, no semantics). The pre-fix code read
        # `fpn_feats.get("P2")` which is conv_proj output — raw patch embeddings
        # with no object/state semantics. The PSR causal-transformer was
        # learning from semantics-free features, which is why PSR loss was flat
        # at base-rate entropy. blocks[14] features carry semantic information
        # after 14 transformer blocks; PSR can finally learn transition events.
        # [OPUS 201] PSRHead now defaults to internal d=256 with input_proj(768→256).
        # Only pass input_dim (P5 source); feat_dim defaults to 256 internally.
        self.psr_head = PSRHead(input_dim=backbone_dim, num_components=num_psr_components)

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
        # [OPUS 192 FC-2 / Layer 5] Use only P3/P4/P5 for detection. P2 reads
        # raw `conv_proj` patch-embeddings (semantics-free, same issue that
        # starved PSR before B-3). Drop P2 from detection — classification on
        # semantic levels (P3=192ch, P4=384ch, P5=768ch) is the load-bearing
        # change. P2 remains in the FPN top-down pathway (so PSR's P5 still
        # gets its top-down context) but is not used for detection heads.
        fpn_out = self.fpn(fpn_feats)
        det_outputs = {}
        for level_name, feat in fpn_out.items():
            if level_name == "P2":
                # Skip P2 (raw conv_proj features, no semantics — FC-2)
                continue
            # Temporal-pool T dimension for 2D detection
            pooled = feat.mean(dim=2)  # [B, 256, H, W]
            det_outputs[level_name] = self.det_head(pooled)

        # Activity
        act_logits = self.act_head(cls_token)

        # PSR (uses blocks[14] features at P5 resolution — semantic-rich post-attention)
        # [OPUS 186 B-6] Was fpn_feats.get("P2") (conv_proj, 96ch, layer 0).
        # Now uses fpn_feats.get("P5") (blocks[14], 768ch, post-all-attention).
        psr_input = fpn_feats.get("P5")  # [B, 768, T=8, 7, 7]
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
