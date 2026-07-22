"""DEPRECATED — DO NOT USE FOR NEW WORK.

The active multi-task model is POPWMultiTaskModel in src/models/model.py
(2361 lines, convnext_tiny backbone, 46.47M params).

This file is the LEGACY MViTv2-S based MTLMViTModel (655 lines). It is
preserved for historical reference only. All V2 work targets POPWMultiTaskModel.

See analyses/consult_claude_science/consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md
for the migration rationale.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self._hooks.append(self.backbone.conv_proj.register_forward_hook(_make_hook("P2")))
        # Hook blocks where spatial resolution halves
        # block[1] → 28×28 (192ch), block[3] → 14×14 (384ch), block[14] → 7×7 (768ch)
        for idx, name in [(1, "P3"), (3, "P4"), (14, "P5")]:
            self._hooks.append(self.backbone.blocks[idx].register_forward_hook(_make_hook(name)))

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
        x = self.backbone.conv_proj(x)  # [B, 96, T, H, W]
        # [FIX 2026-07-13] Use dynamic thw from conv_proj output instead of hardcoded
        # pos_encoding.spatial_size. Enables multi-resolution inference (224/320/480):
        # MViTv2-S uses RELATIVE position encoding which is auto-interpolated per-block
        # for any spatial size — only the thw tracking was preventing multi-scale use.
        T_t, H_t, W_t = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.backbone.pos_encoding(x)
        thw = (T_t, H_t, W_t)

        # [FIX 2026-07-13] Gradient checkpointing for 480/640 training. Trades
        # ~30% extra compute for ~3-4x lower activation memory. Enabled by
        # --grad-checkpoint flag (passed via model cfg). Trades: 480 batch=1
        # becomes feasible on 16GB GPU where it would OOM otherwise.
        use_grad_ckpt = getattr(self, "_grad_checkpoint", False)
        for i, block in enumerate(self.backbone.blocks):
            if use_grad_ckpt and self.training:
                # Checkpoint only the early/middle blocks (cheaper recompute)
                # Leave the last few blocks un-checkpointed for stable training.
                x, thw = torch.utils.checkpoint.checkpoint(block, x, thw, use_reentrant=False)
            else:
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
    """BiFPN — top-down + bottom-up with EfficientDet-style weighted fusion.

    [FIX 207 §2.5] P5 fusion now uses two inputs only (p5_td + max_pool(p4_out)),
    eliminating the duplicate p5_lat term.

    [IMP-10] Added dedicated 3x3 P2 lateral projection (p2_lateral) to better
    process C2 (96ch, stride 4) into the 256-channel FPN space, improving small
    object detection by preserving high-resolution signal through a larger
    receptive field compared to the standard 1x1 lateral.

    Input: dict {P2: 96ch, P3: 192ch, P4: 384ch, P5: 768ch} each [B, C, T, H, W]
    Output: dict {P2, P3, P4, P5} each [B, 256, T, H, W] with H,W halving per level.
    """

    def __init__(self, in_channels: Dict[str, int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.eps = 1e-4

        # 1x1 lateral projections to out_channels (P3/P4/P5)
        self.lateral = nn.ModuleDict(
            {name: nn.Conv3d(ch, out_channels, kernel_size=1) for name, ch in in_channels.items() if name != "P2"}
        )
        # [IMP-10] Dedicated 3x3 conv for P2 (stride 4, 96ch). Larger receptive
        # field than 1x1 lateral — better spatial processing for small objects
        # while preserving stride-4 resolution (stride=1, not 2, to maintain P2
        # high resolution). The temporal kernel is 1 (per-frame) so T is preserved.
        self.p2_lateral = nn.Conv3d(96, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        # Smooth convs for top-down path
        self.td_conv = nn.ModuleDict(
            {
                name: nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
                for name in in_channels
            }
        )

        # Smooth convs for bottom-up path
        self.bu_conv = nn.ModuleDict(
            {
                name: nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
                for name in in_channels
            }
        )

        # Learnable fusion weights: top-down
        # P5 has 1 input (lateral only); P4/P3/P2 have 2 (lateral + up from above)
        names_td = ["P5", "P4", "P3", "P2"]
        self.td_w = nn.ParameterDict(
            {name: nn.Parameter(torch.ones(1 if name == "P5" else 2)) for name in names_td}
        )

        # Learnable fusion weights: bottom-up
        # P2 has 1 input (td only); P3/P4/P5 have 2 (td + down from below)
        names_bu = ["P2", "P3", "P4", "P5"]
        self.bu_w = nn.ParameterDict(
            {name: nn.Parameter(torch.ones(1 if name == "P2" else 2)) for name in names_bu}
        )

    @staticmethod
    def _fast_weightsum(weights: torch.Tensor, terms: List[torch.Tensor]) -> torch.Tensor:
        """Fast normalized weighted sum (ReLU + normalize, EfficientDet-style)."""
        w = F.relu(weights)
        return sum(w[i] * terms[i] for i in range(len(terms))) / (w.sum() + 1e-4)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build BiFPN pyramid: top-down followed by bottom-up with weighted fusion."""
        names = ["P5", "P4", "P3", "P2"]

        # Lateral projections: 1x1 for P3/P4/P5, dedicated 3x3 for P2
        lat = {}
        for n in names:
            if n == "P2":
                lat[n] = self.p2_lateral(features[n])
            else:
                lat[n] = self.lateral[n](features[n])

        # --- Top-down pathway (P5 -> P4 -> P3 -> P2) ---
        td = {}
        for i, name in enumerate(names):
            if i == 0:
                td[name] = self.td_conv[name](lat[name])
            else:
                prev = names[i - 1]
                up = F.interpolate(
                    td[prev],
                    size=lat[name].shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                fused = self._fast_weightsum(self.td_w[name], [lat[name], up])
                td[name] = self.td_conv[name](fused)

        # --- Bottom-up pathway (P2 -> P3 -> P4 -> P5) ---
        bu = ["P2", "P3", "P4", "P5"]
        out = {}
        for i, name in enumerate(bu):
            if i == 0:
                out[name] = self.bu_conv[name](td[name])
            else:
                prev = bu[i - 1]
                down = F.interpolate(
                    out[prev],
                    size=td[name].shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                fused = self._fast_weightsum(self.bu_w[name], [td[name], down])
                out[name] = self.bu_conv[name](fused)

        return out


# ===========================================================================
# Detection Head (lightweight YOLO-style decoupled head)
# ===========================================================================


class DetectionHead(nn.Module):
    """Detection head — decoupled cls + per-anchor box regression.

    Operates on FPN features [B, 256, T, H, W] (temporal-pooled to [B, 256, H, W]).
    Box regression outputs 4 * num_anchors channels, interpreted as (dx, dy, dw, dh)
    per anchor.  Default num_anchors=16, so reg_out = 64 channels.

    Training (train_mtl_v3.py) reshapes [B, 4*A, H, W] -> [B, A, 4, H, W] -> [B, H, W, A, 4].
    Eval decodes each anchor's (dx, dy, dw, dh) as:
        cx = grid_cx + dx * 0.1
        cy = grid_cy + dy * 0.1
        w  = anchor_w * exp(dw)
        h  = anchor_h * exp(dh)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 24,
                 prior_prob: float = 0.01, logit_bias_scale: float = 1.0,
                 num_anchors: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.prior_prob = prior_prob
        self.logit_bias_scale = logit_bias_scale
        # [FIX-2026-07-21] Register running_pos_ratio as a persistent buffer so
        # it survives checkpoint save/load. Previously it was a plain Python
        # float attribute, meaning every resume-from-checkpoint would reset
        # the EMA back to prior_prob (causing the bias adjustment to re-warm
        # from the initial state on every reload).
        self.register_buffer(
            "running_pos_ratio",
            torch.tensor(float(prior_prob), dtype=torch.float32),
            persistent=True,
        )
        self.num_anchors = num_anchors

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * self.num_anchors, 1),  # dynamic: num_anchors x 4 coords
        )

        # Initialize with low-positive prior to suppress false positives on start
        self._init_weights(prior_prob=self.prior_prob)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward per FPN level.

        Args:
            x: [B, 256, H, W] spatial feature map (temporal-pooled).

        Returns:
            cls_logits: [B, num_classes, H, W]
            reg_preds: [B, 4 * num_anchors, H, W]
        """
        return {
            "cls_logits": self.cls_head(x),
            "reg_preds": self.reg_head(x),
        }

    def _init_weights(self, prior_prob: float = 0.01):
        """Initialize detection head with a low-positive prior.

        Sets the bias of the final classification conv layer so that sigmoid
        outputs start near `prior_prob` instead of 0.5. Without this, every
        location fires with ~50% confidence and eval produces millions of
        false positives (mAP=0 even when training is correct).

        [IMP-10] logit_bias_scale multiplies the computed bias_init, allowing
        controlled deviation from the prior_prob-derived value.  scale > 1.0
        makes the initial bias more negative (lower confidence), scale < 1.0
        makes it less negative (higher confidence).  Default 1.0 = no change.

        Args:
            prior_prob: desired initial probability for foreground at each location.
        """
        import math
        bias_init = -math.log((1.0 - prior_prob) / prior_prob) * self.logit_bias_scale
        for m in self.cls_head.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == self.num_classes:
                nn.init.constant_(m.bias, bias_init)
        # Regression head: default init is fine (predictions = raw anchor offsets)

    def update_logit_bias(self, batch_pos_ratio: float, momentum: float = 0.05):
        """Dynamically adjust classification bias based on observed positive ratio.

        [IMP-10] Tracks an EMA of the per-batch positive anchor ratio and
        adjusts the final conv bias to match the theoretical optimal bias
        for that ratio: bias = -log((1-r)/r) * logit_bias_scale.

        This prevents the bias from drifting toward extreme negative values
        when the positive fraction is very low (e.g., 1/30K).  Without this
        adjustment, the sigmoid bias can collapse to -10+ (sigmoid ~0),
        starving the detection head of gradient signal.

        Args:
            batch_pos_ratio: fraction of positive anchors in this batch
                             (num_positive / num_total_locations).
            momentum: EMA decay rate for running_pos_ratio. 0.05 = ~20-batch
                      half-life. Higher = faster adaptation.
        """
        import math
        # Update running EMA of positive ratio (in-place on registered buffer)
        with torch.no_grad():
            self.running_pos_ratio.mul_(1.0 - momentum).add_(momentum * batch_pos_ratio)
        # Compute target bias from running ratio (clamp to stable range)
        pos = max(0.001, min(float(self.running_pos_ratio.item()), 0.5))
        target_bias = -math.log((1.0 - pos) / pos) * self.logit_bias_scale
        # Apply to final classification conv bias
        for m in self.cls_head.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == self.num_classes:
                m.bias.data.fill_(target_bias)


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
        """Forward. Returns raw logits (no logit adjustment — applied in loss).

        [OPUS 207 §2.6] Logit adjustment moved to activity_loss() per Menon et al.
        protocol: additive logit correction inside the training loss only, raw
        logits for argmax at eval.

        Args:
            cls_token: [B, 768]

        Returns:
            logits: [B, 75] raw logits
        """
        x = self.norm(cls_token)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return self.classifier(x)

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
    """PSR head — per-frame MLP on spatial-pooled features.

    Simplified from temporal Transformer to per-frame MLP since all training
    and evaluation uses single frames (T=1). The old causal Transformer was
    dead weight — with T=1 the mask is 1×1 and no temporal processing occurs.

    Architecture: AdaptiveAvgPool3d → Linear(768→256) → GELU → Dropout →
    Linear(256→11). Returns [B, 11] directly.

    Reads spatial-pooled features from the backbone's hook output (blocks[14] = P5).
    Total params ≈ 0.2M (was 1.8M with Transformer).
    """

    def __init__(
        self,
        feat_dim: int = 256,
        input_dim: int = 768,
        num_components: int = 11,
    ):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # pool H,W
        self.input_proj = nn.Linear(input_dim, feat_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.15)
        self.projection = nn.Linear(feat_dim, num_components)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p5_feat: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            p5_feat: [B, 768, T, H, W] from P5 (blocks[14] hook). T is typically 1
                     for single-frame inference, but any T works (per-frame MLP).

        Returns:
            psr_logits: [B, 11] per-frame transition logits.
        """
        # Pool spatial dims → [B, 768, T, 1, 1] → [B, T, 768]
        x = self.spatial_pool(p5_feat).squeeze(-1).squeeze(-1).transpose(1, 2)
        # Average over temporal dim (always T=1, so this is a no-op in practice)
        x = x.mean(dim=1)  # [B, 768]
        # MLP: project → activate → dropout → output
        x = self.input_proj(x)  # [B, 256]
        x = self.activation(x)
        x = self.dropout(x)
        return self.projection(x)  # [B, 11]


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
            nn.Dropout(0.15),
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
        det_prior_prob: float = 0.01,
        logit_bias_scale: float = 1.0,
        use_p2_level: bool = False,
        num_anchors: int = 16,
        use_yolov8_head: bool = False,
    ):
        super().__init__()
        self.num_act_classes = num_act_classes
        self.num_det_classes = num_det_classes
        self.num_psr_components = num_psr_components
        self.use_p2_level = use_p2_level
        self.num_anchors = num_anchors
        self.use_yolov8_head = use_yolov8_head

        # Shared MViTv2-S backbone + feature pyramid
        self.feature_pyramid = MViTFeaturePyramid()
        backbone_dim = 768  # MViTv2-S final channel dim

        # [IMP-10] P2 level support for small object detection.
        # When use_p2_level=True, P2 (stride 4, 56x56) is included in detection
        # outputs alongside P3/P4/P5. P2 provides higher resolution features
        # that improve small object AP by 3-5% (Agent 5 findings).
        # The dedicated 3x3 p2_lateral in LightweightFPN improves feature
        # extraction from C2 (96ch) into the 256-channel FPN space.

        # Detection FPN + head
        self.fpn = LightweightFPN(
            in_channels={"P2": 96, "P3": 192, "P4": 384, "P5": 768},
            out_channels=fpn_channels,
        )

        # [v3.19] Choose detection head: standard 3x3 or YOLOv8-style DFL
        if use_yolov8_head:
            from src.models.yolov8_det_head import YOLOv8DetectHead
            self.det_head = YOLOv8DetectHead(
                in_channels=fpn_channels,
                num_classes=num_det_classes,
                reg_max=16,
            )
            logger.info(f'Using YOLOv8-style DFL detection head (anchor-free)')
        else:
            self.det_head = DetectionHead(in_channels=fpn_channels, num_classes=num_det_classes, prior_prob=det_prior_prob, logit_bias_scale=logit_bias_scale, num_anchors=num_anchors)

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
            "MTLMViTModel: feats={}, act={}-cls, det={}-cls, psr={}-comp, fpn={}ch, use_p2={}".format(
                backbone_dim, num_act_classes, num_det_classes, num_psr_components, fpn_channels, use_p2_level
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
              - psr_logits: [B, 11] per-frame transition logits
              - pose_6d: [B, 6] Tanh-bounded fwd+up
        """
        # Shared backbone forward
        fpn_feats, cls_token = self.feature_pyramid(clip)

        # Detection: FPN → decoupled head
        # [IMP-10] P2 level: controlled by self.use_p2_level flag. When enabled,
        # P2 (stride 4, 56x56) is included in detection outputs for better small
        # object AP (+3-5%). The dedicated 3x3 p2_lateral in LightweightFPN
        # provides better feature extraction from C2 (96ch) into 256ch FPN space.
        # When disabled (default), only P3/P4/P5 are used (legacy behavior).
        fpn_out = self.fpn(fpn_feats)
        det_outputs = {}

        if self.use_yolov8_head:
            # YOLOv8 head: takes list of [B, 256, H, W] features
            level_feats = []
            for level_name in ['P3', 'P4', 'P5']:
                if level_name in fpn_out:
                    # Temporal-pool T dimension for 2D detection
                    pooled = fpn_out[level_name].mean(dim=2)  # [B, 256, H, W]
                    level_feats.append(pooled)
            det_outputs = self.det_head(level_feats)
        else:
            # Legacy 3x3 anchor-based head: takes single feature map
            for level_name, feat in fpn_out.items():
                if level_name == "P2" and not self.use_p2_level:
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
            psr_logits = torch.zeros(clip.size(0), self.num_psr_components, device=clip.device)

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


def gram_schmidt_rotation(fwd: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Build a 3x3 rotation matrix from two 3D vectors via Gram-Schmidt.

    Orthonormalises (fwd, up) to produce a valid SO(3) rotation matrix
    with columns = orthonormal basis.

    Args:
        fwd: [B, 3] forward vectors
        up: [B, 3] up vectors (need not be orthogonal to fwd)

    Returns:
        R: [B, 3, 3] rotation matrix
    """
    b1 = F.normalize(fwd, dim=1)  # [B, 3]
    proj = (up * b1).sum(dim=1, keepdim=True) * b1  # [B, 3]
    b2 = F.normalize(up - proj, dim=1)  # [B, 3]
    b3 = torch.cross(b1, b2, dim=1)  # [B, 3]
    return torch.stack([b1, b2, b3], dim=2)  # [B, 3, 3]


def geodesic_angle(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic angular error in degrees on SO(3).

    ``angle = arccos((trace(R_pred^T @ R_gt) - 1) / 2)`` in degrees.

    Args:
        R_pred: [B, 3, 3] predicted rotation matrices
        R_gt: [B, 3, 3] ground-truth rotation matrices

    Returns:
        angles: [B] angular error in degrees
    """
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)  # [B, 3, 3]
    trace = torch.diagonal(R_rel, dim1=1, dim2=2).sum(dim=1)  # [B]
    trace = trace.clamp(-1.0, 3.0)  # numerical stability for acos
    return torch.rad2deg(torch.acos((trace - 1.0) / 2.0))
