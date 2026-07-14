"""
Video backbone + 4-task multi-task model (MViTv2-S / VideoMAE).

Replaces ConvNeXt-Tiny (frame-level, ImageNet-pretrained) with a video backbone
(Kinetics-pretrained, spatiotemporal) that feeds all 4 task heads from a single
shared encoder.  This is the primary architectural upgrade for activity recognition
SOTA (+5-12% Top-1 expected) and provides richer temporal features for PSR.

Architecture
============
  Input: clip [B, T, 3, 224, 224]  (T=16 frames, center-frame labeled)

  Video Backbone (MViTv2-S, 34.5M params, Kinetics-400 pretrained)
    │
    ├── conv_proj: Conv3d(3, 96, k=(3,7,7), s=(2,4,4))  -> [B, 96, T/2, H/4, W/4]
    │       └── extracted as C2 surrogate for per-frame spatial features
    │
    └── 16x MultiscaleBlocks
            │
            ├── Block 3  -> 384ch, stride 16  ->  C3 surrogate
            ├── Block 15 -> 768ch, stride 32  ->  C5 surrogate
            └── norm  ->  clip embedding [B, 768]

  SpatialFeatureAdapter:
    C3 (14x14) -> stride-2 conv -> C4 (7x7, 384ch)

  FPN -> P3-P7 (256ch) -> DetectionHead + PoseHead
  HeadPoseHead(C4, C5), PoseFiLM(C5), ActivityHead(embed + spatial), PSRHead(P3+P4+P5)

Design decisions
================
  1. Single backbone, all 4 heads share it (no separate spatial stem).
  2. MViTv2-S has spatial pooling only at block 3 (56->14) and block 15 (14->7).
     C4 is created artificially via stride-2 conv on C3 (14x14 -> 7x7).
  3. Center-frame slice for spatial heads (t=T'//2 after tubelet embedding).
  4. PoseFiLM and HeadPoseFiLM kept as-is from POPWMultiTaskModel.
  5. Gradient checkpointing on MViT blocks for memory (target: RTX 3060 11 GB).
  6. Initial backbone freeze (K400-pretrained), staged unfreeze.

Memory budget (estimated, batch=2, T=16, FP16):
  Video backbone forward pass:      ~2.8 GB (activation)
  Heads (FPN + 4 tasks):           ~1.2 GB
  Total train (no checkpoint):     ~7.5 GB
  With gradient checkpointing:     ~5.0 GB  -- fits RTX 3060 11 GB
  With activation offloading:      ~4.0 GB
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C

logger = logging.getLogger(__name__)


# ===========================================================================
# Video Backbone Wrapper -- MViTv2-S / VideoMAE with intermediate features
# ===========================================================================
class VideoBackboneWrapper(nn.Module):
    """Wraps a Kinetics-pretrained video backbone and exposes intermediate
    spatiotemporal features for both spatial and temporal heads.

    Supported backbones:
      - 'mvit_v2_s'  : MViTv2-Small (torchvision, 34.5M params, 768-d clip embed)
      - 'videomae_s' : VideoMAE-Small (HF transformers, 22M params, 384-d clip embed)

    MViTv2-S architecture (16 blocks):
      conv_proj: Conv3d(3, 96, k=(3,7,7), s=(2,4,4)) -> [B, 96, T/2, H/4, W/4]
      Block 0-2:  96ch,  thw=(8, 56, 56)
      Block 3:    SPATIAL POOL: (56,56) -> (14,14), D=384
      Block 4-14: 384ch, thw=(8, 14, 14)  (no further spatial pooling)
      Block 15:   SPATIAL POOL: (14,14) -> (7,7), temporal pool: 8->1, D=768
      norm:       LayerNorm(768)

    Since spatial pooling happens only at block 3 (56->14) and block 15 (14->7),
    there is no native 28x28 intermediate.  The SpatialFeatureAdapter creates
    the missing C4 scale via a lightweight stride-2 conv.

    For spatial heads, the center frame is selected from the spatiotemporal
    feature map (t = T' // 2 after tubelet embedding temporal reduction).
    """

    def __init__(
        self,
        backbone_name: str = "mvit_v2_s",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        use_checkpoint: bool = True,
        clip_frames: int = 16,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.clip_frames = clip_frames
        self.use_checkpoint = use_checkpoint

        if backbone_name == "mvit_v2_s":
            self._build_mvit_v2_s(pretrained)
        elif backbone_name == "videomae_s":
            self._build_videomae_s(pretrained)
        else:
            raise ValueError(f"Unknown video backbone: {backbone_name}")

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("VideoBackboneWrapper: backbone frozen (pretrained)")
        else:
            logger.info("VideoBackboneWrapper: backbone trainable (fine-tune)")

        if use_checkpoint:
            logger.info("VideoBackboneWrapper: gradient checkpointing enabled")

    def _build_mvit_v2_s(self, pretrained: bool):
        """Build MViTv2-Small backbone from torchvision."""
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = mvit_v2_s(weights=weights)
        self.encoder = model
        self.hidden_size = 768

        # Verified dimensions by trace:
        #   conv_proj:   96ch, thw=(8, 56, 56)   -- stride 4
        #   block 3:    384ch, thw=(8, 14, 14)   -- stride 16
        #   block 15+:  768ch, thw=(8, 7, 7)     -- stride 32
        self.feature_dims = {"c2": 96, "c3": 384, "c4": 384, "c5": 768}

    def _build_videomae_s(self, pretrained: bool):
        """Build VideoMAE-Small backbone from HuggingFace transformers."""
        try:
            from transformers import VideoMAEModel, VideoMAEConfig

            config = VideoMAEConfig(
                hidden_size=384,
                num_hidden_layers=12,
                intermediate_size=1536,
                num_attention_heads=16,
                image_size=224,
                patch_size=16,
                num_frames=self.clip_frames,
                tubelet_size=2,
                qkv_bias=True,
                use_mean_pooling=True,
            )
            model = VideoMAEModel(config)
            if pretrained:
                _cache = (
                    "~/.cache/huggingface/hub/"
                    "models--MCG-NJU--videomae-small-finetuned-kinetics/"
                    "snapshots/240e9734611173accbbf74cbdf4b641e4c431264/model.safetensors"
                )
                import os

                _cache = os.path.expanduser(_cache)
                if os.path.exists(_cache):
                    from safetensors.torch import load_file as _load_sf

                    sd = _load_sf(_cache)
                    model.load_state_dict(sd, strict=False)
                    logger.info("VideoMAE-S: loaded K400-pretrained weights")

            self.encoder = model
            self.hidden_size = 384
            self.feature_dims = {"c2": 384, "c3": 384, "c4": 384, "c5": 384}
        except Exception as ex:
            logger.error(f"VideoMAE-S build failed: {ex}")
            raise

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through video backbone.

        Args:
            clip: [B, 3, T, H, W] or [B, T, 3, H, W] video clip.
        Returns:
            dict with:
              - 'clip_embed': [B, hidden_size] clip-level temporal embedding.
              - 'stage_features': {c2, c3, c5} spatial feature maps, each [B, D, H, W].
              - 'c2', 'c3', 'c5': individual spatial features.
        """
        if clip.dim() == 5 and clip.shape[1] != 3:
            clip = clip.permute(0, 2, 1, 3, 4).contiguous()

        if self.backbone_name == "mvit_v2_s":
            return self._forward_mvit(clip)
        elif self.backbone_name == "videomae_s":
            return self._forward_videomae(clip)

        raise RuntimeError(f"Unknown backbone: {self.backbone_name}")

    def _forward_mvit(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """MViTv2-S forward with intermediate feature extraction.

        Feature extraction points (verified by runtime trace):
          conv_proj:  [B, 96, T/2, H/4, W/4]  -> C2 (stride 4)
          block 3:    [B, N, 384], thw=(T/2, H/16, W/16) -> C3 (stride 16)
          block 15:   [B, N, 768], thw=(T/2, H/32, W/32) -> C5 (stride 32)
          norm:       [B, N, 768] -> clip_embed via mean pool

        C4 created artificially by SpatialFeatureAdapter (stride-2 conv on C3).
        """
        x_5d = self.encoder.conv_proj(clip)  # [B, 96, T', H', W']
        t_idx = x_5d.shape[2] // 2
        c2 = x_5d[:, :, t_idx, :, :]  # [B, 96, H/4, W/4] -- center frame

        x = x_5d.flatten(2).transpose(1, 2).contiguous()
        thw = (x_5d.shape[2], x_5d.shape[3], x_5d.shape[4])

        x = self.encoder.pos_encoding(x)

        c3 = None
        c5 = None
        for i, block in enumerate(self.encoder.blocks):
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                x, thw = checkpoint(block, x, thw, use_reentrant=False)
            else:
                x, thw = block(x, thw)

            if i == 3:
                c3 = self._patches_to_spatial(x, thw, center_frame=True)
            if i == 15:
                c5 = self._patches_to_spatial(x, thw, center_frame=True)

        x = self.encoder.norm(x)
        clip_embed = x.mean(dim=1)  # [B, 768]

        return {
            "clip_embed": clip_embed,
            "stage_features": {"c2": c2, "c3": c3, "c5": c5},
            "c2": c2,
            "c3": c3,
            "c5": c5,
        }

    def _forward_videomae(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """VideoMAE forward with hidden states for intermediate features.

        VideoMAE has no hierarchical spatial pooling -- all stage features
        are at the same spatial resolution (14x14 for 224px input).
        """
        outputs = self.encoder(pixel_values=clip, output_hidden_states=True)
        x = outputs.last_hidden_state  # [B, N, D]
        clip_embed = x.mean(dim=1)  # [B, 384]

        N = x.shape[1]
        H_sp = W_sp = int((N - 1) ** 0.5)
        patches = x[:, 1:, :]
        spatial_feat = patches.reshape(-1, H_sp, W_sp, 384).permute(0, 3, 1, 2)

        return {
            "clip_embed": clip_embed,
            "stage_features": {"c2": spatial_feat, "c3": spatial_feat, "c5": spatial_feat},
            "c2": spatial_feat,
            "c3": spatial_feat,
            "c5": spatial_feat,
        }

    @staticmethod
    def _patches_to_spatial(x, thw, center_frame=True):
        """Reshape MViT patch-sequence [B, N, D] to 2D spatial [B, D, H, W]."""
        B, N, D = x.shape
        T_p, H_p, W_p = thw
        offset = 1 if N > T_p * H_p * W_p else 0
        patches = x[:, offset:, :]
        patches = patches.reshape(B, T_p, H_p, W_p, D)
        patches = patches.permute(0, 4, 1, 2, 3)
        if center_frame:
            return patches[:, :, T_p // 2, :, :]  # [B, D, H, W]
        return patches.mean(dim=2)


# ===========================================================================
# Spatial Feature Adapter -- creates C4 (missing scale) from C3
# ===========================================================================
class SpatialFeatureAdapter(nn.Module):
    """Creates a multi-scale spatial feature pyramid from video backbone output.

    MViTv2-S only produces two native spatial scales: C3 at stride 16 (14x14
    for 224px input) and C5 at stride 32 (7x7).  This adapter:
      - Passes C3 and C5 through as-is.
      - Creates an artificial C4 at stride 32 by a stride-2 conv on C3 features.

    This allows the FPN to receive three spatial scales (C3, C4, C5) matching
    the existing POPWMultiTaskModel interface.
    """

    def __init__(self, c3_channels: int = 384, c4_channels: int = 384, c5_channels: int = 768):
        super().__init__()
        self.c3_to_c4 = nn.Conv2d(c3_channels, c4_channels, 3, stride=2, padding=1)
        nn.init.kaiming_uniform_(self.c3_to_c4.weight, a=1)
        nn.init.zeros_(self.c3_to_c4.bias)

    def forward(
        self, stage_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3 = stage_features["c3"]
        c5 = stage_features["c5"]
        if c3.shape[-2:] != c5.shape[-2:]:
            c4 = self.c3_to_c4(c3)
        else:
            c4 = F.avg_pool2d(c3, 2) if c3.shape[-1] > c5.shape[-1] else c3
        return c3, c4, c5


# ===========================================================================
# FPN (adapted for video backbone channel dims)
# ===========================================================================
class VideoFPN(nn.Module):
    """FPN adapted for video backbone features.  Same topology as the existing
    FPN in model.py but accepts per-level channel dims instead of a list.
    """

    def __init__(
        self, c3_channels: int, c4_channels: int, c5_channels: int, out_channels: int = 256
    ):
        super().__init__()
        self.lateral_c3 = nn.Conv2d(c3_channels, out_channels, 1)
        self.lateral_c4 = nn.Conv2d(c4_channels, out_channels, 1)
        self.lateral_c5 = nn.Conv2d(c5_channels, out_channels, 1)

        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.p6_conv = nn.Conv2d(c5_channels, out_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c3, c4, c5):
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        p6 = self.p6_conv(c5)
        p7 = self.p7_conv(F.relu(p6))
        return {"p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7}


# ===========================================================================
# Video Multi-Task Model (single backbone, 4 heads)
# ===========================================================================
class VideoMultiTaskModel(nn.Module):
    """Single video backbone (MViTv2-S / VideoMAE) feeding all 4 task heads.

    This replaces ConvNeXt-Tiny + separate VideoMAEStream with a single
    Kinetics-pretrained video backbone.  Spatial heads (detection, pose,
    head pose) extract features from intermediate stages of the video
    backbone at the clip center frame.  Temporal heads (activity, PSR)
    use the clip-level embedding and temporal aggregation.
    """

    def __init__(
        self,
        backbone_name: str = "mvit_v2_s",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        use_checkpoint: bool = True,
        clip_frames: int = 16,
        num_classes_detection: int = 24,
        num_classes_activity: int = 75,
        num_components_psr: int = 11,
        num_keypoints: int = 17,
        use_headpose_film: bool = True,
        use_hand_film: bool = True,
        train_pose: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_headpose_film = use_headpose_film
        self.use_hand_film = use_hand_film
        self.train_pose = train_pose

        # -- Video Backbone --
        self.video_backbone = VideoBackboneWrapper(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_checkpoint=use_checkpoint,
            clip_frames=clip_frames,
        )
        feat = self.video_backbone.feature_dims
        self.c2_ch = feat["c2"]
        self.c3_ch = feat["c3"]
        self.c4_ch = feat["c4"]
        self.c5_ch = feat["c5"]

        # -- Spatial Feature Adapter --
        self.spatial_adapter = SpatialFeatureAdapter(
            c3_channels=feat["c3"],
            c4_channels=feat["c4"],
            c5_channels=feat["c5"],
        )

        # -- FPN --
        self.fpn = VideoFPN(
            c3_channels=feat["c3"],
            c4_channels=feat["c4"],
            c5_channels=feat["c5"],
            out_channels=256,
        )

        # -- Detection Head (RetinaNet-style) --
        from src.models.model import DetectionHead, AnchorGenerator

        self.detection_head = DetectionHead(
            in_channels=256,
            num_classes=num_classes_detection,
            detach_reg_fpn=getattr(C, "DETACH_REG_FPN", False),
        )
        self.anchor_gen = AnchorGenerator()

        # -- Pose Head (17 keypoints) --
        from src.models.model import PoseHead

        self.pose_head = PoseHead(
            in_channels=256,
            num_keypoints=num_keypoints,
            temperature=getattr(C, "SOFT_ARGMAX_TEMPERATURE", 0.1),
            training_temperature=getattr(C, "SOFT_ARGMAX_TEMP_TRAIN", 1.0),
        )

        # -- PoseFiLM (keypoint-conditioned FiLM on C5) --
        from src.models.model import PoseFiLMModule

        if use_hand_film:
            self.pose_film = PoseFiLMModule(
                num_keypoints=num_keypoints,
                c5_channels=feat["c5"],
                hidden_channels=512,
            )

        # -- HeadPoseFiLM --
        from src.models.model import HeadPoseFiLMModule

        if use_headpose_film:
            self.headpose_film = HeadPoseFiLMModule(
                c5_channels=feat["c5"],
                hidden_channels=256,
            )

        # -- Head Pose Head (9-DoF) --
        if getattr(C, "USE_GEO_HEAD_POSE", False):
            from src.models.head_pose_geo import GeometryAwareHeadPose

            self.head_pose_head = GeometryAwareHeadPose(
                in_channels_c4=feat["c4"],
                in_channels_c5=feat["c5"],
                hidden_dim=512,
            )
        else:
            from src.models.model import HeadPoseHead

            self.head_pose_head = HeadPoseHead(
                c4_channels=feat["c4"],
                c5_channels=feat["c5"],
                hidden_dim=128,
            )

        # -- Activity Head --
        from src.models.model import ActivityHead

        self.activity_head = ActivityHead(
            c5_channels=feat["c5"],
            p4_channels=256,
            det_conf_size=num_classes_detection,
            embed_dim=512,
            num_classes=num_classes_activity,
            dropout=0.1,
            window_size=16,
            use_vit=True,
            use_videomae=feat["c5"] != self.video_backbone.hidden_size,
        )

        # -- PSR Head --
        from src.models.model import PSRHead

        self.psr_head = PSRHead(
            in_channels=256,
            hidden_dim=128,
            num_components=num_components_psr,
            dropout=0.2,
        )

        # -- Feature Bank (temporal, for activity head) --
        from src.models.model import FeatureBank

        self.feature_bank = FeatureBank(embed_dim=512, window_size=16)

        logger.info(
            "VideoMultiTaskModel: backbone=%s, hidden=%d, clip_frames=%d, freeze=%s, checkpoint=%s",
            backbone_name,
            self.video_backbone.hidden_size,
            clip_frames,
            freeze_backbone,
            use_checkpoint,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        clip: torch.Tensor,
        video_ids: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            clip: [B, C, T, H, W] or [B, T, C, H, W] video clip.
            video_ids: list of video IDs for Feature Bank (optional).
        Returns:
            dict matching POPWMultiTaskModel output interface for loss compatibility.
        """
        # -- 1. Video backbone --
        backbone_out = self.video_backbone(clip)
        clip_embed = backbone_out["clip_embed"]  # [B, D]
        stage = backbone_out["stage_features"]  # {c2, c3, c5}
        c2, c3, c5 = stage["c2"], stage["c3"], stage["c5"]

        # -- 2. SpatialFeatureAdapter + FPN --
        c3, c4, c5 = self.spatial_adapter({"c3": c3, "c5": c5})
        pyramid = self.fpn(c3, c4, c5)

        # -- 3. Detection Head --
        cls_preds, reg_preds = self.detection_head(pyramid)
        anchors = self.anchor_gen(pyramid)

        # -- 4. Pose Head (on P3) --
        heatmaps, keypoints, pose_confidence = self.pose_head(pyramid["p3"])

        # -- 5. PoseFiLM (keypoint-conditioned modulation of C5) --
        if self.use_hand_film and hasattr(self, "pose_film"):
            c5_mod = self.pose_film(c5, keypoints.detach(), pose_confidence)
        else:
            c5_mod = c5

        # -- 6. Head Pose --
        if self.train_pose or not self.training:
            head_pose = self.head_pose_head(c4, c5)
            if isinstance(head_pose, tuple):
                _rot6d, _rot_mat, _pos = head_pose
                _forward = _rot_mat[:, :, 0]
                _up = _rot_mat[:, :, 2]
                head_pose = torch.cat([_forward, _pos, _up], dim=1)
            if self.use_headpose_film and hasattr(self, "headpose_film"):
                c5_mod = self.headpose_film(c5_mod, head_pose.detach())
        else:
            head_pose = None

        # -- 7. Activity Head --
        det_conf = torch.sigmoid(cls_preds.max(dim=1)[0])
        activity_proj = torch.cat(
            [
                det_conf,
                F.adaptive_avg_pool2d(c5_mod, 1).flatten(1),
                F.adaptive_avg_pool2d(pyramid["p4"].detach(), 1).flatten(1),
            ],
            dim=1,
        )
        proj_feat = self.activity_head.proj_features(activity_proj)

        _staging_enabled = bool(getattr(C, "STAGED_TRAINING", False))
        bank_output = self.feature_bank(proj_feat, video_ids, None) if _staging_enabled else None

        # Pass clip_embed as videomae_feat if activity head expects fusion
        videomae_feat = (
            clip_embed
            if (hasattr(self.activity_head, "use_videomae") and self.activity_head.use_videomae)
            else None
        )

        act_logits = self.activity_head(
            proj_feat=proj_feat,
            temporal_bank=bank_output,
            videomae_feat=videomae_feat,
        )

        # -- 8. PSR Head --
        psr_full = self.psr_head(pyramid)
        psr_logits = psr_full[..., :11]
        psr_confidence = psr_full[..., 11:]

        # -- 9. Keypoint normalization --
        if self.train_pose:
            _, _, grid_h, grid_w = heatmaps.shape
            kp_scale = torch.tensor(
                [grid_w, grid_h], device=keypoints.device, dtype=keypoints.dtype
            )
            keypoints = keypoints / kp_scale.view(1, 1, 2)

        # -- Output dict (matches POPWMultiTaskModel keys) --
        return {
            "cls_preds": cls_preds,
            "reg_preds": reg_preds,
            "anchors": anchors,
            "heatmaps": heatmaps,
            "keypoints": keypoints,
            "pose_confidence": pose_confidence,
            "head_pose": head_pose,
            "c5_mod": c5_mod,
            "det_conf": det_conf,
            "act_logits": act_logits,
            "psr_logits": psr_logits,
            "psr_confidence": psr_confidence if not self.training else None,
            "clip_embed": clip_embed,
            "c5_raw": c5,
            "proj_feat": proj_feat,
            "p4": pyramid["p4"],
        }


# ===========================================================================
# Parameter counting
# ===========================================================================
def count_parameters(model: VideoMultiTaskModel) -> Dict[str, int]:
    """Count trainable and total parameters per component."""
    components = {
        "video_backbone": [model.video_backbone],
        "spatial_adapter": [model.spatial_adapter],
        "fpn": [model.fpn],
        "detection": [model.detection_head],
        "pose_head": [model.pose_head],
    }
    if hasattr(model, "pose_film"):
        components["pose_film"] = [model.pose_film]
    if hasattr(model, "headpose_film"):
        components["headpose_film"] = [model.headpose_film]
    components.update(
        {
            "activity_head": [model.activity_head],
            "psr_head": [model.psr_head],
            "head_pose_head": [model.head_pose_head],
            "feature_bank": [model.feature_bank],
        }
    )
    result = {}
    total = 0
    for name, modules in components.items():
        count = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
        result[name] = count
        total += count
    result["total_trainable"] = total
    result["total_all"] = sum(p.numel() for p in model.parameters())
    return result


# ===========================================================================
# Training Strategy Helpers
# ===========================================================================
def get_trainable_param_groups(
    model: VideoMultiTaskModel,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
) -> List[Dict]:
    """Parameter groups with different LRs for backbone vs heads."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if "video_backbone" in name else head_params).append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": head_lr, "name": "heads"},
    ]


def unfreeze_backbone_stages(model: VideoMultiTaskModel, num_stages: int = 1):
    """Gradually unfreeze video backbone stages (from last to first)."""
    blocks = model.video_backbone.encoder.blocks
    total_blocks = len(blocks)
    blocks_per_stage = total_blocks // 4
    stages_to_unfreeze = min(num_stages, 4)
    for stage_idx in range(4 - stages_to_unfreeze, 4):
        start = stage_idx * blocks_per_stage
        end = start + blocks_per_stage
        for i in range(start, end):
            for p in blocks[i].parameters():
                p.requires_grad = True
        logger.info("Unfroze backbone stage %d (blocks %d-%d)", stage_idx, start, end - 1)
