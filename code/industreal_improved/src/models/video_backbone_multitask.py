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
    ├── conv_proj: Conv3d(3, 96, k=(3,7,7), s=(2,4,4))  → [B, 96, T/2, H/4, W/4]
    │       └── extracted as "C2" surrogate for per-frame spatial features
    │
    └── 16× MultiscaleBlocks (stages pool spatial/temporal resolution)
            │
            ├── Block 3  →  D~192, pool by 2  →  "C3" surrogate
            ├── Block 7  →  D~384, pool by 2  →  "C4" surrogate
            ├── Block 11 →  D~768, pool by 2  →  "C5" surrogate
            └── Block 15 →  D=768, final      →  clip embedding

  Spatial heads (per-frame, extracted at clip center frame):
    conv_proj / block-X features → reshape [B, D, T', H', W'] → select t=T'//2
      │
      ├── FPN (P3-P7, 256ch) → DetectionHead + PoseHead
      ├── HeadPoseHead(C4, C5)
      ├── PoseFiLM(C5) → HeadPoseFiLM(C5_mod) → c5_mod
      └── PSRHead(P3, P4, P5)

  Temporal heads (clip-level):
    ActivityHead:  clip_embed [B, 768] ‖ GAP(c5_mod) ‖ GAP(P4) ‖ det_conf → classifier
    PSR head also uses temporal sequence via its CausalTransformer (per-frame features)

Design decisions
================
  1. Single backbone, all 4 heads share it (no separate spatial stem).
  2. Intermediate features from MViT blocks are reshaped from patch-sequence
     back to 2D spatial maps for FPN compatibility.
  3. Center-frame slice for spatial heads (t=T'//2 after tubelet embedding).
  4. PoseFiLM and HeadPoseFiML kept as-is from POPWMultiTaskModel.
  5. Gradient checkpointing on MViT blocks for memory (target: RTX 3060 11 GB).
  6. Initial backbone freeze (K400-pretrained), staged unfreeze.

Memory budget (estimated, batch=2, T=16, FP16):
  Video backbone forward pass:      ~2.8 GB (activation)
  Heads (FPN + 4 tasks):           ~1.2 GB
  Total train (no checkpoint):     ~7.5 GB
  With gradient checkpointing:     ~5.0 GB  ← fits RTX 3060 11 GB
  With activation offloading:      ~4.0 GB
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C

logger = logging.getLogger(__name__)


# ===========================================================================
# Video Backbone Wrapper — MViTv2-S / VideoMAE with intermediate features
# ===========================================================================
class VideoBackboneWrapper(nn.Module):
    """Wraps a Kinetics-pretrained video backbone and exposes intermediate
    spatiotemporal features for both spatial and temporal heads.

    Supported backbones:
      - 'mvit_v2_s'  : MViTv2-Small (torchvision, 34.5M params, 768-d clip embed)
      - 'videomae_s' : VideoMAE-Small (HF transformers, 22M params, 384-d clip embed)

    The wrapper runs the backbone's convolutional stem and transformer blocks,
    capturing features at predefined stages.  Stage outputs are reshaped from
    patch-sequence format (B, N, D) back to 2D spatial feature maps (B, D, H, W)
    by using the (T, H, W) metadata from the MViT's internal thw tracking.

    For spatial heads, the center frame is selected from the spatiotemporal
    feature map (t = T' // 2 after tubelet embedding temporal reduction).
    """

    # Stage indices where features are captured
    # MViTv2-S: 16 blocks; capture at block 3, 7, 11, 15
    STAGE_CAPTURE_IDX = {
        'mvit_v2_s':  {'c2_stage': 0, 'c3_stage': 3, 'c4_stage': 7, 'c5_stage': 11},
        'videomae_s': {'c2_stage': 0, 'c3_stage': 3, 'c4_stage': 7, 'c5_stage': 11},
    }

    def __init__(
        self,
        backbone_name: str = 'mvit_v2_s',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        use_checkpoint: bool = True,
        clip_frames: int = 16,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.clip_frames = clip_frames
        self.use_checkpoint = use_checkpoint

        if backbone_name == 'mvit_v2_s':
            self._build_mvit_v2_s(pretrained)
        elif backbone_name == 'videomae_s':
            self._build_videomae_s(pretrained)
        else:
            raise ValueError(f'Unknown video backbone: {backbone_name}')

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info('VideoBackboneWrapper: backbone frozen (pretrained)')
        else:
            logger.info('VideoBackboneWrapper: backbone trainable (fine-tune)')

        if use_checkpoint:
            logger.info('VideoBackboneWrapper: gradient checkpointing enabled')

    def _build_mvit_v2_s(self, pretrained: bool):
        """Build MViTv2-Small backbone from torchvision."""
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = mvit_v2_s(weights=weights)
        self.encoder = model
        self.hidden_size = 768  # final norm output dim
        self.feature_dims = {
            'c2': 96,    # after conv_proj
            'c3': 192,   # after ~block 4 (approximate: MViT doubles dim gradually)
            'c4': 384,   # after ~block 8
            'c5': 768,   # after ~block 12
        }
        self.stage_indices = self.STAGE_CAPTURE_IDX['mvit_v2_s']

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
                    '~/.cache/huggingface/hub/'
                    'models--MCG-NJU--videomae-small-finetuned-kinetics/'
                    'snapshots/240e9734611173accbbf74cbdf4b641e4c431264/model.safetensors'
                )
                import os
                _cache = os.path.expanduser(_cache)
                if os.path.exists(_cache):
                    from safetensors.torch import load_file as _load_sf
                    sd = _load_sf(_cache)
                    model.load_state_dict(sd, strict=False)
                    logger.info('VideoMAE-S: loaded K400-pretrained weights')

            self.encoder = model
            self.hidden_size = 384
            self.feature_dims = {
                'c2': 384,  # VideoMAE uses a single hidden dim throughout
                'c3': 384,
                'c4': 384,
                'c5': 384,
            }
        except Exception as ex:
            logger.error(f'VideoMAE-S build failed: {ex}')
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
              - 'clip_embed': [B, hidden_size] clip-level feature for activity head
              - 'stage_features': dict of {c2, c3, c4, c5} spatial feature maps
                                 each [B, D, H', W'] (center frame, spatial only)
              - 'full_features': dict of {c3, c4, c5} raw block outputs [B, N, D]
        """
        # Normalize input to [B, C, T, H, W]
        if clip.dim() == 5 and clip.shape[1] != 3:
            clip = clip.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = clip.shape

        # ---- conv_proj: [B, C, T, H, W] -> [B, D, T', H', W'] ----
        c2_spatial = None
        c2_full = None
        if self.backbone_name == 'mvit_v2_s':
            x = self.encoder.conv_proj(clip)  # [B, 96, T/2, H/4, W/4]
            c2_full = x  # [B, D, T', H', W'] for later slicing
            # Select center frame for spatial head
            t_idx = x.shape[2] // 2
            c2_spatial = x[:, :, t_idx:t_idx+1, :, :].squeeze(2)  # [B, 96, H/4, W/4]

            # Flatten to patch sequence: [B, D, T', H', W'] -> [B, T'H'W', D] -> [B, N, D]
            thw_shape = (x.shape[2], x.shape[3], x.shape[4])  # (T', H', W')
            x = x.flatten(2).transpose(1, 2)  # [B, T'H'W', D]

            # Position encoding (adds class token)
            x = self.encoder.pos_encoding(x)  # [B, 1 + THW, D]
            N = x.shape[1]
        else:
            # VideoMAE: directly pass pixel_values
            outputs = self.encoder(pixel_values=clip, output_hidden_states=True)
            x = outputs.last_hidden_state  # [B, N, D]
            c2_spatial = outputs.hidden_states[0].mean(dim=2) if hasattr(outputs, 'hidden_states') else None
            thw_shape = None  # not needed for ViT-MAE reshape

        # ---- Blocks ----
        stage_features = {}
        full_features = {}
        stage_outputs = {}

        if self.backbone_name == 'mvit_v2_s':
            # Track thw through blocks
            thw = (c2_full.shape[2], c2_full.shape[3], c2_full.shape[4])
            for i, block in enumerate(self.encoder.blocks):
                if self.use_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint
                    x, thw = checkpoint(block, x, thw, use_reentrant=False)
                else:
                    x, thw = block(x, thw)

                # Capture at predefined stages
                if i == self.stage_indices['c3_stage']:
                    full_features['c3'] = (x, thw)
                    stage_features['c3'] = self._mvit_patches_to_spatial(x, thw, center_frame=True)
                elif i == self.stage_indices['c4_stage']:
                    full_features['c4'] = (x, thw)
                    stage_features['c4'] = self._mvit_patches_to_spatial(x, thw, center_frame=True)
                elif i == self.stage_indices['c5_stage']:
                    full_features['c5'] = (x, thw)
                    stage_features['c5'] = self._mvit_patches_to_spatial(x, thw, center_frame=True)

            # Norm + pool for clip embedding
            x = self.encoder.norm(x)  # [B, N, D]
            clip_embed = x[:, 0] if x.shape[1] > thw[0]*thw[1]*thw[2] else x.mean(dim=1)
            # x[:, 0] is class token when present; otherwise mean pool

        else:
            # VideoMAE: use last hidden state mean
            clip_embed = x.mean(dim=1)  # [B, 384]
            # For VideoMAE, spatial feature extraction is more limited
            # (no hierarchical pooling in ViT). Use simple reshape.
            for layer_name in ('c3', 'c4', 'c5'):
                stage_features[layer_name] = c2_spatial  # placeholder
                full_features[layer_name] = x

        # Store features
        result = {
            'clip_embed': clip_embed,  # [B, hidden_size]
            'stage_features': stage_features,
            'full_features': full_features,
            'c2': c2_spatial,  # [B, D, H/4, W/4] or None
            'hidden_size': self.hidden_size,
        }
        return result

    @staticmethod
    def _mvit_patches_to_spatial(
        x: torch.Tensor,
        thw: Tuple[int, int, int],
        center_frame: bool = True,
    ) -> torch.Tensor:
        """Reshape MViT patch-sequence output [B, 1+T*H*W, D] to spatial feature
        map [B, D, H', W'], optionally extracting the center frame.

        The MViT adds a class token at position 0, so patches start at index 1.
        """
        B, N, D = x.shape
        has_cls = N > thw[0] * thw[1] * thw[2]
        offset = 1 if has_cls else 0
        T_pooled, H_pooled, W_pooled = thw

        # Extract patch tokens (skip class token)
        patches = x[:, offset:, :]  # [B, T*H*W, D]
        # Reshape to spatiotemporal grid
        patches = patches.reshape(B, T_pooled, H_pooled, W_pooled, D)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()  # [B, D, T, H, W]

        if center_frame:
            t_idx = T_pooled // 2
            spatial = patches[:, :, t_idx, :, :]  # [B, D, H, W]
        else:
            # Temporal mean pool
            spatial = patches.mean(dim=2)  # [B, D, H, W]

        return spatial


# ===========================================================================
# Lightweight FPN variant for video backbone features
# ===========================================================================
class VideoFPN(nn.Module):
    """FPN adapted for video backbone features.

    Takes stage features {c3, c4, c5} with backbone-specific channel dims and
    produces P3-P7 pyramid (all 256 channels), matching the existing FPN
    interface consumed by DetectionHead, PoseHead, and PSRHead.
    """

    def __init__(self, in_channels: Dict[str, int], out_channels: int = 256):
        super().__init__()
        self.lateral_c3 = nn.Conv2d(in_channels['c3'], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels['c4'], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels['c5'], out_channels, 1)

        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.p6_conv = nn.Conv2d(in_channels['c5'], out_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
            c3: [B, D_c3, H3, W3]
            c4: [B, D_c4, H4, W4]
            c5: [B, D_c5, H5, W5]
        Returns:
            pyramid dict: {'p3': ..., 'p4': ..., 'p5': ..., 'p6': ..., 'p7'}
        """
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')

        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        p6 = self.p6_conv(c5)
        p7 = self.p7_conv(F.relu(p6))

        return {'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, 'p7': p7}


# ===========================================================================
# Video Multi-Task Model (single backbone, 4 heads)
# ===========================================================================
class VideoMultiTaskModel(nn.Module):
    """Single video backbone (MViTv2-S / VideoMAE) feeding all 4 task heads.

    This replaces ConvNeXt-Tiny + separate VideoMAEStream with a single
    Kinetics-pretrained video backbone. Spatial heads (detection, pose,
    head pose) extract features from intermediate stages of the video
    backbone at the clip center frame.  Temporal heads (activity, PSR)
    use the clip-level embedding and temporal aggregation.

    Architecture (see module docstring for full diagram):
      clip [B, 3, T, H, W]
        └→ VideoBackboneWrapper
              ├→ stage_features: {c3, c4, c5}  →  VideoFPN → P3-P7
              │     ├── DetectionHead(P3-P7)
              │     ├── PoseHead(P3)
              │     ├── PoseFiLM(C5) ─→ HeadPoseFiLM ─→ c5_mod
              │     ├── HeadPoseHead(C4, C5)
              │     └── PSRHead(P3, P4, P5)
              └→ clip_embed: [B, D]  ─→ ActivityHead (fused with spatial)
    """

    def __init__(
        self,
        backbone_name: str = 'mvit_v2_s',
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

        # ---- Video Backbone ----
        self.video_backbone = VideoBackboneWrapper(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_checkpoint=use_checkpoint,
            clip_frames=clip_frames,
        )

        feature_dims = self.video_backbone.feature_dims
        self.c2_channels = feature_dims['c2']  # for potential C2 usage
        self.c3_channels = feature_dims['c3']
        self.c4_channels = feature_dims['c4']
        self.c5_channels = feature_dims['c5']

        # ---- FPN ----
        self.fpn = VideoFPN(
            in_channels={'c3': feature_dims['c3'], 'c4': feature_dims['c4'], 'c5': feature_dims['c5']},
            out_channels=256,
        )

        # ---- Detection Head (RetinaNet-style) ----
        # Uses the same DetectionHead as POPWMultiTaskModel — operates on FPN P3-P7
        from src.models.model import DetectionHead, AnchorGenerator
        self.detection_head = DetectionHead(
            in_channels=256,
            num_classes=num_classes_detection,
            detach_reg_fpn=getattr(C, 'DETACH_REG_FPN', False),
        )
        self.anchor_gen = AnchorGenerator()

        # ---- Pose Head (17 keypoints) ----
        from src.models.model import PoseHead
        self.pose_head = PoseHead(
            in_channels=256,
            num_keypoints=num_keypoints,
            temperature=getattr(C, 'SOFT_ARGMAX_TEMPERATURE', 0.1),
            training_temperature=getattr(C, 'SOFT_ARGMAX_TEMP_TRAIN', 1.0),
        )

        # ---- PoseFiLM (keypoint-conditioned FiLM on C5) ----
        from src.models.model import PoseFiLMModule
        if use_hand_film:
            self.pose_film = PoseFiLMModule(
                num_keypoints=num_keypoints,
                c5_channels=feature_dims['c5'],
                hidden_channels=512,
            )

        # ---- HeadPoseFiLM (Doc 01 E) ----
        from src.models.model import HeadPoseFiLMModule
        if use_headpose_film:
            self.headpose_film = HeadPoseFiLMModule(
                c5_channels=feature_dims['c5'],
                hidden_channels=256,
            )

        # ---- Head Pose Head (9-DoF) ----
        if getattr(C, 'USE_GEO_HEAD_POSE', False):
            from src.models.head_pose_geo import GeometryAwareHeadPose
            self.head_pose_head = GeometryAwareHeadPose(
                in_channels_c4=feature_dims['c4'],
                in_channels_c5=feature_dims['c5'],
                hidden_dim=512,
            )
        else:
            from src.models.model import HeadPoseHead
            self.head_pose_head = HeadPoseHead(
                c4_channels=feature_dims['c4'],
                c5_channels=feature_dims['c5'],
                hidden_dim=128,
            )

        # ---- Activity Head ----
        from src.models.model import ActivityHead
        self.activity_head = ActivityHead(
            c5_channels=feature_dims['c5'],
            p4_channels=256,
            det_conf_size=num_classes_detection,
            embed_dim=512,
            num_classes=num_classes_activity,
            dropout=0.1,
            window_size=16,
            use_vit=True,
            use_videomae=False,  # video backbone replaces VideoMAE stream
        )

        # ---- PSR Head ----
        from src.models.model import PSRHead
        self.psr_head = PSRHead(
            in_channels=256,
            hidden_dim=128,
            num_components=num_components_psr,
            dropout=0.2,
        )

        # ---- Feature Bank (temporal, for activity head) ----
        from src.models.model import FeatureBank
        self.feature_bank = FeatureBank(embed_dim=512, window_size=16)

        logger.info(
            'VideoMultiTaskModel: backbone=%s, hidden=%d, clip_frames=%d, '
            'freeze=%s, checkpoint=%s',
            backbone_name, self.video_backbone.hidden_size, clip_frames,
            freeze_backbone, use_checkpoint,
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
            clip: [B, C, T, H, W] or [B, T, C, H, W] — video clip.
            video_ids: list of video IDs for Feature Bank (optional).
        Returns:
            dict matching POPWMultiTaskModel output interface for loss compatibility.
        """
        B = clip.shape[0]

        # ---- 1. Video backbone ----
        backbone_out = self.video_backbone(clip)
        clip_embed = backbone_out['clip_embed']           # [B, D]
        stage_feats = backbone_out['stage_features']      # {c3, c4, c5}
        c2 = backbone_out['c2']                            # [B, 96, H/4, W/4] or None
        hidden_size = backbone_out['hidden_size']

        # ---- 2. FPN ----
        c3, c4, c5 = stage_feats['c3'], stage_feats['c4'], stage_feats['c5']
        pyramid = self.fpn(c3, c4, c5)  # {p3: [B, 256, H3, W3], p4, p5, p6, p7}

        # ---- 3. Detection Head ----
        cls_preds, reg_preds = self.detection_head(pyramid)
        anchors = self.anchor_gen(pyramid)

        # ---- 4. Pose Head (on P3) ----
        heatmaps, keypoints, pose_confidence = self.pose_head(pyramid['p3'])

        # ---- 5. PoseFiLM (keypoint-conditioned modulation of C5) ----
        if self.use_hand_film and hasattr(self, 'pose_film'):
            c5_mod = self.pose_film(c5, keypoints.detach(), pose_confidence)
        else:
            c5_mod = c5

        # ---- 6. Head Pose ----
        if self.train_pose or not self.training:
            head_pose = self.head_pose_head(c4, c5)
            if isinstance(head_pose, tuple):
                _rot6d, _rot_mat, _pos = head_pose
                _forward = _rot_mat[:, :, 0]
                _up = _rot_mat[:, :, 2]
                head_pose = torch.cat([_forward, _pos, _up], dim=1)
            if self.use_headpose_film and hasattr(self, 'headpose_film'):
                c5_mod = self.headpose_film(c5_mod, head_pose.detach())
        else:
            head_pose = None

        # ---- 7. Activity Head ----
        # Build projected features from det_conf + c5_mod + p4 (same as POPW)
        det_conf = torch.sigmoid(cls_preds.max(dim=1)[0])
        activity_proj = torch.cat([
            det_conf,
            F.adaptive_avg_pool2d(c5_mod, 1).flatten(1),
            F.adaptive_avg_pool2d(pyramid['p4'].detach(), 1).flatten(1),
        ], dim=1)
        proj_feat = self.activity_head.proj_features(activity_proj)

        # Temporal bank (expand path for non-staged training)
        _staging_enabled = bool(getattr(C, 'STAGED_TRAINING', False))
        if _staging_enabled:
            bank_output = self.feature_bank(proj_feat, video_ids, None)
        else:
            bank_output = None

        # Populate videomae_feat from clip_embed if activity head expects it
        # (Use clip_embed as the "videomae" feature to match the interface)
        videomae_feat = clip_embed if hasattr(self.activity_head, 'use_videomae') and self.activity_head.use_videomae else None

        act_logits = self.activity_head(
            proj_feat=proj_feat,
            temporal_bank=bank_output,
            videomae_feat=videomae_feat,
        )

        # ---- 8. PSR Head ----
        psr_full = self.psr_head(pyramid)  # [B, 12]
        psr_logits = psr_full[..., :11]
        psr_confidence = psr_full[..., 11:]

        # ---- 9. Feature normalization (matching POPW output interface) ----
        # Normalize keypoints to [0, 1] for Wing loss compatibility
        if self.train_pose:
            _, _, grid_h, grid_w = heatmaps.shape
            kp_scale = torch.tensor([grid_w, grid_h], device=keypoints.device, dtype=keypoints.dtype)
            keypoints = keypoints / kp_scale.view(1, 1, 2)

        # ---- Output dict (matches POPWMultiTaskModel for loss compatibility) ----
        return {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'anchors': anchors,
            'heatmaps': heatmaps,
            'keypoints': keypoints,
            'pose_confidence': pose_confidence,
            'head_pose': head_pose,
            'c5_mod': c5_mod,
            'det_conf': det_conf,
            'act_logits': act_logits,
            'psr_logits': psr_logits,
            'psr_confidence': psr_confidence if not self.training else None,
            'clip_embed': clip_embed,  # NEW: full clip embedding
            'c5_raw': c5,
            'proj_feat': proj_feat,
            'p4': pyramid['p4'],
        }


# ===========================================================================
# Parameter counting (matching POPWMultiTaskModel.count_parameters)
# ===========================================================================
def count_parameters(model: VideoMultiTaskModel) -> Dict[str, int]:
    """Count trainable and total parameters per component."""
    components = {
        'video_backbone': [model.video_backbone],
        'fpn': [model.fpn],
        'detection': [model.detection_head],
        'pose_head': [model.pose_head],
    }
    if hasattr(model, 'pose_film'):
        components['pose_film'] = [model.pose_film]
    if hasattr(model, 'headpose_film'):
        components['headpose_film'] = [model.headpose_film]
    components['activity_head'] = [model.activity_head]
    components['psr_head'] = [model.psr_head]
    components['head_pose_head'] = [model.head_pose_head]
    components['feature_bank'] = [model.feature_bank]

    result = {}
    total = 0
    for name, modules in components.items():
        count = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
        result[name] = count
        total += count
    result['total_trainable'] = total
    result['total_all'] = sum(p.numel() for p in model.parameters())
    return result


# ===========================================================================
# Training Strategy Helpers
# ===========================================================================
def get_trainable_param_groups(
    model: VideoMultiTaskModel,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
) -> List[Dict]:
    """Return parameter groups with different LRs for backbone vs heads.

    The video backbone (Kinetics-pretrained) should be fine-tuned with a lower
    LR than the randomly initialized heads.  This matches the POPW staged
    training strategy but with backbone fine-tuning from the start.
    """
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'video_backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': head_params, 'lr': head_lr, 'name': 'heads'},
    ]


def unfreeze_backbone_stages(
    model: VideoMultiTaskModel,
    num_stages: int = 1,
) -> None:
    """Gradually unfreeze video backbone stages.

    MViTv2-S has 16 blocks grouped into 4 stages (4 blocks each).
    Stage 0: earliest (spatial pattern processing)
    Stage 3: latest (semantic / motion processing)

    Args:
        model: VideoMultiTaskModel instance.
        num_stages: number of stages from the end to unfreeze (1 = last stage only).
    """
    if not hasattr(model.video_backbone, 'encoder'):
        logger.warning('unfreeze_backbone_stages: no encoder attribute, skipping')
        return

    from torch import nn as _nn
    blocks = model.video_backbone.encoder.blocks
    total_blocks = len(blocks)  # 16 for MViTv2-S
    blocks_per_stage = total_blocks // 4  # 4
    stages_to_unfreeze = min(num_stages, 4)

    for stage_idx in range(4 - stages_to_unfreeze, 4):
        start = stage_idx * blocks_per_stage
        end = start + blocks_per_stage
        for i in range(start, end):
            for p in blocks[i].parameters():
                p.requires_grad = True
        logger.info('Unfroze backbone stage %d (blocks %d-%d)', stage_idx, start, end - 1)
