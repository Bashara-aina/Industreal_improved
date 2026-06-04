"""
POPW: Unified Multi-Task Architecture for Egocentric Assembly Understanding
Full implementation: ConvNeXt-Tiny + FPN + 5 task heads + two-stage FiLM conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from config import C


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt-Tiny backbone for feature extraction using timm."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            import timm
            self.model = timm.create_model(
                'convnext_tiny',
                pretrained=pretrained,
                num_classes=0,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            self.out_channels = (96, 192, 384, 768)
            self._use_timm = True
        except Exception:
            # Fallback to torchvision (no hierarchical features)
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = convnext_tiny(weights=weights)
            self.out_channels = (96, 192, 384, 768)
            self._use_timm = False
            # We'll handle this in forward

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns feature maps at multiple scales.
        C2: stride 4, 96 x H/4 x W/4
        C3: stride 8, 192 x H/8 x W/8
        C4: stride 16, 384 x H/16 x W/16
        C5: stride 32, 768 x H/32 x W/32
        """
        try:
            # timm with features_only
            features = self.model(x)
            # features = [C2, C3, C4, C5]
            return {
                'C2': features[0],
                'C3': features[1],
                'C4': features[2],
                'C5': features[3],
            }
        except (AttributeError, TypeError):
            # torchvision fallback
            x = self.model.features(x)
            x = self.model.norm(x)
            # No hierarchical output - return same feature at different "levels"
            return {
                'C2': x,
                'C3': x,
                'C4': x,
                'C5': x,
            }


class FPN(nn.Module):
    """Feature Pyramid Network with lateral 1x1 convolutions and top-down path."""

    def __init__(self, in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
                 out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1x1 convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])

        # Output 3x3 convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])

        # P6/P7 from C5
        self.p6_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: dict with C2, C3, C4, C5

        Returns:
            P3, P4, P5, P6, P7
        """
        laterals = [lat_conv(features[key]) for lat_conv, key in
                    zip(self.lateral_convs, ['C2', 'C3', 'C4', 'C5'])]

        # Top-down path
        P5 = laterals[3]
        P4 = laterals[2] + F.interpolate(P5, size=laterals[2].shape[2:], mode='bilinear', align_corners=False)
        P3 = laterals[1] + F.interpolate(P4, size=laterals[1].shape[2:], mode='bilinear', align_corners=False)
        P2 = laterals[0] + F.interpolate(P3, size=laterals[0].shape[2:], mode='bilinear', align_corners=False)

        # Apply output convolutions
        P2_out = self.output_convs[0](P2)
        P3_out = self.output_convs[1](P3)
        P4_out = self.output_convs[2](P4)
        P5_out = self.output_convs[3](P5)

        # P6/P7
        P6 = self.p6_conv(P5_out)
        P7 = self.p7_conv(F.relu(P6))

        return {'P2': P2_out, 'P3': P3_out, 'P4': P4_out, 'P5': P5_out, 'P6': P6, 'P7': P7}


class DetectionHead(nn.Module):
    """RetinaNet-style detection head for 24-class ASD."""

    def __init__(self, in_channels: int = 256, num_classes: int = 24,
                 num_anchors: int = 9, feat_channels: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification subnet
        cls_layers = []
        for _ in range(4):
            cls_layers.extend([
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
        cls_layers.append(nn.Conv2d(feat_channels, num_anchors * num_classes, 1))
        self.cls_subnet = nn.Sequential(*cls_layers)

        # Regression subnet
        reg_layers = []
        for _ in range(4):
            reg_layers.extend([
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
        reg_layers.append(nn.Conv2d(feat_channels, num_anchors * 4, 1))
        self.reg_subnet = nn.Sequential(*reg_layers)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Special initialization for classification
        prior = 0.01
        nn.init.constant_(self.cls_subnet[-1].bias, -math.log((1 - prior) / prior))

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: list of [B, C, H, W] feature maps

        Returns:
            cls_preds: list of [B, N_anchors * C] per feature level
            reg_preds: list of [B, N_anchors * 4] per feature level
        """
        cls_preds = []
        reg_preds = []

        for feat in features:
            cls_pred = self.cls_subnet(feat)
            reg_pred = self.reg_subnet(feat)

            B, _, H, W = cls_pred.shape
            cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W)
            cls_pred = cls_pred.permute(0, 1, 3, 4, 2).contiguous()
            cls_pred = cls_pred.view(B, -1, self.num_classes)

            reg_pred = reg_pred.view(B, self.num_anchors, 4, H, W)
            reg_pred = reg_pred.permute(0, 1, 3, 4, 2).contiguous()
            reg_pred = reg_pred.view(B, -1, 4)

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)

        return cls_preds, reg_preds


class BodyPoseHead(nn.Module):
    """Body keypoint pose estimation head (17 keypoints, for IKEA ASM)."""

    def __init__(self, in_channels: int = 768, num_joints: int = 17,
                 heatmap_size: Tuple[int, int] = (180, 320)):
        super().__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        # Upsampling + heatmap prediction
        self.upconv = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(32, 256)
        self.heatmap_pred = nn.Conv2d(256, num_joints, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 768, H/32, W/32]

        Returns:
            kpts: [B, 17, 2] keypoint coordinates
            conf: [B, 17] confidence scores
        """
        from losses import soft_argmax

        h = self.upconv(x)
        h = F.relu(self.norm(h))
        heatmaps = self.heatmap_pred(h)  # [B, 17, H/4, W/4]

        # Soft-argmax for differentiable keypoint extraction
        kpts, conf = soft_argmax(heatmaps, temperature=C.SOFTMAX_TEMP)

        return kpts, conf


class HeadPoseHead(nn.Module):
    """9-DoF head pose estimation head (for IndustReal)."""

    def __init__(self, c4_channels: int = 384, c5_channels: int = 768):
        super().__init__()
        total_channels = c4_channels + c5_channels  # 1152

        mlp_dims = [total_channels, 512, 256, 9]
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.LayerNorm(mlp_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.mlp = nn.Sequential(*layers)

    def forward(self, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c4: [B, 384, H/16, W/16]
            c5: [B, 768, H/32, W/32]

        Returns:
            head_pose: [B, 9] (forward[3], position[3], up[3])
        """
        # Global Average Pooling
        c4_pool = F.adaptive_avg_pool2d(c4, 1).flatten(1)  # [B, 384]
        c5_pool = F.adaptive_avg_pool2d(c5, 1).flatten(1)  # [B, 768]

        # Concatenate
        x = torch.cat([c4_pool, c5_pool], dim=1)  # [B, 1152]

        # MLP
        head_pose = self.mlp(x)

        return head_pose


class PoseFiLM(nn.Module):
    """First-stage FiLM: modulates C5 features using body keypoint confidence."""

    def __init__(self, pose_input_dim: int = 51, hidden_dim: int = 512, output_dim: int = 768):
        super().__init__()
        self.output_dim = output_dim

        self.gamma_net = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        self.beta_net = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, c5_direct: torch.Tensor, keypoints: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c5_direct: [B, 768, H, W]
            keypoints: [B, 34]
            confidence: [B, 17]

        Returns:
            c5_mod: [B, 768, H, W]
        """
        pose_input = torch.cat([keypoints, confidence], dim=1)  # [B, 51]

        gamma = self.gamma_net(pose_input)
        beta = self.beta_net(pose_input)

        gamma = 1 + torch.tanh(gamma)
        gamma = gamma.view(-1, self.output_dim, 1, 1)
        beta = beta.view(-1, self.output_dim, 1, 1)

        c5_mod = gamma * c5_direct + beta

        return c5_mod


class HeadPoseFiLM(nn.Module):
    """Second-stage FiLM: modulates features using 9-DoF head pose."""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 256, output_dim: int = 768):
        super().__init__()
        self.output_dim = output_dim

        self.gamma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, c5_mod: torch.Tensor, head_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c5_mod: [B, 768, H, W]
            head_pose: [B, 9]

        Returns:
            c5_mod2: [B, 768, H, W]
        """
        gamma = self.gamma_net(head_pose)
        beta = self.beta_net(head_pose)

        gamma = 1 + torch.tanh(gamma)
        gamma = gamma.view(-1, self.output_dim, 1, 1)
        beta = beta.view(-1, self.output_dim, 1, 1)

        c5_mod2 = gamma * c5_mod + beta

        return c5_mod2


class FeatureBank:
    """Sliding window feature bank for temporal activity recognition."""

    def __init__(self, max_len: int = 16, dim: int = 512):
        self.max_len = max_len
        self.dim = dim
        self.bank = {}

    def get_sequence(self, video_id: str, camera_view: str = "front") -> torch.Tensor:
        key = (video_id, camera_view)
        if key not in self.bank or len(self.bank[key]) == 0:
            return torch.zeros(1, self.max_len, self.dim)
        bank = self.bank[key]
        seq_len = min(len(bank), self.max_len)
        features = bank[-seq_len:]
        while len(features) < self.max_len:
            features.insert(0, torch.zeros_like(features[0]))
        return torch.stack(features, dim=0).unsqueeze(0)

    def update(self, video_id: str, camera_view: str, feature: torch.Tensor):
        key = (video_id, camera_view)
        if key not in self.bank:
            self.bank[key] = []
        bank = self.bank[key]
        bank.append(feature.detach().cpu())
        if len(bank) > self.max_len:
            bank.pop(0)

    def reset(self, video_id: str, camera_view: str = "front"):
        key = (video_id, camera_view)
        if key in self.bank:
            del self.bank[key]

    def clear(self):
        self.bank.clear()


class ActivityHead(nn.Module):
    """Activity recognition head with TCN + ViT temporal modeling."""

    def __init__(self, det_context_dim: int = 24, spatial_dim: int = 1024,
                 projection_dim: int = 512, num_classes: int = 74,
                 feature_bank_len: int = 16, use_videomae: bool = False):
        super().__init__()
        self.projection_dim = projection_dim
        self.use_videomae = use_videomae

        self.det_proj = nn.Linear(det_context_dim, 64)
        self.spatial_proj = nn.Linear(spatial_dim, projection_dim)

        joint_dim = 64 + projection_dim
        if use_videomae:
            joint_dim += 384

        self.joint_proj = nn.Sequential(
            nn.Linear(joint_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.tcn = nn.Sequential(
            nn.Conv1d(projection_dim, projection_dim, kernel_size=5,
                     stride=1, padding=2, groups=projection_dim),
            nn.GroupNorm(1, projection_dim),
            nn.GELU(),
        )
        self.tcn_fc = nn.Linear(projection_dim, projection_dim)

        self.vit_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                droppath=0.1 if i == 0 else 0.15
            )
            for i in range(2)
        ])

        self.cls_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(projection_dim, num_classes)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, projection_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, feature_bank_len + 1, projection_dim))

    def forward(self, det_context: torch.Tensor, c5_mod2_gap: torch.Tensor,
                p4_gap: torch.Tensor, temporal_seq: Optional[torch.Tensor] = None,
                videomae_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            det_context: [B, 24]
            c5_mod2_gap: [B, 768]
            p4_gap: [B, 256]
            temporal_seq: [B, T, D] optional
            videomae_feat: [B, 384] optional

        Returns:
            act_logits: [B, 74]
        """
        B = det_context.size(0)

        det_feat = self.det_proj(det_context)
        spatial_feat = torch.cat([c5_mod2_gap, p4_gap], dim=-1)
        spatial_feat = self.spatial_proj(spatial_feat)

        if self.use_videomae and videomae_feat is not None:
            joint_feat = torch.cat([det_feat, spatial_feat, videomae_feat], dim=-1)
        else:
            joint_feat = torch.cat([det_feat, spatial_feat], dim=-1)

        proj_feat = self.joint_proj(joint_feat)

        if temporal_seq is not None and temporal_seq.numel() > 0:
            # temporal_seq is [1, T, 1, D] from FeatureBank with B=1 actual video
            # Collapse to [1, T, D]
            if temporal_seq.dim() == 4:
                temporal_seq = temporal_seq.squeeze(2)  # [1, T, D]
            T = temporal_seq.size(1)
            B = det_context.size(0)  # actual batch size
            cls_tokens = self.cls_token.expand(B, 1, -1)  # [B, 1, D]

            tcn_input = temporal_seq.permute(0, 2, 1)  # [1, D, T]
            tcn_out = self.tcn(tcn_input)  # [1, D', T]
            B2, D2, T2 = tcn_out.shape
            # Reshape: keep batch, apply FC to features
            tcn_out = tcn_out.permute(0, 2, 1).reshape(B2 * T2, D2)  # [T, D']
            tcn_out = self.tcn_fc(tcn_out)  # [T, D'']
            tcn_out = tcn_out.reshape(B2, T2, -1)  # [1, T, D'']

            tcn_out = tcn_out + temporal_seq
            tcn_out = tcn_out + self.pos_embed[:, :T]

            # Broadcast temporal_seq to match B
            seq_final = torch.cat([cls_tokens, tcn_out.expand(B, -1, -1)], dim=1)

            for layer in self.vit_layers:
                seq_final = layer(seq_final)

            cls_out = seq_final[:, 0]
        else:
            cls_out = proj_feat

        logits = self.cls_head(cls_out)
        return logits


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm and droppath."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1, droppath: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.droppath = DropPath(droppath) if droppath > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.droppath(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.droppath(self.ffn(self.norm2(x)))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PSRHead(nn.Module):
    """Procedure Step Recognition head with causal transformer."""

    def __init__(self, input_dim: int = 768, mlp_dims: List[int] = [768, 256, 128],
                 transformer_layers: int = 3, transformer_heads: int = 4,
                 d_model: int = 128, num_components: int = 11, max_cache_len: int = 32):
        super().__init__()
        self.num_components = num_components
        self.max_cache_len = max_cache_len

        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.LayerNorm(mlp_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.input_proj = nn.Sequential(*layers)

        self.pos_encoding = nn.Parameter(torch.randn(1, max_cache_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=transformer_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.component_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(num_components)
        ])

        self.cache = {}
        self.cache_len = {}

    def forward(self, p3_gap: torch.Tensor, p4_gap: torch.Tensor, p5_gap: torch.Tensor,
                video_id: str = "", cache: bool = False) -> torch.Tensor:
        """
        Args:
            p3_gap: [B, 256]
            p4_gap: [B, 256]
            p5_gap: [B, 256]
            video_id: for caching
            cache: use cache

        Returns:
            psr_logits: [B, 11]
        """
        x = torch.cat([p3_gap, p4_gap, p5_gap], dim=-1)
        x = self.input_proj(x)

        if cache and video_id:
            if video_id not in self.cache:
                self.cache[video_id] = []
                self.cache_len[video_id] = 0

            self.cache[video_id].append(x)
            self.cache_len[video_id] += 1

            if len(self.cache[video_id]) > self.max_cache_len:
                self.cache[video_id].pop(0)

            seq = torch.stack(self.cache[video_id], dim=0).transpose(0, 1)
            seq_len = seq.size(1)
            seq = seq + self.pos_encoding[:, :seq_len]

            out = self.transformer(seq)
            last_out = out[:, -1, :]
        else:
            last_out = x

        logits = torch.cat([head(last_out) for head in self.component_heads], dim=0)
        logits = logits.t()

        return logits

    def reset_sequence(self, video_id: str):
        if video_id in self.cache:
            del self.cache[video_id]
        if video_id in self.cache_len:
            del self.cache_len[video_id]

    def clear_cache(self):
        self.cache.clear()
        self.cache_len.clear()


class VideoMAEStream(nn.Module):
    """VideoMAE-small stream for activity recognition.

    Loads MCG-NJU/videomae-small-finetuned-kinetics from HuggingFace.
    Takes a batch of video clips (16 frames each, 224x224),
    returns [CLS] token features (384-dim).
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        try:
            from transformers import VideoMAEForVideoClassification, VideoMAEConfig
            self.model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-small-finetuned-kinetics"
            )
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.available = True
            self.hidden_dim = 384  # VideoMAE-small cls dim
        except Exception as e:
            print(f"VideoMAE not available: {e}. Using dummy.")
            self.model = None
            self.available = False
            self.hidden_dim = 384

    def forward(self, clip_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_rgb: [B, T, C, H, W] where T=16, H=224, W=224
                     If None or empty, returns zeros.
        Returns:
            cls_features: [B, 384]
        """
        if clip_rgb is None or not self.available:
            B = clip_rgb.size(0) if clip_rgb is not None else 1
            device = clip_rgb.device if clip_rgb is not None else 'cpu'
            return torch.zeros(B, self.hidden_dim, device=device)

        # clip_rgb is [B, T, C, H, W] from dataset
        # VideoMAE expects [B, C, T, H, W]
        x = clip_rgb.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

        with torch.no_grad():
            outputs = self.model(x)
            # If model has hidden_states, get cls token; else use logits pool
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                cls_features = outputs.hidden_states[-1][:, 0]  # [B, 384]
            else:
                # Pool the logits as fallback
                cls_features = outputs.logits.mean(dim=1)  # [B, num_classes]
                # Project to 384 if needed
                if cls_features.size(-1) != 384:
                    proj = nn.Linear(cls_features.size(-1), 384, device=cls_features.device)
                    cls_features = proj(cls_features)

        return cls_features


class POPWModel(nn.Module):
    """Unified POPW architecture for multi-task egocentric assembly understanding."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or C

        self.backbone = ConvNeXtBackbone(pretrained=self.config.BACKBONEpretrained)

        self.fpn = FPN(
            in_channels=self.config.BACKBONE_channels,
            out_channels=self.config.FPN_CHANNELS
        )

        self.detection_head = DetectionHead(
            in_channels=self.config.FPN_CHANNELS,
            num_classes=self.config.NUM_CLASSES_DET,
            num_anchors=len(self.config.ANCHOR_RATIOS) * len(self.config.ANCHOR_SCALES)
        )

        self.body_pose_head = None
        if self.config.TRAIN_BODY_POSE:
            self.body_pose_head = BodyPoseHead(
                in_channels=self.config.BACKBONE_channels[3],
                num_joints=self.config.NUM_JOINTS,
                heatmap_size=self.config.HEATMAP_SIZE
            )

        self.head_pose_head = HeadPoseHead(
            c4_channels=self.config.BACKBONE_channels[2],
            c5_channels=self.config.BACKBONE_channels[3]
        )

        self.pose_film = PoseFiLM(
            pose_input_dim=self.config.POSE_FILM_INPUT_DIM,
            hidden_dim=self.config.POSE_FILM_HIDDEN_DIM,
            output_dim=self.config.POSE_FILM_OUTPUT_DIM
        )

        self.headpose_film = HeadPoseFiLM(
            input_dim=self.config.HEAD_POSE_FILM_INPUT_DIM,
            hidden_dim=self.config.HEAD_POSE_FILM_HIDDEN_DIM,
            output_dim=self.config.HEAD_POSE_FILM_OUTPUT_DIM
        )

        spatial_dim = self.config.BACKBONE_channels[3] + self.config.FPN_CHANNELS
        self.activity_head = ActivityHead(
            det_context_dim=self.config.ACT_DET_CONTEXT_DIM,
            spatial_dim=spatial_dim,
            projection_dim=self.config.ACT_PROJECTION_DIM,
            num_classes=self.config.NUM_CLASSES_ACT,
            feature_bank_len=self.config.ACT_FEATURE_BANK_LEN,
            use_videomae=self.config.USE_VIDEOMAE
        )

        self.psr_head = PSRHead(
            input_dim=self.config.PSR_INPUT_DIM,
            mlp_dims=self.config.PSR_MLP_DIMS,
            transformer_layers=self.config.PSR_TRANSFORMER_LAYERS,
            transformer_heads=self.config.PSR_TRANSFORMER_HEADS,
            d_model=self.config.PSR_TRANSFORMER_DMODEL,
            num_components=self.config.PSR_NUM_COMPONENTS,
            max_cache_len=self.config.PSR_MAX_CACHE_LEN
        )

        if self.config.USE_VIDEOMAE:
            self.videomae_stream = VideoMAEStream(freeze=True)
        else:
            self.videomae_stream = None

        self.feature_bank = FeatureBank(
            max_len=self.config.ACT_FEATURE_BANK_LEN,
            dim=self.config.ACT_PROJECTION_DIM
        )

    def forward(self, images: torch.Tensor, video_id: str = "",
                return_intermediate: bool = False, **kwargs):
        """Full forward pass."""
        # Get clip_rgb from kwargs if available
        clip_rgb = kwargs.get('clip_rgb', None)
        B, _, H, W = images.shape

        features = self.backbone(images)
        fpn_features = self.fpn(features)
        p3, p4, p5 = fpn_features['P3'], fpn_features['P4'], fpn_features['P5']

        det_features = [fpn_features['P3'], fpn_features['P4'], fpn_features['P5'],
                        fpn_features['P6'], fpn_features['P7']]
        cls_preds, reg_preds = self.detection_head(det_features)

        cls_pred = torch.cat([p.view(B, -1, self.config.NUM_CLASSES_DET)
                              for p in cls_preds], dim=1)
        det_context = cls_pred.max(dim=1)[0]
        det_context = det_context.detach()

        body_kpts = None
        body_conf = None
        if self.body_pose_head is not None:
            body_kpts, body_conf = self.body_pose_head(features['C5'])

        head_pose = self.head_pose_head(features['C4'], features['C5'])
        head_pose_detach = head_pose.detach()

        c5_direct = features['C5']

        if body_kpts is not None and body_conf is not None:
            c5_mod = self.pose_film(c5_direct, body_kpts.view(B, -1), body_conf)
        else:
            c5_mod = c5_direct

        c5_mod2 = self.headpose_film(c5_mod, head_pose_detach)

        c5_mod2_gap = F.adaptive_avg_pool2d(c5_mod2, 1).flatten(1)
        p4_gap = F.adaptive_avg_pool2d(p4, 1).flatten(1)

        temporal_seq = self.feature_bank.get_sequence(video_id, "front")

        # Handle videomae_feat BEFORE feature_bank.update and activity_head call
        videomae_feat = None
        if self.config.USE_VIDEOMAE:
            if self.videomae_stream is not None and self.videomae_stream.available:
                # clip_rgb is [B, T, C, H, W] - need to sample 16 frames from T
                clip = clip_rgb
                if clip is not None and clip.numel() > 0:
                    # Sample 16 frames uniformly from clip
                    T = clip.size(1)
                    indices = torch.linspace(0, T-1, steps=16, device=clip.device).long()
                    clip_16 = clip[:, indices]  # [B, 16, C, H, W]
                    # Resize to 224x224 if needed
                    if clip_16.size(-1) != 224 or clip_16.size(-2) != 224:
                        clip_16 = F.interpolate(clip_16.flatten(0,1), size=(224, 224), mode='bilinear', align_corners=False)
                        clip_16 = clip_16.unflatten(0, (clip.size(0), 16))
                    videomae_feat = self.videomae_stream(clip_16)
            # Fallback to zeros if VideoMAE unavailable or clip not provided
            if videomae_feat is None:
                videomae_feat = torch.zeros(B, 384, device=images.device)

        # Build activity features - include videomae_feat only if USE_VIDEOMAE
        if self.config.USE_VIDEOMAE and videomae_feat is not None:
            activity_proj_feat = torch.cat([
                self.activity_head.det_proj(det_context),
                self.activity_head.spatial_proj(torch.cat([c5_mod2_gap, p4_gap], dim=-1)),
                videomae_feat
            ], dim=-1)
        else:
            activity_proj_feat = torch.cat([
                self.activity_head.det_proj(det_context),
                self.activity_head.spatial_proj(torch.cat([c5_mod2_gap, p4_gap], dim=-1))
            ], dim=-1)

        self.feature_bank.update(video_id, "front",
                                  self.activity_head.joint_proj(activity_proj_feat))

        act_logits = self.activity_head(
            det_context, c5_mod2_gap, p4_gap,
            temporal_seq if temporal_seq.numel() > 0 else None,
            videomae_feat
        )

        p3_gap = F.adaptive_avg_pool2d(p3, 1).flatten(1)
        p4_gap_psr = F.adaptive_avg_pool2d(p4, 1).flatten(1)
        p5_gap = F.adaptive_avg_pool2d(p5, 1).flatten(1)

        psr_logits = self.psr_head(p3_gap, p4_gap_psr, p5_gap, video_id=video_id, cache=True)

        outputs = {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'head_pose': head_pose,
            'act_logits': act_logits,
            'psr_logits': psr_logits,
            'det_context': det_context,
        }

        if body_kpts is not None:
            outputs['body_kpts'] = body_kpts
            outputs['body_conf'] = body_conf

        if return_intermediate:
            outputs['c5_mod2'] = c5_mod2
            outputs['fpn_features'] = fpn_features

        return outputs

    def reset_psr_cache(self, video_id: str):
        self.psr_head.reset_sequence(video_id)


def build_model(config=None) -> POPWModel:
    return POPWModel(config)


if __name__ == "__main__":
    model = POPWModel()
    print(f"POPW model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    x = torch.randn(1, 3, 720, 1280)
    outputs = model(x, video_id="test")

    print(f"Head pose: {outputs['head_pose'].shape}")
    print(f"Activity: {outputs['act_logits'].shape}")
    print(f"PSR: {outputs['psr_logits'].shape}")
    print(f"Detection levels: {len(outputs['cls_preds'])}")