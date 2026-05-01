import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
POPW: Pose-Conditioned Multi-Task Architecture for IKEA/IndustReal Recognition
================================================================================
Matches the XML diagram architecture EXACTLY (as implemented in industriali).

BACKBONE (ConvNeXt-Tiny via torchvision):
  C2 (stride 4, 96ch) → C3 (stride 8, 192ch) → C4 (stride 16, 384ch) → C5 (stride 32, 768ch)
  C5 goes DIRECTLY to PoseFiLM (bypasses FPN)

FPN NECK:
  P3, P4, P5 (lateral 1x1 + top-down upsample + 3x3 smooth) + P6/P7 via stride-2 conv on C5

DETECTION HEAD (RetinaNet-style, P3-P7):
  Cls subnet: 4× Conv3x3+ReLU → Conv(9×num_classes)
  Reg subnet: 4× Conv3x3+ReLU → Conv(9×4)
  Anchors: 3 ratios × 3 scales = 9 per location

POSE HEAD:
  ConvTranspose2d(k=4,s=2,p=1) + GroupNorm(32) + ReLU → P3 resolution
  Conv1x1 → heatmaps [B, NUM_KEYPOINTS, H, W]
  Soft-argmax → keypoints [B, NUM_KEYPOINTS, 2] + confidence [B, NUM_KEYPOINTS]

POSEFILM MODULE:
  keypoints [B,NUM_KEYPOINTS,2] ‖ confidence [B,NUM_KEYPOINTS] → pose_flat [B, 3*NUM_KEYPOINTS]
  γ-net: (3*NUM_KEYPOINTS)→512→C5, 1+tanh ∈ (0,2)
  β-net: (3*NUM_KEYPOINTS)→512→C5, linear (unbounded)
  C5_mod = γ · C5 + β   [B, C5, H/32, W/32]

ACTIVITY HEAD (Feature Bank + TCN + 2×ViT + CLS):
  det_conf = MaxPool(cls_preds) → [B, NUM_DET_CLASSES]  (stop_grad)
  f_joint = [det_conf ‖ GAP(C5_mod)(C5) ‖ GAP(P4)(256)] → [B, NUM_DET_CLASSES+C5+256]
  W_proj: Linear(...) → f̃_t [B, 512]
  Feature Bank B_t = [f̃_{t-T+1}, ..., f̃_t]  T=16, [B, T, 512]
  TCN (depthwise 1D conv, kernel=5) → captures velocity/acceleration
  2× ViT blocks (8 heads, d_k=64) with CLS token
  CLS pooled output → Dropout(0.1) → act_logits [B, NUM_CLASSES_ACT]

LOSSES (from losses.py):
  L_det = Focal Loss (α=0.25, γ=2) + SmoothL1
  L_pose = Wing Loss (ω=0.05, ε=0.005)
  L_act = CB-Focal Loss (β=0.999, γ=2.0, NUM_CLASSES_ACT cls, label_smoothing=0.1)
  L_total = Kendall(s_det,s_pose,s_act) with act_ramp = min(1, epoch/5)
  init: s_det=0, s_pose=-1, s_act=0

Architecture follows industriali_improved exactly.
Only dataset-specific adjustments: class counts, paths, dataset class.

Author: Bashara (adapted from industriali_improved for IKEA ASM)
Date: April 2026
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ===========================================================================
# Soft-Argmax — differentiable keypoint extraction from heatmaps
# ===========================================================================
class SoftArgmax(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, H, W = heatmaps.shape
        flat = heatmaps.view(B, K, H * W)
        weights = F.softmax(flat / self.temperature, dim=-1)

        grid_x_2d = torch.arange(W, device=heatmaps.device, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        grid_y_2d = torch.arange(H, device=heatmaps.device, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        grid_x_2d = grid_x_2d.unsqueeze(0)
        grid_y_2d = grid_y_2d.unsqueeze(0)

        coords_x = (weights.view(B, K, H, W) * grid_x_2d).sum(dim=[-2, -1], keepdim=True)
        coords_y = (weights.view(B, K, H, W) * grid_y_2d).sum(dim=[-2, -1], keepdim=True)
        coords = torch.cat([coords_x, coords_y], dim=-1).squeeze(-2)

        conf = heatmaps.max(dim=-1)[0].max(dim=-1)[0]
        conf = torch.sigmoid(conf)

        return coords, conf


# ===========================================================================
# Wing Loss — robust regression loss for keypoint prediction
# ===========================================================================
class WingLoss(nn.Module):
    def __init__(self, omega: float = 0.05, epsilon: float = 0.005):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * math.log(1 + omega / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C,
        )
        if weight is not None:
            loss = loss * weight.unsqueeze(-1)
        return loss.mean()


# ===========================================================================
# ConvNeXtBackbone — torchvision ConvNeXt-Tiny (matches industriali exactly)
# ===========================================================================
class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt-Tiny pretrained backbone (ImageNet) via torchvision.
    Matches industriali_improved exactly — uses model.features stages.

    Channel outputs:
      C2: 96ch   (after first stage / stem)
      C3: 192ch  (after second stage)
      C4: 384ch  (after third stage)
      C5: 768ch  (after fourth stage)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            c2: [B, 96, H/4, W/4]
            c3: [B, 192, H/8, W/8]
            c4: [B, 384, H/16, W/16]
            c5: [B, 768, H/32, W/32]
        """
        stages = self.model.features

        x = stages[0](x)
        c2 = x

        x = stages[1](x)
        x = stages[2](x)
        c3 = x

        x = stages[3](x)
        x = stages[4](x)
        c4 = x

        x = stages[5](x)
        x = stages[6](x)
        c5 = x

        return c2, c3, c4, c5


def build_backbone(backbone_type: str = 'convnext_tiny', pretrained: bool = True) -> nn.Module:
    if backbone_type == 'convnext_tiny':
        return ConvNeXtBackbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Use 'convnext_tiny'.")


# ===========================================================================
# FPN — Feature Pyramid Network
# ===========================================================================
class FPN(nn.Module):
    def __init__(self, in_channels: List[int] = [192, 384, 768], out_channels: int = 256):
        super().__init__()
        c3_ch, c4_ch, c5_ch = in_channels

        self.lateral_c3 = nn.Conv2d(c3_ch, out_channels, 1)
        self.lateral_c4 = nn.Conv2d(c4_ch, out_channels, 1)
        self.lateral_c5 = nn.Conv2d(c5_ch, out_channels, 1)

        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.p6_conv = nn.Conv2d(c5_ch, out_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c3, c4, c5) -> Dict[str, torch.Tensor]:
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
# Anchor Generator (RetinaNet-style)
# ===========================================================================
class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes: Tuple[int, ...] = C.ANCHOR_SIZES,
        ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        scales: Tuple[float, ...] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
    ):
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(ratios) * len(scales)

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(iter(feature_maps.values())).device
        all_anchors = []
        fpn_keys = ['p3', 'p4', 'p5', 'p6', 'p7']
        strides = [8, 16, 32, 64, 128]

        for level_idx, (key, stride) in enumerate(zip(fpn_keys, strides)):
            feat = feature_maps[key]
            h, w = feat.shape[2:]
            base_size = self.sizes[level_idx]

            cell_anchors = []
            for ratio in self.ratios:
                for scale in self.scales:
                    s = base_size * scale
                    aw = s * math.sqrt(ratio)
                    ah = s / math.sqrt(ratio)
                    cell_anchors.append([-aw / 2, -ah / 2, aw / 2, ah / 2])
            cell_anchors = torch.tensor(cell_anchors, device=device, dtype=torch.float32)

            shifts_x = (torch.arange(w, device=device) + 0.5) * stride
            shifts_y = (torch.arange(h, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shifts = torch.stack([
                shift_x.flatten(), shift_y.flatten(),
                shift_x.flatten(), shift_y.flatten(),
            ], dim=1)

            anchors = shifts.unsqueeze(1) + cell_anchors.unsqueeze(0)
            all_anchors.append(anchors.reshape(-1, 4))

        return torch.cat(all_anchors, dim=0)


# ===========================================================================
# Detection Head (RetinaNet-style)
# ===========================================================================
class DetectionHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 7, num_anchors: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def make_subnet():
            layers = []
            for _ in range(4):
                layers.extend([
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.ReLU(True),
                ])
            return nn.Sequential(*layers)

        self.cls_subnet = make_subnet()
        self.reg_subnet = make_subnet()

        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_subnet, self.reg_subnet]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        pi = 0.01
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, -math.log((1 - pi) / pi))
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)

    def forward(self, features: Dict[str, torch.Tensor]):
        cls_all, reg_all = [], []
        for key in ('p3', 'p4', 'p5', 'p6', 'p7'):
            feat = features[key]
            B, _, H, W = feat.shape

            cls_out = self.cls_score(self.cls_subnet(feat))
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, self.num_classes)
            cls_all.append(cls_out)

            reg_out = self.reg_pred(self.reg_subnet(feat))
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, 4)
            reg_all.append(reg_out)

        return torch.cat(cls_all, dim=1), torch.cat(reg_all, dim=1)


# ===========================================================================
# Pose Head — heatmaps → soft-argmax → keypoints + confidence
# ===========================================================================
class PoseHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_keypoints: int = 17,
                 temperature: float = 0.1):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(True),
        )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_keypoints, 1),
        )

        self.soft_argmax = SoftArgmax(temperature=temperature)

    def forward(self, p3_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.upsample(p3_feature)
        heatmaps = self.heatmap_head(x)
        keypoints, confidence = self.soft_argmax(heatmaps)
        return heatmaps, keypoints, confidence


# ===========================================================================
# PoseFiLM Module — keypoint-conditioned FiLM modulation on C5
# ===========================================================================
class PoseFiLMModule(nn.Module):
    def __init__(self, num_keypoints: int = 17, c5_channels: int = 768,
                 hidden_channels: int = 512):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.c5_channels = c5_channels

        pose_input_dim = 2 * num_keypoints + num_keypoints  # 3 * num_keypoints = 51 for 17 keypoints

        self.gamma_net = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, c5_channels),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, c5_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for net in [self.gamma_net, self.beta_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
        nn.init.ones_(self.gamma_net[-1].bias)

    def forward(self, c5: torch.Tensor,
                keypoints: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        B = keypoints.shape[0]

        scale = torch.tensor([C.IMG_WIDTH, C.IMG_HEIGHT],
                             device=keypoints.device, dtype=keypoints.dtype)
        keypoints_norm = keypoints / scale.view(1, 1, 2)

        kp_flat = keypoints_norm.flatten(1)
        conf_flat = confidence

        pose_flat = torch.cat([kp_flat, conf_flat], dim=1)

        gamma_raw = self.gamma_net(pose_flat)
        beta_raw = self.beta_net(pose_flat)

        gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(-1).unsqueeze(-1)
        beta = beta_raw.unsqueeze(-1).unsqueeze(-1)

        return gamma * c5 + beta


# ===========================================================================
# TemporalConvBlock — 1D conv for short-range motion (matches industriali)
# ===========================================================================
class TemporalConvBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, kernel_size: int = 5,
                 dropout: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1(x_conv)
        x_conv = self.gelu(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = self.conv2(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.dropout(x_conv)
        return residual + x_conv


# ===========================================================================
# ViT Temporal Block — MHSA + FFN with learnable pos embed (matches industriali)
# ===========================================================================
class ViTTemporalBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1, drop_path: float = 0.1,
                 max_seq_len: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        x = x + self.pos_embed[:, :T, :]

        x_normed = self.norm1(x)

        q = self.q_proj(x_normed).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, T, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        x = x + out
        x = x + self.ffn(x)

        return x


# ===========================================================================
# Feature Bank — sliding window temporal memory
# ===========================================================================
class FeatureBank(nn.Module):
    def __init__(self, embed_dim: int = 512, window_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self._bank: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def forward(self,
                projected_features: torch.Tensor,
                video_ids: Optional[List[str]] = None,
                camera_views: Optional[List[str]] = None) -> torch.Tensor:
        if projected_features.dim() == 3:
            return projected_features

        B = projected_features.shape[0]

        if video_ids is None:
            return projected_features.unsqueeze(1).expand(-1, self.window_size, -1)

        outputs = []
        for i in range(B):
            vid = str(video_ids[i]) if video_ids[i] is not None else 'default'
            cam = str(camera_views[i]) if (camera_views is not None and camera_views[i] is not None) else 'default'
            key = (vid, cam)

            feat_i = projected_features[i]

            if key not in self._bank:
                self._bank[key] = []

            self._bank[key].append(feat_i.detach().clone())

            if len(self._bank[key]) > self.window_size:
                self._bank[key].pop(0)

            seq = self._bank[key]
            while len(seq) < self.window_size:
                seq = [feat_i.detach().clone()] + seq
            seq = seq[-self.window_size:]

            bank_i = torch.stack(seq)
            outputs.append(bank_i)

        return torch.stack(outputs)

    def reset(self):
        self._bank.clear()

    def reset_sequence(self, video_id: str, camera_view: str = 'default'):
        key = (str(video_id), str(camera_view))
        self._bank.pop(key, None)


# ===========================================================================
# Activity Head — Feature Bank + TCN + 2×ViT + CLS classifier
# Matches industriali_improved exactly.
# ===========================================================================
class ActivityHead(nn.Module):
    def __init__(self, c5_channels: int = 768, p4_channels: int = 256,
                 det_conf_size: int = 7, embed_dim: int = 512,
                 num_classes: int = 33, dropout: float = 0.3,
                 window_size: int = 16, use_vit: bool = True,
                 vit_drop_path: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_vit = use_vit
        self.window_size = window_size

        self.gap_c5 = nn.AdaptiveAvgPool2d(1)
        self.gap_p4 = nn.AdaptiveAvgPool2d(1)

        proj_input_dim = det_conf_size + c5_channels + p4_channels

        self.proj_features = nn.Linear(proj_input_dim, embed_dim)

        self.tcn = TemporalConvBlock(
            embed_dim=embed_dim,
            kernel_size=5,
            dropout=0.1,
            drop_path=0.1,
        )

        self.vit = nn.ModuleList([
            ViTTemporalBlock(
                embed_dim=embed_dim,
                num_heads=8,
                ff_dim=2048,
                dropout=dropout,
                drop_path=0.1,
            ),
            ViTTemporalBlock(
                embed_dim=embed_dim,
                num_heads=8,
                ff_dim=2048,
                dropout=dropout,
                drop_path=0.15,
            ),
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.activity_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, proj_feat: torch.Tensor,
                temporal_bank: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = proj_feat.shape[0]

        if temporal_bank is not None:
            bank_seq = temporal_bank.clone()
            bank_seq[:, -1, :] = proj_feat
        elif self.use_vit:
            bank_seq = proj_feat.unsqueeze(1).expand(-1, self.window_size, -1)
        else:
            bank_seq = None

        if bank_seq is not None:
            bank_seq = self.tcn(bank_seq)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            bank_seq = torch.cat([cls_tokens, bank_seq], dim=1)

            for vit_block in self.vit:
                bank_seq = vit_block(bank_seq)

            cls_out = bank_seq[:, 0, :]
            feat = cls_out
        else:
            feat = proj_feat

        feat = F.dropout(feat, p=0.1, training=self.training)
        logits = self.activity_classifier(feat)
        return logits


# ===========================================================================
# Full Multi-Task POPW Model for IKEA ASM
# ===========================================================================
class POPWMultiTaskModel(nn.Module):
    """
    POPW: Pose-Conditioned Multi-Task Architecture for IKEA Assembly.

    Backbone: ConvNeXt-Tiny (C3=192, C4=384, C5=768)

    Neck: FPN (P3-P7, 256ch)
    Heads:
      - Detection: RetinaNet-style (NUM_DET_CLASSES, 9 anchors per location)
      - Pose: ConvTranspose2d + GroupNorm + ReLU → heatmaps + soft-argmax → keypoints + confidence
      - PoseFiLM: keypoints + confidence → gamma/beta on C5
      - Activity: det_conf + GAP(C5_mod) + GAP(P4) → concat → proj → FeatureBank → TCN → 2×ViT → CLS → FC(NUM_CLASSES_ACT)
    """
    def __init__(
        self,
        pretrained: bool = True,
        backbone_type: str = 'convnext_tiny',
        use_film: bool = True,
        use_temporal: bool = False,
        use_multiview: bool = False,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.use_film = use_film
        self.use_temporal = use_temporal
        self.use_multiview = use_multiview

        self.backbone = build_backbone(backbone_type, pretrained=pretrained)

        c2_ch, c3_ch, c4_ch, c5_ch = 96, 192, 384, 768
        fpn_in_channels = [c3_ch, c4_ch, c5_ch]

        self.c2_channels = c2_ch
        self.c3_channels = c3_ch
        self.c4_channels = c4_ch
        self.c5_channels = c5_ch

        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=256)

        self.detection_head = DetectionHead(in_channels=256, num_classes=C.NUM_DET_CLASSES)
        self.anchor_gen = AnchorGenerator()

        self.pose_head = PoseHead(in_channels=256, num_keypoints=C.NUM_KEYPOINTS)

        self.pose_film = PoseFiLMModule(
            num_keypoints=C.NUM_KEYPOINTS,
            c5_channels=c5_ch,
            hidden_channels=512,
        )

        self.activity_head = ActivityHead(
            c5_channels=c5_ch,
            p4_channels=256,
            det_conf_size=C.NUM_DET_CLASSES,
            embed_dim=512,
            num_classes=C.NUM_CLASSES_ACT,
            dropout=0.3,
            window_size=16,
            use_vit=True,
        )

        self.feature_bank = FeatureBank(embed_dim=512, window_size=16)

    @staticmethod
    def _decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        a_w = anchors[:, 2] - anchors[:, 0]
        a_h = anchors[:, 3] - anchors[:, 1]

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2].clamp(-4, 4)
        dh = deltas[:, 3].clamp(-4, 4)

        pred_w = torch.exp(dw) * a_w
        pred_h = torch.exp(dh) * a_h
        pred_cx = dx * a_w + a_cx
        pred_cy = dy * a_h + a_cy

        return torch.stack([
            pred_cx - pred_w / 2,
            pred_cy - pred_h / 2,
            pred_cx + pred_w / 2,
            pred_cy + pred_h / 2,
        ], dim=1)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def forward(
        self,
        images: torch.Tensor,
        video_ids: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        B = images.shape[0]

        c2, c3, c4, c5 = self.backbone(images)

        pyramid = self.fpn(c3, c4, c5)

        cls_preds, reg_preds = self.detection_head(pyramid)
        anchors = self.anchor_gen(pyramid)

        heatmaps, keypoints, pose_confidence = self.pose_head(pyramid['p3'])

        if self.use_film and self.pose_film is not None:
            c5_mod = self.pose_film(c5, keypoints, pose_confidence)
        else:
            c5_mod = c5

        with torch.no_grad():
            det_conf = cls_preds.max(dim=1)[0]

        activity_proj = torch.cat([
            det_conf,
            F.adaptive_avg_pool2d(c5_mod, 1).flatten(1),
            F.adaptive_avg_pool2d(pyramid['p4'], 1).flatten(1),
        ], dim=1)

        proj_feat = self.activity_head.proj_features(activity_proj)

        bank_output = self.feature_bank(proj_feat, video_ids, None)

        act_logits = self.activity_head(
            proj_feat=proj_feat,
            temporal_bank=bank_output,
        )

        return {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'anchors': anchors,
            'heatmaps': heatmaps,
            'keypoints': keypoints,
            'pose_confidence': pose_confidence,
            'c5_mod': c5_mod,
            'det_conf': det_conf,
            'act_logits': act_logits,
            'temporal_features': bank_output,
            'c5_raw': c5,
            'pyramid': pyramid,
        }

    def forward_sequence(
        self,
        images_seq: torch.Tensor,
        video_ids: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = images_seq.shape[:2]
        images_flat = images_seq.reshape(B * T, 3, *images_seq.shape[3:])

        c2, c3, c4, c5 = self.backbone(images_flat)
        pyramid = self.fpn(c3, c4, c5)

        cls_preds, reg_preds = self.detection_head(pyramid)
        heatmaps, keypoints, pose_confidence = self.pose_head(pyramid['p3'])
        if self.use_film and self.pose_film is not None:
            c5_mod = self.pose_film(c5, keypoints, pose_confidence)
        else:
            c5_mod = c5

        with torch.no_grad():
            det_conf = cls_preds.max(dim=1)[0]

        activity_proj = torch.cat([
            det_conf,
            F.adaptive_avg_pool2d(c5_mod, 1).flatten(1),
            F.adaptive_avg_pool2d(pyramid['p4'], 1).flatten(1),
        ], dim=1)

        proj_feat = self.activity_head.proj_features(activity_proj)

        proj_feat_per_frame = proj_feat.view(B, T, -1)

        bank_seqs = []
        for t in range(T):
            feat_t = proj_feat_per_frame[:, t, :]
            vid_list = video_ids if video_ids is not None else None
            bank_t = self.feature_bank(feat_t, vid_list, None)
            bank_seqs.append(bank_t)

        bank_seq = torch.stack(bank_seqs, dim=1)

        act_logits_seq_list = []
        for t in range(T):
            logits_t = self.activity_head(
                proj_feat=proj_feat_per_frame[:, t, :],
                temporal_bank=bank_seq[:, t, :, :] if bank_seq.dim() == 4 else None,
            )
            act_logits_seq_list.append(logits_t)
        act_logits_seq = torch.stack(act_logits_seq_list, dim=1)

        tma_logvar = torch.zeros(B, T, device=images_seq.device)

        return {
            'temporal_al': {
                'tma_logvar': tma_logvar,
                'temporal_features': bank_seq,
            },
            'act_logits_seq': act_logits_seq,
            'temporal_ordering': None,
        }


def count_parameters(model: POPWMultiTaskModel) -> Dict[str, int]:
    components = {
        'backbone': [model.backbone],
        'fpn': [model.fpn],
        'detection': [model.detection_head],
        'pose_head': [model.pose_head],
        'pose_film': [model.pose_film],
        'activity_head': [model.activity_head],
        'feature_bank': [model.feature_bank],
    }
    result = {}
    total = 0
    for name, modules in components.items():
        count = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
        result[name] = count
        total += count
    result['total_trainable'] = total
    result['total_all'] = sum(p.numel() for p in model.parameters())
    return result


# Alias matching industriali's train.py import
MultiTaskIKEA = POPWMultiTaskModel