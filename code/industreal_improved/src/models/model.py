import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
POPW: Pose-Conditioned Multi-Task Architecture for IKEA/IndustReal Recognition
================================================================================
Matches popw_paper.tex architecture (ConvNeXt-Tiny backbone).

BACKBONE (ConvNeXt-Tiny, ImageNet pretrained):
  C2 (stride 4, 96ch) -> C3 (stride 8, 192ch) -> C4 (stride 16, 384ch) -> C5 (stride 32, 768ch)
  C5 goes DIRECTLY to PoseFiLM (bypasses FPN)

FPN NECK:
  [C3,C4,C5] -> [P3,P4,P5] (lateral 1x1 + top-down upsample + 3x3 smooth)
  P6/P7 via stride-2 conv on C5. All pyramid levels 256ch.

DETECTION HEAD (RetinaNet-style, P3-P7):
  Cls subnet: 4x Conv3x3+ReLU -> Conv(9x24)
  Reg subnet: 4x Conv3x3+ReLU -> Conv(9x4)
  Anchors: 3 ratios x 3 scales = 9 per location, sizes (24,48,96,192,384)
  Focal(alpha=0.25,gamma=2) + GIoU loss

POSE HEAD (Body -- 17 keypoints):
  Input: P3 (stride 8, 256ch)
  ConvTranspose2d(k=4,s=2,p=1) + GroupNorm(32) + ReLU -> P3 resolution
  Conv1x1 -> heatmaps [B, 17, H, W]
  Soft-argmax(T=0.1) -> keypoints [B, 34] + confidence [B, 17]
  Wing Loss (ω=0.05, ε=0.005) x 0.001

HEAD POSE HEAD (9-DoF):
  GAP(C4) ‖ GAP(C5) -> [B, 384+768=1152] -> MLP 1152->512->256->9
  LayerNorm + GELU + Dropout. MSE x 0.001

POSEFILM MODULE:
  keypoints [B,34] ‖ confidence [B,17] -> [B,51]
  gamma-net: 51->512->768, 1+tanh ∈ (0,2)
  beta-net: same, unbounded
  C5_direct (bypasses FPN). C5_mod = gamma·C5 + beta

HEADPOSEFILM MODULE:
  head_pose [B,9] (stop_grad -- CRITICAL per paper)
  gamma_hp: 9->256->768, 1+tanh. beta_hp: unbounded
  C5_mod2 = gamma_hp·C5_mod + beta_hp

ACTIVITY HEAD (Feature Bank + TCN + 2xViT):
  det_conf = MaxPool(cls_preds) -> [B, 24] (stop_grad)
  f_joint = [24 ‖ GAP(C5_mod2)(768) ‖ GAP(P4)(256)] -> [B, 1048]
  W_proj: Linear(1048->512) -> f̃_t [B, 512]
  Feature Bank T=16: B_t = [f̃_{t-T+1}, ..., f̃_t] [B, T, 512]
  TCN: 1D Depthwise Conv(k=5, dilation=1) -- true depthwise per paper
  2x ViT blocks: CLS token, learnable pos embed, MHSA(8heads, d_k=64), FFN(512->2048->512)
  DropPath 0.10/0.15, pre-norm, attn_dropout=0.1
  CLS readout -> Dropout(0.1) -> Linear(512->74)
  LDAM-DRW

PSR HEAD:
  Multi-scale GAP(P3+P4+P5) -> concat -> MLP(768->256)
  Causal Transformer: 3 layers, 4 heads, d_model=256
  11 per-component tiny MLPs (256->64->1)
  Binary Focal(alpha=0.25,gamma=2.0) + temporal smoothness(w=0.05)

LOSSES:
  Kendall homoscedastic uncertainty: L = Σ_t exp(-s_t)·L_t·ramp_t + s_t
  init: s_det=0, s_pose=-1, s_act=0, s_psr=0. Clamp[-4,2].
  Activity ramp: min(1, epoch/5).

Author: Bashara
Date: April 2026
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C


# ===========================================================================
# Soft-Argmax -- differentiable keypoint extraction from heatmaps
# ===========================================================================
class SoftArgmax(nn.Module):
    """
    Differentiable soft-argmax for keypoint extraction.
    Converts heatmaps to spatial coordinates via:
      coords = sum_{x,y} (x,y) * softmax(heatmap / temperature)
    Temperature controls sharpness -- lower = more peaked.
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmaps: [B, K, H, W] -- K keypoints, raw unnormalized
        Returns:
            coords: [B, K, 2] -- (x, y) spatial coordinates
            confidence: [B, K] -- max-pooled heatmap values (0-1)
        """
        B, K, H, W = heatmaps.shape
        flat = heatmaps.view(B, K, H * W)  # [B, K, HW]
        weights = F.softmax(flat / self.temperature, dim=-1)  # [B, K, HW]

        # 2D coordinate grids [H, W]
        grid_x_2d = torch.arange(W, device=heatmaps.device, dtype=torch.float32).unsqueeze(0).repeat(H, 1)   # [H, W]
        grid_y_2d = torch.arange(H, device=heatmaps.device, dtype=torch.float32).unsqueeze(1).repeat(1, W)   # [H, W]
        # Add head_dim=1 for broadcasting: [1, H, W]
        grid_x_2d = grid_x_2d.unsqueeze(0)  # [1, H, W]
        grid_y_2d = grid_y_2d.unsqueeze(0)  # [1, H, W]

        # Weighted sum over spatial dimensions: [B, K, H, W] x [1, H, W] -> sum -> [B, K, 1, 1]
        coords_x = (weights.view(B, K, H, W) * grid_x_2d).sum(dim=[-2, -1], keepdim=True)
        coords_y = (weights.view(B, K, H, W) * grid_y_2d).sum(dim=[-2, -1], keepdim=True)
        coords = torch.cat([coords_x, coords_y], dim=-1).squeeze(-2)  # [B, K, 2]

        # Confidence: max heatmap value per keypoint
        conf = heatmaps.max(dim=-1)[0].max(dim=-1)[0]  # [B, K]
        conf = torch.sigmoid(conf)  # [B, K]

        return coords, conf


# ===========================================================================
# Wing Loss -- robust regression loss for keypoint prediction
# ===========================================================================
class WingLoss(nn.Module):
    """
    Wing Loss (Wu et al., WACV 2018) for robust keypoint regression.
    L = ω * ln(1 + |x|/ε)  for |x| < ω
       |x| - C              for |x| >= ω
    where C = ω - ω*ln(1+ω/ε)
    """
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
# Backbone factory -- supports ResNet-50 and ConvNeXt-Tiny (Doc 01 D.1)
# ===========================================================================
class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt-Tiny pretrained backbone (ImageNet).

    Doc 01 D.1: Replaces ResNet-50 for improved ImageNet performance (+1.5%).
    ConvNeXt-Tiny: twenty-eight-M params vs ResNet-50s twenty-five-M.
    Channel outputs differ from ResNet-50:
      C2: 96ch   (after first stage / stem)
      C3: 192ch  (after second stage with downsample)
      C4: 384ch  (after third stage with downsample)
      C5: 768ch  (after fourth stage with downsample, NOT 2048 like ResNet)

    BN handling: ConvNeXt uses LayerNorm internally, no frozen BN layers.

    Memory optimization: gradient checkpointing on stages for RTX 3060 11GB.
    Uses torch.utils.checkpoint.checkpoint_sequential to trade ~20% compute
    for ~50% activation memory reduction during backprop.
    """
    def __init__(self, pretrained: bool = True, use_checkpoint: bool = False):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.use_checkpoint = use_checkpoint

        # Stage definitions: (stage_modules, output_name)
        # Stages 0-1: stem + stage1 -> C2 (stride 4, 96ch)
        # Stages 2-3: downsample2 + stage2 -> C3 (stride 8, 192ch)
        # Stages 4-5: downsample3 + stage3 -> C4 (stride 16, 384ch)
        # Stage 6:   downsample4 + stage4 -> C5 (stride 32, 768ch)
        # Stage 7:   3 CNBlocks (no spatial downsample, produces final C5 at stride 32)
        self.stage_groups = [
            nn.ModuleList([self.model.features[0], self.model.features[1]]),   # C2
            nn.ModuleList([self.model.features[2], self.model.features[3]]),   # C3
            nn.ModuleList([self.model.features[4], self.model.features[5]]),   # C4
            nn.ModuleList([self.model.features[6], self.model.features[7]]),   # C5 (downsample + CNBlocks)
        ]

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
        import torch.utils.checkpoint as checkpoint

        def stage0(x):
            for m in self.stage_groups[0]:  # features[0] + features[1] -> C2
                x = m(x)
            return x

        def stage1(x):
            for m in self.stage_groups[1]:  # features[2] + features[3] -> C3
                x = m(x)
            return x

        def stage2(x):
            for m in self.stage_groups[2]:  # features[4] + features[5] -> C4
                x = m(x)
            return x

        def stage3(x):
            for m in self.stage_groups[3]:  # features[6] + features[7] -> C5 (downsample + CNBlocks)
                x = m(x)
            return x

        # Gradient checkpointing: each stage is wrapped with checkpoint to trade
        # ~20% compute for ~50% activation memory reduction during backprop.
        # use_reentrant=False is required for proper gradient flow with autograd.
        if self.use_checkpoint and self.training:
            c2 = checkpoint.checkpoint(stage0, x, use_reentrant=False)
            c3 = checkpoint.checkpoint(stage1, c2, use_reentrant=False)
            c4 = checkpoint.checkpoint(stage2, c3, use_reentrant=False)
            c5 = checkpoint.checkpoint(stage3, c4, use_reentrant=False)
        else:
            c2 = stage0(x)
            c3 = stage1(c2)
            c4 = stage2(c3)
            c5 = stage3(c4)

        return c2, c3, c4, c5


def set_backbone_stage_requires_grad(
    model: nn.Module,
    backbone_type: str,
    stage: int,
    requires_grad: bool,
) -> None:
    """
    Freeze/unfreeze a specific stage of the backbone.

    Doc 01 B.4: Freeze only stage 1 (layer1 for ResNet, stages[0-1] for ConvNeXt)
    during synthetic pretrain for domain adaptation. Stages 2-4 remain trainable.

    Args:
        model: POPWMultiTaskModel (has .backbone attribute)
        backbone_type: 'resnet50' or 'convnext_tiny'
        stage: 0-indexed stage to freeze/unfreeze
               stage=1 -> freeze/unfreeze stage 1
        requires_grad: True=trainable, False=frozen
    """
    if backbone_type == 'resnet50':
        resnet: ResNet50Backbone = model.backbone
        layer_map = {1: resnet.layer1, 2: resnet.layer2, 3: resnet.layer3, 4: resnet.layer4}
        if stage in layer_map:
            for param in layer_map[stage].parameters():
                param.requires_grad = requires_grad

    elif backbone_type == 'convnext_tiny':
        convnext: ConvNeXtBackbone = model.backbone
        features = convnext.model.features
        # Map logical stage index to ConvNeXt feature indices
        # stage 0 -> features[0,1] (stem + stage1 -> produces C2)
        # stage 1 -> features[2,3] (downsample2 + stage2 -> produces C3)
        # stage 2 -> features[4,5] (downsample3 + stage3 -> produces C4)
        # stage 3 -> features[6]   (downsample4+stage4 -> produces C5)
        stage_to_features = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7],
        }
        if stage in stage_to_features:
            for feat_idx in stage_to_features[stage]:
                for param in features[feat_idx].parameters():
                    param.requires_grad = requires_grad


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 pretrained backbone (ImageNet).
    Matches the XML diagram exactly:
      C2: 256ch, stride 4  (after layer1, after initial conv+pool)
      C3: 512ch, stride 8  (after layer2)
      C4: 1024ch, stride 16 (after layer3)
      C5: 2048ch, stride 32 (after layer4)
    BN layers are frozen during training EXCEPT layer4 (C5) which re-learns
    industrial scene statistics. (Doc 01 B.2)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self._freeze_bn()

    def _freeze_bn(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if name.startswith('layer4'):
                    continue
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._freeze_bn()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            c2: [B, 256, H/4, W/4]
            c3: [B, 512, H/8, W/8]
            c4: [B, 1024, H/16, W/16]
            c5: [B, 2048, H/32, W/32]
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        c2 = x

        x = self.model.layer2(x)
        c3 = x

        x = self.model.layer3(x)
        c4 = x

        x = self.model.layer4(x)
        c5 = x

        return c2, c3, c4, c5


def build_backbone(backbone_type: str = 'resnet50', pretrained: bool = True) -> nn.Module:
    """
    Factory function to build backbone by type.
    Doc 01 D.1: Supports 'resnet50' and 'convnext_tiny'.
    """
    if backbone_type == 'convnext_tiny':
        return ConvNeXtBackbone(pretrained=pretrained)
    elif backbone_type == 'resnet50':
        return ResNet50Backbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Use 'resnet50' or 'convnext_tiny'.")


# ===========================================================================
# FPN -- Feature Pyramid Network
# ===========================================================================
class FPN(nn.Module):
    """
    Standard FPN. Takes [C2, C3, C4, C5] -> [P3, P4, P5, P6, P7].
    All pyramid levels have 256 channels.
    Uses lateral 1x1 convs and top-down upsampling with 3x3 smoothing convs.
    P6/P7 derived via stride-2 conv on C5.
    """
    def __init__(self, in_channels: List[int] = [512, 1024, 2048], out_channels: int = 256):
        super().__init__()
        # in_channels: [C3, C4, C5] -- C2 (stride 4) is not used in FPN
        c3_ch, c4_ch, c5_ch = in_channels

        self.lateral_c3 = nn.Conv2d(c3_ch, out_channels, 1)
        self.lateral_c4 = nn.Conv2d(c4_ch, out_channels, 1)
        self.lateral_c5 = nn.Conv2d(c5_ch, out_channels, 1)

        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # P6/P7 from C5 via stride conv
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
        # Standard FPN: top-down pathway with lateral connections
        # C3(512ch) -> lateral -> P3(256ch) at stride 8
        # C4(1024ch) -> lateral -> P4(256ch) at stride 16
        # C5(2048ch) -> lateral -> P5(256ch) at stride 32
        # P6/P7 from stride-2 conv on C5
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
    """
    Generate anchors for RetinaNet. 3 ratios x 3 scales = 9 per location.
    Anchor sizes from C.ANCHOR_SIZES, calibrated via k-means on GT boxes (Doc 01 B.3).
    """
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
        self.num_anchors = len(ratios) * len(scales)  # 9

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
    """
    RetinaNet detection head with shared cls/reg subnets across FPN levels.
    - Cls subnet: 4x Conv3x3+ReLU -> Conv(9x24) for 24 ASD classes
    - Reg subnet: 4x Conv3x3+ReLU -> Conv(9x4)
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 24, num_anchors: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def make_subnet():
            layers = []
            for _ in range(4):
                layers.extend([
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.GroupNorm(8, in_channels),
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

        # Init bias to a less aggressive prior -- pi=0.03 (3% positive rate) instead of 0.01
        # This prevents all predictions collapsing to "no object" when EMA locks in the bias.
        # The bias learns from data during training; starting at -3.4 is not catastrophic.
        pi = 0.03
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
# Pose Head -- heatmaps -> soft-argmax -> keypoints + confidence
# ===========================================================================
class PoseHead(nn.Module):
    """
    Pose head generating heatmaps from P3 features, then extracting
    keypoints via soft-argmax and confidence scores.

    Architecture per diagram:
      ConvTranspose2d(k=4, s=2, p=1) -> GroupNorm(32) + ReLU
      -> Conv1x1 -> heatmaps [B, 17, H, W]
      -> Soft-argmax -> keypoints [B, 17, 2] + confidence [B, 17]

    For 1280x720 input:
      C3 (stride 8) = 192ch -> FPN P3 (stride 8) = 256ch
      ConvTranspose2d on P3: 256 -> 256, k=4, s=2 -> [B, 256, H/4, W/4] = [B, 256, 180, 160]
      Then heatmap head: 256 -> 17
    """
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
        """
        Args:
            p3_feature: [B, 256, H/8, W/8] from FPN P3
        Returns:
            heatmaps: [B, 17, H/4, W/4] raw unnormalized heatmaps
            keypoints: [B, 17, 2] spatial (x, y) coordinates
            confidence: [B, 17] per-keypoint confidence (0-1)
        """
        x = self.upsample(p3_feature)  # [B, 256, H/4, W/4]
        heatmaps = self.heatmap_head(x)  # [B, 17, H/4, W/4]
        keypoints, confidence = self.soft_argmax(heatmaps)
        return heatmaps, keypoints, confidence


# ===========================================================================
# PoseFiLM Module -- keypoint-conditioned FiLM modulation on C5
# ===========================================================================
class PoseFiLMModule(nn.Module):
    """
    PoseFiLM: keypoint + confidence -> gamma/beta -> modulate C5.

    Matches the XML diagram exactly:
      keypoints [B,17, 2] ‖ confidence [B,17] -> pose_flat [B,51]
      gamma-net: 51->512->768, 1+tanh(·) ∈ (0,2)
      beta-net: 51->512->768, linear (unbounded)
      C5_mod = gamma · C5 + beta

    Note: gamma/beta always project to 768 channels (paper spec) regardless of
    backbone C5 channel count. For ConvNeXt-Tiny (C5=768): direct match.
    For ResNet-50 (C5=2048): C5 is reshaped to 768 via its channel dimension
    before modulation. See forward() for the reshape logic.
    """
    # Fixed output dimension -- paper specifies 768 per ?PoseFiLM
    FILM_DIM = 768

    def __init__(self, num_keypoints: int = 17, c5_channels: int = 2048,
                 hidden_channels: int = 512):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.c5_channels = c5_channels

        # 34 keypoint coords + 17 confidence = 51
        self.gamma_net = nn.Sequential(
            nn.Linear(34 + 17, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, self.FILM_DIM),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(34 + 17, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, self.FILM_DIM),
        )

        self._init_weights()

        # 1x1 conv to project C5 from c5_channels -> FILM_DIM (768) per paper spec
        if self.c5_channels != self.FILM_DIM:
            self._c5_proj = nn.Conv2d(self.c5_channels, self.FILM_DIM, kernel_size=1)

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
        """
        Args:
            c5: [B, c5_channels, H/32, W/32] -- C5 from backbone (2048 for ResNet-50, 768 for ConvNeXt-Tiny)
            keypoints: [B, 17, 2] -- (x, y) in image coordinates
            confidence: [B, 17]
        Returns:
            c5_mod: [B, 768, H/32, W/32] -- FiLM-modulated C5 at 768 channels (paper spec)
        """
        B = keypoints.shape[0]

        # Normalize keypoints to [0, 1] based on image size
        scale = torch.tensor([C.IMG_WIDTH, C.IMG_HEIGHT],
                             device=keypoints.device, dtype=keypoints.dtype)
        keypoints_norm = keypoints / scale.view(1, 1, 2)  # [B, 17, 2]

        # Flatten keypoints: [B, 34]
        kp_flat = keypoints_norm.flatten(1)  # [B, 34]
        conf_flat = confidence.detach()  # [B, 17] -- stop gradient

        # Concatenate: [B, 51]
        pose_flat = torch.cat([kp_flat, conf_flat], dim=1)

        # Compute gamma and beta -- always 768-dim per paper spec
        gamma_raw = self.gamma_net(pose_flat)  # [B, 768]
        beta_raw = self.beta_net(pose_flat)  # [B, 768]

        # gamma: 1 + tanh ? (0, 2)
        gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]
        beta = beta_raw.unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]

        # Reshape C5 to 768 channels (FiLM always operates at 768 per paper)
        if self.c5_channels != self.FILM_DIM:
            # Project C5 from c5_channels -> FILM_DIM via 1x1 conv
            c5_768 = self._c5_proj(c5)  # [B, 768, H, W]
        else:
            c5_768 = c5

        return gamma * c5_768 + beta


# ===========================================================================
# HeadPoseFiLM Module -- 9-DoF head pose conditioned FiLM (Doc 01 E)
# ===========================================================================
class HeadPoseFiLMModule(nn.Module):
    """
    HeadPoseFiLM: second-stage FiLM modulation from 9-DoF head pose.

    Doc 01 E: Since IndustReal uses head pose (9-DoF) not body keypoints,
    condition C5 on head pose features in addition to the keypoint-based FiLM.

    Architecture mirrors PoseFiLMModule but takes 9-DoF head pose as input:
      - forward_vector (3): where the person is looking
      - position (3): head position in world/coordinate space
      - up_vector (3): head orientation

    The second FiLM stage refines C5_mod (already modulated by keypoints)
    using ego-centric gaze direction which is critical for industrial actions
    (worker looks at assembly area vs away).

    Flow:
      c5_mod (from PoseFiLM, 768ch) -> HeadPoseFiLM -> gamma_hp, beta_hp (768ch)
      c5_mod_2 = gamma_hp · c5_mod + beta_hp

    Note: gamma/beta always project to 768 channels (FILM_DIM) per paper spec,
    regardless of backbone C5 channel count. The input c5_mod is already
    768 channels from PoseFiLM's projection.
    """
    FILM_DIM = 768  # shared with PoseFiLMModule

    def __init__(self, c5_channels: int = 2048, hidden_channels: int = 256):
        super().__init__()
        self.c5_channels = c5_channels

        self.gamma_net = nn.Sequential(
            nn.Linear(9, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.FILM_DIM),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(9, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.FILM_DIM),
        )

        self._init_weights()

    def _init_weights(self):
        for net in [self.gamma_net, self.beta_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
        nn.init.ones_(self.gamma_net[-1].bias)

    def forward(self, c5_mod: torch.Tensor, head_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c5_mod: [B, 768, H/32, W/32] -- PoseFiLM output (768ch per paper spec)
            head_pose: [B, 9] -- 9-DoF head pose (forward[3] + position[3] + up[3])
        Returns:
            c5_mod_2: [B, 768, H/32, W/32] -- double-modulated C5 (768ch)
        """
        gamma_raw = self.gamma_net(head_pose)
        beta_raw = self.beta_net(head_pose)

        # gamma: 1 + tanh ? (0, 2), beta: unbounded
        gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]
        beta = beta_raw.unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]

        # c5_mod is already 768ch from PoseFiLM -- direct modulation
        return gamma * c5_mod + beta


# ===========================================================================
# VideoMAE V2 Stream -- 2-stream Activity feature extractor (Doc 2 A.1)
# ===========================================================================
class VideoMAEStream(nn.Module):
    """
    VideoMAE V2 stream for activity recognition -- separate temporal modeling.

    Doc 2 A.1: Initialize from VideoMAE V2 fine-tuned on Kinetics-400 (87.4% Top-1).
    This is the single biggest unlock for Activity Top-1 (+5 to +7%).

    Architecture:
      - VideoMAE V2 ViT-S/16 (twenty-two million frozen params) for first 10 epochs
      - Takes same 16-frame clip as POPW feature bank
      - Outputs 384-D feature per clip
      - Fused with existing activity head before the classifier

    2-stream design:
      - Stream 1 (CNN backbone): handles detection, pose, PSR, head pose (per-frame tasks)
      - Stream 2 (VideoMAE): handles ONLY Activity, fused before classifier

    Cost: +twenty-two million params, +6 GFLOPs. FPS drops ~25% but stays >15 on RTX 3060.

    Fallback: If VideoMAE checkpoint unavailable (OSError/HF errors), falls back to a
    lightweight three-D-convolution temporal stream that still captures temporal dependencies
    without requiring external checkpoint download.
    """
    def __init__(self, ckpt: str = 'MCG-NJU/videomae-small-finetuned-kinetics'):
        super().__init__()
        self._use_fallback = False
        self.hidden_size = 384
        self._position_embeddings_len = 1568  # default for T=16, tubelet_size=2, 14x14 spatial

        # Local cache path for the 384-D VideoMAE-Small checkpoint
        # (MCG-NJU/videomae-small-finetuned-kinetics -> hidden_size=384, num_hidden_layers=12)
        _cache_root = os.path.expanduser('~/.cache/huggingface/hub')
        _model_cache = os.path.join(
            _cache_root,
            'models--MCG-NJU--videomae-small-finetuned-kinetics',
            'snapshots',
            '240e9734611173accbbf74cbdf4b641e4c431264',
        )
        _config_path = os.path.join(
            _cache_root,
            'models--MCG-NJU--videomae-small-finetuned-kinetics',
            'snapshots',
            '1bf2b82c809609f5df1ea7648aee577e2e486b63',
            'config.json',
        )
        _ckpt_path = os.path.join(_model_cache, 'model.safetensors')

        try:
            from transformers import VideoMAEModel, VideoMAEConfig

            # Build config explicitly with the 384-D small variant params
            # (avoids default ViT-Base hidden_size=768 mismatch)
            config = VideoMAEConfig(
                hidden_size=384,
                num_hidden_layers=12,
                intermediate_size=1536,
                num_attention_heads=16,
                decoder_hidden_size=192,
                decoder_intermediate_size=768,
                decoder_num_attention_heads=3,
                decoder_num_hidden_layers=12,
                image_size=224,
                patch_size=16,
                num_frames=16,
                tubelet_size=2,
                qkv_bias=True,
                use_mean_pooling=True,
            )
            self.encoder = VideoMAEModel(config)

            # Load state dict from local cache with strict=False
            if os.path.exists(_ckpt_path):
                from safetensors.torch import load_file as _load_safetensors
                state_dict = _load_safetensors(_ckpt_path)
                missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.debug(f'VideoMAE: missing keys (expected): {missing}')
                if unexpected:
                    logger.debug(f'VideoMAE: unexpected keys (ignored): {unexpected}')
                logger.info('VideoMAE: loaded VideoMAE-Small (384-D) from local cache')
            elif os.path.exists(ckpt):
                # Fallback: load from provided path
                self.encoder = VideoMAEModel.from_pretrained(ckpt, local_files_only=True)
            else:
                raise FileNotFoundError(f'VideoMAE checkpoint not found at {_ckpt_path}')

            for p in self.encoder.parameters():
                p.requires_grad = False
            self._forward = self._forward_videomae
            logger.info('VideoMAEStream: VideoMAE-Small (384-D) loaded successfully')

        except Exception as ex:
            import warnings
            warnings.warn(f'VideoMAE checkpoint unavailable ({ex}), using 3D temporal fallback')
            self._use_fallback = True
            self.hidden_size = 384
            self._build_fallback_encoder()

    def _build_fallback_encoder(self):
        """Three-D temporal convolution fallback stream -- captures temporal patterns without checkpoint."""
        # 3D conv stack: temporal mixing via 3D kernels, spatial via 2D after flattening
        self.fb = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # [B,64,T/2,112,112]

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),  # [B,128,T/4,56,56]

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),  # [B,256,T/8,28,28]

            nn.Conv3d(256, 384, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B,384,1,1,1]
        )
        for p in self.fb.parameters():
            p.requires_grad = True

    def _forward_videomae(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        clip = clip.reshape(B, T, C, H, W)

        cfg = self.encoder.config
        tubelet_size = getattr(cfg, 'tubelet_size', 2)
        patch_size = cfg.patch_size
        num_spatial_patches = (cfg.image_size // patch_size) ** 2
        actual_seq_len = (T // tubelet_size) * num_spatial_patches

        if actual_seq_len != self._position_embeddings_len:
            pe = self.encoder.embeddings.position_embeddings
            pe_interp = F.interpolate(
                pe.permute(0, 2, 1),  # [1, 384, 1568]
                size=actual_seq_len,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1)  # [1, actual_seq_len, 384]
            self.encoder.embeddings.position_embeddings = nn.Parameter(
                pe_interp, requires_grad=False
            )
            self._position_embeddings_len = actual_seq_len

        outputs = self.encoder(pixel_values=clip)
        feat = outputs.last_hidden_state.mean(dim=1)
        return feat

    def _forward_fallback(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        x = clip.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, W]
        x = self.fb(x)  # [B, 384, 1, 1, 1]
        x = x.view(B, self.hidden_size)  # [B, 384]
        return x

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if self._use_fallback:
            B, T, C, H, W = clip.shape
            x = clip.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, W]
            x = self.fb(x)  # [B, 384, 1, 1, 1]
            x = x.view(B, self.hidden_size)  # [B, 384]
            return x
        return self._forward_videomae(clip)

    def unfreeze(self, lr: float = 1e-5):
        if self._use_fallback:
            for p in self.fb.parameters():
                p.requires_grad = True
            return [{'params': self.fb.parameters(), 'lr': lr}]
        for p in self.encoder.parameters():
            p.requires_grad = True
        return [{'params': self.encoder.parameters(), 'lr': lr}]


# ===========================================================================
# DropPath (Stochastic Depth) -- paper ?ActivityHead: DropPath 0.1/0.15 applied in TCN + ViT
# ===========================================================================
def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly drop entire residual branches during training."""
    if drop_prob == 0. or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
    return x / keep * mask


# ===========================================================================
# Temporal Convolutional Block -- captures local motion patterns
# ===========================================================================
class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block - captures local motion patterns.

    Replaces simple positional embedding with:
      - Temporal LayerNorm
      - Depthwise Conv (capture short-range motion: velocity, acceleration)
      - Residual connection + DropPath (stochastic depth)

    The TCN output feeds into the ViT for long-range temporal reasoning.
    """
    def __init__(self, embed_dim: int = 512, kernel_size: int = 5,
                 dropout: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)
        # [FIX #5 MEDIUM] True depthwise: groups=embed_dim, single conv per paper ?ActivityHead
        self.depthwise_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=embed_dim
        )
        # [FIX #TCN] Pointwise 1x1 conv after depthwise -- enables cross-channel
        # temporal modulation. Per TCN paper: "The pointwise (1x1) conv ... mixes
        # information across channels." Without this, each channel only sees its own
        # temporal filter (depthwise-only), limiting inter-channel motion modeling.
        self.pointwise_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.drop_path_prob = drop_path

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.depthwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)
        nn.init.xavier_uniform_(self.pointwise_conv.weight)
        nn.init.zeros_(self.pointwise_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise_conv(x_conv)  # depthwise: temporal per channel
        x_conv = self.pointwise_conv(x_conv)  # [FIX #TCN] pointwise: cross-channel modulation
        x_conv = self.gelu(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)
        return residual + _drop_path(x_conv, self.drop_path_prob, self.training)


class ViTTemporalBlock(nn.Module):
    """
    ViT Temporal Block -- Matches XML diagram exactly.

    Diagram spec: Learnable pos embed [1, T, 512] + MHSA (eight_heads, dk=64) + FFN (512->2048->512)

    No RoPE -- diagram does not specify it.

    Note: ActivityHead (line 1183) overrides to num_heads=8 per paper spec
    (embed_dim=512, dk=512/8=64). ViTTemporalBlock default (num_heads=4, dk=128)
    is for other uses (e.g., PSR temporal model).
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1, drop_path: float = 0.1,
                 max_seq_len: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, f"{embed_dim} % {num_heads} != 0"

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Learnable positional embedding per diagram
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

        self.drop_path_prob = drop_path
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

        # Add learnable positional embedding per diagram
        x = x + self.pos_embed[:, :T, :]

        x_normed = self.norm1(x)

        q = self.q_proj(x_normed).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, T, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        # [AUDIT FIX 2026-06-11 / RC-16] was `/ scale`: dividing by d^-0.5
        # MULTIPLIES the attention logits by sqrt(d)=8 instead of dividing —
        # logits 64x larger than standard scaled dot-product attention →
        # softmax saturates to near-one-hot and gradients through attention
        # vanish (activity-ViT collapse mechanism). Standard form: qk^T * d^-0.5.
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        x = x + _drop_path(out, self.drop_path_prob, self.training)
        x = x + _drop_path(self.ffn(x), self.drop_path_prob, self.training)

        return x


# ===========================================================================
# Feature Bank -- sliding window temporal memory
# ===========================================================================
class FeatureBank(nn.Module):
    """
    Feature Bank: sliding window of T=8 temporal projected features.
    Matches the XML diagram exactly.

    Stores [f̃_{t-T+1}, ..., f̃_t] as a ring buffer.
    Keyed by (video_id, camera_view) for multi-sequence handling.
    Memory: T x 512 x 2 bytes (FP16) = 8 KB per sequence
    """
    def __init__(self, embed_dim: int = 512, window_size: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self._bank: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def forward(self,
                projected_features: torch.Tensor,
                video_ids: Optional[List[str]] = None,
                camera_views: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            projected_features: [B, 512] -- current frame's projected feature
            video_ids: list of video IDs for per-frame mode
            camera_views: list of camera view IDs (e.g., 'front', 'top')
        Returns:
            bank: [B, T=8, 512] -- feature bank for temporal processing
        """
        if projected_features.dim() == 3:
            # Batch mode with temporal dim: [B, T, 512] -> use as-is
            # NaN guard: if input has NaN, sanitize to zeros to prevent cascade
            if not torch.isfinite(projected_features).all():
                return torch.zeros_like(projected_features)
            return projected_features

        B = projected_features.shape[0]

        if video_ids is None:
            # Fallback: replicate current frame T times
            return projected_features.unsqueeze(1).expand(-1, self.window_size, -1)

        # Per-frame mode: ring buffer keyed by (video_id, camera_view)
        outputs = []
        for i in range(B):
            vid = str(video_ids[i]) if video_ids[i] is not None else 'default'
            cam = str(camera_views[i]) if (camera_views is not None and camera_views[i] is not None) else 'default'
            key = (vid, cam)

            feat_i = projected_features[i]  # [512]

            # NaN/Inf guard: skip storing corrupted features in the bank.
            # A single NaN feature contaminates all future temporal windows
            # and triggers a permanent NaN cascade (PSR → all heads collapse
            # to 1e-4 fallback). If the bank has prior valid data, pad with
            # the last valid feature instead; otherwise use zeros.
            if key not in self._bank:
                self._bank[key] = []
            feat_is_valid = torch.isfinite(feat_i).all()
            if not feat_is_valid:
                if key in self._bank and len(self._bank[key]) > 0:
                    # Pad with last valid feature
                    self._bank[key].append(self._bank[key][-1].clone())
                else:
                    # No prior data — use zeros
                    zero_feat = torch.zeros_like(feat_i)
                    # Initialize bank with copies of zero_feat
                    self._bank[key] = [zero_feat.clone() for _ in range(self.window_size)]
                    bank_i = torch.stack(self._bank[key])
                    outputs.append(bank_i)
                    continue
            else:
                # [OPUS v5 AUDIT] FeatureBank gradient: when FEATURE_BANK_DETACH=False,
                # gradient flows through bank entries enabling temporal learning (#14-16).
                _bank_detach = bool(getattr(C, 'FEATURE_BANK_DETACH', True))
                _stored = feat_i.detach().clone() if _bank_detach else feat_i.clone()
                self._bank[key].append(_stored)

            if len(self._bank[key]) > self.window_size:
                self._bank[key].pop(0)

            seq = self._bank[key]
            while len(seq) < self.window_size:
                if feat_is_valid:
                    _bank_detach = bool(getattr(C, 'FEATURE_BANK_DETACH', True))
                    pad_feat = feat_i.detach().clone() if _bank_detach else feat_i.clone()
                else:
                    pad_feat = self._bank[key][0].clone()
                seq = [pad_feat] + seq
            seq = seq[-self.window_size:]

            bank_i = torch.stack(seq)  # [T, 512]
            outputs.append(bank_i)

        return torch.stack(outputs)  # [B, T, 512]

    def reset(self):
        """Clear all stored sequences."""
        self._bank.clear()

    def reset_sequence(self, video_id: str, camera_view: str = 'default'):
        """Clear stored sequence for a specific video/camera."""
        key = (str(video_id), str(camera_view))
        self._bank.pop(key, None)


# ===========================================================================
# Activity Head -- Feature Bank + TCN + 2xViT + CLS classifier (Doc 01 A)
# ===========================================================================
class ActivityHead(nn.Module):
    """
    Activity head with architectural improvements from Doc 01 A.

    Improvements over XML diagram:
      A.1: TCN (TemporalConvBlock) before ViT -- captures short-range motion
      A.2: T=16 temporal window (was T=8)
      A.3: 2x ViT blocks with CLS token + cross-attention pooling (replaces last-timestep)
      A.4: Attention dropout 0.1 on QK matrix

    Doc 2 A.1: Optional VideoMAE V2 stream fusion for +5-7% Top-1 gain.
      VideoMAE features (384-D) are fused with CNN features before the classifier.

    Inputs:
      det_conf: [B, 24] -- max-pooled detection cls scores (stop_grad)
      c5_mod: [B, 768, H/32, W/32] -- FiLM-modulated C5 (ConvNeXt-Tiny)
      p4: [B, 256, H/16, W/16] -- FPN P4 spatial features

    Flow:
      GAP(C5_mod) -> [B, 768]  (ConvNeXt-Tiny C5; ResNet-50 would be 2048)
      GAP(P4) -> [B, 256]
      Concat -> [B, 24+768+256] = [B, 1048]  (ConvNeXt-Tiny)
              or [B, 24+2048+256] = [B, 2328] (ResNet-50)
      W_proj (Linear 1048->512) -> f̃_t [B, 512]
      Feature Bank B_t = [f̃_{t-T+1}, ..., f̃_t] [B, T=16, 512]
      TCN (depthwise 1D conv) -> captures velocity/acceleration
      2x ViT blocks (8 heads, d_k=64) with CLS token
      CLS pooled output -> Dropout(0.1) -> act_logits [B, 74]

    With VideoMAE stream:
      VideoMAE(384-D) -> fused with CLS output -> classifier

    Note: The final classifier outputs 75 classes (indices 0-74). The dataset
    emits raw action_id values (0-74) directly as class indices — no shift.
    Action_id=0 is \"take_short_brace\" (a real action, not NA). Only ID 37 is
    absent from the dataset (permanent cold channel). Frames without any AR
    annotation are marked with label=-1 and excluded from the activity loss.
    """
    def __init__(self, c5_channels: int = 2048, p4_channels: int = 256,
                 det_conf_size: int = 24, embed_dim: int = 512,
                 num_classes: int = 74, dropout: float = 0.1,  # [FIX #3 HIGH] attn_dropout=0.1 per paper
                 window_size: int = 16, use_vit: bool = True,
                 vit_drop_path: float = 0.1,
                 use_videomae: bool = False, videomae_hidden: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_vit = use_vit
        self.use_videomae = use_videomae
        self.window_size = window_size

        self.gap_c5 = nn.AdaptiveAvgPool2d(1)
        self.gap_p4 = nn.AdaptiveAvgPool2d(1)

        proj_input_dim = det_conf_size + c5_channels + p4_channels  # 2328

        self.proj_features = nn.Linear(proj_input_dim, embed_dim)

        # A.1: TCN -- short-range motion (velocity/acceleration profiles)
        self.tcn = TemporalConvBlock(
            embed_dim=embed_dim,
            kernel_size=5,
            dropout=0.1,
            drop_path=0.1,
        )

        # A.3: 2x ViT blocks with CLS token (replaces last-timestep pooling)
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

        # CLS token for cross-attention pooling (TimeSformer / ViViT style)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Doc 2 A.1: VideoMAE fusion
        if use_videomae:
            self.videomae_proj = nn.Sequential(
                nn.Linear(videomae_hidden, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            classifier_input_dim = embed_dim * 2
        else:
            self.videomae_proj = None
            classifier_input_dim = embed_dim

        # Activity classifier
        self.activity_classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(0.1),
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, proj_feat: torch.Tensor,
                temporal_bank: Optional[torch.Tensor] = None,
                videomae_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            proj_feat: [B, 512] -- pre-projected joint features
            temporal_bank: [B, T, 512] feature bank sequence, or None for single-frame
            videomae_feat: [B, 384] optional VideoMAE V2 feature (Doc 2 A.1)
        Returns:
            act_logits: [B, num_classes]
        """
        B = proj_feat.shape[0]

        if temporal_bank is not None:
            bank_seq = temporal_bank.clone()
            # [OPUS v5 AUDIT] Slot -1 overwrite: when FEATURE_BANK_SLOT_OVERWRITE=False,
            # the bank accumulates naturally without the live frame replacing the last position.
            # This enables all T positions to contribute to temporal learning (#16).
            if getattr(C, 'FEATURE_BANK_SLOT_OVERWRITE', True):
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

        if self.use_videomae and videomae_feat is not None:
            videomae_emb = self.videomae_proj(videomae_feat)
            feat = torch.cat([feat, videomae_emb], dim=-1)
        elif self.use_videomae:
            feat = torch.cat([feat, torch.zeros_like(feat)], dim=-1)

        feat = F.dropout(feat, p=0.1, training=self.training)
        logits = self.activity_classifier(feat)
        return logits


# ===========================================================================
# Head Pose Head -- 9-DoF head pose from C5 features
# ===========================================================================
class HeadPoseHead(nn.Module):
    """
    Head pose head: predicts 9-DoF head pose from multi-scale C4+C5 features.

    Multi-scale: C4 (384ch, stride 16) + C5 (768ch, stride 32) from ConvNeXt-Tiny
    Fusion MLP: 1152->512->256->9 with LayerNorm

    9-DoF = forward_vector(3) + position(3) + up_vector(3)
    Trained with MSE loss against raw GT from pose.csv.
    """
    def __init__(self, c4_channels: int = 384, c5_channels: int = 768,
                 hidden_dim: int = 128):
        super().__init__()

        self.gap_c4 = nn.AdaptiveAvgPool2d(1)
        self.gap_c5 = nn.AdaptiveAvgPool2d(1)

        total_in = c4_channels + c5_channels  # 3072

        self.head = nn.Sequential(
            nn.Linear(total_in, hidden_dim * 4),  # 3072->512 (hidden_dim=128, so *4=512)
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 512->256
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, 9),
        )

    def forward(self, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        c4_gap = self.gap_c4(c4).flatten(1)
        c5_gap = self.gap_c5(c5).flatten(1)
        fused = torch.cat([c4_gap, c5_gap], dim=1)
        return self.head(fused)  # [B, 9]


# ===========================================================================
# PSR Head -- Causal Transformer + Per-Component Heads (Doc 01 C)
# ===========================================================================
class PSRHead(nn.Module):
    """
    PSR Head with architectural improvements from Doc 01 C.

    Architecture (actual implementation):
      C.1: Causal Transformer Encoder (3 layers, 4 heads, d_model=256) --
           processes per-frame features with upper-triangular causal masking.
           Each position attends only to itself and prior positions (no future).
           At inference with per-video cache: O(1) new frame cost after warmup.
      C.2: Per-component output heads -- each of 11 components has different
           transition statistics. Shared head underfits rare components.

    Architecture:
      - Per-frame feature: multi-scale P3+P4+P5 GAP -> MLP -> 256-D
      - Causal Transformer Encoder (3 layers, 4 heads, d_model=256, gelu, pre-norm)
      - Per-component output heads (11 separate tiny MLPs)

    Doc 2 C.3: Binary focal loss, not BCE. Heavy class imbalance per component.
    """
    def __init__(self, in_channels: int = 256, hidden_dim: int = 128,
                 num_components: int = 11, dropout: float = 0.2,
                 num_scales: int = 3, gru_hidden: int = 256):  # [FIX #2 HIGH] d_model=256 per paper
        super().__init__()
        self.num_components = num_components
        self.gru_hidden = gru_hidden

        self.gap_p3 = nn.AdaptiveAvgPool2d(1)
        self.gap_p4 = nn.AdaptiveAvgPool2d(1)
        self.gap_p5 = nn.AdaptiveAvgPool2d(1)

        per_scale_ch = in_channels * num_scales  # 768
        self.per_frame_mlp = nn.Sequential(
            nn.Linear(per_scale_ch, gru_hidden * 2),
            nn.LayerNorm(gru_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.LayerNorm(gru_hidden),
        )

        # C.1: Causal Transformer (3 layers, 4 heads, d_model=256) -- paper ?PSR Head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gru_hidden,
            nhead=4,
            dim_feedforward=gru_hidden * 4,  # 1024
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm as per paper ViT-style
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # Causal mask: upper-triangular prevents attending to future tokens
        self._causal_mask = None
        # Project back to gru_hidden for per-component heads

        # C.2: Per-component output heads (11 separate tiny MLPs)
        # comp0 (base plate) placed first 95%; comp10 (wheels) come last.
        # Each component has different transition statistics.
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_hidden, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.3),
                nn.Linear(64, 1),
            ) for _ in range(num_components)
        ])

        self._cache: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        self._MAX_CACHE_LEN = 32

    def _get_frame_feat(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        p3_gap = self.gap_p3(pyramid['p3']).flatten(1)
        p4_gap = self.gap_p4(pyramid['p4']).flatten(1)
        p5_gap = self.gap_p5(pyramid['p5']).flatten(1)
        fused = torch.cat([p3_gap, p4_gap, p5_gap], dim=1)
        return self.per_frame_mlp(fused)  # [B, gru_hidden]

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask (upper-triangular, prevents attending to future)."""
        if not hasattr(self, '_cached_mask') or self._cached_mask is None or self._cached_mask.size(0) != seq_len:
            # causal mask: position i can attend to positions 0..i (not i+1..T-1)
            ones = torch.ones(seq_len, seq_len, device=device)
            # torch.triu with diagonal=0 keeps main diagonal and above (past positions)
            mask = torch.triu(ones, diagonal=1).bool()
            # invert so True = ignore (cannot attend), False = attend
            self._cached_mask = mask
        return self._cached_mask

    def forward(self, pyramid: Dict[str, torch.Tensor],
                video_ids: Optional[List[str]] = None,
                camera_views: Optional[List[str]] = None
                ) -> torch.Tensor:
        """
        Args:
            pyramid: FPN feature pyramid
            video_ids: for per-sequence caching (optional)
            camera_views: for per-sequence caching (optional)
        Returns:
            psr_logits: [B, 11] per-frame predictions
        """
        B = pyramid['p3'].shape[0]
        frame_feat = self._get_frame_feat(pyramid)  # [B, gru_hidden]

        should_cache = (
            video_ids is not None
            and camera_views is not None
            and not self.training
        )

        if should_cache and len(video_ids) == B:
            outputs = []
            for i in range(B):
                vid = str(video_ids[i]) if video_ids[i] is not None else 'default'
                cam = str(camera_views[i]) if camera_views[i] is not None else 'default'
                key = (vid, cam)

                feat_i = frame_feat[i:i+1]  # [1, gru_hidden]

                if key not in self._cache:
                    self._cache[key] = []
                self._cache[key].append(feat_i.detach().clone())

                if len(self._cache[key]) > self._MAX_CACHE_LEN:
                    self._cache[key] = self._cache[key][-self._MAX_CACHE_LEN:]

                # Causal Transformer: process sequence with causal masking
                seq = torch.stack(self._cache[key], dim=0).squeeze(1)  # [T, gru_hidden]
                T = seq.size(0)
                seq = seq.unsqueeze(0)  # [1, T, gru_hidden]
                causal_mask = self._get_causal_mask(T, seq.device)
                encoded = self.transformer(seq, mask=causal_mask)  # [1, T, gru_hidden]
                # Use the last position's output (causal = only sees past)
                last_out = encoded[:, -1, :]  # [1, gru_hidden]

                # Per-component heads
                comp_logits = torch.cat([
                    head(last_out) for head in self.output_heads
                ], dim=-1)  # [1, 11]
                # psr_confidence: max sigmoid per component group (Item 32)
                confidence = torch.sigmoid(comp_logits).max(dim=-1, keepdim=True)[0]  # [1, 1]
                comp_out = torch.cat([comp_logits, confidence], dim=-1)  # [1, 12]
                outputs.append(comp_out)

            return torch.cat(outputs, dim=0)  # [B, 12]

        # Non-cached mode: single frame / batch without caching
        # For single frame (T=1): causal mask allows attending to position 0 only
        feat_seq = frame_feat.unsqueeze(1)  # [B, 1, gru_hidden]
        T = feat_seq.size(1)
        causal_mask = self._get_causal_mask(T, feat_seq.device)
        encoded = self.transformer(feat_seq, mask=causal_mask)  # [B, 1, gru_hidden]
        # For single frame, use the last position output
        last_out = encoded.squeeze(1)  # [B, gru_hidden]

        # Per-component heads
        logits = torch.cat([
            head(last_out) for head in self.output_heads
        ], dim=-1)  # [B, 11]
        # psr_confidence: per-frame certainty = max sigmoid across 11 components (Item 32)
        confidence = torch.sigmoid(logits).max(dim=-1, keepdim=True)[0]  # [B, 1]
        return torch.cat([logits, confidence], dim=-1)  # [B, 12]

    def reset_sequence(self, video_id: str, camera_view: str = 'default'):
        key = (str(video_id), str(camera_view))
        self._cache.pop(key, None)

    def reset_all(self):
        self._cache.clear()


# ===========================================================================
# Full Multi-Task POPW Model
# ===========================================================================
class POPWMultiTaskModel(nn.Module):
    """
    POPW: Pose-Conditioned Multi-Task Architecture for IndustReal.

    Supports two backbones (Doc 01 D.1):
      - ResNet-50: C2=256, C3=512, C4=1024, C5=2048
      - ConvNeXt-Tiny: C2=96, C3=192, C4=384, C5=768

    Doc 01 E: HeadPoseFiLM -- second-stage FiLM from 9-DoF head pose.
    Doc 2 A.1: Optional VideoMAE V2 stream for activity recognition.

    Neck: FPN (P3-P7, 256ch)
    Heads:
      - Detection: RetinaNet-style (24 ASD classes, 9 anchors per location)
      - Pose: ConvTranspose2d + GroupNorm + ReLU -> heatmaps + soft-argmax -> keypoints + confidence
      - PoseFiLM: keypoints + confidence -> gamma/beta on C5
      - HeadPoseFiLM: 9-DoF head pose -> second-stage gamma/beta on C5_mod (Doc 01 E)
      - Activity: det_conf(24) + GAP(C5_mod_2)(C5) + GAP(P4)(256) -> concat -> proj -> FeatureBank -> TCN -> 2xViT -> CLS -> FC(74)
                   + optional VideoMAE V2 stream fusion (Doc 2 A.1)
    """
    def __init__(
        self,
        pretrained: bool = True,
        backbone_type: str = 'convnext_tiny',  # [FIX #8 LOW] Paper mandates ConvNeXt-Tiny, not ResNet-50
        use_headpose_film: bool = True,
        use_hand_film: bool = True,  # [FIX] Hand-FiLM conditional — enables ablation (paper: always on)
        use_videomae: bool = False,
        train_pose: bool = True,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.use_headpose_film = use_headpose_film
        self.use_hand_film = use_hand_film
        self.use_videomae = use_videomae
        self.train_pose = train_pose

        # === Backbone (Doc 01 D.1) ===
        self.backbone = build_backbone(backbone_type, pretrained=pretrained)

        # Channel dimensions by backbone type
        if backbone_type == 'convnext_tiny':
            c2_ch, c3_ch, c4_ch, c5_ch = 96, 192, 384, 768
            fpn_in_channels = [c3_ch, c4_ch, c5_ch]
        else:  # resnet50
            c2_ch, c3_ch, c4_ch, c5_ch = 256, 512, 1024, 2048
            fpn_in_channels = [c3_ch, c4_ch, c5_ch]

        self.c2_channels = c2_ch
        self.c3_channels = c3_ch
        self.c4_channels = c4_ch
        self.c5_channels = c5_ch

        # === FPN Neck ===
        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=256)

        # === Detection Head ===
        self.detection_head = DetectionHead(in_channels=256, num_classes=C.NUM_DET_CLASSES)
        self.anchor_gen = AnchorGenerator()

        # Pose head -- paper tau=0.07 (soft-argmax temperature)
        self.pose_head = PoseHead(in_channels=256, num_keypoints=C.NUM_KEYPOINTS, temperature=0.07)

        # === PoseFiLM (keypoint-conditioned, hand-keypoint FiLM) ===
        # [FIX] USE_HAND_FILM now wired: conditional instantiation enables ablation.
        # Paper default: True (always on). Set False to measure PoseFiLM contribution.
        if use_hand_film:
            self.pose_film = PoseFiLMModule(
                num_keypoints=C.NUM_KEYPOINTS,
                c5_channels=c5_ch,
                hidden_channels=512,
            )

        # === HeadPoseFiLM (Doc 01 E) ===
        if use_headpose_film:
            self.headpose_film = HeadPoseFiLMModule(
                c5_channels=c5_ch,
                hidden_channels=256,
            )

        # === Head Pose Head ===
        # [OPUS v5 AUDIT] Geometry-aware head pose: replace raw 9-number MSE MLP
        # with 6D continuous rotation (Zhou et al. CVPR 2019) + geodesic loss.
        # Expected: 10-25° MAE vs 60-70° from raw MSE. No baseline exists → free win.
        if getattr(C, 'USE_GEO_HEAD_POSE', False):
            from src.models.head_pose_geo import GeometryAwareHeadPose
            self.head_pose_head = GeometryAwareHeadPose(
                in_channels_c4=c4_ch,
                in_channels_c5=c5_ch,
                hidden_dim=512,
            )
        else:
            self.head_pose_head = HeadPoseHead(
                c4_channels=c4_ch,
                c5_channels=c5_ch,
                hidden_dim=128,
            )

        # === Activity Head ===
        self.activity_head = ActivityHead(
            c5_channels=c5_ch,
            p4_channels=256,
            det_conf_size=C.NUM_DET_CLASSES,
            embed_dim=512,
            num_classes=C.NUM_CLASSES_ACT,
            dropout=0.1,  # [FIX #3 HIGH] attn_dropout=0.1 per paper
            window_size=16,
            use_vit=True,
            use_videomae=use_videomae,
        )

        # === PSR Head ===
        self.psr_head = PSRHead(
            in_channels=256,
            hidden_dim=128,
            num_components=C.NUM_PSR_COMPONENTS,
            dropout=0.2,
        )

        # === Feature Bank ===
        self.feature_bank = FeatureBank(embed_dim=512, window_size=16)

        # === VideoMAE Stream (Doc 2 A.1) ===
        if use_videomae:
            self.videomae_stream = VideoMAEStream()

    @staticmethod
    def _decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Decode anchor deltas to bounding boxes."""
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
        if hasattr(self.backbone, '_freeze_bn'):
            if mode:
                self.backbone._freeze_bn()
        return self

    def forward(
        self,
        images: torch.Tensor,
        video_ids: Optional[List[str]] = None,
        clip_rgb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            images: [B, 3, H, W] -- current frame
            video_ids: list of video IDs for Feature Bank (optional)
            clip_rgb: [B, T, 3, 224, 224] optional clip for VideoMAE stream (Doc 2 A.1)
        Returns:
            dict with all outputs
        """
        B = images.shape[0]

        # Handle [B, T, C, H, W] sequence input from PSR sequence mode
        # _prepare_images flattens it to [B*T, C, H, W]; detect via _seq_len attribute
        seq_len = getattr(self, '_seq_len', 1)
        BT = B
        if images.dim() == 5:
            BT = images.shape[0] * images.shape[1]
            images = images.reshape(BT, images.shape[2], images.shape[3], images.shape[4])

        c2, c3, c4, c5 = self.backbone(images)

        # NaN guard: sanitize backbone features before FPN. A single NaN in
        # c2-c5 infects all downstream heads. torch.where preserves gradient
        # flow: finite entries pass through, NaN entries are zeroed.
        def _sanitize(x, bound=100.0):
            if torch.isfinite(x).all():
                return x.clamp(-bound, bound)
            return torch.where(torch.isfinite(x), x.clamp(-bound, bound), torch.zeros_like(x))

        _c3, _c4, _c5 = _sanitize(c3), _sanitize(c4), _sanitize(c5)
        pyramid = self.fpn(_c3, _c4, _c5)

        # NaN guard for pyramid: torch.clamp alone does NOT replace NaN values.
        # A single NaN in any pyramid level propagates through all 5 heads.
        for _k in pyramid:
            if not torch.isfinite(pyramid[_k]).all():
                pyramid[_k] = torch.where(
                    torch.isfinite(pyramid[_k]),
                    pyramid[_k].clamp(-100.0, 100.0),
                    torch.zeros_like(pyramid[_k]),
                )
            else:
                pyramid[_k] = pyramid[_k].clamp(-100.0, 100.0)

        cls_preds, reg_preds = self.detection_head(pyramid)
        anchors = self.anchor_gen(pyramid)

        heatmaps, keypoints, pose_confidence = self.pose_head(pyramid['p3'])

        # Doc 2 ?C.3: IndustReal has no COCO keypoint annotations, but the detection
        # head outputs bounding boxes for assembly objects. Use these as pseudo-keypoints
        # to give PoseFiLM a meaningful spatial signal instead of zeroing it out.
        # Generate 17 COCO-style keypoints from the highest-confidence detection box.
        if not self.train_pose:
            with torch.no_grad():
                # cls_preds: [A, NUM_DET_CLASSES], anchors: [A, 4]
                # Find dominant detection per sample (top-1 by confidence across all classes)
                top_cls = cls_preds.argmax(dim=1)          # [A]  # noqa: F841 -- class index tracked but spatial max used for location
                top_conf = cls_preds.max(dim=1)[0]         # [A]
                # Group by sample (anchor assignment to batch is implicit via spatial location)
                # For each image in the batch, pick the highest-confident detection
                B_kp = c5.shape[0]
                H, W = c5.shape[2], c5.shape[3]
                scale_kp = torch.tensor([W, H], device=c5.device, dtype=c5.dtype)  # noqa: F841 -- spatial scale available for future keypoint scaling
                pseudo_kps = torch.zeros(B_kp, 17, 2, device=c5.device, dtype=c5.dtype)
                pseudo_conf = torch.zeros(B_kp, 17, device=c5.device, dtype=c5.dtype)
                for b in range(B_kp):
                    # Use image-wide max detection confidence as proxy for spatial location
                    # Since anchors are spatial, the highest-conf anchor's location ? object center
                    max_idx = top_conf.argmax().item()
                    cx_norm = (anchors[max_idx, 0] / C.IMG_WIDTH).clamp(0, 1)
                    cy_norm = (anchors[max_idx, 1] / C.IMG_HEIGHT).clamp(0, 1)
                    w_norm = ((anchors[max_idx, 2] - anchors[max_idx, 0]) / C.IMG_WIDTH).clamp(0.05, 1)
                    h_norm = ((anchors[max_idx, 3] - anchors[max_idx, 1]) / C.IMG_HEIGHT).clamp(0.05, 1)
                    x0, y0 = cx_norm - w_norm / 2, cy_norm - h_norm / 2  # noqa: F841 -- bbox corners used for keypoint placement
                    x1, y1 = cx_norm + w_norm / 2, cy_norm + h_norm / 2
                    # 17 COCO keypoints: 4 corners, 4 mid-edges, 9 body points (approximated from bbox)
                    # COCO order: nose=0, eyes(1,2), ears(3,4), shoulders(5,6), elbows(7,8),
                    #            wrists(9,10), hips(11,12), knees(13,14), ankles(15,16)
                    kps_b = torch.tensor([
                        [cx_norm, cy_norm],          # 0: nose (bbox center)
                        [cx_norm - w_norm*0.1, cy_norm - h_norm*0.15],  # 1: l_eye
                        [cx_norm + w_norm*0.1, cy_norm - h_norm*0.15],  # 2: r_eye
                        [cx_norm - w_norm*0.2, cy_norm - h_norm*0.1],   # 3: l_ear
                        [cx_norm + w_norm*0.2, cy_norm - h_norm*0.1],   # 4: r_ear
                        [cx_norm - w_norm*0.3, cy_norm + h_norm*0.1],   # 5: l_shoulder
                        [cx_norm + w_norm*0.3, cy_norm + h_norm*0.1],   # 6: r_shoulder
                        [cx_norm - w_norm*0.4, cy_norm + h_norm*0.35],  # 7: l_elbow
                        [cx_norm + w_norm*0.4, cy_norm + h_norm*0.35],  # 8: r_elbow
                        [cx_norm - w_norm*0.5, cy_norm + h_norm*0.55],   # 9: l_wrist
                        [cx_norm + w_norm*0.5, cy_norm + h_norm*0.55],   # 10: r_wrist
                        [cx_norm - w_norm*0.15, cy_norm + h_norm*0.5],  # 11: l_hip
                        [cx_norm + w_norm*0.15, cy_norm + h_norm*0.5],  # 12: r_hip
                        [cx_norm - w_norm*0.15, cy_norm + h_norm*0.75], # 13: l_knee
                        [cx_norm + w_norm*0.15, cy_norm + h_norm*0.75], # 14: r_knee
                        [cx_norm - w_norm*0.15, cy_norm + h_norm*0.95], # 15: l_ankle
                        [cx_norm + w_norm*0.15, cy_norm + h_norm*0.95], # 16: r_ankle
                    ], device=c5.device, dtype=c5.dtype)
                    pseudo_kps_b = kps_b.clone()
                    pseudo_kps_b[:, 0] = pseudo_kps_b[:, 0].clamp(0, 1)
                    pseudo_kps_b[:, 1] = pseudo_kps_b[:, 1].clamp(0, 1)
                    pseudo_kps[b] = pseudo_kps_b
                    # Confidence: high (0.8) for bbox-derived points, lower for extrapolated
                    conf_vals = torch.tensor([0.9, 0.7, 0.7, 0.5, 0.5, 0.8, 0.8,
                                              0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.4, 0.4, 0.3, 0.3],
                                             device=c5.device, dtype=c5.dtype)
                    pseudo_conf[b] = conf_vals
                keypoints = pseudo_kps
                pose_confidence = pseudo_conf

        # [FIX] pose_film conditional on use_hand_film flag (ablation support)
        if self.use_hand_film and hasattr(self, 'pose_film'):
            c5_mod = self.pose_film(c5, keypoints, pose_confidence)
        else:
            c5_mod = c5  # No hand-keypoint conditioning when USE_HAND_FILM=False

        with torch.no_grad():
            # [AUDIT FIX 2026-06-11 / RC-19, P7] sigmoid-bound the detection
            # confidence. Raw max-logits are unbounded: a saturated det head
            # injected O(10-100), near-constant values into the activity-head
            # input (measured L2 243.39 +/- 0.001 across frames), drowning the
            # GAP features. max(sigmoid(x)) == sigmoid(max(x)) (monotonic), so
            # ranking semantics are unchanged; the scale is now [0, 1].
            det_conf = torch.sigmoid(cls_preds.max(dim=1)[0])
        if C.ZERO_DET_CONF_FOR_RECOVERY:
            det_conf = torch.zeros_like(det_conf)

        # PSR Head: pass full temporal sequence for proper Causal Transformer engagement
        # Pyramid dicts have per-level features [BT, C, H, W]. Reshape each to [B, T, C, H, W].
        B_main = BT // seq_len if seq_len > 1 else B
        T_main = seq_len if seq_len > 1 else 1
        if T_main > 1 and B_main * T_main == BT:
            pyramid_seq = {}
            for k, v in pyramid.items():
                pyramid_seq[k] = v.reshape(B_main, T_main, *v.shape[1:])
            # Build per-frame features [B, T, hidden] for PSR head
            frame_feats = []
            for t in range(T_main):
                p3_t = pyramid_seq['p3'][:, t]  # [B, C, H, W]
                p4_t = pyramid_seq['p4'][:, t]
                p5_t = pyramid_seq['p5'][:, t]
                p3_gap = self.psr_head.gap_p3(p3_t).flatten(1)
                p4_gap = self.psr_head.gap_p4(p4_t).flatten(1)
                p5_gap = self.psr_head.gap_p5(p5_t).flatten(1)
                fused = torch.cat([p3_gap, p4_gap, p5_gap], dim=1)
                frame_feats.append(self.psr_head.per_frame_mlp(fused))  # list of [B, hidden]
            frame_feat_seq = torch.stack(frame_feats, dim=1)  # [B, T, hidden]
            # Causal Transformer on full sequence [B, T, hidden]
            seq = frame_feat_seq  # [B, T, hidden]
            T = seq.size(1)
            # Causal mask: [T, T] — PyTorch broadcasts this identically across all batch items
            causal_mask = torch.triu(torch.ones(T, T, device=seq.device), diagonal=1).bool()
            # batch_first=True, so [B, T, hidden]
            encoded = self.psr_head.transformer(seq, mask=causal_mask)  # [B, T, hidden]
            # [AUDIT FIX 2026-06-11 — FIX-4 reapplied] The old code took only
            # encoded[:, -1, :] and .expand()ed ONE prediction across all T
            # frames: the per-frame focal loss then compared one prediction to
            # T different labels (optimum = window-average label = constant
            # output collapse), and the temporal-smooth loss had identically
            # zero gradient (differences of the same expanded tensor). Causal
            # masking already guarantees position t only sees frames <= t, so
            # every position is a valid (and supervised) per-frame prediction.
            enc_flat = encoded.reshape(B_main * T_main, -1)  # [BT, hidden]
            psr_full = torch.cat([
                head(enc_flat) for head in self.psr_head.output_heads
            ], dim=-1)  # [BT, 11]
            confidence = torch.sigmoid(psr_full).max(dim=-1, keepdim=True)[0]  # [BT, 1]
            psr_logits = torch.cat([psr_full, confidence], dim=-1)  # [BT, 12]
            psr_confidence = psr_logits[..., 11:]  # [BT, 1]
        else:
            psr_full = self.psr_head(pyramid, video_ids=video_ids)  # [B, 12]
            psr_logits = psr_full[..., :11]  # [B, 11]
            psr_confidence = psr_full[..., 11:]  # [B, 1]

        # Head pose: computed when TRAIN_HEAD_POSE=True (paper: disabled for IKEA ASM)
        # ALSO compute during eval mode (self.training=False) so evaluate.py gets tensors.
        # During training: respect train_pose flag (may be stage-gated to False in early epochs).
        # During eval: always compute head_pose so evaluate.py doesn't crash on None.
        if self.train_pose or not self.training:
            head_pose = self.head_pose_head(c4, c5)
            # [GAP-C] GeometryAwareHeadPose returns (rot6d, rot_matrix, position) tuple.
            # Convert to [B,9] for downstream compatibility (FiLM, eval, loss).
            if isinstance(head_pose, tuple):
                _rot6d, _rot_mat, _pos = head_pose
                # Reconstruct [B,9] = forward(3) + position(3) + up(3)
                _forward = _rot_mat[:, :, 0]                  # first column = forward dir
                _up = _rot_mat[:, :, 2]                        # third column = up dir
                head_pose = torch.cat([_forward, _pos, _up], dim=1)  # [B, 9]
            if self.use_headpose_film and hasattr(self, 'headpose_film'):
                c5_mod = self.headpose_film(c5_mod, head_pose.detach())  # stop_grad per paper ?HeadPoseFiLM
        else:
            head_pose = None

        activity_proj = torch.cat([
            det_conf,
            F.adaptive_avg_pool2d(c5_mod, 1).flatten(1),
            F.adaptive_avg_pool2d(pyramid['p4'], 1).flatten(1),
        ], dim=1)

        proj_feat = self.activity_head.proj_features(activity_proj)

        # NaN guard: sanitize proj_feat before FeatureBank update to prevent
        # permanent contamination of the temporal ring buffer.
        if not torch.isfinite(proj_feat).all():
            proj_feat = torch.zeros_like(proj_feat)

        bank_output = self.feature_bank(proj_feat, video_ids, None)

        videomae_feat = None
        if self.use_videomae and clip_rgb is not None and hasattr(self, 'videomae_stream'):
            videomae_feat = self.videomae_stream(clip_rgb)

        act_logits = self.activity_head(
            proj_feat=proj_feat,
            temporal_bank=bank_output,
            videomae_feat=videomae_feat,
        )

        # [CRITICAL FIX 2026-06-12] NaN guard on ALL head outputs before return.
        # A single NaN in any output contaminates the loss computation and, via
        # _safe→constant replacement, eventually disconnects the autograd graph.
        # Check each tensor and replace NaN with zeros.
        for _out_name, _out_val in [
            ('cls_preds', cls_preds), ('reg_preds', reg_preds),
            ('heatmaps', heatmaps), ('keypoints', keypoints),
            ('pose_confidence', pose_confidence),
            ('head_pose', head_pose), ('act_logits', act_logits),
            ('psr_logits', psr_logits),
        ]:
            if _out_val is not None and not torch.isfinite(_out_val).all():
                if not self.training:
                    logger.warning(f'[MODEL_NAN] {_out_name} has NaN (eval mode)')
                # Zero the output: preserves shape/dtype, disconnects NaN from loss
                _out_val = torch.zeros_like(_out_val)
                # Reassign to local variable
                if _out_name == 'cls_preds': cls_preds = _out_val
                elif _out_name == 'reg_preds': reg_preds = _out_val
                elif _out_name == 'heatmaps': heatmaps = _out_val
                elif _out_name == 'keypoints': keypoints = _out_val
                elif _out_name == 'pose_confidence': pose_confidence = _out_val
                elif _out_name == 'head_pose': head_pose = _out_val
                elif _out_name == 'act_logits': act_logits = _out_val
                elif _out_name == 'psr_logits': psr_logits = _out_val

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
            'psr_logits': psr_logits[..., :11],  # [BT, 11] for loss compatibility
            'psr_confidence': psr_confidence if not self.training else None,  # Item 32
            'temporal_features': bank_output,
            'c5_raw': c5,
        }


def count_parameters(model: POPWMultiTaskModel) -> Dict[str, int]:
    components = {
        'backbone': [model.backbone],
        'fpn': [model.fpn],
        'detection': [model.detection_head],
        'pose_head': [model.pose_head],
        'pose_film': [model.pose_film],
        'headpose_film': [model.headpose_film] if hasattr(model, 'headpose_film') else [],
        'activity_head': [model.activity_head],
        'psr_head': [model.psr_head],
        'feature_bank': [model.feature_bank],
        'videomae_stream': [model.videomae_stream] if hasattr(model, 'videomae_stream') else [],
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


# ===========================================================================
# EMA -- Exponential Moving Average of model weights
# ===========================================================================

class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains shadow weights: ema_param = decay * ema_param + (1 - decay) * current_param
    Use update() every step after optimizer, and get_ema() / get_current() for eval/train.

    Standard decay = 0.999 for image models, 0.9999 for very large models.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        for name, buffer in self.model.named_buffers():
            self.shadow[name] = buffer.data.clone().detach()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"EMA: shadow missing {name}"
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()
        for name, buffer in self.model.named_buffers():
            if name in self.shadow:
                self.shadow[name] = buffer.data.clone()

    def set_decay(self, decay: float):
        self.decay = decay

    def get_ema(self):
        """Apply EMA weights to model for evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
        for name, buffer in self.model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buffer.data.clone()
                buffer.data.copy_(self.shadow[name].to(buffer.device))
        return self.model

    def restore(self):
        """Restore original (non-EMA) weights after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup.pop(name).to(param.device))
        for name, buffer in self.model.named_buffers():
            if name in self.backup:
                buffer.data.copy_(self.backup.pop(name).to(buffer.device))
