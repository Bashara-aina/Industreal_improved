import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Multi-Task IndustReal Model
=============================
ResNet-50-FPN backbone with four task-specific heads:
  1. DetectionHead  : RetinaNet-style anchor-based (24 ASD assembly states)
  2. HeadPoseHead   : 9-DoF head pose regression from C5 GAP (forward + pos + up)
  3. ActivityHead   : GAP + FC classifier (74 AR classes)
  4. PSRHead        : Multi-label binary classification (11 component completion)

No FiLM, no pose keypoints, no multi-camera.
Single egocentric RGB input.

Author: Bashara
Date: April 2026
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import config as C


# =============================================================================
# Backbone / Neck
# =============================================================================
class FPN(nn.Module):
    """Standard FPN. Takes C3, C4, C5 -> P3, P4, P5, P6, P7. All 256 channels."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.smooth_p3  = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4  = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5  = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.p6_conv    = nn.Conv2d(in_channels_list[2], out_channels, 3, stride=2, padding=1)
        self.p7_conv    = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c3, c4, c5):
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)
        p6 = self.p6_conv(c5)
        p7 = self.p7_conv(F.relu(p6))
        return {'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, 'p7': p7}


class AnchorGenerator(nn.Module):
    """
    Generate anchors for RetinaNet. 3 ratios x 3 scales = 9 per location.

    Loop order: ratios outer, scales inner (matches RetinaNet paper).
    """

    def __init__(
        self,
        sizes:   Tuple[int, ...] = (32, 64, 128, 256, 512),
        ratios:  Tuple[float, ...] = (0.5, 1.0, 2.0),
        scales:  Tuple[float, ...] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
    ):
        super().__init__()
        self.sizes   = sizes
        self.ratios  = ratios
        self.scales  = scales
        self.num_anchors = len(ratios) * len(scales)  # 9

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(iter(feature_maps.values())).device
        all_anchors = []
        fpn_keys = ['p3', 'p4', 'p5', 'p6', 'p7']
        strides   = [8,   16,   32,   64,   128]

        for level_idx, (key, stride) in enumerate(zip(fpn_keys, strides)):
            feat = feature_maps[key]
            h, w = feat.shape[2:]
            base_size = self.sizes[level_idx]

            cell_anchors = []
            for ratio in self.ratios:
                for scale in self.scales:
                    s  = base_size * scale
                    aw = s * math.sqrt(ratio)
                    ah = s / math.sqrt(ratio)
                    cell_anchors.append([-aw / 2, -ah / 2, aw / 2, ah / 2])
            cell_anchors = torch.tensor(
                cell_anchors, device=device, dtype=torch.float32
            )  # [9, 4]

            shifts_x = (torch.arange(w, device=device) + 0.5) * stride
            shifts_y = (torch.arange(h, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shifts = torch.stack([
                shift_x.flatten(), shift_y.flatten(),
                shift_x.flatten(), shift_y.flatten(),
            ], dim=1)  # [H*W, 4]

            anchors = shifts.unsqueeze(1) + cell_anchors.unsqueeze(0)  # [H*W, 9, 4]
            all_anchors.append(anchors.reshape(-1, 4))

        return torch.cat(all_anchors, dim=0)


# =============================================================================
# Task Heads
# =============================================================================
class DetectionHead(nn.Module):
    """RetinaNet classification + regression subnets, shared across FPN levels."""

    def __init__(self, in_channels: int = 256,
                 num_classes: int = C.NUM_DET_CLASSES,
                 num_anchors: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        cls_layers = []
        for _ in range(4):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(True),
            ])
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_score  = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)

        reg_layers = []
        for _ in range(4):
            reg_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(True),
            ])
        self.reg_subnet = nn.Sequential(*reg_layers)
        self.reg_pred   = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_subnet, self.reg_subnet]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
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
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(
                B, H * W * self.num_anchors, self.num_classes
            )
            cls_all.append(cls_out)
            reg_out = self.reg_pred(self.reg_subnet(feat))
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(
                B, H * W * self.num_anchors, 4
            )
            reg_all.append(reg_out)
        return torch.cat(cls_all, dim=1), torch.cat(reg_all, dim=1)


class HeadPoseHead(nn.Module):
    """
    9-DoF head pose regression from C5 feature map.

    Output: [B, 9] = forward_vector[3] + position[3] + up_vector[3]
    Architecture: GAP -> FC(2048->256) -> FC(256->9)
    """

    def __init__(self, in_channels: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, C.NUM_HEAD_POSE_DOF),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, c5: torch.Tensor) -> torch.Tensor:
        x = self.gap(c5).flatten(1)  # [B, 2048]
        return self.fc(x)  # [B, 9]


class ActivityHead(nn.Module):
    """
    GAP on C5 + P4 features -> FC classifier (74 AR classes).

    Always requires p4 (fuse_p4=True). C5-only mode not used in IndustReal.
    """

    def __init__(
        self,
        c5_channels:  int = 2048,
        p4_channels:  int = 256,
        hidden_dim:   int = 512,
        num_classes:  int = C.NUM_CLASSES_ACT,
        dropout:      float = 0.3,
        fuse_p4:      bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.fuse_p4 = fuse_p4
        self.use_residual = use_residual
        in_features = (c5_channels + p4_channels) if fuse_p4 else c5_channels

        self.pool_c5 = nn.AdaptiveAvgPool2d(1)
        self.pool_p4 = nn.AdaptiveAvgPool2d(1) if fuse_p4 else None
        if self.use_residual:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GroupNorm(1, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout),
            )
            self.bottleneck = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GroupNorm(1, hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(dropout),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.GroupNorm(1, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout),
            )
            self.residual_projection = nn.Linear(in_features, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, c5: torch.Tensor,
                p4: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.fuse_p4:
            if p4 is None:
                raise ValueError(
                    'ActivityHead requires p4 when fuse_p4=True. '
                    'Pass pyramid["p4"].'
                )
            x = torch.cat([self.pool_c5(c5), self.pool_p4(p4)], dim=1)
        else:
            x = self.pool_c5(c5)

        x = x.flatten(1)
        if self.use_residual:
            residual = self.residual_projection(x)
            x = self.fc1(x)
            x = self.bottleneck(x)
            x = self.fc2(x)
            x = x + residual
            return self.classifier(x)
        return self.fc(x)


class PSRHead(nn.Module):
    """
    Per-frame procedure step completion (PSR) head.

    Multi-label binary classification for 11 assembly components.
    Input: C5 feature map [B, 2048, H, W]
    Architecture: GAP -> FC(2048->256) -> FC(256->11) -> sigmoid
    Output: [B, 11] multi-label logits (use BCEWithLogitsLoss or sigmoid+BCE)
    """

    def __init__(self, in_channels: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, C.NUM_PSR_COMPONENTS),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, c5: torch.Tensor) -> torch.Tensor:
        x = self.gap(c5).flatten(1)  # [B, 2048]
        return self.fc(x)  # [B, 11]


class MultiTaskIndustReal(nn.Module):
    """
    Unified multi-task model for IndustReal.
    Backbone: ResNet-50 (ImageNet pretrained, frozen BN)
    Neck    : FPN (P3-P7)
    Heads   : Detection (RetinaNet), HeadPose (9-DoF), Activity (74-class), PSR (11-component)
    No FiLM, no pose keypoints, single RGB camera.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # C2: 256ch,  stride 4
        self.layer2 = backbone.layer2   # C3: 512ch,  stride 8
        self.layer3 = backbone.layer3   # C4: 1024ch, stride 16
        self.layer4 = backbone.layer4   # C5: 2048ch, stride 32
        self._freeze_bn()

        self.fpn             = FPN([512, 1024, 2048], 256)
        self.detection_head  = DetectionHead(256, C.NUM_DET_CLASSES)
        self.head_pose_head  = HeadPoseHead(2048, 256)
        self.activity_head   = ActivityHead(2048, 256, 512, C.NUM_CLASSES_ACT,
                                            fuse_p4=True)
        self.psr_head        = PSRHead(2048, 256)
        self.anchor_gen      = AnchorGenerator()

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

    def _freeze_bn(self):
        for module in [
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4
        ]:
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._freeze_bn()
        return self

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x  = self.layer0(images)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        pyramid                  = self.fpn(c3, c4, c5)
        cls_preds, reg_preds     = self.detection_head(pyramid)
        anchors                  = self.anchor_gen(pyramid)

        head_pose = self.head_pose_head(c5)                       # [B, 9]
        psr_logits = self.psr_head(c5)                            # [B, 11]
        act_logits = self.activity_head(c5, pyramid['p4'])        # [B, 74]

        return {
            'cls_preds':   cls_preds,
            'reg_preds':   reg_preds,
            'anchors':     anchors,
            'head_pose':   head_pose,
            'psr_logits':  psr_logits,
            'act_logits':  act_logits,
        }


def count_parameters(model: MultiTaskIndustReal) -> Dict[str, int]:
    components = {
        'backbone':   [model.layer0, model.layer1, model.layer2,
                       model.layer3, model.layer4],
        'fpn':        [model.fpn],
        'detection':  [model.detection_head],
        'head_pose':   [model.head_pose_head],
        'activity':   [model.activity_head],
        'psr':        [model.psr_head],
    }
    result = {}
    total  = 0
    for name, modules in components.items():
        count = sum(
            p.numel() for m in modules for p in m.parameters()
            if p.requires_grad
        )
        result[name] = count
        total += count
    result['total_trainable'] = total
    result['total_all']       = sum(p.numel() for p in model.parameters())
    return result