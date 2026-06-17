"""
Tier 2.5 — ROI-Centric Assembly State Detector
===============================================
Replaces the dense 24-class RetinaNet anchors (173K locations) with a two-stage design:

(a) Class-Agnostic Localizer — single-class anchor-free head (FCOS/CenterNet-style,
    P5-P7 only) that finds "the assembly object" (easy, large object: 146-594px).

(b) ROI-Align State Classifier — crops high-res features (from P3 or raw image at
    224²) around the localized object and classifies into 24 assembly states.

This converts the impossible problem (dense fine-grained detection across 173K anchors)
into two easy ones:
  - Localization: one foreground class on large objects
  - Classification: high-res 24-way state recognition

The state classifier output is exactly what PSR needs (Tier 2.7).

Design:
  Input [B, 3, 720, 1280] → Backbone → FPN
  → CenternessHead [B, H_big, W_big, 1] + BoxHead [B, H_big, W_big, 4]  (P5-P7, large strides)
  → Top-K boxes → ROIAlign(p3_features, boxes) → [B, K, 256, 14, 14]
  → StateClassifier [B, K, 24]

Total anchors replaced: 173K → 0 (anchor-free)
Detection complexity: 24-class dense → 1-class localizer + 24-way RoI classifier
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src import config as C


# ============================================================================
# Anchor-Free Localizer (FCOS-style)
# ============================================================================
class AnchorFreeLocalizer(nn.Module):
    """Single-class FCOS/CenterNet-style localizer on large-stride pyramid levels.

    Uses only P5-P7 (stride 32-128) since assembly objects are large (146-594px).
    Predicts: centerness [B, H, W, 1], box [B, H, W, 4] per location.
    Total predictions: ~3K locations at P5-P7 (vs 173K RetinaNet anchors).
    """

    def __init__(self, in_channels: int = 256, feat_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.levels = ['p5', 'p6', 'p7']  # stride 32, 64, 128

        # Shared conv tower (per-level heads)
        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
        )

        self.reg_tower = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
        )

        # Per-level scales (FCOS learnable)
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in self.levels
        ])

        self.cls_head = nn.Conv2d(feat_channels, 1, 3, padding=1)
        self.reg_head = nn.Conv2d(feat_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(feat_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_tower, self.reg_tower]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Cls head: pi=0.01 prior
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))

        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.constant_(self.reg_head.bias, 0)

        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.constant_(self.centerness.bias, 0)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], ...]:
        """Returns: cls_preds, reg_preds, centers, locations per level."""
        cls_preds = []
        reg_preds = []
        centernesses = []
        locations = []

        for i, level in enumerate(self.levels):
            feat = pyramid[level]  # [B, 256, H, W]
            B, _, H, W = feat.shape

            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_out = self.cls_head(cls_feat)  # [B, 1, H, W]
            reg_out = self.scales[i] * F.relu(self.reg_head(reg_feat))  # [B, 4, H, W]
            center = self.centerness(cls_feat)  # [B, 1, H, W]

            cls_preds.append(cls_out.reshape(B, 1, -1))
            reg_preds.append(reg_out.reshape(B, 4, -1))
            centernesses.append(center.reshape(B, 1, -1))

            # Compute absolute locations for stride
            stride = 2 ** (5 + i)  # 32, 64, 128
            ys = torch.arange(H, device=feat.device).float() * stride + stride / 2
            xs = torch.arange(W, device=feat.device).float() * stride + stride / 2
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)
            locations.append(grid.expand(B, 2, -1))

        cls_preds = torch.cat(cls_preds, dim=-1)  # [B, 1, N_total]
        reg_preds = torch.cat(reg_preds, dim=-1)  # [B, 4, N_total]
        centernesses = torch.cat(centernesses, dim=-1)  # [B, 1, N_total]
        locations = torch.cat(locations, dim=-1)  # [B, 2, N_total]

        return cls_preds, reg_preds, centernesses, locations

    def decode_boxes(self, reg_preds: torch.Tensor, locations: torch.Tensor
                     ) -> torch.Tensor:
        """Decode FCOS (l,t,r,b) predictions to xyxy boxes."""
        # reg_preds: [B, 4, N] in (l, t, r, b) format
        # locations: [B, 2, N] in (cx, cy) format
        l = reg_preds[:, 0:1, :]
        t = reg_preds[:, 1:2, :]
        r = reg_preds[:, 2:3, :]
        b = reg_preds[:, 3:4, :]
        cx = locations[:, 0:1, :]
        cy = locations[:, 1:2, :]
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        return torch.cat([x1, y1, x2, y2], dim=1)  # [B, 4, N]


# ============================================================================
# ROI State Classifier
# ============================================================================
class ROIStateClassifier(nn.Module):
    """Crops high-res P3 features around localized boxes and classifies assembly states.

    Input: P3 features [B, 256, H/8, W/8] + bboxes [B, K, 4] (xyxy, pixel coords)
    Output: state logits [B, K, 24] — 24-way assembly state classification

    Uses ROI-Align (pool_size=14) for each box, then a small conv+fc classifier.
    """

    def __init__(self, in_channels: int = 256, num_states: int = 24,
                 pool_size: int = 14, hidden_dim: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.num_states = num_states
        self.pool_size = pool_size

        # RoI feature extractor
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # State classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_states),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.roi_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p3_features: torch.Tensor, boxes: List[torch.Tensor]
                ) -> torch.Tensor:
        """Forward with RoI-Align.

        Args:
            p3_features: [B, 256, H/8, W/8] P3 FPN features
            boxes: list of [K_i, 4] tensors (pixel xyxy, one per image)

        Returns:
            state_logits: [B, K_max, 24] padded with -inf for shorter K
        """
        B = p3_features.shape[0]
        H_p3, W_p3 = p3_features.shape[2], p3_features.shape[3]
        stride = 8  # P3 stride

        # Normalize boxes to [0,1] in P3 feature space
        results = []
        for i in range(B):
            if boxes[i].shape[0] == 0:
                results.append(torch.zeros(0, self.num_states,
                                           device=p3_features.device))
                continue

            # Scale boxes from pixel coords to P3 feature coords [0,1]
            boxes_norm = boxes[i].clone()
            boxes_norm[:, [0, 2]] = boxes_norm[:, [0, 2]] / (W_p3 * stride)
            boxes_norm[:, [1, 3]] = boxes_norm[:, [1, 3]] / (H_p3 * stride)

            # ROI-Align
            roi_feats = torchvision.ops.roi_align(
                p3_features[i:i+1],  # [1, 256, H, W]
                [boxes_norm],
                output_size=(self.pool_size, self.pool_size),
                spatial_scale=1.0,
                aligned=True,
            )  # [K, 256, 14, 14]

            # Conv + classify
            feats = self.roi_conv(roi_feats).squeeze(-1).squeeze(-1)  # [K, 256]
            logits = self.classifier(feats)  # [K, 24]
            results.append(logits)

        # Pad to uniform K
        max_k = max(r.shape[0] for r in results)
        padded = []
        for r in results:
            if r.shape[0] < max_k:
                pad = torch.full((max_k - r.shape[0], self.num_states),
                                 float('-inf'), device=r.device)
                r = torch.cat([r, pad], dim=0)
            padded.append(r)

        return torch.stack(padded, dim=0)  # [B, max_k, 24]


# ============================================================================
# Full ROI Detector (Localizer + Classifier)
# ============================================================================
class ROIDetector(nn.Module):
    """Complete ROI-centric assembly state detector.

    Replaces the dense RetinaNet DetectionHead. See module docstring for rationale.
    """

    def __init__(self, num_states: int = 24, top_k: int = 10,
                 score_thresh: float = 0.05, nms_thresh: float = 0.5):
        super().__init__()
        self.num_states = num_states
        self.top_k = top_k
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.localizer = AnchorFreeLocalizer(in_channels=256)
        self.classifier = ROIStateClassifier(in_channels=256, num_states=num_states)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full detection pass.

        Returns dict with:
            'cls_preds': [B, max_k, 24] — state classification logits
            'reg_preds': [B, max_k, 4] — box regression (l,t,r,b)
            'boxes': [B, max_k, 4] — decoded xyxy boxes
            'scores': [B, max_k] — centerness * cls_score
            'locations': [B, 2, N] — all grid cell locations
        """
        # Step 1: Localize objects
        cls_raw, reg_raw, centers, locations = self.localizer(pyramid)

        # Step 2: Decode top-K boxes
        scores = centers.squeeze(1) * cls_raw.sigmoid().squeeze(1)  # [B, N]
        boxes_raw = self.localizer.decode_boxes(reg_raw, locations)  # [B, 4, N]

        B = scores.shape[0]
        N = scores.shape[1]

        # Get top-K per image
        top_scores, top_indices = scores.topk(min(self.top_k * 3, N), dim=1)

        top_boxes_per_img = []
        for i in range(B):
            k = min(self.top_k * 3, N)
            boxes_i = boxes_raw[i, :, top_indices[i]]  # [4, k]
            scores_i = top_scores[i]  # [k]

            # Filter by score
            mask = scores_i > self.score_thresh
            boxes_i = boxes_i[:, mask]
            scores_i = scores_i[mask]

            if boxes_i.shape[1] == 0:
                top_boxes_per_img.append(torch.zeros(0, 4, device=boxes_raw.device))
                continue

            # NMS
            keep = torchvision.ops.nms(
                boxes_i.permute(1, 0).contiguous(),  # [k', 4]
                scores_i,
                self.nms_thresh,
            )
            keep = keep[:self.top_k]
            top_boxes_per_img.append(boxes_i[:, keep].permute(1, 0))  # [k'', 4]

        # Step 3: ROI-Align and classify
        p3 = pyramid['p3']
        state_logits = self.classifier(p3, top_boxes_per_img)  # [B, max_k, 24]

        # Padded boxes for uniform output
        max_k = state_logits.shape[1]
        padded_boxes = []
        for bx in top_boxes_per_img:
            if bx.shape[0] < max_k:
                pad = torch.zeros(max_k - bx.shape[0], 4, device=bx.device)
                bx = torch.cat([bx, pad], dim=0)
            elif bx.shape[0] > max_k:
                bx = bx[:max_k]
            padded_boxes.append(bx)
        boxes_all = torch.stack(padded_boxes, dim=0)  # [B, max_k, 4]

        return {
            'cls_preds': state_logits,  # [B, max_k, 24]
            'reg_preds': reg_raw,  # [B, 4, N] — raw FCOS reg
            'boxes': boxes_all,  # [B, max_k, 4]
            'scores': scores,  # [B, N]
            'locations': locations,  # [B, 2, N]
        }


# Late import to avoid circular dependency
try:
    import torchvision
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False
    # Fallback RoI-Align using grid_sample
    logger = __import__('logging').getLogger(__name__)
    logger.warning("torchvision not available — ROIStateClassifier needs torchvision for roi_align")
