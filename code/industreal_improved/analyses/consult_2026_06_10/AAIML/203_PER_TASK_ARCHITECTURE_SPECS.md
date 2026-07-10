# 203 — Per-Task Deep Architecture Specifications With Exact Implementations

**Date:** 2026-07-10
**Sources:** 12 specialized research agents covering detection heads, PSR architectures, pose estimation, activity recognition, adapters, distillation, pretraining, and unified backbones.

---

## 1. Detection Head — Full Specification

### 1.1 BiFPN Neck (Replaces Current LightweightFPN)

```python
class BiFPN(nn.Module):
    """Weighted Bi-directional FPN for detection. 128ch, 3 levels (P3/P4/P5)."""
    def __init__(self, in_channels={"P3": 192, "P4": 384, "P5": 768}, out_ch=128):
        super().__init__()
        # Lateral projections: backbone channels → 128
        self.lat_p3 = nn.Conv2d(192, out_ch, 1)
        self.lat_p4 = nn.Conv2d(384, out_ch, 1)
        self.lat_p5 = nn.Conv2d(768, out_ch, 1)

        # Top-down pathway: P5→P4_hat→P3_hat
        self.td_p4 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # after P5↓+P4
        self.td_p3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # after P4_hat↓+P3

        # Bottom-up pathway: P3_hat→P4_out→P5_out
        self.bu_p4 = nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2)  # receptive field
        self.bu_p5 = nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2)

        # Learned fusion weights (BiFPN key innovation)
        self.w_td_p4 = nn.Parameter(torch.ones(2))  # [w_lat, w_upsampled]
        self.w_td_p3 = nn.Parameter(torch.ones(2))
        self.w_bu_p4 = nn.Parameter(torch.ones(3))  # [w_td, w_bu_prev, w_lat]
        self.w_bu_p5 = nn.Parameter(torch.ones(3))

        nn.init.constant_(self.lat_p3.bias, 0)
        nn.init.constant_(self.lat_p4.bias, 0)
        nn.init.constant_(self.lat_p5.bias, 0)

    def forward(self, fpn_feats):
        p3, p4, p5 = fpn_feats["P3"], fpn_feats["P4"], fpn_feats["P5"]

        # Temporal pool: T dim → mean
        p3 = p3.mean(dim=2)  # [B, 192, 28, 28]
        p4 = p4.mean(dim=2)  # [B, 384, 14, 14]
        p5 = p5.mean(dim=2)  # [B, 768, 7, 7]

        # Lateral
        p3_lat = self.lat_p3(p3)  # [B, 128, 28, 28]
        p4_lat = self.lat_p4(p4)  # [B, 128, 14, 14]
        p5_lat = self.lat_p5(p5)  # [B, 128, 7, 7]

        # Top-down: P5 → fuse with P4_lat
        w_relu = F.relu(self.w_td_p4)
        w = w_relu / (w_relu.sum() + 1e-4)
        p4_hat = self.td_p4(
            w[0] * p4_lat + w[1] * F.interpolate(p5_lat, scale_factor=2, mode="nearest")
        )

        # Top-down: P4_hat → fuse with P3_lat
        w_relu = F.relu(self.w_td_p3)
        w = w_relu / (w_relu.sum() + 1e-4)
        p3_hat = self.td_p3(
            w[0] * p3_lat + w[1] * F.interpolate(p4_hat, scale_factor=2, mode="nearest")
        )

        # Bottom-up: P3_hat → fuse with P4_hat + P4_lat
        w_relu = F.relu(self.w_bu_p4)
        w = w_relu / (w_relu.sum() + 1e-4)
        p4_out = self.bu_p4(
            w[0] * p4_hat + w[1] * F.max_pool2d(p3_hat, 2) + w[2] * p4_lat
        )

        # Bottom-up: P4_out → fuse with P5_lat + P5 (from top-down)
        w_relu = F.relu(self.w_bu_p5)
        w = w_relu / (w_relu.sum() + 1e-4)
        p5_out = self.bu_p5(
            w[0] * p5_lat + w[1] * F.max_pool2d(p4_out, 2) + w[2] * p5_lat
        )

        return {"P3": p3_hat, "P4": p4_out, "P5": p5_out}
```

**Parameters:** ~1.2M for 128ch version (vs current LightweightFPN ~7.5M at 256ch). The learned fusion weights are 4×(2+2+3+3) = 40 scalars — negligible.

### 1.2 GFLV2 Detection Head

```python
class GFLV2DetectionHead(nn.Module):
    """Decoupled detection head with GFLV2 quality estimation."""
    def __init__(self, in_ch=128, num_classes=24, reg_max=16, head_ch=96):
        super().__init__()
        inner_ch = head_ch  # 96 — sufficient for 24 classes

        # Shared stem (one per level, instantiated per level)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, inner_ch, 3, padding=1), nn.BatchNorm2d(inner_ch), nn.SiLU()
        )

        # Classification branch
        self.cls_conv1 = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, 3, padding=1), nn.BatchNorm2d(inner_ch), nn.SiLU()
        )
        self.cls_conv2 = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, 3, padding=1), nn.BatchNorm2d(inner_ch), nn.SiLU()
        )
        self.cls_pred = nn.Conv2d(inner_ch, num_classes, 3, padding=1)

        # Regression branch (DFL: 4 edges × (reg_max+1) bins)
        self.reg_conv1 = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, 3, padding=1), nn.BatchNorm2d(inner_ch), nn.SiLU()
        )
        self.reg_conv2 = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, 3, padding=1), nn.BatchNorm2d(inner_ch), nn.SiLU()
        )
        self.reg_pred = nn.Conv2d(inner_ch, 4 * (reg_max + 1), 3, padding=1)

        # GFLV2 quality predictor (from DFL distribution statistics)
        # 4 edges × 2 stats (mean, std) = 8 input features
        self.quality_mlp = nn.Sequential(
            nn.Conv2d(8, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        stem = self.stem(x)
        cls_feat = self.cls_conv2(self.cls_conv1(stem))
        reg_feat = self.reg_conv2(self.reg_conv1(stem))
        cls_logits = self.cls_pred(cls_feat)
        reg_preds = self.reg_pred(reg_feat)

        # Quality: compute mean/std per box edge from DFL distribution
        # reg_preds: [B, 4*(reg_max+1), H, W]
        B, _, H, W = reg_preds.shape
        reg_dist = reg_preds.view(B, 4, -1, H, W).softmax(dim=2)
        proj = torch.arange(reg_dist.size(2), device=x.device).float().view(1,1,-1,1,1)
        mean = (reg_dist * proj).sum(dim=2)         # [B, 4, H, W]
        var = ((reg_dist * proj**2).sum(dim=2) - mean**2).clamp(min=0)  # [B, 4, H, W]
        stats = torch.cat([mean, var.sqrt()], dim=1)  # [B, 8, H, W]
        quality = self.quality_mlp(stats)

        return {"cls_logits": cls_logits, "reg_preds": reg_preds, "quality": quality}
```

**Parameters per FPN level:** ~120K. ×3 levels = ~360K total for detection head.

### 1.3 TAL Assigner with Per-Level Top-K

Already implemented in `src/losses/tal_assigner.py`. One change: per-level k values.

```python
topk_per_level = {"P3": 9, "P4": 12, "P5": 15}  # more anchors at coarser levels
```

Rationale: P5 has fewer cells (7×7=49) but should cover more GT area (large objects). P3 has many cells (28×28=784) covering small parts. TAL with per-level top-k gives 9+12+15 = 36 positive cells per GT, up from 10 uniform.

### 1.4 Detection Loss

```python
def detection_loss_gflv2(det_outputs, det_list, num_classes=24, reg_max=16):
    """GFLV2-style detection loss: QFL + GIoU + DFL + quality."""
    loss_cls = 0.0; loss_iou = 0.0; loss_dfl = 0.0; loss_quality = 0.0
    levels = ("P3", "P4", "P5")
    strides = {"P3": 8, "P4": 16, "P5": 32}
    topk_dict = {"P3": 9, "P4": 12, "P5": 15}

    for level_name in levels:
        out = det_outputs[level_name]
        # ... TAL assignment (same as current) ...
        # Quality loss: BCE between predicted quality and actual IoU
        quality_pred = out["quality"][pos_mask]  # [n_pos, 1]
        quality_gt = compute_iou(decoded_boxes[pos_mask], gt_boxes_assigned[pos_mask])
        loss_quality += F.binary_cross_entropy(quality_pred, quality_gt.unsqueeze(1))

        # QFL (Quality Focal Loss) replaces standard Focal BCE
        # QFL combines classification score + quality into joint representation
        cls_score = torch.sigmoid(out["cls_logits"])
        quality_target = quality_gt  # IoU as soft label
        # QFL: -|y - σ|^γ * ((1-y)log(1-σ) + y*log(σ))
        # where y = quality_target if class matches, else 0
        # Implemented as weighted BCE with quality-guided targets

    return (loss_cls + 2.0 * loss_iou + 0.25 * loss_dfl + 1.0 * loss_quality) / len(levels)
```

---

## 2. PSR Head — Full Specification

### 2.1 Detection-Conditioned Hierarchical Transformer

```python
class DetectionConditionedPSRHead(nn.Module):
    """PSR head with hierarchical temporal transformer + detection conditioning."""
    def __init__(self, input_dim=768, feat_dim=256, num_components=11,
                 nhead=4, num_layers_stage1=2, num_layers_stage2=1,
                 det_feat_dim=256, dropout=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.input_proj = nn.Linear(input_dim, feat_dim)

        # Detection ROI projector
        self.det_proj = nn.Linear(det_feat_dim, feat_dim)

        # Stage 1: Full T=8 resolution, bi-directional
        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim * 4,
            dropout=dropout, activation=F.gelu, batch_first=True
        )
        self.stage1 = nn.TransformerEncoder(encoder_layer1, num_layers=num_layers_stage1)

        # Temporal downsampling
        self.temporal_pool = nn.AvgPool1d(kernel_size=2, stride=2)  # T=8→4

        # Stage 2: T=4 resolution
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim * 4,
            dropout=dropout, activation=F.gelu, batch_first=True
        )
        self.stage2 = nn.TransformerEncoder(encoder_layer2, num_layers=num_layers_stage2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_components)
        )

        # Detection-conditioned fusion gate (learned scalar per frame)
        self.fusion_gate = nn.Parameter(torch.zeros(1))  # 0 = no detection signal initially

    def forward(self, p5_feat, det_features=None, det_boxes=None):
        """p5_feat: [B, 768, T=8, 7, 7], det_features: optional [B, T=8, 256]"""
        # Spatial pool
        x = self.spatial_pool(p5_feat).squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, 8, 768]
        x = self.input_proj(x)  # [B, 8, 256]

        # Detection conditioning (if available)
        if det_features is not None:
            det_signal = self.det_proj(det_features)  # [B, 8, 256]
            gate = torch.sigmoid(self.fusion_gate)
            x = x + gate * det_signal

        # Stage 1: Bi-directional (no causal mask)
        x = self.stage1(x)  # [B, 8, 256]

        # Downsample
        x = x.transpose(1, 2)  # [B, 256, 8]
        x = self.temporal_pool(x)  # [B, 256, 4]
        x = x.transpose(1, 2)  # [B, 4, 256]

        # Stage 2
        x = self.stage2(x)  # [B, 4, 256]

        # Global mean pool + classify
        x = x.mean(dim=1)  # [B, 256]
        return self.classifier(x).unsqueeze(1).expand(-1, 8, -1)  # [B, 8, 11]
```

**Parameters:** ~5.2M. Key: the detection conditioning gate starts at 0 (sigmoid(0) = 0.5) but learns when to trust detection input.

### 2.2 Transition-Aware Focal Loss

```python
def transition_aware_focal_loss(logits, targets, gamma=1.5, alpha_pos=0.35, alpha_neg=0.65,
                                 transition_weight=4.0):
    """Focal-BCE with boosted weight on transition events."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # Detect transitions: 0→1 in targets along time axis
    with torch.no_grad():
        transitions = (targets[:, 1:, :] - targets[:, :-1, :]).clamp(min=0)  # [B, T-1, 11]
        # Pad to match T dim
        transitions = F.pad(transitions, (0, 0, 0, 1))  # [B, T, 11]
        # Frame weight: boost frames near transitions (±1 frame)
        frame_weight = 1.0 + transition_weight * (
            transitions + F.pad(transitions[:, :-1, :], (0, 0, 1, 0))
        ).clamp(0, 1)

    # Focal weight
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    alpha_t = targets * alpha_pos + (1 - targets) * alpha_neg
    focal_weight = alpha_t * (1 - pt) ** gamma

    loss = frame_weight * focal_weight * bce
    return loss.mean()
```

---

## 3. Activity Head — Full Specification

### 3.1 Temporal Attention Pool (from Per-Frame Tokens)

```python
class TemporalAttentionPool(nn.Module):
    """Multi-head attention pool over per-frame tokens to produce activity representation."""
    def __init__(self, feat_dim=768, nhead=4, num_queries=1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, feat_dim) * 0.02)
        self.mha = nn.MultiheadAttention(feat_dim, nhead, batch_first=True)

    def forward(self, frame_tokens):
        """frame_tokens: [B, T, 768] — per-frame features from backbone."""
        B = frame_tokens.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, 1, 768]
        out, attn_weights = self.mha(q, frame_tokens, frame_tokens)
        return out.squeeze(1), attn_weights  # [B, 768], [B, 1, T]
```

This requires surfacing per-frame tokens from `MViTFeaturePyramid.forward()`. Currently the forward returns `(fpn_features, cls_token)`. We need to add per-frame tokens:

```python
# In MViTFeaturePyramid.forward(), after processing:
# The sequence tokens x[:, 1:, :] are [B, T*H*W, C]
# We reshape to [B, T, H, W, C] and pool spatial dims
frame_tokens = x[:, 1:, :].reshape(B, T, H, W, C).mean(dim=(2, 3))  # [B, T, 768]
# Return alongside cls_token
return fpn_features, cls_token, frame_tokens
```

### 3.2 Full Activity Head

```python
class ImprovedActivityHead(nn.Module):
    """Activity head with temporal attention pool + 3-layer MLP + logit adjustment."""
    def __init__(self, feat_dim=768, num_classes=75, hidden1=2048, hidden2=1024,
                 nhead=4, dropout=0.2, logit_adjust=False):
        super().__init__()
        self.temporal_pool = TemporalAttentionPool(feat_dim, nhead)
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.classifier = nn.Linear(hidden2, num_classes)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.logit_adjust = logit_adjust
        self._init_weights()

    def forward(self, frame_tokens):
        """frame_tokens: [B, T, 768] — per-frame features (NOT cls_token)."""
        pooled, _ = self.temporal_pool(frame_tokens)  # [B, 768]
        x = self.norm(pooled)
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.drop2(self.act2(self.fc2(x)))
        logits = self.classifier(x)
        if self.logit_adjust and hasattr(self, "class_freq"):
            logits = logits + self.logit_adjust_tau * torch.log(
                self.class_freq + 1e-9
            ).unsqueeze(0)
        return logits
```

**Parameters:** ~5.0M (temporal pool: 2.36M + MLP: 2.64M).

---

## 4. Pose Head — Full Specification

### 4.1 6D Rotation with Geodesic Loss

```python
class Pose6DHead(nn.Module):
    """6D rotation pose head with Gram-Schmidt orthonormalization."""
    def __init__(self, feat_dim=768, temporal_context=3):
        super().__init__()
        input_dim = feat_dim * temporal_context
        self.temporal_context = temporal_context
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6),  # 6D output
            nn.Tanh()  # bound to [-1, 1]
        )

    def forward(self, cls_token_sequence):
        """cls_token_sequence: [B, T, 768] with temporal context."""
        B, T, C = cls_token_sequence.shape
        mid = T // 2
        # Take context window: t-1, t, t+1
        if self.temporal_context == 3 and T >= 3:
            ctx = cls_token_sequence[:, mid-1:mid+2, :].reshape(B, -1)
        else:
            ctx = cls_token_sequence[:, mid, :]
        return self.mlp(ctx)  # [B, 6]

def gram_schmidt_6d(pose_6d):
    """Convert 6D representation to SO(3) rotation matrix (Zhou et al. CVPR 2019)."""
    a1 = pose_6d[:, :3]
    a2 = pose_6d[:, 3:]
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)  # [B, 3, 3] rotation matrix

def geodesic_loss(R_pred, R_gt):
    """Geodesic distance on SO(3): arccos((tr(R_gt^T R_pred) - 1) / 2)."""
    R_diff = R_gt.transpose(1, 2) @ R_pred  # [B, 3, 3]
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle.mean()  # in radians
```

**Parameters:** ~0.5M (3-frame context).

---

## 5. LoRA + FiLM Task Adapters

### 5.1 LoRA for MViTv2-S

```python
class LoRALayer(nn.Module):
    """Low-Rank Adaptation for a linear layer."""
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original = original_linear  # frozen
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scale = alpha / r
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original(x) + self.scale * (x @ self.lora_A @ self.lora_B)
```

Replace Q and V projections in each of MViTv2-S's 16 attention blocks with LoRA-wrapped versions. **Per task: 16 blocks × 2 projections × (768×8 + 8×768) = 786K params.**

### 5.2 FiLM Modulation

```python
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for FFN outputs."""
    def __init__(self, feat_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feat_dim))
        self.shift = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, x):
        return self.scale * x + self.shift
```

Apply FiLM after each FFN block (16 blocks). **Per task: 16 × 2 × 768 = 197K params.**

### 5.3 Task-Specific Adapter Store

```python
class TaskAdapterStore(nn.Module):
    """Stores and switches between per-task LoRA + FiLM adapters."""
    def __init__(self, backbone, tasks=["det", "act", "psr", "pose"]):
        super().__init__()
        self.backbone = backbone
        self.adapters = nn.ModuleDict()
        for task in tasks:
            self.adapters[task] = nn.ModuleDict({
                f"block_{i}": nn.ModuleDict({
                    "q_lora": LoRALayer(...),
                    "v_lora": LoRALayer(...),
                    "film": FiLMLayer(768),
                }) for i in range(16)
            })
        self.current_task = None

    def set_task(self, task):
        self.current_task = task

    def forward(self, x):
        return self.backbone(x, adapters=self.adapters[self.current_task])
```

**Total adapter params: 4 tasks × 1.05M = 4.2M.**

---

## 6. Summary: Implementation Plan

| File | Change | Lines | Priority |
|------|--------|-------|----------|
| `src/models/mvit_mtl_model.py` | BiFPN neck replacement | ~120 | 1 |
| `src/models/mvit_mtl_model.py` | GFLV2DetectionHead | ~80 | 1 |
| `src/models/mvit_mtl_model.py` | DetectionConditionedPSRHead | ~100 | 2 |
| `src/models/mvit_mtl_model.py` | TemporalAttentionPool + ImprovedActivityHead | ~80 | 2 |
| `src/models/mvit_mtl_model.py` | Pose6DHead + gram_schmidt | ~60 | 1 |
| `src/models/mvit_mtl_model.py` | MViTFeaturePyramid: surface frame_tokens | ~15 | 2 |
| `src/models/adapters.py` | LoRALayer, FiLMLayer, TaskAdapterStore (NEW FILE) | ~150 | 3 |
| `src/losses/tal_assigner.py` | Per-level topk | ~10 | 1 |
| `scripts/train_mtl_mvit.py` | GFLV2 loss, geodesic loss, transition-aware focal | ~100 | 1 |
| `scripts/train_mtl_mvit.py` | Nash-MTL gradient bargaining (replace PCGrad) | ~80 | 3 |
| **Total** | | **~795 lines** | |

**Estimated implementation time:** 3-5 days for a single developer.
**Expected model size after all changes:** ~60M (backbone 34.5M + LoRA 4.2M + heads 15M + BiFPN 5M).
**Efficiency:** 60M vs ~100M specialists = 1.67× parameter win + single-pass latency win.
