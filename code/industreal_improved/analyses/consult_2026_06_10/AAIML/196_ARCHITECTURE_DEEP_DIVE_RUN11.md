# 196 — Architecture Deep Dive: run11 MViTv2-S Multi-Task Model

**Date:** 2026-07-10
**Model:** MTLMViTModel (`src/models/mvit_mtl_model.py`, 524 lines)
**Training:** `scripts/train_mtl_mvit.py` (1926 lines)

---

## 1. Total Model: 117.7M Parameters

| Component | Params | Trainable | Source |
|-----------|--------|-----------|--------|
| MViTv2-S Backbone | 34.5M | 34.5M | `torchvision.models.video.mvit_v2_s`, Kinetics-400 pretrained |
| FPN Neck | ~1.5M | ~1.5M | Lateral 1×1 + top-down 2× upsample + 3×3 conv, 256ch output |
| Detection Head | ~4.5M | ~4.5M | Decoupled cls_head + reg_head (DFL, reg_max=16) |
| Activity Head | 3.75M | 3.75M | 3-layer MLP: 768→2048→1024→75, GELU, dropout 0.2 |
| PSR Head | 70.9M | 70.9M | 6-layer causal Transformer, d=768, ff=6144, nhead=4 |
| Pose Head | ~0.5M | ~0.5M | MLP: 768→256→6, Tanh |
| Kendall log_vars | 4 params | 4 params | One per task |
| **Total** | **~117.7M** | **117.7M** | |

---

## 2. Feature Flow Diagram

```
Input: [B, 3, T=16, H=224, W=224]
  │
  ▼
╔══════════════════════════════════════════╗
║         MViTv2-S Backbone (34.5M)        ║
║  Temporal: T=16 → 8 → 4 → 2             ║
║  Spatial:  224² → 56² → 28² → 14² → 7² ║
╠══════════════════════════════════════════╣
║  Forward hooks capture:                  ║
║    conv_proj  → P2  [B, 96,  T=8, 56²] ║
║    blocks[1]  → P3  [B, 192, T=8, 28²] ║
║    blocks[3]  → P4  [B, 384, T=8, 14²] ║
║    blocks[14] → P5  [B, 768, T=8, 7²]  ║
║    cls_token  →     [B, 768]            ║
╚══════════════════════════════════════════╝
  │                    │              │
  │ P3,P4,P5          │ P5           │ cls_token
  ▼                    ▼              ├──────────────────┐
╔══════════╗    ╔═══════════╗        ▼                  ▼
║ FPN Neck ║    ║ PSR Head  ║  ╔═══════════╗    ╔═══════════╗
║ 256ch    ║    ║ (70.9M)   ║  ║ Activity  ║    ║ Pose Head ║
║ P3→P4→P5 ║    ║           ║  ║ Head      ║    ║ (0.5M)    ║
╚══════════╝    ║ P5 features║  ║ (3.75M)   ║    ║           ║
  │              ║ [B,768,T=8]║  ║           ║    ║ [B,6]     ║
  ▼              ║ ↓spatial   ║  ║ cls_token ║    ║ → fwd+up  ║
╔══════════╗    ║  pool      ║  ║ [B,768]   ║    ╚═══════════╝
║ Det Head ║    ║ ↓6-layer   ║  ║ ↓3-layer  ║
║ (4.5M)   ║    ║  causal TF ║  ║  MLP      ║
║ cls+reg  ║    ║ ↓Linear    ║  ║ [B,75]    ║
║ P3+P4+P5 ║    ║ [B,8,11]   ║  ╚═══════════╝
╚══════════╝    ╚═══════════╝
  │
  ├─ cls_logits [B, 24, H, W]
  └─ reg_preds  [B, 64, H, W]  (4 × reg_max=16)
```

**Key routing:**
- P2 (conv_proj, 96-dim) is **NOT used** by any head. It exists in hooks but detection skips it (Opus 192 FC-2).
- Detection uses P3/P4/P5 through FPN → 256ch. TAL assignment happens per-level.
- PSR reads P5 features directly (blocks[14] hook), spatial-pools to 1×1, then temporal Transformer at native T=8.
- Activity and Pose share the cls_token [B, 768].

---

## 3. Detection Head & TAL Assigner

### 3.1 Architecture (already decoupled — Opus 192 FC-2 confirmed)

```python
class DetectionHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=24, reg_max=16):
        # Separate classification branch
        self.cls_head = nn.Sequential(
            Conv2d(256,256,3,padding=1), BN, ReLU,
            Conv2d(256,24,1)
        )
        # Separate regression branch (one per level)
        self.reg_head = nn.Sequential(
            Conv2d(256,256,3,padding=1), BN, ReLU,
            Conv2d(256, 4*reg_max, 1)  # DFL: distribution over reg_max bins
        )
```

### 3.2 TAL Assignment (TaskAlignedAssigner from TOOD, ICCV 2021)

- **Alignment score:** `s = cls_score^α × box_iou^β` with α=1.0, β=6.0
- **Per GT:** pick topk=10 cells per FPN level (so up to 30 cells per GT across 3 levels)
- **Per level:** cell centers → decode DFL → xyxy → compute IoU with GTs → alignment metric → topk
- **Loss:** Focal BCE (γ=2.0, α=0.5) + CIoU + DFL on positive cells
- **P2 is skipped** — P3 (stride 8), P4 (stride 16), P5 (stride 32) only
- **Fallback:** sparse 3×3 assignment if `use_tal=False`

### 3.3 DFL Decoding (Generalized Focal Loss, NeurIPS 2020)

```python
# reg_preds: [B, 4*reg_max, H, W]
# Each of 4 box edges predicted as distribution over reg_max=16 bins
reg_dist = reg_preds.view(B, 4, 16, H, W)  # softmax over dim=2
proj = torch.arange(16).float().view(1,1,16,1,1)
decoded = (reg_dist.softmax(dim=2) * proj).sum(dim=2)  # [B, 4, H, W]
# decoded[0] = left, decoded[1] = top, decoded[2] = right, decoded[3] = bottom
# xyxy = [cx - left*stride, cy - top*stride, cx + right*stride, cy + bottom*stride]
```

---

## 4. PSR Head — The 70.9M Transformer

### 4.1 Why So Large?

The old PSR head (run10) read from `conv_proj` features: `[B, 96, T=8, H=56, W=56]`. These are patch-embedding features — edges, textures, no semantics. The head was a 4-layer transformer with d_model=96 and feedforward 4× (384). Total ~3M params. Loss was flat at 1.56.

The new PSR head reads from P5 (blocks[14]): `[B, 768, T=8, H=7, W=7]`. These are high-level semantic features at the model's deepest layer. The transformer must process a 768-dim input sequence (T=8) with sufficient capacity for transition detection.

**Cost breakdown of the 6-layer transformer with d=768, ff=6144:**

| Component | Per-layer params | ×6 layers |
|-----------|-----------------|-----------|
| Self-attention (Q,K,V,O, 4-head) | 768² × 4 = 2.36M | 14.2M |
| Feedforward (768→6144→768) | 768×6144×2 = 9.44M | 56.6M |
| Layer norms + biases | ~0.02M | 0.1M |
| **Total** | **~11.8M/layer** | **70.9M** |

This is large, but PSR is the most complex task (sequence-to-sequence transition detection) and was the most broken head. The 70.9M is the price of giving it real features.

### 4.2 Architecture Detail

```python
class PSRHead(nn.Module):
    def __init__(self, feat_dim=768, num_components=11, nhead=4, num_layers=6):
        self.spatial_pool = AdaptiveAvgPool3d((None, 1, 1))  # pool H,W → 1×1
        encoder_layer = TransformerEncoderLayer(
            d_model=768, nhead=4,
            dim_feedforward=6144,  # 8× d_model
            dropout=0.1,
            activation=LeakyReLU(0.01),
            batch_first=True,
        )
        self.temporal_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        self.projection = Linear(768, 11)  # → 11 per-frame component logits

    def forward(self, conv_proj_feat):  # [B, 768, T=8, 7, 7]
        x = spatial_pool → [B, 768, T=8, 1, 1] → squeeze → [B, T=8, 768]
        mask = causal_mask(8×8)  # prevent future leakage
        x = temporal_encoder(x, mask=mask)  # [B, 8, 768]
        return projection(x)  # [B, 8, 11]
```

**Key design decisions:**
- **Native T=8** (not 8→16 interpolation — Opus 192 FC-4 fix)
- **Causal mask** ensures frame t cannot attend to frame t+1 (can't cheat)
- **LeakyReLU** avoids dead neurons (ReLU saturation was the old PSR killer)
- **Spatial pool** collapses 7×7→1×1 before temporal processing

### 4.3 PSR Loss: Focal-BCE at T=8

Labels come from dataset at T=16. Downsampled via `adaptive_max_pool1d` (T=16→T=8) preserving any positive transition in each 2-frame window.

```python
psr_targets_T8 = adaptive_max_pool1d(psr_targets_T16.transpose(1,2), output_size=8).transpose(1,2)
# Focal-BCE: FL(pt) = -α_t × (1-pt)^γ × log(pt)
# α_t = 0.25 for positives, 0.75 for negatives
# γ = 2.0
```

---

## 5. Activity Head — 3-Layer MLP

### 5.1 Why 3 Layers?

EP10 activity at 0.58% is below random (1/75 = 1.33%). The old 2-layer MLP (768→1024→75, 1.1M params) cannot discriminate 75 fine-grained assembly states from a single pooled class token. The activity SOTA (0.6525) was set by MViTv2-S + single linear layer — but that was **single-task**. In MTL, the class token represents a compromise across all four tasks.

```python
class ActivityHead(nn.Module):
    def __init__(self, feat_dim=768, num_classes=75, hidden1=2048, hidden2=1024):
        self.norm = LayerNorm(768)
        self.fc1 = Linear(768, 2048); self.act1 = GELU(); self.drop1 = Dropout(0.2)
        self.fc2 = Linear(2048, 1024); self.act2 = GELU(); self.drop2 = Dropout(0.2)
        self.classifier = Linear(1024, 75)
        # Xavier uniform init

    def forward(self, cls_token):  # [B, 768]
        x = norm → fc1 → GELU → drop → fc2 → GELU → drop → classifier
        return logits  # [B, 75]
```

### 5.2 Loss & Class Balancing

- **CE with label_smoothing=0.05** (not 0.1 — less aggressive)
- **Class weights:** inverse-frequency, sqrt-tamed. min=0.0, max=11.71, mean=2.29, 72/75 nonzero
- **Ignore index:** -1 (unlabeled clips)
- **Optional logit-adjustment** (Menon et al. 2020) — subtract per-class log-frequency from logits before softmax. Drop-in bias correction for long-tail. Not currently active.

---

## 6. Pose Head — 6D MLP (Unchanged)

```python
class PoseHead(nn.Module):
    def __init__(self, feat_dim=768):
        self.mlp = Sequential(
            Linear(768, 256), ReLU,
            Linear(256, 6), Tanh   # 6D continuous representation
        )
    # Loss: (1-cos(fwd)) + (1-cos(up))
```

Unchanged because it was already healthy (9° fwd MAE). Pose is the most likely positive-transfer story.

---

## 7. Kendall Uncertainty Weighting (Path-D)

### 7.1 The Mechanism

Four learned parameters `log_var = {det, act, psr, pose}`, initialized to -0.5.

```
weight_i = exp(-log_var_i)          # effective loss weight
loss = weight_det * L_det + weight_act * L_act + weight_psr * L_psr + weight_pose * L_pose
       + log_var_det + log_var_act + log_var_psr + log_var_pose  # regularizer
```

### 7.2 The Caps (Path-D Fix D2)

Without caps, Kendall collapses to `weight ≈ 1/(2·loss)` — the highest-loss task gets the lowest weight, starving it. Our caps:

| Task | log_var cap | Max effective weight | Why this value |
|------|------------|---------------------|----------------|
| Activity | ≤ 1.0 | exp(-1.0) ≈ 0.37 | Activity has highest raw loss (4-5 range) |
| PSR | ≤ 0.5 | exp(-0.5) ≈ 0.61 | PSR has low Focal-BCE (0.15-0.25) |
| Detection | ≤ 1.5 | exp(-1.5) ≈ 0.22 | Detection alternates 0.001/4.5 per batch |
| Pose | ≤ 2.0 (via HP cap) | exp(-2.0) ≈ 0.14 | Pose has lowest raw loss (0.01-0.05) |

### 7.3 EMA-Normalized Losses (Path-D Fix D1)

Each task's loss tracked via EMA (momentum=0.99). Logged as diagnostic. Not used in loss computation (that's the Kendall weights), but monitors whether any task's loss is genuinely diverging vs just high.

---

## 8. PCGrad — Gradient Surgery

PCGrad projects conflicting gradients onto each other's normal planes on the shared backbone. Random 4-permutation per batch. Only applied to backbone parameters (not head-specific params).

```
for each pair (i, j):
    if cos(grad_i, grad_j) < 0:   # conflicting
        grad_i = grad_i - proj(grad_i, grad_j)  # remove conflict
```

HM-7.1-IDENTIFIED-INEFFICIENCY: each batch does 6 comparisons on 34.5M backbone params — costly but worth it per Opus 181.

---

## 9. Training Protocol

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Epochs | 100 | Long enough to converge 117.7M on 78K windows |
| Batch size | 2 × grad-accum-2 = effective 4 | VRAM-limited (12GB) |
| Backbone LR | 1e-4 | Standard for pretrained backbone fine-tuning |
| Head LR | 1e-3 | 10× faster for fresh-init heads |
| log_var LR | 1e-3 | Fast enough to adapt uncertainty weights |
| Grad clip | 5.0 norm | High (was 1.0) — protects against spikes |
| Eval every | 10 epochs | cudaErrorLaunchTimeout at 5 → increased to 10 |
| Max batches/epoch | 8000 | Capped from 39,195 — epoch ~35 min |
| Warmup | None (resumed from ep5) | Shape filter handles new layers |
| Optimizer | AdamW, weight_decay=1e-4 | |
| Scheduler | Cosine annealing to 0 | |
| EMA model | momentum=0.999 | Swapped in for eval (not training) |

---

## 10. What We're NOT Using (Per Opus 192)

| Feature | Why Rejected |
|---------|-------------|
| Foundation backbone (300M+) | Inverts efficiency claim, license risk |
| ArcFace / margin-based losses | Unproven for long-tail 75-class, fragile margins |
| Temporal-attention-pool | Per-frame tokens not exposed; redundant with MViT pooling |
| STORM decoder | Unverified; real PSR bottleneck was feature source + temporal resolution |
| Cross-task attention | High risk, redundant per 186 Q7 |
| MMoE | Marginal upside per 186 D-1 |
| MixUp on activity | Opus 192 Q6: skip; do mosaic on detection only |
| Mosaic/mixup detection aug | Implemented but NOT active (--det-aug not passed) |
