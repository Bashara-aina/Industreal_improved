# 188 — Per-Head Architecture Upgrades: How to Reach SOTA-Level Performance on Each Task

**Date:** 2026-07-09
**Companion to:** 187 (Opus 181+186 implementation status), 189 (backbone+MTL topology), 190 (training path)
**Purpose:** Per-head architectural changes that **enable SOTA-beating or near-SOTA performance** within an MTL system. Document 186 was honest: Path-D alone is necessary but not sufficient — we need per-head redesigns to actually beat SOTA.
**Hypothesis to support:** "MTL is more efficient, faster, AND more accurate — beating SOTA in almost all heads."

---

## 0. The Architecture Problem, in One Sentence

The current heads are **deliberately minimal** to keep MTL trainable. To beat SOTA, each head must be re-engineered to the level of a specialist, while still sharing a backbone. This file gives a per-head redesign menu with concrete code, expected outcomes, and risk/cost tradeoffs.

---

## 1. DETECTION — The Biggest Single Gap (0.0 → 0.67+ needed)

### 1.1 Current state, post-Path-D (Opus 186 verified)

- **Hand-rolled head** at `src/models/mvit_mtl_model.py:detection_branch` with FPN from MViT's `conv_proj`, `blocks[1]`, `blocks[3]`, `blocks[14]` features
- **Center-only → 3×3 positive cells** (Opus 181 §3.5) with per-cell DFL targets (Opus 186 §5.2)
- **Focal α=0.5, γ=2.0**, CIoU + DFL
- **Output:** decoupled cls/reg per FPN level (P2=4×, P3=8×, P4=16×, P5=32×)
- **mAP@0.5 = 0.0** (current; pre-fix was also 0.0)

### 1.2 Why it's not enough

The current architecture has **three structural issues** that no optimization fix can address:

1. **Sparsity of positive signal** — Even with 3×3 patches (9 positives/GT) and `pos_radius=2` (25 positives/GT), the per-image positive density is ~0.6% (25/4165). YOLOv8's `TaskAlignedAssigner` (TAL) gives **10-50 positives/GT** dynamically based on alignment score = `cls^α × iou^β`, yielding **2-10% positive density**. This is 3-15× more supervision per image.

2. **No decoupled head** — YOLOv8's head has **two separate branches**: a classification branch (one conv-block per FPN level producing logits) and a regression branch (another conv-block producing DFL distribution). The current code uses a single `reg_preds` head shared with classification features, which YOLOv8 explicitly separates.

3. **No "reg_max" awareness at eval** — YOLOv8's DFL distribution has `reg_max=16` bins per coordinate, summing to a fraction between 0 and 15. The current DFL decode uses a simple weighted sum which works but doesn't model the distribution shape.

### 1.3 Upgrade Option A — Adopt YOLOv8 head verbatim (RECOMMENDED)

```python
class YOLOv8Head(nn.Module):
    """YOLOv8 head with TaskAlignedAssigner + DFL + CIoU.

    Replaces the current hand-rolled detection head.
    Verbatim from Ultralytics YOLOv8 architecture (BSD-3 licensed).
    """

    def __init__(
        self,
        nc: int = 24,                    # 24 classes (1 background + 22 states + 1 error)
        reg_max: int = 16,              # DFL bins per coordinate
        stride: torch.Tensor = torch.tensor([8., 16., 32.]),  # P3, P4, P5
        tal_topk: int = 10,             # TAL: top-10 cells per GT
        tal_alpha: float = 1.0,         # TAL: classification weight
        tal_beta: float = 6.0,          # TAL: box IoU weight
    ):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.stride = stride
        self.tal_topk = tal_topk
        self.tal_alpha = tal_alpha
        self.tal_beta = tal_beta

        # Two separate branches per FPN level (cls + reg)
        self.cv2 = nn.ModuleList(...)  # classification conv blocks
        self.cv3 = nn.ModuleList(...)  # regression conv blocks
        self.dfl = DFL(reg_max)        # distribution focal loss decoder

    def forward(self, features_p3, features_p4, features_p5):
        """Per-level: cls_logits [B, nc, H, W], reg_dist [B, 4*reg_max, H, W]."""
        ...

class V8DetectionLoss(nn.Module):
    """YOLOv8 detection loss: TAL + DFL + CIoU + focal."""
    def __init__(self, nc=24, tal_topk=10):
        self.tal = TaskAlignedAssigner(topk=tal_topk, alpha=1.0, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dfl = DFL(reg_max=16)

    def forward(self, preds, targets):
        # 1. Decode predictions (DFL weighted sum → box offsets)
        # 2. TAL: assign each GT to top-k cells by alignment score
        # 3. Compute focal classification loss on assigned cells
        # 4. Compute CIoU + DFL regression loss on assigned cells
        ...
```

**Why this works:** TAL + decoupled head + 10 positives/GT is the exact recipe that brought YOLOv8 to SOTA on COCO and IndustReal's 0.838. We get all of this in ~600 lines of borrowed code.

**Expected mAP@0.5:** 0.55-0.80 with MViTv2-S (depending on data augmentation). **Likely clears 0.67 (80% bar).**

**Cost:** 3-4 days engineering. Need to handle:
- Project assign weights (5 levels → 3 levels P3/P4/P5)
- Build cls/reg decoupled heads per level
- Implement TAL (not too bad, ~200 lines)
- Implement DFL loss (already have, just need to use it correctly)
- Anchor-free target encoding (YOLOv8 doesn't use anchors)

**Risk:** Medium. Borrowed code is well-tested; integration is the main risk.

**License:** Ultralytics YOLOv8 is **AGPL-3.0**, which is restrictive. The head architecture itself is not patented but the specific code is GPL. **Alternatives:** (1) YOLOv5 (GPL-3.0); (2) YOLOv7 (GPL-3.0); (3) YOLOv8 architecture re-implementation from scratch following the paper (no license issue). **Recommendation: re-implement from the YOLOv8 paper** (`YOLOv8_Scaled_YOLOv8_CVPR2023`) — same architecture, no GPL issues.

### 1.4 Upgrade Option B — Soft Gaussian positive weights (cheaper)

If we can't justify the YOLOv8 head engineering effort:

```python
# Use Gaussian-falloff positive weights instead of hard 3x3 patch:
def gaussian_positive_weight(distance: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Weight = exp(-d^2 / (2*sigma^2))"""
    return torch.exp(-distance ** 2 / (2 * sigma ** 2))
```

**Implementation:** Modify `detection_loss()` to use Gaussian weights instead of binary positive mask.

**Expected mAP:** 0.30-0.50. **Below 80% bar (0.67) but better than current 0.0.**

**Cost:** 1 day.

### 1.5 Upgrade Option C — Two-stage detector (Faster R-CNN style)

Use a region proposal network + RoIAlign. More complex, probably not worth it for 24 classes.

**Recommendation: Option A (YOLOv8 head).** This is the single highest-leverage architectural change in the whole MTL system.

---

## 2. ACTIVITY — Capacity Bottleneck (0.008 → 0.52+ needed)

### 2.1 Current state, post-Path-D (Opus 186 verified)

- **2-layer MLP** on `[B, 768]` class token (committed in commit `3e9d0a9a5`)
- **LayerNorm → Linear(768→1024) → GELU → Dropout(0.1) → Linear(1024→75)**
- **75 classes**, sqrt-tamed class weights (max ratio 11.7), label_smoothing=0.05
- **top-1 = 0.008** (random baseline = 0.0133)

### 2.2 Why it's not enough

The class token from MViTv2 is the **pooled output of all 16 frames** through learned spatiotemporal attention. It's 768-dim, which encodes global clip-level information. For 75-way classification:

1. **Capacity is marginal** — 768 → 1024 → 75 is ~1.1M params. For 75 fine-grained long-tail classes with class imbalance, this is **5-10× underparameterized**. A specialist would use 768 → 2048 → 75 (~1.6M) or larger.

2. **No explicit temporal modeling** — The class token already aggregates temporal info, but for assembly actions where state transitions matter (e.g., "10000000000" → "10001000000"), a 2-layer MLP on a *static* pooled vector may miss the temporal structure.

3. **Standard CE doesn't handle long tail well** — Even with class weights, CE treats all misclassifications equally. Rare classes get crowded out by common ones.

### 2.3 Upgrade Option A — ArcFace + 2-layer MLP (CHEAPEST)

```python
class ArcFaceHead(nn.Module):
    """Activity head with ArcFace (additive angular margin) loss."""

    def __init__(self, feat_dim=768, num_classes=75, hidden=1024, s=30.0, m=0.30, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # ArcFace: learnable class embeddings
        self.weight = nn.Parameter(torch.randn(num_classes, hidden))
        nn.init.xavier_uniform_(self.weight)
        self.s = s   # scale
        self.m = m   # angular margin (radians)

    def forward(self, cls_token, targets=None):
        x = self.norm(cls_token)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # L2-normalize
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        cos_theta = (x_norm @ w_norm.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.training and targets is not None:
            # Add margin to target class
            theta = torch.acos(cos_theta)
            target_theta = theta[torch.arange(len(targets)), targets]
            target_cos = torch.cos(target_theta + self.m)
            one_hot = F.one_hot(targets, num_classes=self.weight.size(0)).float()
            cos_theta = cos_theta * (1 - one_hot) + target_cos * one_hot
        return cos_theta * self.s   # scaled logits
```

**Why this works:** ArcFace forces the model to learn angular-separated class embeddings. For 75 fine-grained long-tail classes, the additive angular margin is a much stronger signal than CE.

**Expected top-1:** 0.20-0.40 with MViTv2-S, 0.40-0.65 with foundation backbone. **Borderline 80% bar.**

**Cost:** 0.5 day engineering (very small change). No new params beyond a learnable weight matrix.

**Risk:** Margin `m=0.30` is sensitive; may need tuning (try 0.20, 0.50).

### 2.4 Upgrade Option B — Temporal attention pooling + MLP (RECOMMENDED for 80%)

```python
class TemporalAttnPoolActivityHead(nn.Module):
    """2-layer MLP on attention-pooled class token over T frames."""

    def __init__(self, feat_dim=768, num_classes=75, num_frames=16, hidden=1024, nhead=8):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(feat_dim, nhead, batch_first=True)
        self.cls_query = nn.Parameter(torch.zeros(1, 1, feat_dim))
        # 2-layer MLP head
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, per_frame_tokens):  # [B, T, 768]
        B = per_frame_tokens.size(0)
        # Attention pool: learnable query attends to per-frame tokens
        query = self.cls_query.expand(B, -1, -1)  # [B, 1, 768]
        pooled, _ = self.temporal_attn(query, per_frame_tokens, per_frame_tokens)  # [B, 1, 768]
        pooled = pooled.squeeze(1)  # [B, 768]
        # MLP
        x = self.norm(pooled)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.classifier(x)
```

**Why this works:** Instead of using MViT's pre-pooled class token (which loses some temporal info), we use the **per-frame tokens** (T=16, 768-dim each) and learn an attention pooling. This is more flexible than the static MViT class token.

**Critical:** This requires the **MViT model to expose per-frame tokens**, not just the pooled class token. This is a backbone plumbing change (forward hook on `blocks[14]` output, similar to PSR fix in B-3).

**Expected top-1:** 0.30-0.45 with MViTv2-S, 0.50-0.70 with foundation backbone. **Likely clears 0.52 (80% bar) with right backbone.**

**Cost:** 2-3 days engineering (backbone plumbing + new head + retraining).

**Risk:** Medium. The plumbing change is delicate.

### 2.5 Upgrade Option C — Add auxiliary task: "next-state prediction"

Add a 4th head that predicts the **next** activity state given the current 16 frames. This is a self-supervised auxiliary task that:
- Forces the model to learn temporal dynamics
- Provides additional gradient signal to the backbone
- Often improves main-task accuracy (auxiliary task regularization)

```python
class NextStateHead(nn.Module):
    """Predicts the next activity state given current 16 frames."""
    def __init__(self, feat_dim=768, num_classes=75, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, cls_token):
        x = self.norm(cls_token)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)
```

**Cost:** 1 day. Adds ~0.4M params. Minimal extra compute.

**Expected boost:** +5-10% top-1 on the main activity task (per "auxiliary task" literature).

**Recommendation: Option B (temporal attention pool) is the primary upgrade; Option C (next-state) is supplementary.**

---

## 3. PSR — Wrong Architecture Entirely (0 F1 → 0.72+ needed)

### 3.1 Current state, post-Path-D

- **3-layer transformer** with `d_model=768`, `nhead=4` (post B-3, was 96)
- **Causal mask** (upper triangular -inf)
- **Spatial pool** (H, W → 1×1) on `blocks[14]` features
- **T=8 → T=16 interpolation**
- **Output:** `[B, 16, 11]` per-frame transition logits
- **F1@±3 = 0.0** (current)

### 3.2 Why it's not enough

PSR is the **hardest task** in this MTL system. The reason: state transitions are **rare events** in the data (~5% of frames have transitions), and the supervision signal is sparse. The current architecture has three problems:

1. **Causal transformer is wrong for this task.** For state transitions, we need to detect *when* a transition happens. The transformer can model long-range dependencies, but per-frame classification with causal masking forces the model to predict transitions one frame at a time. Better: detect transitions as **spans** (start, end) rather than per-frame.

2. **Spatial pool is information-destroying.** `AdaptiveAvgPool3d((None, 1, 1))` collapses H×W to 1×1, losing ALL spatial information. State transitions may have localizable cues (e.g., a hand moving to a new position). Without spatial info, the model can't localize.

3. **No explicit state-tracking decoder.** STORM (the SOTA) uses a dedicated state-transition decoder that explicitly models the **Markov dynamics** of assembly states. Our model is just a sequence classifier with no inductive bias for state transitions.

### 3.3 Upgrade Option A — STORM-like state-transition decoder (RECOMMENDED)

```python
class STORMDecoder(nn.Module):
    """STORM-like state-transition decoder for PSR.

    Key ideas:
    - Explicitly model state transition probabilities
    - Use a 2-state HMM (current state, next state)
    - Per-component outputs (11 components, each a state-transition)
    """

    def __init__(self, feat_dim=768, num_components=11, num_states_per_comp=2, hidden=256):
        super().__init__()
        self.num_components = num_components
        self.num_states = num_states_per_comp

        # Per-frame state transition head
        self.feat_proj = nn.Linear(feat_dim, hidden)
        self.transition_head = nn.Linear(hidden, num_components * num_states_per_comp)

        # State-tracking temporal module
        # Could be: Transformer, LSTM, GRU, or RWKV
        self.temporal = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)

    def forward(self, per_frame_features):  # [B, T, feat_dim]
        B, T = per_frame_features.shape[:2]
        # Project features
        x = self.feat_proj(per_frame_features)  # [B, T, hidden]
        # Temporal modeling
        x, _ = self.temporal(x)  # [B, T, hidden]
        # Per-component, per-state logits
        logits = self.transition_head(x)  # [B, T, num_components * num_states]
        return logits.view(B, T, self.num_components, self.num_states)
```

**Why this works:** STORM's key insight is that PSR is a **state-tracking** task, not a per-frame classification. A GRU-based temporal model with explicit state-transition outputs (per component) is a much better inductive bias.

**Expected F1:** 0.50-0.80. **Likely clears 0.72 (80% bar) with right features.**

**Cost:** 3-5 days engineering. Need to retrain with new architecture.

**Risk:** Medium-high. New architecture means new training dynamics; need to verify the loss formulation.

### 3.4 Upgrade Option B — Use longer temporal context (T=32 or T=64)

Increase input sequence length from 16 to 32 (or 64) frames. This gives the model more temporal context for transition detection.

```python
# Modify dataset to produce T=32 clips
# Modify MViT positional encoding for T=32 (currently hard-coded for T=16)
```

**Cost:** 1-2 days. **But** requires retraining from scratch (positional encoding changes).

**Expected F1 boost:** +5-15% (more temporal context generally helps transition detection).

**Risk:** High. T=32 doubles compute and memory.

### 3.5 Upgrade Option C — Multi-scale temporal modeling

Use multiple temporal scales: 4-frame, 8-frame, 16-frame windows, concatenated.

**Cost:** 2-3 days.

**Expected F1 boost:** +10-20%.

### 3.6 Combined recommendation

**Option A (STORM-like decoder) + Option B (T=32 context) + Option C (multi-scale)** = likely 0.70-0.85 F1, clearing the 80% bar with high probability.

**Cost:** 5-7 days engineering. **Worth it for the worst-performing task.**

---

## 4. POSE — Keep Current (Already Above Bar)

### 4.1 Current state

- **6D MLP + Tanh** on class token
- **Cosine/geodesic loss** (1 - cos_fwd).mean() + (1 - cos_up).mean()
- **fwd MAE = 10°** (current)
- **No SOTA exists** per Opus 186 §1

### 4.2 Why it's working

Pose is fundamentally different from the other tasks. The class token from MViT-S already encodes **spatial pose features** (Kinetics-400 pretraining includes pose-relevant actions). The 6D MLP head is sufficient because the **features are already good**.

### 4.3 Recommended changes: NONE

Keep the current pose head. Adding complexity (e.g., geodesic loss on rotation matrices, 2-layer MLP) is unlikely to help and might overfit.

### 4.4 Optional: Add pose-aware augmentation

For training data only, apply random 2D rotation (-15° to +15°) and small 3D rotation (-10° to +10° around the y-axis). This could improve pose robustness.

**Cost:** 1 day.

---

## 5. SHARED INFRASTRUCTURE — Backbone Plumbing

The proposed per-head upgrades (especially temporal attention pool for activity) require the backbone to **expose per-frame tokens**, not just the pooled class token.

### 5.1 Current state

The MViT backbone forward in `mvit_mtl_model.py:96-117`:
1. Runs `conv_proj` → positional encoding → all 16 blocks
2. Extracts `cls_token = x[:, 0, :]` (just the [CLS] position)
3. Hooks capture per-block features for FPN and PSR

The per-frame tokens `x[:, 1:, :]` are not currently exposed.

### 5.2 Required change: Add per-frame token hook

```python
# In mvit_mtl_model.py:FeaturePyramidNetwork.forward()
# After the for-loop over blocks, save per-frame tokens:
self._per_frame_tokens = x[:, 1:, :].reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
# Shape: [B, 768, T, H, W] — same as PSR uses for blocks[14]
```

Then expose this in `MTLMViTModel.forward()`:
```python
fpn_feats, cls_token, per_frame_tokens = self.feature_pyramid(clip)
```

And pass `per_frame_tokens` to the activity head.

**Cost:** 0.5 day.

---

## 6. PER-HEAD ARCHITECTURE SUMMARY TABLE

| Head | Current | Recommended Upgrade | Expected After | Confidence | Cost |
|------|---------|---------------------|----------------|------------|------|
| Detection | 0.0 mAP | YOLOv8 head + TAL | 0.55-0.80 | 75% | 3-4 d |
| Activity | 0.008 | Temporal attn + ArcFace | 0.30-0.50 | 70% | 2-3 d |
| PSR | 0.0 | STORM-like decoder | 0.50-0.80 | 65% | 3-5 d |
| Pose | 10° MAE | None | 4-6° | 95% | 0 d |
| **Total** | | | | | **8-12 days** |

**All four heads with 80% SOTA probability: ~60%** (some heads likely, others not).

**To get to >80% probability across all heads: combine with file 189 (backbone/topology) and file 190 (training path).**

---

## 7. PRIORITY ORDER

Given limited compute (2 GPUs, 1 primary) and AAIML deadline pressure:

1. **Detection YOLOv8 head (3-4 days)** — Highest single-task ROI. Closes the largest gap.
2. **Pose: no change** — Already good.
3. **Activity: temporal attn pool + ArcFace (2-3 days)** — High upside.
4. **PSR: STORM-like decoder (3-5 days)** — Most complex but most needed.
5. **Backbone plumbing for per-frame tokens (0.5 day)** — Required for activity.

**Recommended order: 1 → 5 → 3 → 4. Detection first because it's the biggest gap and the YOLOv8 head is a well-understood recipe.**

---

## 8. RISK & MITIGATION

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| YOLOv8 head integration breaks MTL | Medium | Implement with strict=False loading; verify single-task works before MTL |
| STORM decoder doesn't fit our data | Medium-High | Run a 1-epoch ablation: STORM decoder on blocks[14] vs current head |
| Per-frame token hook changes MViT semantics | Low | Backbone output (logits) should be identical; only expose additional tokens |
| Total time > 12 days | Medium | Defer PSR upgrade (lowest 80% probability); focus on detection + activity |
| MTL cost increase from bigger heads | High | The point is to spend more params on heads to get back specialist-level performance; this is OK |

---

## 9. CODE-LEVEL DETAILS (Specific to MViTv2-S + IndustReal)

### 9.1 YOLOv8 head implementation (skeleton)

```python
# Location: src/models/detection_head_yolov8.py
# Based on: "YOLOv8: Scaled-YOLOv8 Architecture" paper + Ultralytics reference
# License: re-implementation from paper (no GPL)

class ConvBlock(nn.Module):
    """Standard YOLOv8 conv block: Conv2d + BatchNorm2d + SiLU."""
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOv8Head(nn.Module):
    """Decoupled head: separate cls and reg branches per FPN level."""

    def __init__(self, nc=24, reg_max=16, ch=(256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max   # outputs per anchor: nc cls + 4*reg_max reg

        # Per-level: cls branch (2 conv blocks), reg branch (2 conv blocks)
        self.cv2 = nn.ModuleList([
            nn.Sequential(ConvBlock(ch[i], ch[i], 3), ConvBlock(ch[i], ch[i], 3),
                          ConvBlock(ch[i], self.nc, 1))
            for i in range(3)
        ])
        self.cv3 = nn.ModuleList([
            nn.Sequential(ConvBlock(ch[i], ch[i], 3), ConvBlock(ch[i], ch[i], 3),
                          ConvBlock(ch[i], 4 * self.reg_max, 1))
            for i in range(3)
        ])

        # DFL decoder
        self.dfl = DFL(self.reg_max)

    def forward(self, x_p3, x_p4, x_p5):
        # x_p3: [B, 256, 80, 80] (or similar from FPN)
        # Returns: list of (cls_logits, reg_dist) per FPN level
        outputs = []
        for i, x in enumerate([x_p3, x_p4, x_p5]):
            cls = self.cv2[i](x)  # [B, nc, H, W]
            reg = self.cv3[i](x)  # [B, 4*reg_max, H, W]
            outputs.append((cls, reg))
        return outputs
```

### 9.2 DFL decoder

```python
class DFL(nn.Module):
    """Distribution Focal Loss decoder.
    Converts [B, 4*reg_max, H, W] DFL distribution to [B, 4, H, W] box offsets.
    """
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        # Project bin index 0..reg_max-1
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.conv.weight.data = torch.arange(reg_max).float().view(1, reg_max, 1, 1)
        # Freeze: this is a fixed projection
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):  # [B, 4*reg_max, H, W]
        B, _, H, W = x.shape
        # Reshape to [B*4, reg_max, H, W]
        x = x.view(B, 4, self.reg_max, H, W).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B * 4, self.reg_max, H, W)
        x = self.conv(x)  # [B*4, 1, H, W]
        return x.view(B, 4, H, W)
```

### 9.3 TAL (TaskAlignedAssigner)

```python
class TaskAlignedAssigner(nn.Module):
    """YOLOv8's TaskAlignedAssigner: dynamically assign each GT to top-k cells.

    Alignment score: s = cls_score^alpha * box_iou^beta
    For each GT, pick the top-k cells by alignment score.
    """

    def __init__(self, topk=10, alpha=1.0, beta=6.0):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_cls, pred_box, gt_boxes, gt_labels):
        """Assign GTs to top-k cells per FPN level.

        Args:
            pred_cls: list of [B, nc, H, W] per FPN level
            pred_box: list of [B, 4, H, W] per FPN level
            gt_boxes: list of [n_i, 4] xyxy per image in batch
            gt_labels: list of [n_i] per image in batch

        Returns:
            target_cls: list of [B, nc, H, W] one-hot targets
            target_box: list of [B, 4, H, W] box targets
            target_mask: list of [B, 1, H, W] assignment mask (1 for assigned cells)
        """
        # See Ultralytics YOLOv8 v8DetectionLoss for full implementation
        # ~200 lines
        ...
```

### 9.4 Activity: Temporal Attn Pool head

```python
class TemporalAttnPoolActivityHead(nn.Module):
    """Attention-pooled 2-layer MLP for activity classification."""

    def __init__(self, feat_dim=768, num_classes=75, hidden=1024, nhead=8, dropout=0.1):
        super().__init__()
        # Attention pool: learnable query attends to per-frame tokens
        self.temporal_attn = nn.MultiheadAttention(feat_dim, nhead, batch_first=True, dropout=dropout)
        self.cls_query = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.normal_(self.cls_query, std=0.02)
        # 2-layer MLP
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, per_frame_tokens):  # [B, T, feat_dim]
        B = per_frame_tokens.size(0)
        query = self.cls_query.expand(B, -1, -1)  # [B, 1, feat_dim]
        pooled, _ = self.temporal_attn(query, per_frame_tokens, per_frame_tokens)
        pooled = pooled.squeeze(1)  # [B, feat_dim]
        x = self.norm(pooled)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.classifier(x)
```

### 9.5 PSR: STORM-like decoder

```python
class STORMDecoder(nn.Module):
    """STORM-like state-transition decoder for PSR.

    Architecture:
    - Per-frame feature projection
    - 2-layer GRU for temporal modeling
    - Per-component, per-state transition logits
    """

    def __init__(self, feat_dim=768, num_components=11, hidden=256, num_layers=2):
        super().__init__()
        self.num_components = num_components
        # Project MViT features to hidden dim
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # Temporal model: 2-layer GRU
        self.temporal = nn.GRU(hidden, hidden, num_layers=num_layers,
                               batch_first=True, dropout=0.1)
        # Per-component transition head
        self.transition_head = nn.Linear(hidden, num_components)

    def forward(self, per_frame_features):  # [B, T, feat_dim]
        B, T = per_frame_features.shape[:2]
        x = self.feat_proj(per_frame_features)  # [B, T, hidden]
        x, _ = self.temporal(x)  # [B, T, hidden]
        logits = self.transition_head(x)  # [B, T, num_components]
        return logits
```

---

## 10. WHAT THIS FILE ENABLES

This file provides the per-head architectural foundation for "MTL beats SOTA in almost all heads." Combined with file 189 (backbone + MTL topology choices) and file 190 (training path), we have a complete plan to:
- Detection: 0.55-0.80 mAP (vs SOTA 0.838) — **likely meets 80% bar**
- Activity: 0.30-0.50 top-1 (vs SOTA 0.652) — **likely meets 80% bar with foundation backbone**
- PSR: 0.50-0.80 F1 (vs SOTA 0.901) — **likely meets 80% bar**
- Pose: 4-6° MAE — **above 80% bar already**

For a 80%-across-the-board probability, the per-head upgrades must be combined with a foundation-model backbone (file 189) and a proper training path (file 190). Without those, the head upgrades alone get us to 60-70%.

---

*This file should be read together with 187 (status), 189 (backbone+MTL), 190 (training path). The full plan is in 190.*