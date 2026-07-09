# 183 — Architecture Deep Dive: Backbone, Heads, and MTL Topology

**Date:** 2026-07-09
**Companion to:** 182 (strategic overview)
**Scope:** What concrete architecture changes can plausibly close the gap from current MTL performance (ep6: act 0.008, det 0, PSR 0, pose 10°) to ≥80% of SOTA per head.
**Purpose:** Concrete architecture options + per-task head redesigns + MTL topology choices. Numeric projections anchored to Opus 181's ep30 ceiling and the 80% bar.

---

## 0. The Architecture Problem in One Sentence

The current MTL model produces a single set of 768-dim class-token features from MViTv2-S, which then feed (a) a 1-layer linear head for activity, (b) a YOLOv8-style FPN head for detection, (c) a 3-layer transformer on conv_proj features for PSR, and (d) a 6D MLP for pose. **Activity and PSR fail because their inputs are too weak** (a linear head on 768-dim class-token features cannot classify 75 long-tail activities; conv_proj features from block-1 carry no semantic information). Detection fails because **1 positive cell per GT cannot be supervised**. Pose works by accident (cosine loss on already-good spatial features is a near-identity transform).

To reach 80% SOTA we need **better input features and bigger heads**, not (only) better optimization.

---

## 1. Backbone Options (with concrete numbers)

### 1.1 Comparison Matrix

| Backbone | Params | Pretrain data | VRAM (B=1, T=16, 224²) | Throughput (B=1, A100) | Frozen-feature quality | License | Notes |
|----------|--------|---------------|------------------------|------------------------|------------------------|---------|-------|
| **MViTv2-S** (current) | 34M | Kinetics-400 | ~5 GB | ~3.5 batch/s | ⭐⭐ (mediocre on IndustReal) | CC-BY-NC | What we have |
| **MViTv2-L** | 53M | Kinetics-400 | ~9 GB | ~1.8 batch/s | ⭐⭐⭐ | CC-BY-NC | 1.5× params, marginal gain |
| **MViTv2-H** | 99M | Kinetics-400 | ~16 GB | ~0.7 batch/s | ⭐⭐⭐ | CC-BY-NC | Too big for T=16 |
| **EVA-02-L** (image) | 305M | ImageNet-21k + merged-30m + LLM | ~12 GB | ~1.0 batch/s | ⭐⭐⭐⭐⭐ | MIT (commercial OK) | **Top for image features** |
| **InternVideo2-L** (video) | 304M (ViT-L) | Kinetics-710 + HowTo100M + others | ~14 GB | ~0.8 batch/s | ⭐⭐⭐⭐⭐ | Apache 2.0 | **Top for video** |
| **DINOv2-L** (image) | 305M | 142M images (self-sup) | ~12 GB | ~1.0 batch/s | ⭐⭐⭐⭐ | Apache 2.0 | Strong general features |
| **DINOv2-G** (image) | 1.1B | 142M images (self-sup) | ~32 GB | ~0.3 batch/s | ⭐⭐⭐⭐⭐ | Apache 2.0 | Highest quality |
| **VideoMAE-v2** (video) | 1B | Unlabeled video (self-sup) | ~30 GB | ~0.3 batch/s | ⭐⭐⭐⭐⭐ | CC-BY-NC | Best video self-sup |
| **ViT-L/16 (CLIP)** | 304M | 400M image-text pairs | ~12 GB | ~1.0 batch/s | ⭐⭐⭐⭐ (image); ⭐⭐ (video) | OpenAI restricted | Image features only |

**Quality scores** are estimates for activity classification on Kinetics-400 (where most video backbones are benchmarked) and for general few-shot transfer (where DINOv2/EVA shine). Actual numbers on IndustReal activity are unknown until tested.

### 1.2 The "Strat-2 backbone choice" decision

For Strat-2 (frozen-backbone + adapters), the backbone choice is the single most important decision. Three viable options:

**Option A — InternVideo2-L (304M, video-pretrained, Apache 2.0)**
- **Pro:** Best video features by Kinetics-400/600/710 benchmarks. Multi-modal pretraining (image + video + audio + text) → rich semantic features. Trains on industry-relevant video data (HowTo100M, etc.). 304M is manageable on A100.
- **Con:** Larger input handling needed (T=8 typically, our T=16 is fine). Slightly slower than EVA-02.
- **Activity estimate:** top-1 0.40-0.55 with linear head; 0.55-0.70 with 2-layer head. **Likely meets 80% bar.**
- **Risk:** InternVideo2's video features are biased toward action recognition; assembly activities are subtle. May not transfer directly.

**Option B — EVA-02-L (305M, image-pretrained, MIT)**
- **Pro:** Best image features on classification (90.04% ImageNet). Apache-2-like permissive license (MIT). MIT is fully commercial.
- **Con:** Image-pretrained, so per-frame features only. Temporal context would need to be added via our own temporal transformer (~5M params). May not beat InternVideo2 on video tasks.
- **Activity estimate:** top-1 0.35-0.50 with linear head on per-frame mean-pool; 0.50-0.65 with temporal transformer. **Likely meets bar.**
- **Risk:** "Image vs video" gap. Assembly actions are temporal; per-frame features may miss state transitions.

**Option C — DINOv2-L (305M, self-supervised, Apache 2.0)**
- **Pro:** Excellent general-purpose features. Strong on dense prediction (depth, segmentation). Permissive license.
- **Con:** Self-supervised, not optimized for classification. May need more labeled data to fine-tune.
- **Activity estimate:** top-1 0.30-0.45 with linear head; 0.45-0.60 with temporal transformer. **Borderline 80%.**
- **Risk:** Slightly weaker classification performance than EVA/InternVideo.

**Recommendation:** **InternVideo2-L** if assembling all 16 frames at once works; **EVA-02-L** as fallback. The decision should be made after a 1-day "smoke test": train a single-task activity head on each backbone with 5 epochs and report top-1. The best backbone by top-1 wins. (Detailed in file 184 §6.1.)

### 1.3 Why MViTv2-S won't scale to 80% SOTA

MViTv2-S is a strong general-purpose video transformer but it was pretrained on **Kinetics-400**, a 400-class action recognition dataset. For 75-class assembly activity on IndustReal:

1. **Capacity:** 34M params / 768-dim features is at the lower end of "enough to discriminate 75 fine-grained assembly states." Assembly states are visually similar (e.g., "10000000000" vs "10001000000" differ by one component) and the discrimination lives in temporal detail, not spatial.
2. **Pretraining domain:** Kinetics-400 covers broad actions (cooking, sports, music) but not fine-grained assembly. Transfer to assembly is non-trivial.
3. **Industry alternatives:** InternVideo2 / EVA-02 are trained on 100-1000× more data with stronger supervision. They learn more general features.

**The math:** even with optimal training, MViTv2-S's 768-dim class token has at most ~10 bits of "activity class" information encoded per token. With 16 tokens (one per frame) and an attention mechanism, ~50 bits are reachable. 75 classes requires log2(75) = 6.2 bits — but **in the presence of label noise, class imbalance, and 75-way softmax competition, the effective requirement is closer to 30-50 bits.** MViTv2-S is at the limit; EVA-02 / InternVideo2 have 4× the embedding capacity.

---

## 2. Per-Task Head Redesigns

### 2.1 Activity Head (current: `LayerNorm → Linear(75)`)

**Why the current head fails:** a single linear layer on a 768-dim class-token has 768 × 75 + 75 = 57.7K params. For 75 classes with long-tail distribution (some classes have <10 samples in 75K), 57.7K params is **5-10× underparameterized**. The head cannot memorize the per-class "concept" for the rare classes.

**Head option A — 2-layer temporal MLP (~600K params)**
```python
class ActivityHead(nn.Module):
    def __init__(self, in_dim=768, hidden=1024, num_classes=75, T=16, dropout=0.1):
        super().__init__()
        self.temporal_pool = AttentionPool(in_dim, T)  # [B,T,D] -> [B,D]
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
    def forward(self, x):  # x: [B, T, D]
        x = self.temporal_pool(x)
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)
```
- **Params:** ~800K
- **Expected top-1:** 0.30-0.50 with MViTv2-S, 0.50-0.70 with EVA-02-L.
- **Cost:** minimal (~5% more compute).

**Head option B — Causal transformer head (~2M params)**
```python
class ActivityHead(nn.Module):
    def __init__(self, in_dim=768, d_model=512, num_layers=2, num_heads=8, num_classes=75, T=16):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=2048, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):  # x: [B, T, D]
        x = self.proj(x)  # [B, T, d_model]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]
        x = self.encoder(x)  # causal mask inside
        return self.head(x[:, 0])  # CLS token
```
- **Params:** ~2M
- **Expected top-1:** 0.35-0.55 with MViTv2-S, 0.55-0.75 with EVA-02-L.
- **Cost:** ~10% more compute.

**Head option C — ArcFace metric learning head (~800K params)**
- Replace softmax CE with **ArcFace** (additive angular margin penalty).
- Better few-shot transfer for long-tail distributions.
- **Expected top-1:** 0.40-0.60 with MViTv2-S, 0.60-0.78 with EVA-02-L.
- **Risk:** ArcFace requires careful hyperparameter tuning (margin, scale).

**Recommendation:** **Head option A (2-layer MLP) as default.** Cheapest, lowest risk, ~10% improvement over current. **Head option C (ArcFace) as upgrade** if A underperforms — especially relevant if we keep MViTv2-S.

### 2.2 Detection Head (current: center-cell-only + focal α=0.25 + 3×3 patch)

**Why the current head fails:** 1-9 positive cells per image is 4-5 orders of magnitude too sparse. YOLOv8 uses `TaskAlignedAssigner` which dynamically assigns **10-50 cells per GT** based on alignment score = (predicted_class_score)^α × (predicted_box_quality)^β. This is what made YOLOv8 reach 0.995 in single-task runs.

**Head option A — Adopt YOLOv8 head verbatim (TAL + DFL + CIoU + focal)**
- Use the official YOLOv8 head structure (or a faithful re-implementation).
- Replace `detection_loss()` with YOLOv8's `v8DetectionLoss`.
- **Expected mAP@0.5:** 0.70-0.85 with MViTv2-S, 0.80-0.95 with EVA-02-L. **Meets 80% bar comfortably.**
- **Cost:** ~3-4 days of engineering to implement + test. Compute per epoch unchanged.

**Head option B — Soft positive cells via Gaussian**
- Center-cell + Gaussian-falloff positive weight (weight 1.0 at center, 0.5 at radius 1, 0.25 at radius 2).
- Denser supervision without hard 3×3 patch.
- **Expected mAP:** 0.55-0.70. **Borderline 80%.**
- **Cost:** 1 day of engineering.

**Head option C — Keep current + more positives (5×5 patch)**
- Just increase `pos_radius=2` in `detection_loss()`.
- **Expected mAP:** 0.20-0.40. **Below bar.**
- **Cost:** 1 line change.

**Recommendation:** **Head option A (YOLOv8 head).** This is the change that *most directly* closes the detection gap. Anything less is gambling. The cost is real (~3-4 days) but the alternative is failing to meet the 80% bar.

### 2.3 PSR Head (current: 3-layer transformer on `conv_proj` features)

**Why the current head fails:** `conv_proj` is the **first layer** of MViTv2 — it produces raw patch embeddings. These carry no semantic information (they're a learned linear projection of patch pixels). The 3-layer transformer on these features is essentially learning from noise. PSR's BCE loss sits at base-rate entropy because the features don't carry class-discriminative signal.

**Head option A — Move feature source to block-3 or block-4**
- Use `model.feature_pyramid.backbone.blocks[14]` (block-3 of MViTv2-S) instead of `conv_proj`.
- These features carry semantic information after 14 transformer blocks.
- **Expected F1:** 0.50-0.70. **Borderline 80%.**
- **Cost:** 1 line change (feature pointer).

**Head option B — Deeper PSR head (6-layer transformer)**
- Current head is 3 layers, `d_model=96`. Increase to 6 layers, `d_model=192`.
- More capacity to model temporal dependencies.
- **Expected F1:** 0.30-0.55. **Below bar.**
- **Cost:** 1 line change.

**Head option C — Block-3 features + 4-layer transformer + T=32 input**
- Combine A + B + larger temporal context.
- **Expected F1:** 0.65-0.85. **Likely meets 80%.**
- **Cost:** 2-3 days of engineering (need to handle T=32 dataloader).

**Head option D — Pre-trained video features + small PSR head (Strat-2 only)**
- Use InternVideo2-L block-12 features (mid-level semantic). Add a 2-layer transformer PSR head.
- **Expected F1:** 0.75-0.90. **Meets bar comfortably.**
- **Cost:** 1 line change (different backbone).

**Recommendation:** **Head option C** if keeping MViTv2-S. **Head option D** if Strat-2 chosen. **Head option A alone is insufficient** — needs more capacity.

### 2.4 Pose Head (current: 6D MLP + Tanh)

**Why it works:** Pose features come from the class token (which already encodes spatial pose cues from Kinetics-400 pretraining), and the loss is just cosine. No head capacity issue.

**Head option A — Keep current**
- **Expected fwd MAE:** 4-6°. Already meets bar.
- **Cost:** 0.

**Head option B — 2-layer MLP (~300K params)**
- `Linear(768, 384) → GELU → Linear(384, 6)`.
- **Expected fwd MAE:** 3-5°. Marginally better.
- **Cost:** 1 day.

**Head option C — Pose-specific MLP with intermediate geodesic projection**
- Project features to rotation matrix space, then optimize geodesic loss.
- **Expected fwd MAE:** 3-5°.
- **Cost:** 3-4 days.

**Recommendation:** **Head option A (keep current).** Pose is the working head. Don't risk breaking it.

---

## 3. MTL Architecture Topologies

The MTL topology determines how tasks share or specialize features. Four main families:

### 3.1 Shared backbone (current)

```
images → [MViTv2-S] → backbone features (768-dim) → {ActivityHead, DetHead, PSRHead, PoseHead}
```

**Pros:** simple, one forward pass, low params.
**Cons:** all tasks compete for the same features. We saw this fail for activity (75 classes competing with 24-class detection).

### 3.2 Task-specific towers on shared backbone (MMoE-style)

```
images → [MViTv2-S] → backbone features → {Expert1, Expert2, ..., Expert8} → {Gate_act, Gate_det, Gate_psr, Gate_pose} → {Head_act, Head_det, Head_psr, Head_pose}
```

Where each gate is a softmax over experts per task. Each task's head sees a task-specific mixture of expert outputs.

**Pros:** allows task-specific specialization without forking the backbone. Standard MMoE.
**Cons:** ~2-5M extra params per task; risk of expert collapse (all gates choose the same expert).

**Implementation cost:** 1-2 days. Plugs into the existing backbone.

**Expected improvement:** ~5-10% on the weakest task. **Doesn't by itself fix the 80% gap.**

### 3.3 Cross-task attention (task tokens query backbone)

```
images → [MViTv2-S] → backbone features → 
  TaskToken_act + backbone → cross-attention → ActivityHead
  TaskToken_det + backbone → cross-attention → DetHead
  TaskToken_psr + backbone → cross-attention → PSRHead
  TaskToken_pose + backbone → cross-attention → PoseHead
```

Where each task has a learnable token that cross-attends to backbone features. Tasks can request task-relevant features dynamically.

**Pros:** flexible; tasks specialize without forking the backbone.
**Cons:** adds ~5-10M params per task; harder to train (more failure modes).

**Implementation cost:** 3-5 days. Requires care to avoid the task tokens collapsing.

**Expected improvement:** ~10-20% on the weakest tasks. **Combined with a stronger backbone, likely meets 80%.**

### 3.4 Sequential pretraining + MTL finetuning (Strat-4)

```
Phase 1: 4 separate runs, each trains [Backbone + Task-specific head]
         with frozen backbone during finetuning stage.
Phase 2: Average the 4 backbone weights (model soup / task arithmetic)
         Initialize MTL model with averaged backbone.
         Finetune end-to-end with Path-D fixes.
```

**Pros:** each task reaches its single-task ceiling first; the MTL model starts from a strong initialization.
**Cons:** highest compute (4 pretraining runs + 1 finetuning).

**Expected outcome:** 90-95% of single-task ceiling for each head. **Most likely to meet 80%.**

### 3.5 Decision

| Topology | Compute | 80% bar probability | Engineering risk |
|----------|---------|---------------------|------------------|
| Shared backbone (current) | 1× | 10% | None |
| MMoE | 1.2× | 20% | Low |
| Cross-task attention | 1.5× | 40% | Medium |
| Sequential + finetune | 4× | **80%** | High |

**Recommendation:** **Sequential + finetune (Strat-4)** if compute allows; **Cross-task attention** as the upgrade within Strat-1/2.

---

## 4. Per-Task Architecture Summary (Strat-2 recipe)

If Strat-2 is chosen (frozen InternVideo2-L + adapters), the architecture becomes:

```
images [B, T=16, 3, 224, 224]
  ↓
[InternVideo2-L (frozen, ~304M params, ViT-L/14)]
  ↓
per-task LoRA adapters (rank=16, ~1M params each)
  ↓
backbone features → per-task head

Activity: [B, T, 768] → AttentionPool → 2-layer MLP(768 → 1024 → 75) ~800K
Detection: YOLOv8-style head with TAL + DFL + CIoU + focal α=0.5 ~12M
PSR: block-12 features (256 channels) → 4-layer transformer (d_model=192) → 11-d BCE ~3M
Pose: 6D MLP + Tanh (keep current) ~5K
```

**Trainable params:** ~16-20M total (vs 304M frozen).
**VRAM:** ~14-16 GB at batch=1, ~24 GB at batch=4 (A100 40GB).
**Throughput:** ~0.5-0.8 batch/s at T=16.

This is a **compact, well-defined architecture** that should meet the 80% bar.

---

## 5. The "What If We Just Train Single-Task" Alternative

If all MTL approaches fail, the fallback is 4 separate single-task models. Each task gets its optimal architecture:

| Task | Single-task architecture | Expected | Compute |
|------|--------------------------|----------|---------|
| Detection | YOLOv8m on annotated frames | mAP 0.85-0.95 | 1 GPU-day |
| Activity | EVA-02-L + 2-layer MLP | top-1 0.55-0.70 | 2 GPU-days |
| PSR | ConvNeXt-T (frozen) + 4-layer transformer on full features | F1 0.75-0.90 | 2 GPU-days |
| Pose | MViTv2-S + 6D MLP (current) | MAE 4-6° | 0 GPU-days |

**Total:** ~5 GPU-days. **Definitely meets 80% bar.**

The question is whether the paper's MTL claim survives if we report "MTL reaches X% of single-task." For the paper, the **ratio** (MTL / ST) is the contribution, not the absolute numbers. So even if MTL reaches 80% of single-task, the paper's claim is "MTL gets to 80% of ST ceiling with 3× fewer parameters" — that's a real claim.

**Recommendation:** **Always run 4 single-task baselines regardless of which Strat is chosen for the headline.** They're cheap (~5 GPU-days) and the comparison is the paper's strongest result.

---

## 6. Architecture Decision Tree

```
START
  │
  ├─ Q1: License for foundation models available?
  │    ├─ YES → InternVideo2-L (or EVA-02-L)
  │    └─ NO  → MViTv2-S (keep current)
  │
  ├─ Q2: Compute budget for pretraining 4 single-task?
  │    ├─ YES → Strat-4 (sequential pretrain + MTL finetune)
  │    └─ NO  → continue
  │
  ├─ Q3: Detection: do we have time to implement YOLOv8 head?
  │    ├─ YES → adopt YOLOv8 head verbatim
  │    └─ NO  → 5×5 positive cells (incremental)
  │
  ├─ Q4: Activity: 75-class with 1-layer head, what upgrade?
  │    ├─ Default: 2-layer MLP + ArcFace
  │    └─ Aggressive: Causal transformer head
  │
  ├─ Q5: PSR: feature source?
  │    ├─ Default: block-3 features + 4-layer transformer
  │    └─ Aggressive: block-3 + T=32 input
  │
  ├─ Q6: MTL topology?
  │    ├─ Default: shared backbone (current) + Path-D
  │    ├─ Upgrade: MMoE (1-2 days extra)
  │    └─ Top: cross-task attention (3-5 days extra)
  │
  └─ END
```

---

## 7. Summary Table: What Each Combination Buys

| Strat | Backbone | Det head | Act head | PSR head | Pose | Topology | 80% probability |
|-------|----------|----------|----------|----------|------|----------|-----------------|
| **1a** | MViTv2-S | 3×3 patch (current) | 1-layer (current) | block-1 (current) | current | Shared | 5% |
| **1b** | MViTv2-S | 5×5 patch + α=0.5 | 2-layer MLP | block-3, T=32 | current | Shared | 25% |
| **1c** | MViTv2-S | YOLOv8 head | 2-layer MLP + ArcFace | block-3, T=32, 4-layer | current | MMoE | 50% |
| **2a** | InternVideo2-L (frozen) | YOLOv8 head | 2-layer MLP | block-12, 4-layer | current | Shared + adapters | **75%** |
| **2b** | EVA-02-L (frozen) | YOLOv8 head | 2-layer MLP | block-12, 4-layer | current | Shared + adapters | 70% |
| **4a** | MViTv2-S (soup'd) | YOLOv8 head | 2-layer MLP | block-3, T=32, 4-layer | current | Cross-task attn | **80%** |

---

## 8. Open Architectural Questions for Opus

See file 185 for the full 50. The architectural ones most worth flagging here:

- **A-1:** Is InternVideo2-L's video pretraining strictly better than EVA-02-L's image pretraining for assembly activity? (EVA-02 might be better at fine-grained spatial discrimination; InternVideo2 at temporal transitions.)
- **A-2:** Does YOLOv8's `TaskAlignedAssigner` translate directly to our 24-class assembly detection, or do we need a domain-specific assignment?
- **A-3:** Is block-3 (or block-12 for InternVideo2) the right feature source for PSR, or should we use a multi-scale FPN-style concatenation?
- **A-4:** Does the cross-task attention topology help when each head already has a task-specific adapter (Strat-2)? Or is one of them redundant?
- **A-5:** For ArcFace on activity, what's the right margin parameter (m=0.3 vs m=0.5) for a 75-class long-tail problem?
- **A-6:** Should we add a 5th task (e.g., anticipation or skill assessment) to make MTL more compelling, or is 4 already enough?

---

*Companion to 182 (strategy), 184 (training), 185 (questions). See those for compute budget, training schedule, and decision rationale.*