# 189 — Backbone & Multi-Task Topology: How to Share Features Without Hurting Any Head

**Date:** 2026-07-09
**Companion to:** 187 (Opus 181+186 status), 188 (per-head upgrades), 190 (training path)
**Purpose:** Two strategic decisions outside the per-head layer: (1) which **backbone** to use, (2) which **MTL topology** (how features are shared or specialized across tasks). These decisions determine whether MTL is "1 model that mostly works on each task" or "1 model that beats SOTA on each task."
**Hypothesis to support:** MTL shares a backbone, is more efficient, and **still** beats SOTA across heads.

---

## 0. The Shared-Features Problem in One Sentence

In an MTL system, every task uses the same backbone features. If the backbone is **task-agnostic**, all tasks benefit equally. If the backbone is **task-specialized** (e.g., favoring spatial features for pose), some tasks suffer. The architecture choice is: how do we get **task-specialized features** (each head gets what it needs) while keeping **parameter sharing** (one backbone, MTL efficiency)?

---

## 1. BACKBONE OPTIONS

### 1.1 Current: MViTv2-S (34.5M params, Kinetics-400 pretrained)

**Pros:**
- Already integrated; Path-D run is live
- 768-dim class token has spatial pose features (Kinetics pretraining)
- Pretrained on 400-class action recognition — domain-relevant
- 34.5M params is small enough for fast iteration

**Cons:**
- Capacity: 34.5M / 768-dim is at the lower bound for 75-class fine-grained activity
- Pretrained on broad actions (sports, music), not fine-grained assembly
- All 4 tasks compete for the same 768-dim features → per-task specialization is hard

**Verdict:** Good for current path. Will get to ~50-60% of SOTA across heads. To reach 80%+, we need either:
- A stronger backbone (more capacity, more relevant pretraining)
- OR a better MTL topology (per-task feature specialization)

### 1.2 Option A: Scale up to MViTv2-L (53M params)

**Change:** Same architecture, 1.5× params.

**Pros:**
- 1.5× more capacity (~768 → ~1152 hidden dim)
- No new licensing concerns
- Drop-in replacement (just change model class)

**Cons:**
- 1.5× slower (training + inference)
- 1.5× more VRAM
- Marginal accuracy gain (1.5× params rarely gives 1.5× accuracy)

**Expected per-head gain:** +5-10% on each head. Probably gets us to **60-70% of SOTA**.

**Cost:** 0.5 day (change `MViTv2_S` to `MViTv2_L` in the model class). ~2× compute per epoch.

**Verdict:** Cheap, low-upside. Worth doing if compute allows.

### 1.3 Option B: Frozen foundation model (InternVideo2-L, 304M params, Apache 2.0)

**Change:** Replace MViTv2-S with InternVideo2-L (frozen). Add **LoRA adapters** per task. Heads operate on adapted features.

**Pros:**
- **304M params pretrained on Kinetics-710 + HowTo100M + others** — much stronger than MViTv2-S
- 768-dim features carry rich semantic + temporal info
- Frozen backbone: stable training, no catastrophic forgetting
- LoRA adapters (~1M params/task) keep training fast
- MTL efficiency claim still holds: 304M (frozen) + ~5M (adapters + heads) vs 4 × ~100M (specialists)

**Cons:**
- **License caveat (Opus 186 §5.3):** InternVideo2's *code* is Apache 2.0, but the *weights* have usage restrictions. Verify before committing.
- **Inference latency:** 304M model is 10× slower than MViTv2-S. The "MTL is faster" claim weakens.
- **VRAM:** 304M model needs ~12-14 GB on A100 at batch 1. Doable but tight.
- **Domain transfer:** Kinetics-710 actions ≠ assembly actions. Adaptation may be partial.

**Expected per-head gain:** +15-25% on each head. **Likely gets us to 80% of SOTA across heads.**

**Cost:** 2-3 days to integrate (load weights, add LoRA, modify forward).

**Verdict:** Highest single-decision impact. License must clear.

**Alternatives if license is blocked:**
- **DINOv2-L** (305M, Apache 2.0, image-only) — strong general features
- **EVA-02-L** (305M, MIT, image-only) — best image classifier
- **VideoMAE-v2** (1B, self-supervised video) — best video features
- **DINOv2-L + temporal transformer on top** = ~equivalent to InternVideo2 for video

### 1.4 Option C: Hybrid (frozen DINOv2 + small temporal transformer)

**Change:** Frozen DINOv2-L image features → learnable temporal transformer → per-task heads.

**Pros:**
- DINOv2-L: best for dense prediction, Apache 2.0, no weight restrictions
- Image features + temporal transformer: a clean separation of "what's in the frame" (DINOv2) and "how it changes" (temporal transformer)
- Same trainable params (~5-10M) as InternVideo2-L option
- MTL efficiency claim: image features computed once, temporal transformer shared

**Cons:**
- Two-stage (DINOv2 + temporal) may not match end-to-end video pretraining
- DINOv2 is image-only — temporal dynamics entirely learned from our data
- 75K frames may not be enough to learn rich temporal transformer

**Expected per-head gain:** +10-20% on each head. **Borderline 80% of SOTA.**

**Cost:** 3-5 days (integrate DINOv2, build temporal transformer, add per-frame tokens).

**Verdict:** Good middle ground. Strong features + safe license.

### 1.5 Backbone decision matrix

| Backbone | Params (frozen) | Trainable | License | Expected | Cost |
|----------|----------------|-----------|---------|----------|------|
| MViTv2-S (current) | 0 | 34.5M | CC-BY-NC | 50-60% SOTA | 0 d |
| MViTv2-L | 0 | 53M | CC-BY-NC | 60-70% SOTA | 0.5 d |
| InternVideo2-L (frozen) | 304M | ~5M (LoRA) | Apache (code) | **80% SOTA** | 2-3 d |
| DINOv2-L + temp transformer | 305M | ~10M (frozen+temp) | Apache 2.0 | 70-80% SOTA | 3-5 d |
| EVA-02-L (frozen) | 305M | ~5M (LoRA) | MIT | 70-80% SOTA | 2-3 d |

**Recommendation:** **InternVideo2-L** if license clears (verify within 24h). **DINOv2-L + temp transformer** as fallback. **MViTv2-L** as "we have no time" baseline.

---

## 2. MTL TOPOLOGY OPTIONS

MTL topology = how tasks share or specialize features. Five main families:

### 2.1 Shared backbone (current, baseline)

```
images → [Backbone] → backbone_features → {head_det, head_act, head_psr, head_pose}
```

**Pros:** Simple, one forward pass, low params. The MTL efficiency claim is strongest here.

**Cons:** All tasks compete for the same features. Detection (spatial) competes with activity (semantic) competes with PSR (temporal). The shared representation is a compromise that doesn't favor any task.

**Expected:** 50-70% of specialist SOTA per head, depending on backbone.

**Cost:** 0 (already implemented).

### 2.2 Task-specific adapters (LoRA-style)

```
images → [Backbone (frozen)] → backbone_features
                                    ↓
                            per_task LoRA adapters
                                    ↓
                          {task_features for each head}
```

**Pros:**
- Each task gets task-specific features (specialization)
- Backbone stays frozen (stable, fast)
- Small per-task adapter (~1M params each)
- MTL efficiency claim still strong (one backbone, lightweight adapters)

**Cons:**
- Adds complexity (adapter hooks per task)
- Adapter placement is a hyperparameter (which layers to adapt?)
- Adapter collapse: all adapters might learn similar transforms if not regularized

**Implementation:**
```python
class TaskLoRAAdapter(nn.Module):
    """LoRA adapter for one task. Low-rank bottleneck."""
    def __init__(self, feat_dim, rank=16, alpha=1.0):
        super().__init__()
        self.down = nn.Linear(feat_dim, rank, bias=False)
        self.up = nn.Linear(rank, feat_dim, bias=False)
        # Init: up = 0 so adapter starts as identity
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return x + self.scale * self.up(self.down(x))
```

**Expected:** +10-15% over shared backbone.

**Cost:** 1-2 days engineering.

### 2.3 Mixture-of-Experts (MMoE)

```
images → [Backbone] → backbone_features
                            ↓
                     {Expert_1, Expert_2, ..., Expert_K}
                            ↓
                {Gate_act, Gate_det, Gate_psr, Gate_pose} (softmax over experts)
                            ↓
                  {task-specific feature for each head}
```

**Pros:**
- Each task gets a learnable mixture of expert features
- Experts specialize implicitly (e.g., Expert 1 for spatial, Expert 2 for temporal)
- Soft parameter sharing — backbone is shared, but features diverge per task
- Standard MMoE; well-tested on NLP/recsys

**Cons:**
- K experts × `feat_dim²` params (e.g., 8 experts × 768² ≈ 4.7M extra params)
- Risk of expert collapse (all gates pick the same expert)
- 4 task gates × K softmaxes = some overhead

**Implementation:**
```python
class MMoELayer(nn.Module):
    """Multi-gate Mixture-of-Experts layer."""
    def __init__(self, feat_dim=768, num_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, feat_dim * 2),
                         nn.GELU(),
                         nn.Linear(feat_dim * 2, feat_dim))
            for _ in range(num_experts)
        ])
        # Per-task gate: K experts → softmax weights
        self.gates = nn.ModuleDict({
            'det': nn.Linear(feat_dim, num_experts),
            'act': nn.Linear(feat_dim, num_experts),
            'psr': nn.Linear(feat_dim, num_experts),
            'pose': nn.Linear(feat_dim, num_experts),
        })

    def forward(self, x, task):
        # x: [B, T, feat_dim] (or [B, feat_dim])
        # Returns: [B, T, feat_dim] task-specific features
        gate_logits = self.gates[task](x)  # [B, T, K] or [B, K]
        gate_weights = F.softmax(gate_logits, dim=-1)
        # Mix expert outputs
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-2)  # [B, T, K, feat_dim]
        return (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=-2)
```

**Expected:** +5-10% over shared backbone.

**Cost:** 2-3 days engineering.

### 2.4 Cross-task attention (task tokens query backbone)

```
images → [Backbone] → backbone_features
                            ↓
                    TaskToken_task (learnable per task)
                            ↓
            cross-attention(TaskToken, backbone_features)
                            ↓
                  {task-specific features for each head}
```

**Pros:**
- Most flexible: each task learns a *dynamic* query for backbone features
- Tasks can share information if beneficial (e.g., pose and activity are both temporal)
- Novel for video MTL

**Cons:**
- Most complex to implement (cross-attention with task tokens)
- Task tokens can collapse (all tokens learn similar attention patterns)
- Adds ~5-10M params per task

**Implementation:**
```python
class CrossTaskAttention(nn.Module):
    """Per-task cross-attention over backbone features."""
    def __init__(self, feat_dim=768, nhead=8, num_tasks=4):
        super().__init__()
        # Learnable task tokens
        self.task_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, feat_dim))
            for name in ['det', 'act', 'psr', 'pose']
        })
        for p in self.task_tokens.parameters():
            nn.init.normal_(p, std=0.02)
        # Cross-attention per task
        self.attns = nn.ModuleDict({
            name: nn.MultiheadAttention(feat_dim, nhead, batch_first=True)
            for name in self.task_tokens
        })
        # LayerNorm for stability
        self.norms = nn.ModuleDict({
            name: nn.LayerNorm(feat_dim)
            for name in self.task_tokens
        })

    def forward(self, backbone_features, task):
        # backbone_features: [B, N, feat_dim] (e.g., 16*7*7=784 tokens)
        token = self.task_tokens[task].expand(backbone_features.size(0), -1, -1)
        attn_out, _ = self.attns[task](token, backbone_features, backbone_features)
        return self.norms[task](attn_out.squeeze(1))
```

**Expected:** +5-15% over shared backbone (if it works; high variance).

**Cost:** 3-5 days engineering. **High risk of task-token collapse.**

### 2.5 Sequential pretraining + model soup (Strat-4)

```
Phase 1: 4 single-task runs, each trains [Backbone + Task-specific head]
Phase 2: Average the 4 backbone weights (model soup)
Phase 3: Initialize MTL model with soup'd backbone, finetune end-to-end
```

**Pros:**
- Each task reaches its single-task ceiling in Phase 1
- Soup'd backbone is a strong initialization (Wortsman 2022, Ilharco 2022)
- Phase 3 finetuning at low LR preserves per-task features
- No complex topology changes

**Cons:**
- Total compute: 4 single-task runs (~10-15 GPU-days total) + 1 finetune (~5 GPU-days)
- Requires implementing 4 single-task training scripts
- The soup might still be a compromise

**Expected:** 85-95% of single-task ceiling per head. **Likely 80%+ SOTA across all heads.**

**Cost:** 2-3 weeks wall-clock.

**Verdict:** Most defensible academically; highest probability of meeting 80%; highest cost.

### 2.6 Topology decision matrix

| Topology | Trainable Params | Compute Overhead | Expected | Cost |
|----------|-----------------|------------------|----------|------|
| Shared (current) | 34.5M | 1× | baseline | 0 d |
| + LoRA adapters | 38M | 1.05× | +10-15% | 1-2 d |
| + MMoE | 39M | 1.10× | +5-10% | 2-3 d |
| + Cross-task attn | 44M | 1.20× | +5-15% (high variance) | 3-5 d |
| Sequential + soup (Strat-4) | 34.5M × 5 | 5× | **+30-50%** | 2-3 wk |

**Recommendation: Sequential + soup (Strat-4) is the highest probability of meeting 80% across all heads.** LoRA adapters are a cheap upgrade that could be done in parallel.

---

## 3. THE COMBINED RECIPE

### 3.1 Tier 1 (cheapest, ~70-80% SOTA across heads)

**Components:**
- MViTv2-L backbone (53M, 1.5× current)
- 2-layer MLP activity head (current)
- YOLOv8 detection head (replaces current)
- STORM-like PSR decoder (replaces current)
- 6D pose head (current)
- Shared MTL topology
- Path-D training (current)
- EMA model weights (current)

**Compute:** 8-10 days (1 GPU).

**Expected:** 0.55-0.70 detection, 0.20-0.40 activity, 0.40-0.65 PSR, 4-6° pose. **Likely meets 80% SOTA on detection + PSR; activity is the long-shot.**

### 3.2 Tier 2 (medium cost, ~80% SOTA across heads)

**Components:**
- Frozen InternVideo2-L backbone (304M, Apache 2.0 if cleared; else DINOv2-L)
- LoRA adapters per task (1M each)
- YOLOv8 detection head
- Temporal attn pool + ArcFace activity head
- STORM-like PSR decoder
- 6D pose head
- Shared MTL topology
- Path-D training
- EMA + AdamW + cosine warmup

**Compute:** 12-15 days.

**Expected:** 0.65-0.80 detection, 0.45-0.60 activity, 0.55-0.75 PSR, 4-6° pose. **Likely meets 80% SOTA on all 4 heads.**

### 3.3 Tier 3 (high cost, ~85-95% SOTA across heads)

**Components:**
- Same as Tier 2, but with **sequential pretraining + model soup** (Strat-4)
- 4 single-task pretrainings + 1 MTL finetune
- Cross-task attention layer on top of frozen backbone

**Compute:** 3-4 weeks.

**Expected:** 0.70-0.85 detection, 0.50-0.70 activity, 0.65-0.85 PSR, 3-5° pose. **Likely clears 80% on all 4 heads with margin.**

### 3.4 Recommended tier for AAIML deadline (~3 weeks out)

**Tier 2.** The combination of frozen foundation backbone + LoRA + head upgrades from file 188 gives the best probability/cost tradeoff.

If time permits, layer Strat-4 (sequential pretrain + soup) on top of Tier 2 for Tier 3 numbers.

---

## 4. SPECIFIC DECISION TREE

```
START
  │
  ├─ Q1: Is InternVideo2-L license clear?
  │    ├─ YES → frozen InternVideo2-L (Tier 2)
  │    └─ NO  → frozen DINOv2-L + temporal transformer (Tier 2 fallback)
  │
  ├─ Q2: Do we have 3+ weeks wall-clock?
  │    ├─ YES → Tier 3 (sequential + soup + cross-task attn)
  │    └─ NO  → Tier 2 (frozen + LoRA + head upgrades)
  │
  ├─ Q3: What's the AAIML deadline?
  │    ├─ > 4 weeks → Tier 3
  │    └─ 2-3 weeks → Tier 2
  │    └─ < 2 weeks → Tier 1
  │
  └─ Q4: Is single-task > MTL on detection the expected outcome?
       ├─ YES → accept it; the L2+L3+method story is the paper
       └─ NO  → must implement YOLOv8 head (file 188 §1.3)
```

---

## 5. WHAT THIS FILE ENABLES (vs File 188)

File 188 is per-head architectural changes. **File 189 is the shared infrastructure** that makes the per-head changes actually work in MTL. Specifically:

- **Backbone choice** (file 189 §1) determines the **input quality** to every head.
- **MTL topology** (file 189 §2) determines whether tasks compete for features or get task-specialized features.

Combined, files 188 + 189 + 190 (training path) form the complete plan to:
- Beat SOTA on detection (YOLOv8 head + foundation backbone)
- Beat SOTA on activity (temporal attn + ArcFace + foundation backbone)
- Beat SOTA on PSR (STORM decoder + foundation backbone)
- Stay above 80% on pose (keep current)

**The key insight: the bottleneck is not just one thing. It's backbone + per-task heads + topology + training. All four need to be addressed.**

---

## 6. RISKS & MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Foundation model license blocks | Medium | High | Use DINOv2-L (Apache 2.0) as fallback |
| Foundation model domain transfer insufficient | Medium | Medium | Add domain-specific fine-tuning data |
| Task-token collapse in cross-task attn | Medium | Medium | Add diversity regularizer; or skip cross-task attn |
| LoRA adapter collapse | Low | Low | Use larger rank (16-32); orthogonal init |
| 4 single-task pretrainings fail | Low | High | Verify single-task works on 1 task first |
| MMoE expert collapse | Medium | Medium | Add load-balancing loss; or use fewer experts |

---

## 7. RECOMMENDED IMMEDIATE ACTIONS (next 48 hours)

1. **Verify InternVideo2-L license** (Opus 186 §5.3 caveat). If blocked, switch to DINOv2-L.
2. **Test DINOv2-L** as a backbone drop-in (1-2 hours smoke test):
   - Load DINOv2-L
   - Run 1 epoch on the existing Path-D code with backbone frozen
   - Report activity top-1 at ep1
3. **If foundation backbone works:** start implementing LoRA adapters + per-head upgrades (file 188).
4. **If foundation backbone doesn't pan out:** fall back to MViTv2-L (1.5× scale-up) + per-head upgrades (Tier 1 plan).

---

## 8. KEY REFERENCES

- Wortsman et al. 2022: "Model Soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time" — foundation for model soup / task arithmetic
- Ilharco et al. 2022: "Editing Models with Task Arithmetic" — task arithmetic as a primitive for combining task-specific models
- Ma et al. 2018: "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (MMoE)
- Tang et al. 2020: "Progressive Layered Extraction (PLE)" — a more recent MMoE variant
- Houlsby et al. 2019: "Parameter-Efficient Transfer Learning for NLP" (LoRA-style adapters)
- Ultralytics YOLOv8: AGPL-3.0; the architecture (TAL + DFL + decoupled head) is described in the YOLOv8 paper (no license required to re-implement)
- Schoonbeek et al. 2024 (WACV): "IndustReal: A Dataset for Understanding Industrial Assembly Actions" — primary source for the 0.838 / 0.652 / 0.901 SOTA numbers (verified by Opus 186 directly from arXiv:2310.17323)

---

*This file should be read together with 187 (status), 188 (per-head upgrades), 190 (training path). The full plan is in 190.*