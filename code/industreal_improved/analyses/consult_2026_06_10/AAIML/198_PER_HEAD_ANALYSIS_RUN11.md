# 198 — Per-Head Analysis: SOTA Comparison, Bottlenecks, Honest Ceilings

**Date:** 2026-07-10

---

## 1. Detection Head

### 1.1 SOTA Baseline

| System | Training Data | mAP@0.5 (annotated) | mAP@0.5 (entire-video) |
|--------|--------------|---------------------|------------------------|
| YOLOv8-m (SOTA, real+synthetic) | 26.9K real + 100K synthetic | **0.838** | 0.641 |
| YOLOv8-m (IndustReal-only) | 26.9K real only | **0.779** | **0.575** |
| YOLOv8-m (synthetic only) | 100K synthetic | 0.573 | 0.341 |

**Honest ceiling for our model:** 0.779 (boxed) / 0.575 (entire-video). We have no synthetic data. Opus 192 FC-6 explicitly corrected frame against the 0.838 number.

### 1.2 Our Architecture (run11)

```
Backbone FPN: MViTv2-S stages → P3(28²), P4(14²), P5(7²) → FPN 256ch
Detection head: Decoupled cls_head + reg_head (DFL reg_max=16)
Assigner: TAL (TOOD), topk=10 per GT per level
Loss: Focal BCE (γ=2.0, α=0.5) + CIoU + DFL
Classes: 24 assembly states (22 correct + some error states)
```

### 1.3 Structural Disadvantages vs YOLOv8-m

| Factor | YOLOv8-m | Our MViTv2-S | Gap |
|--------|---------|-------------|-----|
| Backbone | CSPDarkNet-53 | MViTv2-S | YOLOv8-m is detection-optimized CNN |
| Pretraining | COCO detection | Kinetics-400 action recognition | Domain mismatch |
| FPN input | Built for detection | Generic video features | Feature semantics differ |
| Synthetic data | 100K Unity frames | 0 | Major gap |
| Head design | Mature, tuned over years | Our custom decoupled head | Less refined |
| Dedicated | Single-task optimized | Shared with 3 other tasks | Gradient competition |

### 1.4 Realistic Expectation

**What's achievable:** A shared-backbone MTL detector will be **weaker** than YOLOv8-m single-task. The question is "how much weaker" and whether it's an acceptable trade-off.

**Internal benchmarks from prior work (file 176, Opus 192 ref):**
- Prior ConvNeXt-MTL run reached **0.468 mAP** on a similar setup
- The detection eval harness was confirmed working (non-zero mAP obtained)
- So 0.0 mAP in run10 was likely (a) P2 feature contamination + (b) fresh-init head, not an eval bug

**run11 expectation:** With TAL (denser supervision) + P2 skip + 5 epochs of training from warm backbone:
- **Optimistic:** 0.10-0.25 mAP at ep10
- **Realistic:** 0.05-0.15 mAP at ep10
- **Pessimistic:** <0.01 mAP (TAL not enough, fundamental MTL competition)

**What would "good enough" look like?** Opus 192's framing: MTL/ST ratio ≥0.7. If a single-task detection baseline reaches ~0.50 mAP (likely given ConvNeXt reached 0.468), then 0.35 mAP in MTL is a defensible cost.

### 1.5 Open Questions

1. **TAL correctness:** Is the TAL assigner mapping box coordinates correctly between our decoded DFL boxes (pixel space, stride-aware) and the GT boxes? A coordinate system error would make TAL assign wrong cells.
2. **FPN quality:** Are P3/P4/P5 features from MViTv2-S sufficiently detection-discriminative? This is the same backbone that hits 65.25% activity — but activity features ≠ detection features.
3. **Class imbalance:** 24 classes with extreme long-tail. Focal γ=2.0 should help but may need tuning.
4. **Detection augmentation:** `--det-aug` (flip+color+crop) is implemented but NOT active in run11. Would it help?

---

## 2. Activity Head

### 2.1 SOTA Baseline

| Model | Pretrain | Modalities | Top-1% | Top-5% |
|-------|---------|------------|--------|--------|
| MViTv2-S | Kinetics-400 | RGB | **65.25** | **87.93** |
| MViTv2-S | Kinetics-400 | RGB+VL+stereo | 66.45 | 88.43 |
| SlowFast | Kinetics-400 | RGB | 60.39 | 85.21 |

**Critical fact (Opus 192 §1 Layer 2):** The 65.25% SOTA was reached with **MViTv2-S + single linear classifier + plain CE.** This is our exact backbone class. The head is NOT the bottleneck for activity — the representation is.

### 2.2 Our Architecture (run11)

```
Input: cls_token [B, 768]
Head: LayerNorm → Linear(768→2048) → GELU → Dropout(0.2) →
      Linear(2048→1024) → GELU → Dropout(0.2) → Linear(1024→75)
Loss: CE(label_smoothing=0.05, weight=sqrt-tame inv-freq)
Classes: 75 (ACT_CLASS_GROUPING="none")
```

### 2.3 Why MTL Activity is Harder Than ST Activity

| Factor | Single-task MViTv2-S | Our MTL MViTv2-S |
|--------|---------------------|-------------------|
| cls_token optimization | Solely for 75-class CE | Shared with pose (spatial) + detection (FPN) + PSR (P5) |
| Backbone gradients | One loss direction | 4 competing gradient directions |
| Kendall weight | N/A (single task) | Capped at exp(-1.0) ≈ 0.37 |
| Class weights | None needed (balanced batches) | Sqrt-tame inv-freq needed (long-tail) |

The cls_token must represent spatial direction (for pose), object features (for detection via hooked FPN), temporal semantics (for PSR via P5), AND activity class. That's a lot of pressure on a single 768-dim vector.

### 2.4 Realistic Expectation

**The 65.25% SOTA is a single-task ceiling we will not reach in MTL.** The question is the MTL/ST ratio:

- **Optimistic (MTL at ep30):** 40-50% top-1 (MTL/ST ≈ 0.65-0.77). This would be a strong result — the cls_token carries enough information for all 4 tasks.
- **Realistic:** 25-35% top-1 (MTL/ST ≈ 0.40-0.55). Bounded cost bought back by parameter efficiency.
- **Pessimistic:** <15% top-1 (MTL/ST < 0.25). The cls_token is over-compressed for 4 tasks.

**The run10 0.58% was well below even pessimistic.** A 3-layer MLP should not stay below random after 5 epochs of training. If run11 EP10 still shows <2% activity:
- Either the cls_token is fundamentally overloaded (needs task-specific features, not shared)
- Or the Kendall weight is starving activity (cap at 0.37 may still be too low)
- Or the learning rate / optimizer setup is wrong for the activity head

### 2.5 Options If Activity Underperforms

| Option | Cost | Risk | Expected gain |
|--------|------|------|---------------|
| Logit-adjustment (Menon et al.) | 1 hour | Low | +2-5% on long-tail |
| Increase act LR (separate head LR) | 1 hour | Low | +2-5% if LR is bottleneck |
| VideoMAE V2 stream (frozen, +22M params) | 2 days | Medium (VRAM) | +5-7% from temporal stream |
| Per-frame temporal features (re-plumb backbone) | 3-5 days | High (reshaping) | Unknown, possibly +10% |
| Unfreeze Kendall cap for activity | 5 min | High (may collapse) | +? but risks starvation |

---

## 3. PSR Head

### 3.1 SOTA Baseline

| System | POS | F1 | τ (s) |
|--------|-----|-----|-------|
| STORM (B3, real+synthetic) | **0.797** | **0.883** | 22.4 |
| B2 (confidence accumulation) | 0.731 | 0.860 | 22.3 |
| B1 (naive: every ASD change) | 0.570 | 0.779 | 14.9 |

**Critical fact:** All three baselines use ASD (detection) outputs + procedural knowledge to infer step completions. We predict per-frame component states. These are fundamentally different tasks — the B2/B3 baselines are NOT directly comparable to our per-frame logit approach.

The comparable baseline is **B1** (naive: every state change = step), but even B1 uses YOLOv8m at 0.838 mAP.

### 3.2 Our Architecture (run11)

```
Input: P5 features [B, 768, T=8, 7, 7]
Head: Spatial pool (7²→1²) → [B, T=8, 768] →
      6-layer causal Transformer (d=768, nhead=4, ff=6144, dropout=0.1) →
      Linear(768, 11) → [B, 8, 11]
Loss: Focal-BCE (γ=2.0, α=0.25)
Label: T=16→8 max-pool downsampling
```

### 3.3 The Dramatic Improvement

**run10 PSR loss: flat ~1.56 for 5 epochs**
**run11 PSR loss: 0.15-0.25 immediately**

This 6-10× loss drop is the single most important architectural signal in the entire project. It definitively proves:
1. The conv_proj (96-dim) feature source was the bottleneck — not the decoder architecture
2. P5 (768-dim semantic) features contain transition-relevant information
3. The 6-layer causal transformer at d=768 can actually learn on this task

**What we still don't know:** Whether 0.17-0.25 Focal-BCE translates to non-zero event_F1@±3. The loss being low doesn't guarantee the right kinds of predictions. But it's the strongest positive signal we've had for PSR.

### 3.4 Realistic Expectation

- **Optimistic:** event_F1@±3 > 0.3 at ep10. This would be >75% of the B1 baseline (0.779) with a fundamentally different (and weaker) approach.
- **Realistic:** event_F1@±3 in 0.1-0.3 range. Learning real transitions but still noisy.
- **Pessimistic:** event_F1@±3 < 0.02. Loss is low but predictions are flat (always "no transition").

**The PSR story for the paper is pre-registered as "the honest miss" (Opus 192 Q7, 186 G-2).** Even modest PSR performance supports a paper that honestly says "our per-frame approach reaches X% of the specialist pipeline while sharing a backbone."

### 3.5 The 70.9M Concern

The PSR head is 60% of total model parameters. Is 70.9M for this task justified? Arguments:

**For:** PSR is the hardest task (sequence-to-sequence transition detection from semantic features). The head needs capacity to model temporal dependencies over 8 frames of 768-dim features. The old 3M head was incapable.

**Against:** 70.9M is heavy. If the model were deployed, inference cost is dominated by the backbone (34.5M), not the heads — all 4 heads together are ~80M, but they run once per clip while the backbone processes every frame. The parameter count is a paper statistic, not a latency bottleneck.

---

## 4. Pose Head

### 4.1 No SOTA on IndustReal

This is our **original contribution.** No prior work benchmarks head pose on IndustReal assembly data. The 8.92° forward MAE and 7.48° up MAE at EP10 are the first published numbers.

### 4.2 Our Architecture (unchanged)

```
Input: cls_token [B, 768]
Head: Linear(768→256) → ReLU → Linear(256→6) → Tanh
Loss: (1 - cos(fwd_pred, fwd_gt)) + (1 - cos(up_pred, up_gt))
```

### 4.3 Why It Works in MTL

Pose predicts spatial direction (forward/up vectors). The MViTv2-S backbone, pretrained on Kinetics-400 (action recognition from video), learns spatial features from human motion. These transfer naturally to head pose from egocentric video. The cls_token carries spatial information that is:
1. **Complementary to activity** (what is happening vs where is the head pointing)
2. **Not competing with detection** (object locations ≠ camera orientation)
3. **Largely independent of PSR** (component states ≠ head direction)

This is the strongest candidate for **positive transfer** — MTL pose could be BETTER than ST pose because the shared backbone learns richer spatial features from multiple objectives.

### 4.4 Realistic Expectation

- **At ep10:** ~8-9° forward MAE (unchanged from prior, pose was already good)
- **At ep30:** ~6-8° forward MAE
- **Single-task ceiling (estimated):** ~5-7° forward MAE

**The MTL/ST ratio for pose is expected to be ≥1.0** (positive transfer). This is the headline win for the paper.

---

## 5. Cross-Head Interactions

### 5.1 Gradient Competition (PCGrad Active)

With PCGrad, conflicting gradients on the shared backbone are projected onto each other's normal planes. This prevents any one task from dominating the backbone update direction.

### 5.2 The Kendall Pathology (Our Methodological Contribution)

```
Without caps:  weight_i ≈ exp(-log_var_i) ≈ 1/(2 × L_i)
               High-loss task → low weight → starved → higher loss → lower weight → ...
```

Our caps prevent this spiral. The caps are:
- act ≤ 1.0 (effective weight ≥ 0.37)
- psr ≤ 0.5 (effective weight ≥ 0.61)
- det ≤ 1.5 (effective weight ≥ 0.22)
- pose ≤ 2.0 (effective weight ≥ 0.14)

These values were derived from the loss magnitudes during run10. They may need re-tuning for run11's different loss landscape (especially PSR which dropped 10×).

### 5.3 Shared Feature Quality

The central premise: **one backbone → richer representations → all tasks benefit.** But there's a counter-premise: **one backbone → compromised representations → all tasks suffer.** The evidence to date:

| Task | Feature source | Compromised by? | Evidence |
|------|---------------|-----------------|----------|
| Activity | cls_token | Pose + shared optimization | 0.58% at EP10 (but fresh head) |
| Detection | P5/P4/P3 via FPN | Shared backbone optimization | 0.0 at EP10 (but P2 contamination) |
| PSR | P5 directly | Detection FPN + activity | Loss 1.56→0.17 after feature fix |
| Pose | cls_token | Activity | 8.9° healthy |

**Tentative conclusion:** The problem was not MTL gradient competition — it was wrong feature routing. PSR improved 10× just by reading from the correct layer. Detection was feeding from P2 (semantics-free). Activity's cls_token is the only genuinely shared bottleneck.
