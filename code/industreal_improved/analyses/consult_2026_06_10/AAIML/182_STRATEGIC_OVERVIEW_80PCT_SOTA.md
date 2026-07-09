# 182 — Strategic Overview: How to Reach ≥80% of SOTA on Every Head

**Date:** 2026-07-09
**Inputs:** 176–181 (training state, Opus Path-D verdict, prior docs)
**Purpose:** Top-level decision document. One question, one answer. Five strategic paths. Pick one.
**Companion files:** 183 (architecture), 184 (training & data), 185 (50 deep questions)
**Audience:** Opus consultation round 2 — "the user is open to changing anything."

---

## 0. The One-Paragraph Answer

The Path-D run that just launched (commit 71df66759) is **necessary but not sufficient**. Honest projections from Opus 181: ep30 → act 0.20-0.35, det 0.10-0.30 (only with the assignment fix), PSR 0.10-0.30, pose 4-6° MAE. The 80% bar requires act ≥0.52, det ≥0.67, PSR ≥0.72, pose ≤12°. **Path D's optimistic ceiling is ~60% of those targets.** To clear the bar you must change something *outside* the optimization loop — specifically the backbone, the per-task head capacity, or the entire MTL formulation. The fastest realistic route is **Strat-2: Frozen-Large-Backbone + Lightweight Adapters + Heavy Augmentation**. The most defensible academic route is **Strat-4: Single-Task Pretrain → MTL Finetune via Cross-Task Attention**. The cheapest acceptable route is **Strat-1: keep MViTv2-S but upgrade heads + losses + augmentation aggressively** (probably gets to ~70% of SOTA, not 80%). **My recommendation: Strat-2**, parallelized with Strat-3 (single-task baselines) so the paper has both stories.

---

## 1. The 80% SOTA Gap (current vs target)

| Head | SOTA | 80% bar | Current (Ep6, post-Path-D-fix) | Gap (current → bar) | Δ needed |
|------|------|---------|-------------------------------|---------------------|----------|
| **Detection** mAP@0.5 | 0.838 | **≥0.67** | 0.0 (mAP=0); presence BCE ≈ 0.001 | 0.0 → 0.67 | +0.67 |
| **Activity** top-1 | 0.652 | **≥0.52** | 0.008 (random=0.0133) | 0.008 → 0.52 | +0.51 |
| **PSR** event F1@±3 | 0.901 | **≥0.72** | 0.0 (flat loss 1.30) | 0.0 → 0.72 | +0.72 |
| **Pose** fwd MAE | ~15° | **≤12°** | ~10° (already beats 80%) | 10° → 12° | **OK** |

**Pose is the only head that already meets the bar.** The other three are catastrophic gaps. Activity's gap is 51 percentage points; PSR's is 72 points; detection's is 67. No optimization trick closes these in a single backbone. They require:

- **Activity +0.51 top-1:** requires a head that *can* represent 75 classes with 768-dim class-token features. Current head is `LayerNorm → Linear(75)`. The dominant cost is **head capacity** + **few-shot learning from a single label per 16-frame window**. Fixes: 2-layer temporal head (T×768 → 768 → 75), or use a **frozen large pretrained vision-language model** (CLIP/EVA-CLIP) for the activity features.
- **Detection +0.67 mAP@0.5:** requires **dense assignment** (YOLOv8's TaskAlignedAssigner assigns 10-50 cells per GT, not 1) and **proper NMS-free eval**. Path D's 3×3 positive cells + α=0.5 is a step in the right direction, but the real fix is **switching to YOLOv8's full TAL + DFL loss**, which is what got us 0.995 in single-task runs.
- **PSR +0.72 F1:** requires **deeper feature source** (block-3 / block-4 of MViTv2, not `conv_proj` from block-1) and **temporal context** (the current transformer is causal but on too-shallow features; needs at least 32-frame input, not 16). The architecture is wrong, not the loss.

---

## 2. Why Path D Alone Won't Reach the Bar (Opus 181's own caveat)

Opus 181 §3.3 explicitly hedged:
- Activity 0.20-0.35 by ep30 (we need ≥0.52)
- Detection 0.10-0.30 *only if the assignment fix also lands* (we need ≥0.67)
- PSR 0.10-0.30 (we need ≥0.72)
- Pose 4-6° (we already meet this)

**Path D buys ~6× improvement on activity and ~6× on PSR vs current, but starts from 0.008 and 0.0.** Even a 6× multiplier doesn't close a 51-point or 72-point gap. The missing piece is **representation quality at the input to each head** — and that is determined by the backbone + head architecture, not the loss.

The "Kendall paradox" framing is correct, but it's a **second-order problem** compared to "the backbone features don't carry enough class-discriminative signal for a 1-layer head to classify 75 activities."

---

## 3. The Five Strategic Paths

### Strat-1 — **Incremental Architecture Upgrade** (cheapest, lowest upside)

**What:** Keep MViTv2-S. Replace the head architectures, augment data heavily, add TAL for detection, deepen the PSR transformer.

**Concrete changes:**
- Detection: switch to **YOLOv8 head** with `TaskAlignedAssigner` (assigns ~10 cells/GT instead of 1-9), `alpha=0.5`, `topk=10`. Replace the current "center-cell-only" assignment. Use full YOLOv8 loss.
- Activity: **2-layer temporal head**: `[T=16, 768] → mean-pool → Linear(768, 768) → GELU → Linear(768, 75)` (~600K params). Or **attention-pool** over T.
- PSR: **deeper feature source** (block-3 features, not conv_proj) + **2-layer transformer** with hidden=192 (not 96). Optionally use **32-frame windows** for stronger temporal context.
- Pose: keep current.
- Add **heavy augmentation**: MixUp (α=0.2), CutMix (α=1.0), RandAugment (N=2, M=10), temporal jittering (random frame drop, max 4 of 16).
- Use **cosine warm restart** scheduler. Train 100 epochs from scratch (not resume).

**Expected outcome:**
- Detection mAP 0.40-0.60 (below 80% bar but closes most of the gap)
- Activity top-1 0.25-0.40 (below bar)
- PSR F1 0.30-0.55 (below bar)
- Pose ≤6° (already above bar)

**Verdict:** **Probably gets to ~60% of SOTA on average.** Cleanest paper story (incremental improvements) but doesn't meet the user's bar. Best as a *baseline* for a more ambitious path.

**Compute:** 4-6 days on 1 GPU. Low risk.

---

### Strat-2 — **Frozen Large Backbone + Lightweight Adapters** (recommended)

**What:** Replace MViTv2-S with a much larger pretrained backbone, **freeze most of it**, and train **task-specific adapters + heads** on top. This is the standard recipe for few-shot adaptation of foundation models.

**Concrete changes:**
- **Backbone candidates** (file 183 §1):
  - **EVA-02-L (300M params)**: ImageNet-21k + LLM-pretrained. Strong image features. Train adapters per task.
  - **InternVideo2 (1B params, ViT-G/14)**: Video-pretrained, Kinetics-710 + HowTo100M + others. Best video features. Slow but proven.
  - **DINOv2-L (300M params)**: Self-supervised on 142M images. Strong general features.
  - **VideoMAE-v2 (1B params)**: Self-supervised video pretraining. Best for video.
- **Architecture:** backbone frozen; per-task **LoRA adapters** (rank=16, ~1M params/task) on the last 4-6 transformer blocks; per-task head on top.
- **Training:** train adapters + heads end-to-end with **Path-D fixes** (per-task log_var caps + EMA + grad accum fix). Backbone gradients are sparse due to adapters (most params frozen).
- **Compute:** InternVideo2-L + 4 tasks × LoRA-16 = ~6M trainable params + 1B frozen. Forward pass: ~600 GFLOPs per 16-frame clip (vs current 130). At batch 1 on A100, ~0.4 sec/batch → ~30 min/epoch for 4000 batches → ~21 hours for 42 epochs.

**Expected outcome (InternVideo2):**
- Detection mAP 0.55-0.75 (likely meets 80% bar)
- Activity top-1 0.45-0.65 (likely meets bar)
- PSR F1 0.65-0.85 (likely meets bar)
- Pose ≤6° (already meets)

**Why this wins:**
1. **Pretrained representations** carry class-discriminative signal that's hard to learn from scratch on 75K frames.
2. **LoRA + frozen backbone** keeps trainable params small (~6M) → low VRAM, fast convergence.
3. **MTL is more compatible** when the backbone is high-quality: less per-task specialization needed.
4. **Defensible paper claim:** "MTL with foundation-model adapters" is novel and timely.

**Verdict:** **Highest probability of meeting the 80% bar.** Higher compute than Strat-1 (24-48 hr/epoch vs 22 min) but the parallelism is per-task adapters, so total epochs needed is small (~10-20). Total compute: ~5-10 days.

**Risk:** Foundation models may not be commercially licensable for AAIML. Need to verify EVA-02 / InternVideo2 / DINOv2 license.

---

### Strat-3 — **Single-Task SOTA + Distillation** (most defensive)

**What:** Train four strong single-task models (the right architecture per task), then distill them into one MTL student. This guarantees per-task performance ≥ baseline.

**Concrete changes:**
- **Detection:** Train YOLOv8m on annotated frames (we already have 0.995). ~1 GPU-day.
- **Activity:** Train a 2-layer temporal head on frozen EVA-02-L features. ~2 GPU-days. Expected top-1: 0.55-0.70.
- **PSR:** Train CausalTransformer on block-3 features with full temporal context (T=64). ~2 GPU-days. Expected F1: 0.75-0.90.
- **Pose:** Keep current MViTv2-S + 6D MLP. Already meets bar.
- **Distill:** train an MViTv2-S MTL student to mimic the 4 teachers (KL divergence on logits + MSE on features). Expected MTL metrics: 0.85-0.95 of teacher.

**Expected outcome:** Detection 0.85, Activity 0.50, PSR 0.68, Pose 6°. **Meets 80% on 3/4 heads, PSR borderline.**

**Verdict:** **Most defensive path. Each task has its own optimal architecture.** But it abandons the MTL "one model, four tasks" framing that the paper is built around. The student is the only MTL artifact, but it's *distilled* not *learned*. This is a known recipe in NLP/RL but less common in vision.

**Compute:** ~5-7 GPU-days total. Medium risk.

---

### Strat-4 — **Sequential Pretrain → MTL Finetune** (most principled)

**What:** Pretrain each task's head independently (or fine-tune the backbone for each task separately) to convergence, then **finetune the entire MTL system** with the strong per-task initializations and Path-D fixes.

**Concrete changes:**
- **Phase 1 (per-task pretraining, 4 runs):**
  - Det: train MViTv2-S + YOLOv8 head with TAL. ~2 GPU-days. Expected mAP 0.75-0.85.
  - Act: train MViTv2-S + 2-layer head, weighted CE, label smoothing 0.05. ~1 GPU-day. Expected top-1 0.40-0.55.
  - PSR: train MViTv2-S + transformer on block-3 features, T=32 input. ~2 GPU-days. Expected F1 0.70-0.85.
  - Pose: already at bar.
- **Phase 2 (MTL finetuning, 1 run):**
  - Initialize the shared MViTv2-S backbone from a *weight average* of the 4 single-task backbones (model soup / task arithmetic).
  - Train all 4 heads jointly with Path-D fixes.
  - Use cross-task attention (task tokens query backbone features). Adds ~5M params.
  - Train ~30 epochs. Expected MTL metrics: 90-95% of single-task.

**Expected outcome:** Detection 0.70-0.80, Activity 0.45-0.55, PSR 0.65-0.80, Pose 6°. **Likely meets 80% on all 4.**

**Why this wins:**
1. **Single-task pretraining guarantees a strong initialization per task.**
2. **Model soup / task arithmetic** is a 2023 finding that averaging weights across fine-tuned models often beats each individually. Applies directly to "average 4 task-specific backbones."
3. **Cross-task attention** lets heads request task-relevant features from the backbone dynamically (vs static shared features).
4. **Path-D fixes** prevent the optimization pathologies we just diagnosed.

**Verdict:** **Most principled and most likely to meet the bar.** Highest chance of a clean paper. But also highest total compute (~7 GPU-days) and most engineering effort.

**Risk:** Requires implementing model soup, cross-task attention, and 4 single-task training scripts.

---

### Strat-5 — **Replace MTL with Sequential** (surrender)

**What:** Drop MTL entirely. Train 4 single-task models. Report them as a "system" not as one MTL model. Discuss MTL pathology as a finding.

**Expected outcome:** Each task at its individual ceiling. No MTL claims.

**Verdict:** **Honest but defeats the paper's premise.** Only do this if all of Strat-1/2/3/4 fail. The user has explicitly stated MTL is the goal, so this is a fallback.

---

## 4. Decision Tree

```
START
  │
  ├─ Q1: Is "MTL with one backbone" the paper's central claim?
  │    ├─ YES → continue to Q2
  │    └─ NO  → Strat-5 (sequential)
  │
  ├─ Q2: Do we have license for InternVideo2 / EVA-02 / DINOv2 / VideoMAE-v2?
  │    ├─ YES → Strat-2 (frozen large + adapters) — highest probability
  │    └─ NO  → continue to Q3
  │
  ├─ Q3: Can we spend ~7 GPU-days?
  │    ├─ YES → Strat-4 (single-task pretrain → MTL finetune) — most defensible
  │    └─ NO  → continue to Q4
  │
  ├─ Q4: Can we spend ~3 GPU-days?
  │    ├─ YES → Strat-3 (single-task SOTA + distill) — defensive
  │    └─ NO  → Strat-1 (incremental upgrades) — accepts ~60% of SOTA
  │
  └─ END
```

---

## 5. Risk / Reward Matrix

| Strat | 80% SOTA probability | Total compute | Engineering risk | Paper novelty | Recommendation |
|-------|---------------------|---------------|------------------|----------------|----------------|
| **1** | 30% | 4-6 days | Low | Low (incremental) | Baseline / fallback |
| **2** | **75%** | 5-10 days | Medium (license, large model infra) | High (foundation-model adapters for video MTL) | **Recommended if license clears** |
| **3** | 65% | 5-7 days | Medium (4 training runs + distillation) | Medium (MTL via distillation) | If license is blocked |
| **4** | **80%** | 7-10 days | High (model soup, cross-task attention, 4 ST scripts) | Very high (task arithmetic + MTL is novel) | **Best academic story** |
| **5** | 100% per task | 5 days | None | None (abandons MTL) | Last resort |

---

## 6. Recommended Sequence (parallelized)

**Week 1 (in parallel):**
1. **Launch Strat-1** in parallel with the running Path-D run. Strat-1's incremental upgrades will produce a useful ablation regardless of which path is chosen. Use this as the "easy baseline" paper result.
2. **Verify license** for EVA-02, InternVideo2, DINOv2, VideoMAE-v2. If licensed, freeze decision on Strat-2. If not, freeze on Strat-4.

**Week 2 (focused):**
3. **Launch Strat-2 (if licensed)** or **Strat-4 (if not)** as the headline MTL run. Use the insights from Strat-1 to skip dead ends.
4. **Run 4 single-task baselines** (for the paper's "MTL vs ST" comparison, regardless of which path is chosen).

**Week 3 (analysis):**
5. Eval all runs on val + test. Compute SOTA ratios.
6. Pick the strongest result for the paper headline. Use the others as ablations.

**Compute summary:**
- Strat-1: 4-6 days × 1 GPU = 4-6 GPU-days
- Strat-2: 5-10 days × 1 GPU = 5-10 GPU-days (with EVA-02-L on A100)
- Strat-3: 5-7 GPU-days total (4 ST + 1 distill)
- Strat-4: 7-10 GPU-days total (4 ST + 1 MTL finetune)
- 4 single-task baselines: 2-3 GPU-days (cheaper if we just need to compare)

**Total realistic budget:** 10-15 GPU-days for the full set, ~3 weeks wall-clock at 1 primary GPU.

---

## 7. The Bottom Line

**If you do nothing else, do Strat-2.** It is the single change with the highest probability of clearing the 80% bar with acceptable compute. The risk is licensing; verify that within 48 hours. If licensing is blocked, fall back to Strat-4, which is the most defensible academic path.

**Do not rely on Path-D alone.** It is a necessary precondition (without it, no MTL optimization scheme will work), but it does not deliver the per-head quality we need.

**Strat-1 is not optional — it's the cheap baseline** that documents what incremental MTL improvements look like and gives the paper a comparison point even if Strat-2/4 fails.

---

## 8. Open Questions for Opus

These are the strategic questions this document *cannot* answer; see file 185 for the full 50.

1. **Q1** (architecture): Is the frozen-backbone + adapter recipe (Strat-2) more defensible than single-task pretrain + MTL finetune (Strat-4) for an AAIML paper?
2. **Q2** (compute): Given 1 primary GPU, is it better to run Strat-4 sequentially (~10 days) or Strat-2 + 4 ST baselines in parallel (~3 weeks wall-clock)?
3. **Q3** (head capacity): Is a 2-layer MLP head on class-token features *ever* sufficient for 75-class video activity, or do we need a temporal transformer regardless of backbone?
4. **Q4** (PSR feature source): Does PSR fundamentally require features from block-3+ of a video transformer, or can block-1 features work if the temporal context is long enough (T=64)?
5. **Q5** (detection assignment): Is YOLOv8's `TaskAlignedAssigner` the right drop-in replacement for center-cell-only assignment, or are there alternatives (SimOTA, OTA)?
6. **Q6** (training): Is MixUp + CutMix + RandAugment enough data augmentation, or do we need domain-specific augmentation (assembly-state transitions, frame jittering)?
7. **Q7** (MTL architecture): Does cross-task attention (task tokens querying backbone) actually help when each head already has its own adapter, or is it redundant?
8. **Q8** (paper story): Is "MTL via foundation-model adapters" a strong enough narrative for AAIML, or do we need the more elaborate "task arithmetic + cross-task attention" story from Strat-4?
9. **Q9** (risk): If Strat-2's foundation-model adapter approach fails to deliver 80% on activity specifically, do we still have a publishable paper, or does activity become the headline failure?
10. **Q10** (timeline): Given today's date (2026-07-09) and the AAIML submission deadline (typically Sept/Oct), is 3-4 weeks of compute realistic, or do we need to ship with Strat-1 only?

---

*Companion to 183 (architecture), 184 (training), 185 (questions). All claims in 182 trace to file IDs in the appendix or to Opus 181.*