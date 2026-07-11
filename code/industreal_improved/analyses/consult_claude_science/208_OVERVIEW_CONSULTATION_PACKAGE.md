# 208 — Claude Science Consultation: Overview & Package Guide

**Document:** 208 of 227 (Claude Science consultation package, docs 208–227)
**Status:** Entry point — read this first
**Date:** 2026-07-11
**Audience:** Claude Science (Anthropic's academic paper research capability)
**Hypothesis under test:** *MTL helps, not hurts — a single multi-task model can beat its own single-task baselines AND achieve SOTA on all four heads while being dramatically more parameter/FLOPs/time efficient than four separate ST models.*

---

## Table of Contents

1. Who We Are and What We Are Building
2. The Hypothesis: The Claim We Need to Prove
3. What We Have Already Tried (and What Did Not Work)
4. What Claude Science Should Help Us Find
5. How to Use This Document Package
6. Success Criteria: What a Useful Response Looks Like
7. Summary of the 50 Questions (Doc 215)
8. Quick Reference: Package Map

---

## 1. Who We Are and What We Are Building

We are a small academic research team with access to consumer-grade GPUs (RTX 3060 12GB, RTX 5060 Ti 16GB). We are building a multi-task learning (MTL) model for industrial assembly verification on the IndustReal dataset — an egocentric video dataset of real human assembly workers performing procedural tasks on a transmission assembly line.

Our model performs four distinct perception tasks simultaneously from a single video frame input:

**Task 1 — Detection (Assembly State Detection / ASD):** 24-class object detection of assembly components (bolts, washers, plates, fixtures, tools, error states) at 224px input resolution. Metric: mAP@0.5. Uses a feature pyramid network (P2–P5) with decoupled classification + regression heads and Task Alignment Learning (TOOD-style TAL) assigner.

**Task 2 — Activity Recognition:** 75-class fine-grained activity classification of assembly actions. Metric: top-1 accuracy. Uses the backbone's CLS token through a 3-layer MLP (768 -> 2048 -> 1024 -> 75). Long-tail distribution: power-law over 75 classes, tail classes with 1–5 samples.

**Task 3 — PSR State Detection (Pick-Place, State, Reach):** 11-state per-frame transition prediction (PSR: picking, placing, reaching, holding, inspecting, etc.). Metric: event-F1@3-frame tolerance. Uses P5 spatial features -> spatial pooling -> Linear projection -> 2-layer causal Transformer (d=256) -> per-frame logits. After a diet from 70.9M to ~1.78M parameters, this head is now appropriately sized for an 8-token -> 88-logit mapping.

**Task 4 — Head Pose Estimation:** 6D continuous rotation estimation (forward and up vectors) of the worker's head in egocentric video. Metric: forward-vector MAE in degrees. Uses the CLS token through a small MLP (768 -> 256 -> 6) -> Tanh -> Gram-Schmidt orthonormalization to produce valid rotation matrices. Geodesic loss on SO(3).

**Shared backbone:** MViTv2-S (34.5M parameters, Kinetics-400 pretrained, 81.0% K400 top-1). The same backbone serves all four heads through a single forward pass. Feature maps are extracted at stages P2 (96ch, 56x56), P3 (192ch, 28x28), P4 (384ch, 14x14), P5 (768ch, 7x7) via forward hooks.

**Total MTL model:** ~48.6M parameters (34.5M backbone + ~14.1M heads + FPN) versus ~100M parameters for four separate single-task models (~2.06x parameter efficiency). Single-forward-pass latency means the MTL model is also ~4x faster at inference.

**Training configuration:** 39K batches per epoch, effective batch size 16 (4 micro-batches x 4 gradient accumulation steps), mixed precision (bf16), AdamW optimizer, Cosine annealing LR schedule. Kendall uncertainty weighting (learned per-task log variances) + PCGrad gradient surgery for multi-task optimization. EMA loss normalization, log-var caps (det: 4.0, act: 1.0, psr: 0.5, pose: 4.0).

**Six active levers for closing the MTL gap:**
1. PSR monotonicity constraint (once-on, stays-on per recording, median filter)
2. Detection threshold calibration (F1-based threshold sweep)
3. SWA checkpoint averaging (last 5 periodic checkpoints)
4. Head warm-starting from ST best checkpoints
5. Knowledge distillation from ST teachers (KL-div for classification heads, MSE for regression heads)
6. Full training budget (50 epochs)

---

## 2. The Hypothesis: The Claim We Need to Prove

Our central hypothesis is deliberately ambitious. We do not merely want to show that MTL "works reasonably well" or that the efficiency trade-off is acceptable. We want to prove that multi-task learning provides a *net positive* — that sharing a single backbone across four diverse perception tasks results in representations that are *better* for each task than training that task in isolation, while also being dramatically more efficient.

Concretely, we need to demonstrate:

**Claim 1 — MTL beats ST (our own baselines):** For each head, the MTL model achieves at least comparable (and ideally superior) metrics to a matched-architecture single-task model trained on the same data at the same input resolution. Our targets: pose MTL/ST >= 0.95, activity MTL/ST >= 0.70, detection MTL/ST >= 0.60, PSR event-F1 > 0.25 with monotonicity.

**Claim 2 — MTL beats SOTA (published benchmarks):** For each of the four tasks, our single MTL model achieves results competitive with or exceeding published state-of-the-art results for the IndustReal dataset specifically. This is complicated by advantage gaps (SOTA systems may use larger inputs, synthetic data, COCO pretraining, or procedural pipelines), so we decompose the gap honestly: SOTA gap = (SOTA - our ST ceiling) + (our ST ceiling - our MTL). The first term belongs to data/recipe differences; the second term is our paper's claim.

**Claim 3 — Dramatic efficiency advantage:** Our MTL model at ~48.6M parameters in a single forward pass is ~2x more parameter-efficient and ~4x more latency-efficient than deploying four separate ST models totaling ~100M parameters. This claim is the paper's efficiency spine and must survive reviewer scrutiny.

**The paper frames this as:** "What if multi-task learning were not a compromise you accepted for efficiency, but a strategy you chose because it produces better representations?"

This is not a settled question in the literature. The dominant finding in multi-task learning is that task interference degrades per-task performance relative to single-task baselines, and the entire field of MTL optimization (gradient surgery, loss weighting, architectural isolation) exists to *mitigate* this degradation. We want to show that for the specific case of egocentric assembly perception — where tasks are related through the shared structure of a procedural manual task — the interference can become *positive transfer*.

---

## 3. What We Have Already Tried (and What Did Not Work)

This consultation package is the product of a long journey. Documents 150 through 207 (the preceding 58 documents in our analysis series) document five rounds of deep consultation with Claude Opus, extensive architecture iteration, bug discovery and fixing, and methodology refinement. Here is a summary of what we have tried and what it taught us.

### Architecture iterations

**Initial design (Tier A, docs 150–175):** ConvNeXt-Tiny backbone (28M params) with five task heads operating at mixed resolutions. This architecture achieved preliminary results but suffered from: (a) ~117.7M total parameters (an oversized PSR head accounted for 70.9M alone), (b) a detection death spiral where gradient conflict caused collapse to predicting all negatives, (c) activity recognition stuck near random (26% top-1 on 75 classes), (d) PSR loss stuck at ~1.56 with the backbone's conv_proj features.

**PSR feature routing discovery (doc 181):** The critical fix was routing PSR from backbone stage P5 (768-dim, 7x7 spatial) instead of conv_proj features (96-dim, 56x56). This single change dropped PSR loss from 1.56 to 0.17. The feature source, not model capacity, was the bottleneck.

**MViTv2-S migration (docs 185–192):** We migrated from ConvNeXt-Tiny to MViTv2-S (34.5M, Kinetics-400 pretrained). This backbone provides native spatiotemporal pooling, superior Kinetics-400 pretraining (81.0% top-1), and a natural feature pyramid for detection. The activity head immediately improved from 26% to ~40% with a frozen random backbone (which turned out to be a false positive in the probe — the head was learning, but the frozen CLS token was information-limited).

**PSR diet (docs 201–207):** We identified that the PSR head at 70.9M was 40x oversized for its task (8 tokens -> 88 logits). We designed a diet: Linear(768->256) projection + 2-layer Transformer (d=256, ff=1024) + Linear(256->11) = ~1.78M. This restores the efficiency spine (~48.6M total) while maintaining PSR performance.

**Detection-conditioned PSR (proposed doc 203, rejected doc 207):** We proposed a 5.2M detection-conditioned PSR head that would fuse detection features with PSR features. Doc 207 found a critical bug: the spec used mean-pool-then-expand, producing identical logits for all 8 frames, making per-frame transition localization impossible. This architecture was deferred.

### Training methodology tried

**Kendall uncertainty weighting:** Learned per-task log variances are our primary multi-task loss balancing mechanism. We discovered a critical bug where log-var caps in the code (det: 4.0, pose: 4.0) were different from what the documentation claimed (det: 1.5, pose: 2.0). At cap 4.0, detection's effective weight floor is ~0.018 — 12x weaker than intended. We also discovered that EMA normalization (which feeds O(1) losses to Kendall) mitigates cap binding but does not eliminate it.

**PCGrad gradient surgery:** Every training step projects conflicting task gradients to remove conflicting components before updating the shared backbone. This adds ~2x backward computation but has been essential for preventing the detection head from being starved by the activity head's larger gradients.

**Warm-starting from ST checkpoints:** MTL heads are initialized from pre-trained single-task checkpoints. This primarily helps activity (which needs the pre-trained classifier weights) and detection (which needs the TAL assigner to stabilize early in training).

**Knowledge distillation from ST teachers:** The MTL model is distilled from four single-task teacher models via KL-divergence (for detection and activity classification) and MSE (for PSR and pose regression). This is our most powerful lever for closing the MTL gap.

**SWA (Stochastic Weight Averaging):** The last 5 periodic checkpoints are averaged at the end of training for a ~0.5-2% across-task improvement at zero additional training cost.

### Diagnostics and probes

**Overfit probe:** For each head, we train on a tiny subset (50 images) with frozen random backbone to verify that (a) the head can overfit to near-zero loss, and (b) the eval code correctly measures the metric. Results: pose and PSR pass trivially (6.2 deg MAE, 91% positive). Activity and detection produced false negative results — they showed the heads DO learn (activity improved from 26% to 40.5% top-1) but are limited by the frozen random backbone's features. These are false negatives of the probe design, not curriculum failures.

### What we learned and what changed our direction

Several key insights emerged from the Opus consultation rounds that reshaped our approach:

1. **The problem is not architecture capacity; it is data volume and eval correctness.** Opus Round 5 (doc 207) forcefully demonstrated that our architecture documents describe intentions as completions — claimed code changes (PSR diet, logit-adjust, kendall-uncapped flag, overfit_probe rewrite) were absent from the repository. The honest diagnosis: our model's struggles stem from eval bugs yet to be ruled out, loss/imbalance handling, and data volume, in that order.

2. **Per-task backbone adapters break the single-pass claim.** LoRA adapters on backbone Q/V projections would require four backbone forward passes (one per task), deleting the single-forward-pass latency claim that is the surviving half of our efficiency story.

3. **The VideoMAE backbone swap was built on a wrong number.** VideoMAE ViT-B does not score 87.4% on Kinetics-400 (that is a ViT-L or ViT-H number; ViT-B is ~81.5%). At true published numbers, the swap buys ~0.5 points of pretraining quality for 2.5x backbone parameters, making it self-defeating.

4. **Nash-MTL would delete our Kendall contribution.** Swapping from Kendall-caps + PCGrad to Nash-MTL mid-paper forfeits our core methodological contribution (Kendall-collapse characterization and the capped-log-var fix). Nash-MTL belongs as an ablation row in a later paper.

5. **Detection augmentation is a zero-parameter lever with the largest published upside.** Mosaic and Copy-Paste augmentations (already implemented but never activated) can add +3-5 mAP on small detection datasets at zero architecture cost. This should be the first move for detection, before any architecture change.

6. **The paper spine was right from the start.** Three contributions have been stable across all five Opus rounds: (a) Kendall-collapse characterization + fix (the capped log-var ablation), (b) measured per-task transfer map (MTL/ST ratios with confidence intervals), (c) genuine ~2x parameter efficiency at single-pass latency. Everything else is a menu to order from after the diagnostics are complete.

---

## 4. What Claude Science Should Help Us Find

Despite five rounds of consultation with Claude Opus and a thorough reading of the multi-task learning literature we are aware of, we are certain that the academic literature contains methods, architectures, and training recipes we have not considered. Claude Science's ability to scan hundreds of papers systematically is the capability we need.

We need Claude Science to find published work in the following categories:

### Category A: Closing the MTL-to-ST performance gap

The fundamental challenge of multi-task learning is that sharing a backbone creates gradient interference between tasks. We need papers that demonstrate how to *close the gap* — ideally, how to achieve MTL > ST, not just MTL ~= ST.

Specific sub-questions:
- Are there published cases where an MTL model *exceeds* its matched single-task baselines on all tasks? What conditions enable this?
- What is the published evidence for *positive transfer* between perception tasks in egocentric/industrial video? Under what conditions does positive transfer occur?
- Are there training schedules (curriculum learning, progressive task addition, alternating task training) proven to reduce interference?
- What is the role of task similarity? Do more similar tasks produce more positive transfer, or is there a U-shaped relationship (too similar -> redundant, too different -> interference)?
- Are there theoretical bounds on the MTL/ST ratio from the optimization literature?

### Category B: Multi-task training methodologies beyond what we have tried

We currently use Kendall uncertainty weighting + PCGrad + EMA normalization + log-var caps. What else exists?

- Gradient surgery variants: GradVac, CAGrad, GradDrop, IMTL-G, Nash-MTL (real Nash-MTL, not the pseudocode from doc 204), MGDA, and any newer methods.
- Loss weighting: Dynamic Weight Averaging (DWA), Uncertainty Weighting with robust variance estimation, Geometric Loss Strategy, Dynamic Task Prioritization, gradient normalization (GradNorm).
- Two-stage training: Methods that train in a "shared first, specialized second" pattern, or that progressively unfreeze task-specific parameters.
- Regularization-based MTL: Cross-task distillation, contrastive task regularization, information bottleneck approaches for MTL.
- Optimization tricks: Per-task learning rates, per-task normalization statistics, task-specific batch normalization.

### Category C: Architecture designs that beat ST baselines

Are there architectural patterns where shared representations boost rather than hurt individual tasks?

- Task-specific routing: Adapter networks, task-specific attention heads within the shared backbone (not full per-task adapters that break single-pass), conditional computation, mixture-of-experts with task routing.
- Feature gating and modulation: FiLM, task-specific feature selection, attention-based feature routing.
- Decoupling strategies: What is the optimal balance between shared and task-specific parameters? Is there a published "sweet spot" for the ratio of shared to task-specific parameters?
- Multi-task transformer architectures: Any published architectures specifically designed for video MTL (as opposed to applying image MTL methods to video)?
- Neural architecture search for MTL: Any NAS methods that discover MTL-optimal architectures?

### Category D: Multi-task optimization objectives beyond task weighting

We use standard per-task losses (focal-BCE for PSR, cross-entropy for activity, CIoU + DFL + QFL for detection, geodesic loss for pose). Could the objectives themselves be improved?

- Loss function design for MTL: Task-level loss calibration, loss normalization across tasks with different scales, Pareto-optimal multi-task objectives.
- Representation learning objectives: Should we add auxiliary self-supervised losses (contrastive, masked autoencoding, temporal ordering) to the shared representation to improve all tasks?
- Prototypical networks for few-shot MTL tail classes.
- Energy-based models or structured prediction losses that capture task relationships.
- Task-conditional normalization or rescaling of feature representations.

### Category E: Published work where an MTL model beat SOTA on multiple tasks simultaneously

This is the hardest ask but the most valuable for our paper.

- Any published system that achieved simultaneous SOTA on 3+ diverse tasks (e.g., detection + recognition + pose estimation).
- Any MTL system that beats separate SOTA models on the same benchmark, even on a subset of tasks.
- Any MTL paper that uses an *efficiency argument* (params, FLOPs, latency) as a primary contribution alongside accuracy — not as a secondary "oh by the way" figure.
- Any industrial/manufacturing MTL paper (assembly verification, quality inspection, human-robot collaboration) that achieves both accuracy and efficiency claims.

### Category F: Data augmentation and data efficiency for MTL

- Augmentation strategies specifically designed for multi-task learning (across-task consistency, task-specific augmentation, mixup variants for MTL).
- Semi-supervised and self-supervised MTL approaches for small datasets.
- Active learning for MTL: Which tasks benefit most from additional labels?
- Synthetic data generation for industrial assembly: Domain randomization, neural rendering, diffusion-based data augmentation.

### Category G: Efficiency analysis methodology

- How do published MTL papers handle the comparison with ST baselines? What statistical methods do they use?
- How should we report FLOPs and latency fairly (single MTL model vs. ensembled ST models)?
- Are there accepted benchmarks for MTL efficiency, or standard comparison tables we should replicate?
- What is the precedent for "parameter efficiency" as a primary contribution in a vision paper?

---

## 5. How to Use This Document Package

This consultation package contains 20 documents (208–227). They are designed to be consumed in a specific way depending on what you need to know.

### Quick entry points

| If you want to... | Start with... |
|---|---|
| Understand the full picture | This document (208) — it tells you everything at a high level |
| See the complete history of what we tried | Doc 209 — the full journey from concept to current state |
| Understand our architecture in detail | Doc 210 — FPN design, head structures, bottlenecks |
| Understand our training methodology | Doc 211 — Kendall, PCGrad, EMA, SWA, distillation |
| See per-head performance gaps | Doc 212 — per-head: current metrics, ST ceiling, SOTA anchor |
| Know what MTL literature we've already read | Doc 213 — our literature survey and identified gaps |
| Understand our backbone / pretraining options | Doc 214 — VideoMAE, Ego4D, MViTv2-S, ConvNeXt alternatives |
| **Jump straight to the questions we need answered** | **Doc 215 — the 50 deep research questions** |
| Know what a winning AAIML paper looks like | Doc 216 — venue strategy, page budget, contribution framing |
| Dive deep into loss functions | Doc 217 — per-task losses, problems, alternatives |
| Understand our data and augmentations | Doc 218 — dataset stats, augmentation strategies |
| See our efficiency metrics methodology | Doc 219 — FLOPs, params, latency, FPS comparison |
| Read our related work survey | Doc 220 — taxonomy of MTL approaches |
| Understand our benchmarking methodology | Doc 221 — how we compare MTL vs ST fairly |
| See our planned ablations | Doc 222 — all ablations for a winning paper |
| Understand our experimental protocol | Doc 223 — seed control, statistical tests, CIs |
| See our figure/table plan | Doc 224 — what figures/tables the paper needs |
| Assess risks and contingencies | Doc 225 — what could fail, contingency plans |
| See the implementation roadmap | Doc 226 — step-by-step after Claude Science returns |
| Learn how to craft effective queries | Doc 227 — how to prompt for academic paper search |

### Recommended reading order for Claude Science

1. **Doc 208 (this document):** The full picture — what we are building, what we need, what we have tried.
2. **Doc 215 (50 Deep Questions):** The most important document — it contains 50 specific, targeted research questions organized by direction and priority. This is the core ask.
3. **Doc 212 (Per-Head Gap Analysis):** Understand the concrete numbers per head: current MTL metric, ST ceiling, SOTA anchor, and what gap remains.
4. **Doc 213 (MTL Optimization Literature):** Understand what we have already read so you do not recommend papers we have already studied.
5. **Doc 210 (Architecture Space):** Understand our architecture in sufficient detail to recommend architectural modifications.
6. **Doc 211 (Training Methodology):** Understand exactly what training methods we are using so you can recommend improvements.
7. **All other documents as needed:** The remaining documents provide deeper detail on specific aspects. Use them as reference when a question from doc 215 or an answer you are formulating requires deeper context.

### How the documents relate

The package has three layers:

**Layer 1 — Entry and context (docs 208–209):** These two documents provide the bird's-eye view. Doc 208 (this one) explains the project, the hypothesis, and what we need. Doc 209 provides the complete historical timeline from doc 150 through doc 207, documenting every significant experiment, bug discovery, and pivot.

**Layer 2 — Technical deep dives (docs 210–221):** These twelve documents provide detailed technical analysis of specific aspects of our system and the literature. Each is designed to be self-contained enough to answer questions from Claude Science without requiring cross-referencing, but they are also cross-linked where dependencies exist.

**Layer 3 — Planning and execution (docs 222–227):** These six documents describe our forward plan: what experiments we intend to run, how we will run them, what figures and tables the paper needs, what could go wrong, and how to interact with Claude Science effectively.

---

## 6. Success Criteria: What a Useful Claude Science Response Looks Like

Claude Science will scan hundreds of academic papers to find methods relevant to our problem. Not all results will be equally useful. Here is how we will evaluate the responses.

### High-value outputs (what we most need)

1. **Papers we have not read that contain methods directly applicable to closing the MTL gap.** The single most valuable thing Claude Science can do is find a paper from an adjacent domain (robotics, autonomous driving, surgical video, activity understanding) that has *solved* a similar MTL problem — multiple diverse tasks sharing a single backbone — and show us how.

2. **Concrete, specific method recommendations** — not just "you could try gradient surgery," but "at ICLR 2023, Method X was applied to setting Y and achieved Z improvement; here is how it differs from what you are doing and why it might help." The specificity is what distinguishes a useful recommendation from a literature search.

3. **Negative results and cautionary tales.** If a method looks promising on paper but has known failure modes (e.g., "Nash-MTL under batch size 4 is mostly noise"; "GradNorm is known to destabilize with more than 3 tasks"), please tell us. We would rather learn this from the literature than discover it in a 6-day training run.

4. **Gap identification.** If you can tell us, "The literature you cite in doc 213 is comprehensive through 2023, but you appear to have missed the 2024–2025 wave of transformer-based MTL routing methods," that is extremely valuable even without specific paper recommendations.

5. **Contradiction detection.** If our hypothesis or claims contradict published evidence, tell us. If no published MTL system has ever beaten SOTA on 4 diverse tasks simultaneously, we need to know that to adjust our claims.

### Medium-value outputs

1. Papers that broadly confirm our approach without providing new directions.
2. Literature taxonomies that organize what we already know.
3. Method descriptions without concrete implementation guidance.

### Low-value outputs

1. Papers we already cite or discuss in doc 213.
2. Methods that are clearly inapplicable to our setting (e.g., requiring per-pixel dense labels for 1000-class segmentation).
3. Generic recommendations without specificity ("you could try more data augmentation").

### What will cause us to revisit this consultation

1. If Claude Science identifies a method that demonstrably can close a 15+ point gap on a specific head (e.g., detection from 0.000 mAP to 0.30+ mAP through a modification we have not tried).
2. If Claude Science identifies a paper with a MTL-to-ST beat on a similar multi-task set (egocentric vision + detection + activity + pose).
3. If Claude Science provides architecture guidance that would materially change our claimed ~2.06x efficiency ratio.
4. If Claude Science finds a methodological flaw in our experimental protocol that would invalidate our statistical claims.

---

## 7. Summary of the 50 Questions (Doc 215)

Doc 215 contains 50 deep research questions organized into 10 categories. Here is a summary of each category and the nature of the question:

### Questions 1–5: Detection Path
How to close the gap from our current detection mAP (which hovers near zero in full evaluation) to a meaningful mAP that supports our MTL hypothesis. Specific asks: augmentation strategies for small-object detection at 224px, task-specific detection head designs that reduce gradient conflict with activity recognition, anchor-free detection heads suitable for MTL, detection-specific backbone modifications.

### Questions 6–10: Activity Recognition Path
How to close the gap from our current ~40% top-1 activity accuracy to the 65.25% SOTA. Specific asks: long-tail classification methods for MTL settings (beyond standalone decoupled training and logit adjustment), cross-task feature sharing that specifically benefits fine-grained activity classification, temporal pooling strategies that improve activity without damaging PSR.

### Questions 11–15: PSR State Detection Path
How to maximize per-frame transition prediction from a T=8 window. Specific asks: temporal attention mechanisms that work well at short window lengths, transition-aware loss functions (beyond focal-BCE), post-processing that improves event-F1 without hurting per-frame accuracy.

### Questions 16–20: Head Pose Path
How to maximize head pose accuracy from CLS token features shared with activity recognition. Specific asks: rotation representations that play well with gradient surgery, geodesic loss variants with better numerical properties, cross-task pose regularization (e.g., activity-aware pose refinement).

### Questions 21–25: Architecture Design
The shared backbone, FPN, and head architecture questions. Specific asks: optimal FPN channel count for MTL, bottleneck representations that benefit all heads, per-head feature selection from shared representations, architectural decoupling strategies.

### Questions 26–30: Training Methodology
Beyond our current Kendall + PCGrad approach. Specific asks: gradient surgery methods designed for 4+ tasks, loss weighting schedules that adapt over training, per-task learning rate schedules, two-phase training strategies, training recipes proven for MTL.

### Questions 31–35: Loss Functions
Per-task loss design and multi-task loss calibration. Specific asks: loss normalization across vastly different scales (classification entropy 2-5 vs. regression loss 0.002-0.1), task-specific loss modifications for MTL, Pareto-optimal task weighting.

### Questions 36–40: Data and Augmentation
Specific asks: augmentation strategies that benefit MTL specifically, synthetic data generation for industrial assembly, cross-task augmentation consistency, active learning for MTL data collection.

### Questions 41–45: Evaluation and Benchmarking
How to fairly compare MTL vs ST. Specific asks: statistical methodology for MTL comparison, confidence intervals for transfer ratios, standard benchmarks for MTL efficiency, accepted comparison protocols for MTL papers.

### Questions 46–50: Paper Strategy and Framing
Specific asks: what constitutes a publishable MTL contribution at WACV/AAIML, how to frame "MTL beats SOTA" when the comparison is apples-to-oranges, how to handle the efficiency contribution when different heads use different input sizes, precedent for MTL papers that claim SOTA on multiple tasks.

---

## 8. Quick Reference: Package Map

```
consult_claude_science/
  208_OVERVIEW_CONSULTATION_PACKAGE.md     ← THIS FILE — entry point
  209_HISTORY_DOCS_150_207.md              ← Full history of what we tried
  210_ARCHITECTURE_SPACE.md                ← Architecture, FPN, head structures
  211_TRAINING_METHODOLOGY.md              ← Kendall, PCGrad, EMA, SWA, config
  212_PER_HEAD_GAP_ANALYSIS.md             ← Per-head: MTL, ST ceiling, SOTA
  213_MTL_OPTIMIZATION_LITERATURE.md       ← What we've read, what we know
  214_BACKBONE_AND_PRETRAINING.md          ← MViTv2-S, VideoMAE, Ego4D
  215_50_DEEP_QUESTIONS.md                 ← **CORE ASK: 50 research questions**
  216_AAIML_WINNING_PAPER_STRATEGY.md      ← Venue strategy, paper framing
  217_LOSS_FUNCTION_DEEP_DIVE.md           ← Per-task losses, alternatives
  218_DATA_AND_AUGMENTATION.md             ← Dataset stats, augmentation
  219_EFFICIENCY_METRICS.md                ← FLOPs, params, latency, FPS
  220_RELATED_WORK_SURVEY.md               ← Taxonomy of MTL approaches
  221_MTL_BENCHMARK_METHODOLOGY.md         ← MTL vs ST comparison protocol
  222_ABLATION_STUDY_PLANNING.md           ← All ablations for paper
  223_EXPERIMENTAL_PROTOCOL.md             ← Seeds, stats, CIs
  224_FIGURE_AND_TABLE_PLANNING.md         ← Paper figures and tables
  225_RISK_ASSESSMENT.md                   ← Failure modes, contingencies
  226_IMPLEMENTATION_ROADMAP.md            ← Step-by-step plan post-CS
  227_CLAUDE_SCIENCE_PROMPT_GUIDE.md       ← How to query effectively
```

---

**Final note to Claude Science:** This is our most important document request to date. We have invested 58 documents, 5 rounds of deep Opus consultation, months of code development, and thousands of GPU-hours into this project. We are at a critical juncture: our diagnostics are beginning to produce results, our overfit probes have confirmed that our eval infrastructure is sound for PSR and pose, and our ST baselines are launching. What we need most from Claude Science is the academic literature we have missed — the papers, methods, and ideas that could close the remaining gaps and turn this project from a credible effort into a genuinely publishable result.

We are not asking for generic advice. We are asking for specific, actionable research findings that address the concrete problems documented in this package. If Claude Science can find us even one paper whose method closes a 15+ point gap on one of our heads, this consultation will have been worth the entire effort.
