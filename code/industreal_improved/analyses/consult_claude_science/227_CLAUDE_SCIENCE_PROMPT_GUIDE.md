# Doc 227 — Claude Science Prompt Guide + Document Package Index

**Document:** 227 of 227 (Claude Science consultation package, docs 208-227)
**Status:** User manual for the consultation -- read this before querying Claude Science
**Date:** 2026-07-11
**Audience:** Research team members who will interact with Claude Science

---

## Table of Contents

1. How Claude Science Works
2. Query Strategy: Breaking Down the 50 Questions
3. Template Queries
4. Query Priority Order: Highest Impact First
5. How to Interpret Claude Science Responses
6. Document Package Index (Docs 208-227)
7. Quick Reference Card
8. Session Protocol

---

## 1. How Claude Science Works

Claude Science is Anthropic's academic paper research capability. It scans hundreds of thousands of peer-reviewed papers from arXiv, conference proceedings, and open-access journals to answer research questions with citations.

**Strengths:**
- **Broad scanning.** A query about "gradient surgery for MTL" pulls from robotics, autonomous driving, surgical video, and NLP -- not just vision.
- **Specific identification.** Asked the right way, it will identify papers, describe their approach, and cite performance deltas (e.g., "CAGrad improves average task accuracy by 1.2% on NYUv2 over PCGrad").
- **Cross-field synthesis.** Good at connecting ideas across fields -- egocentric vision, robotics manipulation, autonomous driving perception.
- **Follow-up refinement.** After an initial response, we can drill down on the most promising method.

**Limitations:**
- Cannot read paywalled papers (IEEE Xplore, Springer, Elsevier). Expect coverage of CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, WACV, and most arXiv-hosted papers.
- Cannot verify code. Always ask for code availability.
- Cannot test methods on our data. Every recommendation must be validated on our setup.
- **Can hallucinate papers** -- invented titles, authors, or numbers that look plausible. Verify every citation before use.

**Ground rules for using Claude Science output:**
1. Verify every citation against the actual paper before including in our work.
2. Treat numerical claims as directional. The exact number depends on dataset, backbone, and task set, which are never identical to ours.
3. Prioritize papers with published code. We cannot reimplement complex methods from scratch.
4. Explicitly ask for negative results. A method's known failure modes are often more valuable than its success modes.
5. Push back on generic advice. If you get "try more data augmentation," ask for specific strategies proven for MTL with small objects at 224px.

---

## 2. Query Strategy: Breaking Down the 50 Questions

Doc 215 contains 50 questions across 10 categories. Do not paste them all at once -- the response will be too diffuse. Group by answer type, not document section.

### 2.1 Group by Answer Type

| Type | Description | Example | Found In |
|---|---|---|---|
| **Single-paper find** | A specific paper/method that solves a concrete problem | "Find papers where MTL beat all ST baselines" | Q B5, Q A7 |
| **Comparative survey** | Methods compared on the same benchmark | "Compare CAGrad vs PCGrad vs Nash-MTL on 4+ tasks" | Q B2, Q B3 |
| **Architecture guidance** | Design patterns from literature | "What FPN variants work best for multi-task detection?" | Q A5, Q A1 |
| **Gap diagnosis** | Understanding why a task is failing | "Why does activity get 312x less gradient than PSR?" | Q A4, Q 6-10 |
| **Negative evidence** | Published failures to avoid | "What MTL methods fail with batch size 16?" | Implicit in many |

### 2.2 The Cascade: Broad then Narrow

For each topic, start broad and narrow over successive queries:

```
Pass 1: "What methods exist for [problem]?"
Pass 2: "Which have been benchmarked on [specific setup]?"
Pass 3: "What are failure modes of [specific method]?"
Pass 4: "Does published code exist for [specific method]?"
```

Example for gradient surgery:
- Pass 1: "What gradient surgery methods beyond PCGrad exist for MTL with 4+ tasks?"
- Pass 2: "Which have been benchmarked on detection + classification + regression tasks?"
- Pass 3: "What are CAGrad's known failure modes with batch size 16 or less?"
- Pass 4: "Does the CAGrad authors' repo include a PyTorch implementation?"

### 2.3 First Session: Four Queries

The highest-value queries to ask first:

1. **Q B5 + Q A7 combined:** "Find published MTL papers that beat single-task baselines on ALL tasks, especially those including object detection with small objects. What optimization methods and architectural patterns do they share?"

2. **Q B2 focused:** "Compare CAGrad, GradDrop, RotoGrad, and IMTL on 4+ task MTL benchmarks including detection and classification. Which shows the largest Pareto front improvement over PCGrad, and at what computational cost?"

3. **Q A4 + A8 combined:** "What published work shows that routing different backbone layers to different task heads reduces negative transfer? What layer-to-task assignment patterns work best for detection (early layers) vs classification (late layers)?"

4. **Q A1 focused:** "What task-conditional feature modulation methods (FiLM, adapters, gating) have been proven in multi-task vision with 4+ tasks including detection? Do any achieve >90% single-task retention for detection?"

---

## 3. Template Queries

Ready-to-paste templates. Replace bracketed text with our setup specifics.

### 3.1 Finding Methods That Close the MTL Gap

```
Find papers where a multi-task learning model achieved single-task-level or better
performance on 3+ tasks simultaneously, where the task set includes object detection
with small objects (<32px at input resolution). Report:
1. Optimization method used (gradient surgery, loss weighting, etc.)
2. Architecture (shared backbone, task-specific branches, routing)
3. MTL-to-ST ratio per task (preferably including detection mAP)
4. Whether code is publicly available

Our setup: MViTv2-S backbone (34.5M), 4 tasks (detection 24-class, activity 75-class
recognition, PSR 11-state temporal, head pose 6D regression), Kendall uncertainty
weighting + PCGrad, 224px input, 26K training frames.
```

### 3.2 Finding MTL Optimization Methods Beyond PCGrad

```
What MTL optimization methods published since 2021 outperform PCGrad + Kendall
uncertainty weighting on benchmarks with 4+ tasks?

I need benchmark numbers on NYUv4, Taskonomy, or PASCAL-Context for: CAGrad, GradDrop,
RotoGrad, IMTL-G, Nash-MTL, FAMO, and any 2024-2026 methods.

For each: Pareto front improvement over PCGrad, computational overhead (fraction of
training time), batch size sensitivity, whether PyTorch code is available.
```

### 3.3 Finding Architecture Patterns for Positive Transfer

```
What architectures use shared backbone representations to boost detection accuracy
in multi-task settings? I need evidence of positive transfer where sharing a backbone
with other tasks improves detection mAP vs single-task baseline.

Interested in: cross-task feature exchange, task-specific adapters/branches that
don't break single-pass inference, multi-scale feature routing (different backbone
depths for different tasks), and any published per-task MTL/ST ratios for detection.
```

### 3.4 Finding MTL for Assembly/Industrial Vision

```
Find papers about multi-task learning for assembly verification, manufacturing quality
inspection, or industrial computer vision from egocentric video. Task set should
include at least 3 of: object detection, activity recognition, action segmentation,
pose estimation, or state detection.

For each: backbone used, MTL optimization methods, data volumes, and crucially:
did the authors report MTL-vs-ST comparisons and per-task retention rates?
```

### 3.5 Finding Training Recipes for Video MTL

```
What training recipes (LR schedules, batch sizes, optimizers, augmentation strategies,
training durations) are proven for video MTL with 4+ heterogeneous tasks?

Our constraints: 2 consumer GPUs (12GB + 16GB), effective batch 16, bf16 mixed
precision, AdamW, 224px input, T=16 video clips. Papers with released training
configs (YAML/JSON) are most valuable.
```

### 3.6 Follow-Up for Negative Results

```
You mentioned [METHOD] improves [METRIC] by [X]%. What are its known failure modes?
1. Does it require batch sizes larger than 32?
2. Does it work with heterogeneous losses (CE + regression + focal)?
3. Does it add >50% training time overhead?
4. Are there published settings where it performs worse than baseline?
```

---

## 4. Query Priority Order

The 50 questions in doc 215 are ordered by section, not impact. Query in this priority:

### Tier 1: Must Ask First

| Priority | Questions | Rationale |
|---|---|---|
| 1 | Q B5 + A7 combined | If published MTL has beaten ST on all tasks, it changes our paper strategy. If none have, we adjust claims now. |
| 2 | Q B2 + B3 combined | Gradient surgery and loss weighting are our primary optimization levers. Better method than PCGrad could close 5-15% gap. |
| 3 | Q A4 + A8 combined | Feature routing is zero-param-cost if we draw from different backbone depths. |
| 4 | Q A1 | Task-conditional modulation could be the architecture advance we need, but only if proven on 4+ tasks including detection. |

### Tier 2: High Impact, Lower Urgency

| Priority | Questions | Rationale |
|---|---|---|
| 5 | Q A2, A3 | Backbone alternatives and task-grouping evidence for next architecture iteration. |
| 6 | Q B1, B6 | Loss weighting alternatives to Kendall. Our capped-log-var fix may already match published methods. |
| 7 | Q 1-5 (Detection) | Detection-specific MTL. Tier 2 because detection alone doesn't fix the core paper claim. |
| 8 | Q B4 | Curriculum/staged training. Low engineering cost, but need published evidence on ordering. |

### Tier 3: Deferrable

| Priority | Questions | Rationale |
|---|---|---|
| 9 | Q 6-10 (Activity) | Gradient starvation is an architecture problem, not a literature problem. |
| 10 | Q 11-15 (PSR) | Event-F1 near zero is loss function problem. Literature may help. |
| 11 | Q 16-20 (Pose) | Pose is already good. Marginal improvements only. |
| 12 | Q 21-25 (Architecture) | FPN/head/neck questions depend on Tier 1 answers. |

### Tier 4: Nice to Have

| Priority | Questions | Rationale |
|---|---|---|
| 13 | Q 31-35 (Loss design) | Unlikely to close major gaps alone. |
| 14 | Q 36-40 (Data/aug) | We know augmentations help. Low surprise potential. |
| 15 | Q 26-30 (Training) | Mostly confirms current approach. |
| 16 | Q 41-50 (Paper strategy) | Useful for writing, not model improvement. |

---

## 5. How to Interpret Claude Science Responses

### 5.1 Relevance Filter

Apply this filter after every response:

| Criterion | Strong Signal | Weak Signal |
|---|---|---|
| Task set includes detection | Yes, with mAP reported | Detection mentioned but not evaluated |
| Dataset size comparable | 10K-100K frames | 1M+ frames (different regime) |
| Backbone class | ViT, MViT, ConvNeXt (34-50M) | ResNet-18 (<15M) |
| Number of tasks | 3-5 in one model | 10+ tasks (different dynamics) |
| MTL/ST ratio reported | Yes, with confidence intervals | Absolute numbers only |
| Code available | GitHub link | "Code will be released" |

### 5.2 Cross-Referencing Protocol

When Claude Science cites a paper: (1) pull the paper from arXiv, (2) find the exact claim in the results table or ablation, (3) check experimental conditions -- a 5% improvement on Cityscapes does not guarantee improvement on IndustReal, (4) check for disclaimers like "weaker baseline" or "single run," (5) cross-reference with follow-up papers to see if the result was replicated.

### 5.3 When to Follow Up

- A method looks promising but the description lacks implementation detail.
- The response mentions a paper we have not read that directly addresses our problem.
- The response gives generic advice -- push for specifics.
- The response says "no published method has achieved X" -- verify with a second query.

### 5.4 Red Flags

- **"In my training data" or "as I recall"** -- the model is relying on parametric knowledge, not retrieval. Demand citations.
- **Generic numbers** ("improves by 5-10%") without specific paper reference.
- **Missing constraints** -- recommendations that ignore our 28GB VRAM, 2-GPU budget.
- **Self-contradiction across sessions** -- note the discrepancy and ask for resolution.
- **Context saturation** -- if the model repeats answers or gives vague summaries, start a new session.

---

## 6. Document Package Index (Docs 208-227)

### 6.1 Complete Listing

| Doc | File | Type | Description |
|-----|------|------|-------------|
| **208** | `OVERVIEW_CONSULTATION_PACKAGE.md` | **Entry point** | Project description, hypothesis, what we tried, success criteria. READ FIRST. |
| **209** | `COMPLETE_HISTORY_DOC150_TO_207.md` | **Entry point** | Complete experimental history from doc 150 to 207: 15+ phases across 4 months. |
| **210** | `ARCHITECTURE_EXPLORATION_SPACE.md` | **Deep dive** | Architecture survey: backbones, FPN designs, head variants, feature routing, parameter budget. |
| **211** | `TRAINING_METHODOLOGY_DEEP_DIVE.md` | **Deep dive** | Training details: Kendall weighting (capped precision fix), PCGrad, loss functions, LR, curriculum. |
| **212** | `PER_HEAD_GAP_ANALYSIS.md` | **Deep dive** | Per-head: current MTL, ST ceiling, SOTA anchor, gap decomposition, risk, priority. |
| **213** | `MTL_OPTIMIZATION_LITERATURE.md` | **Reference** | MTL optimization survey: what we've read, what we haven't, 2023-2026 targets, taxonomy. |
| **214** | `BACKBONE_AND_PRETRAINING.md` | **Deep dive** | Backbone analysis: MViTv2-S details, K400 quality, VideoMAE/Ego4D comparison. |
| **215** | `50_DEEP_QUESTIONS.md` | **Core ask** | 50 research questions in 10 categories. THIS IS THE CORE ASK. |
| **216** | `AAIML_WINNING_PAPER_STRATEGY.md` | **Reference** | AAIML 2027 strategy: venue values, positioning, reviewer management, disclosure. |
| **217** | `LOSS_FUNCTION_DEEP_DIVE.md` | **Deep dive** | Per-task losses: current implementation, failure modes, literature alternatives. |
| **218** | `DATA_AND_AUGMENTATION.md` | **Deep dive** | Dataset statistics and augmentation strategies. Planned, not yet written. |
| **219** | `EFFICIENCY_METRICS.md` | **Reference** | Efficiency protocol: parameter counting, FLOPs, latency, FPS, memory methodology. |
| **220** | `RELATED_WORK_SURVEY.md` | **Reference** | Taxonomy of MTL approaches by method family. Planned, not yet written. |
| **221** | `MTL_BENCHMARK_METHODOLOGY.md` | **Reference** | MTL vs ST comparison protocol and confidence intervals. Planned, not yet written. |
| **222** | `ABLATION_STUDY_PLANNING.md` | **Reference** | Complete ablation plan for the paper. Planned, not yet written. |
| **223** | `EXPERIMENTAL_PROTOCOL.md` | **Reference** | Statistical rigor: seed control, splits, hyperparameter search, statistical testing. |
| **224** | `FIGURE_AND_TABLE_PLANNING.md` | **Reference** | Paper figure/table layouts. Planned, not yet written. |
| **225** | `RISK_ASSESSMENT.md` | **Reference** | Failure modes and contingency plans. Planned, not yet written. |
| **226** | `IMPLEMENTATION_ROADMAP.md` | **Reference** | Post-Claude Science execution plan. Planned, not yet written. |
| **227** | `CLAUDE_SCIENCE_PROMPT_GUIDE.md` | **User manual** | THIS FILE. How to query effectively with this package. |

### 6.2 Reading Order for Claude Science

When loading context for a session: (1) Doc 227 -- query strategy and protocol, (2) Doc 208 -- full picture, (3) Doc 215 -- the 50 specific questions, (4) Doc 212 -- per-head numbers, (5) Doc 213 -- what we already know. Then as needed: 210 (architecture), 211 (training), 217 (losses), 219 (efficiency).

### 6.3 Document Types

- **Entry points (2):** 208, 209. Start here for full context.
- **Deep dives (8):** 210, 211, 212, 214, 217, 218, 219, 223. Detailed technical analysis.
- **Reference (10):** 213, 215, 216, 220, 221, 222, 224, 225, 226, 227. Surveys, strategy, methodology, this manual.

---

## 7. Quick Reference Card

Present this at the start of every Claude Science session to ground the model in our setup.

### Model Identity

| Property | Value |
|---|---|
| Backbone | MViTv2-S |
| Backbone params | 34.5M |
| Total params | ~48.6M (34.5M backbone + ~14.1M heads + FPN) |
| K400 pretraining | 81.0% top-1 |
| Input resolution | 224px |
| Temporal window | T=16 (T=8 after conv_proj stride) |
| Parameter efficiency | ~2.06x vs 4 separate ST models (~100M) |
| Inference | Single forward pass, all 4 tasks |

### Tasks and Metrics

| Task | Classes | Metric | Current MTL | ST Ceiling | SOTA Anchor |
|---|---|---|---|---|---|
| Detection (ASD) | 24 | mAP@0.5 | 0.202 | 0.40-0.55 | 0.779 (YOLOv8m, 640px, COCO) |
| Activity | 75 | Top-1 acc | ~0.35 | 0.50-0.60 | 0.6525 (WACV 2024) |
| PSR | 11 binary | Event-F1 | ~0.006 | 0.15-0.25 | Not established |
| Head pose | 6D cont. | Fwd MAE | ~8.7 deg | 2-5 deg | Not established |

### Optimization Stack

| Component | Method | Parameters |
|---|---|---|
| Loss weighting | Kendall uncertainty (capped) | det:1.5, act:1.0, psr:0.5, pose:2.0 log-var caps; EMA mom=0.99 |
| Gradient surgery | PCGrad (shared backbone only) | After Kendall scaling, random task order |
| Optimizer | AdamW, 3 groups | Backbone:1e-4, Heads:1e-3, Log-vars:1e-3 (wd=0) |
| LR schedule | CosineAnnealingLR | T_max=epochs, no warm-up |
| Precision | bf16 mixed precision | GradScaler is pass-through |
| Gradient clip | Global norm 5.0 | Raised from 1.0 after audit |
| Effective batch | 16 | 4 micro-batches x 4 grad accum |
| EMA/SWA | EMA mom=0.999 + SWA last 5 | Complementary post-training |

### Data and Compute

| Property | Value |
|---|---|
| Train frames | ~26,000 |
| Val frames | ~38,000 |
| Recordings | 16 (10 train, 6 val) |
| Native resolution | 1280x720 (trained at 224x224) |
| Activity distribution | Power-law, 75 classes, tail 1-7 frames |
| PSR positive rate | <0.5% |
| GPU 1 / GPU 2 | RTX 3060 12GB / RTX 5060 Ti 16GB |
| Total VRAM | 28GB |
| Training time (100 ep) | ~6-7 days estimated |
| Inference FPS | ~11 FPS on single consumer GPU |

---

## 8. Session Protocol

### Before
1. Select 1-2 queries from Tier 1 (Section 4). Do not overload a session.
2. Load context docs in reading order (Section 6.2).
3. Paste the Quick Reference Card (Section 7) into the prompt.
4. Open a fresh session -- context windows degrade with length.

### During
1. Start with the template query from Section 3, adapted to your question.
2. After the initial response, ask for verification: exact paper title, year, venue, result.
3. Drill down on promising methods: algorithm detail, hyperparameters, implementation specifics.
4. Explicitly ask for negative results and failure modes.
5. Ask for code availability (official PyTorch implementation).
6. Maintain a running citation log with arXiv IDs.

### After
1. Verify every paper claim against the actual paper before citing.
2. Categorize each finding: **Adopt now** (code available, fits constraints), **Ablate** (promising but needs validation), **Defer** (prerequisite problem remains), **Reject** (inapplicable or requires too much compute).
3. Log decisions in `.superpowers/homunculus/observations/` with rationale.
4. Flag contradictions between Claude Science findings and our current approach for team discussion.

---

**End of Doc 227. This is the final document in the Claude Science consultation package (docs 208-227).**

Now go find us the papers that close the gap.
