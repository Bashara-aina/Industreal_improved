# 50 — ASK OPUS: The Complete Path from Current State to Benchmarkable Paper

> **Generated:** 2026-06-22 19:45 JST  
> **Purpose:** Every question, every gap, every unknown between where we are now and a finished paper with benchmarkable results.  
> **Target document:** `popw_paper_improved.tex` — 81 KB, ~1310 lines, >30 `\todo`/`\popwres` placeholders  
> **Current training:** **Run 3** (crash-restart), rf2 epoch 17, PID 1204133, batch 2050/3302, det_mAP50_pc=0.304, MAE=9.13°  
> **Run history:** Run 1 (wrong LR/BIAS) → Run 2 (correct LR/BIAS, confirmed identical trajectory) → CRASH during epoch 21 → **Run 3** (restart from epoch 17 best checkpoint)  
> **Key finding:** Structural ceiling at ~0.207 mAP50 confirmed across 2 independent runs with different LR/BIAS. OHEM+FocalLoss gradient suppression is the primary hypothesis.  
> **Monitor loop:** ACTIVE — checking every 300s  
> **Status:** ALL 7 GUIDEs implemented. Paper is the critical path. The "wait for Run 2" question is answered — ceiling is confirmed. OHEM ablation is next.  
> **How to use this document:** Each section is a standalone question cluster. Read the context, answer what you can. Skip what you can't. The goal is to produce a concrete, ordered execution plan for getting benchmarkable numbers into every paper table.

---

## Table of Contents

1. [The Fundamental Strategic Question](#1-the-fundamental-strategic-question)
2. [Detection: Breaking the mAP Ceiling](#2-detection-breaking-the-map-ceiling)
3. [Activity Recognition: Zero Baseline Today](#3-activity-recognition-zero-baseline-today)
4. [PSR: The Never-Trained Head](#4-psr-the-never-trained-head)
5. [Head Pose: The One Working Head](#5-head-pose-the-one-working-head)
6. [Body Pose: IKEA ASM Only](#6-body-pose-ikea-asm-only)
7. [The Ablation Regime: Proving the Idea](#7-the-ablation-regime-proving-the-idea)
8. [Efficiency Numbers: Filling the Compute Tables](#8-efficiency-numbers-filling-the-compute-tables)
9. [IKEA ASM Experiments: The Second Dataset](#9-ikea-asm-experiments-the-second-dataset)
10. [Paper Production: From \todo to Submission-Ready](#10-paper-production-from-todo-to-submission-ready)
11. [The 200-Point Verification Checklist](#11-the-200-point-verification-checklist)
12. [Timeline & Resource Constraints](#12-timeline--resource-constraints)
13. [Risk Register](#13-risk-register)
14. [The Final Ask](#14-the-final-ask)

---

## 1. The Fundamental Strategic Question

### 1.1 What is the minimum publishable paper?

GUIDE 4 defines the bar clearly:

> *"Your thesis is NOT 'we beat YOLOv8m / MViTv2 / STORM-PSR.' Your thesis is: A single shared-backbone model can perform egocentric assembly understanding — detection, body pose, head pose, activity, and procedure-step recognition — in one forward pass, at a fraction of the parameters and compute of separate specialists, without catastrophic interference, with cross-task FiLM conditioning."*

**Question 1.1:** Given our current trajectory (rf2 epoch 17, mAP50_pc=0.304, MAE=9.13°), what is the honest assessment of whether we can achieve GUIDE 4's publishable bar?

The four things needed:
1. ✅ All five heads produce non-trivial, competitive results
2. ❓ Efficiency: 53M params / 1 forward pass vs ~81M / 3 passes (measurable now)
3. ❓ Multi-task competitive with single-task specialists (ablation A — not run)
4. ❓ FiLM conditioning contributes (ablation B — not run)

**Question 1.2:** Is "not yet run" on the two ablations a blocking issue for the paper, or can we write the paper structure now and fill numbers later? What's the right ordering?

### 1.2 What does "benchmarkable result" actually mean for each head?

**Question 1.3:** For each head, define the threshold that counts as "benchmarkable":

| Head | Current best | Target | What's "benchmarkable"? |
|------|-------------|--------|------------------------|
| Detection (ASD) | mAP50=0.207, mAP50_pc=0.304 | mAP50_pc≥0.40 | ??? |
| Activity | N/A — never trained | Top-1 > ??? | ??? |
| PSR | N/A — never trained | F1 > ??? | ??? |
| Head Pose | MAE=9.13° | No baseline exists | ??? |
| Body Pose | N/A — IKEA only | PCK@0.2 > ??? | ??? |

What thresholds should we set for each to claim "the head produces meaningful results"?

### 1.3 What's the actual acceptance criteria for a CVPR/ICCV/WACV submission?

**Question 1.4:** Given the detection gap (our best 0.207 vs YOLOv8m's 0.838 on the same dataset), what venue is realistic? WACV? BMVC? A workshop? What threshold of results would make CVPR worth attempting?

### 1.4 Resource constraints: how far can one RTX 3060 take us?

**Question 1.5:** We have:
- 1× RTX 3060 12GB
- ~12.7h remaining for rf2 (epochs 17-36 at ~86 min/epoch)
- ~15 epochs for rf3 at 35% subset
- Phase B/C decoupled training (requires embedding_cache.py which is implemented but not tested)
- IKEA ASM full run (371 videos, 3 views — estimated 2-3 days)

What is achievable on this single GPU before the submission deadline? What should we prioritize?

---

## 2. Detection: Breaking the mAP Ceiling

### 2.1 The Four Hypotheses for Why mAP Plateaus at ~0.207

**Question 2.1:** Current best evidence shows mAP50 consistently stuck at ~0.204-0.215 across multiple runs with different configs (wrong LR/BIAS, correct LR/BIAS — same ceiling). The 50-image cls-only overfit proves the architecture CAN learn. Which of these hypotheses is most likely, and what's the cheapest experiment to test each?

| Hypothesis | Evidence | Cheapest test |
|-----------|----------|---------------|
| A: OHEM+FocalLoss gradient suppression | 50-image overfit shows 3-regime trajectory. OHEM dominates early gradients. FocalLoss suppresses easy positives. | Train with OHEM off + FocalLoss α=0.5 (not 0.25) for 3 epochs on rf2. |
| B: Anchor geometry mismatch | Only 12/24 classes ever fire. Some assembly parts may not match 24-192 anchor sizes. | Log per-class positive anchor counts. If a class has 0 anchors at IoU>0.2, its AP=0 forever. |
| C: Label quality (synthetic projection errors) | Labels are projected from floor-plan geometry. Never visually inspected. | Manual audit: overlay 100 GT boxes from AP=0 classes. Check box position + class ID. |
| D: Data scarcity for tail classes | 8/24 classes have <50 training instances at 50% subset. | Per-class GT count → compare to threshold for learnability. |

**Question 2.2:** The detach_reg_fpn fix (ba48691) is confirmed applied. Run 2 with correct LR/BIAS=1.0 is in progress but shows the same ceiling as Run 1 (wrong LR/BIAS). Does this prove the fix is insufficient, or is there another mechanism?

### 2.2 The 12 Dead Classes Problem

**Question 2.3:** Across ALL runs, exactly 12/24 classes always have AP=0. The same 12. Not random. The det_n_present_classes is consistently 15-16/24 (some appear only in val). This pattern is epoch-consistent and config-consistent. What causes exactly half the classes to be dead?

The per-class AP data we have:
- Classes that work: consistently work across runs
- Classes that don't: consistently don't work
- The pattern is binary, not graded — no class has AP=0.05 or AP=0.10

**Question 2.4:** Could this be an anchor assignment issue? If a class's objects have sizes/aspect ratios that NO anchor template matches above the IoU threshold (0.4 positive, 0.4 negative), the class gets zero positive assignments → zero learning signal → AP=0 forever. The DET_POS_IOU_TOP_K=9 and DET_POS_IOU_IOU_FLOOR=0.2 mitigations help but only if IoU>0.

**Question 2.5:** What's the single most informative diagnostic we could run RIGHT NOW to determine why the 12 classes are dead? Suggestions:
- A. Per-class positive anchor count at current config (does every class get at least SOME positive assignments?)
- B. Per-class AP at a lower IoU threshold (0.3 instead of 0.4 — does relaxing the assignment help?)
- C. Visual inspection of 10 GT boxes from each dead class (are the labels correct?)
- D. Confusion matrix analysis for the dead classes (are they being predicted as other classes?)

### 2.3 Detection: What's the Real Ceiling?

**Question 2.6:** What's the maximum achievable mAP50 on IndustReal with ConvNeXt-Tiny + FPN + our training setup? We need a realistic target for the paper.

Considerations:
- YOLOv8m achieves 83.80% with COCO pretraining + 260K synthetic images + real fine-tuning
- Our setup: ImageNet-only pretraining, real data only (no synthetic), 50% subset
- Even with 100% data and synthetic pretraining, ConvNeXt-Tiny isn't YOLOv8m

**Question 2.7:** What is the SINGLE most impactful thing we could do to improve detection before running the ablation experiments? (Maximum ROI per unit time.)
- A. Fix OHEM + FocalLoss (config change, ~0 days)
- B. Add synthetic pretraining data (already exists in the dataset, need to wire up)
- C. Per-class anchor optimization (compute optimal anchor sizes per class, config change)
- D. Full data (100% instead of 50% subset)

### 2.4 The Dilution Problem and Honest Metrics

**Question 2.8:** The dilution gap is consistent:
- det_mAP50 = 0.202-0.207 (diluted by 8 zero-GT background channels)
- det_mAP50_pc = 0.304-0.344 (present-class only, honest)
- n_present = 15-16/24

The paper currently doesn't mention det_mAP50_pc at all. The results table (`tab:industreal-headline`) uses the same \popwres placeholder for mAP (b-boxed) as YOLOv8m's 83.80%. If we put 0.207 next to 83.80, it looks terrible. If we put 0.304, it's comparing different protocols.

How should the paper handle this? Options:
- A. Report BOTH det_mAP50_pc and det_mAP50, with explicit discussion of dilution
- B. Report only det_mAP50 (standard protocol), add a limitations note
- C. Create a separate "honest metrics" table
- D. Frame the dilution as a finding: "COCO-24 protocol dilutes IndustReal's 16 active classes, masking the model's true performance"

---

## 3. Activity Recognition: Zero Baseline Today

### 3.1 The Cold-Start Problem

**Question 3.1:** Activity recognition has NEVER been trained in our setup. The activity head exists in the model, but train_act=False in rf2. The activity head will only be trained in rf3 (15 epochs, 35% subset). Is 15 epochs at 35% subset sufficient to produce any meaningful Top-1 accuracy?

Dataset statistics:
- 74 activity classes
- Massive class imbalance (some classes have <15 examples)
- At 35% subset: even fewer samples per class
- MViTv2 baseline: 65.25% Top-1 (full data, K400 pretrained video encoder)

**Question 3.2:** What's a realistic Top-1 for activity after 15 epochs on 35% subset?
- If <5%: the paper can't claim activity is working (effectively random on 74 classes = 1.35%)
- If 15-25%: "above random but far from SOTA" — is this publishable?
- If >30%: credible demonstration of multi-task transfer

**Question 3.3:** Should we run a quick activity-only baseline (single-task, same backbone, same data) to establish a "lower bound" for Ablation A? This would tell us whether multi-task sharing helps or hurts activity recognition.

### 3.2 Activity Head: Component Questions

**Question 3.4:** The activity head is complex:
- Detection context (MaxPool of cls_preds, stop_grad)
- Spatial features (GAP of C5_mod2 after FiLM conditioning)
- TCN block (depthwise conv 1D)
- 2× ViT blocks with CLS token
- VideoMAE V2 fusion (frozen, 384→512 projection)

How many of these are actually testable in the current codebase? Has the activity head ever been verified to produce a forward pass with correct shapes end-to-end?

**Question 3.5:** The LDAM-DRW loss switches from uniform to class-balanced reweighting at epoch 60. With only 15 epochs in rf3, DRW never activates. Should we modify this? Options:
- A. Reduce DRW switch epoch to match rf3's 15-epoch schedule
- B. Use class weights from epoch 1 (no DRW schedule)
- C. Report activity with the default schedule and note the limitation

### 3.3 Activity: VideoMAE V2 Dependency

**Question 3.6:** The paper describes VideoMAE V2 fusion (frozen backbone → linear projection → concat with ViT CLS → classifier). Has VideoMAE V2 ever been integrated and tested? What are the memory requirements?

Current training uses ~5.78GB VRAM on RTX 3060 with batch_size=1 and gradient accumulation. Adding VideoMAE would add:
- ~22M parameters (frozen)
- Additional forward pass through 16-frame clip
- Memory for the 16-frame buffer

Is VideoMAE V2 feasible on 12GB VRAM alongside the existing 5-task model?

---

## 4. PSR: The Never-Trained Head

### 4.1 The 1.546e-08 Mystery

**Question 4.1:** PSR loss has been CONSTANT at 1.546e-08 across EVERY run, EVERY config, ALL phases. Opus v9 explained this as the binary-focal floor of a predictor trivially correct on the ~20/22 always-zero components. Is this explanation correct, or is there a deeper bug?

Evidence:
- PSR causal transformer produces extreme logits (min=-23, max=+22)
- Sigmoid saturates at these logits → gradient ≈ 0 for most components
- In rf1-rf3, train_psr=False (intentionally frozen)
- But in R2.5 era (all heads), it was supposed to be training — and STILL showed 1.546e-08

**Question 4.2:** Has PSR EVER successfully trained? Ever? In any configuration? If the answer is genuinely "no, never," what does that mean for the paper?

**Question 4.3:** The paper dedicates significant space to the PSR head architecture:
- Causal Transformer (3L, 4H, d_model=256)
- Per-component tiny MLPs (11 heads)
- Binary Focal Loss
- Per-video cache for O(1) inference

If PSR has never been validated to train, should we:
- A. Report PSR as "not yet trained" (honest but weakens the paper)
- B. Run a quick PSR-only experiment (Phase-B style, from cached features) to validate
- C. Drop PSR from the paper scope entirely and focus on det+act+head_pose

### 4.2 What Would It Take to Validate PSR?

**Question 4.4:** What's the minimum experiment to prove PSR CAN learn? Opus v9 suggested a 50-sequence PSR-only overfit (fully decoupled from detection). Is this:
- Feasible on current hardware?
- Likely to succeed given the extreme logit issue?
- Worth the time vs. improving detection/activity metrics?

---

## 5. Head Pose: The One Working Head

### 5.1 Current State

**Question 5.1:** Head pose is the only head that consistently works well:
- Forward angular MAE: 9.13° (rf2 gate: ≤60°, rf3 gate: ≤55°)
- Well within all thresholds
- No published supervised baseline for 9-DoF head pose in industrial assembly

This is the paper's strongest uncontested result. How should we maximize it?

### 5.2 The Benchmarking Problem

**Question 5.2:** There is NO published supervised baseline for 9-DoF head pose on IndustReal. The paper's head pose row in Table 2 currently shows "no published supervised baseline" for comparison. What baselines could we create?

Options:
- A. Train a head-pose-only MLP on the same backbone features (single-task baseline)
- B. Compare against a simple regression baseline (mean prediction)
- C. Report only POPW's numbers and frame it as a new capability
- D. Compare against the HeadPoseFiML's input (no head pose head → estimate from detection features)

### 5.3 Head Pose as Conditioning Signal

**Question 5.3:** The head pose head serves dual purpose: (1) standalone output for gaze inference, and (2) input to HeadPoseFiLM which modulates activity features. Ablation B (does FiLM help?) directly tests the second purpose. But if activity never trains (rf3 hasn't run), we can't test this.

How important is the FiLM conditioning result to the paper's contribution claim? If we can't test it (activity head doesn't work), does the paper lose its primary novelty?

---

## 6. Body Pose: IKEA ASM Only

### 6.1 Current Status

**Question 6.1:** Body pose (17 keypoints) is IKEA ASM only — IndustReal has no body pose annotations. The body pose head exists in the model but has never been trained or evaluated.

### 6.2 The IKEA ASM Experiments

**Question 6.2:** What is the minimum IKEA ASM experiment to get body pose numbers for the paper?

Considerations:
- 371 videos, 3 views each (front/top/side), 640×480 resolution
- Body pose: 17 keypoints, PCK@0.2 = 88.0% (MaskRCNN-ft baseline)
- Object segmentation: 7 classes, AP@0.5 = 85.3% (ResNeXt-101-FPN baseline)
- Activity: 33 classes, front view, P3D Top-1 = 60.40%, I3D RGB+pose = 64.15%

**Question 6.3:** The paper claims both IKEA ASM and IndustReal results (Tables 1 and 2). Given our resource constraints, should we:
- A. Run both datasets fully (2-3 days per dataset) → strongest paper
- B. Focus on IndustReal only (egocentric is the primary contribution) → save time
- C. Run a single IKEA ASM experiment for pose numbers → compromise

What does the reviewer expect?

---

## 7. The Ablation Regime: Proving the Idea

### 7.1 Ablation A: Single-Task vs Multi-Task

**Question 7.1:** The paper's core scientific claim rests on Ablation A (Table: single-task specialists vs unified POPW). This requires:

For each head:
1. Train the head alone with shared backbone (detached features) → single-task specialist performance
2. Train all heads together → multi-task performance
3. Compare per-head metrics

This is Phase-B decoupled training (embedding_cache.py → train each head on cached features). Has Phase-B infrastructure been tested? Is embedding_cache.py working?

**Question 7.2:** What's the minimum viable Ablation A?
- Full 5-head × single-task runs: 5 runs × ~1 day each = 5 days (not feasible on single GPU)
- Maybe: run Ablation A only for detection and activity (the two primary heads)
- Maybe: run single-task detection only (most important comparison)

What's the minimum that satisfies a reviewer?

### 7.2 Ablation B: FiLM Conditioning

**Question 7.3:** Ablation B requires: No FiLM → PoseFiLM only → HeadPoseFiLM only → Both. This requires activity recognition to work (since FiLM modulates activity features). If activity can't be trained yet, is Ablation B blockingly dependent on rf3 completion?

Options:
- A. Run Ablation B in Phase-B (cache features, train activity head with different FiLM configs)
- B. Run Ablation B in Phase-C (joint fine-tune with different FiLM configs)
- C. Skip Ablation B and report only if activity works

### 7.3 Do We Need Both Ablations for the Paper to Be Complete?

**Question 7.4:** GUIDE 4 says the paper is done when:
- [ ] All five heads have non-trivial numbers (full test set)
- [ ] Efficiency table filled
- [ ] Ablation A filled
- [ ] Ablation B filled
- [ ] Detection confusion matrix + honest framing
- [ ] Limitations section written
- [ ] Multi-seed mean±std
- [ ] Every \todo replaced

If we can't fill Ablation B (FiLM not testable), is the paper still publishable?

---

## 8. Efficiency Numbers: Filling the Compute Tables

### 8.1 What We Know vs What We Need

**Question 8.1:** Current knowledge:
- Parameters: 53.3M w/o VideoMAE, 75.3M w/ VideoMAE (can calculate from model arch)
- GFLOPs: UNKNOWN (need fvcore or manual calculation)
- FPS: UNKNOWN (need to benchmark on RTX 3060)
- VRAM: ~5.78GB reserved during training

The paper needs GFLOPs and FPS for Tables 5 (efficiency), 8 (complexity). What scripting is needed to produce these numbers?

### 8.2 Streaming vs Batched Inference

**Question 8.2:** The paper distinguishes two modes:
- Batched: process multiple frames with batch_size > 1
- Streaming: process frames one at a time (for real-time deployment)

Are both modes implemented and testable? What numbers should we expect?

---

## 9. IKEA ASM Experiments: The Second Dataset

### 9.1 Dataset Configuration

**Question 9.1:** IKEA ASM has different config requirements:
- Resolution: 640×480 instead of 1280×720
- Tasks: detection (7 objects), body pose (17 kpts), activity (33 classes), phase classification — NO head pose, NO PSR
- 3 camera views (front, top, side) — does training use all 3 or just front?

Does the codebase support IKEA ASM training? Are there presets for it?

### 9.2 What's the Minimum IKEA ASM Experiment?

**Question 9.2:** Given the paper currently has \popwres for every IKEA ASM row (object AP, body pose PCK, activity Top-1, temporal localization mAP), what's the minimum experiment to fill these?

The MTL efficiency story is actually stronger on IKEA ASM: 3 views × 4 tasks = 12 separate specialized models vs 1 POPW. But getting there requires running IKEA ASM training.

---

## 10. Paper Production: From \todo to Submission-Ready

### 10.1 Placeholder Audit

**Question 10.1:** The paper has >30 \todo/\popwres placeholders. Here is the complete list. For each, what's the specific experiment/measurement needed, and what's the minimum acceptable value?

#### Results tables (need benchmark numbers):

**Table: IKEA ASM headline (tab:ikea-headline)**
| Row | Placeholder | Metric | What fills it | Status |
|-----|-------------|--------|---------------|--------|
| Object segmentation, POPW | \popwres | AP@0.5 | IKEA ASM eval | NOT RUN |
| Object segmentation, POPW | \popwres | AP (COCO) | IKEA ASM eval | NOT RUN |
| Body pose, POPW | \popwres | PCK@10px | IKEA ASM eval | NOT RUN |
| Body pose, POPW | \popwres | PCK@0.2 | IKEA ASM eval | NOT RUN |
| Activity, POPW front RGB | \popwres | Top-1 | IKEA ASM eval | NOT RUN |
| Activity, POPW all views | \popwres | Top-1 | IKEA ASM eval | NOT RUN |
| Temporal localization, POPW | \popwres | mAP@0.5 | IKEA ASM eval | NOT RUN |

**Table: IndustReal headline (tab:industreal-headline)**
| Row | Placeholder | Metric | What fills it | Status |
|-----|-------------|--------|---------------|--------|
| ASD POPW | \popwres | mAP (b-boxed) | full-test evaluate.py | NEEDS TRAINING |
| ASD POPW | \popwres | mAP@0.5 (all frames) | full-test evaluate.py | NEEDS TRAINING |
| ASD POPW | \popwres | mAP@[0.5:0.95] | full-test evaluate.py | NEEDS TRAINING |
| Activity POPW | \popwres | Top-1 | rf3 eval | NEEDS rf3 |
| Activity POPW | \popwres | Top-5 | rf3 eval | NEEDS rf3 |
| PSR POPW | \popwres | F1 (±3) | rf3 eval | NEEDS rf3 + PSR working |
| PSR POPW | \popwres | F1 (±5) | rf3 eval | NEEDS rf3 + PSR working |
| PSR POPW | \popwres | POS | rf3 eval | NEEDS rf3 + PSR working |
| Head pose POPW | \popwres | Forward angular MAE° | rf2-3 eval | AVAILABLE (9.13°) |
| Head pose POPW | \popwres | Up angular MAE° | rf2-3 eval | NEEDS LOGGING |
| Head pose POPW | \popwres | Position MAE mm | rf2-3 eval | NEEDS LOGGING |

**Table: Ablation experiments (all \todo)**
| Table | \todo count | What fills it | Status |
|-------|-------------|---------------|--------|
| Backbone choice (tab:abl-backbone) | 8 \todo | ResNet-50 vs ConvNeXt-Tiny runs | NOT RUN |
| Head contributions (tab:abl-heads) | 8 \todo | Single-task head runs | NOT RUN |
| FiLM conditioning (tab:abl-film) | 4 \todo | FiLM ladder experiments | NOT RUN |
| MTL weighting (tab:abl-mtl) | 10 \todo | Weighting strategy experiments | NOT RUN |
| Temporal modeling (tab:abl-temporal) | 8 \todo | Component ablation runs | NOT RUN |

**Other placeholders:**
| Location | \todo/\popwres | What fills it | Status |
|----------|---------------|---------------|--------|
| §1 Contributions, claim 3 | "Kendall combined with staged training enables stable joint optimization" | Ablation MTL table | NEEDS rf3 |
| §1 Contributions, claim 4 | "competitive accuracy against dedicated baselines" | Headline tables | NEEDS TRAINING |
| §3.4 Activity head | "VideoMAE V2 features are... concatenated" | Forward pass test | UNTESTED |
| §5 Efficiency | Reserved GFLOPs/FPS | efficiency_report.py | NEEDS SCRIPTING |
| §5 Efficiency | "representing a \todo parameter reduction" | Calculate from params | CAN DO NOW |
| §5 Efficiency | "approximately \todo FPS" | Benchmark | NEEDS SCRIPTING |
| §6 Training dynamics | 4 \todo for Kendall weight observations | rf3 training logs | NEEDS rf3 |
| §7 Failure cases | 8 \todo for occlusion/rare-class impact | Analysis section | CAN DRAFT NOW |
| Table: Per-component PSR | 11 \todo | rf3 eval | NEEDS rf3 + PSR working |

**Question 10.2:** If we prioritize ONLY the experiments that fill the most placeholders with the least time, what's the optimal order? I estimate:
1. rf2 completes (12.7h) → fills head pose MAE numbers ✅
2. rf2 full evaluation → fills detection mAP (b-boxed + all frames)
3. rf3 training (15 epochs × ~43 min = ~10.7h) → fills activity + PSR
4. Phase-B single-task detection → fills Ablation A first column
5. Efficiency script (compute GFLOPs/FPS from existing model) → fills compute tables

Does this ordering make sense? What would you change?

### 10.2 Conclusion Section

**Question 10.3:** The conclusion section is entirely missing (lines 1099-1111 are comments). What should the conclusion say given our likely results? Write a draft that:
- Doesn't overclaim (no "SOTA" unless we actually beat baselines)
- Honestly states limitations
- Points to future work
- Is grounded in the actual numbers we expect to produce

### 10.3 Figures

**Question 10.4:** The paper needs these figures:
1. Architecture diagram (currently a placeholder box)
2. Kendall weight evolution plot
3. Per-task validation curves
4. Detection confusion matrix (24×24)
5. Activity confusion matrix (74×74) 
6. Qualitative results grid (3×3 IndustReal frames)

Which of these can we produce now with existing code? Which need new plotting code?

---

## 11. The 200-Point Verification Checklist

### 11.1 Mapping GUIDE 6 to Paper Readiness

**Question 11.1:** GUIDE 6 (200-point verification checklist) covers code correctness, training pipeline, and metrics. But how many of those items directly affect the paper's benchmark results?

The critical items for the paper are:
- Metrics correctness: mAP calculation, PSR F1 matching PSRT protocol, activity Top-1 matching MViTv2 protocol
- Multi-seed runs: 3 seeds for headline numbers
- Ablation reproducibility: same training configs across ablation runs

**Question 11.2:** Is the evaluation protocol correct for each benchmark comparison?
- Detection mAP (b-boxed) = COCO 101-point interpolation on annotated frames only?
- Activity Top-1 = clip-level, 16-frame uniform sampling?
- PSR F1 = ±3-frame tolerance, bi-directional greedy matching?
- Head pose MAE = L2-normalized vectors before angular computation?

Where in the evaluation code are these protocols defined, and have they been verified against the original paper definitions?

---

## 12. Timeline & Resource Constraints

### 12.1 The Realistic Timeline

**Question 12.1:** Given one RTX 3060 12GB and current training in progress (rf2 epoch 17/36), what's the realistic timeline to a complete draft?

My estimate:
- rf2 completion: ~12.7h remaining (epoch 17→36)
- rf2 full eval (after training): ~2h
- rf3 training (15 epochs, 35% subset): ~10.7h
- rf3 full eval: ~2h
- Phase-B single-task detection baseline: ~8h
- Efficiency measurement: ~1h (scripting + running)
- Paper writing/editing: ~8h (fill tables, write conclusion, generate figures)

Total: ~44.5h of sequential compute + ~8h of writing

But since not all experiments can run in parallel (single GPU), the wall-clock time is the sum of sequential training runs: ~33.4h for rf2→rf3→Phase-B, plus ~10h for writing/analysis.

**Question 12.2:** Can any of these run in parallel?
- Efficiency measurement: run on existing model, no training needed. Can overlap with rf3.
- Paper writing: can start immediately with current data. Fill with \todo, update as results arrive.
- Confusion matrix: can be plotted from rf2 eval once it completes.

### 12.2 What If We Have Less Time?

**Question 12.3:** If we had to produce a submission-ready draft in 48h from NOW, what's the triage plan?

Priority order:
1. Let rf2 finish (12.7h) → get detection mAP + head pose MAE
2. Evaluate full test set → fill IndustReal detection rows + head pose rows
3. Draft everything we can with \todo for missing numbers
4. Submit with head pose (uncontested) + detection (with honest framing) + efficiency (calculated)
5. Activity and PSR as "ongoing work" or "preliminary results"
6. Promise full results in camera-ready

Is this a viable submission strategy?

---

## 13. Risk Register

### 13.1 Technical Risks

**Risk R1 — rf2 fails to improve:**
If rf2 finishes at mAP50_pc=0.35 (not 0.40), the detection section is weaker. What's the communication strategy?

**Risk R2 — rf3 training crashes:**
Activity head has never been trained. What if the LDAM loss, ViT blocks, or Feature Bank cause OOM or NaN? Is there a fallback to a simpler activity head?

**Risk R3 — PSR cannot be trained:**
If the 1.546e-08 loss is a fundamental bug (not just the frozen-head artifact), and PSR has literally never worked, do we drop PSR from the paper?

**Risk R4 — GPU failure/outage:**
Training has been running for 27h on consumer hardware. What if the GPU crashes, the process OOMs, or the system needs a reboot? Is there a recovery plan?

**Risk R5 — IKEA ASM data pipeline broken:**
IKEA ASM has different annotation format, resolution, and task set. If the eval code doesn't support it, we can't fill those tables. Has IKEA ASM eval ever been tested?

### 13.2 Paper Risks

**Risk R6 — The detection gap dominates the narrative:**
Reviewer sees "0.207 vs 83.80" and stops reading. Is the honest framing strong enough to overcome this?

**Risk R7 — Single GPU → no multi-seed results:**
The paper promises "standard deviations across 3 seeds." On a single GPU running sequential training, this would take 3× the time. Is single-seed acceptable for submission? Can we add the multi-seed promise for camera-ready?

**Risk R8 — Novelty challenge:**
"ConvNeXt-Tiny + FPN + RetinaNet detection + separate heads is not novel." The FiLM conditioning and staged Kendall training are the claimed novelties. Is this sufficient for the target venue?

### 13.3 Mitigation Questions

**Question 13.1:** For each risk above, do you have a mitigation we haven't considered?

Specifically:
- R1 (mAP too low): Should we lower the gate threshold preemptively? What's the minimum publishable detection number?
- R2 (activity crash): Is there a simpler activity head that could work as fallback?
- R3 (PSR broken): Does dropping PSR make the paper incomplete, or just less impressive?
- R6 (narrative dominated by detection gap): Is there a way to lead with efficiency and FiLM, and put detection second?

---

## 14. The Final Ask

### 14.1 What we need from you, Opus

**Question 14.1:** Please produce a single, ordered, day-by-day execution plan that answers:

1. **What to run next** — after rf2 completes, what's the next training command?
2. **What to measure** — for each experiment, what's the exact metric we should log?
3. **What to write** — which paper sections can be drafted before results arrive?
4. **What to cut** — if time runs short, what's the order of sacrifice?
5. **What to expect** — realistic numbers for each head, so we know when something is wrong.

### 14.2 What we know that we don't know

**Question 14.2:** Acknowledging the uncomfortable unknowns:

1. We don't know the true detection ceiling with ALL fixes applied simultaneously (detach=False, correct LR/BIAS, no OHEM, optimal anchors, full data, correct labels).
2. We don't know if PSR can be trained at all.
3. We don't know if activity recognition will produce any signal on 35% subset × 15 epochs.
4. We don't know if IKEA ASM evaluation code works.
5. We don't know if embedding_cache.py produces correct cached features for Phase-B ablations.
6. We don't know if VideoMAE V2 fits in 12GB VRAM alongside the 5-task model.
7. We don't know if the Feature Bank / ViT activity head was ever tested end-to-end.

Given these 7 unknowns, what's the most robust strategy — the one that produces a paper regardless of which unknowns resolve positively?

### 14.3 The meta question

**Question 14.3:** We have:
- 7 GUIDEs, all implemented
- A running training pipeline that produces non-catastrophic results
- A paper with good structure but no numbers
- >30 placeholders to fill
- 1 GPU
- ~2-4 days to results

Is the correct answer: "Run the experiments in the right order, fill the numbers, write the honest story, submit"? Or is there a fundamental problem we're not seeing that makes this impossible?

---

*End of document 50. Awaiting Opus analysis and execution plan.*
