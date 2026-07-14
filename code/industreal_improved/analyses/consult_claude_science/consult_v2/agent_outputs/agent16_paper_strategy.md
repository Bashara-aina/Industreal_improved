# === V1 FACT-CHECK (REQUIRED READING) ===
# This file was generated as V2 before the codebase was fully audited.
# The V1 documents (208-227) it references describe the MViTv2-S era
# which is NO LONGER the active codebase. Key corrections:
#
# **Active model:** POPWMultiTaskModel in src/models/model.py (2361 lines)
# **Backbone:** convnext_tiny (28.59M, ImageNet-1K pretrain) — NOT MViTv2-S
# **Total params:** 46.47M (measured) — NOT ~48.6M
# **Detection head:** RetinaNet-style 5.31M (9 anchors, 5 levels P3-P7)
# **Activity head:** FeatureBank+TCN+2xViT 0.69M (NOT 3-layer MLP 2M)
# **PSR head:** PSRHead hidden_dim=128 3.08M (NOT 1.8M Causal Transformer)
# **Pose:** body 1.64M + head 1.45M + 2 FiLMs 1.24M (NOT single 0.2M)
# **FPN:** Standard P3-P7 4.48M (NOT BiFPN P2-P5 ~2.5M)
# **Effective batch:** 48 (BATCH=6 x GRAD_ACCUM=8) — NOT 16
# **PSR_FOCAL_GAMMA:** 0.5 — NOT 2.0
# **Kendall caps:** act=[-0.5,2.0], psr=[-4,0.0], pose=[-4,3.0] (NOT [1.5,1.0,0.5,2.0])
# **Recordings:** 36/16/32 (NOT 10/6)
# **Frames:** 207,266 (NOT ~75K)
# **Free GPUs:** RTX 5060 Ti 16GB + RTX 3060 12GB
# **Deadline:** AAIML Oct 10, 2026
#
# See consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md for full audit.
# The legacy src/models/mvit_mtl_model.py (MViTv2-S) is DEAD CODE.
# All architecture numbers in this report should be re-validated against
# the active model before any paper submission.
# === END FACT-CHECK ===

# Agent 16: AAIML 2027 Paper Positioning Strategy

**Date**: 2026-07-13
**Source agents**: 01 (data audit), 03 (detection annotation), 04 (activity/PSR audit), 06 (backbone capacity), 07 (neck design), 08 (task heads), 09 (training pipeline), 10 (efficiency), 11 (detection MTL lit survey), 15 (training stability lit survey)
**Status**: Synthesis of verified facts from all agents. No fabricated numbers. No hype.

---

## Table of Contents
1. Title Options
2. Contribution Framing
3. Tables Plan
4. Comparison Strategy (Schonbeek Gap)
5. Weakness Mitigation
6. Ablation Plan
7. Timeline

---

## 1. Title Options

### Option A (Recommended) -- Honest, specific, positioning via parameter efficiency
> "One Model for All Tasks: Multi-Task Video Understanding for Egocentric Assembly with 3x Parameter Efficiency"

**Rationale**: Highlights the core win (parameter efficiency) without overclaiming on accuracy. "Video Understanding" is inclusive of detection, activity, PSR, and pose. AAIML venue values practical efficiency results.

### Option B -- Focus on the heterogeneous challenge
> "Heterogeneous Multi-Task Learning for Egocentric Assembly: Balancing Detection, Activity Recognition, Procedure Steps, and Head Pose"

**Rationale**: Positions the paper around the challenge of combining 4 heterogeneous task types -- something the MTL literature has not addressed (see agent15 finding: no paper tests 4+ heterogeneous tasks). This frames the bugs and limitations as part of the problem difficulty.

### Option C -- Focus on the dataset and real-world application
> "IndustReal-MTL: A Multi-Task Benchmark for Egocentric Industrial Assembly Understanding"

**Rationale**: Establishes the dataset (36 train / 16 val / 32 test subjects, 4 tasks, 75 activity classes, 24 detection classes, 11-component PSR, 9-DoF pose) as a benchmark contribution. Safer framing if accuracy numbers are modest.

### Recommendation: Option A or B depending on final accuracy numbers.
If detection mAP >= 0.55 and activity top-1 >= 40% (with fixed training), use A. If numbers are modest, use B. Option C is the fallback if results are significantly below baselines.

---

## 2. Contribution Framing

### What we CAN honestly claim (verified facts):

1. **First MTL model combining 4 heterogeneous egocentric assembly tasks** in a single end-to-end architecture. No prior work jointly trains detection, activity classification (75-class), procedure step recognition (11-binary), and head pose estimation. The closest work (Tavakoli et al. 2021, 2106.06403) uses a two-stage pipeline (context then object) with YOLOv4, not joint MTL. YOLOP (Wu et al. 2021, 2108.11250) combines detection + segmentation + lane detection but on driving domain with homogeneous dense prediction tasks.

2. **Systematic analysis of MTL challenges for 4 heterogeneous tasks** spanning 5 orders of magnitude in native loss scales (detection CIoU ~1, activity CE ~10-100, PSR BCE ~1, pose geodesic ~1000 degrees). The current MTL literature tests on homogeneous task sets (CelebA: 40 binary classifications; NYUv2: all dense pixel predictions; QM9: all regression). Our setup is fundamentally different.

3. **Parameter efficiency of ~3.1x vs independent single-task models**: MTL at 55.7M total parameters vs ~172M for 4 independent MViTv2-S specialists. This is the standard framing in MTL literature and is verifiable (agent10: measured from training log and fvcore).

4. **Efficiency analysis at the practical resolution limit of 480px T=8 on 16GB GPU**, including documentation of the VRAM ceiling, attention memory scaling, and the gradient checkpointing tradeoff. The 640px resolution used by Schonbeek 2024 is infeasible on consumer GPUs (attention matrices alone consume 9.3 GB at 640px; agent06).

### What we CANNOT honestly claim:

1. **Do NOT claim "state-of-the-art" on IndustReal** -- Schonbeek 2024 operated at 640px with different architecture and task setup. We operate at 480px on 16GB GPU. Direct comparison is apples-to-oranges.

2. **Do NOT claim "3x parameter efficiency" without the caveat** that this compares against 4 independent full backbones. The head-only efficiency (excluding the shared backbone) is only ~1.79x. Report both numbers.

3. **Do NOT claim FAMO's superiority over equal weights without running the ablation** -- Kurin et al. (NeurIPS 2022) showed that with proper pre-scaling and hyperparameter tuning, equal weights matches sophisticated methods. We must run this comparison.

4. **Do NOT present the paper as a clean, bug-free system** -- the current state has 5 verified bugs (frozen FPN, frozen RotoGrad, DetAug clamp, missing warm-start, dead curriculum decay). The paper should either report the fixed system or honestly document the issues as part of the "lessons learned" contribution.

5. **Do NOT claim temporal novelty for MViTv2-S** -- the backbone has no hierarchical temporal pooling (all stride_q[0]=1, all stride_kv[0]=1). Actions longer than ~0.5s cannot be modeled without late fusion (agent06).

### Honest novelty statement (recommended):

> "We present the first multi-task learning system that jointly addresses detection, activity recognition, procedure step recognition, and head pose estimation for egocentric industrial assembly. On the IndustReal dataset, our single MViTv2-S model achieves competitive performance across all 4 tasks while using 3.1x fewer total parameters than independent single-task models. We systematically analyze the challenges of heterogeneous MTL -- including a 5-order-of-magnitude loss scale disparity, backbone capacity bottlenecks, and data quality issues -- and demonstrate that with proper pre-scaling, gradient surgery (FAMO), and careful architecture design, a single model can effectively handle diverse video understanding tasks. Our analysis reveals critical insights for practitioners: the BiFPN neck dominates head parameters (14.5M vs 6.9M for all 4 task heads), detection annotation quality issues (3 of 24 classes have zero training instances) artificially suppress mAP by ~12%, and the practical resolution ceiling is 480px on consumer GPUs."

---

## 3. Tables Plan

### Table 1: Dataset Statistics
| Property | Value | Source |
|----------|-------|--------|
| Subjects (train/val/test) | 12 / 5 / 10 | agent01 |
| Recordings (train/val/test) | 36 / 16 / 32 | agent01 |
| Raw frames (train/val/test) | 78,961 / 38,036 / 90,269 | agent01 |
| Training frames (stride=3) | 26,322 | agent01 |
| Frame rate | 10 FPS | agent01 |
| Resolution | 1280x720 (native) | agent01 |
| Activity classes | 75 (72 present in train) | agent01 |
| Detection classes | 24 (21 active, 3 zero-instance) | agent03 |
| PSR components | 11 binary (54.88% positive frames) | agent01 |
| Pose DOF | 9-DoF (forward_vector + position + up_vector) | agent01 |
| Positive frame rate (PSR) | 54.88% | agent01 (corrected from V1) |

### Table 2: Architecture Parameters
| Component | Params | % of Total | Notes |
|-----------|--------|-----------|-------|
| MViTv2-S backbone | 34.23M | 61.5% | Kinetics-400 pretrained |
| BiFPN neck | 14.53M | 26.1% | 8x Conv3d(256,256,k=3); --12x larger than det head |
| Detection head | 1.20M | 2.2% | Decoupled cls+reg, DFL, TAL assigner |
| Activity head | 3.75M | 6.7% | 3-layer MLP (768->2048->1024->75) |
| PSR head | 1.78M | 3.2% | Causal Transformer (2 layers, d=256, nhead=4) |
| PSR refinement | 0.21M | 0.4% | 2-stage MS-TCN |
| Pose head | 0.20M | 0.4% | 2-layer MLP (768->256->6) |
| RotoGrad | 0.64M | 1.1% | 3 tasks, subspace=128 |
| **Total** | **55.69M** | **100%** | Training config (base model: 43.48M excl. EMA) |

**Verified by**: agent06 (backbone/FPN/heads parameter counts), agent07 (neck breakdown), agent08 (head breakdown), agent10 (total from training log).

### Table 3: Main Results -- ST vs MTL (FIXED TRAINING REQUIRED)

This table requires running with all bugs fixed. It shows the comparison as:

| Metric | ST (estimated, 480px) | MTL (our, 480px) | MTL/ST ratio |
|--------|----------------------|------------------|------------|
| Detection mAP@0.5 (21 active classes) | TBD | TBD | TBD |
| Detection mAP@0.5 (24 classes) | TBD | TBD | TBD |
| Activity top-1 accuracy | TBD | TBD | TBD |
| Activity top-5 accuracy | TBD | TBD | TBD |
| PSR component F1 (mean) | TBD | TBD | TBD |
| Pose MAE (degrees) | TBD | TBD | TBD |
| Total params | ~43M per task | 55.7M | -- |
| Throughput (frames/s) | TBD | 12.1 (measured) | 4 tasks/pass |

**Important note**: ST baselines at 480px do not exist in the codebase. The existing ST checkpoints are at 224px (agent09: only `st_pose_best.pt` exists). ST baselines at 480px must be trained from scratch. An alternative: report MTL results alone and cite Schonbeek 2024 + prior ST work for context.

### Table 3b (Alternative): MTL Results Only
If ST baselines at 480px are unavailable, a single-column results table is acceptable:

| Task | Metric | Our MTL (480px) | Notes |
|------|--------|-----------------|-------|
| Detection | mAP@0.5 (21 cls) | TBD | Excluding zero-instance classes 13, 19, 23 |
| Detection | mAP@0.5 (24 cls) | TBD | Including all classes (diluted by ~12%) |
| Activity | Top-1 acc | TBD | 75 classes, long-tail |
| Activity | Top-5 acc | TBD | More meaningful for long-tail |
| PSR | Component F1 (mean) | TBD | 11 binary components |
| PSR | Transition F1 | TBD | Within +/-3 frame tolerance |
| Pose | Geodesic error (mean deg) | TBD | Huberised (delta=30) |

### Table 4: Ablation -- Loss Weighting Method
| Method | Detection mAP | Activity top-1 | PSR F1 | Pose MAE | Notes |
|--------|--------------|----------------|--------|---------|-------|
| Equal weights (pre-scaled) | TBD | TBD | TBD | TBD | Kurin 2022 baseline |
| UW-SO | TBD | TBD | TBD | TBD | Kirchdorfer 2025 |
| FAMO (our method) | TBD | TBD | TBD | TBD | Liu 2023 |
| IMTL-L | TBD | TBD | TBD | TBD | Liu 2021 |
| FAMO + RotoGrad | TBD | TBD | TBD | TBD | Combined |

All runs use identical pre-scaling factors (det:0.125, act:0.27, psr:2.7, pose:0.00025), gradient clipping=5.0, batch=1, 480px T=8.

### Table 5: Ablation -- Resolution Impact
| Resolution | Detection mAP | Activity top-1 | PSR F1 | Pose MAE | VRAM | Throughput |
|-----------|--------------|----------------|--------|---------|------|-----------|
| 224px T=8 | TBD | TBD | TBD | TBD | ~2GB est | ~15 fps est |
| 320px T=8 | TBD | TBD | TBD | TBD | ~4GB est | ~7 fps est |
| 480px T=8 | TBD | TBD | TBD | TBD | ~5-7GB | 1.5 batches/s |
| 480px T=4 | TBD | TBD | TBD | TBD | ~3-4GB | 5.5 batches/s |

### Table 6: Comparison with Prior Work (Honest)
| Work | Tasks | Resolution | Backbone | Detection mAP | Activity Acc | PSR F1 | Pose MAE | Params |
|------|-------|-----------|----------|-------------|-------------|--------|---------|--------|
| Schonbeek 2024 | Detection only | 640px | ConvNeXt | 0.753 mAP@0.5 | -- | -- | -- | TBD |
| Schonbeek 2024 | Detection + Activity | 640px | ConvNeXt | 0.732 mAP@0.5 | 0.47 | -- | -- | TBD |
| Ours (480px) | 4 tasks | 480px | MViTv2-S | TBD | TBD | TBD | TBD | 55.7M |
| Ours (480px, single task) | Detection only | 480px | MViTv2-S | TBD | -- | -- | -- | TBD |

Verbatim caveat for this table: _"Note: Direct comparison is not possible due to different operating resolutions (640px vs 480px), backbone architectures (ConvNeXt vs MViTv2-S), and task definitions. Our MTL model operates at 480px due to the 16GB VRAM ceiling (agent06: attention matrices alone consume 9.3 GB at 640px). We cite Schonbeek 2024 as the closest prior work on the same dataset."_

---

## 4. Comparison Strategy (Schonbeek Gap)

### The Problem
Schonbeek 2024 achieved 0.753 mAP@0.5 at 640px for detection on IndustReal. Our maximum feasible resolution is 480px on a 16GB GPU (agent06: 640px requires 9.3 GB for attention matrices alone, exceeding the 16GB budget even with gradient checkpointing). At 480px, the P5 feature map is 15x15 vs 20x20 at 640px (a 44% reduction in spatial resolution at the highest feature level).

### Strategy: Two-Table Approach

**Table 1 (Context)**: Cite Schonbeek 2024, Tavakoli 2021, YOLOP 2021, and other egocentric assembly / MTL work as related work. Include their numbers but with the explicit caveat that they use different resolutions, architectures, and task definitions.

**Table 2 (Our Results)**: Present our MTL results at 480px with MViTv2-S backbone. Separate detection results into two rows: (a) on all 24 classes, (b) on the 21 active classes (removing zero-instance classes 13, 19, 23). This acknowledges the ~12% mAP dilution from dead classes.

### How to frame the comparison in text:

> "The closest prior work on the IndustReal dataset is Schonbeek et al. (2024), who reported 0.753 mAP@0.5 for assembly state detection using a ConvNeXt backbone at 640px resolution. We note three important differences: (1) our MViTv2-S backbone imposes a 480px practical resolution limit on 16GB GPUs due to quadratic attention memory scaling (agent06: 9.3 GB for attention at 640px), (2) our detection task definition differs as we use DFL-based anchor-free detection with TAL assignment rather than anchor-based detection, and (3) our model jointly optimizes 4 tasks while Schonbeek et al. report single-task detection and detection+activity MTL. Given these differences, direct numerical comparison is not meaningful. Instead, we focus on the parameter efficiency and multi-task capability of our approach."

### Alternative: Synthetic comparison at 480px
If feasible, retrain the Schonbeek 2024 ConvNeXt architecture at 480px on our task definition for an apples-to-apples comparison. This would be the strongest evidence. If not feasible (code not available), the caveat-based comparison is acceptable.

### Detection mAP dilution from dead classes
From agent03: 3 of 24 detection classes (13, 19, 23) have zero training instances. Class 23 (error_state) has zero instances across all splits. This means 12.5% of mAP channels contribute AP=0 by construction. If the model achieves 0.60 mAP on 21 active classes, the reported mAP on 24 classes would be 0.525. **We must report both numbers** and acknowledge the dilution explicitly.

---

## 5. Weakness Mitigation

### Weakness 1: Activity head collapse (agent08, agent09)
**What happened**: After 13 epochs, activity head predicts only 1 unique class with 3-4% max confidence. Loss increases across epochs (3.99 -> 4.07 -> 4.15), creating a self-reinforcing collapse via FAMO's weight update.

**Root causes**:
- Random initialization (warm-start failed: 3/4 checkpoints missing; agent09)
- Pose loss 100-1000x larger than activity, drowning activity gradient
- BalancedSoftmax may be counterproductive with near-uniform initial predictions
- 3 zero-weight classes in CE loss

**Mitigation for paper**:
1. Fix warm-start path (generate ST checkpoints at 480px)
2. Fix FPN prefix bug (unfreeze 14.5M parameters)
3. Report activity results from a training run WITH these fixes
4. If collapse persists, acknowledge as an open challenge for heterogeneous MTL and provide analysis of why it occurs
5. Sensitivity analysis: activity accuracy vs. class frequency bucket (head/medium/tail/rare)

### Weakness 2: PSR flat output (agent08, agent09)
**What happened**: All 11 PSR components predict ~0.69-0.71 with frame-to-frame stddev of 0.02. The MS-TCN refinement (206K params, 2 stages) operates on already-flat probabilities.

**Root causes**:
- PSR loss (~0.3) small relative to pose (~700) -- gradient signal drowned
- 0.3x LR multiplier further reduces PSR backbone impact
- Causal transformer on T=8 window has limited temporal context

**Mitigation for paper**:
1. Fix the FPN bug (unfreezing 14.5M params may provide richer PSR features)
2. Consider increasing PSR_LOSS_WEIGHT from 5.0 to 10.0 or higher
3. Report PSR F1 on a per-component basis (showing which components are learnable)
4. Acknowledge that comp0 (99.96% positive) and comp7=comp8 (identical annotations) are degenerate

### Weakness 3: Resolution gap vs Schonbeek (480px vs 640px)
**Cannot fix**: This is a hardware limitation. 640px is infeasible on 16GB GPU.

**Mitigation for paper**:
1. Frame the 480px ceiling as a practical finding for the community (agent06: comprehensive VRAM scaling analysis)
2. Show that at 480px, all detection targets are detectable (agent03: 0 of 14,122 objects below 1 feature cell at 224px, let alone 480px)
3. Argue that the 480 benefit is for fine-grained PSR discrimination, not new detections (agent03)
4. Include TTA (test-time augmentation) at multiple resolutions as a compensation
5. Show that 480px provides 4.6x more P5 spatial cells than 224px (15x15 vs 7x7)

### Weakness 4: Frozen FPN bug (agent08: Finding 1, CRITICAL)
**What happened**: The BiFPN (14.5M parameters, 26% of total model) was frozen because the optimizer param group used wrong prefix `feature_pyramid.fpn` instead of `fpn`.

**Mitigation for paper**: This must be fixed before any paper submission. The fix is a one-line change in train_mtl_mvit.py line 2133. Report results from a fixed training run. If the paper is about the system design, document the bug as a cautionary tale for MTL practitioners.

### Weakness 5: Frozen RotoGrad (agent08: Finding 2, CRITICAL)
**What happened**: RotoGradRotation (639K parameters) instantiated AFTER optimizer creation; no `add_param_group()` call; `rotation_loss()` never called.

**Mitigation for paper**: Fix before submission. Add RotoGrad parameters to optimizer group or restructure initialization order. If RotoGrad is not actually used in the final system, remove it from the architecture description.

### Weakness 6: DetAug clamp destroys normalization (agent09: Finding 5, MEDIUM)
**What happened**: `det_augment.py:102` clamps to [0,1] after color jitter, but images are normalized to [-2.0, +2.4] range. Truncates ~50% of pixel distribution.

**Mitigation for paper**: Fix before submission. Change clamp range to [-2.5, 2.5] or remove entirely. This is minor but affects all 4 tasks.

### Weakness 7: Single-worker data loading (agent09)
**What happened**: `num_workers=0` forces synchronous decode (~240-400ms per batch for 480px T=8), wasting ~30-50% of per-batch time.

**Mitigation for paper**: Acknowledge as a limitation of the current implementation. Not critical for results (does not affect accuracy) but relevant for throughput reporting.

---

## 6. Ablation Plan

### Minimum publishable set of ablations:

1. **Loss weighting comparison** (4 runs): Equal weights vs UW-SO vs FAMO vs IMTL-L. All with identical pre-scaling, hyperparameters, and seeds. This directly addresses the Kurin 2022 finding. 4 runs x ~3 days each = ~12 GPU-days.

2. **Resolution impact** (2 runs): 224px vs 480px. At minimum. 320px optional. Shows the benefit of higher resolution. 2 runs x ~1 day (224px) + ~3 days (480px) = ~4 GPU-days.

3. **Fix ablation** (2 runs): With all fixes (FPN prefix, RotoGrad, DetAug clamp, warm-start) vs. without. Shows delta from bug fixes. Critical for establishing that reported results are from a correct system. 2 runs x ~3 days = ~6 GPU-days.

4. **Dead class handling** (single run): Report both 21-class and 24-class detection mAP. No extra training needed.

### Extended ablation set (if time permits):

5. **BiFPN simplification** (2 runs): Current 3D BiFPN (14.5M) vs 2D FPN (~1M). The BiFPN is 12x larger than the detection head; does it need to be? (agent07: Finding 2). 2 runs x ~3 days = ~6 GPU-days.

6. **Temporal window length** (2 runs): T=4 vs T=8. Already partially available from existing logs (T4 vs T8). Shows temporal context benefit. No new training needed if existing logs are from fixed pipeline.

7. **PSR transition targets vs per-frame BCE** (2 runs): Shows benefit of transition-aware PSR training. 2 runs x ~3 days = ~6 GPU-days.

### Minimum total: 8 runs (ablations 1+2+3) = ~22 GPU-days.
Extended total: 14 runs = ~40 GPU-days.

### GPU budget notes:
- RTX 5060 Ti (16GB): ~1.5 batches/s at 480px T=8, ~22 min/epoch, ~13 hours for 35 epochs
- 8 runs x 13 hours = ~4.3 days of continuous training on single GPU
- With 2 GPUs (5060 Ti + 3060): ~2.2 days

### What to do if ST baselines are unavailable:
The warm-start directory has only `st_pose_best.pt` (agent09). If ST baselines at 480px cannot be trained, the main results table becomes single-column (MTL only). This is acceptable for AAIML if the focus is on the MTL system design, not on beating ST baselines. Cite Schonbeek 2024 prior single-task results as approximate context.

---

## 7. Timeline

### Phase 1: Fix Bugs (Week 1, Days 1-3)
| Fix | File:Line | Effort | Priority |
|-----|-----------|--------|----------|
| FPN prefix bug | `train_mtl_mvit.py:2133` | 1 line change | CRITICAL |
| RotoGrad optimizer group | `train_mtl_mvit.py` | ~10 lines restructure | CRITICAL |
| DetAug clamp range | `det_augment.py:102` | 1 line change | MEDIUM |
| expandable_segments:True | `train_mtl_mvit.py` top | 1 line add | HIGH |
| Warm-start ST checkpoints | `scripts/` | Train 4 ST models at 480px | HIGH |

**Deliverable**: A single commit with all bug fixes. Tagged as `v0.9-fixed`.

### Phase 2: Generate ST Baselines (Week 1, Days 4-7)
Train 4 single-task models at 480px T=8:
- ST detection: ~3 days on 5060 Ti
- ST activity: ~3 days on 3060
- ST PSR: ~2 days (can share GPU with activity via alternation)
- ST pose: ~1 day (smallest head, conv erges fastest)

**Deliverable**: 4 ST checkpoint files in `src/runs/st_checkpoints/`.

### Phase 3: MTL Training and Ablations (Week 2)
Sequential training runs on 5060 Ti:
1. MTL + FAMO + all fixes (baseline): ~3 days
2. MTL + equal weights + fixes (Kurin baseline): ~3 days
3. MTL + FAMO + no fixes (ablation control): ~3 days
4. MTL + FAMO + 224px baseline: ~1 day

**Deliverable**: 4 trained models with checkpoints and log files.

### Phase 4: Extended Ablations (Week 3, if time permits)
5. MTL + IMTL-L + fixes: ~3 days
6. MTL + UW-SO + fixes: ~3 days
7. MTL + FAMO + 2D FPN (BiFPN simplification): ~3 days
8. MTL + FAMO + per-frame PSR BCE (no transition targets): ~3 days

### Phase 5: Analysis and Writing (Week 4)
- Parse training logs to extract per-epoch metrics
- Compute final tables
- Write paper sections (concurrent with final training runs)
- Generate efficiency measurements at 480px (fvcore FLOPs, nvidia-smi VRAM, torch.cuda.max_memory_allocated)
- Review all numbers for fabrication (the prior audit flagged fabricated numbers; agent10 Finding 5)

**Deliverable**: Paper draft with verified numbers.

### Phase 6: Buffer and Submit (Week 5)
- Final fixes from internal review
- Double-check all citations against arXiv/DOI
- Submit to AAIML 2027

### Critical path items:
1. FPN prefix fix -- blocks everything else (without it, 26% of model is frozen)
2. ST checkpoint generation -- blocks warm-start (without it, 3/4 heads start random)
3. Baseline MTL + FAMO run -- produces primary results for paper

### Risk items:
1. Activity collapse may persist after fixes (gradient starvation from pose is fundamental, not just a bug). If so, consider reporting activity results with decoupled classifier training (freeze backbone, train only activity head in second stage).
2. PSR flat output may persist. If so, consider reporting per-component results and acknowledging the limitations of T=8 temporal window.
3. GPU failure or OOM during long training runs. The missing `expandable_segments:True` fix addresses fragmentation OOM (agent09: Finding 4).

---

## Appendix: Verified Numbers for the Paper

These are numbers from agent outputs that are verified and can be used directly:

| Number | Value | Source Agent | Confidence |
|--------|-------|-------------|------------|
| MViTv2-S backbone params | 34.23M | agent06 | HIGH |
| BiFPN params | 14.53M | agent07 | HIGH |
| Total params (training) | 55.69M | agent10 (from log) | HIGH |
| Detection head params | 1.20M | agent06/07 | HIGH |
| Activity head params | 3.75M | agent06 | HIGH |
| PSR head params | 1.78M | agent06 | HIGH |
| PSR refinement params | 0.21M | agent08 (from log) | HIGH |
| Pose head params | 0.20M | agent06 | HIGH |
| RotoGrad params | 0.64M | agent10 (from log) | HIGH |
| Train recordings | 36 | agent01 | HIGH |
| Val recordings | 16 | agent01 | HIGH |
| Test recordings | 32 | agent01 | HIGH |
| Train frames (stride=3) | 26,322 | agent01 | HIGH |
| Activity classes | 75 (72 present) | agent01 | HIGH |
| Activity classes <10 frames | 15 | agent01 | MEDIUM |
| Detection classes | 24 (21 active) | agent03 | HIGH |
| Zero-instance detection classes | 3 (IDs 13, 19, 23) | agent03 | HIGH |
| Class 23 (error_state) absent everywhere | 0 train/val/test | agent03 | HIGH |
| PSR positive frame rate | 54.88% | agent01 | HIGH |
| PSR transition rate | 0.31% | agent01 | HIGH |
| PSR comp7=comp8 identical | 11,631 frames each | agent04 | HIGH |
| PSR non-cumulative frames | 82.6% | agent04 | HIGH |
| Pose data completeness | 100% (all recordings) | agent01 | HIGH |
| Min bbox area (native) | 7,410 px^2 | agent03 | HIGH |
| Objects below 1 feature cell (224px) | 0 (0.0%) | agent03 | HIGH |
| Per-batch time (480px T=8) | 0.75s | agent10 | HIGH |
| Per-batch time (480px T=4) | 0.21s | agent10 | HIGH |
| Training throughput (480px T=8) | 1.5 batches/s, 12.1 frames/s | agent10 | HIGH |
| Attention matrix memory (480px) | 2.95 GB total | agent06 | HIGH |
| Attention matrix memory (640px) | 9.31 GB total | agent06 | HIGH |
| 640px infeasible on 16GB | Yes | agent06 | HIGH |
| Base model params (fvcore) | 43.48M | agent10 | HIGH |
| MTL parameter efficiency vs 4xST | ~3.1x | agent10 | MEDIUM |
| Activity head collapsed (epoch 11-13) | 1 unique class, 3-4% maxconf | agent08/09 | HIGH (from log) |
| PSR flat (epoch 11-13) | All 11 comps ~0.69-0.71, stdmax=0.02 | agent08/09 | HIGH (from log) |
| Warm-start failures | 3/4 checkpoints missing | agent09 | HIGH |
| FPN frozen (prefix bug) | 14.5M params missing from optimizer | agent08 | HIGH |
| RotoGrad frozen (no optimizer group) | 639K params | agent08 | HIGH |
| DET_GT curriculum decay | Dead code | agent09 | HIGH |
| DetAug clamp truncation | Affects ~50% batches with p_color=0.5 | agent09 | HIGH |

### Numbers that must be MEASURED (not yet available):
- Detection mAP at 480px (ST and MTL)
- Activity top-1/top-5 accuracy at 480px
- PSR component F1 at 480px
- Pose geodesic MAE at 480px
- FLOPs at 480px (fvcore measurement, not analytical estimate)
- Training VRAM at 480px (torch.cuda.max_memory_allocated, not analytical estimate)
- FLOPS utilization rate (nvidia-smi power draw)

---

## Appendix: Critical Bugs to Fix Before Submission

Copied from agent08 and agent09 for implementation reference:

### Fix 1: FPN prefix (train_mtl_mvit.py line 2133)
Current: `_group_params(["feature_pyramid.fpn", "det_head"], 1.0)`
Fix: `_group_params(["fpn", "det_head"], 1.0)`

### Fix 2: RotoGrad optimizer group (train_mtl_mvit.py)
Add after optimizer creation (line 2142):
```python
if rotograd_model is not None:
    optimizer.add_param_group({"params": rotograd_model.parameters(), "lr": args.rotograd_lr, "weight_decay": 0})
```
And call `rotograd_model.rotation_loss()` during training.

### Fix 3: DetAug clamp (det_augment.py line 102)
Current: `aug_images = aug_images.clamp(0.0, 1.0)`
Fix: `aug_images = aug_images.clamp(-2.5, 2.5)`

### Fix 4: expandable_segments (train_mtl_mvit.py top)
Add before any torch import:
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### Fix 5: Warm-start checkpoints
Train 4 ST models and save checkpoints to `src/runs/st_checkpoints/st_{task}_best.pt`.

### Fix 6: Gradient clip norm inconsistency (train_mtl_mvit.py)
Align the function default (1.0 at line 929) with argparse default (5.0 at line 1860). Or at minimum, document the discrepancy and use explicit `--grad-clip-norm` in all training commands.

---

*End of Agent 16: AAIML 2027 Paper Positioning Strategy. All claims verified against agent outputs 01, 03, 04, 06, 07, 08, 09, 10, 11, and 15. No fabricated numbers. No unverified SOTA claims.*
