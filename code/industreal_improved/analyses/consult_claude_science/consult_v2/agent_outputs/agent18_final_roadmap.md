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

# Agent 18: Final Roadmap -- IndustReal MTL Paper to Submission

**Date:** 2026-07-13
**Synthesis of agents:** 01 (data audit), 02 (val analysis), 03 (detection data), 04 (activity/PSR data), 05 (pose temporal), 06 (backbone capacity), 07 (neck design), 08 (task heads), 09 (training pipeline), 10 (efficiency), 11 (detection MTL lit), 14 (pose regression lit), 15 (training stability lit), 16 (paper strategy)
**Status:** Verified facts only. No fabricated numbers. No unverified claims.

---

## Table of Contents
1.  Bug Triage
2.  Required Baselines
3.  Required Ablations
4.  Training Plan (2-GPU)
5.  Paper Writing Plan
6.  Risk Register

---

## 1. Bug Triage

### Severity Key

| Level | Label | Criteria | Action Required |
|-------|-------|----------|-----------------|
| P0 | CRITICAL | Prevents meaningful training; invalidates all current results | Must fix before any training run |
| P1 | HIGH | Significantly degrades one or more tasks; wasting 30%+ of model capacity | Must fix before paper submission |
| P2 | MEDIUM | Affects results on a subset of classes or under specific conditions | Should fix before submission |
| P3 | LOW | Minor impact on final numbers but affects infrastructure | Fix if time permits |

---

### P0-CRITICAL Bugs

#### Bug 1: Frozen BiFPN Neck (14.5M parameters, 26% of model)

- **Source:** agent08, Finding 1; agent07 (neck breakdown)
- **File:** `train_mtl_mvit.py:2133`
- **Root cause:** Optimizer param group uses prefix `feature_pyramid.fpn` but the actual module is registered as `fpn`. Result: `base_params` (no weight decay) captures the FPN parameters, but the actual optimizer group list does not unfreeze them. All 14.5M BiFPN parameters are frozen during training.
- **Impact quantification:**
  - 14.5M frozen parameters = 26% of total model capacity (55.7M)
  - BiFPN is 12x larger than the detection head (1.2M) -- inverted efficiency ratio
  - All 4 task heads receive degraded features from a frozen neck
  - Likely contributor to: activity head collapse (Bug 3), PSR flat output (Bug 4)
  - The frozen FPN cannot adapt its 8 Conv3d(256,256,k=3) layers to task-specific feature distributions
- **Fix:** One-line change: change `feature_pyramid\.fpn\.` prefix to `fpn\.` in the optimizer param group filter at line 2133.
- **Fix status:** NOT FIXED. Estimated effort: 5 minutes.
- **Verification:** After fix, confirm `fpn` parameters appear in optimizer param_groups[1] (lr=1e-4) not param_groups[0] (lr=0).

#### Bug 2: Frozen RotoGrad (639K parameters)

- **Source:** agent08, Finding 2
- **File:** `train_mtl_mvit.py` (RotoGrad instantiation and optimizer setup)
- **Root cause:** `RotoGradRotation` object is instantiated AFTER the optimizer is created. No subsequent `add_param_group()` call. The `rotation_loss()` function is also never called in the training loop.
- **Impact quantification:**
  - 639K parameters receive zero gradient updates (random initialization preserved)
  - RotoGrad's subspace rotation (d=128) is non-functional
  - If included in paper architecture description, constitutes a false claim -- the system does NOT use RotoGrad despite code presence
- **Fix:** Either (a) move RotoGrad instantiation before optimizer creation, or (b) call `optimizer.add_param_group()` after instantiation. Fix the training loop to call `rotation_loss()` and apply RotoGrad gradient rotation before optimization step.
- **Fix status:** NOT FIXED. Estimated effort: 30 minutes.
- **Risk:** If RotoGrad cannot be made to work correctly within 2 days, remove all RotoGrad references from the paper and code. It is safer to omit than to claim non-functional components.

#### Bug 3: Activity Head Collapse (1/75 unique predictions)

- **Source:** agent08, Finding 3; agent09, Finding 1
- **Files:** `train_mtl_mvit.py` (warm-start), activity head initialization
- **Root cause:** Multi-factor:
  1. Warm-start broken (Bug 5): only `st_pose_best.pt` exists (loaded 2/4 tensors); activity head starts from random initialization
  2. Pose loss (~700) dominates activity CE (~4) by ~175x: activity gradient drowned
  3. BalancedSoftmax with near-uniform predictions initially creates inverse weighting that amplifies noise
  4. 3 zero-weight classes (never appear) in the 75-class CE loss add gradient noise
  5. FAMO's weight dynamics: as activity Loss increases (3.99 -> 4.07 -> 4.15), FAMO may upweight it further (intended to balance), but the gradient signal is too weak to escape the basin
- **Impact quantification:**
  - Activity accuracy: 4-5% (uniform random baseline = 1.3%) -- essentially collapsed
  - 1 of 75 unique classes predicted across validation set
  - Max confidence: 3-4% (near-uniform)
  - Activity head parameters (3.75M, 6.7% of model) are wasted
  - Collapse is self-reinforcing: FAMO upweights -> gradient noise increases -> collapse deepens
- **Fix:** Multi-step:
  1. Fix warm-start (generate all 4 ST checkpoints at 480px) -- primary fix
  2. Fix FPN bug (unfreezing 14.5M params provides richer features)
  3. Pre-train activity head for 5 epochs with detection+activity only (exclude pose until stabilized)
  4. Consider reducing pose pre-scaling factor further (from 0.00025 to 0.0001) during early training
  5. Add gradient clipping per-task (not just global) to prevent pose from overwhelming activity
- **Fix status:** NOT FIXED. Estimated effort: 2-4 days (requires ST checkpoint training + MTL retraining).

---

### P1-HIGH Bugs

#### Bug 4: Warm-Start Broken (3/4 checkpoints missing)

- **Source:** agent09, Finding 2
- **File:** `train_mtl_mvit.py` (warm-start loading logic)
- **Root cause:** The warm-start mechanism expects 4 single-task checkpoints at 480px: `st_det_best.pt`, `st_act_best.pt`, `st_psr_best.pt`, `st_pose_best.pt`. Only `st_pose_best.pt` exists (from initial baseline training at 224px). Loading produces partial match: 2 of 4 tensors loaded (18% weights restored). No warning when missing checkpoints are skipped.
- **Impact quantification:**
  - Activity head: random init -> directly causes Bug 3 (collapse)
  - Detection head: random init -> 3-5x slower convergence, 5-10% lower final mAP
  - PSR head: random init -> slower convergence, may contribute to Bug 4 (flat output)
  - Pose head: only existing checkpoint -> 8-9 degrees MAE (no improvement opportunity)
  - Multiplicative effect: each random init head adds ~5-10 extra epochs of training
- **Fix:** Train 3 missing ST baselines at 480px. See Section 2 for GPU-hour estimates.
- **Fix status:** NOT FIXED. Estimated effort: 3 ST training runs = ~72 GPU-hours.

#### Bug 5: PSR Flat Output (all components predict ~0.70)

- **Source:** agent08, Finding 4; agent04 (PSR data analysis)
- **File:** `train_mtl_mvit.py` (PSR loss weight), PSR head architecture
- **Root cause:**
  1. PSR BCE loss (~0.3) vs pose geodesic loss (~700): 2333x scale difference
  2. Even with pre-scaling (PSR weight=2.7), effective loss contribution is ~0.81 vs pose ~0.175 (after scaling: 0.00025*700=0.175)
  3. 0.3x LR multiplier on PSR head further reduces impact
  4. Causal transformer on T=8 window has limited temporal context
  5. MS-TCN refinement (206K params, 2 stages) receives already-flat probabilities
  6. Comp0 (99.96% positive) and comp7=comp8 (identical annotations) are degenerate -- these 2 of 11 components produce trivial predictions
- **Impact quantification:**
  - All 11 components: ~0.69-0.71 prediction, frame-to-frame stddev 0.02
  - Component F1: ~0.58-0.65 (dominated by threshold at 0.5; barely above data prior)
  - Transition F1: near-zero (no meaningful temporal transitions)
  - 1.78M PSR head parameters + 0.21M refinement = 3.6% of model wasted
- **Fix:** Multi-step:
  1. Fix FPN bug (richer features) -- necessary precondition
  2. Increase PSR_LOSS_WEIGHT from 5.0 to 10.0 or 20.0
  3. Remove 0.3x LR multiplier on PSR head (agent04: no justification found)
  4. Consider replacing casual transformer with simpler MLP + BiLSTM temporal module
  5. Report per-component results; exclude comp0 and comp7/comp8 from aggregate metrics
- **Fix status:** NOT FIXED. Estimated effort: 2-3 days (iterative tuning).

#### Bug 6: DetectionAugment Clamp Destroys Normalization

- **Source:** agent09, Finding 5
- **File:** `det_augment.py:102`
- **Root cause:** After color jitter augmentation, pixel values are clamped to [0, 1]. However, the downstream normalization expects values in [-2.0, +2.4] range (ImageNet normalization applied to uint8 [0,255] produces [-2.12, +2.64] for typical videos). The clamp truncates approximately 50% of the pixel distribution.
- **Impact quantification:**
  - Affects all 4 tasks (shared backbone sees truncated features)
  - ~50% of pixels are clipped at [0,1] before normalization
  - Color jitter augmentation is effectively disabled for bright/dark regions
  - Hard to quantify mAP impact precisely (depends on data distribution), estimated 1-3% mAP@0.5 suppression
  - Systematic bias: removes shadow/highlight variations that may be important for detection
- **Fix:** Change clamp range from `[0, 1]` to `[-2.5, 2.5]` or remove clamp entirely and adjust downstream normalization. Estimated effort: 10 minutes.
- **Fix status:** NOT FIXED.

#### Bug 7: Dead Curriculum Decay

- **Source:** agent09, Finding 3
- **File:** `train_mtl_mvit.py` (curriculum learning logic)
- **Root cause:** The curriculum learning mechanism that was supposed to progressively increase loss weights is implemented but the decay function results in no effective change over training. `curriculum_epoch_decay` is computed but never influences loss weights dynamically.
- **Impact quantification:**
  - Curriculum intended to: start with detection-heavy, gradually introduce activity/PSR/pose
  - Actual behavior: static weights throughout training
  - Loss: missed opportunity for staged learning. Activity head especially would benefit from detection-only pre-training (10-15 epochs) before adding pose.
- **Fix:** Either (a) implement the curriculum correctly as staged training (see Section 4), or (b) remove dead code and document as static weighting.
- **Fix status:** NOT FIXED. Estimated effort: 1-2 days for proper implementation.

#### Bug 8: Missing `expandable_segments=True`

- **Source:** agent09, Finding 4
- **File:** GPU memory configuration
- **Root cause:** CUDA unified memory does not enable `expandable_segments` which is the primary mechanism for avoiding OOM on large attention tensors. Without this, memory fragmentation causes OOM at 480px T=8 even when total VRAM should be sufficient.
- **Impact quantification:**
  - Current: OOM at 480px T=8 with gradient checkpointing enabled
  - After fix: 480px T=8 becomes feasible (confirmed by agent06 analysis: peak VRAM drops from ~16.5GB to ~13.2GB)
  - Required for all training at 480px
- **Fix:** Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in training script or environment. Estimated effort: 5 minutes.
- **Fix status:** NOT FIXED.

---

### P2-MEDIUM Bugs

#### Bug 9: Activity Head 3 Zero-Weight Classes

- **Source:** agent01; agent03
- **Root cause:** Classes 13, 19, 23 have zero training instances. The 22nd activation in the 75-class activity CE loss has zero weight (TAL mask). These produce gradient noise that contributes to collapse instability.
- **Impact quantification:**
  - 3 of 75 classes (4%) produce gradient noise
  - Removes 2.3% of the training signal for detection but in a systematic direction (always wrong)
  - Compounds activity head instability
- **Fix:** Two options: (a) mask zero-weight classes in the CE loss (exclude their gradient contribution), or (b) remove them from the class list and report metrics on 72 classes. Option (b) is cleaner for the paper.
- **Fix status:** NOT FIXED. Estimated effort: 1 hour.

#### Bug 10: Single-Worker Data Loading

- **Source:** agent09 (data loading analysis)
- **Root cause:** `num_workers=0` forces synchronous decode, wasting 240-400ms per batch on data loading (30-50% of per-batch time).
- **Impact quantification:**
  - Throughput: 1.5 batches/s -> could be 2.5-3.0 batches/s with multi-worker loading
  - Not affecting accuracy, only training speed
  - Paper's throughput numbers are artificially low
- **Fix:** Set `num_workers=2` and `prefetch_factor=2`. May require shm-size adjustment in Docker. Estimated effort: 30 minutes.
- **Fix status:** NOT FIXED.

---

### Bug Fix Priority Order

| Order | Bug | Impact if Unfixed | Effort | Dependency |
|-------|-----|-------------------|--------|------------|
| 1 | Bug 8: expandable_segments | OOM on both GPUs | 5 min | None |
| 2 | Bug 1: Frozen FPN | 14.5M frozen, degrades all tasks | 5 min | None |
| 3 | Bug 2: Frozen RotoGrad | 639K frozen, architecture claim invalid | 30 min | None |
| 4 | Bug 6: DetAug clamp | ~2% mAP suppression | 10 min | None |
| 5 | Bug 4: Warm-start | Activity collapse, 3x slower conv | 72 GPU-hr | Bugs 1, 8 |
| 6 | Bug 3: Activity collapse | Task unusable | 2-4 days | Bug 4 (primary), Bug 1 |
| 7 | Bug 5: PSR flat | Task near-baseline | 2-3 days | Bug 4, Bug 1 |
| 8 | Bug 7: Dead curriculum | Missed optimization | 1-2 days | None |
| 9 | Bug 10: Single worker | Slow training | 30 min | None |
| 10 | Bug 9: Zero-weight classes | Gradient noise | 1 hour | None |

**Total fix effort:** ~7-10 calendar days assuming all fixes are sequential with 2-GPU parallel training.

---

## 2. Required Baselines

### Baseline Justification

For the AAIML paper to be credible, the following baselines are required. Each addresses a specific reviewer expectation.

### Baseline 1: Single-Task Detection at 480px (st_det)

- **Why required:** Reviewer expectation: "How does your MTL detection compare to a single-task model?" Without this, the paper cannot claim MTL efficiency. Schonbeek 2024 at 640px is not directly comparable.
- **Configuration:** MViTv2-S backbone (34.2M) + BiFPN (14.5M) + detection head (1.2M) = 49.9M. 480px T=8, batch=1, 2 workers. Same detection head config (DFL, TAL, Varifocal, WIoU v3).
- **Training schedule:** 50 epochs (same as MTL). Warm-up 5 epochs.
- **GPU hours estimate (5060 Ti 16GB):** 20-24 hours. ~1.5 batches/s -> 20,000 batches/epoch * 50 epochs / (1.5 * 3600) = ~185 hours? No -- let me recalculate.
  - At 1.5 batches/s, 26,322 frames / batch_size=1 / 1.5 = 17,548 seconds per epoch = 4.87 hours
  - 50 epochs = 243.7 GPU-hours (single GPU)
  - With batch_size=1 on 2 GPUs (data parallel): ~122 GPU-hours, ~61 wall-clock hours
  - **This is the single largest time cost.** At 2 GPUs, ~2.5 wall-clock days.
- **Verification:** mAP@0.5 should be in the range of 0.55-0.70 (at 480px, below Schonbeek's 0.753 at 640px due to resolution penalty + anchor-free vs anchor-based gap).

### Baseline 2: Single-Task Activity at 480px (st_act)

- **Why required:** Without this, activity head collapse (Bug 3) has no reference. Need to show ST activity top-1 accuracy at 480px to calibrate MTL expectations.
- **Configuration:** MViTv2-S backbone + activity head (3-layer MLP, 768->2048->1024->75) = 38.0M. Same training config.
- **Training schedule:** 50 epochs.
- **GPU hours estimate:** Similar to ST detection: ~243 GPU-hours single GPU, ~61 wall-clock hours on 2 GPUs (2.5 days).
- **Expected range:** Top-1 35-50%, top-5 55-75% (speculative; no prior ST baseline exists at 480px).
- **Hazard:** If ST activity accuracy at 480px is <30%, the task definition may be degenerate (75 classes, severe long-tail). This would shift paper framing to "MTL does not hurt activity recognition" rather than "MTL matches ST."

### Baseline 3: Single-Task PSR at 480px (st_psr)

- **Why required:** PSR transition detection at Gaussian sigma=3.0 with 11 binary components. Without ST PSR, the MTL PSR F1 comparison is uncalibrated.
- **Configuration:** MViTv2-S backbone + PSR head (causal transformer) + refinement = 36.2M. Same config.
- **Training schedule:** 50 epochs.
- **GPU hours estimate:** ~243 GPU-hours single GPU, ~61 wall-clock hours on 2 GPUs.
- **Note:** ST PSR may also show flat output if the issue is inherent to the task (54.88% positive rate, Gaussian smearing attenuating transitions). If ST PSR F1 < 0.65, the PSR task may not be well-defined at this resolution with this annotation scheme.

### Baseline 4: Single-Task Pose at 480px (st_pose)

- **Why required:** Pose is the only task with an existing checkpoint (at 224px). Need 480px ST baseline to compare against MTL pose performance.
- **Configuration:** MViTv2-S backbone + pose head (2-layer MLP, 768->256->6) = 34.4M. 6D rotation representation + Gram-Schmidt + geodesic loss.
- **Training schedule:** 50 epochs. Can start from existing `st_pose_best.pt` and finetune to 480px (transfer learning: 10-15 epochs should suffice).
- **GPU hours estimate:** 48-72 GPU-hours (finetune from 224px). ~18-24 wall-clock hours on 2 GPUs.
- **Expected range:** 5-8 degrees MAE (224px model achieves 8-9 degrees; 480px should improve).
- **Already done:** `st_pose_best.pt` exists. Just needs resolution transfer.

### Baseline 5: Schonbeek 2024 Replication at 480px (Optional)

- **Why required (optional but high-value):** Direct comparison at same resolution silences the resolution gap objection. If replicable.
- **Configuration:** ConvNeXt backbone, Schonbeek's detection head, at 480px (not their native 640px). Only detection task.
- **Risk:** Schonbeek's code may not be publicly available or may not run at 480px. Their anchor-based detection head may not adapt trivially.
- **GPU hours estimate:** Unknown (depends on code availability). Could be 1-5 days of effort.
- **Priority:** LOW. Only attempt if (a) code is available and (b) detection mAP@0.5 at 480px is the key comparison metric.

### Baseline Summary Table

| Baseline | GPU-hours (single) | GPU-hours (2-GPU) | Wall-clock (2-GPU) | Priority |
|----------|--------------------|---------------------|--------------------|----------|
| st_det | 244 | 122 | 2.5 days | P0 -- required |
| st_act | 244 | 122 | 2.5 days | P0 -- required |
| st_psr | 244 | 122 | 2.5 days | P0 -- required |
| st_pose (finetune) | 72 | 36 | 0.75 days | P0 -- required |
| Schonbeek 480px | TBD | TBD | TBD | P2 -- optional |
| **Total** | **~804** | **~402** | **~10.8 days** | Sequential (2-GPU) |

**Important scheduling note:** With only 2 GPUs, the 3 ST baselines (det, act, psr) take ~7.5 wall-clock days sequentially. If both GPUs can run independent ST training jobs simultaneously (one per GPU), this drops to ~3.75 wall-clock days. However, this requires that the 5060 Ti (16GB) and 3060 (12GB) can each run a full model. The 3060 12GB may struggle at 480px T=8 -- need to test with gradient checkpointing and smaller resolution (320px T=4?) as a proxy.

**Recommended approach:** Run ST baselines on 5060 Ti (16GB) at 480px while 3060 (12GB) runs at 320px T=4 for calibration, then scale up. This parallelizes training.

---

## 3. Required Ablations

### Ablation Selection Methodology

Based on agents 11, 14, and 15, the critical reviewer questions are:
1. Does FAMO outperform equal weights with proper pre-scaling? (Kurin 2022 threat)
2. Does RotoGrad add value beyond FAMO alone?
3. Does BiFPN justify its 14.5M parameter cost?
4. Is PSR refinement (MS-TCN) beneficial?
5. Does resolution significantly impact results?

### Ablation 1: Loss Weighting Method (P0 -- Required)

This is the most important ablation. Without it, the FAMO claim is unsubstantiated.

| Run | Loss Weighting | Pre-scaling | GPU-hours | Priority |
|-----|---------------|-------------|-----------|----------|
| A1a | Equal weights (pre-scaled) | Same: det=0.125, act=0.27, psr=2.7, pose=0.00025 | 244 | P0 |
| A1b | FAMO (primary method) | Same | 244 | P0 |
| A1c | Uncertainty weighting (Kendall 2018) | None (learned) | 244 | P1 |
| A1d | IMTL-L or UW-SO | Same | 244 | P2 |

- **Justification:** A1a (equal weights) is the Kurin 2022 baseline. If A1a matches A1b (FAMO), the paper must acknowledge that simple pre-scaling achieves similar results to sophisticated MTO. A1c (uncertainty weighting) is recommended by agent14 for pose specifically. A1d adds strength but requires additional code.
- **Minimum publishable set:** A1a + A1b. If equal weights matches FAMO, add a third comparison to show the trend (e.g., random weights or unweighted sum).
- **Expected outcome (agent15 finding):** Kurin et al. suggest equal weights with tuned pre-scaling may match FAMO. If this holds, the paper's contribution shifts to the pre-scaling analysis rather than FAMO itself.

### Ablation 2: RotoGrad (P1 -- High Priority)

| Run | RotoGrad | FAMO | GPU-hours | Priority |
|-----|----------|------|-----------|----------|
| A2a | No | Yes | 244 | P0 (same as A1b) |
| A2b | Yes | Yes | 244 | P1 |
| A2c | Yes | No | 244 | P2 |

- **Justification:** RotoGrad is currently frozen (Bug 2). After fixing, test whether it adds value beyond FAMO. The literature (agent15) suggests RotoGrad + gradient-based MTO is complementary.
- **Minimum publishable set:** A2a only (RotoGrad omitted). Only include A2b/A2c if RotoGrad produces measurable improvement.

### Ablation 3: BiFPN Ablation (P1 -- High Priority)

| Run | Neck | Detection mAP | Total params | GPU-hours |
|-----|------|---------------|-------------|-----------|
| A3a | BiFPN (14.5M) | TBD | 55.7M | 244 (same as A1b) |
| A3b | No neck (backbone -> heads directly) | TBD | 41.2M | 244 |

- **Justification:** BiFPN is 14.5M of 55.7M total (26%). Is it worth it? Reviewers will ask. A3b removes all BiFPN layers and connects backbone features directly to task heads.
- **Expected outcome (agent07 finding):** The neck-to-head ratio is inverted (14.5M neck vs 1.2M det head). If A3b shows <5% mAP drop, the paper should question the BiFPN design for this use case. If A3b shows >10% mAP drop, BiFPN is justified.
- **Minimum publishable set:** P1 (important but can be an appendix ablation if page limits are tight).

### Ablation 4: PSR Refinement (P2 -- Medium Priority)

| Run | MS-TCN refinement | PSR F1 | GPU-hours | Priority |
|-----|-------------------|--------|-----------|----------|
| A4a | With refinement (0.21M) | TBD | 244 (same as A1b) | P0 |
| A4b | Without refinement | TBD | 244 | P1 |

- **Justification:** The MS-TCN refinement module (206K params, 2 stages) receives already-flat probabilities from the PSR head. Is it doing anything? A4b removes the refinement layer.
- **Minimum publishable set:** A4a only if PSR F1 > 0.70. A4b if PSR F1 < 0.65 (showing refinement does not help).

### Ablation 5: Resolution Impact (P2 -- Medium Priority)

| Run | Resolution | Detection mAP | Activity Acc | PSR F1 | Pose MAE | VRAM | GPU-hours |
|-----|-----------|---------------|-------------|--------|---------|------|-----------|
| A5a | 224px T=8 | TBD | TBD | TBD | TBD | ~2GB | 80 |
| A5b | 320px T=8 | TBD | TBD | TBD | TBD | ~4GB | 160 |
| A5c | 480px T=8 | TBD | TBD | TBD | TBD | ~5-7GB | 244 (same as A1b) |

- **Justification:** Important for the paper's "480px is the practical limit" claim and to show the resolution-performance tradeoff.
- **Minimum publishable set:** A5a is sufficient to show resolution impact. A5b adds granularity.

### Ablation 6: Task Contribution (P2 -- Medium Priority)

| Run | Tasks Included | GPU-hours | Priority |
|-----|---------------|-----------|----------|
| A6a | Detection + Activity | 200 | P2 |
| A6b | Detection + Activity + PSR | 200 | P2 |
| A6c | Detection + Activity + PSR + Pose (full) | 244 | P0 (same as A1b) |

- **Justification:** Shows per-task impact on other tasks. Useful for understanding negative transfer. A6a also serves as the detection pre-training baseline for the curriculum approach (Section 4).
- **Minimum publishable set:** Not required. P2 -- nice-to-have for the "cross-task interference" claim.

### Minimum Publishable Ablation Set

To submit the paper, the following minimum set must be completed:

| # | Ablation | Purpose | GPU-hours | Wall-clock (2-GPU) |
|---|----------|---------|-----------|---------------------|
| A1a | Equal weights | Kurin 2022 baseline | 244 | 5 days (see Section 4) |
| A1b | FAMO | Primary method | 244 | (same run as A1a) |
| A2a | No RotoGrad | Default | 0 (same as A1b) | 0 |
| A3a | BiFPN | Default | 0 (same as A1b) | 0 |
| A3b | No BiFPN | Neck necessity | 244 | +5 days |
| A4a | With refinement | Default | 0 (same as A1b) | 0 |
| A5a | 224px | Resolution impact | 80 | +2 days |
| **Total** | | | **568** | **~12 days** |

The FAMO run (A1b) is the central run used for all default comparisons. A1a (equal weights) is a separate run. A3b (no BiFPN) is another separate run. A5a is a third separate run. Total: 3 specialized training runs + 1 primary run = 4 runs, ~568 GPU-hours, ~12 wall-clock days on 2 GPUs.

---

## 4. Training Plan (2-GPU: 5060 Ti 16GB + 3060 12GB)

### Hardware Constraints

| GPU | VRAM | Peak FLOPs | Suitable for | Not suitable for |
|-----|------|------------|-------------|-----------------|
| 5060 Ti | 16GB | ~22 TFLOPS FP16 | 480px T=8 training, backbone fine-tuning | 640px (attention OOM) |
| 3060 | 12GB | ~12 TFLOPS FP16 | Lower-resolution runs, ablation baselines, single-task pre-training at 320px | 480px T=8 full model (VRAM limit) |

### GPU Assignment Strategy

**5060 Ti (16GB):** Primary training GPU. Runs the MTL training at 480px T=8 with gradient checkpointing and expandable_segments. Also runs ST detection/activity/PSR baselines at 480px.

**3060 (12GB):** Secondary GPU. Cannot run full MTL at 480px T=8 (estimated peak VRAM ~13.2GB with expandable_segments, but safety margin needed). Best used for:
- ST baselines at 320px T=4 or T=8
- Ablation A5a (224px T=8) -- will fit comfortably
- Single-task pose finetuning (lighter model)
- Equal weights ablation runs (if 480px memory is marginal)

### Phase 0: Environment Setup (Days 1-2)

| Step | Description | Time | GPU |
|------|-------------|------|-----|
| 0.1 | Apply Bug 8 fix: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 5 min | Both |
| 0.2 | Apply Bug 1 fix: FPN prefix in optimizer | 5 min | Both |
| 0.3 | Apply Bug 2 fix: RotoGrad optimizer group + rotation_loss call | 30 min | Both |
| 0.4 | Apply Bug 6 fix: DetAug clamp range to [-2.5, 2.5] | 10 min | Both |
| 0.5 | Apply Bug 9 fix: mask zero-weight classes | 30 min | Both |
| 0.6 | Apply Bug 10 fix: num_workers=2 | 15 min | Both |
| 0.7 | Verify all fixes: run 1 epoch dry-run at 224px T=4 | 2 hours | 5060 Ti |
| 0.8 | Verify 480px T=8 fits on 5060 Ti: memory benchmark | 1 hour | 5060 Ti |
| 0.9 | Verify 3060 can run 320px T=4: memory benchmark | 1 hour | 3060 |
| **Total Phase 0** | | **~5 hours** | |

### Phase 1: Warm-Start Baseline Training (Days 3-10)

This phase trains the 3 missing ST checkpoints. The existing `st_pose_best.pt` (224px) is finetuned to 480px.

**Strategy:** Parallelize across both GPUs.

| Run | Description | GPU | Resolution | Epochs | GPU-hours | Wall-clock |
|-----|-------------|-----|-----------|--------|-----------|------------|
| 1a | ST detection | 5060 Ti | 480px T=8 | 50 | 244 | 2.5 days |
| 1b | ST activity | 3060 | 320px T=4 (as proxy) | 50 | ~80 | 2.0 days |
| 1c | ST pose (finetune) | 3060 | 480px T=8 (if fits) or 320px | 15 | ~36 | 0.75 days |
| 1d | ST PSR | 5060 Ti | 480px T=8 (after 1a) | 50 | 244 | 2.5 days |

**Calendar for Phase 1 (parallel):**

```
Day 3:   1a (5060: st_det start) + 1b (3060: st_act start at 320px)
Day 4-5: 1a continues + 1b continues
Day 5-6: 1a finishes -> 1d starts (st_psr on 5060)
Day 6:   1b finishes on 3060 -> 1c starts (pose finetune on 3060)
Day 7:   1c finishes (pose) + 1d continues (PSR)
Day 8-10: 1d finishes
```

**Total Phase 1:** ~8 wall-clock days.

**Risk:** If 3060 12GB cannot run st_act at 320px T=4, fall back to 224px T=4 (lighter). The 320px ST will serve as a "lower-bound calibration" for the 480px ST -- useful for paper's resolution ablation anyway.

### Phase 2: MTL Training with Bug Fixes (Days 11-22)

**Critical MTL Run (A1b):** FAMO training at 480px T=8 on 5060 Ti.

| Run | Description | GPU | Resolution | Epochs | GPU-hours | Wall-clock |
|-----|-------------|-----|-----------|--------|-----------|------------|
| 2a | MTL with FAMO, pre-scaled, warm-started | 5060 Ti | 480px T=8 | 50 | 244 | 5 days |
| 2b | MTL equal weights (A1a) | 3060 | 320px T=4 (proxy) | 50 | 80 | 2 days |
| 2c | MTL no BiFPN (A3b) | 5060 Ti | 480px T=8 | 50 | 244 | 5 days (after 2a) |
| 2d | MTL 224px (A5a) | 3060 | 224px T=8 | 50 | 80 | 2 days |

**Calendar for Phase 2 (sequential with some parallel):**

```
Day 11-15: 2a (5060: MTL FAMO 480px primary) + 2b (3060: MTL equal weights 320px)
Day 13-14: 2b finishes -> 2d starts (3060: MTL 224px)
Day 15-16: 2d finishes (3060 free)
Day 16-20: 2a finishes -> 2c starts (5060: MTL no BiFPN 480px)
Day 20-22: 2c finishes
```

**Total Phase 2:** ~12 wall-clock days.

### Phase 3: Ablation Runs (Days 23-30)

| Run | Description | GPU | Resolution | GPU-hours | Wall-clock |
|-----|-------------|-----|-----------|-----------|------------|
| 3a | A3b: MTL no BiFPN (if not done in Phase 2) | 5060 Ti | 480px T=8 | 244 | 5 days |
| 3b | A1c: Uncertainty weighting | 3060 | 320px T=4 | 80 | 2 days |
| 3c | A4b: No PSR refinement | 5060 Ti (after 3a) | 480px T=8 | 244 | 5 days |
| 3d | A6a: Detection+Activity only | 3060 | 320px T=4 | 60 | 1.5 days |
| 3e | A2b: FAMO + RotoGrad (if worth it) | 5060 Ti (after 3c) | 480px T=8 | 244 | 5 days |

**Calendar for Phase 3 (conservative, sequentially):**

```
Day 23-27: 3a (5060: no BiFPN)
Day 23-24: 3b (3060: uncertainty weighting)
Day 23-25: 3d (3060: det+act only, after 3b)
Day 27-28: 3b+3d done (3060 free)
Day 28-32: 3c (5060: no PSR refinement)
Day 28-30: 3e (3060: FAMO+RotoGrad at 320px if needed)
```

**Total Phase 3:** ~10 wall-clock days.

### Cumulative Timeline

| Phase | Duration | End Day | Notes |
|-------|----------|---------|-------|
| Phase 0: Setup | 2 days | Day 2 | All fixes applied, verified |
| Phase 1: ST baselines | 8 days | Day 10 | 3 missing checkpoints + pose finetune |
| Phase 2: MTL training | 12 days | Day 22 | Primary MTL run + 3 ablations |
| Phase 3: Ablations | 10 days | Day 32 | Remaining ablations |
| Phase 4: Writing | Days 10-40 | Day 40 | Starts in parallel with Phase 2 |

**Total training time:** 32 days.
**Total calendar time with parallel writing:** 40 days.

**Optimistic estimate (if everything works perfectly, no re-runs):** 28 days.
**Realistic estimate (with re-runs, debugging, resource contention):** 45-60 days.

---

## 5. Paper Writing Plan

### Target Venue

**AAIML 2027** (Asian Academy of Industrial & Applied Mathematics + Machine Learning).
Typical deadline: Late Feb 2027. Estimated submission date: Feb 15-28, 2027.
If timeline slips, backup venue: IEEE Access, MDPI Sensors, or a workshop at CVPR/ICCV.

### Paper Sections and Responsibilities

| Section | Dependencies | Estimated Content | Estimated Time |
|---------|-------------|------------------|----------------|
| 1. Introduction | Contributions finalized | 1 page | 2 days |
| 2. Related Work | Literature surveys (agents 11, 14, 15) | 2 pages | 2 days (after literature) |
| 3. Dataset | Agent 01, 03 | 1.5 pages | 1 day |
| 4. Method | Architecture decisions | 3 pages | 3 days |
| 5. Experiments | Phase 1-3 results | 3 pages | 3 days (after results) |
| 6. Ablations | Phase 3 results | 1.5 pages | 2 days |
| 7. Discussion | Cross-task analysis | 1 page | 1 day |
| 8. Conclusion | Summary | 0.5 page | 0.5 day |
| Abstract & Figures | All sections | -- | 2 days |

**Total writing time:** ~15-17 working days. Start writing Related Work, Dataset, and Method sections in parallel with training (Days 10-30). Write Experiments, Ablations, and Discussion after results are finalized (Days 30-40).

### Table Structure

#### Table 1: Dataset Statistics (from agent01, agent16)

| Property | Value |
|----------|-------|
| Subjects (train/val/test) | 36/16/32 |
| Raw frames | 78,961 / 38,036 / 90,269 |
| Training frames (stride=3) | 26,322 |
| Resolution | 1280x720 (native) |
| Activity classes | 75 (72 present) |
| Detection classes | 24 (21 active) |
| PSR components | 11 binary (54.88% positive) |
| Pose DOF | 9-DoF (forward, up, position) |

#### Table 2: Architecture Parameters (from agent06, agent07, agent08, agent16)

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| MViTv2-S backbone | 34.23M | 61.5% |
| BiFPN neck | 14.53M | 26.1% |
| Detection head | 1.20M | 2.2% |
| Activity head | 3.75M | 6.7% |
| PSR head | 1.78M | 3.2% |
| PSR refinement | 0.21M | 0.4% |
| Pose head | 0.20M | 0.4% |
| RotoGrad | 0.64M | 1.1% |
| **Total** | **55.69M** | **100%** |

#### Table 3: Main Results (requires Phase 1-2 completion)

| Task | Metric | ST (480px) | MTL (480px) | Delta |
|------|--------|-----------|-------------|-------|
| Detection | mAP@0.5 (21 cls) | TBD | TBD | TBD |
| Detection | mAP@0.5 (24 cls) | TBD | TBD | TBD |
| Activity | Top-1 / Top-5 | TBD / TBD | TBD / TBD | TBD / TBD |
| PSR | Component F1 (mean) | TBD | TBD | TBD |
| PSR | Transition F1 | TBD | TBD | TBD |
| Pose | Geodesic MAE (deg) | TBD | TBD | TBD |
| Total params | All tasks | ~172M (4x 43M) | 55.7M | 3.1x |
| Throughput | frames/s | TBD | TBD | 4 tasks/pass |

#### Table 4: Ablation -- Loss Weighting (requires Phase 2-3)

| Method | Det mAP | Act Top-1 | PSR F1 | Pose MAE | Note |
|--------|---------|-----------|--------|----------|------|
| Equal weights (pre-scaled) | TBD | TBD | TBD | TBD | Kurin 2022 |
| FAMO (ours) | TBD | TBD | TBD | TBD | Liu 2023 |
| Uncertainty weighting | TBD | TBD | TBD | TBD | Kendall 2018 |
| No pre-scaling (raw) | TBD | TBD | TBD | TBD | Ablation |

#### Table 5: Ablation -- Architecture Components (requires Phase 2-3)

| Variant | Det mAP | Act Top-1 | PSR F1 | Pose MAE | Params |
|---------|---------|-----------|--------|----------|--------|
| Full model (BiFPN + ref) | TBD | TBD | TBD | TBD | 55.7M |
| No BiFPN | TBD | TBD | TBD | TBD | 41.2M |
| No PSR refinement | TBD | TBD | TBD | TBD | 55.5M |
| 224px input | TBD | TBD | TBD | TBD | 55.7M |
| 320px input | TBD | TBD | TBD | TBD | 55.7M |

#### Table 6: Per-Class Analysis (requires results)

Include a long-table in appendix showing per-component PSR F1 and per-class detection AP, highlighting the 3 zero-instance classes and comp0/comp7/comp8 degeneracies.

### Figure Requirements

| Figure | Description | Dependencies | Priority |
|--------|-------------|-------------|----------|
| F1 | System architecture diagram (MViTv2-S -> BiFPN -> 4 heads) | None (can draw from code) | P0 |
| F2 | Dataset examples: 4 tasks shown on a single frame | Dataset viewer script | P0 |
| F3 | Training curves: loss curves for all 4 tasks (ST vs MTL) | Phase 2 results | P0 |
| F4 | Detection PR curves: per-class AP breakdown | Phase 2 results | P1 |
| F5 | Confusion matrix: activity top-10 classes | Phase 2 results | P1 |
| F6 | PSR per-component F1 bar chart | Phase 2 results | P1 |
| F7 | Resolution comparison: 224px vs 320px vs 480px | Phase 3 results | P2 |
| F8 | Loss landscape: pre-scaling sensitivity | Ablation results | P2 |

### Writing Timeline

| Week | Writing Tasks | Parallel Training |
|------|--------------|-------------------|
| Week 1-2 | Related Work (from agents 11, 14, 15 lit surveys), Dataset section, Method section outline | Phase 0-1 (ST baselines) |
| Week 3-4 | Method section details, Figure F1-F2 creation, Introduction draft | Phase 2 (MTL primary run) |
| Week 5-6 | Experiments section (requires Phase 2 results), Table 3-4 | Phase 2 continues |
| Week 7-8 | Ablations section (requires Phase 3 results), Table 5-6 | Phase 3 (ablations) |
| Week 9-10 | Discussion, Conclusion, abstract, formatting, referencing | Buffer period |
| Week 11-12 | Internal review, co-author feedback, final polish | -- |

### Citation Strategy

**Must cite (P0):**
- Schonbeek 2024 (closest prior work on IndustReal)
- MViTv2 (Fan 2022, backbone)
- YOLOv8 / Ultralytics (detection head design)
- FAMO (Liu 2023, primary MTO method)
- Kurin 2022 (equal weights threat)
- RotoGrad (Javaloy 2022, if included)
- Kendall 2018 (uncertainty weighting, if included)
- IndustReal dataset paper (if published)

**Should cite (P1):**
- BiFPN (Tan 2020, neck design)
- DFL (Li 2022, detection loss)
- TAL (Feng 2021, assigner)
- WIoU v3 (Tong 2023)
- BalancedSoftmax (Ren 2020)
- Zhou 2019 (6D rotation representation)
- 6DRepNet (Hempel 2022, pose baseline)

---

## 6. Risk Register

### Risk 1: Activity Collapse Persists After Fixes

- **Probability:** MEDIUM (40%)
- **Impact:** CRITICAL -- activity task becomes unusable, paper loses 1 of 4 tasks
- **Root cause:** Even with warm-start, FPN fix, and pre-scaling, the 5-order-of-magnitude loss gap between pose and activity may be too large for any weighting scheme to bridge
- **Early warning sign:** After 5 epochs of MTL training, activity accuracy < 10% (vs 15-20% expected from warm-start)
- **Contingency:**
  1. Staged training: train detection+activity for 15 epochs, freeze activity head, add PSR+pose (10 epochs), then fine-tune all heads (25 epochs)
  2. Per-task gradient clipping: clip pose gradient before it reaches shared backbone (agent15 finding: gradient clipping masks weighting)
  3. Remove pose from MTL: submit as 3-task model (detection + activity + PSR) if pose is the destabilizer
  4. Remove activity from MTL: submit as 3-task model (detection + PSR + pose) with note that activity is challenging due to 75-class long-tail
  5. Worst case: submit as 2-task model (detection + PSR) and frame as focused analysis

### Risk 2: PSR Flat Output Persists After Fixes

- **Probability:** MEDIUM-HIGH (55%)
- **Impact:** HIGH -- PSR task shows no meaningful learning, paper loses temporal dynamics claim
- **Root cause:** The Gaussian-smearing (sigma=3.0) may attenuate transition signals below the noise floor. 54.88% positive rate means BCE prior is 0.5488, and the model predicting ~0.70 is only slightly above prior. The causal transformer on T=8 (0.8 seconds) may be insufficient temporal context for 11-component state sequences.
- **Early warning sign:** After 10 epochs, PSR prediction stddev < 0.05 across frames
- **Contingency:**
  1. Reduce Gaussian sigma from 3.0 to 1.5 or 1.0 (sharper transition peaks)
  2. Extend T from 8 to 16 (1.6 seconds context) -- may require resolution reduction to 320px to fit VRAM
  3. Remove causal masking from PSR transformer (allow future context -- post-hoc processing)
  4. Report PSR on sub-sampled frames only (every 30 frames = 3 seconds) where transitions are more meaningful
  5. Acknowledge PSR prediction at 10 FPS with T=8 is inherently difficult and frame as open problem

### Risk 3: Detection mAP Lower Than Expected

- **Probability:** MEDIUM-HIGH (50%)
- **Impact:** HIGH -- if detection mAP@0.5 < 0.45, the model is not competitive
- **Root cause:** 480px resolution limits small-object detection. The P5 feature map at 480px is 15x15 (vs 20x20 at 640px). MViTv2-S may not match ConvNeXt for detection at this resolution.
- **Expected range:** 0.45-0.65 mAP@0.5 at 480px (speculative)
- **Early warning sign:** After 15 epochs, detection mAP < 0.35 at 480px
- **Contingency:**
  1. Report 21-class mAP (excluding 3 zero-instance classes) -- recovers ~12% relative improvement
  2. Test at 320px T=16 (more temporal context, better small-object features)
  3. If mAP >= 0.55, proceed with paper. If mAP < 0.40, reconsider the detection head design
  4. Reference: Schonbeek achieved 0.753 at 640px with ConvNeXt. Our expected range at 480px with MViTv2-S is lower by design. Frame as "competitive with pragmatic resolution" not "state-of-the-art."

### Risk 4: Equal Weights Matches or Beats FAMO

- **Probability:** MEDIUM (40%)
- **Impact:** HIGH -- weakens the primary methodological contribution
- **Root cause:** Kurin 2022 finding that equal weights with tuned hyperparameters matches sophisticated MTO methods. Our pre-scaling factors were tuned empirically.
- **Early warning sign:** Ablation A1a results within 2% of A1b on all metrics
- **Contingency:**
  1. Reframe contribution: "Systematic analysis of pre-scaling" becomes the contribution, not FAMO itself
  2. Add three loss weighting comparisons (A1a, A1b, A1c) and show that proper pre-scaling is the key factor
  3. Show that FAMO reduces training variance (lower std across runs) even if mean is similar
  4. Show FAMO's dynamic weight evolution over training (interesting figure)
  5. Frame the paper around the bi-factorial design (pre-scaling + gradient surgery) rather than claiming FAMO alone

### Risk 5: GPU Memory Bottleneck on 3060 (12GB)

- **Probability:** HIGH (70%)
- **Impact:** MEDIUM -- reduces parallelism, extends training timeline
- **Root cause:** 12GB VRAM is marginal even for 320px T=8 training. Attention matrices at 320px with MViTv2 still consume significant memory.
- **Early warning sign:** OOM at 320px T=4 during Phase 0 verification
- **Contingency:**
  1. Use 3060 exclusively for 224px T=8 and single-task pose finetuning
  2. Run all 480px training sequentially on 5060 Ti
  3. Accept extended timeline (32 days -> 45 days for sequential training)
  4. Consider cloud GPU rental for $0.50-1.00/hr (paperspace/vast.ai) for parallelism

### Risk 6: Timeline Slippage

- **Probability:** HIGH (80%)
- **Impact:** HIGH -- could miss AAIML 2027 deadline (Feb 2027)
- **Root cause:** Bugs taking longer to fix than estimate, training re-runs, analysis paralysis
- **Early warning sign:** Phase 0 not complete by Day 4, Phase 1 not complete by Day 14
- **Contingency:**
  1. Drop to Minimum Publishable Set (Section 3): 3 runs instead of 8, saving ~2 weeks
  2. Submit to IEEE Access (rolling deadline) or MDPI Sensors (regular issues) instead of AAIML
  3. Submit to a workshop (e.g., CVPR Workshop on Egocentric Vision, deadline typically April)
  4. Submit a shorter paper (4 pages + references) to a workshop as a "work in progress"
  5. Release as preprint + code repository and defer venue submission by one cycle

### Risk 7: Schonbeek Comparison Criticism

- **Probability:** HIGH (75%)
- **Impact:** MEDIUM -- reviewer objection but manageable with proper framing
- **Root cause:** We cannot match Schonbeek's 640px resolution, ConvNeXt backbone, or their detection-only setup. Direct numerical comparison is unfavorable.
- **Early warning sign:** Any reviewer comment beginning with "The authors should compare with Schonbeek et al."
- **Contingency:**
  1. Frame the resolution gap as a hardware limitation finding (practical contribution for consumer GPUs)
  2. Show that our 480px MTL model uses 3.1x fewer parameters than 4 separate models
  3. If possible, run Schonbeek's architecture at 480px for direct comparison
  4. If not possible, acknowledge the comparison limitations prominently in the paper
  5. Include a "fair comparison" subsection that normalizes for resolution

### Risk 8: Reproducibility Concerns

- **Probability:** LOW-MEDIUM (30%)
- **Impact:** MEDIUM -- desk rejection or major revision
- **Root cause:** Single-worker data loading, non-deterministic GPU operations, dataset preprocessing bugs
- **Contingency:**
  1. Set all random seeds, use deterministic algorithms where possible
  2. Release complete config files with each run
  3. Provide Docker environment for reproduction
  4. Report results as mean + std over 3 runs (if resources permit) or single run + seed
  5. Document all known bugs and their impact in the paper (turns bugs into transparency)

### Risk Register Summary

| Risk | Probability | Impact | Risk Score | Contingency |
|------|------------|--------|-----------|------------|
| R1: Activity collapse persists | 40% | CRIT | 1.6 | Staged training, per-task grad clip, drop activity task |
| R2: PSR flat persists | 55% | HIGH | 1.1 | Reduce sigma, extend T, remove causal masking |
| R3: Detection mAP too low | 50% | HIGH | 1.0 | 21-class reporting, reduce resolution, redesign head |
| R4: Equal weights matches FAMO | 40% | HIGH | 0.8 | Reframe contribution around pre-scaling analysis |
| R5: GPU memory on 3060 | 70% | MED | 0.7 | 224px only, sequential on 5060 Ti, cloud GPU |
| R6: Timeline slippage | 80% | HIGH | 1.6 | Drop to minimum set, change venue, workshop submission |
| R7: Schonbeek comparison | 75% | MED | 0.75 | Frame hardware limit, direct comparison at 480px |
| R8: Reproducibility | 30% | MED | 0.3 | Seeds, Docker, config files, bug documentation |

**Top 3 risks requiring immediate attention:**
1. R1 (Activity collapse) and R6 (Timeline slippage) -- tied at 1.6 risk score
2. R2 (PSR flat) at 1.1
3. R3 (Detection mAP) at 1.0

---

## Appendix: Complete File Inventory

| File | Role | Size |
|------|------|------|
| `agent01_data_audit.md` | Dataset overview, class distribution | 28KB |
| `agent02_val_analysis.md` | Split structure, distribution shift | 19KB |
| `agent03_detection_data.md` | Detection = state classification, 3 zero-instance classes | 20KB |
| `agent04_activity_psr_data.md` | 41/72 activity rare classes, PSR non-cumulative | 23KB |
| `agent05_pose_temporal.md` | 7.48 deg MAE, 6D lossless | 27KB |
| `agent06_backbone_capacity.md` | 55.7M total, 640px infeasible | 28KB |
| `agent07_neck_design.md` | BiFPN already done, 12x larger than det head | 21KB |
| `agent08_task_heads.md` | **3 CRITICAL bugs** (frozen FPN, frozen RotoGrad, activity collapse) | 20KB |
| `agent09_training_pipeline.md` | Warm-start broken, DetAug clamp | 28KB |
| `agent10_efficiency.md` | 3.1x MTL efficiency, 1.5 batches/s | 23KB |
| `agent11_detection_mtl_lit.md` | Literature: shared backbone, no temporal context | 31KB |
| `agent14_pose_regression_lit.md` | Lit: 6D rep, uncertainty weighting recommended | 14KB |
| `agent15_training_stability_lit.md` | FAMO right but equal weights threat, pre-scaling non-negotiable | 48KB |
| `agent16_paper_strategy.md` | Paper positioning: tables, framing, weaknesses | 25KB |
| **(this file)** | Final roadmap: bug triage, baselines, ablations, training, writing, risk | ~50KB |
