# 179 — Experimental Design to Prove MTL Helps (or Hurts)

**Purpose:** Rigorous experimental blueprint to determine whether MTL helps, hurts, or is neutral for each task.
**Guiding principle:** Every claim must be backed by a controlled experiment with a clearly defined counterfactual.
**Target audience:** Opus (for path decision) and AAIML reviewers (for paper claims).

---

## Table of Contents

1. [Experimental Philosophy — Defining "MTL Helps"](#1-experimental-philosophy)
2. [Experiment E1: Single-Task Detection MViTv2-S](#2-experiment-e1-single-task-detection)
3. [Experiment E2: Single-Task Activity MViTv2-S](#3-experiment-e2-single-task-activity)
4. [Experiment E3: Single-Task PSR MViTv2-S](#4-experiment-e3-single-task-psr)
5. [Experiment E4: Single-Task Pose MViTv2-S](#5-experiment-e4-single-task-pose)
6. [Experiment E5: Fixed-Weight MTL (Path C)](#6-experiment-e5-fixed-weight-mtl)
7. [Experiment E6: Per-Task Log_Var Cap Sweep (Path A)](#7-experiment-e6-log-var-cap-sweep)
8. [Experiment E7: Remove PSR Detach](#8-experiment-e7-remove-psr-detach)
9. [Experiment E8: Gradient Flow Analysis](#9-experiment-e8-gradient-flow-analysis)
10. [Experiment E9: Data Augmentation for MTL](#10-experiment-e9-data-augmentation)
11. [Experiment E10: Task Dropout Analysis](#11-experiment-e10-task-dropout)
12. [Computational Budget & Scheduling](#12-computational-budget)
13. [Expected Outcomes & Decision Tree](#13-expected-outcomes)
14. [Appendix: Implementation Commands](#14-appendix-implementation-commands)

---

## 1. Experimental Philosophy

### 1.1 What "MTL Helps" Means (Four Definitions)

We propose four definitions, ordered from strongest to weakest:

| Level | Definition | Bar | Paper Claim |
|-------|-----------|-----|-------------|
| **L1** | MTL increases per-task accuracy vs single-task for ALL tasks | High bar, rarely met | "MTL beats single-task" |
| **L2** | MTL improves at least ONE task vs single-task, others within 90% | Realistic bar | "MTL provides positive transfer" |
| **L3** | MTL matches single-task accuracy with lower total compute/params | Practical bar | "MTL is more efficient" |
| **L4** | MTL enables a use case impossible with single-task (real-time multi-task inference) | Weakest bar | "MTL enables multi-task inference" |

**Current position:** We can claim L4 today (one model does 4 things). We need E1-E4 to know if we can claim L1-L3.

### 1.2 Controlled Variables

Every experiment must keep these constant:
- **Backbone:** MViTv2-S (Kinetics-400 pretrained)
- **Data split:** Same train/val split (36 train recordings, 16 val recordings)
- **Input:** T=16, 224×224, same normalization
- **Batch size:** 2 (effective 4 with grad_accum=2)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=0.05)
- **Epochs:** Minimum 20 (for meaningful comparison)
- **Evaluation:** Same eval code, same metrics

### 1.3 What We're Measuring

| Metric | What It Captures | Single-Task Version |
|--------|-----------------|---------------------|
| Loss | Optimization progress | Same loss function |
| Top-1 accuracy | Activity classification | Same |
| mAP@0.5 | Detection localization + class | Same |
| PSR event F1 | Procedure step temporal accuracy | Same |
| Pose fwd/up MAE | Head pose angular error | Same |
| Params | Model size | Sum of 4 single-task |
| GFLOPs | Compute per forward pass | Sum of 4 single-task |
| Training time | Wall-clock to convergence | Max of 4 (sequential) or sum (parallel) |

### 1.4 Statistical Rigor

- Each experiment trains from scratch (not continued from checkpoint)
- Evaluation at fixed intervals (every 5 epochs)
- Per-seed variance: 3 seeds minimum for critical experiments (E1, E2, E5, E6)
- Bootstrapped confidence intervals on all metrics (1000 resamples)

---

## 2. Experiment E1: Single-Task Detection MViTv2-S

### 2.1 Purpose

Establish the **upper bound** for detection with MViTv2-S backbone. This answers:
- Can MViTv2-S achieve competitive detection mAP on IndustReal?
- How much does MTL cost detection (MTL mAP vs single-task mAP)?

### 2.2 Architecture

```
MViTv2-S backbone → FPN (same as current) → DetectionHead (cls+reg)
```

**Changes from MTL:**
- Remove ActivityHead, PSRHead, PoseHead
- Remove Kendall weighting (single loss: det_loss)
- Remove PCGrad (single task, no gradient conflict)
- Remove DETACH_PSR_FPN (not applicable)

### 2.3 Hyperparameters

| Param | Single-Task | MTL (current) | Notes |
|-------|-------------|---------------|-------|
| LR | 1e-4 | 1e-4 | Same |
| Weight decay | 0.05 | 0.05 | Same |
| LR schedule | cosine decay | none | Single-task can use cosine |
| Epochs | 100 | 100 | Same |
| Batch size | 2 (eff. 4) | 2 (eff. 4) | Same |
| Warmup | 250 steps | 250 steps | Same |
| FPN channels | 256 | 256 | Same |
| Loss | Focal + CIoU + DFL | Same + other task losses | Lighter |

### 2.4 Expected Outcome

| Metric | Single-Task (Ep20) | MTL (Ep20, current) | Expected Gap |
|--------|-------------------|---------------------|--------------|
| mAP@0.5 | 0.40-0.55 | 0.20-0.40 (est) | Single-task leads by 0.15-0.20 |
| Det loss | 0.10-0.20 | 0.15-0.25 | Single-task lower |
| Box precision | Higher | Lower | Single-task better localization |

**Hypothesis:** Single-task detection will achieve ~1.5-2× higher mAP than MTL at the same epoch. This quantifies the "MTL cost" for detection.

### 2.5 Compute

- **Training:** ~3.1 days (100 epochs at ~44 min/epoch)
- **Minimal eval:** ~2.8 hours
- **Total:** ~3.2 days

---

## 3. Experiment E2: Single-Task Activity MViTv2-S

### 3.1 Purpose

Establish the **upper bound** for activity recognition with MViTv2-S backbone. This answers:
- Is the activity head (LayerNorm→Linear) sufficient when it gets full backbone gradient?
- Does MTL completely explain the 0.008 top-1, or is the head architecture also a bottleneck?

### 3.2 Architecture

```
MViTv2-S backbone → ActivityHead (LayerNorm + Linear)
```

**Changes from MTL:**
- Remove detection, PSR, pose heads
- Remove Kendall, PCGrad
- Remove class weights (single-task may not need them)
- Add temporal pooling: AvgPool over T=16 → 1 feature vector per clip

### 3.3 Key Design Decision: Per-Frame vs Clip-Level

The current activity head operates per-frame (B, 768 → B, 75). A single-task MViTv2-S can do clip-level classification (pool over T=16 frames) which is the standard Kinetics protocol. We should test both:

**Variant E2a: Per-frame** (same as MTL)
- Same head architecture
- Per-frame CE loss
- Evaluated as both per-frame and clip-level (majority vote)

**Variant E2b: Clip-level** (standard for MViTv2)
- Temporal pooling (mean or max over T=16)
- Clip-level CE loss
- Clip-level evaluation

### 3.4 Expected Outcome (E2a — Per-Frame)

| Metric | Single-Task (Ep20) | MTL (Ep20, current) | Expected Gap |
|--------|-------------------|---------------------|--------------|
| Per-frame top-1 | 0.15-0.25 | 0.01-0.02 | Single-task 10-25× higher |
| Activity loss | 3-5 | 8-12 | Single-task much lower |
| Top class predicted | Varies with features | Class 11 (bias) | Single-task learns discrimination |

**Key question:** If single-task per-frame top-1 reaches 15-25%, MTL's Kendall weight (0.04) is the primary cause of activity's poor performance. If single-task per-frame top-1 is still <5%, the head architecture is also a bottleneck.

### 3.5 Expected Outcome (E2b — Clip-Level)

| Metric | Single-Task (Ep20) | WACV SOTA | Expected Gap |
|--------|-------------------|-----------|--------------|
| Clip-level top-1 | 0.35-0.50 | 0.6525 | 15-30% gap |
| Clip-level top-5 | 0.60-0.75 | 0.8793 | 15-25% gap |

**Key question:** If clip-level single-task reaches 35-50%, the MViTv2-S backbone is adequate for activity and the problem is purely MTL optimization. If clip-level single-task is <20%, the backbone/head combo is fundamentally insufficient.

### 3.6 Compute

- **Training:** ~3.1 days (100 epochs at ~44 min/epoch)
- **Both variants:** ~6.2 days (could run in parallel on 2 GPUs)
- **E2a only:** ~3.1 days (recommended first — answers the critical question)

---

## 4. Experiment E3: Single-Task PSR MViTv2-S

### 4.1 Purpose

Establish the **upper bound** for PSR with MViTv2-S and a direct training path. This answers:
- Is PSR flat because of DETACH_PSR_FPN or because PSR is fundamentally hard with these features?
- What is the "MTL cost" for PSR?

### 4.2 Architecture

```
MViTv2-S backbone → PSRHead (AdaptiveAvgPool3d → causal Transformer)
```

**Changes from MTL:**
- Remove det, act, pose heads
- Remove DETACH_PSR_FPN (gradient flows freely)
- Remove Kendall, PCGrad
- PSR sequence mode only (no per-frame mode — 100% of batches are temporal)

### 4.3 Key Change: No Detach

This is the most important variable. In MTL, `DETACH_PSR_FPN=True` prevents PSR gradient from ever reaching the backbone. Single-task removes this detach by definition.

### 4.4 Expected Outcome

| Metric | Single-Task (Ep20) | MTL (Ep20, current) | Expected Gap |
|--------|-------------------|---------------------|--------------|
| PSR loss | 0.6-0.9 | 1.28-1.30 | Single-task significantly lower |
| Per-comp F1 | 0.4-0.6 | 0.0 (flat) | Single-task shows real learning |
| Event F1 | 0.3-0.5 | 0.0 | Single-task starts learning |

**Key question:** If single-task PSR shows meaningful learning (loss drops below 1.0, F1 > 0.3), then MTL's DETACH_PSR_FPN + Kendall weight is killing PSR. If single-task PSR is also flat, the PSR head architecture is the root cause.

### 4.5 Compute

- **Training:** ~3.1 days (100 epochs)
- **Note:** 100% sequence mode trains slower than mixed mode

---

## 5. Experiment E4: Single-Task Pose MViTv2-S

### 5.1 Purpose

Establish the **upper bound** for head pose. This answers:
- Does pose benefit from MTL features (detection bounding boxes → head position)?
- How close to the "true best" is our current pose MAE?

### 5.2 Architecture

```
MViTv2-S backbone → PoseHead (MLP → Tanh → 6D)
```

**Changes from MTL:**
- Remove det, act, psr heads
- Remove Kendall, PCGrad
- Remove hp_prec_cap (not needed, single task)

### 5.3 Expected Outcome

| Metric | Single-Task (Ep20) | MTL (Ep20, current) | Expected Gap |
|--------|-------------------|---------------------|--------------|
| Pose loss | 0.10-0.15 | 0.15-0.19 | Single-task slightly lower |
| Fwd MAE | 3-5° | 8-10° | Single-task better |
| Up MAE | 8-12° | 15-18° | Single-task better |

**Hypothesis:** Pose is the least controversial head — it benefits most from shared backbone features because pose regression requires spatial understanding that detection features naturally provide. Single-task may actually be WORSE than MTL because it loses detection's spatial guidance.

### 5.4 Compute

- **Training:** ~3.1 days (100 epochs)
- **Likely quickest to converge:** may finish acceptably in 20-30 epochs

---

## 6. Experiment E5: Fixed-Weight MTL (Path C)

### 6.1 Purpose

Remove Kendall weighting entirely and use manually-set fixed weights. This answers:
- Is the Kendall mechanism specifically causing the starvation?
- Does equal-weight MTL outperform Kendall MTL?

### 6.2 Configuration

```python
# Fixed weights (not learned)
FIXED_TASK_WEIGHTS = {
    'detection': 1.0,   # Baseline
    'activity': 2.0,    # Higher because CE loss is structurally larger
    'psr': 1.0,         # Same as detection
    'pose': 0.5,        # Lower because it converges faster
}

# Remove Kendall
use_kendall = False
use_pcgrad = True  # Keep PCGrad for conflict resolution
```

### 6.3 Weight Justification

The weights are chosen so that each task contributes approximately equally to the total gradient:

| Task | Loss (Ep6) | Weight | Weighted Contribution | % of Total |
|------|-----------|--------|----------------------|------------|
| Det | 0.31 | 1.0 | 0.31 | 7% |
| Act | 12.31 | 2.0 | 24.62 (too high!) | 83% |
| PSR | 1.30 | 1.0 | 1.30 | 4% |
| Pose | 0.19 | 0.5 | 0.10 | 0.3% |

**Problem:** With fixed weights = [1, 2, 1, 0.5], activity still dominates (83% of total gradient). The 40× loss scale mismatch means activity must have a much lower weight to equalize contributions.

**Corrected weights:**

| Task | Loss (Ep6) | Weight | Weighted Contribution | % of Total |
|------|-----------|--------|----------------------|------------|
| Det | 0.31 | 1.0 | 0.31 | 30% |
| Act | 12.31 | **0.025** | 0.31 | 30% |
| PSR | 1.30 | 0.24 | 0.31 | 30% |
| Pose | 0.19 | 1.63 | 0.31 | 10% (pose converges fast, needs less) |

**Therefore, Path C fixed weights = [1.0, 0.025, 0.24, 1.63].** This equalizes per-task gradient contributions.

### 6.4 Variants

| Variant | Weights | PCGrad | Expected Signal |
|---------|---------|--------|-----------------|
| E5a | Equal gradient [1.0, 0.025, 0.24, 1.63] | ON | Balanced learning |
| E5b | Equal gradient [1.0, 0.025, 0.24, 1.63] | OFF | Balanced + no projection |
| E5c | Naive equal [1, 1, 1, 1] | ON | Activity dominates (baseline for comparison) |

### 6.5 Expected Outcome

| Variant | Act top-1 (Ep10) | Det mAP (Ep10) | PSR F1 (Ep10) | Pose MAE (Ep10) |
|---------|------------------|----------------|----------------|-----------------|
| E5a (balanced + PCGrad) | 0.05-0.08 | 0.03-0.08 | 0.02-0.05 | 8-10° |
| E5b (balanced, no PCGrad) | 0.08-0.12 | 0.02-0.05 | 0.03-0.08 | 7-9° |
| E5c (equal weights) | 0.01-0.02 | 0.0 | 0.0 | 8-10° |

**Key prediction:** E5b (balanced + no PCGrad) will show the fastest early learning but may have higher negative transfer at later epochs. E5a is the safer bet.

### 6.6 Compute

- **Each variant:** ~3.1 days (100 epochs)
- **All 3 variants:** ~9.3 days (could run in parallel on 3 GPUs)
- **Minimal (E5a only):** ~3.1 days

---

## 7. Experiment E6: Per-Task Log_Var Cap Sweep (Path A)

### 7.1 Purpose

Find the optimal per-task log_var caps that prevent starvation while preserving Kendall's adaptive balancing. This answers:
- What is the right set of caps for each task?
- How sensitive is MTL performance to cap values?

### 7.2 Sweep Design

| Task | Caps to Test | Current Value | Rationale |
|------|-------------|---------------|-----------|
| Detection | [-4, 4] (no change) | -0.41 | Already stable |
| Activity | [0.5, 1.0, 1.5, 2.0, 4] | 3.20 | Find cap that balances learning vs spiral |
| PSR | [0.0, 0.5, 1.0, 1.5] | 0.94 | Find cap that prevents PSR regression |
| Pose | [-4, 4] (no change) | -0.49 | Already stable |

### 7.3 Grid

| Experiment | Act cap | PSR cap | Expected Act Weight | Expected PSR Weight |
|-----------|---------|---------|-------------------|-------------------|
| E6a (baseline) | 4.0 (no cap) | 4.0 (no cap) | 0.04 | 0.39 |
| E6b | 1.0 | 0.5 | 0.37 | 0.61 |
| E6c | 1.0 | 1.0 | 0.37 | 0.37 |
| E6d | 1.5 | 0.5 | 0.22 | 0.61 |
| E6e | 2.0 | 1.0 | 0.14 | 0.37 |
| E6f | 0.5 | 0.5 | **0.61** | **0.61** (strongest protection) |
| E6g | 0.5 | 0.0 | 0.61 | 1.00 (strongest activity, moderate PSR) |

### 7.4 Expected Outcome

**Prediction:** E6f (act cap=0.5, psr cap=0.5) will give the best balance — both activity and PSR get meaningful gradient (weight=0.61 each).

| Metric | E6a (current, Ep10) | E6f (best, Ep10) | Improvement |
|--------|--------------------|-------------------|-------------|
| Act top-1 | 0.01 | 0.05-0.08 | 5-8× |
| Det mAP | 0.05-0.10 | 0.04-0.08 | Slight decrease |
| PSR loss | 1.28-1.30 | 1.10-1.20 | Noticeable drop |
| Pose MAE | 8-10° | 8-10° | No change |

**Risk:** Cap=0.5 for activity gives weight=0.61. This is 15× more gradient than current. Activity may dominate and corrupt detection features. PCGrad must protect against this.

### 7.5 Early Stopping Criteria

Run each sweep variant for 5 epochs (enough for log_vars to start diverging). If log_var_act hits the cap within 5 epochs, the cap is too high. If activity loss doesn't decrease within 5 epochs, the cap is too low (activity needs MORE gradient).

### 7.6 Compute

- **Each variant:** ~3.7 hours (5 epochs × 44 min)
- **All 7 variants:** ~25.9 hours (sequential) or ~3.7 hours (parallel on 7 GPUs)
- **Recommended:** Run E6b, E6f, E6g first (3 variants, 11 hours sequential, 3.7h parallel)

---

## 8. Experiment E7: Remove PSR Detach

### 8.1 Purpose

Isolate the effect of DETACH_PSR_FPN on PSR learning. This experiment is simpler than a full single-task PSR run.

### 8.2 Configuration

```python
# Change in config
DETACH_PSR_FPN = False  # Was True

# Everything else unchanged
# Kendall, PCGrad, all heads active
```

### 8.3 Expected Outcome

| Metric | Detach=True (Ep10) | Detach=False (Ep10) | Expected Change |
|--------|-------------------|--------------------|-----------------|
| PSR loss | 1.28-1.30 | 1.15-1.25 | 0.05-0.15 drop |
| PSR per-comp activation | Pure bias | Slight input variation | Feature utilization starts |
| Detection mAP | 0.05-0.10 | 0.03-0.08 | May decrease slightly |
| PSR gradient to backbone | 0 | Small positive | Feature shaping begins |

**Risk:** If PSR gradient conflicts with detection gradient at conv_proj (shared early features), detection mAP may drop. This quantifies the negative transfer that DETACH_PSR_FPN was designed to prevent.

### 8.4 Compute

- **Training:** ~3.1 days (100 epochs)
- **Can stop at epoch 20 for signal:** ~14.7 hours

---

## 9. Experiment E8: Gradient Flow Analysis

### 9.1 Purpose

Diagnose WHY each task isn't learning by measuring gradient properties directly. This is NOT a training run — it's a one-time diagnostic that instruments the existing training loop.

### 9.2 What to Measure

For each task, over 100 training batches:

| Measurement | Method | What It Tells Us |
|------------|--------|------------------|
| Grad L2 norm per task (pre-PCGrad) | Hook after backward | Which task produces strongest gradient signal |
| Grad L2 norm per task (post-PCGrad) | After PCGrad projection | How much PCGrad attenuates each task |
| Cosine similarity task pairs (pre-PCGrad) | Cosine between flattened grad vectors | Which tasks naturally align/conflict |
| Cosine similarity task pairs (post-PCGrad) | After projection | How much PCGrad changes relationships |
| Grad variance across batch | Multiple micro-batches | Gradient noise level per task |
| Backbone grad magnitude at each block | Hook at conv_proj, blocks[1,3,14] | Where gradients vanish in the network |

### 9.3 Implementation

Add to `train_step()` in `train_mtl_mvit.py`:

```python
# After each task's backward() but before PCGrad
for task_name, task_grads in all_task_grads.items():
    grad_norm = sum(p.grad.norm().item()**2 for p in shared_backbone_params if p.grad is not None)**0.5
    logger.log(f"grad_norm/{task_name}", grad_norm)

# PCGrad projection magnitudes
for task_name, projected_grad in projected_grads.items():
    proj_norm = projected_grad.norm().item()
    logger.log(f"grad_proj/{task_name}", proj_norm)
    logger.log(f"grad_proj_ratio/{task_name}", proj_norm / pre_norm)
```

### 9.4 Expected Outcome

| Measurement | Expected Value | What It Means |
|------------|---------------|---------------|
| Activity grad norm (pre-PCGrad) | Very high (loss is 40× larger) | Activity produces strong gradient |
| Activity grad norm (post-PCGrad) | Near zero | PCGrad projects activity away |
| Cosine(det, pose) | > 0.5 | Detection and pose naturally align |
| Cosine(det, act) | < 0.1 | Detection and activity use different features |
| Cosine(psr, det) | < 0.1 | PSR and detection conflict at conv_proj |
| Backbone grad at block 14 | Low for activity | Activity doesn't influence high-level features |

### 9.5 Compute

- **Instrumentation:** ~2 hours to add hooks and run diagnostics
- **No full training required:** Just 100-200 batches

---

## 10. Experiment E9: Data Augmentation for MTL

### 10.1 Purpose

Test whether stronger data augmentation improves MTL by making per-task gradients more aligned (less noise, more signal).

### 10.2 Augmentations to Add

| Augmentation | Affects Tasks | Rationale |
|-------------|--------------|-----------|
| Random frame masking (zero out 20% frames) | Act, PSR | Forces temporal feature utilization |
| Color jitter (brightness, contrast, saturation) | Det, Act, PSR | Standard robust training |
| Random horizontal flip | Det, Pose | Spatial augmentation |
| Mixup (mix two videos) | Act, Det | Strong augmentation, improves gradient alignment |
| CutMix (patch-level mix) | Det | Detection-specific augmentation |

### 10.3 Implementation

Add to `industreal_dataset.py`:

```python
# In __getitem__, after loading clip:
if self.augment:
    clip = color_jitter(clip)
    if random.random() < 0.5:
        clip = horizontal_flip(clip)
        # Flip bounding boxes, pose
    # MixUp
    if random.random() < 0.3 and self.use_mixup:
        clip2, labels2 = self._get_second_sample(index)
        lam = np.random.beta(2, 2)
        clip = lam * clip + (1 - lam) * clip2
        # Mix labels
```

### 10.4 Expected Outcome

| Metric | No Aug (Ep10) | With Aug (Ep10) | Expected Change |
|--------|---------------|-----------------|-----------------|
| Act top-1 | 0.01 | 0.02-0.04 | Modest improvement |
| Det mAP | 0.05-0.10 | 0.06-0.12 | Small improvement |
| PSR loss | 1.30 | 1.25-1.28 | Minimal (PSR is flat regardless) |
| Grad cosine alignment | Low | Slightly higher | Aug reduces gradient noise |

### 10.5 Compute

- **Training:** ~3.1 days (100 epochs)
- **Minimal (20 epochs):** ~14.7 hours

---

## 11. Experiment E10: Task Dropout Analysis

### 11.1 Purpose

Determine which tasks contribute positive transfer and which contribute negative transfer by systematically dropping tasks from MTL.

### 11.2 Dropout Combinations

| Run | Active Tasks | Answers |
|-----|-------------|---------|
| E10a | Det + Act | Does activity help or hurt detection? |
| E10b | Det + PSR | Does PSR help or hurt detection? |
| E10c | Det + Pose | Does pose help or hurt detection? |
| E10d | Act + PSR | Do temporal tasks help each other? |
| E10e | Det + Act + Pose | Full minus PSR |
| E10f | Det + Act + PSR | Full minus Pose |
| E10g | Full (reference) | Current state |

### 11.3 Expected Outcome

| Run | Det mAP | Act Top-1 | PSR F1 | Pose MAE | Key Insight |
|-----|---------|-----------|--------|----------|-------------|
| E10a | 0.25 | 0.15 | — | — | Activity hurts det? |
| E10c | 0.30 | — | — | 6° | Pose helps det? |
| E10e | 0.22 | 0.12 | — | 7° | PSR hurts others? |
| E10g (full) | 0.20 | 0.01 | 0.0 | 8° | Reference |

**Hypothesis:** Detection and pose will have positive transfer (spatial features help both). Activity will have negative transfer (its loss structure corrupts detection features). PSR is neutral (detached, so minimal effect on others).

### 11.4 Compute

- **Each variant:** ~3.1 days (100 epochs)
- **All 7 variants:** ~21.7 days (sequential)
- **Recommended 3 most critical:** E10a, E10c, E10e — ~9.3 days
- **Parallel on 3 GPUs:** ~3.1 days

---

## 12. Computational Budget & Scheduling

### 12.1 Available Hardware

| GPU | VRAM | Role | Priority |
|-----|------|------|----------|
| RTX 5060 Ti | 16 GB | Primary | Current MTL run |
| RTX 3060 | 12 GB | Secondary | Ablations |

### 12.2 Recommended Schedule (Path A Focus)

Running MTL training (current run) on RTX 5060 Ti is the baseline. Ablations go on RTX 3060.

**Phase 1 — Diagnostics (Days 1-2):**
| Day | GPU | Experiment | Duration |
|-----|-----|-----------|----------|
| 1 | 5060 Ti | E8: Gradient flow analysis (current run + hooks) | 2 hours |
| 1 | 3060 | E6b: Act cap=1.0, PSR cap=0.5 | 14.7 hours (20 epochs) |
| 2 | 3060 | E5a: Fixed-weight MTL | 14.7 hours (20 epochs) |

**Phase 2 — Single-Task Baselines (Days 3-10):**
| Day | GPU | Experiment | Duration |
|-----|-----|-----------|----------|
| 3 | 5060 Ti | E1: Single-task detection | 3.1 days |
| 3 | 3060 | E2a: Single-task activity (per-frame) | 3.1 days |
| 6 | 5060 Ti | E3: Single-task PSR | 3.1 days |
| 6 | 3060 | E4: Single-task pose | 1.5 days (20 epochs sufficient) |

**Phase 3 — Task Dropout (Days 10-14):**
| Day | GPU | Experiment | Duration |
|-----|-----|-----------|----------|
| 10 | 5060 Ti | E10a: Det + Act | 3.1 days |
| 10 | 3060 | E10c: Det + Pose | 3.1 days |

### 12.3 Total Compute Budget

| Phase | Duration | Wall-Clock (parallel) |
|-------|----------|----------------------|
| Phase 1 | 31.6 hours training | ~14.7 hours |
| Phase 2 | ~10.8 days training | ~3.1 days |
| Phase 3 | ~6.2 days training | ~3.1 days |
| **Total** | **~12.9 days** | **~7 days** |

### 12.4 Minimal Viable Experiments (If Compute-Constrained)

If only 7 days of wall-clock time is available:

**Highest Priority (must do):**
1. E6b (log_var cap act=1.0, psr=0.5) — quick fix test — 14.7 hours
2. E2a (single-task activity) — core hypothesis test — 3.1 days
3. E1 (single-task detection) — MTL cost for detection — 3.1 days
4. E7 (remove PSR detach) — PSR-specific fix — 14.7 hours

**Total: ~8 days wall-clock from a single GPU, ~4 days on two GPUs.**

---

## 13. Expected Outcomes & Decision Tree

### 13.1 Decision Tree

```
Start here
│
├─ E2a (single-task activity top-1 > 15% at Ep20)?
│   ├─ YES → MTL Kendall weight (0.04) is the primary cause of activity failure.
│   │        → Path A (cap log_var) or Path C (fixed weights) is correct.
│   │
│   └─ NO → Head architecture (LayerNorm→Linear) is also a bottleneck.
│           → Activity needs a better head (temporal pooling, transformer) 
│             regardless of MTL optimization.
│
├─ E1 (single-task detection mAP > 2× MTL detection mAP)?
│   ├─ YES → MTL is hurting detection. The shared backbone sacrifices
│   │        detection-specific features. Consider:
│   │        → Task-specific backbone branches
│   │        → Detection-only training with MTL fine-tuning (staged)
│   │        → Accept the MTL cost as efficiency tradeoff (L3 claim)
│   │
│   └─ NO → MTL doesn't significantly hurt detection. MTL's efficiency
│           advantage (one model for all tasks) is real with minimal accuracy cost.
│
├─ E6b (capped log_var improves activity)?
│   ├─ YES → Path A works. Continue with caps, monitor log_var trajectory.
│   │
│   └─ NO → Caps alone are insufficient. Need Path C (fixed weights).
│
├─ E5a (fixed-weight MTL improves all tasks)?
│   ├─ YES → Kendall is the problem. Replace with fixed weights.
│   │
│   └─ NO → MTL optimization is not the only problem.
│           → Architecture issues (head capacity, feature sources) must be addressed.
│
└─ E7 (remove PSR detach improves PSR)?
    ├─ YES → DETACH_PSR_FPN was the main cause of PSR collapse.
    │         Keep detach removed, monitor detection mAP for negative transfer.
    │
    └─ NO → PSR has deeper problems (feature quality, transformer capacity).
```

### 13.2 Combined Scenario Analysis

| Scenario | E1 (Det) | E2a (Act) | E6b (Caps) | E7 (PSR) | Recommended Path |
|----------|---------|----------|-----------|---------|-----------------|
| A | MTL > ST | Act > 15% | Works | Works | **Path A — keep MTL with caps** |
| B | MTL < ST | Act > 15% | Works | Works | **Path A + accept det cost** |
| C | MTL < ST | Act < 5% | Fails | Fails | **Architecture redesign needed** |
| D | MTL > ST | Act < 5% | Fails | Fails | **Drop activity task** |
| E | MTL ≈ ST | Act > 15% | Works | Works | **Path A — ideal scenario** |

### 13.3 What the Paper Can Claim Based on Results

| Experiment Outcome | Paper Claim | Contribution Level |
|-------------------|-------------|-------------------|
| Single-task > MTL for ALL tasks | "MTL provides efficiency at the cost of per-task accuracy" | Weak | 
| MTL matches single-task for 2+ tasks | "MTL achieves competitive accuracy with 60% fewer params" | Moderate |
| MTL EXCEEDS single-task for any task | "MTL provides positive transfer for [task], proving shared features benefit multi-task learning" | **Strong** |
| All heads converge in MTL (regardless of single-task) | "First multi-task MViTv2-S system for IndustReal with all 4 tasks functional" | Foundational |

---

## 14. Appendix: Implementation Commands

### 14.1 E1: Single-Task Detection

```bash
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
python scripts/train_mtl_mvit.py \
  --output-dir src/runs/rf_stages/checkpoints/mtl_ablations/single_det \
  --single-task detection \
  --epochs 100 \
  --batch-size 2 \
  --grad-accum 2
```

### 14.2 E2a: Single-Task Activity (Per-Frame)

```bash
python scripts/train_mtl_mvit.py \
  --output-dir src/runs/rf_stages/checkpoints/mtl_ablations/single_act \
  --single-task activity \
  --epochs 100 \
  --batch-size 2 \
  --grad-accum 2
```

### 14.3 E5a: Fixed-Weight MTL

```bash
python scripts/train_mtl_mvit.py \
  --output-dir src/runs/rf_stages/checkpoints/mtl_ablations/fixed_weight_v1 \
  --fixed-weights 1.0 0.025 0.24 1.63 \
  --no-kendall \
  --pcgrad \
  --epochs 100
```

### 14.4 E6b: Log_Var Caps (Act max=1.0, PSR max=0.5)

```bash
python scripts/train_mtl_mvit.py \
  --output-dir src/runs/rf_stages/checkpoints/mtl_ablations/cap_act1.0_psr0.5 \
  --log-var-caps -4 4 -4 1.0 -4 0.5 -4 4 \
  --epochs 100
```

### 14.5 E7: Remove PSR Detach

```bash
python scripts/train_mtl_mvit.py \
  --output-dir src/runs/rf_stages/checkpoints/mtl_ablations/no_psr_detach \
  --no-detach-psr \
  --epochs 100
```

### 14.6 E8: Gradient Flow Analysis (Hooks on Current Run)

```bash
# Add --grad-trace flag to enable gradient logging
python scripts/train_mtl_mvit.py \
  --resume src/runs/rf_stages/checkpoints/mtl_mvit_run/latest.pt \
  --grad-trace \
  --trace-batches 100 \
  --epochs 1
```

---

*Refer to file 177 for the current training status, file 178 for 50 deep questions this experimental design addresses, and file 180 for the Opus consultation prompt to get a path decision.*
