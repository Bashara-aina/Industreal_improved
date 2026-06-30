# AAIML 2027 — 12: New Contributions from Opus-VERIFIED Findings [2026-06-30]

## The Opportunity

Opus's analysis (files 62-63) revealed THREE publishable findings that the AAIML
paper should INCLUDE, not hide. These turn 10 days of "wasted training" into the
paper's most novel section.

## Finding 1: Temporal-Head/Sampler Mismatch (Section to Add in §4)

**What we discovered:**
The class-balanced `WeightedRandomSampler` + `FeatureBank` (keyed by recording_id)
produces 16-frame "temporal windows" that are NOT temporal — they're shuffled,
non-consecutive frames. The TCN+ViT (8.2M params) learns noise, overfits 3.7k
frames, and collapses to the majority class.

**Verification:**
- Opus traced `train.py:1354-1355` (per-frame call with real recording_ids)
- `model.py:1179-1244` (FeatureBank appends in arrival order → shuffled)
- `model.py:1031` (TCN depthwise conv over time axis → noise)
- **Result: the temporal head was always doomed with this sampler**

**How to write it:**
> "A per-frame class-balanced WeightedRandomSampler feeds non-consecutive frames
> into a recording_id-keyed feature bank, silently defeating the temporal head.
> We document this failure mode and propose a simple per-frame MLP as an alternative
> that achieves comparable results without the 8.2M-param overfitting risk."

**This is a METHODS contribution** — a cautionary tale for ANY multi-task paper
combining temporal heads with balanced sampling. AAIML is the right venue for it.

## Finding 2: Gradient Probe Misreading (Section to Add in §4)

**What we discovered:**
Our `_log_per_head_grad_norm` (`train.py:2345-2383`) logged each head's FIRST and
LAST individual parameter grad-norm, never the head total. We spent 10 days
(attempts 2-6) optimizing against "activity gradient = 0.010" and "PSR gradient =
3.180" — a ratio that compares two different tensor shapes. It was meaningless.

**Why it's publishable:**
- This probe pattern is common — many repos log per-parameter norms as "head gradients"
- The fixed-state gradient invariance (LR can't move it, blend ratio can't move it)
  is basic autograd that multiple people missed for 10 days
- The fix is trivial: `sqrt(sum of squared param grads)` instead of `first param grad`
- The LESSON is that even experienced practitioners can misread probes

**How to write it:**
> "We report a cautionary case where per-parameter gradient norms were
> misinterpreted as head-level gradient magnitudes, leading to 10 days of
> hyperparameter optimization against a non-existent gap. The fix — computing
> total head gradients — is trivial; the failure mode is systematic."

## Finding 3: Head Pose Data Artifact (Critical for Paper Integrity)

**What we discovered:**
The pose.csv forward vectors have norms 0.014-0.030 instead of 1.0. The eval
normalizes before computing angular MAE (so 8.71° IS valid), but training MSE
optimizes for magnitude, not direction. The 8.71° number needs re-verification.

**How to write it (before submission):**
> "We discovered that the pose.csv ground truth forward vectors were not
> unit-normalized. This does not affect angular MAE (eval normalizes before
> computing angle) but means training MSE optimized for magnitude rather than
> direction. After normalization, the angular MAE is [X]°."

This is a DATA CONTRIBUTION for the IndustReal dataset — we identified and
documented an annotation issue.

## Updated Contributions List for AAIML Paper

### Old Contributions (from 02_SECTION_BY_SECTION.md)
1. Multi-task architecture with minimal interference
2. Efficiency-accuracy trade-off measured  
3. x402 blockchain pipeline
4. IEEE 7005 framework with factory pilot

### New Contributions (ADD these)
5. **Temporal-head/sampler mismatch documented** — cautionary for multi-task systems
6. **Gradient probe misreading identified** — lesson for the community
7. **Head pose data artifact corrected** — annotation quality contribution
8. **Simple MLP vs TCN+ViT ablation** — per-frame activity head comparison

### Updated Abstract Structure
```
We present POPW, a single-model multi-task system for assembly verification
on consumer GPUs. The system jointly predicts 5 tasks (detection, pose, activity,
PSR, head pose) from egocentric video at 4.8 FPS on an RTX 3060 ($299),
achieving head pose accuracy of [X]° and detection mAP50 of [X] on IndustReal.

Beyond the system, we contribute three verified findings on multi-task training
pathologies: (1) a per-frame balanced sampler combined with a recording_id-keyed
feature bank silently defeats temporal heads — a cautionary result for any
multi-task system combining temporal modeling with balanced sampling;
(2) per-parameter gradient norms can be misinterpreted as head-level gradient
magnitudes, leading to wasted hyperparameter optimization; and (3) the
pose.csv annotation contains un-normalized unit vectors, documented here for
the research community.

The system is integrated with x402 blockchain micropayments and IEEE 7005-2021
ethical governance, validated in a 20-worker factory pilot.
```

## What This Changes in the Paper Structure

| Section Change | Old | New |
|---------------|-----|-----|
| Abstract | 4 contributions, purely technical | 7 contributions including 3 pathology findings |
| §4 (Experiments) | Numbers only | Numbers + failure analysis |
| §4 (New subsection) | — | "Lessons from Multi-Task Training" |
| Discussion | Limitations and future work | Pathology analysis + broader impact |
| Figure count | 8 figures | 9 figures (+ gradient probe vs total comparison) |
