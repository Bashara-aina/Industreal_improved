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

# Agent 02: Validation Split Analysis (v2)

**Date**: 2026-07-13
**Status**: v2 (v1 from Jul 11 outdated)
**Context**: Currently running 480 T=8 FixRes training (started 21:33), ST-det 320 also running.
**1-line THW fix already in** `mvit_mtl_model.py:104-107`.

---

## 1. Split Structure -- Actual Counts

**SOURCE**: `/home/newadmin/swarm-bot/master/POPW/datasets/industreal/` (raw data) and `train.csv`/`val.csv`/`test.csv`.

### Recording counts

| Split | Recordings | Raw frames | Loaded frames (stride-applied) |
|-------|-----------|------------|-------------------------------|
| Train | **36**     | 78,961     | **26,322** (stride=3)         |
| Val   | **16**     | 38,036     | **38,036** (stride=1)         |
| Test  | **32**     | 90,269     | **90,269** (stride=1)         |

### CORRECTION vs v1 claims

The v1 analysis claimed "10 train recordings + 6 val recordings". This is **incorrect at the recording level**. The actual split has:
- **36 training recordings** across 12 unique participants
- **16 validation recordings** across 5 unique participants
- **32 test recordings** across 10 unique participants

The "10 + 6" confusion likely came from counting only the `{participant}_main_0_1` recordings (10 in train, 5 in val, 11 in test) which are the "main assembly" recordings. But the dataset also includes `{participant}_assy_*` recordings (the "assy" task views) and `{participant}_{task}_{variant}` recordings (different assembly variants by the same participant).

**Correct breakdown**: Each participant has up to 3 recording types: one `main_0_1` (standard assembly), one `assy_0_1` (first person assembly view), and task-specific variants (e.g., `assy_2_2`, `main_2_3`). All 36/16/32 recordings are used for training/val/test respectively.

**Confidence: HIGH** -- directly verified from filesystem.

---

## 2. Temporal Overlap: Are Train/Val Recordings Disjoint?

### Participant-level split

The splits are **entirely participant-disjoint** -- no participant appears in more than one split.

| Split | Participant IDs |
|-------|----------------|
| Train | 01, 02, 04, 06, 07, 11, 15, 16, 21, 22, 25, 27 |
| Val   | 05, 14, 20, 24, 26 |
| Test  | 03, 08, 09, 10, 12, 13, 17, 18, 19, 23 |

- Train-Val intersection: **empty**
- Train-Test intersection: **empty**
- Val-Test intersection: **empty**

### Task-type balance

| Task type | Train | Val | Test |
|-----------|-------|-----|------|
| assy      | 20    | 9   | 19   |
| main      | 16    | 7   | 12   |

Both splits have a roughly 55-45% assy-to-main ratio. No systematic imbalance.

### Are recordings from the same data collection session?

The recording naming convention `{participant}_{task}_{variant}` encodes: participant identity, task type (assy=assembly-first-person, main=overhead), and variant number. Since participants are disjoint across splits, **no two recordings across splits share any participant or session**. This eliminates temporal leakage.

**Confidence: HIGH** -- verified through participant ID extraction from all recording names.

### Caveat: Distribution shift from participant pool

The 5 val participants (05, 14, 20, 24, 26) may have systematically different assembly styles, speeds, or hand positions than the 12 train participants. This is intentional (tests generalization) but means val performance is a lower bound on in-distribution performance.

**Confidence: MEDIUM** -- plausible but unverified without participant motion analysis.

---

## 3. Val Distribution Per Task (vs Train)

### 3.1 Action Recognition (AR) -- Segment-level distribution

Train has **3,667 AR segments** across 72 unique action classes (IDs 0-74, excluding 9 absent).
Val has **1,928 AR segments** across **65 unique action classes**.

#### Classes present in train but absent in val (9 classes):

| ID | Action | Train count |
|----|--------|-------------|
| 25 | plug_partial_model | 3 |
| 28 | plug_wheel | 2 |
| 49 | pull_partial_model | 4 |
| 52 | put_short_brace | 12 |
| 59 | put_wing_beam | 6 |
| 64 | fit_partial_model | 1 |
| 70 | pull_pin_long | 2 |
| 73 | tighten_tooth_washer | 1 |
| 74 | loosen_tooth_washer | 1 |

These are all **rare classes** (each with 1-12 segments in train). Their absence from val is expected for a 1928-segment sample from a long-tailed distribution.

#### Classes present in val but absent in train (2 classes):

| ID | Action | Val count | Recordings |
|----|--------|-----------|------------|
| 66 | plug_small_screw_pin | 4 | 05_assy_0_1, 26_assy_1_5 |
| 72 | pull_small_screw_pin | 1 | 26_assy_1_5 |

**This IS a distribution shift**: two action classes (`plug_small_screw_pin`, `pull_small_screw_pin`) appear in validation but have ZERO training examples. Any model will score 0% on these classes. They involve small screw pin manipulation, a class of actions that only appears in the participant-05 and participant-26 recordings assigned to val.

Impact: The reported val top-1 accuracy is automatically capped at `(1928-5)/1928 = 99.7%` for these 5 segments. A more realistic impact is that any model that cannot zero-shot generalize to "small screw pin" actions will have a structural disadvantage.

#### Classes with significant distribution shift (>2x or <0.5x, >0.5% in either):

| ID | Action | Train% | Val% | Ratio |
|----|--------|--------|------|-------|
| 0  | take_short_brace | 1.72% | 0.67% | **0.39x** |
| 7  | check_instruction | 11.02% | **18.52%** | **1.68x** |
| 14 | put_pin_long | 0.14% | 0.31% | 2.28x |
| 35 | put_screw_pin | 0.14% | 0.36% | 2.66x |
| 60 | put_pulley | 0.25% | 0.57% | 2.32x |
| 62 | put_acorn_nut | 0.05% | 0.26% | 4.75x |
| 63 | loosen_acorn_nut | 0.03% | 0.10% | 3.80x |
| 65 | take_small_screw_pin | 0.03% | 0.21% | 7.61x |
| 67 | put_small_screw_pin | 0.05% | 0.16% | 2.85x |

The **most impactful shift** is `check_instruction`: **18.5% of val segments vs 11.0% of train segments** (1.68x over-represented). This means val accuracy is disproportionately determined by performance on `check_instruction`. A model that memorizes this class well gets an automatic ~7.5% boost on val vs what it would get on a train-distributed sample.

**Confidence: HIGH** -- verified by counting AR segments per recording per split.

### 3.2 PSR -- Per-component prevalence

| Component | Train prevalence | Val prevalence | Shift |
|-----------|-----------------|----------------|-------|
| C0  | 99.96% | 100.00% | minor |
| C1  | 80.79% | 90.31% | **+9.5pp** |
| C2  | 81.51% | 90.31% | **+8.8pp** |
| C3  | 52.22% | 58.05% | +5.8pp |
| C4  | 17.93% | 9.97% | **-8.0pp** |
| C5  | 62.16% | 64.31% | minor |
| C6  | 59.98% | 55.01% | -5.0pp |
| C7  | 42.30% | 43.02% | minor |
| C8  | 42.30% | 41.86% | minor |
| C9  | 32.90% | 28.63% | -4.3pp |
| C10 | 19.75% | 22.15% | minor |

**Notable shifts**: Components C1 and C2 (which parts?) are ~9 percentage points more prevalent in val than train. Component C4 is 8pp less prevalent. The always-zero predictor for PSR gets **0% binary accuracy** on val since **100.00% of val frames have at least one PSR component = 1** (vs 99.96% in train).

**Confidence: HIGH** -- computed from PSR_labels_raw.csv per recording.

### 3.3 Head Pose

Train pose mean magnitude: 12.55 (dominated by position terms ~110 raw, divided by HEAD_POSE_POS_SCALE).
- Forward direction vectors are near-unit (as expected after the Opus 126 normalization fix).
- Position components are O(1) after scaling.

**Confidence: HIGH** -- computed from pose.csv across all train/val recordings.

---

## 4. Constant-Predictor Baselines (Val Floor)

### 4.1 Action Recognition: Always-predict-NA (class 0)

| Baseline | Val accuracy |
|----------|-------------|
| Always predict 0 (NA/background) | **0.7%** (13/1928 segments) |
| Always predict 7 (check_instruction, most common non-NA in train) | **18.6%** (357/1915 non-NA) |

**Floor for AR**: The naive constant predictor (always-0) gives **0.7%** on val. This is negligible. A more informed constant predictor (always-predict-most-frequent-non-NA-train-class) gives 18.6%.

However, the actual task is multi-class classification with 75 classes; a random classifier would get **1/75 = 1.3%**. The model's current performance (~30-45% top-1) is well above both floors.

### 4.2 PSR: Always-predict-all-zeros

| Baseline | Binary accuracy |
|----------|----------------|
| Always predict zeros on val | 0.0% (every frame has some component=1) |
| Always predict zeros on train | 0.04% (almost every frame has some component=1) |

The PSR always-zero predictor achieves **0% accuracy** on both train and val because PSR transitions happen throughout the assembly: at any given frame, the probability that NEITHER component has been activated is essentially zero.

A better baseline: always predict the **prevalence** (proportion of frames where each component = 1). For the 11 binary tasks, constant-predicting the training prevalence on val gives:
- Expected BCE loss = `-p*log(p) - (1-p)*log(1-p)` for each component, summed.
- Approx 0.4-0.7 nats per component depending on prevalence.

### 4.3 Head Pose: Predict train mean on val

| Metric | Value |
|--------|-------|
| Val MAE (train mean predictor) | **1.4884** mean across 9 DoF |
| Val MAE (naive unit predictor) | **0.4182** mean across 9 DoF |

The naive unit predictor (`[1,0,0, 0,0,0, 0,1,0]` -- forward=+x, up=+y) gets **lower** MAE (0.418) than the train-mean predictor (1.488). This is because the position components dominate the mean MAE: the train mean position is some arbitrary offset (~O(1) in scaled units), while the naive predictor guesses `pos=(0,0,0)` which is closer to the mean position of most recordings.

**The true floor metric for pose should be the within-recording mean predictor** (because the model can learn per-recording biases from visual context). The between-recording mean predictor we computed here is an over-estimate of the true floor.

### 4.4 Detection: Always-predict-no-boxes

| Baseline | mAP |
|----------|-----|
| Predict no boxes | 0.0 mAP |

Detection is fundamentally different: predicting no boxes gets 0 mAP. The "floor" is the minimum-confidence threshold performance.

**Confidence: HIGH** -- all baselines computed from raw data.

### 4.5 Summary: What is the ceiling (oracle on training distribution)?

An oracle that perfectly predicts the **training distribution** (not the training labels) for AR:
- Predict the correct class for all 65 classes that appear in both splits: **perfect score** on the 1923 segments from overlapping classes.
- Gets 0 on the 5 segments from val-only classes (IDs 66, 72).
- **Val upper bound**: 1923/1928 = **99.7%**.

This is theoretical: the real ceiling is limited by annotation noise, ambiguity, and video quality. But importantly, **5 segments are literally impossible to classify correctly** because they belong to classes unseen in training.

For PSR, the oracle upper bound is **100%** (the task is deterministic per recording).
For pose, the oracle upper bound depends on sensor noise in HoloLens pose tracking.

---

## 5. Per-Recording Val Metrics: Do Some Recordings Dominate?

### Frame distribution across val recordings

| Recording | Frames | % of val | AR segs | Unique actions |
|-----------|--------|----------|---------|----------------|
| 26_assy_1_5 | 4,587 | **12.1%** | 207 | 48 |
| 26_assy_0_1 | 3,093 | **8.1%** | 177 | 49 |
| 14_assy_0_1 | 3,005 | **7.9%** | 156 | 42 |
| 20_assy_3_6 | 2,967 | **7.8%** | 121 | 42 |
| 24_assy_2_4 | 2,952 | **7.8%** | 169 | 40 |
| 05_assy_0_1 | 2,918 | **7.7%** | 160 | 46 |
| 20_assy_0_1 | 2,854 | **7.5%** | 129 | 42 |
| 05_assy_2_2 | 2,323 | **6.1%** | 126 | 40 |
| 24_assy_0_1 | 2,158 | **5.7%** | 128 | 38 |
| 20_main_0_1 | 2,066 | **5.4%** | 68 | 28 |
| 14_main_0_1 | 1,685 | **4.4%** | 77 | 31 |
| 14_main_2_3 | 1,679 | **4.4%** | 77 | 30 |
| 26_main_0_1 | 1,594 | **4.2%** | 99 | 33 |
| 14_main_2_2 | 1,404 | **3.7%** | 62 | 30 |
| 05_main_0_1 | 1,380 | **3.6%** | 90 | 34 |
| 24_main_0_1 | 1,371 | **3.6%** | 82 | 28 |

### Dominance analysis

| Metric | Value |
|--------|-------|
| Max recording share | 12.1% (26_assy_1_5) |
| Min recording share | 3.6% (24_main_0_1) |
| Max/min ratio | 3.3x |
| Top-3 share | 28.1% |
| Top-5 share | 43.7% |

**Conclusion: Mild dominance, not pathological.** The largest recording is 12% of total val frames. Top-3 recordings constitute 28% of val. This is reasonable for a multi-participant setup -- no single recording dominates the evaluation. The 3.3x ratio between largest and smallest is moderate.

However, **participant 26 alone accounts for 24.4% of val frames** (26_assy_1_5 + 26_assy_0_1 + 26_main_0_1 = 9,274 / 38,036 = 24.4%). If participant 26 has idiosyncratic assembly style (speed, hand positions, head motion), their recordings will disproportionately influence val metrics. This is a form of **participant-level dominance**.

**Confidence: HIGH** -- computed from exact frame counts.

---

## 6. Detection Coverage on Val

All 16 val recordings have OD_labels.json with annotations:

| Recording | Annotation images | Total annotations |
|-----------|-----------------|-------------------|
| 24_assy_2_4 | 537 | 537 |
| 26_assy_1_5 | 325 | 325 |
| 20_assy_0_1 | 228 | 228 |
| 26_main_0_1 | 227 | 227 |
| 24_main_0_1 | 212 | 212 |
| 24_assy_0_1 | 207 | 207 |
| 20_assy_3_6 | 188 | 188 |
| 14_main_0_1 | 162 | 162 |
| 05_assy_0_1 | 155 | 155 |
| 05_assy_2_2 | 155 | 155 |
| 05_main_0_1 | 142 | 142 |
| 20_main_0_1 | 131 | 131 |
| 26_assy_0_1 | 126 | 126 |
| 14_assy_0_1 | 124 | 124 |
| 14_main_2_3 | 100 | 100 |
| 14_main_2_2 | 83 | 83 |

**Total**: 3,106 annotated images across val. Each annotation image = 1 annotation (exactly one object labeled per image). Coverage density = 3,106 / 38,036 = **8.2% of val frames have OD labels**. This is extremely sparse.

The per-recording annotation count varies from 83 (14_main_2_2) to 537 (24_assy_2_4), a 6.5x range. Recordings with more OD annotations will have more detection gradient signal during evaluation.

**Confidence: HIGH** -- directly from OD_labels.json.

---

## 7. Verdict on v1 Claims

| v1 Claim | Actual | Corrected? | Confidence |
|----------|--------|------------|------------|
| "10 train recordings" | **36** train recordings (12 participants, up to 3 views each) | WRONG | HIGH |
| "6 val recordings" | **16** val recordings (5 participants) | WRONG | HIGH |
| "10 + 6 recordings" | Probably counted only `*_main_0_1` recordings | REVISED | HIGH |
| "No temporal overlap" | 12 train participants, 5 val participants -- **disjoint** | CONFIRMED | HIGH |
| "Distribution shift exists" | 2 val-only actions (IDs 66, 72), check_instruction 1.68x over-represented | CONFIRMED, QUANTIFIED | HIGH |
| "Val dominated by few recordings" | Top-3 = 28%, max = 12.1%, participant 26 = 24.4% | PARTIALLY (mild dominance, participant-level) | HIGH |

---

## 8. Implication for Training Runs

1. **The val set is larger than the v1 analysis suggested** (16 recordings, 38K frames vs previously thought ~6 recordings). This means val metrics are more statistically stable but also have more distribution diversity.

2. **The train set is also larger** (36 recordings, 26K stride-3 frames). With 36 recordings, the training data covers more assembly variants than the v1 "10 recordings" estimate.

3. **check_instruction over-representation** on val (18.5% vs 11.0% in train) means val top-1 accuracy is inflated relative to train-distributed performance by approximately 7.5 percentage points * the model accuracy on check_instruction. If the model gets check_instruction right 80% of the time, that's a ~6pp boost.

4. **Two val-only action classes** (IDs 66 and 72: small screw pin actions) mean the model will score exactly 0% on at least 5 val segments. This is a structural cap, not a model limitation.

5. **Participant 26** accounts for 24.4% of val frames. If the training participants have different assembly styles, this single val participant's recordings will dominate the reported metrics.

6. **OD annotation coverage**: 8.2% of val frames have annotations (3,106 of 38,036). The per-recording range is 83-537 annotations (6.5x). Detection metrics may be dominated by the best-annotated recordings.

---

## 9. Claude Science Queries

### Query 1: What is the actual class distribution shift between train and val for all 75 AR classes?

See Section 3.1 above. Key findings:
- 2 val-only classes (66, 72: small screw pin actions) with 5 segments total
- check_instruction (ID 7) over-represented 1.68x (18.5% val vs 11.0% train)
- 9 rare train classes absent from val (IDs 25, 28, 49, 52, 59, 64, 70, 73, 74)
- Overall distribution is moderately shifted but not catastrophically

### Query 2: Do action classes co-vary with recording? Is the split stratified?

The split is **participant-based**, not stratified by class. Classes co-vary with recordings because:
- Each recording captures one assembly attempt by one participant
- The specific actions performed depend on the assembly variant and participant choices
- Some assembly variants involve specific parts (screw pins, small screw pins, wheels)

Result: val-only classes exist because the small-screw-pin variants were assigned entirely to val participants. Class stratification would have prevented this.

### Query 3: Is the val set harder or easier than the test set?

This is a separate analysis. The test set (32 recordings, 10 participants, 90K frames) is entirely disjoint from both train and val. Without running the model on test, we can note:
- Test has more participants (10) than val (5), so participant-specific effects are diluted
- Test has 32 recordings vs val's 16, giving better statistical coverage
- Need to run the same analysis on test to compare distributions

### Query 4: Does frame stride create a bias in val evaluation?

Train uses stride=3 (every 3rd frame), val uses stride=1 (all frames). This means:
- Training sees temporally subsampled data (26K frames from 79K raw)
- Validation evaluates on ALL frames (38K frames)
- This is asymmetric: the model is trained on sparser temporal data than it is evaluated on
- For PSR (which has temporal structure through the MonotonicDecoder), stride mismatch could create a systematic bias -- the val evaluation measures performance at 10 FPS while training signal is at 3.3 FPS
- Impact: estimated 1-3% metric deflation on temporally-sensitive tasks (PSR event-F1, AR temporal boundaries) due to the model not being optimized for the evaluation temporal resolution

**Confidence: MEDIUM** -- the exact impact depends on the temporal granularity of learned features.

---

## 10. Recommended Actions

1. **Track per-class val accuracy separately for check_instruction (ID 7)** to understand if reported top-1 is inflated by its over-representation.

2. **Track per-recording val metrics** to identify if participant 26 recordings drive a disproportionate share of results.

3. **Compute train-distributed val accuracy** by reweighting per-class val accuracy by train class frequency. This gives a "debiased" estimate.

4. **Evaluate val-only classes (66, 72) separately** -- flag them as zero-shot generalization tests, not in-distribution performance.

5. **Run val at stride=3 to match training** and compare with stride=1 results to measure the temporal resolution bias.
