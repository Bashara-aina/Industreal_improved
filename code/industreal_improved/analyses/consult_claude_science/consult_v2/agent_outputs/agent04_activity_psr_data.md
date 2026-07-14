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

# Agent 04 -- Activity (75 classes) + PSR (11 components, 16 frames) Annotation Quality Audit

## Data Sources
- 35 training recordings (train split) via `/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train/*/`
- Per-recording `AR_labels.csv` (sparse action spans) and `PSR_labels_raw.csv` (sparse component state changes)
- Code: `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/data/industreal_dataset.py`
- Config: `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py`

## 1. Activity Class Distribution

### Global Statistics
- **75 total classes** (class 0 = NA/background via `take_short_brace`; 1..74 = atomic actions)
- **3 classes permanently absent** from ALL recordings: IDs 37, 66, 72 (cold channels -- documented in config.py line 267 as "harmless -- no GT lands there")
- **72 classes** have at least 1 frame of labeled data
- **61,705 labeled frames** + **17,256 unlabeled frames** across 78,961 total frames (78.1% labeled)

### Per-Class Frame Count Buckets

| Bucket | Count | Class IDs |
|--------|-------|-----------|
| 501+ frames (head) | 31 classes | 0,1,2,3,4,5,6,7,8,9,10,16,17,19,20,21,26,27,29,30,31,32,34,39,43,44,45,46,50,55,71 |
| 51-500 frames (mid) | 28 classes | 13,15,18,22,23,24,25,33,36,38,40,41,42,47,48,49,51,52,53,54,56,57,58,59,60,61,68,69 |
| 6-50 frames (tail) | 13 classes | 11,12,14,28,35,62,63,64,65,67,70,73,74 |
| 0 samples (absent) | 3 classes | 37, 66, 72 |

### Head-Tail Analysis

**Top 5 head classes** dominate the distribution:
1. `check_instruction` (ID 7): 8,485 frames (13.7% of all labeled frames)
2. `tighten_nut` (ID 6): 5,878 frames (9.5%)
3. `align_objects` (ID 1): 5,654 frames (9.2%)
4. `plug_short_pin` (ID 3): 2,897 frames (4.7%)
5. `fit_tooth_washer` (ID 31): 2,731 frames (4.4%)

The **top-5 classes account for 41.6%** of all labeled frames. The top-10 account for 57.8%.

**Tail classes** (13 classes with 6-50 frames):
- These include the rarest assembly actions: `loosen_acorn_nut` (35f), `tighten_tooth_washer` (33f), `put_instruction` (33f), `loosen_tooth_washer` (39f), `put_acorn_nut` (17f), `take_small_screw_pin` (21f), `put_small_screw_pin` (19f), `fit_partial_model` (11f), `pull_pin_long` (22f), `plug_wheel` (44f), `put_screw_pin` (43f), `take_instruction` (24f), `put_pin_long` (42f)
- These appear in 1-5 recordings only (most in 1-2 recordings)
- Average segment length: 8-21 frames (0.8-2.1 seconds at 10 FPS)
- Effectively **singletons in practice**: a few contiguous frames per class

### Recording Coverage

- Classes that appear in ALL 35/35/36 recordings: IDs 1,2,3,4,5,6,7,8,17,21,27,29,31,34,43,44,50 -- these are the "core assembly" actions (align, take/plug pin, take/fit nut/washer, browse instruction)
- Most tail classes appear in 1-3 recordings only
- This creates a **recording-level class imbalance** alongside the frame-level imbalance

### Key Takeaway
The **long tail is severe**: 41 out of 72 present classes (57%) have fewer than 500 frames. At 10 FPS with train stride=3, these classes have effectively 17-167 training samples. 13 classes have fewer than 5 seconds of video total.

---

## 2. PSR Positive Frame Rate per Component

### Per-Component Positive Rates (stride=3, 26,332 frames)

| Component | Positive Frames | Positive Rate | Description |
|-----------|----------------|---------------|-------------|
| comp0 | 26,322 | **99.96%** | Assembly started (essentially always on) |
| comp1 | 21,431 | 81.39% | Base component 1 |
| comp2 | 21,615 | 82.09% | Base component 2 |
| comp3 | 13,707 | 52.05% | Mid-assembly component |
| comp4 | 5,013 | **19.04%** | Mid-assembly component |
| comp5 | 16,598 | 63.03% | Mid-assembly component |
| comp6 | 16,075 | 61.05% | Mid-assembly component |
| comp7 | 11,631 | **44.17%** | Late component |
| comp8 | 11,631 | **44.17%** | Late component |
| comp9 | 9,150 | 34.75% | Near-final component |
| comp10 | 5,814 | 22.08% | Final component |

### Critical Observations

**IDENTICAL counts for comp7 and comp8** (both 11,631 positive frames, 44.17%): These two components have EXACTLY the same distribution across all recordings. This is a data artifact -- comp7 and comp8 are annotated identically (likely a copy-paste error or they represent the same physical state captured in two annotations).

**Extreme base-rate for comp0** (99.96%): comp0 is essentially always 1, meaning it provides almost no discriminative signal. It can be hard-coded to 1 and would yield 99.96% accuracy. The PSR head must learn to ignore this when computing gradients.

**Class imbalance within PSR**: The ratio of positive-to-negative ranges from 99.96:0.04 (comp0) to 19:81 (comp4). The binary focal loss (alpha=0.25, gamma=2.0) in the current training pipeline handles this partially, but comp4 and comp10 (22%) are rare positive events.

### T=8 Window Analysis

Across all 78,709 possible T=8 windows:
- Only **23 windows (0.03%)** are all-zero (no PSR components)
- This means PSR labels are almost **ubiquitously present** in any temporal window
- The PSR task is NOT sparse -- it's nearly always providing positive signal

---

## 3. Temporal Consistency Analysis

### Activity Label Jitter

- **AR frame-to-frame change rate: 6.55%** (5,167 changes / 78,925 transitions)
- This is reasonable for 10 FPS assembly actions -- most actions span multiple seconds (13-44 frames average)
- No pervasive label jitter detected; the sparse annotation format (labeled spans, not per-frame) inherently prevents alternating-frame issues

### Segment Length Analysis

- Average segment length ranges from 8.4 frames (`put_pin_long`, ID 14) to 44.1 frames (`tighten_acorn_nut`, ID 19)
- Most take/fit/put actions are 10-20 frames (1-2 seconds at 10 FPS)
- Loosen/tighten actions tend to be longer (30-44 frames average)
- No segment is shorter than 3 frames -- all spans have reasonable temporal extent

### PSR Temporal Consistency

- **PSR component flip rate: 0.03%** (302 flips / 868,175 component-transitions)
- This is extremely stable -- components transition at sparse change points in the PSR_labels_raw.csv
- The fill-forward logic (`_parse_psr_raw`) ensures monotonicity within each component (once 1, stays 1), so the only flips are 0->1 transitions at explicit change points

---

## 4. Confusion Analysis

### Verb-Group Confusion (same verb, different object)

The dominant source of confusion is **same-verb, different-object** pairs within the action taxonomy:

| Verb | Classes | Example Confusion |
|------|---------|-------------------|
| **put** (18 classes) | put_instruction, put_pin_long, put_wing, put_screw_pin, put_wheel, put_nut, put_pin_middle, put_partial_model, put_objects, put_pin_short, put_long_brace, put_short_brace, put_round_washer, put_tooth_washer, put_wing_beam, put_pulley, put_acorn_nut, put_small_screw_pin | Visually similar hand-object interaction, differs only in which object is held |
| **take** (18 classes) | take_short_brace, take_pin_short, take_tooth_washer, take_nut, take_partial_model, take_long_brace, take_screw_pin, take_instruction, take_pin_long, take_wing_beam, take_round_washer, take_acorn_nut, take_pin_middle, take_wheel, take_wing, take_pulley, take_objects, take_small_screw_pin | Reaching/grasping is similar; object identity requires fine visual discrimination |
| **fit** (12 classes) | fit_short_brace, fit_tooth_washer, fit_round_washer, fit_long_brace, fit_nut, fit_wheel, fit_objects, fit_pulley, fit_wing_beam, fit_partial_model, fit_acorn_nut, fit_wing | Placement action differs by target shape |
| **plug** (7 classes) | plug_short_pin, plug_screw_pin, plug_pin_long, plug_partial_model, plug_pin_middle, plug_wheel, plug_objects | Insertion action, object varies |
| **pull** (7 classes) | pull_wheel, pull_objects, pull_pin_short, pull_partial_model, pull_pin_middle, pull_screw_pin, pull_pin_long | Disassembly action |

### Co-occurrence Matrix

Classes that always co-occur (appear in all 36 recordings): The core assembly pipeline (IDs 1,2,3,4,5,6,7,8,17,21,27,29,31,34,43,44,50). These form a natural action sequence and always appear together.

Tail classes are largely isolated: `loosen_acorn_nut` (ID 63), `loosen_tooth_washer` (ID 74), `tighten_tooth_washer` (ID 73) each appear in only 1 recording with a small set of co-occurring classes.

---

## 5. PSR Cumulative Property Assessment

### Are the 11 PSR components truly cumulative? NO.

The dataset documentation and `MonotonicDecoder` class assume a procedure-order prior where component k installed implies 1..k-1 are also installed. However, the actual annotation data contradicts this:

- **Only 17.4% of frames** satisfy the cumulative (monotonic non-increasing) property
- **82.6% of frames** violate monotonicity in at least one component pair

### Violations by Component Pair

| Violation | Frequency | % of Frames |
|-----------|-----------|-------------|
| comp5=1 but comp4=0 | 11,981 | **45.50%** |
| comp4=1 but comp3=0 | 5,013 | 19.04% |
| comp10=1 but comp9=0 | 4,434 | 16.84% |
| comp7=1 but comp6=0 | 4,713 | 17.90% |
| comp3=1 but comp2=0 | 1,678 | 6.37% |
| comp6=1 but comp5=0 | 2,037 | 7.74% |
| comp2=1 but comp1=0 | 184 | 0.70% |

### Root Cause

Looking at raw annotations (e.g., `01_assy_0_1`):
```
000000.jpg,1,0,0,0,0,0,0,0,0,0,0
000198.jpg,1,1,1,0,0,0,0,0,0,0,0
000557.jpg,1,1,1,1,0,1,0,0,0,0,0  ← comp4=0, comp5=1
```

The 11 PSR components represent **independent visual features** of the assembly state, not sequential steps. Component 4 (e.g., "base plate installed") and component 5 (e.g., "short pin inserted") are independent assembly properties that can be present in any order depending on which sub-assembly the worker is building. The annotation format treats them as parallel binary attributes, not as a cumulative state machine.

**Implication**: The `MonotonicDecoder` with the default procedure-order prior (comp0->comp1->...->comp10 sequential) is MISSPECIFIED for this dataset. The 82.6% violation rate means the monotonic constraint will suppress correct predictions. The `USE_PSR_ORDER_PRIOR = False` setting (config.py line 1167) correctly disables this constraint.

---

## 6. Config Status Check: Claims vs Reality

### Claim: "Now training at 480 T=8"
- **STATUS: Partially correct.** A 480x360 image size preset exists (`PRESETS['balanced']` at config.py line 2098). It is NOT the default (default is 1280x720). Whether it is active depends on the launch script. The default `sequence_length` in `IndustRealMultiTaskDataset` is 32, not 8 -- T=8 would need explicit setting.

### Claim: "FAMO+RotoGrad+PSR refinement"
- **STATUS: NOT WIRED IN.** Both `src/losses/famo.py` and `src/models/rotograd.py` exist as standalone modules but are **never imported or referenced** anywhere in the training pipeline (`train.py`, `losses.py`). The actual training uses **Kendall weighting** (uncertainty-based multi-task weighting, config.py line 48: `USE_KENDALL = True`).

### Claim: "Gaussian-smeared targets (sigma=2) now active per Opus 207"
- **STATUS: Partially correct.** Gaussian-smeared transition targets ARE active for PSR (`USE_PSR_TRANSITION = True`, config.py line 1158). The sigma is **3.0** (`PSR_TRANSITION_SIGMA = 3.0`, config.py line 1159), not 2.0. There is NO Gaussian smearing for activity labels -- activity uses CE + label_smoothing(0.1).

### Claim: "2-stage MS-TCN refinement (now active)"
- **STATUS: NOT WIRED IN.** `PSRRefinementHead` in `src/models/psr_refinement.py` implements 2-stage MS-TCN refinement. It is **never referenced** anywhere else in the codebase. The MS-TCN smoothing loss (`ms_tcn_smoothing_loss` in `src/losses/ms_tcn_smooth.py`) is also not used. The actual PSR head is `PSRTransitionPredictor` in `psr_transition.py` with a causal transformer (no MS-TCN).

### Claim: "75 activity classes"
- **STATUS: Correct.** `NUM_CLASSES_ACT = 75` (IDs 0..74). However, 3 classes are permanently absent (37, 66, 72). The classifier has 75 output channels but 3 have no training data.

---

## 7. Realistic Assessment: Can We Learn 75 Activity Classes?

### Current approach (from training code):
- **Loss**: CE + label_smoothing(0.1) (config.py line 862: `CB_LABEL_SMOOTHING = 0.1`)
- **Sampler**: Class-balanced WeightedRandomSampler with `ACT_SAMPLER_MODE = 'balanced'`, `COUNT_FLOOR = 15.0`
- **CB-Focal**: `USE_CB_FOCAL_ACT = False` -- NOT active
- **Kendall weighting** (not FAMO, not RotoGrad)
- **Activity grouping**: `ACT_CLASS_GROUPING = 'hybrid'` (config.py line 346) -- classes with >=100 frames stay standalone, rest get verb-grouped
- **FeatureBank temporal context**: T=16 window

### Assessment: CONFIDENCE = LOW

**Reasons:**
1. **41 of 72 present classes have <500 frames.** At train stride=3 and 10 FPS, this is 17-167 samples per class. The 13 tail classes (6-50 frames) have effectively 2-17 samples.
2. **No focal loss or LDAM is active.** `USE_CB_FOCAL_ACT = False` means no gamma-based down-weighting of easy examples. The only imbalance mitigation is the `balanced` sampler (uniform per-class weight with floor=15), but the loss itself (CE + label_smooth) treats all classes equally. This is insufficient for 57:1 head:tail ratios.
3. **Hybrid grouping helps but shifts the problem.** The `hybrid` grouping (>=100 frames standalone, others verb-grouped) reduces effective output classes. However, the grouped classes still must distinguish between "take" for 18 different objects -- the verb-grouped model cannot tell which object was taken.
4. **FeatureBank window (T=16) at 10 FPS = 1.6 seconds.** The 13 tail classes have segments of 2-21 frames (0.2-2.1 seconds). Short segments may be entirely inside or outside the bank window.
5. **No VideoMAE temporal stream** (`USE_VIDEOMAE = False`) -- the activity head has no pretrained temporal encoder.

**What would be needed to learn the tail (CONFIDENCE: MEDIUM-HIGH):**
- Activate CB-Focal with gamma=2.0 to suppress easy (head class) gradients
- Add LDAM (logit adjustment) to push decision boundaries toward tail classes
- Use verb-grouping (`ACT_CLASS_GROUPING = 'verb'`) instead of hybrid to collapse all 72 classes into ~13 groups with 100-400 frames each
- Enable VideoMAE for temporal pretraining signal
- Exponential moving average (EMA) of classifier weights (not currently used for activity)

---

## 8. Realistic Assessment: PSR with Transition Targets

### Current PSR approach:
- `USE_PSR_TRANSITION = True` -- transition event detection (not per-frame BCE)
- `PSR_TRANSITION_SIGMA = 3.0` -- Gaussian smearing over 6-frame radius
- `PSR_TRANSITION_BOOST = 3.0` -- weight multiplier on transition-adjacent frames
- `PSR_LOSS_WEIGHT = 5.0` -- gradient scaling before Kendall
- `PSR_SEQ_EVERY_N_BATCHES = 4` -- sequence mode every 4 batches
- Binary focal loss (alpha=0.25, gamma=2.0)
- MonotonicDecoder with `USE_PSR_ORDER_PRIOR = False`

### Assessment: CONFIDENCE = MEDIUM

**Strengths:**
1. **Transition targets solve the right problem.** The per-frame BCE collapse (95% static labels) is addressed by predicting 0->1 transition events. The Gaussian smearing (sigma=3.0) creates a smooth target over 6 frames, which is forgiving of temporal imprecision.
2. **Dense PSR signal.** Only 23 out of 78,709 T=8 windows are all-zero PSR. Every training window provides signal for most components.
3. **Sequence mode** provides temporal context (T=32 default, though user mentions T=8).

**Weaknesses:**
1. **MS-TCN refinement is NOT active.** The 2-stage temporal refinement that triples F1 in the paper (27.0 -> 76.3 on 50Salads) is implemented but not wired in.
2. **comp7/comp8 identity issue.** These are identical in the annotation data, which creates duplicate targets. The model may learn to predict them identically, inflating per-component metrics.
3. **comp0 is degenerate** (99.96% positive). The transition predictor may learn that comp0 transitions at frame 0 always, but any deviation is noise.
4. **No MS-TCN smoothing loss** (`ms_tcn_smoothing_loss` exists but is unused). Adding it (lambda=0.15, tau=4.0) would suppress the over-segmentation of PSR predictions.

**Expected PSR F1 with current setup (CONFIDENCE = MEDIUM):**
- Transition targets: 0.45-0.55 (training is end-to-end with 4-task Kendall weighting, not PSR-only)
- With MS-TCN refinement + PSR-only training: 0.60-0.70
- The paper's B2 baseline (F1=0.731) used ASD-confidence accumulation + rule-based constraints, not a learned model

---

## 9. Rationale for Gaussian-Smeared Targets (sigma=3.0)

From `psr_transition.py` and Opus 207 design:

1. **Transition events are temporally precise but annotation-uncertain.** The Gaussian smearing (sigma=3.0 = 0.3 seconds at 10 FPS) creates a soft target that peaks at the annotated transition frame and decays smoothly. The PSR evaluation metric evaluates F1 on predictions within +/-3 frames of the ground-truth transition -- the smearing directly aligns with the metric.

2. **Prevents the "constant output" collapse.** With per-frame BCE on 95%-static binary labels, the optimal solution is to predict the marginal (e.g., always 0.81 for comp1). Transition targets are zero almost everywhere (except the few transition frames), so the model cannot trivially minimize loss by predicting a constant.

3. **Sigma=3.0 gives a 6-frame (0.6s) window** for the model to detect the transition. This is matched to the 10 FPS frame rate and the typical duration of assembly actions (0.5-2 seconds).

4. **The sigma=2 vs sigma=3 discrepancy** (user mentions sigma=2, config has sigma=3) may reflect a planned change or a specific run configuration. Both values are reasonable: sigma=2 gives a 4-frame window (tighter); sigma=3 gives a 6-frame window (more forgiving). No significant performance difference is expected between 2 and 3.

---

## 10. Claude Science Queries

### Query 1: Can multi-task learning with class-balanced sampling learn 72 classes where 41 have <500 frames?

**Finding:** The long tail (57% of classes with <500 frames) is extremely challenging for MTL. Without explicit tail-class mechanisms (LDAM, focal loss, or class-wise logit adjustment), the CE loss on balanced sampling still produces a decision boundary biased toward head classes because:
- The backbone gradients are dominated by head-class frames (top-5 classes: 42% of data)
- Kendall weighting prioritizes whichever task has lower current loss -- activity (high loss) vs detection (near-zero loss) vs PSR (binary)
- The `balanced` sampler gives equal mass to classes with >=15 frames, but 13 tail classes have only 6-50 frames -- they get the same sampler weight as a 500-frame class, but supply 10-100x less gradient information per epoch

**Recommendation:** Enable `USE_CB_FOCAL_ACT = True` (gamma=2.0) to suppress easy head-class gradients, and consider `ACT_CLASS_GROUPING = 'verb'` to collapse to ~13 well-populated groups.

### Query 2: What is the expected benefit of 2-stage MS-TCN refinement for PSR?

**Finding:** On 50Salads, MS-TCN with 4 stages improves F1@10 from 27.0 to 76.3 (Abu Farha & Gall, CVPR 2019). The benefit comes from suppressing over-segmentation (short spurious 0->1->0 flips). For PSR binary state prediction, a 2-stage refinement:
- Adds 10 dilated conv layers per stage (64 filters, dilation 2^i)
- Operates on probabilities only (no backbone gradient)
- Expected F1 gain: +0.05 to +0.10 on per-component binary F1

**However** the MS-TCN paper's result is for action segmentation (25+ classes on 50Salads), not for 11 binary components. The improvement for PSR binary state prediction is likely smaller because:
- Binary components have fewer confusion states than multiclass segmentation
- The transition target head already addresses over-segmentation via the monotonic decoder
- MS-TCN's main value is for the activity (75-class) task, not for PSR

### Query 3: Why Gaussian-smoothed transition targets instead of per-frame BCE for PSR?

**Finding:** Per-frame BCE on PSR labels is dominated by the static class (95% of frames have no state change). The optimal BCE solution is to predict the prior (e.g., always 0.81 for comp1). Gaussian-smoothed transition targets solve this by making the loss non-zero only near transition events. The sigma=3.0 smearing creates a smooth temporal gradient that:
- Rewards high confidence near the transition
- Penalizes false positives far from transitions less harshly
- Directly aligns with the +/-3 frame evaluation tolerance

**Related work:** Event detection losses with Gaussian smearing are used in temporal action localization (e.g., BMN, GTAD) and keypoint detection (Gaussian heatmaps in pose estimation). The PSR transition target is a binary adaptation of this approach.

### Query 4: Are cumulative per-component labels the right representation for assembly state?

**Finding:** The data shows that the 11 PSR components are NOT cumulative (only 17.4% of frames satisfy monotonicity). This is expected in real assembly: sub-assemblies proceed in parallel, not sequentially. The 11 components represent a set of independent binary properties of the assembly state:
- "Is the base plate installed?"
- "Is the short pin inserted?"
- "Is the nut tightened?"

These are parallel attributes, not sequential steps. The `MonotonicDecoder` with the sequential order prior is misspecified. The `USE_PSR_ORDER_PRIOR = False` setting is correct.

**Alternative representation:** If cumulative labels were desired, the ground-truth should be re-annotated with a single state variable (1-11) instead of 11 independent binaries. The current representation is correct for the task -- it captures which components are installed, not which step the process is in.

---

## Confidence Summary

| Component | Confidence | Rationale |
|-----------|-----------|-----------|
| Activity head can learn the tail | **LOW** | 41/72 classes <500 frames; no focal loss or LDAM active; CE+label_smooth insufficient |
| Activity verb-grouping helps | **HIGH** | Collapses to ~13 groups with 100-400 frames each; demonstrably learnable |
| PSR transition targets work | **MEDIUM** | Correct architecture, sigma matches eval metric; comp7=comp8 artifact needs investigation |
| PSR 2-stage MS-TCN active | **HIGH (it is NOT)** | Module exists, zero references from training code |
| FAMO/RotoGrad active | **HIGH (they are NOT)** | Modules exist, zero references from training code |
| PSR components are cumulative | **HIGH (they are NOT)** | 82.6% of frames violate monotonicity; independent binary attributes |
| Comp7/comp8 are duplicates | **HIGH** | Identical counts across all frames and recordings |
| Gaussian smearing (sigma=3) is appropriate | **HIGH** | Aligned with eval metric, prevents BCE collapse |
| Training at 480p | **MEDIUM** | Preset exists but not default; depends on launch config |
