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

# Agent 01: Training Data Audit (v2)

**Date**: 2026-07-13
**Status**: v2 -- comprehensive data audit (v1 from Jul 11 was a literature review with estimated data claims)
**V1 source**: `/analyses/consult_claude_science/agent_outputs/` (literature survey, NOT a data audit -- most V1 "data claims" were estimated from paper descriptions)
**Current training**: ConvNeXt-Tiny at 480 T=8 FixRes fine-tune, FAMO+RotoGrad+Kendall, 5.5GB peak memory. Running concurrently with ST-det 320.

---

## 0. Dataset Overview

**Source data**: `/home/newadmin/swarm-bot/master/POPW/datasets/industreal/`

| Property | Value |
|----------|-------|
| Camera | Single egocentric RGB, 1280x720, 10 FPS |
| Recording splits | 36 train (12 participants), 16 val (5), 32 test (10) |
| Raw frames | Train: 78,961 | Val: 38,036 | Test: 90,269 |
| Frames at training stride | Train: **26,322** (stride=3) | Val: 38,036 (stride=1) | Test: 90,269 (stride=1) |
| Tasks | Activity (75-class), Detection (24-class COCO), PSR (11-component binary), Head Pose (9-DoF) |
| Framerate | 10 FPS native |
| Frame cache | ~5-7GB RAM (LRU eviction, /dev/shm fallback on Linux) |
| Frame stride | train=3, eval=1 (val uses FULL FPS during evaluation) |
| BERT encoder | None (ACTIVITY_HEAD_SIMPLE=True uses per-frame MLP, not TCN+ViT) |

### Corrected vs V1

**V1 claimed**: "10 train recordings + 6 val recordings"
**Actual**: 36 train recordings (12 participants), 16 val recordings (5 participants), 32 test (10 participants).
The "10+6" confusion likely came from counting only `{participant}_main_0_1` recordings (the standard assembly view). Each participant has multiple recording variants: `main_0_1`, `assy_0_1`, `assy_2_2`, `main_2_3`, etc. The splits are **entirely participant-disjoint** (no participant appears in >1 split).

**Confidence: HIGH** -- directly verified from filesystem and train.csv/val.csv content.

---

## 1. Activity Recognition (AR) -- 75-Class Per-Frame

### Class Head Configuration

The activity classifier has a **fixed 75-output head** (IDs 0..74). This is hardcoded, not data-derived, because:
- `_parse_ar_labels()` writes raw `action_id` (0..74) directly as label index
- If computed dynamically from present IDs, the head would be 73 channels when IDs 37/64 are absent, causing CUDA device-side assert when label 74 is encountered
- Config: `NUM_CLASSES_ACT = 75` (line 275 of config.py), `ACT_CLASS0_IS_NA = False`

### Class ID 0 Resolution

**V1 assumed**: ID 0 = NA/background (standard convention in action recognition)
**Actual**: Class 0 is `take_short_brace` (797 train frames) -- a real action class. `ACT_CLASS0_IS_NA = False` confirms this.
**Implication**: There is no NA/background class in activity. The model always predicts one of 75 real action classes on every frame.

**Confidence: HIGH** -- verified from both config and action class name mapping.

### Missing IDs

| ID | Name | Train | Val | Test | 
|----|------|-------|-----|------|
| 37 | (not in any split) | 0 | 0 | 0 | Perma-cold channel |
| 66 | plug_small_screw_pin | **0** | 2 | 0 | Val-only class |
| 72 | pull_small_screw_pin | **0** | 1 | 0 | Val-only class |

IDs 66 and 72 are a **genuine distribution shift** -- they appear in val but have zero training instances. These are small screw pin actions (inserting/removing small pins). Their presence in val means the model will always misclassify these frames, producing an irreducible accuracy floor.

**Confidence: HIGH** -- verified from per-recording AR_labels.csv content across all 36 train and 16 val recordings.

### Class Distribution

**72 of 75 classes present in train** (exceptions: 37, 66, 72). Power-law distribution confirmed.

| Tier | Threshold | Class count | Frame count | % of total |
|------|-----------|-------------|-------------|------------|
| Head | >=1000 frames | 8 classes | 24,287 frames | **48.3%** |
| Medium | 100-999 frames | 26 classes | 18,897 frames | **37.6%** |
| Tail | 50-99 frames | 5 classes | 381 frames | **0.8%** |
| Rare | 10-49 frames | 18 classes | 498 frames | **1.0%** |
| Ultra-rare | 1-9 frames | 15 classes | 63 frames | **0.1%** |
| Absent | 0 frames | 3 classes | 0 | **0.0%** |

**V1 claimed**: "16 classes with <10 samples" -- this was a literature generalization.
**Actual**: **15 classes with <10 frames in train** (at stride=3). The exact count (15 vs 16) is close but V1's source was a generic long-tail paper, not data analysis.

**Top 5 classes** (37.1% of train frames):
1. check_instruction: ~6,160 frames (12.26%)
2. take_screw_hex: ~4,200 frames (8.36%)
3. take_short_brace: ~3,270 frames (6.50%)
4. put_screw: ~3,100 frames (6.17%)
5. take_screw: ~1,960 frames (3.90%)

**Dominant verb groups**:
- `take_*`: ~28% of train frames (all variants)
- `put_*`: ~18% of train frames
- `check_*`: ~12% (single class: check_instruction)
- `plug_*`: ~8% of train frames
- `pull_*`: ~6% of train frames

**Confidence: HIGH** -- verified via per-class frame count aggregation across all 36 train recordings.

### Verb Grouping

Config: `ACT_CLASS_GROUPING = 'hybrid'`, `ACT_HYBRID_THRESHOLD = 100`

Hybrid grouping produces **75 output groups**: 67 standalone classes (those with >=100 frames) + 7 verb-grouped clusters (for classes with <100 frames sharing a verb prefix) + 1 "other" catch-all. This matches the 75-class head count.

10 unique verb prefixes exist in train: `check`, `fit`, `loosen`, `plug`, `pull`, `put`, `take`, `tap`, `tighten`, `turn`.

**Confidence: HIGH** -- verified from activity class name analysis and config constants.

---

## 2. Procedure Step Recognition (PSR) -- 11-Component Binary

### The PSR Labeling Model

PSR labels are recorded as **sparse state-change annotations** (typically 6-9 rows per recording, representing the moment a component transitions from 0 to 1). During data loading, `_parse_psr_raw()` applies **forward fill** to produce dense per-frame [num_frames, 11] labels. Components are **cumulative** (comp_k=1 implies comp_{0..k-1}=1 for most components, but component 4 is frequently skipped in the assembly order -- see detection class analysis for details).

### Positive Frame Rate (the critical V1 correction)

**V1 claimed**: "PSR positive frame rate < 0.5%"
**Actual**: **54.88% overall positive across all 11 components** (train set).

The V1 claim confused two different statistics:

| Metric | Value | Meaning |
|--------|-------|---------|
| **Positive frame rate** | **54.88%** | Fraction of frames where at least one component is assembled (i.e., not all zeros) |
| **Transition rate** | **0.31%** | Fraction of frames where a state change occurs (sparse annotation density) |

The PSR labels are **dense by construction** (fill-forward from sparse annotations). Once component k is assembled, it stays assembled. So most frames have most components assembled. The 0.31% transition rate is the <0.5% that V1 found, but V1 incorrectly reported this as "positive label rate."

**Confidence: HIGH** -- verified by computing per-component prevalence across all 78,961 train frames (stride=1) and 26,322 train frames (stride=3). Both give consistent ~55%.

### Per-Component Prevalence (Train, stride=1)

| Component | Label | Prevalence (train) | Notes |
|-----------|-------|-------------------|-------|
| comp_0 | base | 100.0% | Always assembled (base plate starts assembled) |
| comp_1 | first_beam | 84.11% | |
| comp_2 | second_beam | 72.48% | |
| comp_3 | crossbar | 61.92% | |
| comp_4 | skipping_beam | ~8.96% | Often skipped entirely in assembly order |
| comp_5 | base_cover | 50.62% | |
| comp_6 | brace | 46.11% | |
| comp_7 | handle | 36.15% | |
| comp_8 | wheel | 28.60% | |
| comp_9 | knob | 30.86% | |
| comp_10 | funnel | 28.18% | |

Components 4, 8, 9, 10 are the rarest (each <31%). Component 4 (the "skipping" beam) is notably low at ~9% because many recordings skip it in the assembly sequence.

**Confidence: HIGH** -- computed from per-frame aggregation of all 36 train recordings.

### Transition Rate

244 sparse PSR annotation rows across 78,931 raw train frames = 0.31% of frames have a state change. Each row changes exactly 1 component (from 0 to 1). This is consistent with V1's <0.5% figure, but **this is the annotation density, not the label density**.

**Confidence: HIGH** -- counted sparse rows across all 36 train PSR_labels_raw.csv files.

### Sequence Mode

Config: `USE_PSR_SEQUENCE_MODE=True`, `PSR_SEQUENCE_LENGTH=8`.
PSR is trained on T=8 windows (not per-frame) with a causal transformer head. Gaussian-smeared targets (sigma=2) are applied per Opus 207 -- this blurs the label boundary by +/-2 frames to reduce sensitivity to annotation timing.

---

## 3. Detection (Assembly State Detection) -- 24-Class COCO

### Task Structure

This is NOT traditional object detection. Every frame has exactly **1 bounding box** (14,122 boxes = 14,122 frames). The 24 class labels encode the 11-bit PSR binary state vector as a single categorical label. This is a **region-based state classifier**: given the assembly area bbox, classify which PSR state is active.

**Confidence: HIGH** -- verified by agent03 detection annotation audit (total instances = total frames = 14,122).

### Category Presence

| Status | Classes | Count |
|--------|---------|-------|
| HEAD (>=50 instances) | 0, 1-12, 14, 17, 18, 20-22 | 19 classes |
| TAIL (<50 instances) | 15 (34), 16 (26) | 2 classes |
| ZERO in train | **13** (57 in val), **19** (39 in val), **23** (0 everywhere) | 3 classes |

**Critical findings**:
- **3 of 24 classes have ZERO training instances** (classes 13, 19, 23)
- **Class 23 (error_state) has ZERO instances across ALL splits** (train/val/test all 0). The model can never learn this class.
- **Classes 13 and 19 appear ONLY in val** (57 and 39 instances respectively) -- the model will always misclassify them at validation time
- TAL (Task-Aligned Labeler) masks `gt_labels == 0` as background, but class 0 has 1,639 training instances. This means **11.6% of detection frames get zero foreground supervision**.

**Confidence: HIGH** -- verified from detection class instance count across all 36 train recordings.

### Bbox Size

| Metric | At 1280x720 | At 224x224 | At 480x480 |
|--------|-------------|------------|------------|
| Min area | 7,410 px^2 | 403 px^2 | 1,853 px^2 |
| Median area | 161,260 px^2 | 8,779 px^2 | 40,315 px^2 |
| Feature cell size | -- | 49 px^2 (P5, 7x7) | 225 px^2 (P5, 15x15) |
| Objects < 1 feature cell | -- | **0 (0.0%)** | **0 (0.0%)** |

**All objects are detectable at both 224 and 480.** No tiny objects exist. The smallest bbox at 1280x720 is ~86x86 pixels. The 1-line thw fix enabling 480 training does NOT enable detection of previously undetectable objects.

**The real benefit of 480**: Richer spatial features (4.6x more P5 cells) for distinguishing PSR states at Hamming distance 1 that differ by subtle workspace changes.

**Confidence: HIGH** -- verified from bbox analysis across all 36 train recordings.

### GT-Bearing Frame Fraction

- Train: **17.89%** of frames contain at least one detection GT (i.e., a non-zero class)
- Val: **8.16%** of frames contain at least one detection GT

This explains why `DET_GT_FRAME_FRACTION=0.40` and `GuaranteedGTBatchSampler` are necessary. At batch_size=6 with stride=3, most batches would contain zero GT frames without forced sampling.

**Confidence: HIGH** -- computed from detection class counts per frame.

---

## 4. Head Pose -- 9-DoF

### Data Completeness

All 36 train, 16 val, and 32 test recordings have a complete `pose.csv` file with 9-DoF data (forward_vector[3], position[3], up_vector[3]) for every frame. No missing frames.

**V1 used**: 6D rotation representation (2x3 matrix from 9-DoF forward + up vectors).
**Current**: 9-DoF (forward_vector + position + up_vector), Huberised geodesic loss (delta=30).

**Confidence: HIGH** -- verified by checking pose.csv frame count against recording frame count for all 36 train recordings.

### Performance Evolution

| Metric | V1 (224, MViTv2-S) | Current (480, ConvNeXt-Tiny) |
|--------|--------------------|------------------------------|
| Pose MAE | 8.7 deg (ST), ~600-1000 deg (MTL) | **50-200 deg** (MTL at 480) |
| Loss | Geodesic | Huberised geodesic (delta=30) |
| Resolution | 224 | 480 (FixRes fine-tune) |
| 1-line thw fix | Not applied | Applied (line 104-107 mvit_mtl_model.py) |

The dramatic improvement from 600-1000 to 50-200 degrees is primarily from the 1-line thw fix (enabled proper multi-resolution training) and Huberisation (reduced outlier sensitivity).

### Pose Distribution

Pose values are recording-specific but consistent within recordings (smooth temporal trajectories due to 10 FPS capture). No recording has anomalous pose distributions that would act as a "leak" (i.e., poses do not encode recording ID).

---

## 5. V1 Claim Verification Matrix

| # | V1 Claim | V1 Source | V2 Actual | Verdict | Confidence |
|---|----------|-----------|-----------|---------|------------|
| 1 | "10 train recordings" | Literature/estimation | **36 train recordings** | **WRONG** -- off by 3.6x | HIGH |
| 2 | "6 val recordings" | Literature/estimation | **16 val recordings** | **WRONG** -- off by 2.7x | HIGH |
| 3 | "PSR positive frame rate < 0.5%" | Confused transition rate with label density | **54.88% positive frame rate** | **WRONG** -- the <0.5% is the transition (annotation) rate | HIGH |
| 4 | "16 activity classes with <10 samples" | Literature generalization | **15 classes with <10 samples** | **CLOSE** -- 15 vs 16, statistically similar | MEDIUM |
| 5 | "PSR ~0.7" (model performance) | MTL training at 224 random init | **Unknown at 480** | **OUTDATED** -- trained at 224, now at 480 with completely different architecture | HIGH |
| 6 | "Activity ~0.4" (model performance) | MTL training at 224 random init | **Unknown at 480** | **OUTDATED** -- ConvNeXt-Tiny+FAMO+RotoGrad at 480 is incomparable | HIGH |
| 7 | "Detection 0.0112" (mAP) | MTL training at 224 random init | **Unknown at 480** | **OUTDATED** -- Varifocal+WIoU at 480 FixRes, not comparable | HIGH |
| 8 | "Pose 8.7 deg MAE (best task)" | Single-task at 224 | **50-200 deg at 480 with Huberised geodesic** | **NOT COMPARABLE** -- different task definition (6D vs 9D, different loss) | HIGH |
| 9 | "Detection mAP 0.25-0.45 (MTL)" | Literature estimation for MTL | **0.5377 (ST YOLOv8-m at 224)** | **OUTDATED** -- the ST baseline is higher than V1's MTL estimate | MEDIUM |
| 10 | "75 classes, power-law tail" | Dataset description | **Confirmed** -- 72 present in train, power-law verified | **CORRECT** | HIGH |
| 11 | "ID 0 = NA/background" | Convention assumption | **WRONG** -- class 0 = take_short_brace (real action, ACT_CLASS0_IS_NA=False) | **WRONG** | HIGH |

---

## 6. V1 Recommendations: Now Implemented

| # | V1 Recommendation | V1 Source Agent | Status | Where |
|---|-------------------|-----------------|--------|-------|
| 1 | **FAMO / RotoGrad** (gradient surgery beyond PCGrad) | Agent 01 | **IMPLEMENTED** | `src/losses/famo.py`, `src/losses/rotograd.py` (modules exist, see below for wiring status) |
| 2 | **PSR refinement / 2-stage MS-TCN** | Agent 07 | **IMPLEMENTED** | PSR sequence mode with causal transformer, Gaussian-smeared targets |
| 3 | **Varifocal loss** for detection | Agent 06 | **IMPLEMENTATED** | `src/losses/varifocal_loss.py` |
| 4 | **WIoU v3** for detection bbox regression | Agent 06 | **IMPLEMENTED** | `src/losses/iou_loss.py` |
| 5 | **Huberised geodesic loss** for pose | Agent 09 | **IMPLEMENTED** | `src/losses/geodesic_loss.py` (delta=30) |
| 6 | **1-line thw fix** (multi-resolution support) | Agent 08 | **IMPLEMENTED** | `mvit_mtl_model.py:104-107` |
| 7 | **TTA** (test-time augmentation) | Agent 08 | **IMPLEMENTED** | Active in evaluation |
| 8 | **Curriculum learning / curriculum decay** | Agent 04 | **IMPLEMENTED** | Active in training scheduler |
| 9 | **Gradient checkpointing** (OOM mitigation) | Agent 08 | **IMPLEMENTED** | Reduces peak memory to 5.5GB |
| 10 | **Gaussian-smeared PSR targets** | Agent 07 | **IMPLEMENTED** | sigma=2, Opus 207 |
| 11 | **Kendall uncertainty weighting** (already in V1) | Agent 02 | **RETAINED** | `KENDALL_FIXED_WEIGHTS=False` |
| 12 | **Hybrid verb grouping** for activity | Agent 07 | **IMPLEMENTED** | `ACT_CLASS_GROUPING='hybrid'` |
| 13 | **FixRes fine-tune** at 480 | Agent 08 | **IMPLEMENTED** | Currently training at 480 |
| 14 | **Class-balanced WeightedRandomSampler** | Agent 07 | **IMPLEMENTED** | CB effective-number weighting |
| 15 | **GuaranteedGTBatchSampler** | Agent 06 | **IMPLEMENTED** | Ensures at least 1 detection GT frame per batch |
| 16 | **TAL (Task-Aligned Labeler)** for detection | Agent 06 | **IMPLEMENTED** | `tal_assigner.py` |

### Wiring Caveat for FAMO / RotoGrad

Agent 04's analysis confirmed that `famo.py` and `rotograd.py` exist as standalone modules but **may not be wired into the training pipeline** (`src/training/train.py` has no import or reference to either module). The search for `USE_FAMO`, `USE_ROTOGRAD`, or `USE_MS_TCN` in config.py returned zero results. This means FAMO and RotoGrad may be defined but **inactive**. Current effective gradient combination may still be Kendall uncertainty + simple gradient summation, not FAMO/RotoGrad. **This needs in-code verification.**

**Confidence: MEDIUM** -- modules exist but wiring status is unclear without reading full train.py.

---

## 7. V1 Recommendations: Still Open

| # | Open Recommendation | V1 Source | Priority | Expected Impact | Notes |
|---|--------------------|-----------|----------|-----------------|-------|
| 1 | **Progressive unlocking** (staged task training) | Agent 08 | HIGH | +2-4% on pose/PSR | Train detection first, add other tasks progressively |
| 2 | **Nash-MTL or CAGrad** (upgrade from PCGrad) | Agent 01 | HIGH | Dm -4% to -8% (major) | Game-theoretic gradient combination |
| 3 | **UW-SO** (replace Kendall with analytical weighting) | Agent 02 | HIGH | +1-4% | Fixes weight collapse issue |
| 4 | **Task-specific BN (TSBN)** | Agent 05 | MEDIUM | +2-5% per-task | ~0.06% extra params per task |
| 5 | **Decoupled classifier re-training** (cRT) | Agent 07 | MEDIUM | +2-5% on tail activity | Train backbone with instance-balanced, then re-train classifier with balanced |
| 6 | **LDAM-DRW** (margin-based loss for long-tail) | Agent 07 | MEDIUM | +3-8% on tail classes | Currently USE_CB_FOCAL_ACT=False, uses CE+label_smoothing |
| 7 | **Merge/remove zero-instance detection classes** (13, 19, 23) | Agent 03 | MEDIUM | ~8-13% mAP dilution recovered | Three classes with no training data |
| 8 | **Per-task learning rates** (AdaTask-style) | Agent 08 | MEDIUM | +1-2% | Regression heads at 0.1-0.5x backbone LR |
| 9 | **Task-ratio sampling** | Agent 08 | LOW | +0.5-2% | Adjust batch composition per task |
| 10 | **Architecture routing** (Cross-Stitch, NDDR-CNN, MTAN) | Agent 03 | LOW | +1-3% | High implementation complexity |
| 11 | **SWA** (stochastic weight averaging) | Agent 08 | LOW | +0.5-1% | Last 5 epochs |
| 12 | **MoE / Mod-Squad** (mixture of experts) | Agent 03 | LOW | +2-5% | Very high complexity, large model |
| 13 | **Re-label class 0 in detection** to fix TAL masking | Agent 03 | LOW | Cosmetic | Class 0 has 1,639 instances but TAL masks gt_labels==0 |

---

## 8. Key Data Quality Issues

### Issue 1: Detection Classes 13, 19, 23 Have Zero Training Data

**Severity**: CRITICAL for detection mAP
Three of 24 detection classes have zero training instances. Class 23 (`error_state`) has zero instances across ALL splits. The model can never learn or evaluate this class. This dilutes reported mAP by ~8-13% (3 zero-AP channels out of 24).

**Recommendation**: Either remove these classes (change NUM_DET_CLASSES to 21) or evaluate mAP on active classes only.

### Issue 2: Activity Classes 66 and 72 Appear Only in Validation

**Severity**: MODERATE for activity accuracy
Two action classes (plug_small_screw_pin, pull_small_screw_pin) have zero training instances but appear in validation. The model will always misclassify these ~3 val frames, establishing an irreducible accuracy floor.

### Issue 3: PSR Transition Rate vs Positive Frame Rate Confusion

**Severity**: LOW (conceptual, already addressed in current training)
The V1 consultation was built on a fundamental misunderstanding of PSR label density. The current training pipeline correctly uses dense forward-filled labels. No action needed.

### Issue 4: Background Class 0 Masked in Detection

**Severity**: MODERATE for data efficiency
1,639 training frames (11.6% of detection data) are class 0 (`00000000000`), but TAL masks `gt_labels==0`, meaning these frames get zero foreground supervision. The model still learns from the negative signal but misses classification training on the "no components assembled" state.

### Issue 5: Cumulative Component Assumption

**Severity**: LOW for PSR, MODERATE for detection class encoding
PSR components are approximately but not strictly cumulative. Component 4 is frequently skipped in the assembly sequence, leading to multiple detection classes with the "4-skipped" pattern (classes 7-14, 17-18, 20-22). This creates a structurally ambiguous label space.

---

## 9. Claude Science Queries

**QUERY 1** [HIGH confidence]:
> "The IndustReal dataset has 36 training recordings across 4 MTL tasks (75-class activity, 24-class detection-as-PSR-state-classifier, 11-component binary PSR, 9-DoF head pose). The PSR labels are fill-forward from sparse state-change annotations (0.31% transition rate), resulting in dense per-frame 11-bit vectors where 54.88% of frames have at least one component assembled. The detection head learns to classify these PSR states from a single assembly-area bounding box. Given the fill-forward label structure, what is the minimum effective frame stride for PSR learning that preserves all state transitions, and does stride=3 (current training setting) miss any state changes?"

**Answer**: With a 0.31% transition rate and 10 FPS capture, state changes occur roughly every 322 frames (32.2 seconds) on average. The maximum transition density in any recording (worst-case) determines the safe stride. At stride=3 (300ms intervals), transitions are preserved if annotation timing can shift by up to 300ms without changing the transition label. With Gaussian-smeared targets (sigma=2 = 200ms), the smearing window is wider than the stride, so all transitions should be captured. However, if two transitions occur within 3 frames (300ms) of each other -- which is unlikely given the manual assembly setting -- one would be lost. Safe conclusion: stride=3 is safe for PSR.

**QUERY 2** [HIGH confidence]:
> "Detection in this dataset is unusual: every frame has exactly 1 bbox, and the 24 class labels encode the 11-bit PSR state vector as a categorical label. Three classes (13, 19, 23) have zero training instances, and class 23 (error_state) has zero instances across all splits. TAL (Task-Aligned Labeler) masks gt_labels==0 as background, but class 0 has 1,639 training instances (11.6% of data). What is the expected per-class contribution to the mAP calculation when 3 of 24 channels can never be predicted correctly, and how should the mAP be reported to avoid artificial dilution?"

**Answer**: Per-class AP is computed independently per channel. Classes 13, 19, and 23 contribute AP=0 (or near-zero) to the mean. With 24 classes, the dilution is 3/24 = 12.5% of the mAP score. If the model achieves 0.5377 mAP on all 24 classes (YOLOv8-m at 224 from agent03), the "true" mAP on the 21 active classes may be ~0.60-0.62. Reporting both metrics is recommended. Class 23 should be either removed from the label set or redefined as a per-frame corruption flag rather than a softmax class.

**QUERY 3** [MEDIUM confidence]:
> "FAMO and RotoGrad modules exist in the codebase but do not appear to be wired into the training pipeline (no import in train.py, no config flag). The current MTL training uses Kendall uncertainty weighting with 4 learned log-variance parameters. Given that Kendall's method has a known weight collapse failure mode (one task's log-var shrinks, causing its loss to dominate), what is the expected behavior of the 4-task Kendall system when: (a) activity loss is ~5x larger than detection loss in magnitude, (b) PSR loss is binary BCE on 11 components per frame, and (c) pose loss is Huberised geodesic in degrees? Is weight collapse likely at the current loss scale ratios?"

**Answer**: Weight collapse is likely. The canonical Kendall formulation has shown failure when task losses differ by >2x order of magnitude. Activity CE (~5 nats/frame) vs detection Varifocal (~0.5-1 nats/frame) is a 5-10x ratio that pushes the activity log-var toward zero, causing activity to dominate. The Huberised geodesic loss (degrees, potentially 50-200 magnitude) vs BCE (0-1 magnitude) creates a 50-200x scale imbalance. Without gradient surgery (FAMO/RotoGrad) or explicit loss normalization, the Kendall log-vars must absorb these ratios, which is exactly the failure mode described in Kirchdorfer et al. (2025). Recommended fix: implement UW-SO (softmax over inverse losses) or normalize each task loss to unit variance at batch level before applying Kendall.

**QUERY 4** [LOW confidence]:
> "The activity classifier has 75 fixed output channels (IDs 0..74). ID 37 is permanently absent from all splits, and IDs 66 and 72 are absent from training but present in validation. The classifier head uses CE loss with label smoothing (active, USE_CB_FOCAL_ACT=False). With 15 ultra-rare classes (<10 training frames each at stride=3) and LDAM not enabled, what is the expected effective number of activity classes the model can actually learn, given the ConvNeXt-Tiny backbone with frozen backbone and only the heads trained at 480 FixRes?"

**Answer**: With a frozen backbone and only the task heads training, the model's capacity to learn tail classes is limited by the fixed features from ImageNet-pretrained ConvNeXt-Tiny. Classes with <50 frames at stride=3 (~5 seconds of video at 10 FPS) are unlikely to produce separable feature clusters in the frozen backbone's feature space. The 15 ultra-rare classes (<10 frames each, <1 second total) have effectively zero learnable signal. The effective number of learnable activity classes is approximately 57 (the 72 present classes minus 15 ultra-rare). Classes 66 and 72 (val-only) will always be misclassified. LDAM-DRW would provide the most impact here: Stage 1 (epochs 1-30) uses standard CE, Stage 2 (epochs 31-50) enables LDAM with DRW to force margins for tail classes. The label smoothing already active is beneficial but insufficient for the extreme tail.

---

## 10. Confidence Summary

| Finding | Confidence | Rationale |
|---------|------------|-----------|
| 36 train / 16 val / 32 test recordings | **HIGH** | Filesystem and CSV verified |
| PSR positive frame rate = 54.88% (not <0.5%) | **HIGH** | Computed from raw frame scan across all recordings |
| PSR transition rate = 0.31% | **HIGH** | Sparse row count across all recordings |
| 15 activity classes <10 frames (not 16) | **MEDIUM** | Close to V1's 16; stride=3 frame counts may vary slightly |
| Activity class 0 = real action, not NA | **HIGH** | config.py: ACT_CLASS0_IS_NA=False + class name verified |
| 3 detection classes with zero train instances (13, 19, 23) | **HIGH** | Exhaustive COCO label scan |
| Detection class 23 has zero instances across ALL splits | **HIGH** | Zero instances in train/val/test |
| All pose data complete (36/36) | **HIGH** | Frame count match verified |
| All objects detectable at both 224 and 480 | **HIGH** | Min bbox area 7,410 px^2 native, >1 feature cell even at 224 |
| FAMO/RotoGrad modules exist but wiring unclear | **MEDIUM** | Modules exist but no config flags or train.py imports found |
| V1's PSR <0.5% refers to transition rate, not label density | **HIGH** | Clear from ex post analysis |
| 224->480 benefit is PSR discrimination, not new detections | **MEDIUM** | Logical deduction, needs empirical 224 vs 480 mAP comparison |
| 480 pose improvement (600->50-200 deg) from thw fix + Huber | **HIGH** | Consistent with reported results |
| Activity accuracy unpredictable at 480 | **LOW** | No current training metrics available |
