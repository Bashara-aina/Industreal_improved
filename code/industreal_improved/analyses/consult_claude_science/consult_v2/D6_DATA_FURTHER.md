# D6 — Data Detailed Adversarial Debate

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D6 (continues D1's data challenges with deeper investigation)

---

## 1. Methodology

D6 challenges R1 with **deeper investigations** into specific data claims that D1 left pending. Focus:
- Per-class instance counts at stride=3
- Activity confusion pairs
- PSR transition timing precision
- Annotation noise quantification

---

## 2. Per-Class Activity Analysis

### 2.1 Power-Law Distribution Verification

**R1 claim:** Top class ~3200 frames, 16 classes <10 frames, 48 classes <100 frames.

**D6 challenge:** Need exact counts. At our effective training size (26,322 frames @ stride=3), 16 classes with <10 frames means 16 classes with <0.04% of data. These are **statistically unrecoverable** even with LDAM-DRW.

**Counter-evidence:** Published long-tail work (OLTR 2019, BBN 2019) typically uses minimum 100-1000 frames per class for tail recovery claims. Our 1-9 frame classes are 10-1000x below this.

**Implication:** Even with logit-adjustment (already implemented), 16 classes will have near-zero recall. Activity top-1 might cap at ~30% even with perfect architecture.

### 2.2 Confusion Pairs

**Need:** Sample 50 misclassified frames and identify which classes are confused.

**Hypothesis:** Fine-grained classes (e.g., `take_pin_short` vs `take_pin_long`) will dominate the confusion matrix. Verb-grouping (collapse 75→20-30 coarse classes) might improve top-1 by 10-20%.

**Action:** Run confusion matrix analysis on current best checkpoint.

---

## 3. PSR Transition Precision

### 3.1 Transition Frame Annotation Accuracy

**R1 claim:** PSR positive rate <0.5%; transitions are sparse events.

**D6 challenge:** Need to know:
- Average transition duration (in frames)
- Average "settled state" duration
- Inter-annotator agreement on transition frames

**Hypothesis:** If transitions occur over 5-10 frames (gradual), our per-frame binary is asking the wrong question. If transitions are 1-2 frames (sudden), our T=8 sequence model should capture them.

**Action:** Plot transition timing histograms per component.

### 3.2 Ground-Truth vs Predicted Transition Timing

**Need:** Compare our predicted transitions vs GT transitions with various tolerances (1, 3, 5, 10 frames).

**Hypothesis:** Our event-F1@3 might be 0.0, but event-F1@10 might be 0.15 (model predicts transitions 7 frames late, but within 10-frame tolerance).

**Action:** Re-run eval with tolerance sweep.

---

## 4. Annotation Noise Quantification

### 4.1 Head Pose Interpolation

**R1 claim:** Head pose is 9-DoF from HL2 sensor, present on every frame.

**D6 challenge:** HL2 sensor provides poses at 30-60 Hz, but our video is 10 FPS. Are poses interpolated? Linear vs spline vs nearest-neighbor?

**Verification needed:** Read `pose.csv` parsing code in `industreal_dataset.py`.

**Implication:** Linear interpolation introduces smoothing that reduces high-frequency noise. This could artificially lower our MAE (model predicts smooth trajectories).

### 4.2 Detection Box Noise

**R1 claim:** Bounding boxes per `COCO format` in `ASD_labels.csv`.

**D6 challenge:** Are boxes annotated:
- Single frame per frame (with possible jitter)
- Tracked across frames (one annotation, propagated)
- Manually verified or auto-generated?

**Implication:** If auto-generated from tracking, ground truth has its own errors. Our detection mAP upper bound is GT noise floor.

---

## 5. Temporal Consistency

### 5.1 Activity Label Jitter

**D6 challenge:** Activity labels are per-frame, but a "natural" activity lasts 30+ frames. Are there label discontinuities (frame N=action_A, frame N+1=action_B, frame N+2=action_A)?

**Hypothesis:** Yes, jitter is common in per-frame labels. This makes per-frame classification harder than necessary.

**Action:** Quantify jitter: count label transitions per recording.

### 5.2 PSR Component Coupling

**D6 challenge:** PSR has 11 components. Are some components always coupled (e.g., "wheel_1_on" → "wheel_2_on" always follows)?

**Implication:** If components are strongly coupled, modeling them jointly (e.g., as a single 2^11 = 2048-class problem) might outperform independent binary classification.

---

## 6. Concrete Action Items

1. **Confusion matrix analysis** on activity (50 misclassified frames)
2. **PSR transition timing histograms** per component
3. **Event-F1 tolerance sweep** (1, 3, 5, 10 frames)
4. **Head pose interpolation check** in dataset.py
5. **Detection box source check** (manual vs tracked)
6. **Activity label jitter quantification**
7. **PSR component coupling analysis**

---

## 7. Survived Findings

| Claim | Status |
|---|---|
| 36/16/32 splits | HIGH |
| 75 activity classes | HIGH |
| 24 detection classes | HIGH |
| 11 PSR components | HIGH |
| Activity power-law | HIGH |
| ConvNeXt-Tiny = 28.59M | HIGH |
| Pose 9-DoF HL2 | HIGH (need interpolation check) |

---

## 8. Output

D6 adds deeper data challenges. The action items above are all runnable in 1-2 days each. They refine our understanding of data quality before designing the final synthesis.
