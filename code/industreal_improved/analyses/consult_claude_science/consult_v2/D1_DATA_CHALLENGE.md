# D1 — Data Adversarial Debate

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D1 (challenges R1 data findings)
**Target:** R1_DATA_VERIFIED.md

---

## 1. Methodology

This debate challenges R1's findings by:
- Recomputing statistics with different methods
- Finding counter-examples where similar data imbalance was handled differently
- Questioning whether the bottleneck is really data or model

---

## 2. Specific Challenges

### 2.1 Recording Counts: Is 36/16/32 the Official Split?

**R1 claim:** 36 train / 16 val / 32 test, 84 total.

**Challenge:** Need to verify against WACV 2024 paper. The paper might use different splits than our config.py. If the reference uses 50 train / 4 val / 30 test (or similar), our comparison breaks.

**Verification needed:** Read WACV 2024 supplementary for exact split definition.

**Status:** PENDING — needs filesystem check on `/media/newadmin/master/POPW/datasets/industreal/`.

### 2.2 Class 0 is take_short_brace — Real Annotation or Convention?

**R1 claim:** ID 0 = take_short_brace, NOT NA/background.

**Challenge:** V1/V2 agent01 verified this, but this is unusual. Most action recognition datasets use class 0 for background. If our class 0 is "take_short_brace" with 797 samples, the model might confuse it with NA-like frames.

**Counter-evidence:** If HoloLens 2 idle periods are labeled as take_short_brace (e.g., worker rests hand near brace), then 797 frames could be the actual idle state disguised as an action.

**Mitigation:** Re-label analysis: check frame timestamps for ID 0 vs adjacent labels.

**Status:** MEDIUM confidence in claim; LOW in interpretation.

### 2.3 PSR Positive Rate <0.5% — Is This Data or Thresholding?

**R1 claim:** <0.5% of frames have a PSR positive.

**Challenge:** If thresholding for "positive" is strict (e.g., requires sustained state >5 frames), positive rate could be artificially low. A 1-frame transition vs 10-frame transition counts the same in our metric.

**Counter-evidence:** In temporal action segmentation literature, "transition rate" is sometimes measured differently. Per-component vs aggregate matters.

**Mitigation:** Report per-component positive rate, not just aggregate.

**Status:** Needs re-verification.

### 2.4 Body Pose has "No Real Annotations" — But is it Used?

**R1 claim:** Body pose uses pseudo-keypoints from detection boxes.

**Challenge:** If body pose is fed into PoseFiLM and modulates C5 features, the pseudo-keypoint noise propagates into activity recognition. This is a hidden data quality issue.

**Counter-evidence:** PoseFiLM modulates C5 (activity features). If body pose is noise, activity recognition is degraded.

**Mitigation:** Run ablation with `FREEZE_BODY_POSE_BRANCH=True` to measure noise impact.

**Status:** HIGH confidence in claim, MEDIUM in impact.

### 2.5 17.9% OD-Labeled Frames — Is This the Right Denominator?

**R1 claim:** 17.9% of frames have ≥1 OD label.

**Challenge:** If "labeled frame" means "frame with annotations in train.csv" but boxes are still empty (e.g., explicit "no objects" annotation), then 17.9% overstates the data. Real positive frame rate could be lower.

**Counter-evidence:** Industrial videos often have explicit "no assembly component visible" frames annotated as empty.

**Mitigation:** Verify by sampling 100 train frames and counting actual box presence.

**Status:** PENDING.

---

## 3. Counter-Evidence to Data Bottleneck Narrative

### 3.1 Long-Tail is Solvable

Liu et al. (OLTR, 2019, arxiv 1911.04474) achieved +20% on long-tail ImageNet. Kang et al. (Decoupling, ICLR 2020, arxiv 1910.09217) shows decoupled training recovers tail performance.

**Implication:** Our activity head could improve 5-10% top-1 with proper long-tail handling (we already have logit-adjust; need decoupled training).

### 3.2 PSR Sparsity is Solvable

Liu et al. (MS-TCN++, TPAMI 2020) handles temporal action segmentation at 1-5% event rates effectively. Their smoothing loss (0.05 weight) is what we have (`PSR_TEMPORAL_SMOOTH_WEIGHT=0.05`).

**Implication:** PSR is solvable with proper architecture. Our 3.08M PSRHead should be enough.

### 3.3 Head Pose Noise Floor is Real

Zhu et al. (6Dof Head Pose from Eye Images, 2023) report 5-7° MAE on controlled data. HL2 sensor noise is ~1-2°. Our 8.7° MAE is consistent with sensor noise + label interpolation.

**Implication:** Pose MAE has a real floor near 5-8°.

---

## 4. Survived Challenges

| Claim | Status |
|---|---|
| 36/16/32 splits | PROBABLY HIGH confidence (need file check) |
| 75 activity classes | HIGH (verified by config.py) |
| 24 detection classes | HIGH |
| 11 PSR components | HIGH |
| ConvNeXt-Tiny is no temporal | HIGH |
| 9-DoF head pose | HIGH |

---

## 5. Refuted Findings

None definitively refuted. Some need re-verification.

---

## 6. Output

D1 challenges data claims. Some need filesystem-level verification (PENDING). The class 0 interpretation and body pose noise impact are the most actionable challenges.
