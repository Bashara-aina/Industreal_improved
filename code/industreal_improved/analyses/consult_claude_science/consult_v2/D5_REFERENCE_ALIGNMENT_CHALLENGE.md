# D5 — Reference Alignment Adversarial Debate

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D5 (challenges R5 reference alignment)
**Target:** R5_REFERENCE_ALIGNMENT.md

---

## 1. Methodology

Challenges R5 alignment claims by:
- Re-examining reference code (if available)
- Questioning interpretation of WACV 2024 baselines
- Looking for misalignment in evaluation protocol

---

## 2. Specific Challenges

### 2.1 PSR Paradigm Mismatch — Is This Our Mistake or Theirs?

**R5 claim:** WACV 2024 PSR baselines use detection-triggered transition events. Our PSR is per-frame binary. Different paradigms.

**Challenge:** If WACV 2024 considers PSR as "transition detection," then our per-frame binary classification is NOT a PSR contribution — it's a "state estimation" contribution.

**Counter-evidence:** WACV 2024 paper does mention "per-frame binary state" as one of the annotation types (PSR_labels.csv has per-frame state). Our approach is valid but may be a "weak" baseline.

**Mitigation:** Re-frame PSR as "frame-level state estimation" with distinct contribution. OR add transition detection post-processing.

**Status:** MEDIUM severity. Defensible with framing.

### 2.2 Reference Code Availability — Where Is It?

**R5 claim:** Reference repo at `/media/newadmin/master/POPW/datasets/industreal/` is the WACV 2024 code.

**Challenge:** Dataset path != reference code path. The reference code is separate.

**Verification needed:** Read filesystem to confirm whether `industreal_github/` (or similar) exists.

**Status:** PENDING — need filesystem check.

### 2.3 Action Recognition Metric Alignment

**R5 claim:** Both we and WACV 2024 use top-1 accuracy.

**Challenge:** WACV 2024 reports "clip-level" top-1 (16 frames → 1 label). Our MTL also reports clip-level. But our stride=3 sampling means each clip covers 1.6s at 30fps — different from WACV 2024's full 10fps sampling.

**Implication:** Direct comparison might be unfair if WACV 2024 uses denser temporal sampling.

**Mitigation:** Match WACV 2024's exact sampling protocol if possible. Document any deviation.

**Status:** LOW severity if documented.

### 2.4 Detection Resolution — 224px vs 1280px

**R5 claim:** We train at 224px; WACV 2024 at 1280px or 640px.

**Challenge:** Detection mAP at 224px is fundamentally lower than 640px. COCO shows 224px caps at ~30 mAP; 640px reaches ~50 mAP.

**Counter-evidence:** Some papers (e.g., AnchorDet) achieve 40+ mAP at 224px. Our gap might be architecture, not resolution.

**Mitigation:** Run detection at 480px as Tier 2 ablation. If still poor, our backbone/head is the bottleneck, not resolution.

**Status:** MEDIUM severity. Both resolution AND architecture are at play.

### 2.5 Pose Baseline — Is Our 8.7° MAE Even Comparable?

**R5 claim:** No reference for IndustReal pose. Original contribution.

**Challenge:** Reviewers might want comparison to:
- MediaPipe Face Mesh (Google, free, real-time)
- 6Dof Head Pose from Eye Images (Zhu et al., 2023) on similar data

If MediaPipe achieves 5° MAE on our test set, our 8.7° is worse than off-the-shelf.

**Mitigation:** Run MediaPipe on our test set as comparison baseline. (Tier 2 task, ~1 day.)

**Status:** HIGH severity for paper acceptance.

---

## 3. Counter-Evidence: Reference Code May Not Be Available

### 3.1 If reference code is unavailable

If `/media/newadmin/master/POPW/datasets/industreal/` only has the dataset (not the code), we cannot directly run reference baselines.

**Implication:** Our ST baselines must be OUR implementations. We can cite WACV 2024 numbers but cannot reproduce them.

**Mitigation:** Clearly state "ST baselines implemented by us, not by Schoonbeek et al." Cite WACV 2024 numbers as reference anchors only.

### 3.2 If reference code is available but outdated

Reference code may use older dependencies, different splits, or different evaluation.

**Mitigation:** Run reference code on our splits to verify metrics transfer.

---

## 4. Survived Findings

| Claim | Status |
|---|---|
| 24/75/11 task taxonomy alignment | HIGH |
| Splits match | HIGH (per V2 agent01) |
| No pose in WACV 2024 | HIGH |
| WACV 2024 PSR is transition-based | HIGH (per paper text) |

---

## 5. Refined Findings

| Finding | Refinement |
|---|---|
| Reference code available | PENDING filesystem check |
| Pose novelty defensible | Need MediaPipe comparison |
| PSR re-framing | "State estimation" instead of "PSR" |
| Detection comparison fair | Need 480px ablation |

---

## 6. Action Items

1. **Verify reference code presence** in `/media/newadmin/master/POPW/datasets/industreal/`
2. **Run MediaPipe on test set** for pose baseline comparison
3. **Run detection at 480px** as resolution ablation
4. **Re-frame PSR** as "frame-level state estimation" in paper text

---

## 7. Output

D5 challenges R5 reference alignment. The biggest risk is the pose comparison vs MediaPipe — if our 8.7° MAE is worse than off-the-shelf, our "novel baseline" claim weakens. Run MediaPipe comparison as Tier 2 task.
