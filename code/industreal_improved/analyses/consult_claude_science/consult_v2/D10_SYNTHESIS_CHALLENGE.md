# D10 — Synthesis Cross-Challenge

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D10 (cross-checks all R1-R5 and D1-D9 for contradictions)

---

## 1. Methodology

D10 looks for **cross-agent contradictions** and **action conflicts** that emerge when all R and D findings are combined.

---

## 2. Contradiction Map

### 2.1 R2 vs R5: Architecture Alignment with Reference

**R2 finding:** Our architecture is fundamentally different from WACV 2024 (ConvNeXt-Tiny vs MViTv2-S, RetinaNet vs YOLOv8m, Standard FPN vs no FPN).

**R5 finding:** WACV 2024 baselines use detection-triggered PSR; ours uses per-frame binary.

**Synthesis implication:** Our comparison to WACV 2024 is fundamentally **unfair on multiple axes**:
- Different backbone
- Different detection head
- Different PSR paradigm

**Resolution:** Frame WACV 2024 as "dataset anchor" not "fair comparison." Use our ST baselines (with same backbone) as fair comparison.

### 2.2 R1 vs D1: Class 0 Interpretation

**R1 finding:** Class 0 = take_short_brace (797 frames), NOT NA/background.

**D1 challenge:** If HL2 idle periods are labeled as take_short_brace, the 797-frame count is artificial.

**Synthesis implication:** Activity taxonomy may need re-examination. 75 classes might be ill-defined if class 0 conflates idle + action.

**Resolution:** Run confusion matrix analysis. If class 0 is confused with NA-like frames, regroup.

### 2.3 R4 vs D9: AAIML Fit

**R4 claim:** AAIML aligned with industrial AI topic.

**D9 challenge:** AAIML scope unverified. Might be "AI for Manufacturing" or "Augmented AI" or "AI in Medicine."

**Synthesis implication:** Need to verify AAIML scope before committing to venue.

**Resolution:** Search AAIML 2024/2025/2026 proceedings. If misaligned, consider WACV or ICRA.

### 2.4 R2 vs D2: ConvNeXt-Tiny for Video

**R2 finding:** ConvNeXt-Tiny + TMA + FeatureBank is the active setup.

**D2 challenge:** No published evidence for ConvNeXt-Tiny on video tasks. MViTv2-S with K400 pretraining would likely give +10-15% activity top-1.

**Synthesis implication:** Our backbone choice trades 10-15% accuracy for ~5M params saved and (potentially) 1.5x throughput.

**Resolution:** Either (a) accept the trade-off and justify with efficiency narrative, or (b) run MViTv2-S as Tier 2 backbone ablation.

### 2.5 D4 vs D9: Pose Novelty

**D4 challenge:** Need MediaPipe comparison to defend pose novelty.

**D9 finding:** Pose alone might be publishable as fallback.

**Synthesis implication:** Run MediaPipe comparison regardless of outcome. Either we beat it (pose is novel) or we don't (pose is commodity, MTL is the contribution).

### 2.6 R1 vs D6: Body Pose Noise

**R1 finding:** Body pose has no real annotations (pseudo-keypoints).

**D6 challenge:** Body pose feeds PoseFiLM. Even with pseudo-keypoints, this could help OR hurt activity.

**Synthesis implication:** Need ablation. `FREEZE_BODY_POSE_BRANCH=True` already exists. Run with this on/off.

---

## 3. Action Conflicts

### 3.1 Time Conflict: ST Pose Baseline vs Main MTL Run

**D4 finding:** Need ST-pose baseline to defend pose novelty.

**D9 plan:** Run main MTL (5 seeds, 100 epochs) on RTX 5060 Ti starting Day 7.

**Conflict:** ST-pose baseline runs ~3-4 hours per seed × 5 seeds = ~20 GPU-hours. While main MTL uses the same GPU.

**Resolution:** Run ST-pose on RTX 3060 (which is faster for ST-only), or schedule ST-pose FIRST (before main MTL on RTX 5060 Ti).

### 3.2 Cost Conflict: Tier 1 Ablations vs Statistical Rigor

**D9 finding:** Tier 1 ablations (GeometryAwareHeadPose, LDAM-DRW, etc.) take 100-150 GPU-hours.

**D9 finding:** Multi-seed (5 seeds × 100 epochs) takes 250-300 GPU-hours.

**Total Tier 1 + main:** 350-450 GPU-hours. Available: ~300.

**Conflict:** Compute-constrained. Can't run both fully.

**Resolution:** Tier 1 ablations on RTX 3060 (smaller batches, fewer seeds), main MTL on RTX 5060 Ti. Cloud backup if needed.

### 3.3 Schedule Conflict: Reference Verification vs Synthesis

**D8 finding:** Need systematic arXiv search for 2025-2026 papers.

**Synthesis phase:** Should be done by Day 28.

**Conflict:** If literature search takes 5+ days, it eats into writing time.

**Resolution:** Parallelize: 1 person doing literature search while 1 runs ablations. Or, accept "search done by Day 21" with gap.

---

## 4. Recommendation Conflicts

### 4.1 "Submit to AAIML" vs "Submit to WACV"

**R4 recommendation:** AAIML 2027.

**D9 recommendation:** Verify AAIML scope first; WACV is also possible.

**Conflict:** AAIML scope unknown.

**Resolution:** Verify both AAIML and WACV CFPs. Choose based on fit.

### 4.2 "Run TOOD-TAL" vs "Stick with RetinaNet"

**D2 recommendation:** Wire TOOD-TAL for +3-5 mAP.

**D7 recommendation:** Verify if the module is correctly implemented.

**Conflict:** Implementation risk.

**Resolution:** Quick smoke test (1 day). If TOOD-TAL wires cleanly, run ablation. If not, skip.

---

## 5. Hidden Assumptions

### 5.1 "R4-V1 doc dates are accurate"

V1 docs are dated 2026-07-11. We assume:
- AAIML deadline Oct 10, 2026 (per V1 doc 216)
- Pose MAE 8.7° target (per V1 doc 220)

**Risk:** If V1 dates are off, our plan is off. V1 dates need cross-check.

### 5.2 "Our codebase is stable"

We assume the active codebase will not undergo major changes between now and Oct 10.

**Risk:** Recent commits (e.g., HOT-FIX 2026-06-07, etc.) suggest code is still being modified. If a major refactor happens, our ablations become invalid.

**Mitigation:** Freeze architecture after Day 14.

### 5.3 "GPU resources are stable"

We assume RTX 3060 and RTX 5060 Ti will be available.

**Risk:** Hardware failure, OS update, etc.

**Mitigation:** Cloud backup ready.

---

## 6. Final Verification Checklist

Before submission, verify:

| Item | Status |
|---|---|
| All 4 heads have multi-seed (5) results | TBD |
| Detection ST baseline complete | TBD |
| Activity ST baseline complete | TBD |
| PSR ST baseline complete | TBD |
| Pose ST baseline complete | TBD |
| GeometryAwareHeadPose ablation run | TBD |
| LDAM-DRW ablation run | TBD |
| 6D rotation claim verified (vs 9-DoF) | TBD |
| Reference code availability checked | TBD |
| MediaPipe pose baseline run | TBD |
| AAIML CFP verified | TBD |
| 2025-2026 literature search complete | TBD |

---

## 7. Survived Findings

| Claim | Status |
|---|---|
| Pose is novel on IndustReal | HIGH |
| Our backbone differs from WACV 2024 | HIGH |
| PSR paradigm mismatch | HIGH |
| Resource budget tight | HIGH |

---

## 8. Refined Findings (Synthesis-Ready)

| Finding | Refinement |
|---|---|
| WACV 2024 anchors are reference, not direct comparison | Document explicitly |
| Pose novelty needs MediaPipe baseline | Tier 1 task |
| AAIML scope unverified | Verify from proceedings |
| Body pose noise impact | Run freeze-body-pose ablation |
| Compute-constrained | Use cloud backup |
| Architecture freeze by Day 14 | Strict |

---

## 9. Output

D10 reveals 6 cross-agent contradictions and 3 action conflicts. The synthesis must address these explicitly.

Key synthesis decisions:
1. **Treat WACV 2024 as reference, not comparison** (use ST baselines for fair comparison)
2. **Verify AAIML scope** before submission
3. **Tier 1 ablations first** (Days 0-7)
4. **Architecture freeze Day 14**
5. **Cloud backup budget $200-500**
6. **Pre-empt reviewer concerns** in supplementary
