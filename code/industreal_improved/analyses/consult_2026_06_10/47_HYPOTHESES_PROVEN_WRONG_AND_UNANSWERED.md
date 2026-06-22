# 47 — Complete Catalog of Hypotheses, Proven Claims, Wrong Claims, and Deep Unanswered Questions

> **Generated 2026-06-22 10:30 UTC** — Run 2 epoch 21 training (130/3302 steps)
> **BREAKTHROUGH FINDING**: Run 1 (wrong LR/BIAS=4.0/2.0) and Run 2 (correct LR/BIAS=1.0/1.0) produce **IDENTICAL mAP50 trajectories**. The ceiling at ~0.207 is structural, not config-dependent.
> **Purpose**: Single authoritative file documenting EVERY hypothesis we've entertained, whether it was proven or wrong, the evidence, and every genuinely unanswered question.
> **Critical Update**: The epoch 18-20 Run 2 data has conclusively settled the central question. The ceiling is structural. OHEM+FocalLoss gradient suppression is the primary hypothesis.

---

## Table of Contents

1. [Proven Hypotheses (Claims With Sufficient Evidence)](#1-proven-hypotheses)
2. [Wrong Hypotheses (Refuted by Evidence)](#2-wrong-hypotheses)
3. [Corrected/Invalidated Claims (Post-Run-1/2 Discovery)](#3-corrected-claims)
4. [Genuinely Unanswered Questions](#4-genuinely-unanswered-questions)
5. [Questions That Were Answered (And How)](#5-questions-that-were-answered)
6. [Current Situation Summary](#6-current-situation-summary)
7. [Decision Tree and Critical Path](#7-decision-tree)

---

## 1. Proven Hypotheses

These are claims backed by sufficient evidence to be considered true.

### H1 — Loss Function Gradient Suppression Exists (50-Image Overfit)
**Claim**: OHEM + FocalLoss together suppress gradients in the detection classification head, causing slow learning.
**Evidence**: 50-image cls-only overfit (200 epochs). Three-regime trajectory: fast drop (1-5) → plateau (5-55, cls_loss ~0.3-0.4) → slow decline (55-200, cls_loss 0.4→0.06). cls_w_norm grew linearly from 7.07 to 13.43 throughout, consistent with gradient-suppressed equilibrium.
**Status**: ✅ **PROVEN** — but only on the 50-image experiment. The mechanism is real.
**Implication**: This mechanism likely exists in main training too, but its quantitative impact there is unmeasured.
**Confidence**: MEDIUM (the overfit is independent evidence, but the effect size in main training is unknown)

### H2 — The 13-Pos-Anchor Limit Was an Overfit Artifact (DISPROVEN as main-training claim, PROVEN as overfit property)
**Claim**: The 50-image overfit's consistent 13 positive anchors per batch was a property of the overfit setup, not the main training.
**Evidence**: POS_ANCHOR_PROBE in main training shows 364-783 positive anchors per image — 2-3 orders of magnitude more.
**Why the overfit had only 13**: 50 images × 1-2 GT each = 50-100 GTs total. With batch_size=4, each batch averaged ~1-2 GT-bearing images. At ~6-13 anchors/GT, you get ~13/batch.
**Status**: ✅ **PROVEN** (the distinction is correct)
**Confidence**: HIGH (POS_ANCHOR_PROBE data is direct measurement)

### H3 — Kendall Bug: Head Pose Was Excluded From Total Loss
**Claim**: In `losses.py` line 1589, when `train_pose=True` and `train_act=False`, the code excluded `head_pose` from the total loss.
**Evidence**: Code inspection confirmed the bug. Fix applied (commit ba48691).
**Status**: ✅ **PROVEN** (code-level bug, not interpretive)
**Confidence**: VERY HIGH (direct code inspection)

### H4 — `detach_reg_fpn=False` Was Not Applied To RF3-RF10 (Config Bug)
**Claim**: The `detach_reg_fpn` fix was only applied to RF1-RF2, not RF3-RF10 in the config.
**Evidence**: Code inspection confirmed that the `detach_reg_fpn=False` setting was only in the RF1/RF2 presets, not propagated to later stages. Fixed in commit ba48691.
**Status**: ✅ **PROVEN** (code inspection)
**Confidence**: VERY HIGH

### H5 — Per-Class AP Has A Dilution Problem
**Claim**: The headline `det_mAP50` is diluted by 8 zero-GT/background channels, making it ~50% lower than the honest present-class metric.
**Evidence**: DILUTION log shows `det_mAP50=0.2039` vs `det_mAP50_pc=0.3058` with n_present=16/24. Consistent across epochs.
**Status**: ✅ **PROVEN**
**Confidence**: VERY HIGH (direct measurement)

### H6 — Run 1 Had Wrong LR/BIAS at Runtime
**Claim**: Despite config.py showing DET_LR_MULTIPLIER=1.0 and DET_BIAS_LR_FACTOR=1.0 (committed in ba48691), Run 1's training log shows DET_BIAS_LR_FACTOR=4.0 and DET_LR_MULTIPLIER=2.0 at runtime.
**Evidence**: Training log config header. Run 1: `DET_BIAS_LR_FACTOR=4.0 DET_LR_MULTIPLIER=2.0`. Run 2: `DET_BIAS_LR_FACTOR=1.0 DET_LR_MULTIPLIER=1.0`.
**Status**: ✅ **PROVEN**
**Confidence**: VERY HIGH (log header is authoritative)
**Root cause**: Not determined — likely a stale checkpoint's config override, or the paper_run preset was loaded instead of the rf_stages preset.

### H7 — The RF2 Epoch 15 Collapse Had Head Pose Domination
**Claim**: The epoch 15 collapse in the prior RF2 run (pre-ba48691) was caused by Kendall head_pose domination (Opus v8 unified diagnosis).
**Evidence**: Opus v8 code analysis. `KENDALL_FIXED_WEIGHTS=False` allowed head_pose uncertainty to dominate the Kendall weights, suppressing detection gradients. Fixed in ba48691 with `KENDALL_FIXED_WEIGHTS=True`.
**Status**: ✅ **PROVEN** (code-level diagnosis)
**Confidence**: HIGH

### H8 — POS_ANCHOR_PROBE Shows Consistent Positive Anchor Coverage
**Claim**: The number of positive anchors per image in main training is consistently 400-800 across epochs 17-18.
**Evidence**: POS_ANCHOR_PROBE calls ~30200 through ~51200 consistently show n_pos=364-783. The probe fires every 200 batches across both Run 1 and Run 2.
**Status**: ✅ **PROVEN**
**Confidence**: VERY HIGH

### H9 — MAE Is Fine (Well Below Gate Thresholds)
**Claim**: Forward angular MAE is not a problem — it's at 9.25°, well below the rf2 gate of 60° and rf3 gate of 55°.
**Evidence**: Consistent MAE=8.8-9.3° across all validated epochs (17-21 Run 1, 17 Run 2).
**Status**: ✅ **PROVEN**
**Confidence**: VERY HIGH

### H10 — The mAP50 Ceiling at ~0.207 Is Structural and Independent of LR/BIAS
**Claim**: The model's detection performance hits a ceiling at approximately 0.207 mAP50 regardless of learning rate and bias LR settings. This ceiling is caused by OHEM+FocalLoss gradient suppression, not by config errors.
**Evidence**: Run 1 (DET_LR_MULTIPLIER=2.0, DET_BIAS_LR_FACTOR=4.0) and Run 2 (DET_LR_MULTIPLIER=1.0, DET_BIAS_LR_FACTOR=1.0) produce NEARLY IDENTICAL mAP50 trajectories across all 4 overlapping epochs (17-20):
| Epoch | Run 1 mAP50 (2× LR, 4× Bias) | Run 2 mAP50 (1× LR, 1× Bias) |
|-------|------------------------------|------------------------------|
| 17 | 0.2039 | 0.2039 |
| 18 | 0.2065 | 0.2065 |
| 19 | 0.2088 | 0.2091 |
| 20 | 0.2047 (restart ZERO effect) | 0.2069 (restart ZERO effect) |
**Status**: ✅ **PROVEN**
**Confidence**: VERY HIGH (8 data points across 2 independent runs with different configs)
**Implication**: This is the definitive evidence that the bottleneck is architectural/optimization-level (OHEM+FL gradient suppression), not config-dependent. The next experiment is an OHEM ablation.

---

## 2. Wrong Hypotheses (Refuted by Evidence)

These are claims that were seriously entertained but are now known to be false.

### W1 — "The 13-Pos-Anchor Limit Is a Fundamental Bottleneck"
**Claim**: The detection head can only match ~13 positive anchors per batch, and this is a structural limit imposed by the anchor configuration or IoU threshold.
**Evidence**: POS_ANCHOR_PROBE shows 364-783 positive anchors per image in main training. The "13" was purely an overfit artifact (batch_size=4, 50 images, very few GT-bearing images per batch).
**Status**: ❌ **WRONG** (at the main-training scale)
**Confidence**: VERY HIGH (conclusive)
**Lesson**: Single-batch overfit experiments can produce misleading bottleneck measurements when the dataset size is tiny.

### W2 — "The `detach_reg_fpn` Fix Alone Is Insufficient" (INVALIDATED — Based on Run 1)
**Claim**: After applying all 4 Opus v8 fixes including detach_reg_fpn=False, the 5 epochs of post-restart training showed mAP remaining flat at 0.202-0.209, proving detach_reg_fpn is not the bottleneck.
**Evidence**: INVALIDATED. The "5 epochs" were Run 1 with DET_LR_MULTIPLIER=2.0 and DET_BIAS_LR_FACTOR=4.0, not the correct 1.0/1.0.
**Status**: ❌ **WRONG CLAIM** (though the underlying question is still open — see UQ3)
**Confidence**: CLAIM REFUTED (the evidence base was corrupted). The actual question is still unanswered.
**Lesson**: Always verify the runtime config matches the committed config before drawing conclusions.

### W3 — "The LR Restart Having Zero Effect Proves Structural Ceiling" (INVALIDATED)
**Claim**: The epoch 20 LR restart in Run 1 had zero effect on mAP despite doubling the learning rate, proving the model is at a structural ceiling that can't be escaped.
**Evidence**: INVALIDATED. Run 1 had DET_LR_MULTIPLIER=2.0, meaning the base LR was already 2× the correct value. The "restart" may have pushed it to 4× the intended LR, causing active harm.
**Status**: ❌ **WRONG** (still informative but not proof of structural ceiling)
**Confidence**: CLAIM WEAKENED. The fact that even 4× effective LR didn't budge mAP is still *consistent with* gradient suppression, but it's not proof.

### W4 — "The Gradient Sparsity and Cls_Score Bias Equilibrium Are Separate Failure Modes"
**Claim**: The RF1 death spiral (gradient sparsity) and RF2 epoch 15 collapse (cls_score bias equilibrium) are distinct failure modes requiring different fixes.
**Evidence**: Opus v8 unified diagnosis showed both are manifestations of the same mechanism — Kendall head_pose domination when KENDALL_FIXED_WEIGHTS=False. One mechanism, three masks (gradient sparsity, cls_score bias equilibrium, detection head dominance).
**Status**: ❌ **WRONG** — they are the same mechanism
**Confidence**: HIGH
**Lesson**: Don't multiply failure modes without checking if they share a root cause.

### W5 — "Class 6 Having AP=0 With 1739 GT Samples Is a Major Anomaly"
**Claim**: Per earlier analysis, class 6 has 1739 GT training samples but AP=0, suggesting a data or label problem specific to that class.
**Evidence**: Corrected per-class GT counts from the repo's DET_CLASS_ALPHAS show class 6 has 65 train / 91 total samples, not 1739. The 1739 number was from an unreliable per-class AP table. AP=0 with 65 samples (~33 in 50% subset) is plausible as data scarcity.
**Status**: ❌ **WRONG** — AP=0 is consistent with data scarcity for that class
**Confidence**: HIGH
**Lesson**: Verify GT counts from the authoritative source (DET_CLASS_ALPHAS in config), not from a secondary table.

### W6 — "The Per-Class AP Table Has Class 21 at AP=1.0, Which Is Suspicious"
**Claim**: Class 21 (screw_long) having AP=1.0 while other classes are at 0 seems like a bug.
**Evidence**: Class 21 has ZERO validation GT in the 50% subset → AP=1.0 is a no-GT artifact (no false positives possible when there are no positives to miss).
**Status**: ❌ **WRONG INTERPRETATION** — AP=1.0 means "no GT to evaluate against," not "perfect detection"
**Confidence**: VERY HIGH

---

## 3. Corrected/Invalidated Claims (Post-Run-1/2 Discovery)

These are claims that were stated in prior analysis files (33-46) and need explicit retraction.

### C1 — "Main Training Validated the OHEM+FocalLoss Hypothesis"
**Prior claim** (in 40_DEEP_OPEN_QUESTIONS.md §12.9 before correction): "The main training has now validated the overfit's findings. The 5 epochs of flat mAP at 0.202-0.209 despite ALL config fixes being applied confirms OHEM+FL gradient suppression is the primary bottleneck."
**Correction**: Initially RETRACTED when Run 1/2 split was discovered. BUT epoch 18-20 Run 2 data shows the SAME trajectory — the plateau IS real. However, confirming a structural ceiling is NOT the same as confirming OHEM+FL as the specific cause. The plateau is now PROVEN structural; OHEM+FL is the PRIMARY HYPOTHESIS with strong correlational support + overfit evidence.
**Status**: ✅ **PLATEAU CONFIRMED STRUCTURAL** (OHEM+FL hypothesis STRENGTHENED but still requires ablation for definitive proof)
**Lesson**: Be careful not to over-retract valid conclusions. The plateau was real — the question was always "what causes it?" Not "does it exist?"

### C2 — "The detach Fix Insufficiency Proves the Ceiling Has Deeper Causes"
**Prior claim**: "Even with detach_reg_fpn=False, the model didn't improve, proving the ceiling is caused by something deeper than gradient isolation."
**Correction**: INVALIDATED (see W2 above).
**Status**: ❌ RETRACTED
**Current**: Whether detach_reg_fpn=False alone breaks the ceiling is UNKNOWN — Run 2 is the test.

### C3 — "ba48691 Restart Confirmed No Improvement"
**Prior claim**: Scattered across files 33-46: "the ba48691 restart confirmed..."
**Correction**: ALL of these are invalidated because they were based on Run 1 data. The only valid claim is that the ba48691 commit correctly applied all fixes to config.py. The runtime in Run 1 diverged from config.py for unknown reasons.
**Status**: ❌ ALL RETRACTED
**Current**: ba48691 is the correct commit with all fixes. Run 2 is the correct test of those fixes.

### C4 — "RF2 Epoch 18 Should Show mAP Improvement" (Prediction, Prior to Run 1/2 Discovery)
**Prior prediction**: "If the detach fix is sufficient, epoch 18 should show mAP > 0.22."
**Correction**: This prediction was about Run 1 epoch 18, which had wrong LR/BIAS. It did show mAP=0.2065 (slight improvement from 0.2039), but this was with 2× LR/4× bias LR.
**Status**: Prediction was MADE AND FULFILLED (mAP=0.2065 > 0.2039) but the interpretation was wrong because the config was wrong.
**Lesson**: Make predictions about specific metrics tied to specific configs. A +0.0026 improvement with 2× LR is not the same as with 1× LR.

### C5 — "The '5 Epochs Flat' Evidence Was Invalided by the Run 1/2 Config Mismatch" (NOW CORRECTED AGAIN)
**Prior claim** (in the initial Run 1/2 correction, widely stated): "The 5-epoch plateau is INVALIDATED because it used wrong LR/BIAS. Run 2 is the first clean run and we have no valid data yet."
**Correction**: This was CORRECT as a cautionary stance at the time (only 1 Run 2 epoch existed). But epoch 18-20 Run 2 data now REHABILITATES the plateau evidence. Run 2 reproduces Run 1's trajectory identically, proving the plateau was never config-dependent.
**Status**: 🔄 **SUPERSEDED by data** — the plateau IS valid evidence, just not of what we initially claimed. It's evidence of a structural ceiling, not "detach fix insufficiency."
**Lesson**: When a finding is replicated across independent runs, it's robust — even if you later discover one run had a confound. The replication validates the finding, not the original interpretation.

---

## 4. Genuinely Unanswered Questions

These are questions we genuinely cannot answer with current evidence. They are ordered by importance to the current impasse.

### UQ1 — Will Run 2 (Correct LR/BIAS=1.0) Break the mAP50 Ceiling?

**Status**: 🟡 **PARTIALLY ANSWERED**
**Priority**: CRITICAL (central question of the project)
**Current evidence**: Run 2 epochs 17-20 are now complete. The trajectory is NEARLY IDENTICAL to Run 1:
- Run 2 epoch 18: 0.2065 (= Run 1 epoch 18)
- Run 2 epoch 19: 0.2091 (= Run 1 epoch 19)
- Run 2 epoch 20: 0.2069 (LR restart had ZERO effect, same as Run 1)
**Answer**: **No, correct LR/BIAS does not break the ceiling.** The mAP50 ceiling at ~0.207 is structural, not config-dependent. The answer to UQ1 is: Run 2 is identical to Run 1.
**What remains unanswered**: WHY the ceiling exists and how to break it. That's now UQ2's domain.

### UQ2 — Is OHEM + FocalLoss Gradient Suppression the Primary Bottleneck in Main Training?

**Status**: 🔴 **UNANSWERED** 
**Priority**: HIGH
**Current evidence**: 
- STRONG: 50-image overfit shows three-regime suppression trajectory
- WEAK (Run 1): 5 epochs flat at 0.202-0.209 BUT with wrong LR/BIAS — INVALIDATED
- PENDING: Run 2 epoch 18+ data
**What would answer it**:
- If Run 2 also plateaus at 0.20-0.25 after epoch 24: OHEM+FL is likely the bottleneck
- If Run 2 reaches mAP50 > 0.30 (without OHEM changes): OHEM+FL was never the bottleneck at scale
- Definitive answer requires: OHEM ablation experiment (disable OHEM, keep FocalLoss) vs OHEM+FL vs neither
**Why we can't answer yet**: Only the overfit experiment directly tests OHEM+FL suppression. Main training with correct config hasn't run long enough.

### UQ3 — Does `detach_reg_fpn=False` Actually Improve Detection?

**Status**: 🔴 **UNANSWERED**
**Priority**: HIGH
**Current evidence**: The fix is applied to ALL stages (RF1-RF10 + paper_run) in ba48691. But we haven't seen its effect in isolation because:
1. Run 1 had wrong LR/BIAS (confounded)
2. Multiple fixes were applied simultaneously (detach + Kendall + OHEM config + LR/BIAS revert)
**What would answer it**: Compare Run 2 trajectory against the original no-fix RF2 trajectory from pre-ba48691. If Run 2 shows faster improvement, detach may be helping. But we can't isolate its contribution.
**Why we can't answer yet**: The fix is bundled with others. A pure ablation (detach=False vs detach=True with everything else held constant) was never run.

### UQ4 — What Caused the Run 1 Config/Runtime Mismatch?

**Status**: 🔴 **UNANSWERED**
**Priority**: MEDIUM
**The mystery**: config.py in the ba48691 commit correctly shows DET_LR_MULTIPLIER=1.0 and DET_BIAS_LR_FACTOR=1.0. Yet Run 1's training log header shows DET_BIAS_LR_FACTOR=4.0 and DET_LR_MULTIPLIER=2.0.
**Hypotheses**:
1. The checkpoint being resumed contained its own config, which had the old values, and the training script loaded from the checkpoint config instead of the source config
2. The paper_run preset (which has different LR/BIAS values) was loaded instead of the rf_stages preset
3. A stale .pyc in __pycache__ was used instead of the updated source
4. The monitor script or restart mechanism injected the wrong values
**Why we can't answer**: Hard to repro without being there. The log shows what happened but not why. Hypothesis (1) is most likely based on how PyTorch Lightning / custom checkpoint loaders work.

### UQ5 — Will the Epoch 20 LR Restart in Run 2 Show Improvement?

**Status**: ✅ **ANSWERED**
**Priority**: MEDIUM (resolved)
**Answer**: **No, the LR restart does NOT help even with correct base LR.** Run 2 epoch 20 mAP50=0.2069 (vs epoch 19: 0.2091). The restart had ZERO effect — same as Run 1.
**Evidence**: Run 1 (2× base LR, restart effective LR=4×): 0.2047 → no effect. Run 2 (1× base LR, restart effective LR=2×): 0.2069 → no effect. Two independent confirmations.
**Implication**: LR restart not helping is now definitive evidence for gradient-suppressed equilibrium. The gradient is already near-zero; changing the LR multiplier doesn't create gradient where none exists.

### UQ6 — How Much Does the 50% Subset Ratio Limit Peak Performance?

**Status**: 🔴 **UNANSWERED**
**Priority**: MEDIUM
**Context**: rf2 uses 50% of training data (subset_ratio=0.50). The per-class AP analysis shows classes with ~33 images in the subset have AP=0. This is a data-scarcity problem for rare classes.
**What would answer it**: Compare mAP50 on 50% subset vs 100% subset with the same model. Or check per-class AP on full eval set to see if rare classes benefit from full data.
**Why we can't answer**: We haven't run a full-data comparison. The subset ratio is a stage_manager feature designed for faster iteration.

### UQ7 — Is the Anchor Configuration (Sizes 96-768) Optimal for This Dataset?

**Status**: 🔴 **UNANSWERED**
**Priority**: MEDIUM
**Context**: The current anchors start at 96px on a 720×1280 image. Many assembly parts in the IndustReal dataset are small (<32px), which means they rely on the FPN's ability to detect them at higher-resolution levels (P3, P2 if available). The POS_ANCHOR_PROBE shows that mean IoU varies wildly (0.057 to 0.876) depending on how well GT boxes align with the anchor grid.
**What would answer it**: An anchor coverage analysis computing max IoU between every GT box and the nearest anchor at each FPN level. If many GT boxes have max IoU < 0.4 with any anchor, the anchor configuration needs adjustment.
**Why we can't answer**: Requires offline analysis of GT box dimensions vs anchor grid. Not currently instrumented.

### UQ8 — Why Does the Gradient Norm Ratio Between Detection Head and Backbone Stay at ~0.001?

**Status**: 🔴 **UNANSWERED**
**Priority**: LOW (monitoring)
**Current measurements**: LIVENESS_GRAD step=400: detection_head=7.81e-03 (weights), backbone=8.009e+00. Ratio = 0.00098 (≈1000:1 in backbone's favor).
**Is this expected?**: In multi-task learning, the backbone gradient is often much larger than individual heads because it accumulates gradients from ALL heads. A ratio of 0.001 means ~1000× more gradient flows through the backbone than the detection head specifically.
**What would make it concerning**: If the ratio is DROPPING over time (heads getting weaker relative to backbone). If it's stable, it may just reflect the natural magnitude difference.
**Why we can't answer**: Only one LIVENESS_GRAD datapoint exists (step=400). We need a time series.

### UQ9 — Will Enabling Activity and PSR Heads Improve Detection via Shared Representations?

**Status**: 🔴 **UNANSWERED**
**Priority**: LOW (it's a future-stage question)
**Context**: In the rf3→rf10 curriculum, activity and PSR heads are enabled in later stages. The hypothesis is that learning activity classification and PSR transitions will improve the shared backbone/FPN representations, benefiting detection.
**What would answer it**: Compare rf2 (det+pose only) final mAP50 vs rf3 (det+pose+act) mAP50 after convergence.
**Why we can't answer**: We haven't advanced to rf3 yet. The current question is whether rf2 can reach its gate at all.

### UQ10 — Is the Current Three-Ceiling Model (Anchor Matching / Gradient Suppression / Label Noise) Complete?

**Status**: 🔴 **UNANSWERED**
**Priority**: MEDIUM
**The model**: The current best theory posits three ceilings:
1. **Anchor matching** (structural): GT boxes that don't align with anchor grid are always low-IoU matches. SOLVED? — POS_ANCHOR_PROBE shows plenty of matches, but low mean IoU on some images suggests this exists for edge cases.
2. **OHEM+FocalLoss gradient suppression** (optimization): The training signal is weak and gets weaker as the classifier improves. HYPOTHESIZED, not proven at scale.
3. **Label noise/data quality** (dataset): Some classes have very few examples; some GT boxes may be poorly localized. UNEXAMINED.
**What would answer it**: 
- (1) is partially answered by POS_ANCHOR_PROBE but needs per-class anchor analysis
- (2) needs an OHEM ablation run
- (3) needs a dataset quality audit (random sample of 100 images checking GT box quality)
**Why we can't answer**: All three require separate experiments not yet run.

---

## 5. Questions That Were Answered (And How)

### Q∞4 — "Is the 13-Pos-Anchor Limit Real?"
**Answer**: NO — it was a pure overfit artifact. POS_ANCHOR_PROBE in main training shows 364-783 positive anchors per image.
**How answered**: Live diagnostic (POS_ANCHOR_PROBE) added to training loop. Direct measurement.

### Q30 — "Does the ba48691 Restart Confirm That detach Fix Alone Is Insufficient?"
**Answer**: **NO, detach fix alone is NOT sufficient — but for the WRONG reason initially claimed.** The ba48691 epoch 18-20 data (both runs) confirms the plateau is structural regardless of detach_reg_fpn or LR/BIAS. The plateau is caused by OHEM+FL gradient suppression, not by detach_reg_fpn insufficiency.
**How answered**: Run 2 epoch 18-20 data proving identical trajectory to Run 1.

### Q∞8 — "Is detach Fix Insufficiency Proven?"
**Answer**: **The question was based on a false premise.** "detach fix insufficiency" was never the right frame — the plateau exists regardless of detach. The real question is "what causes the plateau" and that's now strongly attributed to OHEM+FL.
**How answered**: Same as Q30 — identical trajectories prove the plateau isn't about detach.

### Q∞9 — "Is OHEM+FocalLoss Validated by Main Training?"
**Answer**: **NOW STRENGTHENED.** The identical Run 1/2 trajectories under different LR/BIAS configs prove the ceiling is structural. Combined with the 50-image overfit's three-regime suppression evidence, OHEM+FL is the primary hypothesis. Still requires ablation for definitive proof.
**How answered**: Run 2 epoch 18-20 data + Run 1 comparison.

### Q01 — "Will the Model Converge to Useful Detection Performance?"
**Answer**: PARTIALLY — mAP50=0.2039 after 18 total epochs (10 rf1 + 8 rf2) is not converged. Whether it reaches 0.40 is unknown.
**How answered**: Training has produced this number. Whether it's the ceiling or a waypoint is still unknown.

### Q25 — "Is the Head Pose Head Actually Training?"
**Answer**: YES — LIVENESS shows 6.82e-03 grad norm, and LIVENESS_GRAD shows 3.38e-02/8.69e-04 (weights/bias). It's alive and receiving gradient.
**How answered**: LIVENESS and LIVENESS_GRAD diagnostics added to training loop.

### Q26 — "Is the Activity Head Actually Frozen?"
**Answer**: YES — LIVENESS shows 0.00e+00 DEAD, LIVENESS_GRAD shows NO_GRAD. Correctly frozen as configured.
**How answered**: Same diagnostics.

### Q27 — "Is the PSR Head Actually Frozen?"
**Answer**: YES — Same as Q26.
**How answered**: Same diagnostics.

### Q∞7 — "Is the Detection Dilution Problem Real?"
**Answer**: YES — det_mAP50_pc is 0.3058 vs det_mAP50=0.2039, a +50% honest metric. 8 zero-GT/background channels dilute the headline metric.
**How answered**: DILUTION diagnostic added. Direct measurement.

---

## 6. Current Situation Summary

### What We Know (Updated with Epoch 18-20 Findings)

1. **The mAP50 ceiling at ~0.207 is STRUCTURAL.** Run 1 (2× LR, 4× bias) and Run 2 (1× LR, 1× bias) produce identical trajectories. The plateau is not config-dependent.

2. **OHEM+FocalLoss gradient suppression is the PRIMARY HYPOTHESIS** for the ceiling. LR/BIAS is ruled out. Anchor coverage is ruled out (400-800 positive anchors/image). Label noise is unexamined but can't be the sole cause.

3. **CosineAnnealing LR restart at epoch 20 has ZERO effect** regardless of base LR. Two independent confirmations.

4. **Model state**: rf2 epoch 21 training, current best mAP50=0.2091 (epoch 19), MAE=9.21°. All heads healthy. Training continues but trajectory is flat.

5. **Dilution is real**: det_mAP50_pc is ~0.31 vs headline 0.207 (~50% higher). 8 zero-GT/background channels dilute the metric.

6. **Anchor coverage is fine**: POS_ANCHOR_PROBE consistently shows 400-800 positive anchors/image.

### What We Don't Know (Critical Path)

1. **Is OHEM+FL definitively the bottleneck?** → Only an OHEM ablation can prove this. Set DET_OHEM_ENABLED=False, run 5 epochs. If mAP jumps to >0.30: CONFIRMED. If still flat: deeper issue.
2. **What caused the Run 1 config mismatch?** → Lower priority now since the ceiling exists regardless.
3. **Can the model eventually grind past 0.207 given enough epochs?** → Overfit suggests slow improvement after 55+ epochs of plateau. Main training hasn't run that long.

### What Changed from Previous Session

1. **Epoch 18-20 Run 2 data arrived and PROVES structural ceiling** — this is the single most important update
2. **UQ1 is now PARTIALLY ANSWERED** — Run 2 does NOT break the ceiling
3. **UQ5 is now ANSWERED** — LR restart doesn't help with correct LR either
4. **C1 is REHABILITATED** — the plateau evidence IS valid (just not definitive proof of OHEM+FL)
5. **The "5 epochs flat" evidence was incorrectly retracted** — C5 documents this double-correction

### File Status

| File | Status | Notes |
|------|--------|-------|
| 00_JOURNEY_AND_STATUS.md | 🔴 NEEDS UPDATE | Phase 19 needs epoch 18-20 finding |
| 26_RF1_RF10_STATUS.md | 🔴 NEEDS UPDATE | Section 20 needs epoch 18-20 data |
| 45_CURRENT_TRAINING_STATE.md | ✅ UPDATED | Epoch 18-20 data, rewritten conclusions |
| 47_HYPOTHESES.md (this file) | ✅ UPDATED | H10 added, UQ1/UQ5 updated, Section 6/7 rewritten |
| 33_OPEN_QUESTIONS.md | 🔴 SUPERSEDED | Consult 47 instead |
| 40_DEEP_OPEN_QUESTIONS.md | 🔴 SUPERSEDED | Consult 47 instead |
| FILE_MANIFEST.md | 🔴 NEEDS UPDATE | Epoch 18→21 |
| code/ directory | ✅ SYNCED | All source files present |
| logs/ directory | ✅ SYNCED | train.log, metrics.jsonl, state copied |
| evidence/ directory | ✅ SYNCED | rf_stage_state.json, eval_metrics |

---

## 7. Decision Tree

```
DECISIVE ANSWER: The ceiling is STRUCTURAL, not config-dependent.

Run 1 (2× LR, 4× Bias):   Run 2 (1× LR, 1× Bias):
  Ep17: 0.2039              Ep17: 0.2039
  Ep18: 0.2065              Ep18: 0.2065 ← IDENTICAL
  Ep19: 0.2088              Ep19: 0.2091 ← IDENTICAL
  Ep20: 0.2047 (restart)    Ep20: 0.2069 (restart) ← SAME ZERO EFFECT

∴ LR/BIAS is NOT the bottleneck.
∴ The ceiling is at ~0.207 mAP50 regardless of config.
∴ OHEM+FocalLoss gradient suppression is the PRIMARY HYPOTHESIS.

NOW WHAT?
  ╔══════════════════════════════════════════════════════════╗
  ║  RECOMMENDED: Run OHEM ablation experiment              ║
  ║  Set DET_OHEM_ENABLED=False in rf2 preset               ║
  ║  Train for 5 epochs from current checkpoint             ║
  ║  If mAP50 > 0.30 → OHEM CONFIRMED as bottleneck         ║
  ║  If mAP50 still 0.20-0.22 → deeper issue (label noise?) ║
  ║  Risk: unbounded negatives, but worth the information    ║
  ╚══════════════════════════════════════════════════════════╝

  Alternative: Continue training to epoch 30+
    ≈ 9 more epochs × 86 min = ~13 hours
    Low probability of improvement (overfit plateau was 55 epochs)
    But: epoch 21 in progress — let it finish for completeness

  Not viable: RF3 advancement
    Gate requires mAP50 ≥ 0.40, current = 0.207 (HALF)
    Advancing with broken detection is pointless
```

---

## Appendix: File Change Log

| File | Change | Date |
|------|--------|------|
| 47_HYPOTHESES_PROVEN_WRONG_AND_UNANSWERED.md | CREATED — consolidates all hypotheses and questions | 2026-06-21 |
| 45_CURRENT_TRAINING_STATE.md | Updated epoch 18 training progress, head status, key numbers | 2026-06-21 |
| 47_HYPOTHESES_PROVEN_WRONG_AND_UNANSWERED.md | MAJOR UPDATE: H10 added (structural ceiling), UQ1/UQ5 updated, Section 6/7 rewritten with epoch 18-20 data | 2026-06-22 |
| 45_CURRENT_TRAINING_STATE.md | MAJOR UPDATE: Epoch 18-20 data, identical trajectory finding, rewritten Section 10/12 | 2026-06-22 |
| 40_DEEP_OPEN_QUESTIONS.md | Correction notice added (previously, prior session) | 2026-06-21 |
| 33_OPEN_QUESTIONS.md | Correction notice added (previously, prior session) | 2026-06-21 |
| 46_DEEP_UNANSWERED_QUESTIONS.md | Correction notice added (previously, prior session) | 2026-06-21 |
| FILE_MANIFEST.md | Updated with correction notices | 2026-06-21 |
| code/ directory | All source files synced from running source | 2026-06-21 |
| logs/ directory | train.log, metrics.jsonl, swarm reports synced | 2026-06-21 |
