# UNANSWERED QUESTIONS — IndustReal Consultation

**Date:** 2026-07-14
**Purpose:** Items requiring action before AAIML 2027 submission (Oct 10, 2026)
**Companion file:** `MASTER_VERIFICATION.md`

---

## CRITICAL — Block Submission

### Q1: Have we run the ST baselines?
- **Items:** Items 1, 2, 3, 4, 66
- **Why critical:** Without ST baselines, no MTL/ST ratios exist. The paper's core quantitative claim depends on this.
- **Blocker:** ST pose, detection, PSR, activity × 5 seeds = ~100 GPU-hours. Need actual training runs.
- **Estimated GPU-hours:** ~100 (RTX 3060)
- **Action:** Run `bash scripts/launch_st_baselines.sh` (created by A14)
- **Owner:** TBD
- **Status:** ❌ NOT IMPLEMENTED

### Q2: Have we run multi-seed main MTL?
- **Item:** Item 67
- **Why critical:** Statistical rigor requires 5-seed runs with bootstrap CIs. AAIML reviewers will demand this.
- **Blocker:** ~250-300 GPU-hours on RTX 5060 Ti.
- **Action:** Schedule after Tier 1 architecture freeze (Day 14).
- **Status:** ❌ NOT IMPLEMENTED

### Q3: What is the current detection mAP@0.5?
- **Item:** Item 57 (CRIT-1 risk)
- **Why critical:** V1 said "current trajectory positive" but no recent run. Detection is the most vulnerable head.
- **Blocker:** Need a 100-epoch training run to measure.
- **Action:** Run baseline MTL (5 seeds × 100 epochs) and report mAP.
- **Status:** 🔴 NEEDS ATTENTION

### Q4: Does our pose 8.7° MAE beat MediaPipe?
- **Item:** Item 68 (T1.7)
- **Why critical:** If MediaPipe (5° on controlled data) beats us, our "novel pose baseline" claim weakens.
- **Action:** Run `python scripts/mediapipe_pose_baseline.py --checkpoint best.pth` (script created by A28).
- **Status:** 🔵 PARTIALLY DONE (script exists, not run)

---

## HIGH PRIORITY — Resolve Before Submission

### Q5: UW-SO loss weighting implementation
- **Item:** Item 8
- **Why important:** V1 Priority 1 recommendation (Kirchdorfer IJCV 2025). Eliminates Kendall collapse pathology.
- **Blocker:** ~1.5 hours of integration work. Currently module exists (`src/losses/uw_so.py`) but not wired into train.py.
- **Action:** Integrate `uw_so_loss` into `src/training/losses.py` as alternative to Kendall.

### Q6: Per-task learning rates
- **Item:** Item 9
- **Why important:** V1 Priority 3. PSR/pose regression heads need lower LR.
- **Blocker:** Per-task LR config not found in codebase.
- **Action:** Add `PSR_LR_MULTIPLIER` and `POSE_LR_MULTIPLIER` to optimizer groups in train.py.

### Q7: Task-specific BN (TSBN)
- **Item:** Items 24, 40
- **Why important:** V1 Tier 3 item. Recovers ~75% of detection mAP gap.
- **Blocker:** Standard BN only in current code. Needs separate affine params per task.
- **Action:** Implement in `src/models/model.py` FPN.

### Q8: Decoupled activity training (Kang ICLR 2020)
- **Item:** Item 25
- **Why important:** Major paradigm for long-tail activity recognition.
- **Blocker:** Not implemented. Would require new training pipeline.
- **Action:** Implement using existing `decoupled_act_retrain.py` as starting point.

### Q9: Activity top-1 current performance
- **Item:** Item 58 (CRIT-2 risk)
- **Why important:** V1 risk: "30% probability of <20% top-1". Need to know actual current state.
- **Action:** Run ST activity baseline (Item 3).

### Q10: PSR event-F1 current performance
- **Item:** Item 59 (CRIT-3 risk)
- **Why important:** V1 risk: "60% probability of <0.05 F1". Need to verify PSR_TRANSITION fix works.
- **Action:** Run ST PSR baseline (Item 3) and MTL PSR.

### Q11: Is V1 312x gradient ratio still relevant?
- **Item:** Item 50
- **Why important:** V2 measured new ratio: 20,245x (pose vs psr). Pathology 3 mechanism still valid, but numbers updated.
- **Action:** Update paper to cite new gradient norm measurements.

### Q12: ST baselines performance ceiling
- **Item:** Item 62 (HIGH-2 risk)
- **Why important:** V1 risk: "35% probability that ST baselines also poor". Affects MTL/ST ratio story.
- **Action:** Run ST baselines (Items 1-4).

### Q13: Multi-seed result variance
- **Item:** Item 64 (MED-3 risk)
- **Why important:** V1 risk: "30% probability of inconsistent results across seeds".
- **Action:** Run multi-seed main MTL (Item 67).

### Q14: Do we have working consistent-train-val results?
- **Item:** Item 63 (HIGH-5)
- **Why important:** V1 risk: "25% probability of test-val overfitting gap". Need full eval.
- **Action:** After ST baselines done, run test set evaluation.

---

## MEDIUM PRIORITY — Fix If Time Permits

### Q15: ASL for PSR
- **Item:** Item 15
- **Module:** `src/losses/asymmetric_loss.py` exists, NOT wired into PSR loss.
- **Action:** Wire as alternative to focal-BCE.

### Q16: Balanced Softmax for activity
- **Item:** Item 10
- **Module:** `src/losses/balanced_softmax.py` exists, NOT wired.
- **Action:** Replace current activity loss with BalancedSoftmax.

### Q17: DB-MTL log-transform
- **Item:** Item 19
- **Module:** NOT implemented.
- **Action:** Add `log(1 + loss)` to UW-SO pipeline.

### Q18: Two-stage activity training
- **Item:** Item 28
- **Module:** Partial (`decoupled_act_retrain.py`).
- **Action:** Integrate as standard training pipeline option.

### Q19: PSR transition prediction
- **Item:** Item 29
- **Module:** `src/models/psr_transition.py` exists. `PSR_TRANSITION` flag exists.
- **Action:** Verify PSR_TRANSITION is enabled in main config.

### Q20: Per-task augmentation
- **Item:** Item 30
- **Module:** Spatial aug enabled. Detection aug enabled. Activity/PSR/pose per-task aug not separated.
- **Action:** Add per-task augmentation pipelines.

### Q21: TOOD-TAL wiring
- **Item:** Item 72
- **Module:** `src/losses/tal_assigner.py` exists. NOT wired into DetectionHead.
- **Action:** Wire TAL into `DetectionHead.forward()`.

### Q22: Confusion matrix analysis
- **Item:** Item 73
- **Script:** `scripts/activity_confusion_matrix.py` created.
- **Action:** Run on best activity checkpoint.

### Q23: Reference code presence check
- **Item:** Item 84
- **Module:** Need to verify `/media/newadmin/master/POPW/datasets/industreal/` has reference code.
- **Action:** `ls` the directory.

### Q24: EMA warmup verification
- **Item:** Item 12
- **Module:** No `ema_start_epoch` flag found.
- **Action:** Verify whether EMA has warmup or starts at epoch 0.

### Q25: Task head dropout
- **Item:** Item 16
- **Module:** PSR head has `dropout=0.2`. Pose head dropout unverified.
- **Action:** Verify pose head has dropout.

### Q26: SWA window expansion
- **Item:** Item 14
- **Module:** SWA exists but no SWA_WINDOW config flag.
- **Action:** Add SWA_WINDOW=10 to config.py.

### Q27: Mosaic augmentation activation
- **Item:** Item 21
- **Module:** `PRETRAIN_MOSAIC_PROB = 0.3` exists. Need to verify activation in main training.
- **Action:** Verify training script applies mosaic.

### Q28: OHEM ablation
- **Item:** Item 23
- **Module:** `DET_OHEM_ENABLED = True` exists. No ablation run.
- **Action:** Run with OHEM off to measure impact.

---

## LOW PRIORITY — Document As Future Work

### Q29: Nash-MTL-50 gradient surgery
- **Item:** Item 31
- **Module:** Not implemented. PCGrad is sufficient for now.

### Q30: CAGrad gradient surgery
- **Item:** Item 33
- **Module:** Not implemented.

### Q31: Anchor-free detection
- **Item:** Item 34
- **Module:** `src/models/roi_detector.py` exists. Not wired.
- **Note:** V1 said structural ceiling at 224px — diminishing returns.

### Q32: Nash-MTL (full)
- **Item:** Item 32
- **Module:** Not implemented. V1 doc 213 says complex.

### Q33: ConsMTL bi-level optimization
- **Item:** Item 35
- **Module:** Not implemented. V1 marked as "next paper".

### Q34: 9D+SVD pose representation variant
- **Status:** Rejected by V1 doc 215 (marginal improvement, adds complexity).

### Q35: Geodesic loss replacement
- **Status:** Already using 6D+geodesic (SOTA). No further action.

---

## Category: Data

### Q36: What is the actual positive rate for PSR transitions?
- **Source:** V1 doc 218 (claimed <0.5%)
- **Status:** ⚪ PARTIALLY DONE — V2 agent01 verified at <0.5%. Per-component breakdown not done.
- **Action:** Run per-component PSR positive rate analysis.

### Q37: What is the activity class 0 semantics?
- **Source:** V1 doc 209 (claimed NA/background); V2 agent01 (claimed take_short_brace)
- **Status:** ⚪ CONFLICT
- **Resolution needed:** Confirm with annotation documentation.

### Q38: Body pose annotation source
- **Source:** V1 doc 211 (claimed pseudo-keypoints from detection boxes)
- **Status:** ⚪ PARTIALLY ANSWERED — confirmed by V2 R1 §2.5
- **Action:** Document body pose handling in paper.

---

## Category: Architecture

### Q39: Does our backbone choice (ConvNeXt-Tiny) cost accuracy?
- **Source:** V2 D2 challenge
- **Evidence:** MViTv2-S would give +10-15% activity top-1 per V1 doc 214.
- **Status:** ❌ NOT IMPLEMENTED (no comparison run)
- **Action:** Run MViTv2-S as Tier 2 ablation.

### Q40: Is our 6D rotation implementation correct?
- **Source:** V2 R2 + A3 finding
- **Evidence:** Column-swap bug at model.py:2177-2178 FOUND AND FIXED by A11.
- **Status:** ✅ ANSWERED

### Q41: Does BiFPN improve over standard FPN?
- **Source:** V2 T2.1
- **Evidence:** BiFPN module created (model.py:443-540). Ablation not run.
- **Status:** 🔵 PARTIALLY DONE

---

## Category: Training

### Q42: Does distillation actually help?
- **Source:** V2 T1.2
- **Evidence:** Stub implemented (train.py:1567-1623). TEACHER_CACHE_DIR needs ST teachers.
- **Status:** 🔵 PARTIALLY DONE — ST teachers not yet generated.

### Q43: Does FAMO outperform Kendall?
- **Source:** V2 D8
- **Evidence:** USE_FAMO flag added. Not run.
- **Status:** 🔵 PARTIALLY DONE

### Q44: Does MetaBalance help with gradient starvation?
- **Source:** V2 D8
- **Evidence:** USE_METABALANCE flag added. mtl_balancer.py mode="metabalance" implemented.
- **Status:** 🔵 PARTIALLY DONE

### Q45: Does RotoGrad align gradient directions?
- **Source:** V2 D8
- **Evidence:** USE_ROTOGRAD flag added. RotoGradRotation integrated in model.py:2009-2016.
- **Status:** 🔵 PARTIALLY DONE

---

## Category: Paper

### Q46: Is the paper contribution still novel after V2 discoveries?
- **Status:** ⚪ NEEDS PAPER FRAMEWORK UPDATE
- **V2 found:** 11 papers 2025-2026 (Nardon et al. closest). No 4-task MTL on IndustReal.
- **Action:** Verify Nardon doesn't pre-empt our novelty claim.

### Q47: What is the final paper title?
- **Source:** V2 FINAL_PAPER_FRAMEWORK.md
- **Recommendation:** "Multi-Task Industrial Assembly Perception: A Single-Backbone System for Detection, Activity, Procedure State, and Head Pose on IndustReal"
- **Status:** 🟡 DRAFT, needs finalization.

### Q48: Are all 23 R3 citations properly formatted in paper bibliography?
- **Status:** ❓ UNVERIFIED
- **Action:** Cross-check bib format.

### Q49: Has the FABRIC ATRE benchmark been added to related work?
- **Status:** ❌ NOT DONE
- **Action:** Add to paper §2.

---

## Category: Timeline

### Q50: What's the actual days-remaining to AAIML Oct 10?
- **From 2026-07-14 to 2026-10-10:** 88 days
- **Phase 1 (Tier 1 wiring):** Days 0-14 — partially done (8/14 modules wired)
- **Phase 2 (Training runs):** Days 14-42 — NOT STARTED
- **Phase 3 (Paper writing):** Days 35-65
- **Phase 4 (Submission):** Days 65-88

**Risk:** Phase 2 (ST baselines + multi-seed MTL) requires ~400 GPU-hours. Available: ~300. **TIGHT.**

---

## Summary by Status (UNANSWERED only)

| Status | Count |
|--------|-------|
| ❌ NOT IMPLEMENTED (CRITICAL) | 4 |
| ❌ NOT IMPLEMENTED (HIGH) | 10 |
| ❌ NOT IMPLEMENTED (MEDIUM) | 14 |
| ❌ NOT IMPLEMENTED (LOW) | 7 |
| 🟡 POSTPONED | 1 |
| ⚪ CONFLICT | 1 |
| ❓ UNVERIFIED | 1 |
| **Total items needing action** | **38** |

---

**Top 5 Priorities Before AAIML Submission:**

1. **Run ST baselines** (Q1, Items 1-4, 66) — 100 GPU-hours, CRITICAL
2. **Run multi-seed main MTL** (Q2, Item 67) — 250-300 GPU-hours, CRITICAL
3. **Run MediaPipe pose baseline** (Q4, Item 68) — 1-2 hours, CRITICAL
4. **Wire UW-SO** (Q5, Item 8) — 1.5 hours, HIGH
5. **Run current MTL 100-epoch to measure actual mAP** (Q3, Item 57) — 50 GPU-hours, CRITICAL

**End of UNANSWERED_QUESTIONS.md**
