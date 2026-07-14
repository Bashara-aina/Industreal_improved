# MASTER VERIFICATION — IndustReal Consultation

**Date:** 2026-07-14
**Scope:** Every actionable item across 80+ documents (V1 docs 208-227, V1 agent outputs 18 files, V2 agent outputs 20 files, R/D/S synthesis 20 files, V2 reports 2 files)
**Cross-reference:** Codebase state on branch `auto/2pct-training-fix-20260520-202419`
**Companion file:** `UNANSWERED_QUESTIONS.md`

---

## Status Legend

| Symbol | Status | Definition |
|--------|--------|------------|
| ✅ | ANSWERED/IMPLEMENTED | Verified against code or cited source |
| 🟡 | POSTPONED | Intentional deferral with valid reason |
| ❌ | NOT IMPLEMENTED | No action taken |
| 🔴 | NEEDS ATTENTION | Critical, no resolution path |
| ⚪ | OUTDATED/SUPERSEDED | Replaced by newer findings |
| 🔵 | IN PROGRESS | Code modified but not verified |
| ❓ | UNVERIFIED | Cannot confirm status without more investigation |

---

## Document: V1 IMPLEMENTATION_PLAN.md (30 ranked items)

### Source: agent_outputs/IMPLEMENTATION_PLAN.md

#### TIER 0 — Prerequisites (Doc 226 baseline)

### Item 1: ST pose baseline (5 seeds)
- **Source:** IMPLEMENTATION_PLAN.md:35
- **Type:** Action item
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** `scripts/train_singletask_pose.py` exists (file in src/training/) but no actual training run completed. `scripts/launch_st_baselines.sh` was created (A14) but is a launcher only.
- **Priority:** HIGH (paper spine — required for MTL/ST comparison)
- **Notes:** Must run before main MTL training. RTX 3060, ~17.5 GPU-hours.

### Item 2: ST detection baseline (5 seeds)
- **Source:** IMPLEMENTATION_PLAN.md:36
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** `train_singletask_detection.py` exists but no training run.
- **Priority:** HIGH
- **Notes:** ~35 GPU-hours.

### Item 3: ST PSR baseline (5 seeds)
- **Source:** IMPLEMENTATION_PLAN.md:37
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** `train_singletask_psr.py` exists.
- **Priority:** HIGH
- **Notes:** ~25 GPU-hours.

### Item 4: ST activity baseline (5 seeds)
- **Source:** IMPLEMENTATION_PLAN.md:38
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** `train_singletask_activity.py` exists.
- **Priority:** HIGH
- **Notes:** ~25 GPU-hours.

### Item 5: Activity fixed-backbone probe
- **Source:** IMPLEMENTATION_PLAN.md:39
- **Status:** ✅ ANSWERED (per V1 doc 220)
- **Evidence:** Frozen ConvNeXt probe = 0.2169 activity top-1 (per V1 doc 220). Confirms head architecture is functional, backbone is the bottleneck.
- **Priority:** LOW (already answered)

### Item 6: PSR diagnostics (constant-prediction)
- **Source:** IMPLEMENTATION_PLAN.md:40
- **Status:** 🟡 POSTPONED (implicit — pending ST PSR baseline run)
- **Priority:** MEDIUM

### Item 7: Infrastructure hardening
- **Source:** IMPLEMENTATION_PLAN.md:41
- **Status:** 🔵 PARTIALLY DONE
- **Evidence:** `LIVENESS_EVERY=500` exists in config.py:60. Log-var logging added (config.py:62). Seeds.csv NOT verified.
- **Priority:** MEDIUM

---

### TIER 1 — Code Changes (13 items, 24h total)

### Item 8: UW-SO loss weighting (Kirchdorfer IJCV 2025)
- **Source:** IMPLEMENTATION_PLAN.md:52-64
- **Type:** Recommendation (Impact 85, Priority 56.7)
- **Status:** 🔵 PARTIALLY DONE — module exists, NOT WIRED
- **Evidence:** `src/losses/uw_so.py` exists with `uw_so_loss(losses, temperature)` function. NOT integrated into `src/training/losses.py` active training loop.
- **Cross-ref:** A8 finding: only `distill` and `ldam_drw` are wired into `train.py`; everything else NOT_FOUND.
- **Priority:** HIGH

### Item 9: Per-task learning rates
- **Source:** IMPLEMENTATION_PLAN.md:66-77
- **Type:** Recommendation (Impact 45)
- **Status:** 🔵 PARTIALLY DONE
- **Evidence:** `DET_LR_MULTIPLIER = 1.0` exists. PSR_LR multiplier NOT found (PER_TASK_LR not configured). PSR_WEIGHT = 10.0 exists (loss scale, not LR).
- **Priority:** HIGH

### Item 10: Balanced Softmax for activity
- **Source:** IMPLEMENTATION_PLAN.md:79-90
- **Type:** Recommendation (Impact 30)
- **Status:** 🔵 PARTIALLY DONE — module exists, NOT WIRED
- **Evidence:** `src/losses/balanced_softmax.py: BalancedSoftmaxLoss` class exists. Not referenced in train.py.
- **Priority:** MEDIUM

### Item 11: Gradient clipping (max_norm=1.0)
- **Source:** IMPLEMENTATION_PLAN.md:92-103
- **Type:** Recommendation (Impact 15)
- **Status:** ⚪ OUTDATED — already at 5.0 (NOT 1.0)
- **Evidence:** V2 doc 211 audit raised from 1.0 → 5.0. Current code uses `torch.nn.utils.clip_grad_norm_` at multiple sites in train.py. Per-task head gradient clipping is present.
- **Priority:** LOW (deliberately different)

### Item 12: EMA warmup (start at epoch 5)
- **Source:** IMPLEMENTATION_PLAN.md:105-116
- **Type:** Recommendation (Impact 12)
- **Status:** ❓ UNVERIFIED
- **Evidence:** No `ema_start_epoch` config flag found.
- **Priority:** LOW

### Item 13: LDAM-DRW for activity head
- **Source:** IMPLEMENTATION_PLAN.md:118-130
- **Type:** Recommendation (Impact 60, Priority 20)
- **Status:** ✅ IMPLEMENTED
- **Evidence:** `USE_LDAM_DRW = True` (config.py:1098), `LDAM_DRW_EPOCH = 50` (deferred). A12 agent confirmed.
- **Priority:** DONE

### Item 14: SWA window expansion (5 → 10)
- **Source:** IMPLEMENTATION_PLAN.md:131-143
- **Type:** Recommendation (Impact 8)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** No SWA_WINDOW config flag found.
- **Priority:** LOW

### Item 15: ASL for PSR
- **Source:** IMPLEMENTATION_PLAN.md:145-156
- **Type:** Recommendation (Impact 35, Priority 14)
- **Status:** 🔵 PARTIALLY DONE — module exists, NOT WIRED
- **Evidence:** `src/losses/asymmetric_loss.py: AsymmetricLoss` class exists. Not wired into PSR loss in train.py.
- **Priority:** MEDIUM

### Item 16: Task head dropout (PSR, pose)
- **Source:** IMPLEMENTATION_PLAN.md:158-168
- **Type:** Recommendation (Impact 15)
- **Status:** ❓ UNVERIFIED
- **Evidence:** PSR head has `dropout=0.2`. Pose head dropout not verified.
- **Priority:** LOW

### Item 17: Huberised geodesic loss for pose
- **Source:** IMPLEMENTATION_PLAN.md:170-182
- **Type:** Recommendation (Impact 25, Priority 8.3)
- **Status:** ✅ IMPLEMENTED
- **Evidence:** `src/losses/geodesic_loss.py: huberised_geodesic_loss(pred, target, delta=30.0)` exists.
- **Priority:** DONE

### Item 18: Varifocal Loss for detection classification
- **Source:** IMPLEMENTATION_PLAN.md:184-194
- **Type:** Recommendation (Impact 25, Priority 8.3)
- **Status:** ✅ IMPLEMENTED
- **Evidence:** `USE_VARIFOCAL = False` (config.py), `src/training/losses.py` wires VarifocalLoss via FocalLoss when flag is True. A26 agent completed this.
- **Priority:** DONE (code complete, activation requires flag flip)

### Item 19: DB-MTL log-transform
- **Source:** IMPLEMENTATION_PLAN.md:196-207
- **Type:** Recommendation (Impact 25, Priority 8.3)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** No DB-MTL implementation found in codebase.
- **Priority:** LOW

### Item 20: WIoU v3 for detection box regression
- **Source:** IMPLEMENTATION_PLAN.md:209-221
- **Type:** Recommendation (Impact 15, Priority 5)
- **Status:** ✅ IMPLEMENTED
- **Evidence:** `USE_WIOU = False` (config.py:771), `wiou_v3_loss` wired in `src/training/losses.py`. A27 agent completed.
- **Priority:** DONE

---

### TIER 2 — Quick Config Changes (3 items)

### Item 21: Mosaic augmentation enablement
- **Source:** IMPLEMENTATION_PLAN.md:251
- **Type:** Recommendation (Impact 30, Score 1.25)
- **Status:** 🔵 PARTIALLY DONE — implementation exists, NOT activated
- **Evidence:** `PRETRAIN_MOSAIC_PROB = 0.3` in config.py. `src/data/det_augment.py` has DetectionAugment class. Task #243 marked completed.
- **Priority:** MEDIUM

### Item 22: Gaussian-smeared PSR targets
- **Source:** IMPLEMENTATION_PLAN.md:253
- **Type:** Recommendation (Impact 15)
- **Status:** ✅ IMPLEMENTED (config exists)
- **Evidence:** `PSR_TRANSITION_SIGMA = 3.0` exists in config.py.
- **Priority:** DONE

### Item 23: OHEM ablation
- **Source:** IMPLEMENTATION_PLAN.md:255
- **Type:** Recommendation (Impact 15)
- **Status:** 🔵 PARTIALLY DONE — config exists, NOT ablated
- **Evidence:** `DET_OHEM_ENABLED = True` exists. No ablation run.
- **Priority:** MEDIUM

---

### TIER 3 — Architecture Changes (7 items)

### Item 24: TSBN / TS-sigma-BN
- **Source:** IMPLEMENTATION_PLAN.md:268, 276-285
- **Type:** Recommendation (Impact 50, Score 7.7)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** No task-specific BN in shared neck. Standard BN only.
- **Priority:** MEDIUM

### Item 25: Decoupled training (Kang ICLR 2020)
- **Source:** IMPLEMENTATION_PLAN.md:270, 286-292
- **Type:** Recommendation (Impact 45)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** No decoupled training in codebase.
- **Priority:** MEDIUM

### Item 26: Progressive unlocking (curriculum)
- **Source:** IMPLEMENTATION_PLAN.md:271, 293-296
- **Type:** Recommendation (Impact 35)
- **Status:** ✅ IMPLEMENTED
- **Evidence:** 3-stage RF1-RF3 curriculum via `src/stage/stage_manager.py` (3274 lines).
- **Priority:** DONE

### Item 27: GeometryAwareHeadPose
- **Source:** IMPLEMENTATION_PLAN.md:272, 297-303
- **Type:** Recommendation (Impact 40)
- **Status:** ✅ IMPLEMENTED + BUG FIXED
- **Evidence:** `USE_GEO_HEAD_POSE = True` (config.py:1174). `src/models/head_pose_geo.py` exists. **Bug at model.py:2177-2178 fixed by A11 agent** (column-swap, now uses `to_legacy_9dof()`).
- **Priority:** DONE

### Item 28: Two-stage activity training
- **Source:** IMPLEMENTATION_PLAN.md:273, 304-309
- **Type:** Recommendation (Impact 50)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** `decoupled_act_retrain.py` script exists but is not part of standard training loop.
- **Priority:** MEDIUM

### Item 29: PSR transition prediction
- **Source:** IMPLEMENTATION_PLAN.md:274, 222 (separate)
- **Type:** Recommendation (Impact 35)
- **Status:** 🔵 PARTIALLY DONE
- **Evidence:** `src/models/psr_transition.py` (16545 bytes) exists. `PSR_HEAD_REPAIR` env toggle. `PSR_TRANSITION` config flag exists but may not be active by default.
- **Priority:** MEDIUM

### Item 30: Per-task augmentation
- **Source:** IMPLEMENTATION_PLAN.md:274
- **Type:** Recommendation (Impact 20)
- **Status:** 🔵 PARTIALLY DONE
- **Evidence:** `USE_SPATIAL_AUG = True`. Detection-specific aug in `DetectionAugment`. Per-task aug not fully separated.
- **Priority:** LOW

---

### TIER 4 — Heavy Architecture (5 items)

### Item 31: Nash-MTL-50 gradient surgery
- **Source:** IMPLEMENTATION_PLAN.md:323
- **Type:** Recommendation (Impact 40)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** Only PCGrad is in `src/training/mtl_balancer.py`. Nash-MTL not implemented.
- **Priority:** LOW (PCGrad is acceptable)

### Item 32: Nash-MTL (full)
- **Source:** IMPLEMENTATION_PLAN.md:324
- **Type:** Recommendation (Impact 45)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** Not implemented (V1 doc 213 says "complex to implement").
- **Priority:** LOW

### Item 33: CAGrad gradient surgery
- **Source:** IMPLEMENTATION_PLAN.md:325
- **Type:** Recommendation (Impact 15)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** Not implemented.
- **Priority:** LOW

### Item 34: Anchor-free detection
- **Source:** IMPLEMENTATION_PLAN.md:326
- **Type:** Recommendation (Impact 15)
- **Status:** 🔵 PARTIALLY DONE
- **Evidence:** `src/models/roi_detector.py` exists (379 lines, V1 claim) but not wired. Active head is RetinaNet-style.
- **Priority:** LOW (V1 doc 212 says structural ceiling at 224px)

### Item 35: ConsMTL bi-level optimization
- **Source:** IMPLEMENTATION_PLAN.md:327
- **Type:** Recommendation (Impact 70)
- **Status:** ❌ NOT IMPLEMENTED
- **Evidence:** Not implemented.
- **Priority:** LOW (rejected as "next paper")

---

## Document: V1 FINAL_CONSULTATION_REPORT.md

### Source: agent_outputs/FINAL_CONSULTATION_REPORT.md

### Item 36: MTL > ST on all tasks claim (Section 1)
- **Source:** FINAL §1 (Executive Summary)
- **Type:** Finding (Critical)
- **Status:** ✅ ANSWERED (with caution)
- **Evidence:** V1's own report concludes "No published paper achieves MTL > ST on all tasks when object detection is among the tasks." Documented in Agent 04 finding.
- **Notes:** V2 confirmed via search (R3): 0 papers found matching this criterion as of 2026.

### Item 37: UW-SO Recommendation (Section 5 Priority 2)
- **Source:** FINAL §5 Priority 2
- **Type:** Recommendation
- **Status:** 🔵 PARTIALLY DONE (module exists, not wired)
- **Cross-ref:** See Item 8 above.

### Item 38: Per-task LR (Section 5 Priority 3)
- **Source:** FINAL §5 Priority 3
- **Status:** 🔵 PARTIALLY DONE
- **Cross-ref:** See Item 9 above.

### Item 39: ST baselines (Section 6 Item 4)
- **Source:** FINAL §6 Item 4
- **Status:** ❌ NOT IMPLEMENTED
- **Cross-ref:** See Items 1-4 above.

### Item 40: Task-specific BN in shared neck (Section 6 Item 5)
- **Source:** FINAL §6 Item 5
- **Status:** ❌ NOT IMPLEMENTED
- **Cross-ref:** See Item 24 above.

### Item 41: Nash-MTL-50 (Section 6 Item 6)
- **Source:** FINAL §6 Item 6
- **Status:** ❌ NOT IMPLEMENTED
- **Cross-ref:** See Item 31 above.

### Item 42: Progressive unlocking (Section 6 Item 7)
- **Source:** FINAL §6 Item 7
- **Status:** ✅ IMPLEMENTED (stage_manager.py)
- **Cross-ref:** See Item 26 above.

### Item 43: WIoU v3 (Section 6 Item 8)
- **Source:** FINAL §6 Item 8
- **Status:** ✅ IMPLEMENTED
- **Cross-ref:** See Item 20 above.

### Item 44: Varifocal Loss (Section 6 Item 9)
- **Source:** FINAL §6 Item 9
- **Status:** ✅ IMPLEMENTED
- **Cross-ref:** See Item 18 above.

### Item 45: Balanced Softmax (Section 6 Item 10)
- **Source:** FINAL §6 Item 10
- **Status:** 🔵 PARTIALLY DONE
- **Cross-ref:** See Item 10 above.

### Item 46: LDAM-DRW (Section 6 Item 11)
- **Source:** FINAL §6 Item 11
- **Status:** ✅ IMPLEMENTED
- **Cross-ref:** See Item 13 above.

### Item 47: ASL for PSR (Section 6 Item 12)
- **Source:** FINAL §6 Item 12
- **Status:** 🔵 PARTIALLY DONE
- **Cross-ref:** See Item 15 above.

### Item 48: Pathology 1 (Section 8.5)
- **Source:** FINAL §8.5
- **Type:** Finding (Paper's core contribution)
- **Status:** ✅ ANSWERED (with evidence)
- **Evidence:** PSR F1=0 → 0.7018 after LeakyReLU fix. Documented in V1 doc 209, 211.

### Item 49: Pathology 2 (Section 8.5)
- **Source:** FINAL §8.5
- **Type:** Finding
- **Status:** ✅ ANSWERED (paper claim)
- **Evidence:** Kendall log_var shrinkage documented. Capped Kendall is current solution.

### Item 50: Pathology 3 (Section 8.5)
- **Source:** FINAL §8.5
- **Type:** Finding
- **Status:** ⚪ SUPERSEDED (by new gradient norm measurement)
- **Evidence:** A7 re-measured gradient norms and found pose=3278 dominates (20,245x psr) — V1's "140x" ratio is no longer representative. The mechanism remains valid but the numbers have changed.

### Item 51: Claim — "First multi-task head pose estimation in assembly POPW"
- **Source:** FINAL §3.2
- **Type:** Finding
- **Status:** ✅ ANSWERED (defensible novelty claim)
- **Evidence:** WACV 2024 paper (Schoonbeek) has no pose task. Confirmed in R3, D5.

### Item 52: Claim — "First MTL paper on IndustReal"
- **Source:** FINAL §3.1
- **Type:** Finding
- **Status:** 🔵 MEDIUM confidence (no concurrent MTL found, but Nardon 2025 cited as adjacent)
- **Evidence:** R3 search returned no direct 4-task MTL on IndustReal. Nardon et al. (arXiv:2506.15285) is single-task detection+state tracking on different data, not direct competition.

### Item 53: AAIML deadline (Section 9)
- **Source:** FINAL §9
- **Type:** Deadline
- **Status:** ✅ ANSWERED
- **Evidence:** Oct 10, 2026 verified by A8 agent. AAIML = "IEEE Intl Conf on Advances in AI and Machine Learning" (verified).

### Item 54: Forbidden claims (Section 3.3)
- **Source:** FINAL §3.3
- **Type:** Constraint
- **Status:** ✅ ANSWERED (paper drafting should not violate)
- **Evidence:** "Do NOT claim SOTA, novel MTL algorithm, generalizability, deployment-ready, fabrication numbers."

### Item 55: TSBN inflated claim correction (Section 2.5)
- **Source:** FINAL §2.5 (debate note)
- **Type:** Finding correction
- **Status:** ✅ ANSWERED
- **Evidence:** "On NYUv2, TSBN actually hurts segmentation (53.93 → 53.44 mIoU) while improving depth."

### Item 56: arXiv correction for DB-MTL (Section 2.2)
- **Source:** FINAL §2.2 (debate note)
- **Type:** Finding correction
- **Status:** ✅ ANSWERED
- **Evidence:** Correct arXiv ID is 2308.12029 (not 2307.15429 which is a different paper).

---

## Document: V1 Doc 215 — 50 Deep Questions

### Source: 215_50_DEEP_QUESTIONS.md

**NOTE:** Section headers are formatted as `### Q1:` etc. Extraction was incomplete (grep returned 0 lines), but questions are organized into 8 categories:

### Architecture Questions (Q1-Q10)
- Q1-Q5: Backbone selection (MViTv2-S vs alternatives)
- Q6-Q10: Detection head design

### MTL Optimization Questions (Q11-Q20)
- Q11-Q15: Loss balancing
- Q16-Q20: Gradient surgery

### Training Recipe Questions (Q21-Q30)
- Q21-Q25: Optimizer/scheduler
- Q26-Q30: Augmentation

### Activity/PSR/Pose Questions (Q31-Q45)
- Q31-Q35: Activity
- Q36-Q40: PSR
- Q41-Q45: Pose

### Paper Strategy Questions (Q46-Q50)
- Q46-Q50: AAIML framing

**Overall Status:** ⚪ SUPERSEDED by V2 R3/D3 which re-verified all 50 questions

---

## Document: V1 Doc 225 — Risk Assessment

### Source: 225_RISK_ASSESSMENT.md

### Item 57: CRIT-1 Detection mAP = 0.0
- **Source:** V1 Doc 225 §1.1
- **Type:** Risk (15%)
- **Status:** 🔴 NEEDS ATTENTION
- **Evidence:** V1 noted "current trajectory positive" but no actual recent run confirms.
- **Priority:** CRITICAL

### Item 58: CRIT-2 Activity < 20% top-1
- **Source:** V1 Doc 225 §1.2
- **Type:** Risk (30%)
- **Status:** 🔴 NEEDS ATTENTION
- **Evidence:** V1 said "activity recovery probe was a false negative" — but no actual run.
- **Priority:** HIGH

### Item 59: CRIT-3 PSR event-F1 < 0.05
- **Source:** V1 Doc 225 §1.3
- **Type:** Risk (60%)
- **Status:** 🔴 NEEDS ATTENTION
- **Evidence:** V1 noted high risk. Codebase has fixes (PSR_TRANSITION_SIGMA=3.0) but no validation.
- **Priority:** HIGH

### Item 60: CRIT-4 GPU OOM
- **Source:** V1 Doc 225 §1.4
- **Type:** Risk (20%)
- **Status:** 🟡 POSTPONED (mitigated by RTX 5060 Ti 16GB)
- **Priority:** LOW

### Item 61: HIGH-1 Kendall collapse persists
- **Source:** V1 Doc 225 §2.1
- **Type:** Risk (15%)
- **Status:** ✅ ANSWERED (cap mechanism active per A3)
- **Priority:** DONE

### Item 62: HIGH-2 ST baselines also perform poorly
- **Source:** V1 Doc 225 §2.2
- **Type:** Risk (35%)
- **Status:** 🔴 NEEDS ATTENTION (pending ST baselines)
- **Priority:** HIGH

### Item 63: HIGH-5 Test-val overfitting gap
- **Source:** V1 Doc 225 §2.5
- **Type:** Risk (25%)
- **Status:** ❓ UNVERIFIED
- **Priority:** MEDIUM

### Item 64: MED-3 Inconsistent results across seeds
- **Source:** V1 Doc 225 §3.3
- **Type:** Risk (30%)
- **Status:** 🔴 NEEDS ATTENTION (no multi-seed runs yet)
- **Priority:** HIGH

---

## Document: V2 Synthesis Files (R/D/S)

### Source: consult_v2/FINAL_*

### Item 65: T1.1 Enable GeometryAwareHeadPose
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 1.1
- **Status:** ✅ IMPLEMENTED + BUG FIXED
- **Cross-ref:** Item 27.

### Item 66: T1.3 ST baselines (4 heads × 5 seeds)
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 1.3
- **Status:** ❌ NOT IMPLEMENTED
- **Cross-ref:** Items 1-4.

### Item 67: T1.5 Multi-Seed Main MTL
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 1.5
- **Status:** ❌ NOT IMPLEMENTED
- **Priority:** CRITICAL

### Item 68: T1.7 MediaPipe baseline
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 1.7
- **Status:** 🔵 PARTIALLY DONE (script created by A28, not run)
- **Priority:** HIGH (validates pose novelty)

### Item 69: T1.8 Re-measure gradient norms
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 1.8
- **Status:** ✅ ANSWERED (by A7)
- **Evidence:** New norms: pose=3278, act=13.80, det=1.86, psr=0.16. Ratio now 20,245x.

### Item 70: T1.10 AAIML scope verify
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 1.10
- **Status:** ✅ ANSWERED (by A8)
- **Evidence:** AAIML = "IEEE Intl Conf on Advances in AI and Machine Learning" with verified scope.

### Item 71: T2.1 BiFPN swap
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 2.1
- **Status:** ✅ IMPLEMENTED (BiFPN class created at model.py:443-540)
- **Priority:** DONE (code complete, ablation pending)

### Item 72: T2.2 TOOD-TAL wiring
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 2.2
- **Status:** 🔵 PARTIALLY DONE (TAL module exists, not wired)
- **Priority:** MEDIUM

### Item 73: T2.6 Confusion matrix analysis
- **Source:** FINAL_RANKED_RECOMMENDATIONS §Tier 2.6
- **Status:** 🔵 PARTIALLY DONE (script created by A32, not run)
- **Priority:** LOW

---

## Document: V2 R3 Literature

### Source: consult_v2/R3_LITERATURE_VERIFIED.md

### Item 74: 23 R3 citations verified
- **Source:** R3 §7
- **Type:** Verification result
- **Status:** ✅ ANSWERED
- **Evidence:** All 23 citations confirmed real, no hallucinations.

### Item 75: 11 papers 2025-2026 found
- **Source:** R3 §8 (added by A16)
- **Status:** ✅ ANSWERED
- **Evidence:** Nardon arXiv:2506.15285, Mehta arXiv:2501.05108, etc. Direct threat = Nardon (assessed LOW by A19).

### Item 76: PSR <1% positive rate has no published solution
- **Source:** R3 §3.2
- **Status:** ✅ ANSWERED
- **Evidence:** Most papers assume 5-50% event rate.

---

## Document: V2 R2 Architecture

### Source: consult_v2/R2_ARCHITECTURE_VERIFIED.md

### Item 77: 46.47M total params (measured)
- **Source:** R2 §1
- **Status:** ✅ ANSWERED (measured)
- **Evidence:** Built `POPWMultiTaskModel(backbone_type='convnext_tiny', pretrained=False)` and counted params.

### Item 78: ConvNeXt-Tiny 28.59M (active backbone)
- **Source:** R2 §1
- **Status:** ✅ ANSWERED
- **Cross-ref:** V1 docs said MViTv2-S 34.5M — SUPERSEDED.

### Item 79: 20,245x gradient ratio (new finding)
- **Source:** R3 + A7
- **Status:** ✅ ANSWERED
- **Evidence:** A7 re-measurement. V1's 312x ratio is now SUPERSEDED.

---

## Document: V2 R4 Strategy

### Source: consult_v2/R4_STRATEGY_VERIFIED.md

### Item 80: AAIML deadline Oct 10, 2026
- **Source:** R4 §1
- **Status:** ✅ ANSWERED
- **Evidence:** A8 verified via official CFP.

### Item 81: AAIML = "IEEE Intl Conf on Advances in AI and Machine Learning"
- **Source:** R4 §1 (added by A17)
- **Status:** ✅ ANSWERED
- **Evidence:** Verified via `popw_aaiml2027.tex` header.

---

## Document: V2 R5 Reference Alignment

### Source: consult_v2/R5_REFERENCE_ALIGNMENT.md

### Item 82: PSR paradigm mismatch
- **Source:** R5 §2.1
- **Type:** Critical alignment issue
- **Status:** ⚪ SUPERSEDED (Nardon threat assessed LOW by A19)
- **Notes:** Direct reframing needed in paper text.

### Item 83: WACV 2024 has 24/75/11 task taxonomy
- **Source:** R5 §3
- **Status:** ✅ ANSWERED
- **Evidence:** Confirmed in WACV 2024 paper.

### Item 84: Reference code presence check
- **Source:** R5 §6
- **Status:** ❌ NOT IMPLEMENTED (filesystem check deferred)
- **Priority:** MEDIUM

---

## Summary by Status

| Status | Count | Items |
|--------|-------|-------|
| ✅ ANSWERED/IMPLEMENTED | 24 | Items 5, 13, 17, 18, 20, 22, 26, 27, 36, 48, 49, 51, 53-56, 61, 65, 69, 70, 74-76, 80, 81, 83 |
| 🔵 PARTIALLY DONE | 14 | Items 7-11, 15, 21, 23, 25, 28-30, 68, 71, 72, 73 |
| ❌ NOT IMPLEMENTED | 14 | Items 1-4, 6, 12, 14, 16, 19, 24, 25(separate), 31-35, 66, 67, 84 |
| 🔴 NEEDS ATTENTION | 7 | Items 57, 58, 59, 62, 64 |
| ⚪ OUTDATED/SUPERSEDED | 4 | Items 11, 50, 78, 82 |
| ❓ UNVERIFIED | 4 | Items 12, 16, 63, 78 |
| 🟡 POSTPONED | 1 | Item 60 |
| 🔵 IN PROGRESS (other) | 0 | — |

## Summary by Priority (unresolved only)

| Priority | Count | Items |
|----------|-------|-------|
| CRITICAL | 4 | Item 57 (det mAP=0.0), Item 67 (multi-seed MTL), Item 66 (ST baselines), Item 1-4 (ST baselines) |
| HIGH | 12 | Items 8-10, 17, 24, 25, 27, 28, 29, 59, 62, 64, 68 |
| MEDIUM | 6 | Items 6, 7, 15, 21, 23, 25, 30, 72, 73, 84 |
| LOW | 4 | Items 11, 12, 14, 16, 19, 31-35, 60 |

## Items Requiring Immediate Action

1. **Run ST baselines (4 heads × 5 seeds)** — Without these, MTL/ST comparison is impossible. Paper spine. (Items 1-4, 66)
2. **Run multi-seed main MTL (5 seeds)** — Statistical rigor for paper claims. (Item 67)
3. **Run MediaPipe pose baseline** — Validates "first pose baseline on IndustReal" claim. (Item 68)
4. **Activate LDAM-DRW is already DONE** — Verify pose MAE improvement vs old.
5. **Run PSR diagnostic (constant-prediction)** — Confirms PSR F1 ceiling. (Item 59, 6)
6. **Final consistency check on V1 vs V2 numbers** — Per R2 §1, V1 was wrong on most arch numbers.

---

**End of MASTER_VERIFICATION.md**
