# FINAL RANKED RECOMMENDATIONS — ULTIMATE Consultation V2

**Phase:** ULTIMATE Consultation V2 — Phase 3 Final Synthesis (Synthesizer S2)
**Date:** 2026-07-14
**Author:** Synthesizer S2
**Inputs:** FINAL_VERIFIED_FINDINGS.md + all R/D outputs + 20 V2 agent outputs

---

## Executive Summary

Recommendations ranked by **Impact on AAIML Acceptance** ÷ **Implementation Cost**. Tier 1 (must do), Tier 2 (should do), Tier 3 (if time), Rejected (with evidence).

**Total recommendations:** 20 across all categories.

---

## Tier 1 — MUST IMPLEMENT (10 items)

### T1.1 — Enable GeometryAwareHeadPose for Pose Head
- **Impact:** HIGH (30-50% MAE reduction per Zhou et al. CVPR 2019)
- **Cost:** LOW (0.5 day to enable, 50-epoch training ~10 GPU-hours)
- **Risk:** LOW (module exists, just toggle env flag)
- **Evidence:** R2 (module exists), D7 (currently disabled)
- **Steps:**
  1. Set `USE_GEO_HEAD_POSE=1` env var
  2. Run 100-epoch training
  3. Compare MAE: 9-DoF MSE vs 6D rotation + geodesic
- **Expected outcome:** Pose MAE 8.7° → 5-6°

### T1.2 — Wire Distillation Module
- **Impact:** HIGH (+1-3% all heads per V1 doc 211)
- **Cost:** MEDIUM (1-2 days: verify wiring, train ST teachers, integrate)
- **Risk:** MEDIUM (depends on ST teacher quality)
- **Evidence:** R2 (module exists), D7 (status uncertain)
- **Steps:**
  1. Grep `train.py` for `distill-teacher-dir` to verify wiring
  2. Train ST teachers (4 heads, 5 seeds each, ~100 GPU-hours)
  3. Enable distillation in main MTL run
- **Expected outcome:** Activity +2-3%, detection +1-2%

### T1.3 — Complete ST Baselines (4 heads × 5 seeds)
- **Impact:** HIGH (mandatory for MTL/ST comparison)
- **Cost:** MEDIUM (~100-150 GPU-hours)
- **Risk:** LOW
- **Evidence:** R1-R2, V2 agent05-09
- **Steps:**
  1. ST pose baseline (RTX 3060, 50 epochs × 5 seeds = 17.5 hours)
  2. ST detection baseline (RTX 3060, 50 epochs × 5 seeds = 35 hours)
  3. ST PSR baseline (RTX 3060, 50 epochs × 5 seeds = 25 hours)
  4. ST activity baseline (RTX 3060, 50 epochs × 5 seeds = 25 hours)
- **Expected outcome:** Establishes fair ST ceiling for all 4 heads

### T1.4 — Verify LDAM-DRW + Long-Tail Activity
- **Impact:** HIGH (+5-10% tail class recall for activity)
- **Cost:** MEDIUM (2 days)
- **Risk:** MEDIUM (sensitivity to schedule)
- **Evidence:** D7 (module exists, status uncertain)
- **Steps:**
  1. Grep for LDAM-DRW wiring
  2. If not wired: implement LDAM-DRW schedule
  3. Run 100-epoch ablation
- **Expected outcome:** Activity top-1 +2-5%

### T1.5 — Run Multi-Seed (5) Main MTL
- **Impact:** HIGH (mandatory for statistical rigor)
- **Cost:** HIGH (~250-300 GPU-hours on RTX 5060 Ti)
- **Risk:** MEDIUM (timing risk)
- **Evidence:** R4, V1 doc 223
- **Steps:**
  1. Freeze architecture by Day 14
  2. Run 5 seeds × 100 epochs
  3. Compute bootstrap CIs
- **Expected outcome:** 5-seed mean + 95% CI per head

### T1.6 — Uncapped Kendall Ablation
- **Impact:** MEDIUM (validates our cap configuration)
- **Cost:** LOW (1 run, ~50 GPU-hours)
- **Risk:** LOW
- **Evidence:** D8 (reproducibility on V2 codebase unverified)
- **Steps:**
  1. Set `KENDALL_LOG_VAR_MIN_ACT=-4.0`, `KENDALL_LOG_VAR_MAX_PSR=2.0` etc.
  2. Run 100-epoch ablation
  3. Compare log-var trajectories
- **Expected outcome:** Validates or refutes Kendall collapse hypothesis

### T1.7 — Run MediaPipe Pose Baseline
- **Impact:** HIGH (validates pose novelty claim)
- **Cost:** LOW (1-2 days)
- **Risk:** LOW
- **Evidence:** D4, D5 (off-the-shelf comparison missing)
- **Steps:**
  1. Run MediaPipe Face Mesh on test set
  2. Compute MAE vs HL2 ground truth
  3. Compare to our head pose MAE
- **Expected outcome:** Establishes pose baseline comparison

### T1.8 — Re-measure Gradient Norms
- **Impact:** MEDIUM (validates 312x ratio claim)
- **Cost:** LOW (1 day)
- **Risk:** LOW
- **Evidence:** D8 (V1 measurement on old codebase)
- **Steps:**
  1. Run `e8_gradient_diagnostic.py` on current model
  2. Compute per-task gradient norms at epochs 5, 25, 50
  3. Compare to V1's 312x ratio
- **Expected outcome:** Refreshed gradient starvation evidence

### T1.9 — Verify All Module Wiring
- **Impact:** MEDIUM (catches unused Tier 1 modules)
- **Cost:** LOW (1 day)
- **Risk:** LOW
- **Evidence:** D7, D8 (modules exist, wiring status uncertain)
- **Steps:**
  1. Grep `train.py` for each of: distillation, FAMO, RotoGrad, MetaBalance, LDAM-DRW, IMTL-L, TAL
  2. Document which are wired, which exist-only
  3. Wire any Tier 1 ones that exist but are unused
- **Expected outcome:** Clean module status report

### T1.10 — AAIML Scope Verification
- **Impact:** MEDIUM (avoids venue mismatch)
- **Cost:** LOW (0.5 day)
- **Risk:** LOW
- **Evidence:** D4, D9 (AAIML scope unverified)
- **Steps:**
  1. Search AAIML 2024, 2025, 2026 proceedings
  2. Verify topic alignment with industrial AI / MTL / vision
  3. If misaligned: consider WACV or ICRA as alternatives
- **Expected outcome:** Confirmed venue or pivot decision

---

## Tier 2 — SHOULD IMPLEMENT (6 items)

### T2.1 — BiFPN Swap Ablation
- **Impact:** MEDIUM (+0.4-0.7 mAP per Tan et al. CVPR 2020)
- **Cost:** MEDIUM (1-2 days implementation + 100-epoch run)
- **Risk:** LOW
- **Evidence:** D2, D7 (BiFPN legacy code exists in mvit_mtl_model.py)
- **Steps:**
  1. Port BiFPN from `mvit_mtl_model.py` to `model.py`
  2. Run 100-epoch comparison vs standard FPN
- **Expected outcome:** +0.4-0.7 mAP if gains transfer

### T2.2 — TOOD-TAL Wiring
- **Impact:** MEDIUM (+3-5 mAP per Wang et al. ICCV 2021)
- **Cost:** MEDIUM (2-3 days wiring + verification)
- **Risk:** MEDIUM (integration risk)
- **Evidence:** D2, D7, Task #226 (module exists)
- **Steps:**
  1. Smoke test TAL module
  2. Wire into `DetectionHead`
  3. Run 100-epoch comparison
- **Expected outcome:** +3-5 mAP if integration clean

### T2.3 — Detection at 480×480 Resolution
- **Impact:** MEDIUM (resolution-bound small object detection)
- **Cost:** MEDIUM (1 day + VRAM check)
- **Risk:** MEDIUM (may OOM)
- **Evidence:** R5, D2 (resolution ablation missing)
- **Steps:**
  1. Enable `USE_BACKBONE_CHECKPOINT=True`
  2. Run at 480×480, batch=2
  3. Compare to 224×224
- **Expected outcome:** +2-5 mAP if VRAM permits

### T2.4 — Anchor-Free Detection (YOLOX-style)
- **Impact:** MEDIUM (+4.3 mAP per Ge et al. 2021)
- **Cost:** HIGH (5 days implementation)
- **Risk:** HIGH (rewrite detection head)
- **Evidence:** D2, D7
- **Steps:**
  1. Implement anchor-free `DetectionHead` from scratch
  2. Add simOTA assignment
  3. Run 100-epoch comparison
- **Expected outcome:** +4 mAP if successful

### T2.5 — 2025-2026 Literature Search
- **Impact:** MEDIUM (catches missed papers)
- **Cost:** LOW (1-2 days)
- **Risk:** LOW
- **Evidence:** D3, D8 (search was incomplete)
- **Steps:**
  1. Systematic arXiv search with date filter 2025-2026
  2. Check for: "IndustReal multi-task", "industrial MTL video", "head pose MTL"
  3. Update R3 with any findings
- **Expected outcome:** Comprehensive 2025-2026 coverage

### T2.6 — Confusion Matrix Analysis for Activity
- **Impact:** LOW-MEDIUM (reveals failure modes)
- **Cost:** LOW (0.5 day)
- **Risk:** LOW
- **Evidence:** D6 (action-level confusion analysis missing)
- **Steps:**
  1. Run inference on test set
  2. Compute per-class confusion matrix
  3. Identify top confusion pairs
- **Expected outcome:** Insight into class confusability

---

## Tier 3 — IF TIME/COMPUTE PERMITS (4 items)

### T3.1 — Enable VideoMAE Stream
- **Impact:** MEDIUM (+5-7% activity top-1 per V1 doc 01 §B.1)
- **Cost:** HIGH (3 days + +22M frozen params + +600MB VRAM)
- **Risk:** MEDIUM (may OOM on RTX 3060)
- **Evidence:** R2, config.py:154
- **Steps:**
  1. Set `USE_VIDEOMAE=True`
  2. Run on RTX 5060 Ti (16GB)
  3. Compare activity top-1
- **Expected outcome:** +5-7% activity top-1 if VRAM permits

### T3.2 — Activate FAMO + RotoGrad + MetaBalance
- **Impact:** LOW-MEDIUM (per-task MTL optimization)
- **Cost:** MEDIUM (2-3 days wiring)
- **Risk:** MEDIUM (each method has documented failure modes per D8)
- **Evidence:** R2 (modules exist, not wired)
- **Steps:**
  1. Wire FAMO (CVPR 2023)
  2. Wire RotoGrad (ICML 2022)
  3. Wire MetaBalance (WWW 2022)
  4. Compare each vs Kendall+PCGrad baseline
- **Expected outcome:** +1-2% per head if wiring clean

### T3.3 — CAGrad Comparison Ablation
- **Impact:** LOW (alternative gradient surgery)
- **Cost:** MEDIUM (3 days)
- **Risk:** MEDIUM (large batch requirement)
- **Evidence:** D3 (CAGrad module not implemented)
- **Steps:**
  1. Implement CAGrad in `mtl_balancer.py`
  2. Run with batch=48 (our setup OK per D8)
  3. Compare vs PCGrad
- **Expected outcome:** +1-2% per head if implementation correct

### T3.4 — Cloud GPU Backup Run
- **Impact:** LOW (insurance against local failure)
- **Cost:** MEDIUM ($200-500 for RunPod/Lambda)
- **Risk:** LOW
- **Evidence:** D9 (hardware failure risk)
- **Steps:**
  1. Set up RunPod/Lambda account
  2. Run final 5-seed MTL on cloud as backup
- **Expected outcome:** Backup if local GPU fails

---

## Rejected (with evidence)

### R1 — Drop Pose Head Entirely
- **Evidence:** Pose is novel on IndustReal (no WACV 2024 baseline)
- **Why rejected:** Pose novelty is one of our strongest claims. Dropping it weakens the paper.

### R2 — Replace ConvNeXt with MViTv2-S
- **Evidence:** MViTv2-S has K400 pretraining, likely +10-15% activity
- **Why rejected:** V2 codebase has committed to convnext_tiny. Swap is too risky on timeline.

### R3 — Switch from Kendall to Equal Weights
- **Evidence:** Kendall can fail (per D8)
- **Why rejected:** Our per-task cap configuration is novel. Switching loses this contribution.

### R4 — Drop Activity Head (Class 0 issue)
- **Evidence:** D1 raised idle-period concern about class 0
- **Why rejected:** Activity is a primary task. The class 0 issue is minor and can be reframed.

### R5 — Use 3-task MTL (drop PSR)
- **Evidence:** PSR F1 may stay near 0
- **Why rejected:** PSR with <0.5% positive rate is novel problem. Document as negative result if it fails.

### R6 — Implement B3-style transition detection
- **Evidence:** Reference code (WACV 2024) uses transition-based PSR
- **Why rejected:** Our per-frame approach is novel framing. Mixing paradigms complicates the story.

### R7 — Submit to WACV instead of AAIML
- **Evidence:** AAIML scope unverified
- **Why rejected:** WACV is a possibility (per D9) but AAIML is more aligned with industrial AI. Verify AAIML first (T1.10).

---

## Priority Matrix

| Rank | Recommendation | Impact | Cost | Effort |
|---|---|---|---|---|
| 1 | T1.1 Enable GeoHeadPose | HIGH | LOW | 0.5d |
| 2 | T1.3 ST Baselines (5 seeds) | HIGH | MEDIUM | 5d |
| 3 | T1.5 Multi-Seed Main MTL | HIGH | HIGH | 5d |
| 4 | T1.7 MediaPipe Baseline | HIGH | LOW | 2d |
| 5 | T1.10 AAIML Scope Verify | MEDIUM | LOW | 0.5d |
| 6 | T1.4 LDAM-DRW Activity | HIGH | MEDIUM | 2d |
| 7 | T1.2 Wire Distillation | HIGH | MEDIUM | 2d |
| 8 | T1.9 Verify Module Wiring | MEDIUM | LOW | 1d |
| 9 | T1.8 Re-measure Gradients | MEDIUM | LOW | 1d |
| 10 | T1.6 Uncapped Kendall | MEDIUM | LOW | 1d |
| 11 | T2.1 BiFPN Swap | MEDIUM | MEDIUM | 2d |
| 12 | T2.2 TOOD-TAL Wiring | MEDIUM | MEDIUM | 3d |
| 13 | T2.5 2025-2026 Lit Search | MEDIUM | LOW | 2d |
| 14 | T2.3 480px Resolution | MEDIUM | MEDIUM | 1d |
| 15 | T2.6 Confusion Matrix | LOW | LOW | 0.5d |
| 16 | T2.4 Anchor-Free Detection | MEDIUM | HIGH | 5d |
| 17 | T3.1 VideoMAE Stream | MEDIUM | HIGH | 3d |
| 18 | T3.4 Cloud Backup | LOW | MEDIUM | 1d |
| 19 | T3.2 FAMO/RotoGrad/MB | LOW | MEDIUM | 3d |
| 20 | T3.3 CAGrad Ablation | LOW | MEDIUM | 3d |

---

## Summary by Category

| Category | Tier 1 | Tier 2 | Tier 3 | Rejected |
|---|---|---|---|---|
| Data | T1.7, T1.10 | T2.5, T2.6 | — | R4 |
| Architecture | T1.1 | T2.1, T2.2, T2.3, T2.4 | T3.1 | R2 |
| Training | T1.2, T1.3, T1.4, T1.5, T1.6, T1.8, T1.9 | — | T3.2, T3.3 | R3 |
| Pose | T1.1 | T2.6 | — | R1 |
| Strategy | T1.10 | T2.5 | T3.4 | R7 |
| Detection | — | T2.1, T2.2, T2.3, T2.4 | — | — |

---

## Output

This file is the ranked recommendations index. S3 (implementation plan) should sequence Tier 1 by day, S4 (paper framework) should include these as the contribution list, and S5 (Claude Science queries) should cover the open questions.
