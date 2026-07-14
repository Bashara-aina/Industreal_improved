# Implementation Plan: Ranked Recommendations from Claude Science

**Date:** 2026-07-11
**Source synthesis:** 10 Agent Discovery Reports + FINAL_CONSULTATION_REPORT + Doc 226 (Execution Roadmap) + Doc 212 (Per-Head Gap Analysis) + Doc 225 (Risk Assessment)

---

## Scoring Methodology

Each recommendation is scored using the user-specified formula:

```
Priority Score = Impact / (Effort_Hours + Compute_GPU_Hours)
```

**Impact** is a 0-100 numeric score reflecting the expected improvement on the paper-critical metrics. Based on doc 226's 5-point scale:
- 80-100: Solves a critical failure (activity collapse, PSR F1=0, detection starvation)
- 50-79: Major measurable gain on a critical head (+5-15% metric improvement)
- 30-49: Substantial gain on a task (+2-5%), or eliminates a known failure mode
- 15-29: Moderate gain (+1-2%), or high diagnostic value
- 1-14: Small improvement or diagnostic only

**Effort_Hours** = person-hours of coding, debugging, testing.
**Compute_GPU_Hours** = GPU-hours for validation (probes, overfit tests). Main training runs are budgeted separately in doc 226.

**Key caveat:** The formula penalizes GPU-intensive items heavily. Zero-GPU-cost code changes score highest, which is intentional -- they should be done first. Items requiring long training runs (ST baselines, full Nash-MTL validation) score lower but are still mandatory -- their priority reflects "implement order" within their phase, not importance.

---

## TIER 0: PREREQUISITES (Already Planned in Doc 226, Not Scored)

These runs are the paper's spine. They must execute regardless of other priorities. Compute and effort are already budgeted in doc 226.

| Item | Effort (h) | GPU-h | Purpose | Doc 226 Phase |
|------|-----------|-------|---------|--------------|
| ST pose baseline (5 seeds) | 0 | 17.5 | Paper's anchor MTL/ST ratio | Ph1 |
| ST detection baseline (5 seeds) | 0 | 35 | Diagnostic: ST ceiling vs MTL ceiling | Ph1 |
| ST PSR baseline (5 seeds) | 0 | 25 | Diagnostic: architectural vs MTL failure | Ph1 |
| ST activity baseline (5 seeds) | 0 | 25 | Diagnostic: data ceiling vs MTL collapse | Ph1 |
| Activity fixed-backbone probe | 1 | 6 | Verify head can learn at all | Ph1 |
| PSR diagnostics (constant-prediction) | 1 | 8 | Verify eval pipeline, Gaussian targets | Ph1 |
| Infrastructure hardening | 8 | 0 | Eval uncap, LIVENESS_GRAD, log-var logging, seeds.csv | Ph1 |

**Total prerequisite GPU-hours:** ~116.5 (already allocated in doc 226's 165 GPU-h Phase 1 budget)

---

## TIER 1: IMMEDIATE CODE CHANGES (Zero GPU Cost, Highest ROI)

Implement these while ST baselines train. They change only config/hyperparameters and take effect before the main MTL training run.

### 1. UW-SO Loss Weighting (Replace Kendall UW)

| Field | Value |
|-------|-------|
| **Impact** | 85 -- Eliminates the weight collapse pathology entirely. No more learnable log-var parameters that shrink during training. UW-SO (Kirchdorfer, IJCV 2025) validated across NYUv2, Cityscapes, CelebA. Direct drop-in replacement. |
| **Effort** | 1.5 hours |
| **GPU cost** | 0 (no probe needed; run alongside ST baselines) |
| **Expected improvement** | +3-5% on detection metrics (removes starvation effect). +1-4% Delta m overall (per UW-SO published results). |
| **Risk of not working** | VERY LOW. UW-SO is published in IJCV 2025 with 3-benchmark validation. Our situation (heterogeneous loss scales) is exactly its design target. |
| **Priority Score** | **85 / 1.5 = 56.7** |
| **Source** | Agent 02; FINAL_CONSULTATION Section 5, Priority 2 |
| **Implementation** | `weights = F.softmax(-detach(losses) / temperature, dim=0)`. Delete the 4 learnable log-var params. Remove caps. Add temperature T (start at 1.0). ~10 lines changed. |
| **Rollback** | Keep `--loss-weighting kendall-uncapped` flag for ablation |

### 2. Per-Task Learning Rates

| Field | Value |
|-------|-------|
| **Impact** | 45 -- Regression heads (PSR, pose) have naturally larger gradient magnitudes. AdaTask (AAAI 2023) and GradNorm (ICML 2018) both confirm lower LR for regression prevents destabilization. |
| **Effort** | 1 hour |
| **GPU cost** | 0 |
| **Expected improvement** | +2-4% on PSR/pose metrics. Could help pose break its 9 deg plateau. |
| **Risk of not working** | VERY LOW. Hparam change, backed by published theory. At worst, no effect. |
| **Priority Score** | **45 / 1 = 45** |
| **Source** | Agent 08 (G1 finding); FINAL_CONSULTATION Section 5, Priority 3 |
| **Implementation** | Backbone LR=1e-4. Detection=1x. Activity=1x. PSR=0.3x. Pose=0.3x. One line per optimizer group. |

### 3. Balanced Softmax for Activity Head

| Field | Value |
|-------|-------|
| **Impact** | 30 -- Replaces hand-tuned CE+logit_adj+sqrt_tamed_weights with a principled loss that handles long-tail by construction. No hand-tuned hyperparameters. |
| **Effort** | 1 hour |
| **GPU cost** | 0 |
| **Expected improvement** | +2-5% top-1 on activity (based on long-tail benchmarks). More importantly, eliminates the fragile tuning stack (logit_adj, sqrt weights, tau) that has 3 interdependent knobs. |
| **Risk of not working** | LOW. Balanced Softmax is well-established for long-tail classification. Our 75-class power-law distribution is its target regime. |
| **Priority Score** | **30 / 1 = 30** |
| **Source** | Agent 10; Agent 07 |
| **Implementation** | Replace `logit_adj + F.cross_entropy(weight=sqrt_tamed)` with `BalancedSoftmaxLoss(pi=class_priors)` |

### 4. Gradient Clipping (max_norm=1.0)

| Field | Value |
|-------|-------|
| **Impact** | 15 -- Prevents gradient explosion from heterogeneous task gradients. Standard practice in MTL with 4+ tasks. |
| **Effort** | 0.5 hours |
| **GPU cost** | 0 |
| **Expected improvement** | Stability. Prevents divergence when adding new heads. Reduces NaN risk. |
| **Risk of not working** | NEGLIGIBLE. Standard practice, no downside when set to 1.0. |
| **Priority Score** | **15 / 0.5 = 30** |
| **Source** | Agent 08 (G4 recommendation, rank 7) |
| **Implementation** | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before optimizer step |

### 5. EMA Warmup (Start at Epoch 5)

| Field | Value |
|-------|-------|
| **Impact** | 12 -- Early EMA averages noisy initial weights. Starting at epoch 5 gives the model 10% of training to stabilize before averaging begins. |
| **Effort** | 0.5 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +0.5-1% on final metrics |
| **Risk of not working** | NEGLIGIBLE. Standard practice. |
| **Priority Score** | **12 / 0.5 = 24** |
| **Source** | Agent 08 (G6 finding 1) |
| **Implementation** | Set `ema_start_epoch=5` instead of `ema_start_epoch=0` |

### 6. LDAM-DRW for Activity Head

| Field | Value |
|-------|-------|
| **Impact** | 60 -- LDAM (Cao et al., NeurIPS 2019) gave 10.86% absolute top-1 improvement on iNaturalist 2018. Our tail classes (16 classes with <10 samples) are exactly the LDAM target regime. DRW schedule activates re-weighting after LR drop, aligning naturally with MTL training. |
| **Effort** | 3 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +3-10% top-1 on activity (published: 10.86% on iNaturalist, 5.42% on CIFAR-100-LT). Tail class recall specifically improves due to per-class margin adjustment. |
| **Risk of not working** | LOW-MEDIUM. LDAM is published NeurIPS 2019, well-cited. The DRW schedule may need tuning for our 50-epoch MTL schedule vs. the 200-epoch image classification schedule in the paper. |
| **Priority Score** | **60 / 3 = 20** |
| **Source** | Agent 07 (Paper D2) |
| **Implementation** | Replace CE logits with `logits - delta_y * margin_y` where `margin_y = C / n_y^{1/4}`. DRW: switch from vanilla CE to re-weighted after LR drop (epoch 35 in 50-epoch schedule). |

### 7. SWA Window Expansion (5 -> 10 Epochs)

| Field | Value |
|-------|-------|
| **Impact** | 8 -- SWA over 10 epochs captures wider optima than 5. Published pattern (Model Fusion survey, 2023). |
| **Effort** | 0.5 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +0.3-0.5% |
| **Risk of not working** | NEGLIGIBLE. |
| **Priority Score** | **8 / 0.5 = 16** |
| **Source** | Agent 08 (G6 finding 2) |
| **Implementation** | Set `SWA_WINDOW=10` instead of `SWA_WINDOW=5` |

### 8. ASL (Asymmetric Loss) for PSR

| Field | Value |
|-------|-------|
| **Impact** | 35 -- Replaces BCE+focal which produces "predict all zeros" failure mode. ASL hard-thresholds negative gradients so ultra-easy negatives (99.5% of frames) contribute nothing. Directly targets the constant-prediction pathology. |
| **Effort** | 2.5 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +1-3% PSR F1 on the binary components. More importantly, eliminates the "predict all zeros" local minimum that the current BCE+focal falls into. |
| **Risk of not working** | LOW. ASL is published (CVPR 2020 oral, 2000+ citations). The hard-threshold mechanism directly addresses the >99.5% negative rate problem. |
| **Priority Score** | **35 / 2.5 = 14** |
| **Source** | Agent 10; Agent 07 (Paper E6); FINAL_CONSULTATION Section 2.10 |
| **Implementation** | Replace BCE focal with `ASL(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)`. Remove sensitivity penalty cap increase needed. |

### 9. Task Head Dropout (PSR and Pose Heads)

| Field | Value |
|-------|-------|
| **Impact** | 15 -- Dropout (0.1-0.2) on task-specific MLP heads prevents overfitting on smaller tasks. |
| **Effort** | 1.5 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +0.5-2% on PSR and pose metrics |
| **Risk of not working** | LOW. Standard regularization. |
| **Priority Score** | **15 / 1.5 = 10** |
| **Source** | Agent 08 (G2, rank 6) |
| **Implementation** | Add `nn.Dropout(0.15)` after the hidden layer in PSR and pose MLP heads |

### 10. Huberised Geodesic Loss for Pose

| Field | Value |
|-------|-------|
| **Impact** | 25 -- Replaces standard geodesic loss with Huberised variant that caps outlier gradients from extreme pose errors. Our 9.13 deg MAE may be pulled up by a few hard samples. |
| **Effort** | 3 hours |
| **GPU cost** | 0 |
| **Expected improvement** | -1 to -3 deg MAE on pose (by reducing impact of extreme outliers) |
| **Risk of not working** | LOW. Standard robust loss technique. |
| **Priority Score** | **25 / 3 = 8.3** |
| **Source** | Agent 09; Agent 10 |
| **Implementation** | `hinge_l1 = where(geodesic_error < delta, 0.5*e^2, delta*(e - 0.5*delta))` with delta=30 degrees |

### 11. Varifocal Loss for Detection Classification

| Field | Value |
|-------|-------|
| **Impact** | 25 -- Replaces Focal Loss with IoU-aware asymmetric loss. Positive gradients weighted by IoU quality. CVPR 2021 Oral, +2.0 AP on COCO. |
| **Effort** | 3 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +1-2 AP detection |
| **Risk of not working** | LOW. Published, peer-reviewed. |
| **Priority Score** | **25 / 3 = 8.3** |
| **Source** | Agent 10 |
| **Implementation** | `VFL(pred, target) = target * ( -log(sigmoid(pred)) * (1 - sigmoid(pred))^gamma ) + (1 - target) * ( -log(1 - sigmoid(pred)) * sigmoid(pred)^gamma )` |

### 12. DB-MTL (Log-Transform Loss Scale Normalization)

| Field | Value |
|-------|-------|
| **Impact** | 25 -- Log-transform normalizes the loss scale mismatch (detection ~2 vs activity ~5). Secondary to UW-SO; can be stacked. |
| **Effort** | 3 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +1-3% overall. Directly addresses the 2.5x loss scale gap. |
| **Risk of not working** | LOW. Published in Neural Networks 2025. |
| **Priority Score** | **25 / 3 = 8.3** |
| **Source** | Agent 02; FINAL_CONSULTATION Section 2.2 |
| **Implementation** | Apply `log(1 + loss)` transform to each task loss before UW-SO weighting |

### 13. WIoU v3 for Detection Box Regression

| Field | Value |
|-------|-------|
| **Impact** | 15 -- Replaces CIoU's static penalty with dynamic non-monotonic focusing. Reduces harmful gradients from very-low-quality anchors. |
| **Effort** | 3 hours |
| **GPU cost** | 0 |
| **Expected improvement** | +1-2 AP detection |
| **Risk of not working** | LOW. Published, validated on COCO with YOLOv7. |
| **Priority Score** | **15 / 3 = 5** |
| **Source** | Agent 10 |
| **Implementation** | Replace CIoU with WIoU v3 formulation from Tong et al. 2023 |

### Tier 1 Summary (Implementation Order)

| Rank | Item | Score | Effort (h) | GPU-h | Cum. Effort |
|------|------|-------|-----------|-------|-------------|
| 1 | UW-SO loss weighting | 56.7 | 1.5 | 0 | 1.5 |
| 2 | Per-task LR | 45.0 | 1.0 | 0 | 2.5 |
| 3 | Balanced Softmax (activity) | 30.0 | 1.0 | 0 | 3.5 |
| 4 | Gradient clipping | 30.0 | 0.5 | 0 | 4.0 |
| 5 | EMA warmup (epoch 5) | 24.0 | 0.5 | 0 | 4.5 |
| 6 | LDAM-DRW (activity) | 20.0 | 3.0 | 0 | 7.5 |
| 7 | SWA window 5->10 | 16.0 | 0.5 | 0 | 8.0 |
| 8 | ASL (PSR) | 14.0 | 2.5 | 0 | 10.5 |
| 9 | Task head dropout | 10.0 | 1.5 | 0 | 12.0 |
| 10 | Huberised geodesic (pose) | 8.3 | 3.0 | 0 | 15.0 |
| 11 | Varifocal Loss (det cls) | 8.3 | 3.0 | 0 | 18.0 |
| 12 | DB-MTL log-transform | 8.3 | 3.0 | 0 | 21.0 |
| 13 | WIoU v3 (det box) | 5.0 | 3.0 | 0 | 24.0 |
| | **TOTAL** | | **24 hours** | **0 GPU-h** | |

**All 13 items can be implemented in 3 person-days while ST baselines train.**

---

## TIER 2: QUICK CONFIG CHANGES (Need GPU Validation)

These are config-flag changes or short probes that need 1-3 seeds of validation. They are already in doc 226's Phase 1 plan.

| Rank | Item | Impact | Effort (h) | GPU-h | Score | Source |
|------|------|--------|-----------|-------|-------|--------|
| 14 | **Mosaic augmentation enablement** | 30 | 0 | 24 | 1.25 | Doc 226 Ph1 |
| 15 | **Gaussian-smeared PSR targets** | 15 | 2 | 2 | 7.5 | Doc 212, Agent 07 |
| 16 | **OHEM ablation** | 15 | 0 | 24 | 0.63 | Doc 226 Ph1 |

**Notes:**
- **Mosaic augmentation** (+3-5 mAP published): Already implemented but never activated. Config flag only. Runs as part of doc 226's detection quick fixes. Expected +3-5 mAP from published results.
- **Gaussian-smeared PSR targets** (+0.05-0.15 F1): Smear hard binary transition labels with Gaussian kernel (sigma=2 frames). Gives gradient signal around transition boundaries. Can be validated in 1-2 GPU-hours with a short probe.
- **OHEM ablation** (+0.05-0.10 mAP): Disable OHEM_RATIO=2.0, run 3 seeds. Already in doc 226's plan. Diagnostic even if mAP doesn't improve (confirms bottleneck is elsewhere).

---

## TIER 3: MEDIUM-EFFORT ARCHITECTURE CHANGES (Phase 2 of Doc 226)

These require 4-12 hours of implementation plus GPU probes. They change the model architecture or training loop.

| Rank | Item | Impact | Effort (h) | GPU-h | Score | Source |
|------|------|--------|-----------|-------|-------|--------|
| 17 | **TSBN / TS-sigma-BN** | 50 | 6 | 5 | 7.7 | Agent 03, 05, 06 |
| 18 | **Decoupled training (Kang ICLR 2020)** | 45 | 6 | 0 | 7.5 | Agent 07 |
| 19 | **Progressive unlocking (curriculum)** | 35 | 6 | 0 | 5.8 | Agent 08 |
| 20 | **GeometryAwareHeadPose** | 40 | 6 | 10 | 5.7 | Agent 09 |
| 21 | **Two-stage activity training** | 50 | 8 | 35 | 4.3 | Doc 212, Agent 07 |
| 22 | **PSR transition prediction** | 35 | 12 | 40 | 2.2 | Doc 212, Agent 07 |
| 23 | **Per-task augmentation** | 20 | 12 | 0 | 1.7 | Agent 08 |

### 17. TSBN / TS-sigma-BN (Priority: DO FIRST among Tier 3)

**What:** Replace shared BatchNorm affine parameters with per-task affine parameters in the BiFPN neck. TSBN recovers ~75% of detection mAP gap at near-zero parameter overhead (~0.06% per task). Only the gamma/beta parameters are per-task; the running statistics remain shared.

**Evidence level:** HIGH. Confirmed by Agents 03, 05, and 06 independently. Published in multiple papers. TSBN achieves 98-100% ST retention on NYUv2.

**Implementation:** 6 hours. Modify the BiFPN's BN layers to maintain K sets of affine params (K=4 tasks). At inference, route to the correct set based on which task is predicting. For single-forward-pass mode, use the detection task's BN during shared backbone computation.

**Probe:** 5 GPU-hours (1 seed, 10 epochs). Compare TSBN vs shared BN on detection mAP. If TSBN doesn't improve, revert -- but published evidence strongly suggests it will.

### 18. Decoupled Training (Activity)

**What:** Train backbone with instance-balanced sampling for 50 epochs (keeping natural data frequencies). Freeze. Retrain activity classifier with class-balanced sampling. Kang et al. (ICLR 2020) showed this is the dominant paradigm for long-tail recognition.

**Evidence level:** HIGH. Foundational paradigm (ICLR 2020, 4000+ citations). Directly applicable to our activity head's 75-class power-law distribution.

### 19. Progressive Unlocking

**What:** Train backbone + detection for 15 epochs (anchor task), add activity at epoch 20, PSR at epoch 30, pose at epoch 40. Continue to 50. Curriculum training with detection as anchor yields 1-3% improvement on other tasks per the Vandenhende et al. survey.

**Evidence level:** MODERATE-HIGH. Supported by CAGrad/Recon theoretical framework. Curriculum/staged training is recommended by the MTL dense prediction survey.

### 20. GeometryAwareHeadPose

**What:** The 6D rotation representation + geodesic loss (251 lines in head_pose_geo.py). Already partially implemented. Expected -2 to -5 degrees MAE improvement.

**Evidence level:** HIGH. 6DRepNet (ICIP 2022) confirmed SOTA. Zhou et al. (CVPR 2019) proved continuity.

### 21. Two-Stage Activity Training

**What:** Freeze backbone, train activity head alone on cached backbone embeddings. After head converges (10-15 epochs), unfreeze backbone for joint fine-tuning. This eliminates gradient conflict entirely during the head initialization phase.

**Evidence level:** MODERATE. Standard MTL technique noted in doc 209 as never tried. The embedding cache approach is novel for our codebase but standard in MTL.

### Tier 3 Decisions (from doc 226 G2, Day 7):

These are gated on the 10-epoch probe results. The order above reflects the probability-weighted priority:
- **TSBN and Decoupled Training** should be implemented regardless of probe results (strong evidence, low risk)
- **Progressive Unlocking and GeometryAwareHeadPose** are high-confidence, implement if resources permit
- **Two-stage Activity Training and PSR Transition Prediction** are gated on Phase 1 diagnostics showing the underlying failure is recoverable

---

## TIER 4: HEAVY ARCHITECTURE CHANGES (Phased by Gate G2 Results)

These require 12+ hours of implementation plus significant GPU validation. Implement only if G2 (Day 7) criteria are met.

| Rank | Item | Impact | Effort (h) | GPU-h | Score | Source |
|------|------|--------|-----------|-------|-------|--------|
| 23 | **Nash-MTL-50 gradient surgery** | 40 | 20 | 10 | 1.9 | Agent 01 |
| 24 | **Nash-MTL (full, daily update)** | 45 | 20 | 50+ | 1.8 | Agent 01 |
| 25 | **CAGrad gradient surgery** | 15 | 12 | 30 | 1.0 | Agent 01 |
| 26 | **Anchor-free detection** | 15 | 28 | 50 | 0.45 | Doc 212, Agent 06 |
| 27 | **ConsMTL-style bi-level optimization** | 70 | 32 | 100+ | 0.53 | Agent 04 |

### 23. Nash-MTL-50 (Recommended over full Nash-MTL)

**What:** Replace PCGrad with Nash-MTL's bargaining solution. Nash-MTL-50 variant updates the bargaining weights every 50 steps instead of every step, achieving ~90% of full Nash-MTL benefit at ~1/50th the computational cost.

**Evidence level:** HIGH. ICML 2022. Nash-MTL achieves Dm=-4.04% on NYUv2 vs PCGrad's Dm=+3.97% -- an 8.01% absolute improvement. CAGrad (Dm=+0.20%) is simpler but gives less benefit.

**Risk:** MEDIUM. The Nash-MTL implementation (20h) requires care: the quadratic program solver must be numerically stable with bf16. The Nash-MTL-50 variant reduces overhead but the implementation complexity is similar.

**Decision rule:** If TSBN + loss weighting changes recover >50% of the detection gap, Nash-MTL adds marginal benefit. If detection is still struggling after Tier 1+2+3, implement Nash-MTL-50 as it directly targets gradient conflict which is the second-largest cause of detection degradation (Agent 06: gradient conflict recovers ~55% of gap).

### 26. Anchor-Free Detection

**What:** Replace RetinaNet-style anchor-based detection with FCOS-style anchor-free (roi_detector.py, 379 lines exists but not enabled).

**Evidence level:** MODERATE. YOLOPX showed +4.2pp gain from anchor-free, but assumes sufficient resolution and pretraining. At 224px with ImageNet-only features, anchor-free alone cannot overcome the structural ceiling.

**Risk:** HIGH. 28h effort, 50 GPU-h compute. The per-head gap analysis (doc 212) states the structural ceiling at 224px is ~0.40-0.55 mAP regardless of head architecture. If OHEM ablation and TSBN already close the gap, anchor-free adds diminishing returns.

**Decision rule (from doc 226):** Implement only if OHEM ablation from Tier 2 shows <0.02 mAP improvement. If OHEM is the bottleneck, fix it first.

---

## SUMMARY: 30-DAY EXECUTION ORDER

### Days 1-3 (Parallel: Code Changes + ST Baselines)

**Code changes (all 13 Tier 1 items, 24h total):**
```
Day 1: UW-SO + Per-task LR + Balanced Softmax + Gradient clipping (4h)
Day 2: EMA warmup + LDAM-DRW + SWA window + ASL (6.5h)
Day 3: Task dropout + Huberised geodesic + Varifocal + DB-MTL + WIoU (13.5h)
```

**Training (same 3 days, second GPU):**
```
ST baselines: pose (17.5h) + detection (35h) + PSR (25h) + activity (25h)
Quick probes: Mosaic (24h) + OHEM (24h) + PSR Gauss (2h)
= ~116.5 GPU-hours, back-to-back on one GPU
```

### Day 3 Gate G1 Results

Evaluate ST baselines against doc 226 criteria:

| Condition | Action |
|-----------|--------|
| ST detection mAP > 0.35 | Proceed with TSBN + anchor-free decision |
| ST detection mAP < 0.15 | Re-evaluate detection head architecture |
| ST activity top-1 > 15% | Proceed with LDAM-DRW + two-stage training |
| ST activity top-1 < 5% | Prepare to drop activity from paper |
| ST PSR event-F1 > 0.10 | Proceed with transition predictor |
| ST PSR event-F1 < 0.05 | Drop PSR from paper |

### Days 4-10 (Phase 2: Architecture Changes)

```
Day 4-5:   TSBN implementation + probe (6h code + 5h GPU)
Day 5-6:   Decoupled activity training implementation (6h)
Day 6-7:   Progressive unlocking schedule (6h)
Day 7-8:   GeometryAwareHeadPose + probe (6h code + 10h GPU)
Day 8-9:   Two-stage activity + PSR transition predictor (20h code + 75h GPU)
Day 9-10:  Integration testing + freeze architecture
```

### Day 7 Gate G2 Results

Evaluate architecture change probes:

| Condition | Action |
|-----------|--------|
| TSBN probe shows +0.03+ mAP | Integrate permanently |
| TSBN probe no improvement | Revert |
| PSR transition predictor F1 > 0.10 | Integrate |
| Activity two-stage training top-1 > 10% | Integrate |
| Detection still struggling >15% degradation | Implement Nash-MTL-50 |

### Days 11-20 (Phase 3: Full Training + Ablation Matrix)

```
Days 11-15: MTL main (5 seeds, 50 GPU-h) on RTX 5060 Ti
Days 12-15: ST re-runs (if architecture changed) on RTX 3060
Days 15-18: Ablation matrix (8-12 ablations, 240-330 GPU-h) on both GPUs
Days 16-20: Statistical analysis (CPU)
```

### Days 21-30 (Phase 4: Paper Writing)

Per doc 226 Phase 4 timeline.

---

## GLOBAL PRIORITY RANKING (All Items)

| Rank | Item | Score | Phase |
|------|------|-------|-------|
| 1 | UW-SO loss weighting | 56.7 | Day 1 |
| 2 | Per-task LR | 45.0 | Day 1 |
| 3 | Balanced Softmax (activity) | 30.0 | Day 1 |
| 4 | Gradient clipping | 30.0 | Day 1 |
| 5 | EMA warmup | 24.0 | Day 2 |
| 6 | LDAM-DRW (activity) | 20.0 | Day 2 |
| 7 | SWA window 5->10 | 16.0 | Day 2 |
| 8 | ASL (PSR) | 14.0 | Day 2 |
| 9 | Task head dropout | 10.0 | Day 3 |
| 10 | Huberised geodesic (pose) | 8.3 | Day 3 |
| 11 | Varifocal Loss (det cls) | 8.3 | Day 3 |
| 12 | DB-MTL log-transform | 8.3 | Day 3 |
| 13 | Gaussian-smeared PSR targets | 7.5 | Day 2 |
| 14 | TSBN / TS-sigma-BN | 7.7 | Day 4 |
| 15 | Decoupled training (activity) | 7.5 | Day 5 |
| 16 | Progressive unlocking | 5.8 | Day 6 |
| 17 | GeometryAwareHeadPose | 5.7 | Day 7 |
| 18 | WIoU v3 (det box) | 5.0 | Day 3 |
| 19 | Two-stage activity training | 4.3 | Day 8 |
| 20 | Infrastructure hardening | 3.75 | Day 1 (parallel) |
| 21 | ST baselines | 3.03 | Day 1-3 |
| 22 | PSR transition prediction | 2.2 | Day 8 |
| 23 | Nash-MTL-50 | 1.9 | Day 10 (gated) |
| 24 | Nash-MTL (full) | 1.8 | Day 10 (gated) |
| 25 | Mosaic augmentation | 1.25 | Day 1 |
| 26 | Per-task augmentation | 1.7 | Day 8 |
| 27 | CAGrad | 1.0 | Day 10 (fallback) |
| 28 | ConsMTL bi-level optimization | 0.53 | _Next paper_ |
| 29 | OHEM ablation | 0.63 | Day 1 |
| 30 | Anchor-free detection | 0.45 | Day 8 (gated) |

---

## WHAT NOT TO IMPLEMENT (Rejected with Evidence)

| Method | Why Rejected | Source |
|--------|-------------|--------|
| RLW (Random Loss Weighting) | Fails when losses differ by 2.5x | Agent 02 |
| GradNorm | Underperforms UW, unstable at extreme gradient ratios | Agent 02 |
| Full MoE (Mod-Squad) | 2-5x param pool, incompatible with single-pass | Agent 05 |
| TAPS layer gating | 15-50% param overhead, breaks efficiency claim | Agent 05 |
| Auto-Lambda / MetaWeighting | 1-3% improvement at significant complexity | Agent 02 |
| Full re-write to anchor-free | Cannot overcome 224px structural ceiling alone | Doc 212 |
| Adding temporal head (TCN+ViT for activity) | Would reintroduce Pathology 1 (sampler destroys temporal coherence) before the sampler is fixed | Doc 212, Agent 07 |

---

## COMPUTE BUDGET SUMMARY

| Phase | GPU-h (RTX 3060) | GPU-h (RTX 5060 Ti) | Total | Calendar Days |
|-------|-------------------|---------------------|-------|---------------|
| Prerequisites (ST baselines) | 0 | 116.5 | 116.5 | 3 |
| Tier 1 (code changes) | 0 | 0 | 0 | 1 |
| Tier 2 (config probes) | 50 | 0 | 50 | 2 |
| Tier 3 (arch probes) | 90 | 0 | 90 | 4 |
| Tier 4 (if gated) | 80 | 10 | 90 | 3 |
| Phase 3 (full training) | 200 | 180 | 380 | 10 |
| Phase 4 (writing) | 0 | 0 | 0 | 10 |
| Buffer | 30 | 58.5 | 88.5 | 0 |
| **TOTAL** | **450** | **365** | **815** | **30** |

---

## RISK-ADJUSTED IMPACT PROJECTION

If all Tier 1, 2, and 3 items succeed:

| Head | Current | Projected (Post-Fix) | Improvement Source |
|------|---------|---------------------|-------------------|
| Detection mAP | 0.202 | **0.30-0.40** | UW-SO (+3-5%), TSBN (+2-4 AP), Varifocal (+1-2 AP), WIoU (+1-2 AP), OHEM (+0.05-0.10) |
| Pose MAE | 9.13 deg | **6-8 deg** | Per-task LR (-2-4%), Huberised geodesic (-1-3 deg), GeometryAwareHeadPose (-2-5 deg) |
| Activity top-1 | ~0% | **15-30%** | LDAM-DRW (+3-10%), Balanced Softmax (+2-5%), Decoupled training (+15-25%), Two-stage (+15-25%) |
| PSR event-F1 | ~0.0 | **0.10-0.30** | ASL (+1-3%), Gaussian targets (+0.05-0.15), Transition prediction (+0.15-0.30 F1) |

**Paper narrative post-fix:** "MTL achieves 60-80% of ST performance on detection and pose at 30% parameter savings, while activity and PSR reveal two previously uncharacterized training pathologies."

---

## KEY DECISION CHAIN

```
Day 1: Start ST baselines + implement all Tier 1 code changes (24h)
Day 3: G1 -- evaluate ST baselines
  ├── Good: TSBN + Decoupled training + GeometryAwareHeadPose
  ├── Mixed: Drop lowest-performing head(s), focus compute
  └── Bad: Pivot to 3-task or pose-only paper
Day 7: G2 -- evaluate architecture probes
  ├── Good: Freeze architecture, proceed to Phase 3
  ├── Mixed: Nash-MTL-50 as additional fix
  └── Bad: Revert failed changes, proceed with what works
Day 15: G3 -- Phase 3 mid-point check
  ├── TRAJECTORY GOOD: Continue, prepare Phase 4
  └── TRAJECTORY POOR: Initiate fallback framing (doc 225 Section 4)
Day 20: Full results -> paper writing
Day 30: Submit
```
