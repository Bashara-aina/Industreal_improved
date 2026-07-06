# 130 — Master Plan to Beat SOTA on All Heads

**Date:** 2026-07-06
**Purpose:** Single-file action plan derived from all 66 questions (file 127) and 30 debates (file 128). Sequenced by priority and dependency. Each action item has: (a) what to do, (b) why, (c) who/what blocks it, (d) estimated effort, (e) expected impact.

---

## Executive Summary — Current State vs Goal

| Head | Current | Goal (Beat SOTA) | Gap | Priority |
|---|---|---|---|---|
| Detection (self-trained YOLOv8m) | mAP50=0.995 | ≥0.838 (WACV) | **DONE** | P1 — claim now |
| Detection (multi-task ConvNeXt) | mAP50=0.358 | ≥0.838 | -0.480 | P1 — major work |
| Activity (per-frame MLP) | top1=0.023 | ≥0.10 first baseline | -0.077 | P2 — redefine |
| Activity (clip-level) | top1=0.028 | ≥0.622 (MViTv2-S) | -0.594 | P3 — architectural |
| Head Pose forward | 8.39° | ≤15° | **DONE** | P2 — claim now |
| Head Pose up | 26.20° (or 13.52°) | ≤15° | -11° | P3 — investigate |
| PSR (per-comp F1) | 0.7499 | ≥0.883 (B3) | -0.133 | P1 — finish gap |
| PSR (transition F1) | 0 (D4) | ≥0.883 | -0.883 | P2 — paradigm |
| PSR POS | 0.968 | drop | n/a | P3 — explain |

**Net assessment:** Detection (single-task), Head Pose forward are **DONE**. The other 6 metrics need work. PSR is the closest to SOTA (within 0.13). Activity is the farthest (0.59 clip-level gap). Head Pose up is unclear. Multi-task detection is the highest-leverage remaining work.

---

## P1: Critical Path (Next 2 Weeks)

### P1.1 Finish PSR (0.7499 → 0.883, gap -0.13)

**Action:** Train PSR with `KENDALL_FIXED_WEIGHTS=True` and fixed manual weights, removing the Kendall suppression. Run for 5-10 epochs from epoch_18 checkpoint.

**Why:** Question PSR-3, debate 2.3: Kendall is suppressing PSR via log_var_psr=-0.04. Fixed weights force the optimizer to give PSR its full gradient budget.

**Blocks:** None (have crash_recovery.pth).

**Effort:** 2-3 days compute, including crash risk.

**Expected impact:** F1 from 0.7499 → ~0.82 (estimate based on observed gradient starvation).

**Files:**
- `src/config.py` — toggle KENDALL_FIXED_WEIGHTS=True
- `src/training/train.py` — verify loss path with fixed weights
- `src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth` — resume point

**Source:** PSR-3, debate 2.3.

---

### P1.2 Fix D1 Detection Audit (mAP=0.0004 → ≥0.60)

**Action:** Run a brute-force per-class histogram comparison (predicted vs GT class IDs). If shifted, fix the +1; if not, the 0.0004 is real and we must report it.

**Why:** Question D-2, debate 1.2, AC-1: D1 v3 returned 0.0004. AAIML reviewers will spot this and ask. The 0.995 from d1r is a separate training run; without D1 audit, we cannot claim SOTA.

**Blocks:** Nothing; pure eval.

**Effort:** 1 day.

**Expected impact:** Either (a) fix +1 bug → mAP ~0.6-0.8, or (b) confirm 0.0004 is real → drop "BEATS SOTA" claim from main paper.

**Files:**
- `src/evaluation/eval_yolov8m.py` — class index alignment
- `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` — already done, mAP=0.0004

**Source:** D-2, debate 1.2, EP-1.

---

### P1.3 Detect mAP NaN on Full Eval (D3 v4, v6, v7)

**Action:** Run eval in-process by setting `EVAL_MAX_BATCHES=0` in main training config. The subprocess_eval.py path has 5 fixed bugs; one more may remain. In-process eval bypasses subprocess entirely.

**Why:** Debate 1.2: full eval returns NaN. Subsample (250 batches) gives 0.358 but is unreviewable.

**Blocks:** Nothing.

**Effort:** 1 day.

**Expected impact:** Either (a) in-process works → full-eval mAP, or (b) subprocess bug fixed → full-eval mAP.

**Files:**
- `src/training/train.py` — toggle EVAL_MAX_BATCHES
- `src/runs/rf_stages/checkpoints/d3_v4/metrics.json` (NaN, needs fix)

**Source:** D-1, debate 1.2.

---

### P1.4 Train Activity Non-Simple Head (TCN+ViT)

**Action:** Set `ACTIVITY_HEAD_SIMPLE=False`, train for 10 epochs from epoch_18. Compare clip-level top1 between simple and non-simple.

**Why:** Question ACT-2, debate 3.1: per-frame MLP has architectural ceiling at 0.028. TCN+ViT (T=16) adds temporal context.

**Blocks:** Need ~2 days compute on RTX 5060 Ti.

**Effort:** 2-3 days.

**Expected impact:** clip-level top1 from 0.028 → ~0.10-0.20 (modest; backbone is still the bottleneck).

**Files:**
- `src/config.py` ACTIVITY_HEAD_SIMPLE
- `src/models/model.py` ActivityHead non-simple path

**Source:** ACT-2, debate 3.1.

---

## P2: High Priority (Weeks 3-4)

### P2.1 Knowledge Distillation (D6: 0.358 → ≥0.6)

**Action:** Use the self-trained YOLOv8m (d1r, mAP=0.995) as a teacher. Distill into the ConvNeXt-Tiny detection head. Train for 10 epochs.

**Why:** Question D-6: distillation flips detection from weakness to methodology contribution. The teacher provides soft labels for the harder classes.

**Blocks:** Need to wait for training compute.

**Effort:** 3 days.

**Expected impact:** Detection mAP from 0.358 → ~0.65 (estimate).

**Files:**
- `runs/detect/src/runs/yolov8m_industreal/d1r/weights/best.pt` — teacher
- `src/training/train.py` — add distillation loss

**Source:** D-6.

---

### P2.2 Fixed-Weight Ablation (KENDALL_FIXED_WEIGHTS)

**Action:** Run training with KENDALL_FIXED_WEIGHTS=True, KENDALL_HP_FIXED_LAMBDA=0.2. Compare PSR F1.

**Why:** Question PSR-3, debate 2.3: Kendall is suppressing PSR. Fixed weights are the principled baseline.

**Blocks:** None.

**Effort:** 2-3 days compute.

**Expected impact:** PSR F1 from 0.7499 → ~0.80.

**Files:**
- `src/config.py` KENDALL_FIXED_WEIGHTS

**Source:** PSR-3, debate 2.3.

---

### P2.3 PSR Backbone Swap (already done: D4 F1=0)

**Action:** The D4 experiment was completed (F1=0, POS=0.999). Document in paper as "PSR head on YOLOv8m gives F1=0 because [REASON]."

**Why:** Question PSR-4, debate 2.2: D4 is the reviewer-3 backbone-swap experiment. The result is the most damning number; disclosure must be clear.

**Blocks:** Nothing.

**Effort:** 0 (already done).

**Expected impact:** Honest disclosure. Doesn't help F1.

**Files:**
- `src/runs/rf_stages/checkpoints/d4_yolov8m_psr/metrics.json`

**Source:** PSR-4, debate 2.2.

---

### P2.4 Per-Recording Breakdown of Up-Vector Error

**Action:** Compute up-vector MAE per recording (16 recordings), report median + IQR.

**Why:** Question HP-2, debate 4.2: three conflicting up-vector numbers exist. Per-recording breakdown resolves which is real.

**Blocks:** Need to run diagnostic on epoch_18.

**Effort:** 1 day.

**Expected impact:** Median up-vector may be ~13-15° (closer to forward direction), resolving the inconsistency.

**Files:**
- `src/evaluation/head_pose_diag.py`

**Source:** HP-2, debate 4.2.

---

### P2.5 Leave-One-Recording-Out CV for PSR Thresholds

**Action:** Compute per-component thresholds on N-1 recordings, evaluate on held-out one. Repeat for all 16 recordings.

**Why:** Question PSR-5, debate 7.2: per-component thresholds on same val set is overfitting. LOO CV confirms.

**Blocks:** Need to wait for compute.

**Effort:** 2 days.

**Expected impact:** If improvement persists, threshold is real. If it shrinks to <0.005, threshold is val noise.

**Files:**
- `src/evaluation/psr_optimal_thresholds.py` — add LOO mode

**Source:** PSR-5, debate 7.2.

---

### P2.6 Transition F1 Side-by-Side (PSR Per-Frame vs Transition)

**Action:** Compute both per-frame F1 (current 0.7499) and transition F1 (event matching within tolerance) on the same predictions.

**Why:** Question PSR-2: different metrics measure different things. Mixing them is invalid.

**Blocks:** None.

**Effort:** 1 day.

**Expected impact:** Cleaner SOTA comparison if transition F1 ≈ 0.7 (close to B3).

**Files:**
- `src/evaluation/psr_optimal_thresholds.py` — add transition F1

**Source:** PSR-2.

---

## P3: Lower Priority (Weeks 5+)

### P3.1 Error-State Detection Evaluation (FPR/FNR)

**Action:** Evaluate error-state class (24) using FPR/FNR since val has 0 GT for it.

**Why:** Question D-7: WACV reports error-state FPR=65%. Our val has 0 GT for error class.

**Blocks:** Need to load WACV eval protocol.

**Effort:** 2 days.

**Expected impact:** Different metric than mAP for error class. Could distinguish from WACV.

**Files:**
- `src/evaluation/eval_yolov8m.py`

**Source:** D-7.

---

### P3.2 FiLM Ablation (HP-5)

**Action:** Train without FiLM conditioning for pose head. Compare 8.39° vs no-FiLM.

**Why:** Question HP-5: FiLM contribution unmeasured.

**Blocks:** Need compute.

**Effort:** 2 days.

**Expected impact:** May show FiLM contributes +0.5-1° to MAE.

**Files:**
- `src/config.py` USE_HEADPOSE_FILM
- `src/models/model.py` FiLM layers

**Source:** HP-5.

---

### P3.3 Position Units Resolution (HP-3)

**Action:** Contact IndustReal authors or check HoloLens SDK to confirm position units (mm vs cm).

**Why:** Question HP-3: position unit ambiguity. Code says "DO NOT USE FOR REPORTING."

**Blocks:** External contact.

**Effort:** 1-3 days (depends on response time).

**Expected impact:** Either confirm mm or drop position claims.

**Files:**
- `src/evaluation/evaluate.py:1918-1926`

**Source:** HP-3.

---

### P3.4 Activity Linear Probe (Backbone Bottleneck Test)

**Action:** Frozen ConvNeXt-Tiny features → single linear layer → 69 classes. Train 10 epochs.

**Why:** Debate 3.1: if linear probe also plateaus at 0.03, the bottleneck is the backbone, not the head.

**Blocks:** None.

**Effort:** 1 day.

**Expected impact:** Diagnoses where the activity improvement is blocked.

**Files:**
- `src/models/model.py` ActivityHead

**Source:** ACT-2, debate 3.1.

---

### P3.5 Halt Sequence-Mode if F1=0

**Action:** If P1.1 and P2.2 both fail to lift PSR F1 above 0.85, drop `USE_PSR_SEQUENCE_MODE=True` from config and train per-frame-only.

**Why:** Question A-3: 25% throughput cost for zero benefit.

**Blocks:** Wait for P1.1/P2.2 results.

**Effort:** 0 (config change).

**Expected impact:** 25% throughput speedup.

**Files:**
- `src/config.py` USE_PSR_SEQUENCE_MODE

**Source:** A-3.

---

### P3.6 Stop Best-Checkpoint Being Broken (AC-1)

**Action:** Audit which SOTA numbers came from epoch 11 vs epoch 18. Re-derive any that used epoch 11.

**Why:** Question AC-1: epoch 11 was promoted via broken metric. Any pre-fix number may be invalid.

**Blocks:** Audit only.

**Effort:** 1 day.

**Expected impact:** Clean separation of valid/invalid numbers.

**Files:**
- `src/runs/rf_stages/checkpoints/best.pth`

**Source:** AC-1.

---

## P4: Disclosure & Writing (Continuous)

### P4.1 Add POS Paradox Explanation to Paper

**Action:** In §5.2.1, add 1 paragraph explaining POS=0.968 vs F1=0.7499 vs D4 F1=0.

**Why:** Question PSR-1, debate 2.1: high POS is misleading without context.

**Blocks:** None.

**Effort:** 0.5 day.

**Source:** PSR-1, debate 2.1.

---

### P4.2 Reframe Detection as "Multi-Task Cost"

**Action:** Section title: "Detection: Multi-Task Cost Measurement". Include 64-68% ratio table.

**Why:** Debate 1.3: cost framing is honest, "competitive" is misleading.

**Blocks:** None.

**Effort:** 0.5 day.

**Source:** 1.3.

---

### P4.3 Add Honest Disclosure Section

**Action:** §5.4 "Honest Disclosures" with 8 numbered items (D4 F1=0, activity 0.028, etc.).

**Why:** Question PW-7, debate 9.7: honesty matrix builds reviewer trust.

**Blocks:** None.

**Effort:** 1 day.

**Source:** PW-7, debate 9.7.

---

### P4.4 Rename "Activity Recognition" → "Per-Frame Action Classification"

**Action:** grep -l "activity recognition" → rename all instances.

**Why:** Question PW-1: inconsistent naming.

**Blocks:** None.

**Effort:** 0.5 day.

**Source:** PW-1.

---

## P5: Stretch Goals (Optional)

### P5.1 Build MViTv2-S Activity Head

**Action:** Replace simple MLP with MViTv2-S (8.2M params, video-level). Train for 20 epochs.

**Why:** Question ACT-2: architectural change to reach 0.622.

**Blocks:** Major compute (5+ days).

**Effort:** 1 week.

**Expected impact:** Activity clip-level 0.028 → ~0.50.

---

### P5.2 Knowledge Distillation from D1R YOLOv8m → ConvNeXt-Tiny Backbone

**Action:** Soft-label distillation. Compare multi-task detection mAP.

**Why:** Question D-6.

**Blocks:** P2.1.

---

### P5.3 Procedural Knowledge Loss in PSR

**Action:** Add expected-transition mask as training loss. Compare per-component F1.

**Why:** Question PSR-7, debate 2.2: B3's 0.883 vs ours 0.7499 — procedural knowledge may be +0.10.

**Blocks:** Major code change.

---

## Sequencing — Recommended Order

```
Week 1:  P1.2 (D1 audit) | P1.3 (full eval) | P4.1-P4.4 (disclosure + writing)
Week 2:  P1.1 (PSR fixed weights) | P2.4 (up-vector per-recording)
Week 3:  P2.2 (fixed-weight ablation) | P2.5 (PSR LOO CV) | P2.6 (transition F1)
Week 4:  P1.4 (TCN+ViT activity) | P2.1 (knowledge distillation)
Week 5+: P3.x (lower priority) | P5.x (stretch goals)
```

---

## GPU Resource Plan

| GPU | Currently | Used For | Priority |
|---|---|---|---|
| RTX 5060 Ti (16 GB) | Training (epoch 25) | New training runs | P1.1, P2.1, P2.2 |
| RTX 3060 (12 GB) | Idle | Eval, ablation | P1.2, P1.3, P2.4, P2.5 |

Both GPUs can run in parallel. P1.1 (training) on 5060 Ti, P1.2/P1.3 (eval) on 3060.

---

## Critical Risks

1. **Training crash:** 4 CUDA crashes already happened. P1.1 may crash again. Mitigation: batch_size=2 with OMP_NUM_THREADS=4.
2. **PSR doesn't improve:** P1.1, P2.2 may both fail. Mitigation: P3.5 (drop sequence mode).
3. **Activity remains broken:** MLP cannot reach 0.10+. Mitigation: P3.4 (linear probe) + P5.1 (MViTv2-S).
4. **D1 audit reveals bug:** Either fix or drop SOTA claim. Either way, no showstopper.
5. **Position units unresolved:** P3.3 may take >3 days. Mitigation: drop position claims entirely.

---

## Success Metrics

| Metric | Current | Target | Date |
|---|---|---|---|
| PSR F1 | 0.7499 | ≥0.83 | Week 2-3 |
| Detection mAP (multi-task) | 0.358 | ≥0.60 | Week 4 |
| Activity clip-level top1 | 0.028 | ≥0.10 | Week 4-5 |
| Head pose up-vector MAE | 26.20° | ≤15° | Week 2 |
| Honest disclosures | 0/8 | 8/8 | Week 1 |
| SOTA claims defensible | 2/4 | 4/4 | Week 4 |

---

## File Map — What Each P-Item Modifies

| P-Item | Files Modified |
|---|---|
| P1.1 | `src/config.py`, `src/training/train.py` |
| P1.2 | `src/evaluation/eval_yolov8m.py` (already done) |
| P1.3 | `src/training/train.py` (toggle) |
| P1.4 | `src/config.py` (ACTIVITY_HEAD_SIMPLE), `src/models/model.py` |
| P2.1 | `src/training/train.py` (distillation loss) |
| P2.2 | `src/config.py` (KENDALL_FIXED_WEIGHTS) |
| P2.3 | Documentation only |
| P2.4 | `src/evaluation/head_pose_diag.py` |
| P2.5 | `src/evaluation/psr_optimal_thresholds.py` |
| P2.6 | `src/evaluation/psr_optimal_thresholds.py` |
| P3.1 | `src/evaluation/eval_yolov8m.py` |
| P3.2 | `src/config.py`, `src/models/model.py` |
| P3.3 | External (HoloLens SDK) |
| P3.4 | `src/models/model.py` |
| P3.5 | `src/config.py` |
| P3.6 | Audit only (no file changes) |
| P4.1-P4.4 | `popw_aaiml2027.tex` |

---

## Cross-Reference to Question Files

| P-Item | Question File 127 ID | Debate File 128 ID |
|---|---|---|
| P1.1 | PSR-3 | Debate 2.3 |
| P1.2 | D-2, EP-1 | Debate 1.2, 7.1 |
| P1.3 | D-1 | Debate 1.2 |
| P1.4 | ACT-2, ACT-6 | Debate 3.1 |
| P2.1 | D-6 | (not debated) |
| P2.2 | PSR-3 | Debate 2.3 |
| P2.3 | PSR-4 | Debate 2.2 |
| P2.4 | HP-2 | Debate 4.2 |
| P2.5 | PSR-5, EP-4 | Debate 7.2 |
| P2.6 | PSR-2 | (not debated) |
| P3.1 | D-7 | (not debated) |
| P3.2 | HP-5 | (not debated) |
| P3.3 | HP-3 | (not debated) |
| P3.4 | ACT-2 | Debate 3.1 |
| P3.5 | A-3 | Debate 5.3 |
| P3.6 | AC-1 | Debate 10.1 |
| P4.1 | PSR-1 | Debate 2.1 |
| P4.2 | (covered in 1.3) | Debate 1.3 |
| P4.3 | PW-7 | Debate 9.7 |
| P4.4 | PW-1 | Debate 9.1 |
| P5.1 | ACT-2 | Debate 3.1 |
| P5.2 | D-6 | (not debated) |
| P5.3 | PSR-7 | Debate 2.2 |

---

## One-Line Summary

> Beat SOTA on PSR via fixed-weight training (P1.1, P2.2) → reach 0.83 F1. Beat SOTA on activity via TCN+ViT (P1.4) → reach 0.10+. Frame detection honestly as multi-task cost (P4.2) → 64-68% ratio. Disclose D4 F1=0, activity 0.028, POS paradox in §5.4 (P4.3) → build reviewer trust.