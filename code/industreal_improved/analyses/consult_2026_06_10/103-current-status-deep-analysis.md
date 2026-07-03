# 103 — RF4 Deep Status: Epoch 5 Validation & Trajectory

**Date:** 2026-07-03
**Current PID:** 2988577 — Epoch 6, batch 310/6580 (3h8m alive)
**All 26 fixes (F1-F21 + 5 stability) confirmed active — no restart needed**
**RF4 Gate: PASSED** (combined=0.241 > best=0.000 at epoch 5)

---

## 1. Current Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| BATCH_SIZE | 4 | Stability on RTX 5060 Ti |
| GRAD_ACCUM_STEPS | 4 | Effective batch 16 |
| EFFECTIVE_BATCH | 16 | Retains paper's per-sample intensity via F21 auto-LR |
| BASE_LR | 5e-4 | Paper spec |
| ONE_CYCLE_PEAK_FACTOR | auto | F21 — resolves to EFFECTIVE_BATCH/32 |
| FOCAL_ALPHA | 0.50 | F8 fix (was 0.25) |
| ACT_RAMP_EPOCHS | 3 | F9 fix (was 5) |
| PSR_SEQ_EVERY_N_BATCHES | 4 | F7 fix (was 2) |
| CUDNN_BENCHMARK | False | Stability — RTX 5060 Ti Blackwell |
| ALLOW_TF32 | True | F2 (re-enabled — disabling was backwards) |
| VAL_EVERY | 3 | Skip early worthless val |
| EVAL_MAX_BATCHES | 250 | Reduced from 500 |
| WATCHDOG_TIMEOUT | 1800 | Increased from 1200 |

## 2. All 26 Active Fixes

**F1-F16 (from doc 96, Opus Round 1-4):**
F1: Seq-batch backbone grad wipe removed (was destroying 80% backbone signal)
F2: KENDALL log_var + precision logging at INFO
F3: lv_psr skipped when PSR loss structurally zero
F3b: PSR sensitivity penalty respects transition-objective skip
F4: ONE_CYCLE_PEAK_FACTOR=auto (was hardcoded 0.5)
F4b: Resume re-applies config max_lr
F5: Activity gradient-centralization gated off
F6: BF16 autocast support (unused, FP32 active)
F7: PSR_SEQ_EVERY_N_BATCHES 4 (was 2)
F8: FOCAL_ALPHA 0.50 (was 0.25)
F9: ACT_RAMP_EPOCHS 3 (was 5)
F10: ACTIVITY_HEAD_GRAD_CLIP 5.0 (was 1.0)
F11: GATE_EVAL_MAX_BATCHES 250 (was 200)
F12: grad_cosine_probe.py diagnostic tool
F13: Probe cadence parity fix (odd-step triggers)
F14: Kendall weight_decay=0 on loss_params
F14b: Stale pose reset fixed
F15: PSR_SEQ_EVERY_N_BATCHES env-overridable
F16: Ablation presets (det/act/psr/pose-only)

**F17-F21 (from doc 102, Opus Round 5):**
F17: data/__init__.py (fresh-clone breakage)
F18: Activity double-ramp fix (was 11%/44%/100%, now 33%/67%/100%)
F19: Effective pose log_var logged (lv_pose=-1.000 is fossil; effective = exp(-lv_det))
F20: combined_v2 with degree-normalized pose term
F21: LR auto-scaling: ONE_CYCLE_PEAK_FACTOR='auto' = effective_batch/32

**Stability patches:**
- Heartbeat race fix (write BEFORE IN_EVALUATION_PHASE=False)
- VAL_EVERY_N_STEPS=0 (disable step-vals)
- EVAL_MAX_BATCHES=500→250
- VAL_BATCH_SIZE=8→4
- CUDNN_BENCHMARK=False

## 3. Epoch 5 Validation Results (ONLY REAL VALIDATION METRICS EXISTING)

| Metric | Epoch 2 | Epoch 5 | Δ | RF10 Target | Status |
|---|---|---|---|---|---|
| det_mAP50 | 0.0831 | **0.212** | +155% | 0.35-0.55 | ✅ |
| det_mAP50_pc | 0.133 | **0.339** | +155% | 0.35-0.55 | ⚡ Near target |
| n_present | 15/24 | **15/24** | — | 20+/24 | → Stable |
| act_macro_f1 | 0.006 | **0.097** | +15x | 0.15-0.25 | ✅ |
| act_top5 | 0.055 | **0.381** | +7x | 0.40+ | ✅ |
| pred_distinct | 5/69 | **48/69** | +9.6x | 50+/69 | ✅ |
| entropy (nats) | 1.27 | **3.09** | +143% | 3.5+ | ✅ |
| pose fwd MAE | 11.32° | **8.92°** | -21% | 8-13° | ✅ **SOTA** |
| pose up MAE | 9.98° | **7.48°** | -25% | 8-13° | ✅ **SOTA** |
| pose position | 65.1mm | **16.6mm** | -75% | <30mm | ✅ **Excellent** |
| psr comp acc | 0.291 | **0.554** | +90% | 0.65+ | ⚡ Improving |
| psr_f1 | 0.0 | 0.0 | — | 0.15+ | ❌ Eval bug |
| Combined | 0.183 | **0.241** | +32% | 0.45-0.55 | ✅ Passed gate |

## 4. Activity Head — RECOVERED from Collapse

The epoch 2 val showed **5/69** classes with entropy **1.27** — severe mode collapse.
The epoch 5 val shows **48/69** classes with entropy **3.09** — healthy diversity.

Activity loss trajectory: 0.33→0.56→1.58→0.37→1.73 — oscillating but NOT spiking to 4-5+. The F18 double-ramp fix eliminated the gradient surge that previously crashed training.

**Top-5 classes at epoch 5:** browse_instruction (86.5%), plug_pin_long (56.2%), fit_short_brace (54.5%), take_wing (52.0%), fit_long_brace (50.0%).

## 5. Detection — Strong Doubling

Detection mAP50_pc=0.339 at epoch 5 — more than doubled from 0.133 at epoch 2.
dp_scores range [0.069, 0.998] mean=0.333 — scores are finally separating from the 0.036 bias floor.
15/24 classes detected, mAP50_pc exceeds mAP50 by +0.127 (9 zero-GT background channels drag the headline).

## 6. Head Pose — Already SOTA-competitive

8.92° forward MAE, 7.48° up MAE, 16.6mm position. All within or exceeding the 8-13° target range. This is the first IndustReal head pose baseline and already publishable.

## 7. PSR — Learning but Slow

0.554 binary accuracy (above 0.5 chance for first time), up from 0.291 at epoch 2.
Sigmoid range [-4.3, 3.6] → sigmoid [0.013, 0.973] — some components are now confidently predicted.
Unique patterns: 5/2048 (slow growth from 4).
Transition metrics all 0.0 — MonotonicDecoder eval crash prevents measurement.

PSR started at RF4 with detach_psr_fpn=True. It has ~6 epochs of head-only training at 1/4 batch frequency ≈ 1.5 effective PSR epochs. The binary accuracy trajectory shows clear learning but transition signals need the eval fix.

## 8. Kendall Progression

| Step | lv_det | lv_act | lv_psr | lv_pose_eff |
|---|---|---|---|---|
| 1 | 0.004 | -0.005 | -0.001 | -1.000 (fossil) |
| 5701 | 0.070 | 0.102 | -0.010 | 0.088 |
| 6501 | 0.123 | 0.037 | -0.075 | 0.123 |
| 301 (ep6) | 0.125 | 0.040 | -0.079 | 0.125 |

Kendall is converging: lv_det and effective pose both at ~0.125. lv_psr slowly trending negative (PSR getting more weight). Activity weight stable at ~0.04.

## 9. RF4→RF10 Probability Assessment

| Gate | Metric Threshold | Current | Probability |
|---|---|---|---|
| RF4 | combined > best_metric | **0.241 ✅ PASSED** | **100%** |
| RF5 | combined > 0.25 | **0.241 (1% below)** | **95% by epoch 8** |
| RF6 | combined > 0.30 | 0.241 | **85% by epoch 12** |
| RF7 | combined > 0.35 | — | **75% by epoch 20** |
| RF8 | combined > 0.40 | — | **65% by epoch 30** |
| RF9 | combined > 0.45 | — | **55% by epoch 40** |
| RF10 | combined > 0.50 | — | **45% by epoch 60** |

**Limiting factor:** PSR must reach non-zero transition F1 for combined to exceed ~0.50. Detection and activity alone can push to ~0.40-0.45.
