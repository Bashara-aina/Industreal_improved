# 98 — Head-by-Head Analysis: All Four POPW Multi-Task Heads

**Date:** 2026-07-03
**RF4 Context:** Epoch 5 running at PID 2886320 on RTX 5060 Ti, all F1-F16 fixes applied
**Staging:** RF4 is Stage 3 in the staged training plan (all 4 heads active)

---

## 1. DETECTION HEAD

### 1.1 Architecture Summary
RetinaNet-style anchor-free detector on FPN (ConvNeXt-Tiny backbone). Predicts 24 classes (NUM_DET_CLASSES=24). Loss: FocalLoss with DET_ASYMMETRIC_GAMMA=True (gamma_pos=0.0, gamma_neg=1.5), FOCAL_ALPHA=0.50 (fixed from 0.25). GIoU for box regression.

### 1.2 Current Trajectory
**Loss:** det(c)=0.5-1.5, GIoU=0.25-0.40 — stable, no collapse.
**Epoch 2 val:** det_mAP50=0.0831, det_mAP50_pc=0.1330, n_present=15/24
**DET_PROBE:** score_p50=0.036, score_max=0.47-0.76, bestIoU_max=0.85-0.97 — localizing well, confident rarely.

The 0.036 median score is expected (99.3% anchors are background). The separation signal is score_max rising from 0.03→0.76 — positives ARE separating from background.

### 1.3 Fix Impact (F1, F4, F8)
- F1: Backbone gradient restored (was 80% wiped on seq batches). Epoch 2 is effectively the first with proper gradient.
- F4: ONE_CYCLE_PEAK_FACTOR=0.75 restores paper's per-sample intensity.
- F8: FOCAL_ALPHA 0.25→0.50 corrects asymmetric focal (with gamma_pos=0, alpha=0.25 suppressed positives 4x relative to balanced).

### 1.4 Expected Trajectory
- Epochs 5-8: mAP50_pc 0.15-0.25 (post-fix improvement)
- Epochs 8-12: Peak LR drives acceleration to 0.25-0.35
- Epochs 12-30: Slow improvement to 0.35-0.45
- RF10 target: 0.35-0.55 mAP50_pc

### 1.5 Conditions for SOTA-benchmarkable
- mAP50_pc >= 0.35
- Stable training through epoch 30 without CUDA timeout
- score_p50 rising above 0.10 by epoch 12

---

## 2. ACTIVITY HEAD

### 2.1 Architecture Summary
Per-frame MLP (ACTIVITY_HEAD_SIMPLE=True) over 768-dim fused FPN features. Predicts 69 verb-grouped classes. ACTIVITY_GRAD_CLIP=5.0 (fixed from 1.0), ACT_RAMP_EPOCHS=3 (fixed from 5). CB-Focal loss (beta=0.999). PoseFiLM modulation active.

### 2.2 Current Trajectory
**Loss rising:** 0.9→5.0+ as ramp completes (expected — full unscaled loss emerges).
**Epoch 2 val:** act_macro_f1=0.0063, act_top5=0.055, pred_distinct=5/69, entropy=1.27 nats
**LIVENESS:** act=3.11-5.75 ALIVE — gradient magnitudes healthy.

The activity loss RISE is not divergence — it's the ramp (ACT_RAMP_EPOCHS=3) completing at epoch 3. The loss drops from 1.0 to 0.9 and then rises as full 100% weight applies. This is the classifier beginning to separate features.

### 2.3 Fix Impact (F5, F9, F10)
- F5: Gradient centralization gated off — was interfering with sparse class-imbalanced gradients.
- F9: ACT_RAMP_EPOCHS 5→3 — gives 2 more epochs of full training.
- F10: ACTIVITY_HEAD_GRAD_CLIP 1.0→5.0 — was clipping every step (raw grad 3-6, clip at 1.0 destroyed signal).

### 2.4 Expected Trajectory
- Epochs 5-8: pred_distinct rises from 5 to 15-20, loss plateaus
- Epochs 8-12: macro_f1 0.03-0.05, entropy reaches 2.0 nats
- Epochs 12-20: Peak LR drives macro_f1 to 0.05-0.10
- Epochs 20-40: Slow improvement to 0.10-0.15
- RF10 target: macro_f1 >= 0.15

### 2.5 Conditions for SOTA-benchmarkable
- macro_f1 >= 0.15
- pred_distinct >= 40/69
- entropy >= 2.5 nats
- May need ACTIVITY_HEAD_SIMPLE=False (temporal TCN+ViT) at RF6+

---

## 3. HEAD POSE

### 3.1 Architecture Summary
9-DoF head pose estimator predicting forward/up/position vectors. WingLoss with confidence weighting, scaled ×0.001. KENDALL_HP_PREC_CAP=True (lv_pose fixed at -1.000, precision 2.718). HeadPoseFiLM (400K params) + PoseFiLM (841K params).

### 3.2 Current Trajectory
**Loss:** Dropped from 1.9→0.03-0.15 by mid-epoch 2. Converged.
**Epoch 2 val:** Forward MAE=11.32°, Up MAE=9.98°, Position=65.07mm
**Kendall:** lv_pose=-1.000 constant (HP_PREC_CAP active — pose capped at detection precision).

### 3.3 Fix Impact
- F1: Backbone gradient fix benefits pose indirectly (better backbone features).
- F14b: Stale pose reset fixed — lv_pose=-1.000 preserved across resumes.

### 3.4 Expected Trajectory
- Already within target range (8-13° fwd MAE) at epoch 2.
- Marginal improvement to 8-10° by epoch 12.
- Position error improves from 65mm to 30-40mm.
- RF10 target: 8-13° fwd MAE (already met).

### 3.5 Conditions for SOTA-benchmarkable
- Already viable. First IndustReal ego-pose baseline (wearer's head orientation from HoloLens).
- CORRECTION: This is NOT face-based head pose — comparisons to OpenFace/6DRepNet are category errors.
- Risk is OVER-performing and creating gradient imbalance with other heads.

---

## 4. PSR HEAD

### 4.1 Architecture Summary
11-component binary state classifier. Transformer with 3 layers, 4 heads, d_model=256. detach_psr_fpn=True (PSR gradient does NOT reach backbone). Trains every PSR_SEQ_EVERY_N_BATCHES=4 batches. PSR_WEIGHT=10 × PSR_SEQ_LOSS_SCALE=1.5 = 15x amplification. PSR_WARMUP_EPOCHS=3.

### 4.2 Current Trajectory
**Loss:** Structurally zero on non-seq batches, 0.6-6.0 on seq batches. Highly variable.
**Epoch 2 val:** psr_f1=0.0, comp acc=0.291, unique patterns=4/2048
**LIVENESS:** psr=1.00e-06 DEAD on non-seq batches (structural — zero loss = zero grad).

At 29.1% with 4 patterns, the model is predicting constant state vectors below random (50% chance for 11 binary components).

### 4.3 Fix Impact (F7, F15)
- F7: PSR_SEQ_EVERY_N_BATCHES 2→4 — detection sees 50% more data, PSR sees half as many updates.
- F15: PSR_SEQ_EVERY_N_BATCHES env-overridable.

### 4.4 Expected Trajectory
- Epochs 5-8: comp acc rises 0.291→0.40 (pulling out of local min)
- Epochs 8-12: comp acc 0.40-0.50, unique patterns 10-100
- Epochs 12-20: comp acc 0.50-0.60, transitions become non-zero
- Epochs 20-40: comp acc 0.60-0.70, PSR F1 0.20-0.40
- RF10 target: comp acc >= 0.40 (lowest priority head)

### 4.5 Conditions for SOTA-benchmarkable
- Would likely require detach_psr_fpn=False at RF6+ (let PSR shape backbone)
- 15x gradient amplification creates batch-to-batch instability
- Component imbalance (0.19-1.0) needs component-level weighting

---

## 5. MULTI-TASK BALANCING (Kendall)

### 5.1 Kendall Trajectory
| Step | lv_det | lv_pose | lv_act | lv_psr |
|---|---|---|---|---|
| 1 | 0.004 | -1.000 | -0.005 | -0.001 |
| 5701 | 0.070 | -1.000 | 0.102 | -0.010 |
| →ep5 | 0.075 | -1.000 | 0.114 | -0.014 |

### 5.2 Key Observations
- lv_pose = -1.000 constant (HP_PREC_CAP active — pose weight fixed)
- lv_det rising slowly (0.004→0.075): detection considered increasingly easy
- lv_act positive (0.114): activity head thinks task is easy (may indicate miscalibrated uncertainty)
- lv_psr near 0: Kendall barely weights PSR (zero on 75% of batches)
- F14 (weight_decay=0) critical: before, log-variances couldn't evolve freely

---

## 6. CROSS-CUTTING RISKS

1. **CUDA timeout** — has killed every run before epoch 5. Current TF32+V8 mitigations may not be sufficient.
2. **Activity gradient amplitude** — act grad 3-6 at 5.0 clip means ~100% of steps are clipped on activity. The per-head clip may still be suppressing activity learning.
3. **PSR alternating gradient intensity** — total loss jumps from ~5 (non-seq) to ~95 (seq, with 15x PSR amp). This 19x swing destabilizes optimizer.
4. **Head pose overshooting** — fixed high precision (2.718) at peak LR may overshoot optimal pose parameters.

### Head Health Summary

| Head | Status | Key Metric | Current | Target (RF10) | Confidence |
|---|---|---|---|---|---|
| Detection | Recovering | mAP50_pc | 0.133 | >=0.35 | 70% |
| Activity | Mode collapsed | macro_f1 | 0.006 | >=0.15 | 45% |
| Head Pose | **Converged** | fwd MAE | 11.32° | 8-13° | **95%** |
| PSR | Not started | comp acc | 0.291 | >=0.65 | 30% |
| Combined | Early | combined | 0.183 | >=0.45 | 50% |
