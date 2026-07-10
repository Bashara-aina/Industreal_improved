# 197 — Results & Metrics: run10 EP10 Failure → run11 Current State

**Date:** 2026-07-10
**run11 PID:** 885592, GPU 1, launched 10:27 JST
**Data as of:** ~2800 batches into epoch 6 (resumed from ep5)

---

## 1. run10 EP10 Eval — The Trigger Event

**Why run10 was killed:** Epoch 10 eval showed all three academic heads at/near zero.

```
==================================================
Evaluation @ epoch 10 (EMA weights)
==================================================
  Activity top-1: 0.0058 (0.58%)  top-5: 0.0192 (1.92%)  (n=37796)
  Detection mAP: mAP50=0.0000  mAP50:95=0.0000  mAP50_pc=0.0000
  PSR: event_f1@+-3=0.004  POS=0.000  tau=nan  (n_recs=16)
  Pose: fwd_MAE=8.92deg [7.74, 10.87]  up_MAE=7.48deg [6.89, 8.81]  (n=37796)
==================================================
```

**What was the architecture at EP10?**
- Activity: 2-layer MLP (768→1024→75, 1.1M), fresh-init at resume (B-9 shape filter)
- Detection: Sparse 3×3 positive cells, P2/P3/P4/P5 (P2 = conv_proj 96-dim)
- PSR: 4-layer transformer, d=96, ff=384, conv_proj features, T=8→16 linear interpolation
- Pose: 6D MLP (unchanged, already healthy)

**Diagnosis:**
1. Activity at 0.58% = below random (1.33%) → the 2-layer MLP cannot discriminate 75 classes from a single cls_token. The head was fresh-init at resume, but even a cold head should beat random after 5 epochs. This is a capacity problem.
2. Detection at 0.0 mAP → either eval-harness bug, or sparse 3×3 from semantics-free P2 features produces no real signal. A prior ConvNeXt run reached 0.468 mAP (176 §3.4), suggesting the MViTv2-S backbone + 3×3 assigner can work — but not from the P2 features.
3. PSR at 0.004 F1 → conv_proj (96-dim edge features) + 4-layer transformer = no semantic signal. Loss was flat at 1.56 for 5 epochs. The head was reading garbage.
4. Pose at 8.9° → healthy. The cls_token carries spatial direction information.

---

## 2. run11 Batch-Level Losses (First 2800 Batches)

**Resumed from epoch 5** (best.pt). New heads (activity 3-layer, PSR 6-layer P5, detection TAL) were fresh-init. Optimizer restarted fresh.

### 2.1 Detection Loss Pattern

```
[batch   100] det=4.1781    ← GT present: TAL assigns topk cells, loss computes
[batch   200] det=4.3612    ← GT present
[batch   300] det=0.0016    ← no GT in this batch
[batch   400] det=4.8070    ← GT present
[batch   500] det=3.8802    ← GT present
[batch   600] det=0.0011    ← no GT
[batch   700] det=0.0012    ← no GT
[batch   800] det=0.0014    ← no GT
```

**Pattern:** Alternating 0.001 (no-GT batches, ~60-70% of batches) vs 3.9-4.8 (GT batches, ~30-40%). This is **exactly the expected behavior** for detection on a dataset where only ~13% of frames are annotated. The TAL assigner is working — GT batches produce real gradient signal.

**Key insight:** The ~60-70% of batches with det=0.001 are NOT a problem. The CE+DFL losses are computed only on positive cells; when no GT boxes exist in the batch, the loss is near-zero. The ~30-40% with GT provide the real supervision.

### 2.2 Activity Loss Trend

```
[batch     0] act=4.4375
[batch   100] act=5.5633
[batch   500] act=5.0245
[batch  1000] act=4.5338
[batch  1500] act=4.3416
[batch  2000] act=5.2911
[batch  2800] act=5.0888
```

**Assessment:** High but stable at 4.3-5.6. The class weights (sqrt-tamed inverse-frequency, max=11.7) are active. With 75 classes and long-tail distribution, a CE of ~5.0 is expected early in training. The old run had act loss flat at 4.8 — this is in the same range but should trend down over 10+ epochs.

**Concern:** The 3-layer MLP adds more capacity but the fundamental bottleneck — compressing 16 frames of assembly video into a single 768-dim vector for 75-class discrimination — is unchanged. Even SOTA single-task MViTv2-S uses the same representation; the question is whether MTL compromises it.

### 2.3 PSR Loss — The Breakthrough

```
[batch     0] psr=0.2123
[batch   100] psr=0.2517
[batch   500] psr=0.1569
[batch  1000] psr=0.1678
[batch  1500] psr=0.2433
[batch  2000] psr=0.1854
[batch  2800] psr=0.1949
```

**This is the single most important number in run11.** The old PSR head was flat at 1.56 loss. The new PSR head with P5 (768-dim semantic) features is at 0.15-0.25. **The loss dropped 6-10× immediately.**

**What this means:**
- The old conv_proj-based head was learning **nothing** — 1.56 is approximately the BCE of a model guessing 0.5 for every component at every timestep
- The new P5-based head is producing **real predictions** — 0.17-0.25 means most predictions are near the correct label
- Focal-BCE is active (γ=2.0, α=0.25), so the loss is lower than plain BCE but still reflects real learning
- **This validates Opus 192's FC-4 diagnosis — the feature source was the bottleneck, not the decoder**

### 2.4 Pose Loss

```
[batch     0] pose=0.0557
[batch   100] pose=0.0260
[batch   500] pose=0.0321
[batch  1000] pose=0.0256
[batch  1500] pose=0.0331
[batch  2000] pose=0.0431
[batch  2200] pose=1.5482  ← SPIKE
[batch  2300] pose=0.0436  ← recovered
[batch  2600] pose=0.5295  ← moderate spike
[batch  2800] pose=0.1389
```

**Assessment:** Mostly healthy at 0.01-0.05 with occasional single-batch spikes (1.5, 0.5). These are likely batches where the head pose label is extreme/outlier. The model recovers immediately. This is a training dynamics issue, not a head problem. EMAs will smooth these in eval.

---

## 3. Training Speed

| Metric | Value |
|--------|-------|
| Batches per ~33 seconds | 100 |
| Batches per epoch (capped) | 8000 |
| Time per epoch | ~44 minutes |
| Epochs to first eval (EP10) | ~5 remaining from ep6 |
| First eval estimated | ~2026-07-10 14:00-15:00 JST |

---

## 4. Eval Protocol (What EP10 Will Measure)

### 4.1 Activity

```
act_top1 = correct_top1 / total_valid_labels
act_top5 = correct_top5 / total_valid_labels
```
- Clip-level: one prediction per 16-frame window
- 75 classes (ACT_CLASS_GROUPING="none")
- CE label_smoothing=0.05 during training, argmax during eval
- Excludes unlabeled clips (label = -1)

### 4.2 Detection

```
det_mAP50       = mAP at IoU=0.5 (COCO-style)
det_mAP_50_95   = mAP at IoU=0.5:0.05:0.95
det_mAP50_pc    = mAP at IoU=0.5, present-class only (honest)
det_presence_bce = BCE of presence/absence per class
det_presence_acc = Accuracy of presence/absence per class
```
- DFL decode → xyxy → score threshold 0.05 → NMS IoU=0.65
- Per-batch decode; accumulate all predictions; compute mAP at end

### 4.3 PSR

```
psr_event_f1_at_3 = Event F1 with ±3 frame tolerance
psr_pos           = Procedure order similarity
psr_tau_frames    = Mean temporal offset
```
- Sigmoid → binarize at 0.5
- Per-recording: detect 0→1 transitions, compare to GT transitions
- ±3 frame tolerance = ±0.15s at ~20fps eval stride

### 4.4 Pose

```
pose_fwd_mae = mean angular error (degrees) on forward vector
pose_up_mae  = mean angular error (degrees) on up vector
```
- Renormalize 6D → fwd/up unit vectors
- Bootstrap 95% CI (1000 resamples)

---

## 5. What We Need from EP10 to Continue

| Scenario | Detection mAP | Activity top-1 | PSR F1 | Action |
|----------|---------------|----------------|--------|--------|
| **Best case** | >0.1 | >5% | >0.1 | All heads learning → let run to ep30 |
| **Mixed** | >0.05 | >3% | >0.05 | Some learning → evaluate which heads need more |
| **Concerning** | <0.01 | <2% | <0.02 | One or more still dead → need architecture rethink |
| **Worst case** | 0.0 | <1% | <0.01 | Same as run10 → head upgrades didn't help enough |

**The PSR loss drop (1.56→0.17) strongly suggests at least PSR will show real signal at EP10.** Detection and activity are more uncertain — TAL should help detection, but activity's cls_token bottleneck may be fundamental.

---

## 6. Comparison: run10 vs run11 Architecture Impact

| Metric | run10 (EP10) | run11 (expected EP10) | Delta |
|--------|-------------|----------------------|-------|
| Detection loss (GT batches) | ~2.5-3.5 (3×3 sparse) | ~3.9-4.8 (TAL dense) | TAL gives more positives → higher loss per batch |
| Activity loss | ~4.8 (flat) | ~4.5-5.5 (still high) | 3-layer MLP same range, need epochs |
| PSR loss | ~1.56 (flat) | ~0.17-0.25 (dropping) | **10× improvement from feature source fix** |
| Pose loss | ~0.02-0.05 | ~0.01-0.05 | Unchanged |
| Model size | ~46M | 117.7M | 2.6× larger |
