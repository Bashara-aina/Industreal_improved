# 04 — Validation Metrics & Gate RF4 Status

## Current Validation State

**VALIDATION HAS NEVER SUCCEEDED on this codebase.** Every attempt either:
1. Hung on CUDA kernel during evaluation
2. Was killed by watchdog after eval completed (>1200s stale heartbeat)
3. Was killed manually for restart

We have partial validation data from two step-val attempts:

### Step Val at global_step=2500 (epoch 1, batch ~600)

```
[STEP VAL gs=2500] det_mAP50=0.0000  act_F1=0.0000  psr_F1=0.0000  pose_MAE=0.0000
```

This is a **gated eval** (200 batches only, GATE_EVAL_MAX_BATCHES=200). All zeros because the model was only 600 batches into epoch 1 with OneCycleLR still in warmup phase. NOT representative.

### Step Val Activity Results at gs=2500

```
[EVAL COLLAPSE] activity head predicts only 1/69 classes (top-1 class=12 with 100.0% of frames)
[DIVERSITY] pred_distinct=1/69  entropy=-0.000 nats  gt_distinct=55/69
[GAP-B] Activity Segment — Top-1: 0.0000  Top-5: 0.0000  Segments: 1915
Head Pose [unit_vectors_ok] — Forward angular: 13.3958 deg  Up angular: 14.8029 deg  Position: 130.4724 mm  fwd_raw: n/a (L1)  Overall raw: 0.1006
ASD — mAP@0.5: nan  mAP@[0.5:0.95]: nan  mAP@0.5 (all frames): nan
```

The `nan` for mAP is because DET_METRICS_EVERY_N=3 skips the full detection eval. Head pose MAE of 13.4° forward angular is reasonable for epoch 1 with 2 warmup epochs remaining.

### Step Val at global_step=5000 (epoch 2, batch ~1220)

```
[EVAL COLLAPSE] activity head predicts only 3/69 classes (top-1 class=0 with 50.5% of frames)
[DIVERSITY] pred_distinct=3/69  entropy=1.036 nats  gt_distinct=56/69
```

**Activity improvement detected:** went from 1→3 predicted classes, entropy from 0→1.036 nats. The activity head is beginning to discriminate but still collapsed for practical purposes.

### DET_PROBE Results (epoch 0 validation, max 500 batches)

Every DET_PROBE line showed `verdict: LOCALIZING` with:
- 1000-4000 predictions at IoU>0.5 per 8-frame batch
- bestIoU_max consistently 0.85-0.97 (excellent localization quality)
- score_p50: ~0.036, score_p99: ~0.147, score_max: 0.47-0.76
- This means detection is producing GOOD boxes but with LOW CONFIDENCE scores

The low confidence (score_p50=0.036) at threshold DET_EVAL_SCORE_THRESH=0.001 means most predictions pass the filter but the PR curve will suffer from high false positive rate. The scores should separate as training progresses.

**CORRECTION: What score_p50=0.036 Actually Means.** The median score of ~0.036 across all ~1.38M anchor locations is expected to be near the bias initialization (sigmoid(-3.4) ≈ 0.033) because ~99.3% of anchors are background. The presence of many low-confidence predictions is the expected distribution for a detection head with heavy class imbalance. The actual pathology is that scores are NOT SEPARATING for positive vs. negative anchors. In a healthy detector, the small fraction of positive anchors should have significantly higher scores than the vast background. Monitor score_p90 and score_max for separation from score_p50 — this spread is the true indicator of detection health.

## Expected Target Ranges (from Architecture Verification Report)

| Metric | Target Range | SOTA Baseline | Gap |
|---|---|---|---|
| ASD mAP@0.5 | 70-78% | 83.8% (YOLOv8m) | -5 to -14% |
| ASD mAP@[0.5:0.95] | Not specified | — | — |
| Activity Top-1 (RGB) | 55-63% | 65.25% (MViTv2) | -2 to -10% |
| Activity Top-1 (w/ VideoMAE) | 62-68% | 65.25% | Comparable |
| PSR F1@±3f | 0.50-0.65 | 0.731 (B2) | -0.08 to -0.23 |
| PSR POS | 0.70-0.80 | 0.816 (B2) | -0.02 to -0.12 |
| Head Pose Fwd Angular | 5-10° | — | — |
| Head Pose Position | <50mm | — | — |
| Combined Metric | 0.50-0.60 | — | — |

These are **projections from the paper, not measurements**. No run has ever produced these values.

## Gate RF4 Criteria

The "gate" is the combined metric improving over best_metric. Early stopping patience=10 epochs. The combined metric computation:

```python
combined = (0.30/1.0) * det_mAP50 + (0.35/1.0) * act_F1 + (0.15/1.0) * (1/(1+pose_MAE)) + (0.20/1.0) * psr_F1
```

For example, if at convergence:
- det_mAP50=0.75, act_F1=0.55, pose_MAE=8°, psr_F1=0.55
- combined = 0.30*0.75 + 0.35*0.55 + 0.15*(1/9) + 0.20*0.55
- combined = 0.225 + 0.193 + 0.017 + 0.110 = 0.544

The gate passes when combined > best_metric (starts at 0). Patience fires if no improvement in 10 epochs. With 100 epochs total, there's plenty of runway.

## Activity Collapse Context: The Ramp Window

**CORRECTION: Epoch 0-2 activity results are NOT true collapse.** The activity head uses ACT_RAMP_EPOCHS=5, meaning:
- Epoch 0: activity loss weight = 0% — classifier receives NO training signal
- Epoch 1: ~20% weight — only mild supervision
- Epoch 2: ~40% weight — insufficient for full convergence
- Epoch 3: ~60% — first meaningful gradient
- Epochs 5+: 100% — full supervision

At epoch 1-2 where step-val data was collected, the activity head has received ~0-40% of intended supervision. The "collapse" to 1-3 classes is expected for a barely trained classifier. Improvement from 1→3 classes with entropy 0→1.036 is a healthy sign.

**Do not use epoch 0-2 activity metrics for any go/no-go decisions.** First meaningful eval at epoch 3, first reliable eval at epoch 5+.

## Timeline to First Real Validation

- **Epoch 3** (~1 hour remaining from current epoch 2 batch 150): First VAL_EVERY=3 validation
- Evaluation scope: Both VAL_EVERY=3 and DET_METRICS_EVERY_N=3 fire at same epochs. **CORRECTION: GATE_EVAL_MAX_BATCHES=200 is dead code at this cadence** — every val runs the full 500-batch evaluation. If intent was fast gate-checks, schedules need offsetting.
- The epoch 3 val will show: activity macro-F1, head pose MAE, PSR F1, but NOT detection mAP
- Full detection mAP at epoch 6 (DET_METRICS_EVERY_N=3)
- **Note:** 200 batches at VAL_BATCH_SIZE=8 evaluates only 1,600 of 1,928 validation frames (83%). Tail activity classes appear <10 times in this sample. Metrics like macro-F1 are sensitive to undersampling.
- **CORRECTION on LR schedule:** OneCycleLR pct_start may be 0.3 (optimizer.py line 58), not 0.1. If so, peak LR at epoch 31, not epoch 10 — shifting the meaningful validation timeline dramatically.

## Healthy vs. Intervention Thresholds

### Epoch 3 Thresholds (First Valid Validation)

At epoch 3, activity has received ~60% of ramp and is past warmup start:

| Metric | Healthy | Needs Intervention | Notes |
|---|---|---|---|
| pred_distinct | ≥ 5 classes | 3-4 classes | Current (ep 2): 3. Should double |
| Entropy | ≥ 1.5 nats | 1.0-1.5 | Current (ep 2): 1.036 |
| Activity macro-F1 | ≥ 0.05 | < 0.02 | First non-trivial F1 |
| Head pose MAE | < 15° fwd | > 20° | Current (ep 1): 13.4° — healthy |

### Epoch 6 Thresholds (First Full Detection Eval)

By epoch 6, activity ramp is complete (100% since epoch 5). First DET_METRICS full eval:

| Metric | Paper-Quality Trajectory | Acceptable | Intervention |
|---|---|---|---|
| Detection mAP@0.5_pc | ≥ 0.40 | 0.25-0.40 | < 0.15 |
| pred_distinct | ≥ 15 classes | 8-15 | < 8 |
| Activity macro-F1 | ≥ 0.15 | 0.08-0.15 | < 0.05 |
| Combined metric | > 0.25 | 0.15-0.25 | < 0.15 |
| Head pose MAE | < 12° | 12-20° | > 25° |

If combined < 0.15 at epoch 6, activate Tier 1 contingency (fix FeatureBank, det_conf, ViT attention).

## Why Validation Kept Failing (All Fixed)

1. **SIGALRM in thread:** Segment metrics code tried to set SIGALRM inside ThreadPoolExecutor thread which raises ValueError. The exception handler set `_run_seg_metrics=False` but the code still ran `compute_activity_segment_metrics()` without timeout protection. **FIXED: gated behind `if _seg_have_alarm:`.**

2. **Stale heartbeat after eval:** Watchdog checks heartbeat file written every 100 training steps. After 20+ minutes of eval with no training steps, heartbeat was 1217s stale. On eval end, watchdog immediately killed. **FIXED: fresh heartbeat written immediately after eval ends.**

3. **Subprocess eval GPU contention:** `USE_SUBPROCESS_EVAL=true` spawns a subprocess that tries to use GPU 0, but main training also uses GPU 0 → CUDA context deadlock. **FIXED: disabled subprocess eval, running eval in main thread with all other fixes.**

4. **Epoch 0 evaluation is worthless:** Activity shows 1-class collapse, all metrics near zero. **FIXED: VAL_EVERY=3 skips epochs 0, 1, 2.**

5. **Watchdog timeout too tight:** 1200s (20 min) was barely enough for 500-batch eval at FP32. **FIXED: increased to 1800s.**
