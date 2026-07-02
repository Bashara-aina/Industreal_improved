# 03 — Loss Analysis & Multi-Task Balancing

## Critical Gaps Discovered by Agent Review

The following 8 issues undermine all loss-balancing analysis below. They must be resolved before multi-task balancing can be properly evaluated or tuned.

1. **Kendall log_vars are NEVER logged.** The Kendall log_var values (lv_det, lv_hp, lv_act, lv_psr) are never emitted in standard training output. Without these, we cannot determine if multi-task balancing is functioning or if bounds are clamping adaptation. This is the single most important missing diagnostic.

2. **PSR zero is STRUCTURALLY zero, not a sampling artifact.** On non-seq batches under USE_PSR_TRANSITION=True, PSR loss is EXPLICITLY zeroed (losses.py line 1449). Combined with detach_psr_fpn=True, PSR contributes ZERO gradient to any shared (backbone) parameter on ANY batch, ever. PSR is a pure downstream beneficiary of other tasks' feature learning.

3. **Activity receives ~20x less gradient signal than PSR.** Activity head (687K params, ACTIVITY_LOSS_WEIGHT=1.0, KENDALL_LOG_VAR_MIN_ACT=-0.5) vs PSR (3.08M params, PSR_WEIGHT=20.0 on seq batches). Effective gradient magnitudes: activity ~0.9 vs PSR ~10.0 on seq batches. This massive imbalance likely explains slow activity convergence.

4. **Pose convergence may be magnitude-matching, not directional learning.** GT vector norms are 0.014-0.030; loss sqrt(0.1) ≈ 0.3 means errors 10-20x target magnitude. x0.001 scaling reduces an already small loss to near-zero. With KENDALL_LOG_VAR_MAX_POSE=3.0 (20x suppression), pose is effectively gradient-starved and may output near-constant predictions.

5. **Combined metric sensitivity is untested.** Pose contributes only ~2% of total (0.15*(1/14+1) ≈ 0.01 out of ~0.55). The weight scheme (det=0.30, act=0.35, pose=0.15, psr=0.20) has never been sensitivity-analyzed. It may be dominated by a single head's noise.

6. **Effective batch 48 vs paper's 16 creates LR scaling mismatch.** Per linear scaling rule, LR should be 3x higher. Without adjustment, the model sees 1/3 the parameter updates per epoch vs the reference — slowing all heads.

7. **PSR weight of 0.20 is unsubstantiated.** This head contributes zero backbone gradient and trains on 50% of batches. Justification is needed for its evaluation weight.

8. **Detection-to-backbone gradient ratio is ~0.001 (1000:1).** Detection is effectively a free rider on other heads' feature learning. If other heads underperform, the backbone receives almost no task-relevant gradient.

## Loss Formulas Per Head

### Detection Loss
```
L_det = FocalLoss(cls_pred, cls_gt, α=0.25, γ=2.0) + IoULoss(reg_pred, reg_gt) + GIoULoss(reg_pred, reg_gt)
```

Detection loss logging shows `det=c(g)` where `c` is class loss and `g` is GIoU regression loss. The ratio `c/g` is typically 3:1 to 6:1, meaning classification dominates detection loss.

### Activity Loss
```
L_act = CrossEntropyLoss(act_logits, act_gt)
```
The activity uses LDAM (label-distribution-aware margin) with DRW (deferred re-weighting). The LDAM margin is computed from class frequencies. DRW is disabled (USE_LDAM_DRW=False) in current runs.

### PSR Loss
```
L_psr = FocalLoss(psr_logits, psr_gt, γ_pos=0, γ_neg=2) + temporal_smoothness_loss + transition_consistency_loss
```
PSR has three components:
1. **Focal loss** on per-frame PSR states
2. **Temporal smoothness** — penalizes state flips in neighboring frames
3. **Transition prior** — learned order constraints (went from/to transitions)

The PSR_SEQ_EVERY_N_BATCHES=2 means every other batch is PSR-sequence-only (not detection), where the model sees 8-frame sequences via USE_PSR_SEQUENCE_MODE=True.

### Head Pose Loss
```
L_pose = WingLoss(forward_pred, forward_gt, weight=confidence) + 
         WingLoss(up_pred, up_gt, weight=confidence) +
         WingLoss(position_pred, position_gt, weight=confidence)
L_pose_total = L_pose × 0.001 (per paper)
```

## Combined Metric Formula

```
combined = (0.30/total_w) × mAP50 + (0.35/total_w) × act_F1 + (0.15/total_w) × pose_acc + (0.20/total_w) × psr_F1

where:
  pose_acc = 1/(1 + MAE_head_pose)
  total_w = sum of active head weights (all 4 in RF4 = 1.0)
```

### Weight Definitions (train.py lines 164-167)
- _W_DET = 0.30
- _W_ACT = 0.35
- _W_POSE = 0.15
- _W_PSR = 0.20

## Current Loss Analysis

### Loss Component Magnitudes (epoch 2, typical non-seq batch)

| Component | Raw Value | Scaled by Kendall | Gradient Impact |
|---|---|---|---|
| Det(c) | ~2.0 | Kendall-adjusted | Moderate |
| Det(g) | ~0.4 | Kendall-adjusted | Low |
| Pose | ~0.1-1.5 | Kendall-adjusted + ×0.001 | Very low (converged) |
| Act | ~1.7 | Kendall-adjusted | Moderate (still random) |
| PSR | ~0.0 (non-seq) / ~0.5 (seq) | Kendall-adjusted | Low except seq batches |
| Weight Decay | ~0.29 | N/A | Constant regularizer |

### Critical Observations

1. **PSR is zero on non-seq batches** — This is EXPECTED behavior. The PSR head only gets gradient on sequence batches (every 2nd batch). On non-seq batches, `seq=1` indicates a PSR-only forward pass where detection/pose/act gradients are explicitly zeroed (criterion.train_psr=True, all others False). The PSR logits show healthy activations (pre_linear std=1.6, post_gelu mean=0.26) on seq batches.

2. **Pose has converged** — From 8.38 in early epoch 0 to 0.09-1.5 in epoch 2. This means the pose head has learned the mapping and is now outputting near-constant predictions. The validation MAE of 13.4° forward angular and 14.8° up angular is reasonable for a single-frame estimator.

3. **Activity loss is NOT decreasing** — It oscillates between 0.7-2.1 across 3 epochs without trend. This is the main concern. The activity head:
   - Has 5-epoch ramp-up (ACT_RAMP_EPOCHS), meaning it's only been fully active for ~0.5 epochs
   - Uses verb-grouping (41 outputs from 69 raw classes) which may confuse the classifier
   - On epoch 0 val: predicted only 1 class (class 12, `put`) for all 4000 frames
   - On epoch 2 val: predicted 3 classes (class 0, 63, 12) with entropy=1.036 nats
   - **Slight improvement** — diversity went from 1→3 classes, entropy from 0→1.036 nats

4. **Detection is stable but not improving** — c=1.2-2.6 consistently. The detection head may need more backbone adaptation (backbone LR=5e-5, 10x lower than heads).

5. **Kendall log_vars** are not being logged in standard output — need to verify they're adapting correctly. At init: det=0.0, head_pose=-1.0, act=0.0, psr=0.0. The KENDALL_HP_PREC_CAP ensures head pose precision doesn't exceed detection.

## What the Loss Numbers Tell Us vs What They Don't

### Known (from training)
- All 5 heads have ALIVE gradients (verified by liveness probes)
- Pose is essentially learned
- PSR is alive on seq batches
- Detection produces boxes with IoU>0.5 after epoch 0 (LOCALIZING verdict from DET_PROBE)
- Activity is struggling but showing microscopic improvement

### Unknown (no validation data yet)
- Actual detection mAP@0.5 (epoch 3 will be first real measurement)
- Actual PSR F1 with temporal consistency
- Actual head pose MAE with proper evaluation
- Activity diversity beyond the 200-batch gated eval (which showed collapse)
- Combined metric value
- Whether Kendall weighting is properly balancing or if a head is being suppressed
