# 70: Master Prompt — Round 3 Opus Consult [2026-07-01]

## The Problem

After implementing all of Opus's RF10 decisions (files 62-69), we completed two full
epoch-0 training runs. Both collapsed:

| Run | Loss Function | pred_distinct | entropy | Macro-F1 | Verdict |
|-----|--------------|--------------|---------|----------|---------|
| Run 1 | CE + label_smooth=0.1 | 1/75 | 0.000 | 0.0002 | Total collapse |
| Run 2 | CB-Focal (β=0.999, γ=2.0, dropout=0.3) | 2/75 | 0.152 | 0.0000 | Marginal improvement, still collapsed |

**The task:** 74-class activity recognition from egocentric video, multi-task with
detection/pose/PSR. Simple MLP head (LayerNorm → Linear(512→256) → GELU → Dropout →
Linear(256→75)), 150K params, trained with all 5 heads sharing ConvNeXt-Tiny backbone.

## What We've Tried (Listed for Opus to Evaluate)

### Architecture (already on disk)
1. **ACTIVITY_HEAD_SIMPLE=True** — bypasses TCN+ViT, uses 150K MLP directly on joint
   projection features (model.py:1374-1420). Logit bias=-0.5 init.
2. **Feature bank bypassed** when STAGED_TRAINING=False (model.py:2193-2198).
3. **Gradient centralization** on activity head params (train.py:1280-1295, 1730-1745).

### Loss function (tried)
4. **CE + label_smoothing=0.1** (losses.py:1054) — collapsed to 1 class.
5. **CB-Focal** β=0.999, γ=2.0 (losses.py:1057, wired 2026-06-30) — collapsed to 2 classes.

### Training hyperparameters (current)
6. ACTIVITY_LR_MULTIPLIER=1.0, ACTIVITY_HEAD_GRAD_CLIP=1.0
7. ACTIVITY_HEAD_DROPOUT=0.3 (was 0.2)
8. ACTIVITY_LOSS_WEIGHT=0.8
9. USE_CB_FOCAL_ACT=True, CB_FOCAL_BETA=0.999, CB_FOCAL_GAMMA=2.0
10. OneCycleLR pct_start=0.1, backbone LR=5e-5, head LR=5e-4

### Data
11. NUM_WORKERS=0 (deadlock fix), RAM_CACHE_MAX_IMAGES=8000
12. DET_GT_FRAME_FRACTION=0.4
13. WeightedRandomSampler for class balance
14. 3,667 train frames, 1,928 val frames. 46/72 classes have <1% annotation.

## What Has NOT Been Tried (Questions for Opus)

### Loss function variants
Q1. **LDAM-DRW** (already implemented in losses.py:1048, config.py:763):
    Currently USE_LDAM_DRW=False. Should we try LDAM-DRW with reduced LDAM_S
    (e.g. 5 instead of 30)? Or CB-Focal with higher gamma (5.0)?

Q2. **Class-balanced weights on CE** (already implemented in losses.py:1125-1135):
    The code path at losses.py:1125-1135 applies CB weights to standard CE loss.
    It runs after set_class_counts() when act_loss_fn is CE. This was active in
    Run 1 (CE+label_smooth). Did the CB weighting work? The effective number
    formula with β=0.99 gives weights that cap at 50× via the clamp. Check if
    the weights actually propagated to the loss function.

Q3. **Focal loss without class-balancing**: Just CE + focal γ=2 without the CB
    effective-number weighting. The CB weighting may be over-regularizing the
    already-rare classes. Should we test vanilla Focal Loss (no CB)?

### Architecture variants
Q4. **Deeper MLP**: Current = 2-layer. Would 3-layer (512→512→256→75) with
    residual connections help? More capacity on 3.7k frames risks overfitting,
    but the current 2-layer may lack capacity to learn 74 classes.

Q5. **Wider MLP**: 512→512→75 instead of 512→256→75? 400K params vs 150K.

Q6. **Add temporal context without feature bank**: The root cause of the TCN+ViT
    collapse was that the shuffled sampler fed non-temporal sequences. But
    per-frame MLP has zero temporal context. Could we add a lightweight temporal
    module (e.g. 1D conv with kernel=3 over 3 consecutive frames) that does NOT
    depend on the feature bank's recording_id ordering? The frames within a batch
    can be consecutive if we disable shuffling for activity only.

### Data-level interventions
Q7. **Train/val split for activity**: The val set has 1,928 frames from different
    recordings. But the train set only has 3,667 frames. Is the train set simply
    too small for 74-way classification? What's the minimum annotated frames per
    class needed?

Q8. **Downsample activity classes**: Merge semantically similar classes. The 74
    classes come from 36 assembly actions × 2 hands + "NA". Many are visually
    similar (e.g. "take_pin_short" vs "take_pin_long"). Should we group into
    10-15 super-classes?

Q9. **Oversample minority classes**: Every class with <10 frames gets copied 10×
    in the dataset. Does this help, or does it just cause overfitting?

Q10. **Synthetic data augmentation**: Flip, color jitter, cutout. Currently
     USE_MIXUP=False in RF4 preset. Would mixup or cutout help the activity head
     generalize?

### Multi-task gradient dynamics
Q11. **Freeze backbone for activity only**: The backbone carries gradients from
     all 5 heads. Detection dominates (3.5× larger gradient than activity).
     If we freeze backbone for activity head's first 3 epochs, does activity
     learn its own projection features without competition?

Q12. **ACTIVITY_LOSS_WEIGHT=3.0**: Currently 0.8. Raising it gives activity more
     influence on the shared backbone. With CB-Focal producing 0.02-0.3 loss
     (vs detection 2-4), Kendall weighting already re-balances. But would a
     manual 3× boost help?

### Paper strategy
Q13. **4-task paper**: If activity remains collapsed, we drop it and document
     the collapse as a training pathology (§5). The paper becomes:
     "A multi-task assembly verification model on a single consumer GPU:
     which tasks share a backbone gracefully, which don't, and why."
     Is this publishable at AAIML?

Q14. **5-task paper with "activity failed" admission**: Instead of dropping,
     keep activity in the title but report it as a verified failure case.
     Both CE and CB-Focal collapsed → this is a genuine negative result.
     Is negative-result publishing acceptable at AAIML?

## Data Constraint (Must Read)

The binding constraint is NOT architecture:

```
Action classes: 75 (class 0 = "NA", 74 real actions)
Training frames: 3,667
Classes with >100 frames: 5
Classes with 10-99 frames: 21
Classes with <10 frames: 48
Classes with 1 frame: 7
```

The ConvNeXt-T backbone is ImageNet-pretrained (28M params, 1,000-class classifier
head). The bottleneck is that per-frame 74-way classification on 3.7k frames with
46/72 classes <1% is an extremely long-tailed problem.

## What We Need From Opus

1. **Is the activity task salvageable** with the interventions listed above
   (LDAM-DRW, higher gamma, deeper MLP, temporal context, data aug)?
   Or is 3.7k frames of 72-way long-tail data fundamentally insufficient?

2. **If salvageable**: which SINGLE intervention has the highest probability of
   success? We have ~3 hours of GPU compute for experiments. We need a strong
   recommendation, not a grid search.

3. **If not salvageable**: confirm the 4-task paper plan is viable for AAIML.
   Does the documented activity collapse strengthen or weaken the paper?

4. **For the 4-task paper**: should we keep the activity head code in the
   repository (for reproducibility) but exclude results from the paper?

## File Reference

- Master prompt (this file): `70_MASTER_PROMPT_OPUS_ROUND3.md`
- Collapse analysis: `71_ACTIVITY_COLLAPSE_DEEP_ANALYSIS.md`
- Alternative architectures: `72_ALTERNATIVE_ARCHITECTURES.md`
- Data strategies: `73_DATA_AUGMENTATION_STRATEGIES.md`
- 4-task paper fallback: `74_4_TASK_PAPER_FALLBACK.md`

## Git History

All changes from rounds 1-3 are on main:
```
18e0160 fix: apply simple_classifier init (logit bias=-0.5)
8207632 fix: bypass feature bank in non-staged mode
ea325d2 chore: raise RAM_CACHE_MAX_IMAGES to 8000
(plus 69_OPUS_RESPONSE_FINAL.md changes — subprocess eval, diversity instrumentation,
 CB-Focal wiring, per-epoch checkpoints, auto-load crash recovery)
```
