# 71: Activity Collapse — Deep Analysis of Both Runs [2026-07-01]

## Run 1: CE + label_smooth=0.1 (baseline)

### Config
```
ACTIVITY_HEAD_SIMPLE=True
ACTIVITY_HEAD_DROPOUT=0.2  (default from nn.Sequential)
ACTIVITY_LOSS_WEIGHT=0.8
ACTIVITY_LR_MULTIPLIER=1.0
GRAD_CLIP=1.0, ACTIVITY_HEAD_GRAD_CLIP=1.0
Loss: CrossEntropyLoss(label_smoothing=0.1)
  → then replaced by CB-weighted CE at train time (losses.py:1125-1135)
```

### Training Loss Trajectory
```
Batch 0:     act=1.4673    (random init, high entropy)
Batch 100:   act=1.3-1.5   (stable, normal range)
Batch 1000:  act=1.0-1.5   (stable, no collapse visible in loss)
Batch 2000:  act=1.0-1.5   (stable — looks healthy!)
Batch 3500:  act=0.85-1.5  (still stable)
Batch 5000:  act=0.89-1.5  (still stable!)
```

**Key observation:** The training loss never looked bad. It stayed at 1.0-1.5 nats
throughout epoch 0 — which is exactly what random initialization looks like for
75-class CE. There was NO signal in the training loss that the model had collapsed
to 1 class.

### Step 2500 Validation
```
pred_distinct: 1/75 (class 66 = "plug_small_screw_pin", 100% of frames)
entropy:       0.000 nats
act_macro_f1:  0.0002
```

The model predicted class 66 for ALL 200 validation frames. Class 66 appears in
1,044 training frames (28% prevalence — the most common class). The model learned
to predict the majority class and stopped exploring.

### Gradient Liveness at Same Step
```
Run 1 (CE):    activity_head:ALIVE[1.08e-02]
```

Gradient was flowing (1.08e-02), but the network had already converged to the
trivial solution. The gradient wasn't zero — it was just pushing along the
majority-class attractor.

## Run 2: CB-Focal (β=0.999, γ=2.0, dropout=0.3)

### Config changes from Run 1
```
USE_CB_FOCAL_ACT=True
CB_FOCAL_BETA=0.999
CB_FOCAL_GAMMA=2.0
ACTIVITY_HEAD_DROPOUT=0.3
```

### Training Loss Trajectory
```
Batch 0:     act=0.0331   (CB-Focal starts much lower due to γ down-weighting)
Batch 100:   act=0.02-0.17 (oscillating, some batches higher than others)
Batch 1000:  act=0.02-0.32 (diverse range — good sign)
Batch 2000:  act=0.02-0.17 (still oscillating)
Batch 3500:  act=0.02-0.31 (diverse)
Batch 4500:  act=0.02-0.32 (still diverse)
```

**Key observation:** The CB-Focal loss oscillated between 0.02 and 0.32 throughout
the entire epoch. This was PROMISING — it suggested the network was trying different
classes (if it had collapsed to 1 class, the loss would flatline at a consistent
value). The oscillation meant predictions were changing.

### Step 2500 Validation
```
pred_distinct: 2/75 (class 73 = 96.5%, class 52 = 3.5%)
entropy:       0.152 nats
act_macro_f1:  0.0000
```

**Marginal improvement:** 2 classes instead of 1. Class 73 ("loosen_tooth_washer")
and class 52 ("pull_small_screw_pin"). Entropy of 0.152 is higher than 0.000 but
still far below the 1.5 nats gate.

### Gradient Liveness at Same Step
```
Run 2 (CB-Focal):    activity_head:ALIVE[1.49e-01]
```

The CB-Focal produced **10× more gradient** than CE (0.149 vs 0.0108). This is
evidence that the focal loss is correctly up-weighting hard examples. But even
with 10× more gradient, the network still collapsed to 2 classes.

## Root Cause Analysis

### Data Constraint (Primary)

The fundamental problem:

```
Total training frames:     3,667
Real action classes:       74
Frames per class (median): 8
Classes with >100 frames:  5
Classes with <10 frames:   48
```

A 150K-param MLP with LayerNorm and dropout processes ONE frame at a time with
NO temporal context. Each of 74 classes needs enough exemplars for the network
to learn discriminative features. The majority class (class 66) has 1,044 frames
(28% of data). The median class has 8 frames (0.2% of data). The network learns
to predict the majority class because it's the only one with enough support.

### Architecture Constraint (Secondary)

The simple MLP head receives a 512-dim joint projection. This projection is:
```
det_conf ⊕ GAP(c5_mod_blend) ⊕ GAP(p4)
```
from model.py activity_proj. The projection is trained jointly with all 5 heads.

The detection head dominates the shared backbone (gradient 2.8-3.5 vs activity
0.0108-0.149). Even with gradient centralization, the backbone features are
optimized for detection, not activity. The MLP head tries to classify 74 classes
from features that are primarily shaped by detection loss.

### Loss Function Constraint (Tertiary)

CB-Focal partially addresses the imbalance (2 classes vs 1, 10× more gradient)
but did not overcome it. The effective-number weighting caps at 50×, but for
classes with <10 frames, even 50× weighting may be insufficient.

## Failure Mode Categorization

```
Type A: "Majority-class Attractor"
- Network predicts the most common class for all frames
- Training loss looks normal (1.0-1.5 CE, 0.02-0.3 CB-Focal)
- Gradient flows but pushes along the attractor, not away from it
- CE run: Type A
- CB-Focal run: Type A (partial — escaped to 2 classes)

Type B: "Zero-Gradient Dead Head"
- Network predicts uniform distribution or zeros
- Loss does not decrease
- Gradient is DEAD (0.0)
- NOT our case — activity gradient is always ALIVE

Type C: "Oscillating Confusion"
- Network predicts many classes but all with low confidence
- Loss is high but entropy is high
- Gradient is healthy
- We WANT to reach this state — it means the network is trying
```

We are firmly in Type A. The network needs to be PUSHED OUT of the attractor.

## Candidate Interventions (Ranked by Likely Impact)

| # | Intervention | Expected Effect | Risk |
|:-|--------------|----------------|------|
| 1 | Higher γ (5.0) | More aggressive down-weight of easy (majority) examples | May destabilize training |
| 2 | LDAM with reduced s (5-10) | Margin-based separation for rare classes | More stable than CB-Focal? |
| 3 | Oversample minority 10× | More gradient updates for rare classes | Overfitting on repeated frames |
| 4 | Merge to 10-15 super-classes | Fewer classes, more examples per class | Loses fine-grained actions |
| 5 | Temporal context (3-frame 1D conv) | Temporal signal without feature bank | May not help per-frame |
| 6 | Deeper MLP (512→512→256→75) | More capacity | Overfitting risk on 3.7k frames |
| 7 | Mixup augmentation | Synthetic training examples | May help generalization |
| 8 | Freeze backbone for activity epochs | Features specialized for activity | Hurts detection/pose/PSR |
| 9 | ACTIVITY_LOSS_WEIGHT=3.0 | More gradient on shared backbone | May hurt other tasks |
| 10 | Drop activity (4-task paper) | Remove the failing task | Paper scope reduction |
