# 56: Activity Head Collapse — Root Cause Analysis [2026-06-30]

## Status: UNSOLVED — Training Has NEVER Produced Meaningful Activity Metrics

After 6 full training runs across 10 days (Jun 20-Jun 30), the activity head has NEVER
produced an act_macro_f1 > 0.0022. The head always collapses to predicting 1-4 out of 75
possible classes, with the dominant class receiving 87-100% of frame assignments.

## Timeline of Failed Fixes

### Attempt 1: Default config (Jun 20, RF1-RF2)
- ACTIVITY_LR_MULTIPLIER: not defined (shared PSR param group, head_lr=5e-4)
- ACTIVITY_GRAD_BLEND_RATIO: 0.05 (5% gradient to backbone)
- Result: act_macro_f1 = 0.0 (not trained at all in RF1-RF2)

### Attempt 2: RF3 with blend=0.10 (Jun 28)
- ACTIVITY_GRAD_BLEND_RATIO: 0.10
- ACTIVITY_HEAD_GRAD_CLIP: 0.3
- Result: act_macro_f1 = 0.0022, act_top1 = 0.125 (from Epoch 2 metrics file)
  - NOTE: The 0.125 act_top1 from RF3 stage_history conflicts with the epoch 2 validation
    which shows act_clip=0.0000 and act_macro_f1=0.0022. This may indicate the stage_history
    recorded a different metric or data subset.

### Attempt 3: RF4 with blend=0.70 (Jun 29 pre-consult)
- ACTIVITY_GRAD_BLEND_RATIO: 0.70
- ACTIVITY_HEAD_GRAD_CLIP: 0.3
- ACTIVITY_LR_MULTIPLIER: 1x (shared PSR group)
- Result: [EVAL COLLAPSE] predicts 4/75 classes, act_macro_f1 = 0.0000

### Attempt 4: RF4 with blend=1.0 + 3x LR + GC (Jun 30, session 1)
- ACTIVITY_GRAD_BLEND_RATIO: 1.0
- ACTIVITY_HEAD_GRAD_CLIP: 1.0
- ACTIVITY_LR_MULTIPLIER: 3.0 (separate param group)
- Gradient centralization on activity head params
- Result: [EVAL COLLAPSE] predicts 4/75 classes, class 12=87.5%, act_macro_f1 = 0.0005

### Attempt 5: RF4 with blend=1.0 + 10x LR + GC (Jun 30, session 2)
- Same as Attempt 4 but ACTIVITY_LR_MULTIPLIER = 10.0
- Result: [EVAL COLLAPSE] predicts 4/75 classes, class 12=98%, act_macro_f1 = 0.0005
- Gradient norm at LIVENESS_GRAD step=0: activity_head=0.010 (unchanged across ALL attempts)

### Attempt 6 (CURRENT): RF4 with classifier reinit + 20x LR + GC (Jun 30, session 3, 18:50 UTC)
- activity_head.activity_classifier.2.weight: reinitialized (xavier uniform, bound=0.1083)
- activity_head.activity_classifier.2.bias: zeroed
- activity_head.activity_classifier.0.weight/bias: reset to LayerNorm defaults
- ACTIVITY_LR_MULTIPLIER: 20.0 → activity_head_lr = 1.0e-2
- PID 3369770, Epoch 3, awaiting first validation result

## Measured Data

### Gradient Liveness (ALL attempts, ALL epochs, step 0)
```
activity_head: ALIVE[0.0102]/ALIVE[0.00255]  ← UNCHANGED across ALL configs
```

### Comparison with Other Heads
```
detection_head:    ALIVE[0.479]/ALIVE[0.060]   → 47x activity
head_pose_head:    ALIVE[0.561]/ALIVE[0.078]   → 55x activity
pose_head:         ALIVE[0.443]/ALIVE[0.004]   → 43x activity
psr_head:          ALIVE[3.180]/DEAD[1.5e-08]  → 312x activity (per-param dead, total alive)
backbone:          ALIVE[2.366|n=178]
fpn:               ALIVE[1.154|n=16]
```

The activity gradient norm of 0.010 is STATIONARY across all config changes. This is NOT
a learning rate problem — it is a structural gradient starvation issue.

### Effective Weight Updates
At ACTIVITY_LR_MULTIPLIER = 20.0: 0.010 × 1.0e-2 = 1.0e-4 per step
At ACTIVITY_LR_MULTIPLIER = 10.0: 0.010 × 5.0e-3 = 5.0e-5 per step
At ACTIVITY_LR_MULTIPLIER =  3.0: 0.010 × 1.5e-3 = 1.5e-5 per step
Detection comparison: 0.479 × 5.0e-7 (backbone_lr) = 2.4e-7 per step (but AdamW normalizes)

### Activity Head Architecture (8.2M params = 15.2% of total)
```
proj_features (Linear 1048→512) → 2-layer ViT → TCN (depthwise+pointwise) → LayerNorm → Linear(512→75)
    1048-D input: c5_mod_blend (512) + det_conf (512) + p4_pooled (24)
```

### Training Loss Breakdown (epoch 2, batch 3469)
```
Train: loss=8.591  det=1.124  pose=1.699  act=0.600  psr=1.147
```

Activity loss = 0.600. This is NOT zero — the loss computes and backpropagates.
The gradient just vanishes somewhere in the chain:
  CE → classifier → LayerNorm → TCN → ViT → proj_features → c5_mod_blend → backbone

## Diagnostic Questions for Opus

1. **Why is the gradient norm 0.010 regardless of LR, blend ratio, clip, or reinit?**
   - This suggests the gradient is being numerically determined by the architecture
     (e.g., LayerNorm in eval mode, or a gradient-blocking operation).

2. **Could the LayerNorm in activity_classifier be in eval mode** while training?
   - model.eval() would freeze LayerNorm statistics, making the output distribution
     depend only on the input statistics, which would be nearly constant.

3. **Is the gradient vanishing in the TCN?**
   - depthwise_conv with kernel_size=5 and 512 channels, followed by pointwise_conv.
   - With sequence_length=1 (no staging), the 1D convolution operates on a single-element
     sequence, which may produce numerically zero gradients.

4. **Could the ViT with pos_embed for sequence_length=1024 but only 1 frame** produce
   essentially random attention patterns with near-zero gradient?

5. **Is this a data issue?** The training set has 72 activity classes but some have
   only 1 frame. With CB-Balanced CE loss weighting, rare classes get very high weights.
   Could the reweighted CE loss be pathological with the current architecture?

6. **Would removing the ViT + TCN and using a simple MLP** (proj_features → classifier)
   fix the gradient starvation? This would reduce activity params from 8.2M to ~0.5M.

7. **Would gradient checkpointing or gradient accumulation mode affect the TCN/ViT?**

8. **Could the feature_bank be blocking activity gradients?**
   The activity_proj passes through feature_bank before reaching activity_head.
   If feature_bank.detach_grad_entries_only or feature_bank_detach is True,
   the activity gradient may be severed before it reaches the backbone.

## Activity Feature Path (forward pass)

```
c5_mod (FPN top) → c5_mod_blend = blend*c5_mod + (1-blend)*c5_mod.detach()
  → concat[det_conf, pool(c5_mod_blend), pool(p4.detach())] = activity_proj
  → proj_features(activity_proj) → proj_feat
  → NaN guard → feature_bank(proj_feat, video_ids) → bank_output
  → concat[bank_output, det_conf] → ViT + TCN → LayerNorm → Linear → logits
```

KEY QUESTION: Does feature_bank detach gradients from proj_feat?
