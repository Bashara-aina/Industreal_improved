# 57: Multi-Task Gradient Imbalance — The Structural Problem [2026-06-30]

## The Core Challenge

We are training 5 task heads on a single ConvNeXt-Tiny backbone with FPN.
The heads compete for gradient through the shared backbone. The imbalance ratios
are extreme and systematic — they do NOT respond to any hyperparameter tuning.

## Gradient Ratios (Measured at Step 0, ALL runs)

| Head | Gradient Norm | Ratio (vs activity) | Ratio (vs mean) |
|------|:-:|:-:|:-:|
| PSR head (total) | 3.180 | **312x** | 4.4x |
| Detection head | 0.479 | **47x** | 0.7x |
| Head pose head | 0.561 | **55x** | 0.8x |
| Pose head | 0.443 | **43x** | 0.6x |
| **Activity head** | **0.010** | **1x** | **0.01x** |
| Backbone (178 params) | 2.366 | 232x | 3.3x |
| FPN (16 params) | 1.154 | 113x | 1.6x |

**The activity gradient is 312x weaker than PSR at the parameter level.**

This is not a scaling issue — it persists across ALL attempted fixes:
- blend_ratio 0.05 → 1.00
- LR multiplier 1x → 20x
- Clip 0.3 → 1.0 → removed
- Gradient centralization on/off
- With and without classifier reinit

## Why It Persists: Forward Propagation Analysis

### Detection Head
```
FPN features → conv(256→cls) → sigmoid → Focal Loss
```
Sigmoid on 75M predictions × 24 classes → dense gradient signal even on bad predictions.
The logits are in healthy range (cls_mean=-3.5 to -5.1, std=0.4-0.9) with near_zero=0.0000.

### PSR Head
```
FPN features → Transformer → 11 binary classifiers → Focal Loss × 11 components
```
Sequence mode (T=8, stride=1) produces 41K windows. The sequence loss gradients
are strong (psr_seq_loss_scale active). PSR oscillates between ALIVE (3.18) and
DEAD (1e-08) on a ~500-step cycle, but the TOTAL gradient is ALWAYS high.

### Head Pose Head
```
9-DoF targets → MSE × LOSS_WEIGHT=5.0
```
The loss starts high (6.88 in epoch 1) and drops fast. MAE=8.71° by epoch 2.
Gradient stays high at 0.561 even as loss drops.

### Activity Head
```
activity_proj → proj_features (1048→512) → NaN guard → feature_bank
  → bank_output → [ViT × 2 → TCN → LayerNorm → Linear(512→75) → CE]
```

The gradient path is the LONGEST of any head:
1. CE loss → classifier (75×512)
2. → LayerNorm (512)
3. → TCN (depthwise conv 512×1×5 + pointwise 512×512×1)
4. → ViT (2 layers, each with self-attention + FFN)
5. → proj_features (512×1048)
6. → c5_mod_blend = blend*c5_mod + (1-blend)*c5_mod.detach()
7. → c5_mod (from FPN top)
8. → FPN features
9. → backbone stages

With sequence_length=1 (no staging in RF4), the TCN operates on a single-element
sequence, and the ViT sees only one token + pos_embed.

## Kendall Uncertainty Weights (epoch 2, start)

```
log_var_det = 0.000  →  w_det = 0.250
log_var_pose = 0.000 →  w_pose = 0.250
log_var_act = 0.000  →  w_act = 0.250
log_var_psr = 0.000  →  w_psr = 0.250
```

All Kendall weights are equal at initialization. By epoch 1 they shift slightly,
but the KENDALL_LOG_VAR bounds prevent extreme values:
- KENDALL_LOG_VAR_MIN_ACT = -0.5 (can boost precision 1.65x max)
- KENDALL_LOG_VAR_MAX_PSR = 0.0 (can't suppress PSR)
- KENDALL_LOG_VAR_MAX_POSE = 3.0 (can suppress pose 20x)

This means Kendall cannot compensate for the 312x gradient gap.

## PSR Oscillation: A Secondary Problem

The PSR head cycles between ALIVE and DEAD on a ~500-step cycle.

### LIVENESS Snapshots (epoch 2)
```
step=2000: psr=2.48e+00 ALIVE  | psr_c=2.05e-01/1.08e+00/5.30e-01
step=2500: psr=4.56e-01 ALIVE  | psr_c=6.37e-02/1.24e+00/3.37e-01
step=3000: psr=1.00e-06 DEAD   | psr_c=2.05e-01/1.08e+00/5.30e-01
step=3500: psr=8.17e-02 ALIVE  | psr_c=1.53e-01/1.09e+00/4.73e-01
step=4000: psr=1.00e-06 DEAD   | psr_c=1.49e-01/1.34e+00/5.84e-01
step=5500: psr=1.00e-06 DEAD   | psr_c=1.41e-01/1.28e+00/5.50e-01
step=6000: psr=1.00e-06 DEAD   | psr_c=1.59e-01/9.69e-01/4.48e-01
```

The per-component PSR heads (11 of them, h0-h10) stay ALIVE throughout.
The DEAD signal comes from psr_head total norm oscillating. The component means
psr_c show stable values around 0.1-1.6 across all components.

This suggests the PSR sequence output head is where the oscillation occurs,
not the per-component classifiers. The sequence mode (seq_every=2) alternates
between det batches and seq batches, causing PSR to oscillate.

## Detection Performance

### DET_PROBE Results (epoch 2 validation, 200 batches)
```
bestIoU>0.5: 921-1354 preds (LOCALIZING verdict)
bestIoU_max: 0.79-0.81
```

Detection is improving but from a very low base (det_mAP50=0.023 epoch 1 → 0.053 epoch 2).
At 0.053/epoch improvement, it needs 3 more epochs just to reach 0.20 gate threshold
and 12 more for 0.30 RF10 target.

### DET-HEALTH Trajectory
```
Epoch 1, step 2501: cls_mean=-4.614943 std=0.781770 near_zero=0.0000
Epoch 2, step 2501: cls_mean=-4.615525 std=0.782043 near_zero=0.0000
Epoch 2, step 1501: cls_mean=-4.410868 std=0.669610 near_zero=0.0000 (improved)
Epoch 2, step 501:  cls_mean=-3.472174 std=0.409101 near_zero=0.0000 (best at step 501)
```

cls_mean oscillates between -3.5 and -5.1 depending on where the batch is in
the seq/det cycle. The detection head recovers quickly after seq steps.

## Head Pose

Actually exceeds RF10 final target (35°) at epoch 2 (forward_MAE=8.71°).
The pose.csv forward vector norms average 0.014-0.030 instead of 1.0, suggesting
the data unit vectors are not normalized. This means absolute MAE numbers are
artificially low, but relative improvement is still genuine.

## Data Characteristics

| Property | Value |
|----------|-------|
| Training recordings | 36 |
| Training frames | 3,667 (subset_ratio=0.5 → ~18 recordings) |
| Validation frames | 1,928 (subset_ratio=0.5 → ~8 recordings) |
| Activity classes | 72 (75 with NA) |
| Long-tail classes (<1%) | 46/72 (64%) |
| Detection classes | 24 COCO-style assembly states |
| PSR components | 11 |
| PSR steps | 36 |
| Head pose | 9-DoF (forward, position, up vectors) |

## Questions for Opus

1. **Should we remove the ViT + TCN from activity_head for joint training?**
   They may be designed for VideoMAE features (384-D, temporal) but we feed
   single-frame bank_output (512-D). The ViT's pos_embed assumes T=1024 but we
   have 1 frame in non-staging mode.

2. **Would a direct shortcut help?** E.g., adding a skip connection from
   proj_features output to classifier input, bypassing ViT+TCN.

3. **Is there a gradient-blocking path in feature_bank?** The bank stores
   per-frame features and retrieves them by video_id. If the retrieval uses
   .detach() or .data, gradients never reach proj_features.

4. **Should we freeze all other heads and train only activity for 2 epochs?**
   This would tell us if the head can learn AT ALL with the current architecture.

5. **Is the NaN guard (line 2117-2118) killing gradients?**
   `if not torch.isfinite(proj_feat).all(): proj_feat = torch.zeros_like(proj_feat)`
   If proj_feat is mostly zeros, gradients through the guard are zero.

6. **Does the blend ratio formula actually propagate gradients?**
   `c5_mod_blend = blend * c5_mod + (1 - blend) * c5_mod.detach()`
   When blend=1.0, this equals c5_mod (full gradient). Verified correct in practice.

7. **Is sequence_length=1 causing the TCN/ViT to produce degenerate outputs?**
   With 1-element sequence, convolutions and attention operate on a single position.
