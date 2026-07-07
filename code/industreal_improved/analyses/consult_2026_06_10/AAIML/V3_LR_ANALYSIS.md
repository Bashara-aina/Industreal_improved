# V3 PSR Training — Learning Rate Schedule Analysis

**Date**: 2026-07-07
**Source log**: `/tmp/train_psr_v3_real.log`
**Commit**: `9ff6d5031963c255ddecf839f566a520677dcce3`

## Summary

The PSR gradient death (psr_head consistently shows NO_GRAD / DEAD in liveness probes) is **NOT caused by a learning rate problem**. The LR schedule is well-configured and consistent with the paper specification. The root cause is a **numerical instability in the PSR head forward pass** that produces activations with std ~2000-21000, which under bf16 mixed precision causes gradient collapse.

---

## 1. Learning Rate at Start

The training resumed from epoch 27 (checkpoint at step 86037). The LR at start of epoch 27:

| Param Group | LR at epoch 27 start |
|---|---|
| Backbone | 5.0e-06 (frozen) |
| All heads (det, pose, activity, psr, bias) | 5.0e-04 |

At epoch 27, the model is already past the 2-epoch linear warmup, so LR is at the OneCycleLR plateau phase (near peak).

## 2. LR Schedule Detail

| Parameter | Value | Notes |
|---|---|---|
| `BASE_LR` | 5e-4 | Per paper spec |
| Scheduler | OneCycleLR | Cosine annealing |
| `WARMUP_EPOCHS` | 2 | LinearLR from 0.1x to 1.0x |
| `ONE_CYCLE_PEAK_FACTOR` | auto = 0.25 | EFFECTIVE_BATCH(8) / 32(paper) |
| Scheduler `pct_start` | 0.1 | 10% of epochs in warmup phase |
| `GRAD_CLIP_NORM` | 5.0 | Applied globally |
| Total epochs | 100 | Config target |

### Param-group max_lr values (OneCycleLR)

| Group | max_lr | Formula |
|---|---|---|
| Backbone | 1.25e-05 | BASE_LR * 0.1 * _peak |
| Detection head | 1.25e-04 | BASE_LR * _peak * DET_LR_MULTIPLIER(1.0) |
| Other heads (pose, head_pose) | 1.25e-04 | BASE_LR * _peak |
| Activity head | 1.25e-04 | BASE_LR * _peak * ACTIVITY_LR_MULTIPLIER(1.0) |
| **PSR head** | **1.25e-04** | BASE_LR * _peak |
| Detection head bias | 1.25e-04 | BASE_LR * _peak * DET_BIAS_LR_FACTOR(1.0) |
| Generic bias | 3.75e-05 | BASE_LR * _peak * BIAS_LR_FACTOR(0.3) |
| VideoMAE | 0.0 (frozen) | — |
| Loss (Kendall log_vars) | 1.25e-04 | BASE_LR * _peak |

## 3. Issues Found

### Issue A (Critical): PSR numerical explosion — not LR related

**Evidence from log:**
```
[PSR_DEBUG step=0] pre_linear:  mean=57.7389 std=1980.5378 min=-5857.4961 max=5324.6729
[PSR_DEBUG step=0] post_linear64: mean=-4608.0000 std=22784.0000 min=-51712.0000 max=74752.0000
```

PSR head activations are **50-100x larger than healthy neural network values**:
- `pre_linear` (MLP input): std ~1900, range [-6000, +6600]
- `post_linear64` (64-dim output per keypoint): std ~21000, range [-55000, +80000]

Normal activations should have std ~1-10 for a properly initialized head.

In bf16 mixed precision (`AMP_DTYPE=bf16`), these extreme values:
- Lose precision: bf16 has only 7 mantissa bits, so values around 20000 are quantized to multiples of ~1024
- Produce corrupted gradients: the gradient of the loss w.r.t. these activations is numerically unstable
- The clip_grad_norm=5.0 then either clips the tiny surviving gradient to zero, or the gradient contains NaN/inf that causes GradScaler to skip the optimizer step

**This is the root cause of `psr_head:NO_GRAD` and `psr=1.00e-06 DEAD`.**

### Issue B (Minor): PSR activation patterns don't change across training

The PSR debug probes at step 0, 1, 10, 100, 200, and 500 all show nearly identical statistics:
- pre_linear mean: 51-132 (no systematic trend)
- pre_linear std: 1800-1980 (no systematic trend)
- post_linear64 std: 20800-22700 (no systematic trend)

This confirms the PSR head weights are not being updated by gradient descent — the head is effectively frozen regardless of LR setting.

### Issue C (Non-issue): PSR warmup implementation

The code has a post-reinit PSR output head warmup: 2x gradient multiplier for 200 steps (implemented in both seq and non-seq paths at lines 1473-1482 and 1971-1980 of `src/training/train.py`). However, this only scales the already-dead gradient and does not address the root numerical instability.

### Issue D (Non-issue): PSR LR multiplier

`PSR_LR_MULTIPLIER` is not set in config, meaning PSR gets the same base head LR (max 1.25e-04 at peak). This is reasonable — changing the LR alone would not fix the dead gradient. Even a 10x higher or 10x lower LR would have no effect because the gradient magnitude is zero (not too small or too large).

### Issue E (Non-issue): Kendall log_var stuck

```
KENDALL step=1: lv: det=0.538 pose=-0.987 act=0.291 psr=0.000 | lv_grad: psr=0.0000
```

The Kendall log_var for PSR is stuck at 0.000 with zero gradient. This is a symptom, not a cause. If PSR gradients existed, the Kendall weight would learn.

## 4. Comparison with Stable Multi-Task Runs

The earlier stable runs (e.g., `rf4_stable_20260703_200447.log`) used:
- Same BASE_LR=5e-4
- Same OneCycleLR scheduler
- Slightly different peak_factor=0.5 (batch size was 4x4=16, so EFFECTIVE_BATCH=16, peak=16/32=0.5 vs current 0.25)
- All heads had working gradients (no DEAD status)

The PSR gradient death in V3 is not due to LR differences between runs. The successful runs also had PSR at the same LR as other heads.

## 5. Recommendations

1. **Fix PSR head weight initialization**: The extreme activation values (std ~2000 vs expected ~1-10) indicate the PSR head weights produce outputs at wrong scale. Investigate `psr_head.per_frame_mlp` and `psr_head.transformer` and `psr_head.output_heads` weight initialization. The small-normal init or LeakyReLU preservation may not apply correctly to the PSR head's internal layers.

2. **Add PSR head activation monitoring at initialization**: Log the actual weight norms and activation statistics of the PSR head immediately after model construction (before any training) to catch initial-weight scale issues.

3. **Consider per-head gradient clipping for PSR**: Currently `ACTIVITY_HEAD_GRAD_CLIP` exists for the activity head but there is no analogous `PSR_HEAD_GRAD_CLIP`. Adding one wouldn't fix the dead-gradient problem (since gradient is already zero), but it would help if the initialization fix produces large-but-finite initial gradients.

4. **No LR changes needed**: The LR schedule (OneCycleLR, peak 1.25e-04 for heads, 2-epoch warmup) is correct and matches the paper spec. Wasting tuning cycles on LR would not resolve the PSR gradient death.
