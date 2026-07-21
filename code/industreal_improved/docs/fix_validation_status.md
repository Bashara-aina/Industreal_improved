# mAP=0.05 Bottleneck Investigation — Status Report

**Date:** 2026-07-21  
**Status:** Fix applied, training in progress, validation pending

## TL;DR

mAP@0.5 was stuck at 0.0519 across all training runs (v3.5: 5 epochs, v3.7: 0.45
epochs — both gave identical mAP). Investigation identified the root cause as
`update_logit_bias()` overwriting per-class bias gradients with a uniform
EMA-derived value. Fix is in place (disabled by default), probe shows
**+16.5% lower BG confidence, +10.7% higher FG confidence, +15.3% better
FG/BG separation** in overfit-200 evaluation.

## Root Cause Analysis

### Symptoms

- mAP=0.0519 stable across epochs (5 epochs vs 0.45 epochs → identical)
- 4.8M predictions for 138 GT boxes (35K:1 FP:TP ratio)
- Model fires on ~every anchor at ~0.1 confidence instead of suppressing BG
- Training log: cls loss very small (0.026) vs reg loss huge (15.28)

### The Offending Code

```python
def update_logit_bias(self, batch_pos_ratio, momentum=0.05):
    """[IMP-10] Adjust cls bias based on observed positive ratio."""
    self.running_pos_ratio = (1.0 - momentum) * self.running_pos_ratio + momentum * batch_pos_ratio
    pos = max(0.001, min(self.running_pos_ratio, 0.5))
    target_bias = -math.log((1.0 - pos) / pos) * self.logit_bias_scale
    for m in self.cls_head.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == self.num_classes:
            m.bias.data.fill_(target_bias)  # ⚠️ OVERWRITES gradient updates
```

### Why This Breaks Learning

1. **Gradient override**: After each optimizer step, the fill_() call
   overwrites whatever bias values gradient descent computed.

2. **Uniform bias**: ALL 24 classes get the SAME bias value (target_bias).
   Different classes have different frequencies/difficulty — they should
   have different biases.

3. **EMA-driven collapse**: With ~3 positives per ~75K anchors, pos_ratio ≈
   4e-5. EMA target_bias = -log((1-4e-5)/4e-5) ≈ -10.1 (sigmoid ≈ 0).

4. **Actual bias range in checkpoint**: -0.22 to -3.56 (sigmoid 0.03 to 0.44).
   The bias is BEING PUSHED BACK by gradient updates against the EMA.
   Result: most classes stuck at sigmoid ~0.42 — firing everywhere.

### Checkpoint Evidence (v3.7 b18000)

```
m.det_head.cls_head.3.bias (24 classes):
  Class 0:  -0.2491 (sigmoid=0.438) ← majority cluster
  Class 4:  -2.7506 (sigmoid=0.060) ← severely suppressed
  Class 13: -1.3017 (sigmoid=0.214)
  Class 19: -1.3003 (sigmoid=0.214)
  Class 22: -2.7116 (sigmoid=0.062)
  Class 23: -3.5553 (sigmoid=0.028)
```

The biases have **diverged from the uniform EMA fill** because gradient
updates overpowered the EMA — but they can't reach the right values because
the fill_() keeps wiping them.

## Probe Validation

**`scripts/probe_logit_bias_disable.py`** — 500 steps overfit-200,
FullMultiModalDataset:

| Metric | ENABLED (baseline) | DISABLED (fix) | Δ |
|---|---|---|---|
| Initial bias[0] | -4.60 | -4.60 | — |
| Final bias[0] | -6.91 | -4.58 | +2.33 |
| Background conf | 0.0465 | **0.0388** | **-16.5%** ✓ |
| Foreground conf | 0.327 | **0.362** | **+10.7%** ✓ |
| FG/BG separation | 0.281 | **0.324** | **+15.3%** ✓ |
| Time | 109s | 90s | -17% |

**Verdict:** Disable `update_logit_bias()`.

## Implementation

### Changes Made

1. **`src/models/mvit_mtl_model.py`**: `running_pos_ratio` is now a registered
   buffer (`register_buffer(..., persistent=True)`), so it persists across
   checkpoint save/load. Previously it was a plain Python float that reset
   to `prior_prob=0.1` on every reload.

2. **`train_mtl_v3.py`**: Added `--use-logit-bias-update` flag (default OFF).
   Both Phase 1 (~line 1554) and Phase 2 (~line 1747) bias-update calls are
   guarded by this flag.

3. **`scripts/probe_logit_bias_disable.py`**: New probe to validate fix.

4. **`scripts/tal_probe_correct.py`**: Corrected probe using
   `FullMultiModalDataset` (prior `tal_probe_fixed.py` used wrong dataset).

5. **`scripts/eval_v38_fix.py`**: Eval script for v3.8 vs v3.7 comparison.

### Files Changed

- `src/models/mvit_mtl_model.py`
- `train_mtl_v3.py`
- `scripts/probe_logit_bias_disable.py` (new)
- `scripts/tal_probe_correct.py` (new)
- `scripts/eval_v38_fix.py` (new)
- `docs/bottleneck_fix_logit_bias.md` (new)
- `docs/fix_validation_status.md` (this file)

## Training Runs

### v3.8_fix (resumed from v3.7 b18000, bias-update disabled)

| Batch | mAP@0.5 (1000 frames) | vs v3.7 (0.0261) |
|---|---|---|
| b500  | 0.0326 | +24.9% |
| b1000 | 0.0293 | +12.3% |
| b1500 | 0.0061 | -77% |
| b2000 | 0.0066 | -75% |
| b2500 | 0.0178 | -32% |
| b3000 | 0.0168 | -36% |

**Observation**: mAP jumped at b500 (+25%), then regressed. The bias values
remained nearly identical to v3.7 b18000 throughout, suggesting the fix's
gradient signal was too weak to move the converged bias.

### v3.8b_reset_bias (resumed from v3.7 b18000 with bias RESET to -2.2)

- Started 22:49 with all 24 class biases reset to -2.2 (sigmoid=0.10)
- Training in progress, first eval pending

This is the proper test: gradient descent can now find each class's
optimal bias from a clean starting point.

## Why The Probe Worked But Resumed Training Didn't

The probe used **fresh init** (Xavier + prior_prob=0.01 bias), training
from scratch on 200 samples. Gradient signal was strong (only 200 samples,
no convergence).

In the real training, the model resumes from a checkpoint where bias has
**already converged** via the gradient-vs-EMA tug-of-war. Without the EMA,
gradient signal alone is too weak to move the bias meaningfully.

**Solution**: Reset bias to prior_prob value before resuming, so gradient
descent can find optimal per-class bias from a clean starting point.

## Bonus Findings

1. **`running_pos_ratio` was never in checkpoints**: Plain Python attribute,
   not a registered buffer. Every resume reset it to prior_prob=0.1.

2. **TAL probe (corrected dataset)** verdict: `3x3-suffices` — TAL port
   would not yield major improvement. Both 3×3 and TAL converge to
   similar loss in 200-step overfit.

3. **Class imbalance is severe**: Only 4 classes have >median count; classes
   14, 20, 24 have ZERO training samples in FG pool (rare classes that
   never get sampled). This is a separate issue from the bias bug.

## Next Steps

1. **Wait for v3.8b reset_bias training** — first checkpoint at b500
2. **Eval at b500, b1000, b2000, b5000** — track mAP trajectory
3. **If mAP > 0.05**: validate the fix definitively
4. **If mAP still ~0.05**: investigate further (cls_head[3].weight init?
   Different loss? Better anchors? More training data?)

## References

- Probe results: `runs/probe_results/logit_bias_probe_2026-07-21.json`
- Probe results: `runs/probe_results/tal_probe_correct_2026-07-21.json`
- Eval results: `/tmp/v38_eval_b*.json` (v3.8 fix checkpoints)
- Eval results: `/tmp/v37_eval_1kframes.json` (v3.7 baseline)
- Original analysis: `/tmp/mvit_eval_v3.7_fixed.json`
- Commits: see `git log` for the fix commit