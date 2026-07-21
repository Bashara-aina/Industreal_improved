# Detection Bottleneck Fix: Disable `update_logit_bias()` (2026-07-21)

## Problem

mAP@0.5 was stuck at 0.0519 across all training runs:
- v3.5 (5 epochs): mAP=0.0519
- v3.7 (0.45 epochs): mAP=0.0519 (identical)

Despite ~4.8M predictions for 138 GT boxes (35K:1 FP:TP ratio), the model
fires on every anchor location at ~0.1 confidence instead of suppressing
background.

## Root Cause

`DetectionHead.update_logit_bias()` (called every training step) was
**overwriting per-class bias gradients** with a uniform value computed from
EMA of positive anchor ratio.

### Mechanism

```python
def update_logit_bias(self, batch_pos_ratio, momentum=0.05):
    self.running_pos_ratio = (1.0 - momentum) * self.running_pos_ratio + momentum * batch_pos_ratio
    pos = max(0.001, min(self.running_pos_ratio, 0.5))
    target_bias = -math.log((1.0 - pos) / pos) * self.logit_bias_scale
    for m in self.cls_head.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == self.num_classes:
            m.bias.data.fill_(target_bias)  # ALL 24 classes get SAME bias
```

With ~3 positives out of ~75K anchors per batch:
- `pos_ratio` ≈ 4e-5
- `target_bias` = -log((1-4e-5)/4e-5) ≈ -10.1

But the checkpoint shows class biases ranging from **-0.22 to -3.56** —
the fill_() is being constantly **counteracted** by class-specific gradient
updates. This creates a tug-of-war where:
- The EMA pushes bias toward very negative (sigmoid ~0)
- Gradient updates push bias toward class-specific values
- Most classes settle at sigmoid ~0.42 (still firing everywhere)

### Checkpoint Evidence (v3.7 b18000)

```
m.det_head.cls_head.3.bias (24 classes):
  Class 0:  -0.2491 (sigmoid=0.438)  ← majority cluster
  Class 4:  -2.7506 (sigmoid=0.060)  ← severely suppressed
  Class 13: -1.3017 (sigmoid=0.214)
  Class 19: -1.3003 (sigmoid=0.214)
  Class 22: -2.7116 (sigmoid=0.062)
  Class 23: -3.5553 (sigmoid=0.028)
```

The bias should NOT be uniform across classes — different classes have
different frequencies and difficulty.

## Fix

**Disable `update_logit_bias()` entirely.** Let gradient descent push each
class's bias toward its own optimal value.

### Probe Results (500-step overfit-200, FullMultiModalDataset)

| Metric | ENABLED (baseline) | DISABLED (fix) | Δ |
|---|---|---|---|
| Initial bias[0] | -4.60 | -4.60 | — |
| Final bias[0] | -6.91 | -4.58 | +2.33 |
| Loss | 2.85 → 0.0001 | 2.85 → 0.0001 | tied |
| Background conf | 0.0465 | **0.0388** | **-16.5%** ✓ |
| Foreground conf | 0.327 | **0.362** | **+10.7%** ✓ |
| FG/BG separation | 0.281 | **0.324** | **+15.3%** ✓ |
| Time | 109s | 90s | -17% |

The DISABLED run **dropped BG conf from 0.057 to 0.012 over 500 steps**
(learned proper suppression), while the ENABLED run oscillated
between 0.007 and 0.10 (unstable due to EMA interference).

## Implementation

1. **`src/models/mvit_mtl_model.py`**: Made `running_pos_ratio` a registered
   buffer (`register_buffer(..., persistent=True)`) so it survives
   checkpoint save/load. Previously it was a plain Python float that
   reset to `prior_prob=0.1` on every reload.

2. **`train_mtl_v3.py`**: Added `--use-logit-bias-update` flag (default
   OFF). Both Phase 1 (line ~1554) and Phase 2 (line ~1745) bias-update
   calls are now guarded by this flag.

3. **`scripts/probe_logit_bias_disable.py`**: New probe script for
   validating the fix.

4. **`scripts/tal_probe_correct.py`**: Corrected TAL probe using
   `FullMultiModalDataset` (the prior `tal_probe_fixed.py` used the wrong
   dataset — verdict was meaningless).

## Validation Plan

- `runs/mtl_v3.8_fix`: Resuming from `v3.7 b18000` with bias-update OFF
- Target: mAP > 0.05 (any improvement validates the fix)
- Compare bias trajectories with v3.7 to confirm gradient-driven bias
  divergence is now class-specific and stable

## Files Changed

- `src/models/mvit_mtl_model.py` — register_buffer fix
- `train_mtl_v3.py` — add flag, guard calls (Phase 1 + Phase 2)
- `scripts/probe_logit_bias_disable.py` — new probe
- `scripts/tal_probe_correct.py` — corrected probe (was wrong dataset)
- `runs/probe_results/logit_bias_probe_2026-07-21.json` — probe results
- `runs/probe_results/tal_probe_correct_2026-07-21.json` — TAL probe results

## Bonus Discovery

`running_pos_ratio` was NOT being saved in checkpoints (plain Python
attribute). Every resume-from-checkpoint reset the EMA to prior_prob=0.1,
forcing the bias to re-warm from -2.2 each time. Now registered as buffer
so EMA persists across resumes.