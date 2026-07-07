# 149 — Implementation Fixes Summary: 9 Fixes This Session

## §1. PSR Head Repair (e618d929a, 6defe1f5f)
- GELU 99.7% saturated → LeakyReLU(0.01) + small-normal init (std=0.01) + zero bias
- Post_gelu mean: -130 (dead GELU) → +4608 (V3 training log step 10) on sequence frames
- Gradient flow restored

## §2. Head Pose Index Fix (bff38b790)
- 3.5-month bug: up-vector read from [3:6] (position)
- Fixed to [6:9] (correct slice)
- Result: 26.20° → 7.78°

## §3. Detection Fixes (8cef56fc2, cd901f655, 10d5ab596, a0ffb9aa8)
- GT-balanced sampler: 100% of batches have GT (was ~95%)
- DET_GAMMA_NEG 1.5→2.0: harder negative mining
- Anchor audit: confirmed not the root cause
- Class index verification: mapping correct

## §4. Full-Eval V2 with Corrected Indices (216566da0)
- forward 9.136° (vs 9.14° from Kalman)
- up 7.784° (vs 7.78° from Kalman)
- 0.005° agreement between methods

## §5. Multi-Task Training: FREEZE_BACKBONE Flag (bc6bebdb7)
- New FREEZE_BACKBONE config flag
- BACKBONE_LR_MULT for fine-tuning
- Enables backbone gradient flow when unfrozen

## §6. Temporal Probe Fix (7001107de)
- Bare except Exception removed
- Metadata extraction fixed
- ClipDataset now builds correctly

## §7. What Each Fix Enables
- PSR repair: in-flight training will produce non-saturated logits
- Head pose: corrected numbers throughout
- Detection: next training run will use these fixes
- Multi-task fine-tuning: enabled by new config flag

## §8. What's Still Pending
- Detection fix evaluation: single-task training with these fixes
- PSR head repair: training in flight, activations confirmed alive
- MViTv2-S fine-tuning: launch script ready, blocked on GPU
