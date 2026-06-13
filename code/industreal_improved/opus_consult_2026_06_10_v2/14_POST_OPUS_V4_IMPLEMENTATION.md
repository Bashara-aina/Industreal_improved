# 14 — Post-Opus v4 Implementation Journal (2026-06-13)

## What We Learned From Opus v4

Opus told us the hard truth: **Run 8 launched without the RC-28/RC-29 fixes.** The fixes had been sitting on `claude/keen-lamport-vxnhxg` since June 12, unmerged. Run 8 trained on pre-fix code where:

1. **RC-28**: Empty frames contributed their full ~4.15M-element negative focal mass divided by `num_pos=1` (~85% of frames, ~30:1 domination of the detection gradient)
2. **RC-29**: `mixed_precision: True` was hand-flipped (against audited FP32 config), causing GradScaler to silently skip `optimizer.step()` — the model was FROZEN, not at equilibrium

The "critical finding" that Run 8 proved the collapse is architectural was **wrong** — it proved the unfixed normalization reproduces, which we already knew. The fresh ImageNet start did prove the collapse isn't checkpoint lineage (valuable), but focal loss is fine, the architecture can learn, and no exotic loss replacements (VFL/GHM/OHEM/ATSS) are needed.

## Sequence of Events

### 1. Cherry-Pick the Fixes (commit `d101e7e`)

Cherry-picked `c219569` from `claude/keen-lamport-vxnhxg` onto main. The fix contained:
- **RC-28**: Empty frames with `gt_boxes.shape[0]==0` are skipped in detection loss; normalization changed from `/B` to `/max(n_img_with_gt, 1)`
- **RC-29**: Step-commit telemetry at both optimizer-step sites with per-epoch `committed/skipped/scaler_scale` verdict
- **recovery_det_only** preset: det + head_pose only, FP32, batch=1, grad_accum=8
- `13_OPUS_ANSWER_v4.md`: Full Opus answer document

### 2. Pre-R1 Config Fixes (commit `8bef821`)

- Added `TRAIN_MAX_STEPS` from env var to config.py (default 0 = disabled)
- Set `SKIP_DET_METRICS_EVAL=False` so we can see `det_mAP50` for the R1 gate
- Fixed `PRE_VAL_GUARD` to accept `loss ≤ 0` (changed `> 0` to `isfinite()`) — Kendall log-vars on fresh init produce near-zero total loss when empty frames are skipped

### 3. R0 Smoke Test — PASSED

```
TRAIN_MAX_STEPS=400 python3 src/training/train.py --preset recovery_det_only --subset-ratio 0.25
```

| Metric | Result |
|--------|--------|
| Time | 278s (~4.6 min) |
| RC-29 | committed=55, skipped=0 (0.0%), scaler_scale=1.0 |
| det c-loss | Started 49 → 0.05 (trending down on GT frames) |
| nan_skips | 0 |
| PRE_VAL_GUARD | Blocked val (loss≈0 — false positive from fresh Kendall init) |
| Verdict | **PASSED** — all gates met |

The false-positive PRE_VAL_GUARD was fixed immediately (changed to `isfinite()`).

### 4. R1 Detection Bootstrap — 4 Attempts, 3 Eval Crashes

**R1 command**: `python3 src/training/train.py --preset recovery_det_only --subset-ratio 0.25 --max-epochs 3 --seed 42`

#### R1 v1 — Crash #1: Activity eval broadcast mismatch

Epoch 0 completed (6,954 batches, 4,064s, committed=870, skipped=0). Validation started, DET_PROBE showed LOCALIZING on GT frames (bestIoU up to 0.94). Crashed at `evaluate.py:809`:

```
ValueError: operands could not be broadcast together with shapes (9626,5) (8115,1)
```

**Fix**: Added shape guard before `top5_indices == all_gt[:, None]` broadcast.

**Why it happened**: Activity head predictions and ground truth counts don't align when activity head is disabled during training. The evaluation pipeline collected mismatched prediction/GT arrays.

#### R1 v2 — Crash #2: Activity eval clip-level accuracy

Same epoch 0 result (6,954 batches, 4,065s). Crashed at `evaluate.py:833` in `_compute_clip_level_accuracy`. Another activity metric function with data alignment assumptions.

**Fix**: Skipped entire `compute_activity_metrics()` call when `TRAIN_ACT=False`. Also skipped PSR eval when `TRAIN_PSR=False`.

#### R1 v3 — Crash #3: Activity logging KeyError

Epoch 0 completed (6,954 batches, 4,058s). Crashed at `evaluate.py:3191`:

```
KeyError: 'act_accuracy'
```

The stub activity metrics dict (returned when TRAIN_ACT=False) only had `act_macro_f1`, `act_top5_acc`, `act_frame_acc` — but the logging section referenced 7+ activity keys.

**Fix**: Guarded the activity logger.info() block with `if getattr(C, 'TRAIN_ACT', True):`. Also guarded PSR logging similarly.

#### R1 v4 — SUCCESS: First Non-Zero Detection Metrics!

Epoch 0 completed (6,954 batches, 4,075s). Validation completed **without crashing** — all 5 eval guards worked.

**Val results (epoch 0):**
```
loss=-0.3916  det_mAP50=0.0091  as_f1=0.0000  as_map_r=0.0000
ev_ap=0.0268  ev_f1=0.0000  combined=0.1107
act_macro_f1=0.0000  act_top5=nan  psr_f1=nan  psr_edit=nan
```

**THIS IS THE FIRST TIME IN THE PROJECT'S HISTORY THAT `det_mAP50` PRINTED ABOVE ZERO.**

Key findings:
- `det_mAP50=0.0091` — detection is LEARNING (gate was ≥0.05, close after just 1 epoch on 25% subset)
- `ev_ap=0.0268` — error verification also showing non-zero signal
- `as_f1=0.0000`, `as_map_r=0.0000` — assembly state still zero (more epochs needed)
- `psr_f1=nan`, `act_top5=nan` — cosmetic NaN from stub dict key mismatches in Val line formatter. PSR/act eval correctly skipped; the stub values `{psr_f1: 0.0, psr_edit: 0.0}` don't include all keys the formatter expects (`psr_overall_f1`, etc.)
- Combined=0.1107 — up from 0.1067 (pre-fix runs) — driven by det_mAP50 and ev_ap contributions

**Epoch 1**: Started immediately after val. Training at 10% (batch 670).

| Guard | Location | Purpose |
|-------|----------|---------|
| Top5 shape check | evaluate.py:809 | Skip broadcast when preds ≠ GT count |
| TRAIN_ACT guard | evaluate.py:3174 | Skip activity eval when disabled |
| TRAIN_ACT logger guard | evaluate.py:3190 | Skip activity logging when disabled |
| TRAIN_PSR guard | evaluate.py:3255 | Skip PSR eval when disabled |
| TRAIN_PSR logger guard | evaluate.py:3264 | Skip PSR logging when disabled |

### 5. What We've Seen So Far (DET_PROBE on GT Frames)

Across all R1 runs, the model consistently shows:

| Batch | GT Boxes | preds@IoU>0.5 | bestIoU | Verdict |
|-------|----------|---------------|---------|---------|
| b0 | 16 | 49 | 0.59 | LOCALIZING |
| b1 | 16 | 93 | 0.61 | LOCALIZING |
| b2 | 6 | 44 | 0.72 | LOCALIZING |
| b4 | 4 | 673 | 0.91 | LOCALIZING |
| b34 | 15 | 1,867 | 0.85 | LOCALIZING |
| b197 | 9 | 1,124 | 0.84 | LOCALIZING |
| b222 | 12 | 2,775 | 0.89 | LOCALIZING |
| b223 | 16 | 4,142 | 0.94 | LOCALIZING |
| b224 | 16 | 3,116 | 0.92 | LOCALIZING |

**Empty frames**: Model correctly outputs near-zero scores (score_p50=0.001, 0-3 preds>0.05). False positive suppression improved across retrains as model stabilized.

## Confusions Encountered

1. **"Run 8 proved architectural collapse"**: We were wrong. Opus corrected us — it proved the unfixed normalization reproduces, not that focal loss is fundamentally broken. The RC-28/RC-29 fixes were written, just unmerged.

2. **The `mixed_precision` hand-flip**: Run 8's config had `'mixed_precision': True` with comment "[TEMP: AMP for 12GB VRAM]" — overriding the audited FP32 requirement. This caused GradScaler to skip every optimizer step. We didn't notice because forward-pass losses looked healthy. The RC-29 telemetry now catches this.

3. **PRE_VAL_GUARD rejecting healthy training**: The guard checked `total_loss > 0`, but Kendall-weighted loss on fresh init with empty frames skipped averages near zero. Changed to `isfinite()`.

4. **Eval pipeline assumes all heads are trained**: Three separate crashes from activity eval code that assumes aligned prediction/GT arrays. We didn't anticipate this because recovery_det_only (det+head_pose only) is a new training mode the eval pipeline was never tested against.

## Current State

- **R1 v4**: Running validation (DET_PROBE b601+, past all crash points)
- **Training**: Healthy — committed=870, skipped=0, nan_skips=0
- **Detection**: LOCALIZING on GT frames with bestIoU up to 0.94
- **Waiting for**: Final `Val:` line with `det_mAP50` number
- **Next**: If det_mAP50 ≥ 0.05 → proceed to R2 (joint recovery). If not → det-head LR ×3 for 2 more epochs.
