# Master Prompt v4 — RC-28 Deadlock: 3-Way Coupled Collapse (2026-06-12)

## Context

This is a follow-up to `10_OPUS_ANSWER_v2.md` (your previous answer, June 11). We implemented ALL your v2 prescriptions (RC-25 FPN reinit, RC-27 GroupNorm, step-0 assertion, det_conf zeroing, EMA re-anchor, D7-D9 diagnostics). **RC-25 is SOLVED** (step-0 PASSED: median |z| = 2.95 < 8). But a NEW deadlock (RC-28) prevents any head from reaching non-zero metrics.

## What We Did

### Run 1 (June 11–12): Fresh reinit from epoch-43 checkpoint
```
python3 src/training/train.py --preset recovery --reinit-heads \
  --resume epoch43_latest.pth --subset-ratio 0.25 --max-epochs 45 --seed 42
```
- 2 epochs (44–45), ~3 hours
- Step-0 assertion PASSED
- Train loss: 25–160 (non-zero, non-NaN)
- Val after 2 epochs: det_mAP50=0.0000, act_macro_f1=0.0007, psr_f1=0.0000, combined=0.1067
- Saved crash_recovery.pth (301MB)

### Run 2 (June 12): Continue without reinit
```
python3 src/training/train.py --preset recovery \
  --resume crash_recovery.pth --subset-ratio 0.25 --max-epochs 55 --num-workers 0 --seed 42
```
- 2 more epochs (46–47), ~3.5 hours, KILLED at epoch 47
- Val after 2 more epochs: **IDENTICAL** to 4 decimal places
- 4 validation cycles across 2 runs: ZERO change in any metric

## The Deadlock (RC-28)

### Architecture coupling chain
```
Backbone+FPN → detection_head → cls_preds
                                    ↓
                              det_conf = sigmoid(max(cls_preds))
                                    ↓
                              if ZERO_DET_CONF_FOR_RECOVERY: det_conf = zeros
                                    ↓
                              concat(det_conf, GAP_features) → activity_head → 75 classes
                                    ↓
                              PSR ← FPN features ← backbone (shared with detection)
```

### Deadlock table
| Head | Observed State | Deadlock Mechanism |
|------|---------------|-------------------|
| **Detection** | Scores flat at 0.154 (std=0.0095), 194K preds/90 GT | Focal Loss(α=0.25,γ=2) at prior π=0.05: ~2.76M negative anchors/batch × FL per neg ≈ 200+ cls_loss. Only 10–15% of batches have any GT. Positive signal from ~10–50 anchors can't overcome negative mass. |
| **Activity** | 1/75 classes (class 20, 100% frames), macro_f1=0.0007 | `ZERO_DET_CONF_FOR_RECOVERY=True` → det_conf = zeros → activity head gets ZERO detection signal. Can't learn frame differentiation. |
| **PSR** | 1 unique binary pattern across 1200 frames, f1=0.0000 | Depends on backbone features. Backbone gradient dominated by detection's negative Focal Loss → no useful signal for PSR-relevant features. |

### The recovery flag IS the deadlock
`ZERO_DET_CONF_FOR_RECOVERY` (model.py:1819-1820) was designed for the ORIGINAL collapse where det_conf was O(10-100) and saturated. But after RC-25 fix (FPN reinit + prior π=0.05), det_conf is HEALTHY at 0.154 — the zeroing is now STARVING the activity head.

## Evidence

### Validation results (all 4 cycles identical)
```
Val: loss=132.4092  det_mAP50=0.0000  act_clip=0.0000  act_frame=0.0100
     act_macro_f1=0.0007  act_top5=0.2425  forward_angular_MAE_deg=66.00
     psr_f1=0.0000  psr_edit=0.4773  psr_pos=0.0000
     as_f1=0.0000  as_map_r=0.0000  ev_ap=0.0000  ev_f1=0.0000
     combined=0.1067
```

### Collapse diagnostics (all 4 cycles identical)
```
[EVAL COLLAPSE] detection head produces flat scores (std=0.0095 < 0.01, all ≈ 0.154)
[EVAL COLLAPSE] activity head predicts only 1/75 classes (top-1 class=20 with 100.0% of frames)
[EVAL COLLAPSE] PSR head produces only 1 unique binary pattern(s) across 1200 frames
[EVAL COLLAPSE] excessive prediction count: 194831 preds across 90 GT boxes (ratio=2165x)
```

### Training loss pattern
- GT batches: loss ~5–30 (cls=3–4, g=0.8–0.9) — box regression works well
- No-GT batches: loss ~120–160 (cls=200+, g=0.0) — Focal Loss dominates on 2.76M negatives
- ~85% of batches have no GT → negative Focal Loss dominates overall gradient

### What IS working
- Step-0 assertion PASSED: median |z| = 2.95 < 8
- Train loss non-zero, non-NaN
- Box regression on GT batches: GIoU = 0.8–0.9 (good localization)
- FPN feature magnitudes healthy (confirmed by D7)
- act_top5=0.2425 above random (0.067) — activity has SOME signal via GAP features
- psr_edit=0.4773 non-zero — PSR binary strings have some structure

## Questions for Opus

1. **Verify RC-28**: Is the 3-way coupling deadlock diagnosis correct? Or is there a simpler explanation (e.g., LR too low, insufficient epochs)?

2. **Escape strategy**: What's the minimum set of changes to get ANY core metric non-zero?
   - Disable `ZERO_DET_CONF_FOR_RECOVERY` immediately?
   - Train detection solo first (staged: epoch 1–N det-only, then unfreeze others)?
   - Higher LR for detection head (1e-3 vs current)?
   - Subsample negatives in Focal Loss (e.g., top-k hardest negatives)?
   - Different prior (π=0.01 instead of π=0.05)?

3. **Staged recovery protocol**: Design a concrete training schedule with:
   - Which heads active at each stage
   - LR per head per stage
   - Number of epochs per stage
   - When to disable ZERO_DET_CONF_FOR_RECOVERY
   - When to enable EMA, Mixup, etc.

4. **Fresh start vs salvage**: At this point, would a clean ImageNet-init retrain with ALL fixes applied from scratch be more likely to succeed than continuing to fight the epoch-43 lineage?

5. **Focal Loss mass problem**: Is Focal Loss at π=0.05 on 2.76M negatives fundamentally the wrong loss for this architecture? Should we consider alternatives (e.g., Varifocal Loss, GHM, OHEM, ATSS adaptive anchor selection)?

## Source Files

All latest source code is in the `code/` directory. Key files:
- `code/model.py` — lines 503 (GroupNorm), 1195–1260 (ActivityHead), 1805–1820 (det_conf zeroing), 1650 (activity_head init)
- `code/train.py` — lines 140 (_REINIT_HEADS_ACTIVE), 1096–1116 (step-0 assertion), 1654–1780 (_reinit_dead_heads), 2595–2650 (reinit orchestration)
- `code/config.py` — lines 252 (ZERO_DET_CONF_FOR_RECOVERY), 535–553 (recovery preset), 292 (EVAL_MAX_BATCHES=75), 320 (CUDA_MEMORY_FRACTION=0.80)
- `code/losses.py` — Focal Loss implementation
- `code/diag_feature_magnitude.py` (D7), `code/diag_step0_logits.py` (D8)

Training logs in `logs/`:
- `logs/recovery_train1_run1.log` — Run 1 (with --reinit-heads)
- `logs/recovery_train2_run2.log` — Run 2 (without --reinit-heads)

Previous Opus answers and context:
- `10_OPUS_ANSWER_v2.md` — Your previous answer (RC-25, RC-27, recovery plan)
- `00_JOURNEY_AND_STATUS.md` — Full project timeline updated through Phase 7
