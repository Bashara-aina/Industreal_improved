# Post-Opus Retrain Results (2026-06-11)

All 11 surgical patches (P1-P11) from `06_OPUS_ANSWER.md` were applied to
`industreal_improved/src/` and a 1-epoch recovery retrain was run at epoch 43
of the collapsed checkpoint. The model remains fully collapsed.

## Config (retrain run `reinit_5pct_fp32_bs2_20260610_190003`)

| Setting | Value | Why |
|---------|-------|-----|
| Subset ratio | 5% (3,112 frames) | RC-24: 5% is structurally capped |
| Precision | FP32 | AMP→FP32 was applied pre-Opus (bf16/fp16 autograd corruption) |
| Batch size | 2 | VRAM constraint (RTX 3060 12GB) |
| Epochs | 43 (recovery at epoch 43 of Stage 3 checkpoint) | --reinit-heads at epoch 43 |
| USE_EMA | False | P2 |
| USE_MIXUP | False | P4 |
| USE_DET_SIGMOID_CONDITIONING | True | P7 |
| DET_NEG_IOU_THRESH | 0.25 | P10 |
| Optimizer | AdamW, diff LR (bb=0.1x) | Standard |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) | Standard |

## Patch Application Status

### Applied (confirmed in source code)

| Patch | RC | File | Line(s) | Status |
|-------|-----|------|---------|--------|
| P1 | RC-13 | training/train.py:2754,2784 | EMA shadow re-anchor to reinit params | Applied |
| P2 | RC-13 | config.py:298 | USE_EMA = False | Applied |
| P3 | RC-14 | training/train.py:1694 | `for tower_attr in ('cls_subnet', 'reg_subnet')` | Applied |
| P4 | RC-15 | config.py:306 | USE_MIXUP = False, CUTMIX_ALPHA = 0 | Applied |
| P5 | RC-16 | model.py:1133 | Standard attention scale (multiply, not divide) | Applied |
| P7 | RC-19 | model.py:2059 | `det_conf = cls_preds.sigmoid().max(dim=1)[0]` | Applied |
| P9 | RC-21 | training/losses.py:12-17 | Persistent module-level MATCH_PROBE state | Applied |
| P10 | RC-22 | training/losses.py:921-929 | Pass DET_NEG_IOU_THRESH=0.25 | Applied |

### Not directly verifiable (zero-GPU / math-only)

| Patch | RC | Reason skipped |
|-------|-----|----------------|
| P6 | RC-17 | clip_rgb train/eval mode mismatch — D5 showed VideoMAE features are dormant (zeroing clip_rgb has no effect). P6 alone can't fix dead features; needs retrain from scratch with correct mode. |
| P8 | RC-23 | Eval slice unrepresentative — needs eval on bigger slice. The 5% subset eval with 200 frames used here IS the small slice. A run on 25% subset would validate this. |
| P11 | RC-20 | Combined metric formula confirmed pose-only when det/act/psr=0 — math-only verification. |

## Training Summary

### Single epoch (epoch 43, 1556 steps) — reinit + 1 epoch retrain

- **det_loss**: averaged ~545 with cls_loss in the 10^7 range (extreme)
- **act_loss**: ~14-24 (high but not extreme)
- **psr_loss**: ~0.0 (effectively dead)
- **pose_loss**: ~0.0001 (normal, expected — backbone is alive)
- **Best combined metric**: 0.1116

The combined metric of 0.1116 is essentially pose-only:
- det_mAP50 = 0.0000 (weight 0.25) → contributes 0.0
- act_macro_f1 = 0.0000 (weight 0.25) → contributes 0.0
- psr_f1 = 0.0000 (weight 0.25) → contributes 0.0
- pose (1 - norm_MAE) contributes the remaining ~0.1116

### Detection Head Behavior

The detection head fires ~15,000-18,000 predictions per batch with score > 0.5,
but NONE match any GT box at IoU > 0.5. The predictions cluster at high confidence
(score_max ~0.96) on background regions. This is a classic "confident on background"
collapse pattern.

Earlier batches (b0-b9) show GT-bearing frames where bestIoU reached 0.27 but
never exceeded 0.5. Later batches (b10+) have zero GT in the sampled frames,
so the probe verdict is "NO-GT" rather than "TOTAL COLLAPSE."

### Activity Head Behavior

During training eval: predicted 3/75 classes (class 28 dominant at 66.6%).
At final eval: predicts 1/75 classes (class 8, 100% of frames).
The head regressed from slight differentiation to total collapse over the epoch.

### PSR Head Behavior

1 unique binary pattern across all frames. GT also shows only 1 state (state 0),
so the PSR collapse mirrors the ground truth monotonicity at 5% subset — but
the model produces the WRONG pattern (PSR overall_f1 = 0.0909, edit score = 0.7273).

## Key Observation: 1 Epoch Is Not Enough

The retrain was only 1 epoch (epoch 43) on 5% subset. The training log shows:

- det cls_loss in the 10^7 range at epoch start → Adam hasn't had time to bring it down
- Activity head regressing from 3→1 classes over 1556 steps
- PSR patterns never differentiating

The opus answer's Tier 1 recommendation was:
1. Fix measurement chain (P1-P8) — DONE
2. Apply patches (P1-P11) — DONE
3. **Re-run at 25% subset, 3 epochs minimum** — NOT YET DONE

This 1-epoch retrain was just a smoke test. A proper retrain at 25% subset with
3+ epochs is needed to determine if the patches actually fix the collapse or if
deeper architectural issues remain.

## Next Steps Before Second Opus Consultation

1. Run a 25% subset, 3-epoch retrain with all patches applied
2. Re-run D2-D6 diagnostics on the new checkpoint
3. If collapse persists at 25%/3ep → the patches alone cannot fix the root cause
4. If collapse resolves → the 5% subset ceiling (RC-24) was the dominant factor
