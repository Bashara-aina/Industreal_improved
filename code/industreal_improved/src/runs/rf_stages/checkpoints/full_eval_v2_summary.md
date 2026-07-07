# full_eval_stream v2 — Corrected-Index Head Pose Results

**Checkpoint:** `best.pth` (epoch 18)
**Date:** 2026-07-07
**Script:** `full_eval_stream.py` with corrected up-vector indices [6:9]

## Head Pose Angular MAE (38,036 frames, 16 recordings)

| Vector | MAE (°) | Basis |
|--------|---------|-------|
| Forward | **9.136** | full_eval_stream v2 |
| Up | **7.784** | full_eval_stream v2 (corrected indices [6:9]) |

## Comparison with Kalman eval

| Source | Forward MAE (°) | Up MAE (°) |
|--------|----------------|------------|
| full_eval_stream v2 (this) | **9.136** | **7.784** |
| Kalman eval (single-frame) | 9.14 | 7.78 |
| Kalman eval (RTS-smoothed) | 9.00 | 7.58 |
| Difference (stream vs Kalman single) | -0.004 | +0.004 |

## Verification

The full_eval_stream v2 results confirm the corrected-index head pose numbers to within 0.005°. The 3.5-month index bug ([3:6] instead of [6:9] for up-vector) was inflating up-vector error to 26.20°. With corrected indices, up-vector MAE = 7.78°, matching the Kalman eval and establishing the first ego-pose baseline.

## Source

- `src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json`
- `SOTA_STATUS.md` rows 18-19 (headline table)
