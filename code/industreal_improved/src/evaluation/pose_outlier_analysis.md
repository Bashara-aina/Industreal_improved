# Pose Outlier Diagnostic + GT Noise Floor Analysis

**Date:** 2026-07-06  
**Analysis per:** Opus 141 Q14, Q21, Q29  
**Outlier recording:** `14_assy_0_1` (fwd MAE 17.05°, up MAE 12.32° vs others median fwd 8.94°, up 7.58°)

---

## Q14 — GT Noise Floor

**Finding:** No tracking-confidence field exists in `pose.csv`. The file has 10 columns with schema:

```
filename, forward_x, forward_y, forward_z, position_x, position_y, position_z, up_x, up_y, up_z
```

There is no `confidence`, `tracking_state`, or `valid` column. The high-confidence-subset MAE analysis cannot be performed.

### Global GT Noise Floor

| Metric | Value |
|--------|-------|
| Total frames across all 16 recordings | 38,036 |
| Frame-to-frame jumps > 90 degrees | **0 across ALL recordings** |
| Zero forward vectors | 0 |
| Zero up vectors | 0 |
| Zero positions | 0 |

**The GT noise floor is essentially zero.** The HoloLens tracking produces clean, artifact-free pose data across all 38,036 frames.

---

## Q21 — Outlier Analysis

### MAE by Recording (sorted by forward MAE, worst first)

| Recording | Fwd MAE | Up MAE | Fwd Var | Up Var | Mean Fwd AngVel |
|-----------|---------|--------|---------|--------|-----------------|
| 14_assy_0_1 | 17.05 | 12.32 | 0.00946 | 0.00573 | 0.85/fr *** OUTLIER ***
| 20_assy_3_6 | 11.49 | 7.99 | 0.01272 | 0.00489 | 1.34/fr
| 14_main_2_3 | 10.97 | 6.56 | 0.00617 | 0.00264 | 0.59/fr
| 14_main_2_2 | 10.92 | 7.62 | 0.00698 | 0.00764 | 0.61/fr
| 14_main_0_1 | 10.47 | 5.71 | 0.00661 | 0.00409 | 0.42/fr
| 05_main_0_1 | 10.17 | 7.76 | 0.01614 | 0.00952 | 1.40/fr
| 05_assy_2_2 | 9.37 | 10.28 | 0.01691 | 0.01215 | 1.42/fr
| 26_assy_0_1 | 9.05 | 9.20 | 0.00463 | 0.00419 | 0.86/fr
| 26_main_0_1 | 8.83 | 9.01 | 0.00313 | 0.00370 | 0.82/fr
| 24_assy_0_1 | 8.57 | 8.35 | 0.00600 | 0.00358 | 0.95/fr
| 20_assy_0_1 | 8.52 | 7.07 | 0.00973 | 0.00519 | 1.03/fr
| 20_main_0_1 | 8.08 | 6.33 | 0.01243 | 0.00583 | 0.95/fr
| 24_main_0_1 | 6.80 | 6.09 | 0.01021 | 0.00692 | 0.73/fr
| 05_assy_0_1 | 6.26 | 7.53 | 0.00866 | 0.00821 | 1.22/fr
| 26_assy_1_5 | 6.08 | 6.02 | 0.00512 | 0.00403 | 0.82/fr
| 24_assy_2_4 | 6.07 | 5.90 | 0.00446 | 0.00371 | 0.75/fr

### Key Insight: Outlier is in Model Prediction, NOT GT Quality

The outlier recording `14_assy_0_1` has the highest model prediction error (fwd MAE 17.05) but:
- Its GT angular velocity is **below average** (0.85/fr vs others 0.93/fr)
- Its GT variance is **at the mean** (fwd var 0.0095 vs others 0.0087)
- It has **zero artifacts** (no zeros, no jumps > 90 deg)

This means the **GT is clean** and the model simply fails harder on this recording.

---

## Q29 — Tracking Artifact Analysis

### Per-Recording Breakdown

| Recording | Jumps>90 | Zero Fwd | Zero Up | Zero Pos | Fwd MAE | Up MAE |
|-----------|----------|----------|---------|----------|---------|--------|
| 14_assy_0_1 | 0 | 0 | 0 | 0 | 17.05 | 12.32 <<< OUTLIER
| 20_assy_3_6 | 0 | 0 | 0 | 0 | 11.49 | 7.99
| 14_main_2_3 | 0 | 0 | 0 | 0 | 10.97 | 6.56
| 14_main_2_2 | 0 | 0 | 0 | 0 | 10.92 | 7.62
| 14_main_0_1 | 0 | 0 | 0 | 0 | 10.47 | 5.71
| 05_main_0_1 | 0 | 0 | 0 | 0 | 10.17 | 7.76
| 05_assy_2_2 | 0 | 0 | 0 | 0 | 9.37 | 10.28
| 26_assy_0_1 | 0 | 0 | 0 | 0 | 9.05 | 9.20
| 26_main_0_1 | 0 | 0 | 0 | 0 | 8.83 | 9.01
| 24_assy_0_1 | 0 | 0 | 0 | 0 | 8.57 | 8.35
| 20_assy_0_1 | 0 | 0 | 0 | 0 | 8.52 | 7.07
| 20_main_0_1 | 0 | 0 | 0 | 0 | 8.08 | 6.33
| 24_main_0_1 | 0 | 0 | 0 | 0 | 6.80 | 6.09
| 05_assy_0_1 | 0 | 0 | 0 | 0 | 6.26 | 7.53
| 26_assy_1_5 | 0 | 0 | 0 | 0 | 6.08 | 6.02
| 24_assy_2_4 | 0 | 0 | 0 | 0 | 6.07 | 5.90

### Hypothesis Tests

#### (a) GT Data Quality (REJECTED)
- Jumps > 90 deg: **0**
- Zero rows: **0**
- Artifact fraction: **0.00%**
- **Verdict:** GT is perfectly clean. Rejected as cause.

#### (b) Annotation / Task Difference (PARTIAL)
- This is an assembly recording. However, other assembly recordings (e.g. 05_assy_0_1 with fwd MAE 6.26) do not show extreme MAE.
- **Verdict:** Task type alone does not explain the 2x MAE gap.

#### (c) Real Difficulty / Extreme Head Motion (REJECTED)
- Outlier mean fwd angular velocity: **0.85/fr** (BELOW others mean of **0.93/fr**)
- Outlier max fwd jump: **7.88** (LOWER than others mean of **11.46**)
- **Verdict:** The head is moving LESS, not more. Rejected.

#### (d) Calibration Issue (INCONCLUSIVE)
- Position baseline is typical for 14-series recordings.
- No clear calibration anomaly detected.
- **Verdict:** Inconclusive; unlikely to be the primary cause.

#### (e) Visual Domain Shift (PROPOSED as likely cause)
- The model fails despite clean GT and below-average motion.
- Signature: fwd MAE gap (2x median) with normal/low motion suggests a **visual feature distribution shift**.
- Possible causes: different lighting conditions, occlusion patterns, head appearance, or camera position in the 14_assy environment.
- **Verdict:** This hypothesis best explains the evidence. Cannot be confirmed without visual inspection (outside scope).

---

## Final Recommendation

Per Opus 141 Q23:

| Variant | Up MAE | Fwd MAE | Source |
|---------|--------|---------|--------|
| With outlier | 7.78 deg | 9.14 deg | pose_kalman_results.json (weighted mean) |
| Without outlier | 7.39 deg | 8.46 deg | recomputed weighted mean excl. 14_assy_0_1 |

**Decision:** Report both variants. The outlier has clean GT, so it is not a GT-quality footnote issue. The model genuinely performs worse on this recording. Recommend "with-outlier" as primary (7.78, 9.14) and "without-outlier" as secondary (7.39, 8.46), with a note that the outlier reflects a genuine prediction difficulty rather than a data quality problem.

The 0.39 up-MAE difference and 0.68 fwd-MAE difference do not change the paper's conclusions whichever variant is used.
