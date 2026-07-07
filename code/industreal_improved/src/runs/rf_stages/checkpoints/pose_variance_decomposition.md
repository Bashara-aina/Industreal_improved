# Pose Variance Decomposition: Between vs Within Recording

**Analysis:** ANOVA-style random-effects decomposition of head pose MAE variance into between-recording and within-recording components (Opus 141 Q25).

## Data

- **Frames:** 38036 across 16 recordings
- **Grand mean forward:** 9.14 deg
- **Grand mean up:** 7.78 deg
- **Total per-frame std forward:** 7.85 deg
- **Total per-frame std up:** 6.84 deg

## ANOVA Table

### Forward MAE
| Source | SS | df | MS | F |
|---|---|---|---|---|
| Between recordings | 325731 | 15 | 21715.38 | 408.7 |
| Within recordings | 2019977 | 38020 | 53.13 | |
| Total | 2345708 | 38035 | | |

### Up MAE
| Source | SS | df | MS | F |
|---|---|---|---|---|
| Between recordings | 130284 | 15 | 8685.61 | 200.4 |
| Within recordings | 1648229 | 38020 | 43.35 | |
| Total | 1778514 | 38035 | | |

## Variance Components (Random Effects Model)

### Forward MAE
- **Between-recording variance:** 9.1914 deg^2 (14.7%)
- **Within-recording variance:** 53.1293 deg^2 (85.3%)
- **Total:** 62.3207 deg^2
- **ICC:** 0.1475

### Up MAE
- **Between-recording variance:** 3.6670 deg^2 (7.8%)
- **Within-recording variance:** 43.3516 deg^2 (92.2%)
- **Total:** 47.0186 deg^2
- **ICC:** 0.0780

## Interpretation

**Dominant component: WITHIN-RECORDING (per-frame noise)**

For both forward and up-vector, within-recording variance accounts for the large majority of total error variance:
- Forward: 85.3% within vs 14.7% between
- Up: 92.2% within vs 7.8% between

**Recommendation: TEMPORAL SMOOTHING**

The model's MAE is dominated by per-frame noise rather than recording-specific biases. The Kalman smoother (params Q=0.01, R=0.05) already achieves ~1.5-2.7% improvement. Further gains are expected from tuning smoothing parameters or applying more sophisticated temporal filters. Data diversification would provide limited benefit since recording-specific effects are a small fraction of total variance.

## Method

Unbalanced one-way random-effects ANOVA. Variance components estimated via method-of-moments:
- Between-recording component = (MS_between - MS_within) / n0
- Within-recording component = MS_within
- n0 = (sum(n_i) - sum(n_i^2) / sum(n_i)) / (K-1) = 2356.8
