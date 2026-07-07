# PSR Full-38k Reconciliation: 10k vs 38k Gap Analysis

## Background
- 10k subset (val-selected upper bound): global_0.10 F1 = 0.7257
- Full 38k stream: global_0.10 F1 = 0.6788
- Gap: 0.0470

## Per-component optimal thresholds
- 10k: [0.05, 0.2, 0.15000000000000002, 0.8500000000000001, 0.8, 0.5, 0.45, 0.9000000000000001, 0.9000000000000001, 0.05, 0.7000000000000001]
- 38k: [0.05, 0.05, 0.05, 0.8, 0.95, 0.8, 0.65, 0.95, 0.95, 0.95, 0.95]

## Full-38k optimal macro-F1: 0.7018
## Full-38k LOO-CV: 0.0148 ± 0.0158

## Gap Analysis per component (global 0.10 F1)
- comp0: 1.0000 (10k) vs 1.0000 (38k), gap=+0.0000
- comp1: 0.9753 (10k) vs 0.9584 (38k), gap=+0.0169
- comp2: 0.9753 (10k) vs 0.9593 (38k), gap=+0.0159
- comp3: 0.7056 (10k) vs 0.7361 (38k), gap=-0.0305
- comp4: 0.2482 (10k) vs 0.1979 (38k), gap=+0.0503
- comp5: 0.7735 (10k) vs 0.7919 (38k), gap=-0.0185
- comp6: 0.7057 (10k) vs 0.7217 (38k), gap=-0.0159
- comp7: 0.8002 (10k) vs 0.6224 (38k), gap=+0.1777
- comp8: 0.8002 (10k) vs 0.6074 (38k), gap=+0.1928
- comp9: 0.6900 (10k) vs 0.4721 (38k), gap=+0.2179
- comp10: 0.3092 (10k) vs 0.3991 (38k), gap=-0.0898

## Root cause
The gap is primarily due to:
1. **Sampling variance**: The 10k subset was the first 10k frames of the val set. If
   recordings are ordered non-randomly (e.g., easier recordings first), the 10k subset
   may be an overestimate of the full-set performance.
2. **Prevalence shift**: Component prevalence varies across recordings. The 10k subset
   may have different per-component positive fractions than the full 38k set.
3. **The gap (0.0470) is the same order as every claimed improvement from
   per-component calibration** — which is why Opus 140 Q2 flags this as blocking.

## Key takeaway
The honest primary is now: full-38k per-comp-optimal macro-F1 = 0.7018
(previously reported: 0.7499 on 10k). This is the number to report in the paper.

### LOO-CV bound
- Improvement from per-comp calibration: 0.0148 ± 0.0158
- This bound is consistent with the 10k-subset bound of +0.0358 ± 0.0216
