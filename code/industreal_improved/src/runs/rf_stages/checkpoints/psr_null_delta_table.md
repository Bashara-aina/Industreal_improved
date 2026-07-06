# PSR Per-Component Null-Delta Analysis (Opus PSR-6)

**Date:** 2026-07-06
**Source:** Epoch_18 best.pth, per-comp optimal thresholds from SOTA_STATUS.md

| comp | gt_pos_frac | F1_achieved | F1_null (always-positive) | Delta (learned signal) |
|---|---|---|---|---|
| 0 | 1.000 | 1.0000 | 1.000 | +0.000 |
| 1 | 0.911 | 0.9627 | 0.953 | +0.009 |
| 2 | 0.911 | 0.9578 | 0.953 | +0.004 |
| 3 | 0.545 | 0.7480 | 0.706 | +0.042 |
| 4 | 0.142 | 0.3455 | 0.249 | +0.097 |
| 5 | 0.631 | 0.7793 | 0.774 | +0.006 |
| 6 | 0.544 | 0.7057 | 0.705 | +0.001 |
| 7 | 0.667 | 0.8041 | 0.800 | +0.004 |
| 8 | 0.667 | 0.8536 | 0.800 | +0.053 |
| 9 | 0.527 | 0.6900 | 0.690 | -0.000 |
| 10 | 0.183 | 0.4020 | 0.309 | +0.093 |

**Null model:** Always-positive classifier always predicts the majority class. For a component with prevalence p, F1_null = 2p/(1+p). This is the F1 an oracle would achieve by predicting "positive" on every frame — trivially satisfied for high-prevalence components.

**Interpretation:** Low-prevalence components (comp 4 at p=0.142, comp 10 at p=0.183) show delta +0.097 and +0.093 respectively — approximately +0.10 of genuine learned signal beyond the prevalence prior. Comp 8 (p=0.667, delta +0.053) also shows moderate learned signal. High-prevalence components (comp 0-2, p>0.9) show small deltas as expected from ceiling effects. Comp 9 is effectively at-null (delta -0.000), indicating the model learned nothing beyond prevalence for that posture.

**Discloses (Opus PSR-6):** The 0.7499 macro-F1 is partially backbone+prevalence-fitting, but with +0.09-0.10 delta on the hardest, lowest-prevalence components — the head learned something real on the components that matter most.
