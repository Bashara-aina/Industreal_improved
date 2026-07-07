# Per-Class Activity Accuracy Report (Opus 141 ACT-LP-8 / ACT-CM-1)

**Date:** 2026-07-07
**Source:** `checkpoint_35000frames.pkl` (MLP per-frame predictions)
**Labeled frames:** 28665
**Overall accuracy:** 0.0236
**Activity classes:** 69

## Summary

- **Classes with zero accuracy:** 41
- **Classes with accuracy > 0.10:** 9
- **Classes with accuracy > 0.50:** 3
- **Classes with no validation samples:** 6

## Top-5 Most Accurate Classes

| Rank | Class ID | Name | Accuracy | Correct/Total |
|------|----------|------|----------|---------------|
| 1 | 12 | put | 0.6429 | 9/14 |
| 2 | 56 | pull_pin_middle | 0.6207 | 18/29 |
| 3 | 13 | take_pin_long | 0.5044 | 57/113 |
| 4 | 28 | browse_instruction | 0.4160 | 270/649 |
| 5 | 35 | put_wheel | 0.3317 | 69/208 |

## Top-5 Least Accurate Classes

| Rank | Class ID | Name | Accuracy | Correct/Total |
|------|----------|------|----------|---------------|
| 1 | 68 | plug_objects | 0.0000 | 0/781 |
| 2 | 65 | fit_acorn_nut | 0.0000 | 0/79 |
| 3 | 63 | plug_small_screw_pin | 0.0000 | 0/62 |
| 4 | 62 | take_small_screw_pin | 0.0000 | 0/40 |
| 5 | 61 | loosen_acorn_nut | 0.0000 | 0/78 |

## Classes with Zero Accuracy

| Class ID | Name | Samples |
|----------|------|---------|
| 8 | check_instruction | 6217 |
| 7 | tighten_nut | 2415 |
| 2 | align_objects | 1839 |
| 33 | fit_nut | 1350 |
| 30 | fit_tooth_washer | 1079 |
| 41 | take_objects | 1063 |
| 4 | plug_short_pin | 1052 |
| 6 | take_nut | 852 |
| 37 | loosen_nut | 828 |
| 68 | plug_objects | 781 |
| 31 | fit_round_washer | 581 |
| 48 | fit_wheel | 545 |
| 17 | take_round_washer | 527 |
| 21 | take_wheel | 514 |
| 3 | take_pin_short | 495 |
| 43 | put_objects | 459 |
| 53 | fit_pulley | 386 |
| 19 | tighten_acorn_nut | 332 |
| 39 | pull_objects | 304 |
| 26 | take_pulley | 233 |
| 16 | plug_screw_pin | 186 |
| 44 | pull_pin_short | 162 |
| 11 | take_screw_pin | 157 |
| 20 | take_pin_middle | 140 |
| 38 | put_nut | 138 |
| 55 | put_tooth_washer | 106 |
| 52 | put_round_washer | 99 |
| 18 | take_acorn_nut | 97 |
| 65 | fit_acorn_nut | 79 |
| 61 | loosen_acorn_nut | 78 |
| 46 | put_long_brace | 66 |
| 63 | plug_small_screw_pin | 62 |
| 14 | put_pin_long | 43 |
| 40 | put_pin_middle | 43 |
| 62 | take_small_screw_pin | 40 |
| 34 | put_screw_pin | 34 |
| 59 | pull_screw_pin | 33 |
| 24 | put_wing | 30 |
| 36 | pull_wheel | 21 |
| 60 | put_acorn_nut | 14 |
| 51 | fit_objects | 6 |

## Full Per-Class Accuracy Table

| Class ID | Name | Accuracy | Correct/Total |
|----------|------|----------|---------------|
| 12 | put | 0.6429 | 9/14 |
| 56 | pull_pin_middle | 0.6207 | 18/29 |
| 13 | take_pin_long | 0.5044 | 57/113 |
| 28 | browse_instruction | 0.4160 | 270/649 |
| 35 | put_wheel | 0.3317 | 69/208 |
| 42 | put_partial_model | 0.1199 | 85/709 |
| 54 | fit_wing_beam | 0.1176 | 16/136 |
| 64 | put_small_screw_pin | 0.1176 | 4/34 |
| 10 | take_long_brace | 0.1010 | 29/287 |
| 58 | put_pulley | 0.0988 | 8/81 |
| 15 | take_wing_beam | 0.0952 | 10/105 |
| 1 | take_short_brace | 0.0875 | 14/160 |
| 45 | put_pin_short | 0.0787 | 10/127 |
| 49 | check_partial_model | 0.0672 | 16/238 |
| 22 | plug_pin_long | 0.0526 | 9/171 |
| 9 | take_partial_model | 0.0472 | 25/530 |
| 23 | take_wing | 0.0472 | 5/106 |
| 25 | plug_pin_middle | 0.0406 | 11/271 |
| 66 | fit_wing | 0.0305 | 4/131 |
| 32 | fit_long_brace | 0.0154 | 1/65 |
| 29 | fit_short_brace | 0.0149 | 4/268 |
| 5 | take_tooth_washer | 0.0040 | 3/747 |
| 2 | align_objects | 0.0000 | 0/1839 |
| 3 | take_pin_short | 0.0000 | 0/495 |
| 4 | plug_short_pin | 0.0000 | 0/1052 |
| 6 | take_nut | 0.0000 | 0/852 |
| 7 | tighten_nut | 0.0000 | 0/2415 |
| 8 | check_instruction | 0.0000 | 0/6217 |
| 11 | take_screw_pin | 0.0000 | 0/157 |
| 14 | put_pin_long | 0.0000 | 0/43 |
| 16 | plug_screw_pin | 0.0000 | 0/186 |
| 17 | take_round_washer | 0.0000 | 0/527 |
| 18 | take_acorn_nut | 0.0000 | 0/97 |
| 19 | tighten_acorn_nut | 0.0000 | 0/332 |
| 20 | take_pin_middle | 0.0000 | 0/140 |
| 21 | take_wheel | 0.0000 | 0/514 |
| 24 | put_wing | 0.0000 | 0/30 |
| 26 | take_pulley | 0.0000 | 0/233 |
| 30 | fit_tooth_washer | 0.0000 | 0/1079 |
| 31 | fit_round_washer | 0.0000 | 0/581 |
| 33 | fit_nut | 0.0000 | 0/1350 |
| 34 | put_screw_pin | 0.0000 | 0/34 |
| 36 | pull_wheel | 0.0000 | 0/21 |
| 37 | loosen_nut | 0.0000 | 0/828 |
| 38 | put_nut | 0.0000 | 0/138 |
| 39 | pull_objects | 0.0000 | 0/304 |
| 40 | put_pin_middle | 0.0000 | 0/43 |
| 41 | take_objects | 0.0000 | 0/1063 |
| 43 | put_objects | 0.0000 | 0/459 |
| 44 | pull_pin_short | 0.0000 | 0/162 |
| 46 | put_long_brace | 0.0000 | 0/66 |
| 48 | fit_wheel | 0.0000 | 0/545 |
| 51 | fit_objects | 0.0000 | 0/6 |
| 52 | put_round_washer | 0.0000 | 0/99 |
| 53 | fit_pulley | 0.0000 | 0/386 |
| 55 | put_tooth_washer | 0.0000 | 0/106 |
| 59 | pull_screw_pin | 0.0000 | 0/33 |
| 60 | put_acorn_nut | 0.0000 | 0/14 |
| 61 | loosen_acorn_nut | 0.0000 | 0/78 |
| 62 | take_small_screw_pin | 0.0000 | 0/40 |
| 63 | plug_small_screw_pin | 0.0000 | 0/62 |
| 65 | fit_acorn_nut | 0.0000 | 0/79 |
| 68 | plug_objects | 0.0000 | 0/781 |
| 0 | other | N/A (0 samples) | 0/0 |
| 27 | plug_wheel | N/A (0 samples) | 0/0 |
| 47 | pull_partial_model | N/A (0 samples) | 0/0 |
| 50 | put_short_brace | N/A (0 samples) | 0/0 |
| 57 | put_wing_beam | N/A (0 samples) | 0/0 |
| 67 | pull_pin_long | N/A (0 samples) | 0/0 |

## Class Distribution (Train + Val)

- **Train:** 69189 labeled frames
- **Val:** 37280 labeled frames
- **Total:** 106469 labeled frames
- **Dominant class:** check_instruction (Out 8) = 16871 (15.85% of train+val)

### Top-10 Most Frequent Classes (Train+Val)

| Rank | Class ID | Name | Train | Val | Total | % of Total |
|------|----------|------|-------|-----|-------|------------|
| 1 | 8 | check_instruction | 8485 | 8386 | 16871 | 15.85% |
| 2 | 7 | tighten_nut | 5911 | 2863 | 8774 | 8.24% |
| 3 | 2 | align_objects | 5654 | 2703 | 8357 | 7.85% |
| 4 | 4 | plug_short_pin | 2975 | 1274 | 4249 | 3.99% |
| 5 | 33 | fit_nut | 2727 | 1490 | 4217 | 3.96% |
| 6 | 41 | take_objects | 2696 | 1457 | 4153 | 3.90% |
| 7 | 30 | fit_tooth_washer | 2731 | 1254 | 3985 | 3.74% |
| 8 | 68 | plug_objects | 2362 | 1208 | 3570 | 3.35% |
| 9 | 6 | take_nut | 2351 | 1079 | 3430 | 3.22% |
| 10 | 5 | take_tooth_washer | 2182 | 1007 | 3189 | 3.00% |

### Bottom-10 Least Frequent Classes (Train+Val)

| Rank | Class ID | Name | Train | Val | Total | % of Total |
|------|----------|------|-------|-----|-------|------------|
| 1 | 47 | pull_partial_model | 77 | 0 | 77 | 0.07% |
| 2 | 63 | plug_small_screw_pin | 0 | 68 | 68 | 0.06% |
| 3 | 60 | put_acorn_nut | 17 | 48 | 65 | 0.06% |
| 4 | 57 | put_wing_beam | 64 | 0 | 64 | 0.06% |
| 5 | 62 | take_small_screw_pin | 21 | 42 | 63 | 0.06% |
| 6 | 64 | put_small_screw_pin | 19 | 34 | 53 | 0.05% |
| 7 | 12 | put | 33 | 14 | 47 | 0.04% |
| 8 | 27 | plug_wheel | 44 | 0 | 44 | 0.04% |
| 9 | 67 | pull_pin_long | 22 | 0 | 22 | 0.02% |
| 10 | 0 | other | 0 | 0 | 0 | 0.00% |
