# PSR LOO-CV: Train/Val Membership Breakdown

Per Opus 141 Q20 — verifying whether the +0.0148 +/- 0.0158 LOO bound is
contaminated by training-set recordings appearing in the held-out set.

## Recording Membership

| Split  | Recordings |
|--------|------------|
| TRAIN  | 01_assy_0_1, 01_assy_1_1, 01_main_0_1, 02_assy_0_1, 02_assy_1_2, 02_main_0_1, 04_assy_0_1, 04_assy_2_1, 04_main_0_1, 06_assy_0_1, 06_assy_1_4, 06_main_0_1, 07_assy_0_1, 07_assy_2_3, 07_main_0_1, 11_assy_0_1, 11_assy_3_3, 11_main_0_1, 15_assy_0_1, 15_main_3_1, 15_main_3_2, 16_assy_0_1, 16_main_0_1, 16_main_3_3, 21_assy_0_1, 21_main_0_1, 22_assy_0_1, 22_assy_2_3, 22_main_0_1, 25_assy_0_1, 25_assy_2_1, 25_main_0_1, 27_assy_0_1, 27_main_0_1, 27_main_1_3, 27_main_3_1 (36 total) |
| VAL    | 05_assy_0_1, 05_assy_2_2, 05_main_0_1, 14_assy_0_1, 14_main_0_1, 14_main_2_2, 14_main_2_3, 20_assy_0_1, 20_assy_3_6, 20_main_0_1, 24_assy_0_1, 24_assy_2_4, 24_main_0_1, 26_assy_0_1, 26_assy_1_5, 26_main_0_1 (16 total) |

## Membership Cross-Reference

Of the 16 recordings in the LOO-CV evaluation set:

- **In TRAIN split: 0** (none)
- **In VAL split only: 16** (all 16)

The LOO-CV recordings are entirely from the validation split. The model
was trained on the train split (36 recordings) which is disjoint from
these 16. No train/val membership overlap exists.

## Group A (Clean VAL-only held-out)

All 16 recordings fall into this group:

- Mean LOO improvement: **+0.0148**
- Standard deviation: 0.0163
- Global F1 (mean): 0.6710
- Optimal F1 (mean): 0.6858

Sign-positive ratio: 12/16 = 75% of recordings improve.

## Group B (Contaminated TRAIN held-out)

**Empty set.** No LOO-CV recording belongs to the training split, so
there is no contaminated group to report.

## Per-Recording Detail

| Recording | Improvement | Global F1 | Optimal F1 |
|-----------|------------|-----------|------------|
| 05_assy_0_1 | +0.0298 | 0.6042 | 0.6340 |
| 05_assy_2_2 | +0.0075 | 0.5372 | 0.5447 |
| 05_main_0_1 | -0.0048 | 0.7944 | 0.7896 |
| 14_assy_0_1 | +0.0377 | 0.5764 | 0.6142 |
| 14_main_0_1 | +0.0027 | 0.7972 | 0.7999 |
| 14_main_2_2 | -0.0006 | 0.7814 | 0.7808 |
| 14_main_2_3 | -0.0023 | 0.7306 | 0.7283 |
| 20_assy_0_1 | +0.0161 | 0.6619 | 0.6780 |
| 20_assy_3_6 | +0.0187 | 0.6096 | 0.6283 |
| 20_main_0_1 | -0.0014 | 0.7979 | 0.7966 |
| 24_assy_0_1 | +0.0165 | 0.5985 | 0.6150 |
| 24_assy_2_4 | +0.0425 | 0.5693 | 0.6118 |
| 24_main_0_1 | +0.0080 | 0.7840 | 0.7920 |
| 26_assy_0_1 | +0.0186 | 0.5975 | 0.6161 |
| 26_assy_1_5 | +0.0446 | 0.4971 | 0.5417 |
| 26_main_0_1 | +0.0032 | 0.7983 | 0.8015 |

## Implication for the +0.0148 +/- 0.0158 Bound

The bound is **NOT overstated**. There is no train/val membership
contamination. All LOO-CV recordings are from the validation split, the
model was trained on the disjoint train split, and the per-recording
threshold transfer is evaluated on truly held-out recordings with no
training-set overlap.

The original bound of +0.0148 +/- 0.0158 stands as-is.
