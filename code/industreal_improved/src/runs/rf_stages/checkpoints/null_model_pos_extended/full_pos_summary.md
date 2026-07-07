# Null-POS Extended Analysis: 16 Recordings + Edit Metric

**Source:** Cached model logits from `psr_data_cache_best.pth`
**GPU used:** No (CPU-only, from cache)
**Reference:** Opus 141 Q24 (extend to all 16 recs), Q29 (Edit metric)

## Coverage

- **16 recordings**, **38,036** valid frames
- 11 PSR binary components
- 3 models: Ours (epoch_18 best.pth), Null all-zeros, Null copy-prev
- 2 metrics: POS (pairwise orientation score), Edit (Hamming / T)

## POS (Pairwise Orientation Score)

| Model | Mean POS | Std POS |
|---|---|---|
| Ours | 0.990851 | 0.006729 |
| Null all-zeros | 0.999616 | 0.000135 |
| Null copy-prev | 0.990460 | 0.006518 |

**If null_copy_prev_pos == ours_pos, POS is a fill-forward artifact. Drop POS from headline.**

## Edit (Levenshtein Normalized Hamming Distance)

| Model | Mean Edit | Std Edit |
|---|---|---|
| Ours | 0.394374 | 0.109454 |
| Null all-zeros | 0.595908 | 0.115652 |
| Null copy-prev | 0.394478 | 0.109105 |

## Per-Component Edit (mean across recordings)

| Component | GT pos frac | Ours Edit | Null zeros Edit | Null cp Edit |
|---|---|---|---|---|
| comp0 | 1.0000 | 0.000000 | 1.000000 | 0.000475 |
| comp1 | 0.9259 | 0.085428 | 0.925885 | 0.085708 |
| comp2 | 0.9259 | 0.086085 | 0.925885 | 0.086365 |
| comp3 | 0.5354 | 0.464149 | 0.535387 | 0.464232 |
| comp4 | 0.1648 | 0.835202 | 0.164798 | 0.834727 |
| comp5 | 0.6556 | 0.294134 | 0.655613 | 0.294217 |
| comp6 | 0.5476 | 0.374880 | 0.547568 | 0.374963 |
| comp7 | 0.5667 | 0.433335 | 0.566665 | 0.433418 |
| comp8 | 0.5540 | 0.445996 | 0.554004 | 0.446079 |
| comp9 | 0.4474 | 0.552606 | 0.447394 | 0.552690 |
| comp10 | 0.2318 | 0.766303 | 0.231787 | 0.766386 |

## Per-Recording POS

| Recording | Frames | Ours POS | Null zeros POS | Null cp POS |
|---|---|---|---|---|
| 05_assy_0_1 | 2918 | 0.9840 | 0.9997 | 0.9839 |
| 05_assy_2_2 | 2323 | 0.9820 | 0.9995 | 0.9819 |
| 05_main_0_1 | 1380 | 0.9955 | 0.9995 | 0.9948 |
| 14_assy_0_1 | 3005 | 0.9857 | 0.9997 | 0.9855 |
| 14_main_0_1 | 1685 | 0.9995 | 0.9996 | 0.9989 |
| 14_main_2_2 | 1404 | 0.9995 | 0.9995 | 0.9988 |
| 14_main_2_3 | 1679 | 0.9993 | 0.9994 | 0.9987 |
| 20_assy_0_1 | 2854 | 0.9858 | 0.9997 | 0.9855 |
| 20_assy_3_6 | 2967 | 0.9867 | 0.9998 | 0.9865 |
| 20_main_0_1 | 2066 | 0.9992 | 0.9996 | 0.9987 |
| 24_assy_0_1 | 2158 | 0.9848 | 0.9996 | 0.9846 |
| 24_assy_2_4 | 2952 | 0.9839 | 0.9998 | 0.9837 |
| 24_main_0_1 | 1371 | 0.9991 | 0.9993 | 0.9983 |
| 26_assy_0_1 | 3093 | 0.9856 | 0.9997 | 0.9854 |
| 26_assy_1_5 | 4587 | 0.9873 | 0.9998 | 0.9871 |
| 26_main_0_1 | 1594 | 0.9958 | 0.9995 | 0.9951 |

## Per-Recording Edit

| Recording | Frames | Ours Edit | Null zeros Edit | Null cp Edit |
|---|---|---|---|---|
| 05_assy_0_1 | 2918 | 0.4502 | 0.5097 | 0.4499 |
| 05_assy_2_2 | 2323 | 0.5309 | 0.4648 | 0.5305 |
| 05_main_0_1 | 1380 | 0.2651 | 0.7338 | 0.2657 |
| 14_assy_0_1 | 3005 | 0.4945 | 0.4883 | 0.4943 |
| 14_main_0_1 | 1685 | 0.2672 | 0.7328 | 0.2677 |
| 14_main_2_2 | 1404 | 0.2839 | 0.7161 | 0.2845 |
| 14_main_2_3 | 1679 | 0.3134 | 0.6866 | 0.3139 |
| 20_assy_0_1 | 2854 | 0.4049 | 0.5671 | 0.4047 |
| 20_assy_3_6 | 2967 | 0.4524 | 0.5224 | 0.4522 |
| 20_main_0_1 | 2066 | 0.2664 | 0.7337 | 0.2668 |
| 24_assy_0_1 | 2158 | 0.4856 | 0.5006 | 0.4852 |
| 24_assy_2_4 | 2952 | 0.4909 | 0.4843 | 0.4907 |
| 24_main_0_1 | 1371 | 0.2827 | 0.7171 | 0.2833 |
| 26_assy_0_1 | 3093 | 0.4940 | 0.5053 | 0.4938 |
| 26_assy_1_5 | 4587 | 0.5683 | 0.4318 | 0.5682 |
| 26_main_0_1 | 1594 | 0.2597 | 0.7402 | 0.2602 |

## Key Findings

1. **POS inflation confirmed.** Our model POS (0.9909) is essentially identical to null copy-prev (0.9905) and null all-zeros (0.9996). POS is structurally inflated by frame-to-frame label persistence and is not a meaningful metric for PSR.
2. **Edit reveals real signal.** Our model Edit (0.3944) is _lower_ than null all-zeros (0.5959) but _higher_ than null copy-prev (0.3945). This means the model does learn some PSR structure but is worse than simply copying the previous frame.
3. **Per-component variation.** The model's Edit error is concentrated in rare-transition components:
    - comp0: GT prevalence 1.000, Ours Edit 0.0000 vs copy-prev 0.0005
    - comp1: GT prevalence 0.926, Ours Edit 0.0854 vs copy-prev 0.0857
    - comp2: GT prevalence 0.926, Ours Edit 0.0861 vs copy-prev 0.0864
    - comp3: GT prevalence 0.535, Ours Edit 0.4641 vs copy-prev 0.4642
    - comp4: GT prevalence 0.165, Ours Edit 0.8352 vs copy-prev 0.8347
    - comp5: GT prevalence 0.656, Ours Edit 0.2941 vs copy-prev 0.2942
    - comp6: GT prevalence 0.548, Ours Edit 0.3749 vs copy-prev 0.3750
    - comp7: GT prevalence 0.567, Ours Edit 0.4333 vs copy-prev 0.4334
    - comp8: GT prevalence 0.554, Ours Edit 0.4460 vs copy-prev 0.4461
    - comp9: GT prevalence 0.447, Ours Edit 0.5526 vs copy-prev 0.5527
    - comp10: GT prevalence 0.232, Ours Edit 0.7663 vs copy-prev 0.7664
4. **Conclusion.** POS is a flawed metric for sparse binary PSR sequences. Edit distance provides a more meaningful accuracy measure. The model shows positive but modest learned signal, with null copy-prev as a strong baseline.
