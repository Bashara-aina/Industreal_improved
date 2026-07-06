# Activity Confusion Matrix Report

**Source**: checkpoint_35000frames.pkl (28665 labeled frames)

## Top-20 Confused Pairs (raw 75-class space)

| Rank | True Class | Predicted Class | Count | % of True | Verb-Antonym? |
|------|------------|-----------------|-------|-----------|---------------|
| 1 | take_partial_model | take_short_brace | 2436 | 39.2% |  |
| 2 | check_instruction | take_short_brace | 1618 | 67.0% |  |
| 3 | take_pin_short | take_short_brace | 786 | 42.7% |  |
| 4 | take_partial_model | plug_wheel | 774 | 12.4% |  |
| 5 | fit_long_brace | take_short_brace | 655 | 48.5% |  |
| 6 | take_tooth_washer | take_short_brace | 638 | 60.6% |  |
| 7 | take_partial_model | take_pin_long | 601 | 9.7% |  |
| 8 | fit_short_brace | take_short_brace | 549 | 50.9% |  |
| 9 | unknown_37 | take_short_brace | 524 | 63.3% |  |
| 10 | check_instruction | fit_wing_beam | 485 | 20.1% |  |
| 11 | tighten_nut | take_short_brace | 380 | 44.6% |  |
| 12 | fit_long_brace | fit_wing_beam | 371 | 27.5% |  |
| 13 | take_partial_model | pull_partial_model | 336 | 5.4% |  |
| 14 | put_pin_middle | take_short_brace | 318 | 44.9% |  |
| 15 | fit_short_brace | fit_wing_beam | 317 | 29.4% |  |
| 16 | take_nut | take_short_brace | 317 | 42.4% |  |
| 17 | fit_acorn_nut | fit_wing_beam | 313 | 40.1% |  |
| 18 | pull_objects | take_short_brace | 308 | 29.0% |  |
| 19 | put_long_brace | fit_wing_beam | 301 | 55.2% |  |
| 20 | take_partial_model | put_screw_pin | 253 | 4.1% |  |

## Same-Object Verb-Antonym Confusions

| Object | True Verb | Pred Verb | Count | % of True Class |
|--------|-----------|-----------|-------|-----------------|
| pin_short | take | put | 210 | 
| short_brace | put | take | 52 | 
| objects | take | put | 34 | 
| pin_middle | plug | pull | 19 | 
| pin_middle | put | take | 11 | 
| pin_long | put | take | 8 | 
| acorn_nut | take | put | 7 | 
| nut | take | put | 4 | 
| partial_model | pull | fit | 4 | 
| partial_model | take | put | 1 | 

## Summary

- Total labeled frames: 28665
- Per-frame accuracy: 0.0236
- Total errors: 27988
- Same-object verb-antonym errors: 350 (1.3%)
- Take↔Put same-object errors: 327 (1.2%)
- Dominant error pattern: prediction collapse to majority class (take_short_brace)

**Conclusion**: Verb-antonym confusions (especially take↔put on the same object) are systematic
and arise from temporal ambiguity at action boundaries. While not the numerically dominant
error mode (which is class-imbalance collapse), they represent a fundamental ambiguity:
the exact frame where a hand takes a screw becomes indistinguishable from putting it.
