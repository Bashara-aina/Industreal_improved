# Per-Class Accuracy Comparison: MViTv2-S Linear Probe vs ConvNeXt MLP (Opus 144)

**Date:** 2026-07-07
**MViTv2-S overall top-1:** 0.3810
**ConvNeXt overall top-1:** 0.0236
**Mean per-class accuracy (MViTv2-S):** 0.0545
**Mean per-class accuracy (ConvNeXt):** 0.0580

## Summary

- **Classes with non-zero accuracy (MViTv2-S):** 12
- **Classes with non-zero accuracy (ConvNeXt):** 22
- **ConvNeXt zero-accuracy classes:** 41
- **Zero-to-nonzero transitions:** 11
- **Classes MViTv2-S still at 0.0:** 43

## Top-10 Most Improved Classes (Largest Delta)

| Class | ConvNeXt | MViTv2-S | Delta | MViTv2-S Count |
|-------|----------|----------|-------|----------------|
| check_instruction | 0.0000 | 0.8771 | +0.8771 | 529 |
| tighten_nut | 0.0000 | 0.7149 | +0.7149 | 235 |
| plug_objects | 0.0000 | 0.3558 | +0.3558 | 104 |
| take_objects | 0.0000 | 0.2989 | +0.2989 | 87 |
| align_objects | 0.0000 | 0.1547 | +0.1547 | 181 |
| fit_wheel | 0.0000 | 0.1395 | +0.1395 | 43 |
| take_wheel | 0.0000 | 0.1250 | +0.1250 | 16 |
| fit_tooth_washer | 0.0000 | 0.0727 | +0.0727 | 55 |
| fit_nut | 0.0000 | 0.0405 | +0.0405 | 74 |
| fit_pulley | 0.0000 | 0.0294 | +0.0294 | 34 |

## Zero-to-Nonzero Transitions

| Class | MViTv2-S Accuracy | MViTv2-S Count | Delta |
|-------|-------------------|----------------|-------|
| check_instruction | 0.8771 | 529 | +0.8771 |
| tighten_nut | 0.7149 | 235 | +0.7149 |
| plug_objects | 0.3558 | 104 | +0.3558 |
| take_objects | 0.2989 | 87 | +0.2989 |
| align_objects | 0.1547 | 181 | +0.1547 |
| fit_wheel | 0.1395 | 43 | +0.1395 |
| take_wheel | 0.1250 | 16 | +0.1250 |
| fit_tooth_washer | 0.0727 | 55 | +0.0727 |
| fit_nut | 0.0405 | 74 | +0.0405 |
| fit_pulley | 0.0294 | 34 | +0.0294 |
| plug_short_pin | 0.0227 | 88 | +0.0227 |

## Classes That Worsened

| Class | ConvNeXt | MViTv2-S | Delta | MViTv2-S Count |
|-------|----------|----------|-------|----------------|
| pull_pin_middle | 0.6207 | 0.0000 | -0.6207 | 3 |
| take_pin_long | 0.5044 | 0.0000 | -0.5044 | 3 |
| put_wheel | 0.3317 | 0.0000 | -0.3317 | 3 |
| browse_instruction | 0.4160 | 0.1642 | -0.2518 | 67 |
| put_partial_model | 0.1199 | 0.0000 | -0.1199 | 21 |
| fit_wing_beam | 0.1176 | 0.0000 | -0.1176 | 17 |
| take_long_brace | 0.1010 | 0.0000 | -0.1010 | 8 |
| put_pulley | 0.0988 | 0.0000 | -0.0988 | 2 |
| take_wing_beam | 0.0952 | 0.0000 | -0.0952 | 3 |
| put_pin_short | 0.0787 | 0.0000 | -0.0787 | 1 |
| check_partial_model | 0.0672 | 0.0000 | -0.0672 | 23 |
| plug_pin_long | 0.0526 | 0.0000 | -0.0526 | 19 |
| take_partial_model | 0.0472 | 0.0000 | -0.0472 | 15 |
| take_wing | 0.0472 | 0.0000 | -0.0472 | 6 |
| plug_pin_middle | 0.0406 | 0.0000 | -0.0406 | 27 |
| fit_wing | 0.0305 | 0.0000 | -0.0305 | 7 |
| fit_long_brace | 0.0154 | 0.0000 | -0.0154 | 4 |
| fit_short_brace | 0.0149 | 0.0000 | -0.0149 | 9 |
| take_tooth_washer | 0.0040 | 0.0000 | -0.0040 | 22 |

## Full Comparison Table (All Matched Classes)

| Class | ConvNeXt | MViTv2-S | Delta | Fixed? |
|-------|----------|----------|-------|--------|
| align_objects | 0.0000 | 0.1547 | +0.1547 | YES |
| browse_instruction | 0.4160 | 0.1642 | +-0.2518 |  |
| check_instruction | 0.0000 | 0.8771 | +0.8771 | YES |
| check_partial_model | 0.0672 | 0.0000 | +-0.0672 |  |
| fit_acorn_nut | 0.0000 | 0.0000 | +0.0000 |  |
| fit_long_brace | 0.0154 | 0.0000 | +-0.0154 |  |
| fit_nut | 0.0000 | 0.0405 | +0.0405 | YES |
| fit_pulley | 0.0000 | 0.0294 | +0.0294 | YES |
| fit_round_washer | 0.0000 | 0.0000 | +0.0000 |  |
| fit_short_brace | 0.0149 | 0.0000 | +-0.0149 |  |
| fit_tooth_washer | 0.0000 | 0.0727 | +0.0727 | YES |
| fit_wheel | 0.0000 | 0.1395 | +0.1395 | YES |
| fit_wing | 0.0305 | 0.0000 | +-0.0305 |  |
| fit_wing_beam | 0.1176 | 0.0000 | +-0.1176 |  |
| loosen_acorn_nut | 0.0000 | 0.0000 | +0.0000 |  |
| loosen_nut | 0.0000 | 0.0000 | +0.0000 |  |
| plug_objects | 0.0000 | 0.3558 | +0.3558 | YES |
| plug_pin_long | 0.0526 | 0.0000 | +-0.0526 |  |
| plug_pin_middle | 0.0406 | 0.0000 | +-0.0406 |  |
| plug_screw_pin | 0.0000 | 0.0000 | +0.0000 |  |
| plug_short_pin | 0.0000 | 0.0227 | +0.0227 | YES |
| plug_small_screw_pin | 0.0000 | 0.0000 | +0.0000 |  |
| pull_objects | 0.0000 | 0.0000 | +0.0000 |  |
| pull_pin_middle | 0.6207 | 0.0000 | +-0.6207 |  |
| pull_pin_short | 0.0000 | 0.0000 | +0.0000 |  |
| pull_screw_pin | 0.0000 | 0.0000 | +0.0000 |  |
| pull_wheel | 0.0000 | 0.0000 | +0.0000 |  |
| put_long_brace | 0.0000 | 0.0000 | +0.0000 |  |
| put_nut | 0.0000 | 0.0000 | +0.0000 |  |
| put_objects | 0.0000 | 0.0000 | +0.0000 |  |
| put_partial_model | 0.1199 | 0.0000 | +-0.1199 |  |
| put_pin_long | 0.0000 | 0.0000 | +0.0000 |  |
| put_pin_short | 0.0787 | 0.0000 | +-0.0787 |  |
| put_pulley | 0.0988 | 0.0000 | +-0.0988 |  |
| put_round_washer | 0.0000 | 0.0000 | +0.0000 |  |
| put_tooth_washer | 0.0000 | 0.0000 | +0.0000 |  |
| put_wheel | 0.3317 | 0.0000 | +-0.3317 |  |
| take_acorn_nut | 0.0000 | 0.0000 | +0.0000 |  |
| take_long_brace | 0.1010 | 0.0000 | +-0.1010 |  |
| take_nut | 0.0000 | 0.0000 | +0.0000 |  |
| take_objects | 0.0000 | 0.2989 | +0.2989 | YES |
| take_partial_model | 0.0472 | 0.0000 | +-0.0472 |  |
| take_pin_long | 0.5044 | 0.0000 | +-0.5044 |  |
| take_pin_middle | 0.0000 | 0.0000 | +0.0000 |  |
| take_pin_short | 0.0000 | 0.0000 | +0.0000 |  |
| take_pulley | 0.0000 | 0.0000 | +0.0000 |  |
| take_round_washer | 0.0000 | 0.0000 | +0.0000 |  |
| take_screw_pin | 0.0000 | 0.0000 | +0.0000 |  |
| take_small_screw_pin | 0.0000 | 0.0000 | +0.0000 |  |
| take_tooth_washer | 0.0040 | 0.0000 | +-0.0040 |  |
| take_wheel | 0.0000 | 0.1250 | +0.1250 | YES |
| take_wing | 0.0472 | 0.0000 | +-0.0472 |  |
| take_wing_beam | 0.0952 | 0.0000 | +-0.0952 |  |
| tighten_acorn_nut | 0.0000 | 0.0000 | +0.0000 |  |
| tighten_nut | 0.0000 | 0.7149 | +0.7149 | YES |

## MViTv2-S Only Classes


## ConvNeXt Only Classes

| Class | ConvNeXt Accuracy |
|-------|-------------------|
| fit_objects | 0.0000 |
| put | 0.6429 |
| put_acorn_nut | 0.0000 |
| put_pin_middle | 0.0000 |
| put_screw_pin | 0.0000 |
| put_small_screw_pin | 0.1176 |
| put_wing | 0.0000 |
| take_short_brace | 0.0875 |
