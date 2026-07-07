# Activity Temporal Probe - Per-Clip Mean-Pool Results

- **Date:** 2026-07-07 14:12:50
- **Reference:** Opus 141 ACT-ARCH-2
- **Clip size:** 16 frames, stride=8
- **Pooling:** Mean

## Summary

| Metric | Value |
|--------|-------|
| Per-frame linear probe (reference) | 0.2169 |
| Per-frame majority baseline | 0.2217 |
| Clip majority baseline | 0.2288 |
| **Clip mean-pool top-1** | **0.0723** |
| Macro F1 | 0.0458 |
| Weighted F1 | 0.0501 |
| Delta vs majority | -0.1493 |
| Delta vs per-frame | -0.1446 |
| Gating | **FAIL** |

## Data

- Train: 26322 frames -> 3239 clips
- Val: 35456 frames -> 4396 clips
- Val recordings: 16

## Per-Class F1 (sorted by support)

| Class | F1 | Precision | Recall | Support |
|-------|----|-----------|--------|--------|
| take_partial_model (id=8) | 0.0549 | 0.5800 | 0.0288 | 1006 |
| check_instruction (id=7) | 0.0000 | 0.0000 | 0.0000 | 347 |
| take_pin_short (id=2) | 0.0000 | 0.0000 | 0.0000 | 280 |
| fit_long_brace (id=33) | 0.0000 | 0.0000 | 0.0000 | 174 |
| pull_objects (id=41) | 0.0943 | 0.2381 | 0.0588 | 170 |
| fit_short_brace (id=30) | 0.0000 | 0.0000 | 0.0000 | 156 |
| take_tooth_washer (id=4) | 0.1293 | 0.1301 | 0.1284 | 148 |
| tighten_nut (id=6) | 0.0000 | 0.0000 | 0.0000 | 147 |
| take_nut (id=5) | 0.0000 | 0.0000 | 0.0000 | 124 |
| put_pin_middle (id=42) | 0.0819 | 0.1346 | 0.0588 | 119 |
| unknown_37 (id=37) | 0.1059 | 0.1636 | 0.0783 | 115 |
| fit_acorn_nut (id=68) | 0.2048 | 0.2881 | 0.1589 | 107 |
| plug_wheel (id=28) | 0.2081 | 0.1179 | 0.8878 | 98 |
| take_round_washer (id=17) | 0.0000 | 0.0000 | 0.0000 | 88 |
| take_long_brace (id=9) | 0.0000 | 0.0000 | 0.0000 | 87 |
| take_wheel (id=21) | 0.0000 | 0.0000 | 0.0000 | 85 |
| fit_tooth_washer (id=31) | 0.0000 | 0.0000 | 0.0000 | 84 |
| plug_short_pin (id=3) | 0.0426 | 0.0329 | 0.0602 | 83 |
| put_long_brace (id=48) | 0.1254 | 0.0826 | 0.2603 | 73 |
| take_objects (id=43) | 0.0299 | 0.0294 | 0.0303 | 66 |
| fit_objects (id=53) | 0.1696 | 0.1092 | 0.3800 | 50 |
| tighten_acorn_nut (id=19) | 0.1856 | 0.1800 | 0.1915 | 47 |
| plug_partial_model (id=25) | 0.1053 | 0.0667 | 0.2500 | 44 |
| plug_pin_middle (id=26) | 0.0000 | 0.0000 | 0.0000 | 43 |
| loosen_nut (id=39) | 0.0000 | 0.0000 | 0.0000 | 42 |
| take_screw_pin (id=10) | 0.0925 | 0.0507 | 0.5278 | 36 |
| browse_instruction (id=29) | 0.0865 | 0.0537 | 0.2222 | 36 |
| pull_partial_model (id=49) | 0.0149 | 0.0101 | 0.0286 | 35 |
| put_screw_pin (id=35) | 0.0000 | 0.0000 | 0.0000 | 32 |
| plug_screw_pin (id=16) | 0.0476 | 0.0370 | 0.0667 | 30 |
| put_partial_model (id=44) | 0.1364 | 0.2143 | 0.1000 | 30 |
| plug_pin_long (id=22) | 0.1124 | 0.0667 | 0.3571 | 28 |
| align_objects (id=1) | 0.1277 | 0.1364 | 0.1200 | 25 |
| take_instruction (id=11) | 0.0000 | 0.0000 | 0.0000 | 24 |
| take_acorn_nut (id=18) | 0.0000 | 0.0000 | 0.0000 | 23 |
| take_pin_middle (id=20) | 0.0769 | 0.0545 | 0.1304 | 23 |
| pull_wheel (id=38) | 0.0000 | 0.0000 | 0.0000 | 23 |
| put_round_washer (id=54) | 0.0488 | 0.0500 | 0.0476 | 21 |
| plug_small_screw_pin (id=66) | 0.0937 | 0.0682 | 0.1500 | 20 |
| take_wing (id=23) | 0.0556 | 0.0377 | 0.1053 | 19 |
| fit_pulley (id=55) | 0.0000 | 0.0000 | 0.0000 | 19 |
| take_pin_long (id=13) | 0.0923 | 0.0536 | 0.3333 | 18 |
| put_objects (id=45) | 0.0000 | 0.0000 | 0.0000 | 17 |
| put_short_brace (id=52) | 0.0000 | 0.0000 | 0.0000 | 17 |
| take_wing_beam (id=15) | 0.0458 | 0.0254 | 0.2308 | 13 |
| pull_pin_middle (id=58) | 0.0000 | 0.0000 | 0.0000 | 12 |
| put_nut (id=40) | 0.0462 | 0.0252 | 0.2727 | 11 |
| take_small_screw_pin (id=65) | 0.0370 | 0.0227 | 0.1000 | 10 |
| fit_round_washer (id=32) | 0.0286 | 0.0164 | 0.1111 | 9 |
| pull_pin_short (id=46) | 0.0645 | 0.0357 | 0.3333 | 9 |
| pull_screw_pin (id=61) | 0.0000 | 0.0000 | 0.0000 | 9 |
| loosen_acorn_nut (id=63) | 0.0000 | 0.0000 | 0.0000 | 9 |
| fit_partial_model (id=64) | 0.0000 | 0.0000 | 0.0000 | 9 |
| put_pin_long (id=14) | 0.0000 | 0.0000 | 0.0000 | 7 |
| fit_wing_beam (id=56) | 0.0392 | 0.0227 | 0.1429 | 7 |
| put_pulley (id=60) | 0.0000 | 0.0000 | 0.0000 | 6 |
| put_acorn_nut (id=62) | 0.0000 | 0.0000 | 0.0000 | 6 |
| put_wing (id=24) | 0.0000 | 0.0000 | 0.0000 | 5 |
| fit_nut (id=34) | 0.0000 | 0.0000 | 0.0000 | 5 |
| put_wheel (id=36) | 0.0150 | 0.0078 | 0.2500 | 4 |
| put_instruction (id=12) | 0.0392 | 0.0208 | 0.3333 | 3 |
| put_wing_beam (id=59) | 0.0000 | 0.0000 | 0.0000 | 3 |

Non-zero classes: 62/69
Non-predicted (zero F1 with support): 29
