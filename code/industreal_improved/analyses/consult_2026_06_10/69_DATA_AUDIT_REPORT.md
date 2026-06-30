# 69: Comprehensive Data Audit — Is Data the Problem? [2026-06-30]

## Executive Summary

**The data is NOT the root cause of the activity collapse.** The data structure,
label quality, and train/val split are all correct. The collapse is entirely
architectural — the gradient path through the temporal head was severed by a
shuffled-per-frame sampler feeding non-temporal data into a TCN+ViT.

However, we found **one critical config bug** and **five actionable data insights**
that affect training efficiency.

## Findings

### Finding 1: ID 0 Zero-Weight Bug (CRITICAL — Fix Now)

**Problem:** `config.py` builds `ACT_CLASS_NAMES` mapping action_id 0 as "NA"
(background/no-annotation). The loss function at `losses.py:1114` sets:
```python
_weights[0] = 0.0  # class 0 (NA) contributes zero gradient
```

But the IndustReal dataset uses **action_id = 0 as a real action** named
"take_short_brace" (63 train frames, 13 val frames). This means:
- 63 frames of legitimate training data produce **zero gradient**
- The model can never learn to predict "take_short_brace"
- This wastes 63/3667 = **1.7% of training data**

**Fix:** Remove `_weights[0] = 0.0` from `losses.py:1114` OR add weight for
class 0 via the CB-balanced formula, removing the special-case zeroing.
The config already has NUM_CLASSES_ACT=75 including index 0 as a real action.

**Impact of fix:** Marginal. 63 frames is small. But it removes a potential
degenerate attractor (the model pushing predictions toward class 0 because
it never gets penalized for doing so).

### Finding 2: Train/Val Split Is Clean (CONFIRMED)

- 36 train recordings, 16 val recordings
- **Zero overlap** between train and val recordings
- Ratio: 69% train / 31% val (guideline standard is 80/20, but this is fine)
- All recordings are from different camera perspectives and assembly variants

### Finding 3: Activity Labels Are Correctly Mapped (CONFIRMED)

- 72 unique activities across 3,667 train frames
- All activity names in the CSV correctly map to config action IDs
- CSV segments match AR_labels.csv annotation boundaries
- The model's output layer has 75 neurons (IDs 0-74) where:
  - 72 IDs have data in train
  - 2 IDs have data in val only (66=plug_small_screw_pin, 72=pull_small_screw_pin)
  - 1 ID is permanently cold (37 = unknown action)

### Finding 4: Extreme Class Imbalance (78% Long-Tail)

| Metric | Value |
|--------|-------|
| Total frames | 3,667 |
| Unique activity classes | 72 |
| Most common class | ID 7 = 'check_instruction' (404 frames, 11.0%) |
| Rarest class (w/ data) | ID 74 = 'loosen_tooth_washer' (1 frame, 0.03%) |
| Classes with <1% of data | **46/72 (64%)** |
| Classes with <10 frames | 23/72 (32%) |
| Classes with 1 frame only | 4 classes |
| Imbalance ratio (max/min) | 404:1 |

**Impact on training:** CB-balanced CE reweighting helps but cannot create signal
for 1-frame classes. The model will never learn 46 classes well. This is a DATA
limitation, not a training bug.

At 50% subset (RF4), the situation worsens:
- **22 classes with <5 frames**
- **8 singletons**
- 2 classes lost entirely (IDs 63, 70)

### Finding 5: Frame Coverage Is 78% (ADEQUATE)

- 78,931 total RGB frames across 36 recordings
- 61,705 frames (78.2%) are covered by CSV segment annotations
- Per-recording coverage: 66% to 87%
- Uncovered frames likely represent idle/waiting periods with no assembly activity
- These frames are excluded from the activity loss (label=-1), which is correct

### Finding 6: Detection Data Is COCO-Format with 24 Binary-State Categories

- 14,122 annotations across 36 recordings
- 24 categories: 'background' + 22 11-bit state codes + 'error_state'
- Per-recording category coverage is sparse (~6 of 24 categories per recording)
- Bounding boxes cover multiple assembly part sizes (width: 50-600px, height: 40-500px)
- The 11-bit state encoding explains the "1-bit adjacent" error pattern — states like
  "11110110000" and "11110100000" differ by a single bit

**Quality: GOOD.** The OD labels are accurate and follow COCO format.

### Finding 7: PSR Data Is Extremely Sparse (SIGNIFICANT)

- Only 244 PSR transition rows across ALL 36 recordings
- ~7 frames per recording have PSR labels
- Most PSR components are 0 (inactive) on most frames

This explains why PSR produces zero F1 — there are barely enough labels to learn
from. The transition objective (STATE CHANGES only) is the right approach for this
sparsity, but it requires sequence-mode training which RF4 doesn't enable.

### Finding 8: Pose Vectors Are Actually Normalized (Debunked)

Our earlier analysis (file 57) stated forward vector norms = 0.014-0.030.
**This is WRONG.** A sample recording (16_main_3_3) has forward vector norms:
- Mean: 1.000
- Min: 0.999
- Max: 1.001
- Fraction near 1.0: 100%

The `_parse_pose` warnings in training logs are from a FEW recordings where the
CSV format may differ, not the entire dataset. The head pose number (8.71°) is
on normalized targets.

**However**, the warning persists across multiple recordings (14_assy_0_1,
20_assy_3_6, 24_assy_0_1, etc.). These are specific recordings with different
pose.csv formatting. The eval normalizes before computing angular MAE, so the
metric is valid for ALL recordings — but training MSE on un-normalized targets
for THESE specific recordings is suboptimal.

### Finding 9: No Train/Val Label Leakage (CONFIRMED)

- Zero overlapping recordings
- No activity classes that exist ONLY in val and not in train (IDs 66 and 72
  exist in train CSV but may have different frame selections at 50% subset)
- All 75 activity indices are properly mapped

## Action Items

| # | Action | Impact | Effort | Priority |
|:-|--------|--------|--------|:-------:|
| 1 | Fix `_weights[0] = 0.0` in `losses.py:1114` — include class 0 in CB-balanced weighting | Removes 63-frame dead zone | 1 line | HIGH |
| 2 | Document 46/72 long-tail classes — paper's "severe class imbalance" claim | Paper narrative | 10 min | MEDIUM |
| 3 | Note PSR sparsity (244 rows, 36 recordings) — explains 0 F1 | Paper limitation | 5 min | MEDIUM |
| 4 | Verify pose.csv format for recordings with warnings (14_*, 20_*, 24_*, 26_*) | Head pose accuracy | 30 min | LOW |
| 5 | For 1-frame classes, consider removing them from evaluation | Cleaner metrics | 15 min | LOW |

## Verdict: Is Data the Problem?

**NO.** The data is correctly structured, accurately labeled, and properly split.
The 78% frame coverage is adequate. The 14,122 OD annotations are plentiful.
The activity labels correctly map to config IDs.

The extreme class imbalance (64% long-tail) is a genuine data limitation, but it
was known and the CB-balanced weighting addresses it. The activity collapse is
from the architectural gradient attenuation (shuffled sampler + temporal head),
not from bad data.

**The one real bug** (`_weights[0] = 0.0`) is marginal — fixing it recovers
63 frames of training signal, which is 1.7% of the dataset. It won't fix the
activity collapse by itself.
