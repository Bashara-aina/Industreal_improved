# Q36 / Q37 Data Resolution

**Date**: 2026-07-14
**Dataset root**: `/media/newadmin/master/POPW/datasets/industreal`
**Total recordings**: 84 (train=36, val=16, test=32)
**Total frames**: 207,072

---

## Q36: Per-Component PSR Positive Rate

### Per-frame occupancy (state=1 / total component-frames)

| Component | State=1 frames | State=0 frames | Pos rate (%) |
|-----------|---------------|---------------|-------------|
| comp0     | 207,042       | 30            | 99.99       |
| comp1     | 174,342       | 32,730        | 84.19       |
| comp2     | 176,235       | 30,837        | 85.11       |
| comp3     | 115,925       | 91,147        | 55.98       |
| comp4     | 29,700        | 177,372       | 14.34       |
| comp5     | 133,721       | 73,351        | 64.58       |
| comp6     | 121,448       | 85,624        | 58.65       |
| comp7     | 93,283        | 113,789       | 45.05       |
| comp8     | 92,562        | 114,510       | 44.70       |
| comp9     | 69,157        | 137,915       | 33.40       |
| comp10    | 44,468        | 162,604       | 21.47       |
| **ALL**  | **1,257,883** | **1,019,909** | **55.22**   |

- comp0 is always 1 from frame 0 (partial/sub-assembly model pre-assembled).
- Per-frame occupancy: **55.22%** -- well above 0.5%.

### Assembly event rate (0->1 transitions only)

| Metric                  | Value        |
|-------------------------|-------------|
| Total events            | 967          |
| Total comp-frame pairs  | 2,277,792    |
| **Event rate**          | **0.0425%**  |

- The V1 doc 218 claim "<0.5% overall" refers to **assembly event sparsity** (0.0425%), NOT per-frame occupancy.
- **Action needed**: Ensure paper text clearly says "assembly events occur in <0.5% of component-frame pairs" to avoid confusion with the 55% occupancy rate.

---

## Q37: AR_labels Class-0 Semantics

### ID 0 resolution

| Source                  | Claim         | Matches data? |
|-------------------------|---------------|---------------|
| V1 doc 218 (comment)    | ID 0 = 'NA'   | NO            |
| V2 dataset (actual CSV) | ID 0 = 'take_short_brace' | YES  |

- Actual max ID seen: 74 (IDs 0..74, 74 unique action classes; IDs 37 and 64 absent).
- Background/NA frames (no action span covering them): **42,292 frames (20.42%)**.
- These background frames are **implicit** -- they fall outside all `[start_frame, end_frame]` intervals in AR_labels.csv.

### Adjacent class info

| ID | Name               | Frames |
|----|--------------------|--------|
| 0  | take_short_brace   | 1,845  |
| 1  | align_objects      | 16,168 |
| 2  | take_pin_short     | 4,148  |
| 3  | plug_short_pin     | 7,599  |

### Codebase discrepancy

`src/config.py` line 306 comment says `index 0 = 'NA'`, but the actual implementation `_load_act_class_names()` scans AR_labels.csv and correctly discovers ID 0 = `take_short_brace`. The comment is **stale**; runtime behavior is correct.

---

## Actions Needed

1. **Paper text**: Clarify the PSR statistic -- "0.0425% assembly events" not "55% occupancy" or just "less than 0.5%".
2. **config.py line 306**: Fix stale comment `index 0 = 'NA'` -> `index 0 = 'take_short_brace'`.
3. **background mention**: When describing AR action classes, note that 20.42% of frames are implicit background (not ID 0). ID 0 is a real action class.
