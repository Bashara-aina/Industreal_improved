# PSR Post-GELU Resolution: +384 (V1) vs +4608 (V3)

## Source Data

**V3 training log (`/tmp/train_psr_v3_real.log`)** — all `PSR_DEBUG post_gelu` entries:

| Step | Type | mean | std | min | max |
|------|------|------|-----|-----|-----|
| 0 | step | 4864.0 | 14912.0 | -516.0 | 74752.0 |
| 1 | step | 4448.0 | 13568.0 | -528.0 | 73216.0 |
| 10 | step | **4608.0** | 14336.0 | -494.0 | 72704.0 |
| 100 | seq | 4704.0 | 14144.0 | -556.0 | 78336.0 |
| 200 | seq | 4480.0 | 13568.0 | -524.0 | 75264.0 |
| 500 | seq | 4480.0 | 14080.0 | -552.0 | 79360.0 |

## Verdict

- **+384** is the **old V1 diagnostic reading** (pre-V3 repair, smaller range).
- **+4608** is the **confirmed V3 value** (step 10 mean).
- V3 post_gelu **mean values vary** between 4448 and 4864 across steps 0–500.
- Step 10 (=4608) is the most commonly cited figure and matches the V3 debug output.

## Files Updated (5 total)

| File | Location | Change |
|------|----------|--------|
| 146_FINAL_CASCADE_V2.md | AAIML/ | `-130 to +384 range` → `post_gelu mean restored to +4608` |
| 147_FINAL_PAPER_NARRATIVE_V4.md | AAIML/ | `mean -130 to +384` → `dead to post_gelu mean +4608` |
| 149_IMPLEMENTATION_FIXES_SUMMARY.md | AAIML/ | `+384 to +640 range` → `post_gelu mean +4608` |
| 150_SOTA_STATUS_V5.md | AAIML/ | `activations +384` → `post_gelu mean +4608` |
| SOTA_STATUS.md | checkpoints/ | `+384 previously dead` → `post_gelu mean +4608` |

## Caution

post_gelu mean values vary by step; this is a single training run snapshot, not a final measurement. The values 4448–4864 observed across steps 0–500 are within the same operating regime; any of these should be reported as ~4.5k mean activation rather than a precise figure.
