# F-8 LOO-CV Standard Deviation Correction

**Date:** 2026-07-07
**Source:** File-157 F-8 audit
**Correction:** LOO-CV std 0.0158 -> 0.0163 (correct value: 0.0148 ± 0.0163)

## Search Results

### Pattern: `0.0148 ± 0.0158`
- **Occurrences found:** 0
- **Occurrences fixed:** 0

The string "0.0148 ± 0.0158" does not appear in any file in the repository. An exhaustive grep across all tracked files returned no matches.

### Scope Check

| Item | Status |
|---|---|
| Files 150-156 | Do not exist in AAIML directory (files present: 127, 129, 130, 132, 134, 135, 145) |
| SOTA_STATUS.md LOO row | No numeric LOO row present; SOTA_STATUS.md at `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` contains PSR metrics but no "0.0148 ± 0.0158" value |
| disclosures_v1.md | File does not exist; disclosure content is in `135_HONEST_DISCLOSURES_AND_PLAN_AMENDMENT.md` — contains textual LOO references but no numeric value |
| 156 section 9 "CI includes zero" | File 156 does not exist |

### LOO References Found (textual only, non-numeric)
- `135_HONEST_DISCLOSURES_AND_PLAN_AMENDMENT.md` — LOO-CV threshold stability placeholder (line 33), LOO-CV action items (lines 55, 71, 99)
- `132_OPUS_ANSWERS.md` — Week 2 plan mentions LOO CV thresholds (line 130)
- `130_MASTER_PLAN_TO_BEAT_SOTA.md` — LOO CV rationale and code todo (lines 192, 201, 442)

## Conclusion

Zero corrections applied. The incorrect value "0.0148 ± 0.0158" was not present in the codebase. No files required modification.
