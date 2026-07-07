# F-3 / F-5 Corrections Log (File-157 Audit)

**Date:** 2026-07-07
**Agent:** Agent 97 (F-3 F-5 CORRECTIONS SPECIALIST)

---

## F-3 Correction: Never-Predicted Class List

**Error:** Files incorrectly listed never-predicted classes as `{1, 2, 3, 14, 23}`
**Correction:** Changed to `{1, 13, 16, 19, 23}`

### Files Corrected

| File | Status | Notes |
|------|--------|-------|
| 150_MASTER_SYNTHESIS.md | Already correct in HEAD | Fix committed by prior agent |
| 151_PER_HEAD_DEEP_ANALYSIS.md | Already correct in HEAD | Fix committed by prior agent |
| 152_IMPLEMENTATION_BUG_CATALOG.md | Already correct in HEAD | Fix committed by prior agent |
| 155_FINAL_PAPER_NARRATIVE.md | Already correct in HEAD | Fix committed by prior agent |
| 156_100_DEEP_QUESTIONS.md | Already correct in HEAD | Correction documentation preserved intentionally |
| 147_FINAL_PAPER_NARRATIVE_V4.md | **Corrected** | Line 38: replaced `(1, 2, 3, 14, 23)` with `(1, 13, 16, 19, 23)` |
| 154_SOTA_COMPARISON.md | N/A | File does not reference the never-predicted class list |

### Files with Intentional Old-List References (NOT modified)

| File | Reason |
|------|--------|
| 156_100_DEEP_QUESTIONS.md | "correct list is {1, 13, 16, 19, 23} **not** {1, 2, 3, 14, 23}" — correction documentation |
| 157_ULTIMATE_ANSWERS_150_156.md | References 150's error in context of describing the correction |

---

## F-5 Correction: D4+D1R 0.6364 3-Video Caveat

**Error:** D4+D1R 0.6364 reported without sample-size caveat
**Correction:** Added "(3-video subset)" to every occurrence

### Files Verified (all already correct in HEAD)

| File | Occurrences | Status |
|------|-------------|--------|
| 150_MASTER_SYNTHESIS.md | 7 | All caveated in HEAD |
| 151_PER_HEAD_DEEP_ANALYSIS.md | 1 | All caveated in HEAD |
| 154_SOTA_COMPARISON.md | 6 | All caveated in HEAD |
| 155_FINAL_PAPER_NARRATIVE.md | 1 | All caveated in HEAD |
| 150_SOTA_STATUS_V5.md | 1 | All caveated in HEAD |
| 144_VIDEO_BACKBONE_OPUS_BRIEF.md | 2 | All caveated in HEAD |

---

## Summary

- **Total files in 150-156 range:** 8 (144, 150, 150_SOTA_STATUS_V5, 151, 152, 153, 154, 155, 156)
- **Files needing F-3 correction:** 1 new (147) + 5 already committed in HEAD (150, 151, 152, 155, 156)
- **Files needing F-5 correction:** 0 new — all 6 files already caveated in HEAD
- **New edit applied:** 147_FINAL_PAPER_NARRATIVE_V4.md (F-3 class list fix)
- **Commit:** `fix: F-3 class list + F-5 3-video caveat corrections (File-157 audit)`
