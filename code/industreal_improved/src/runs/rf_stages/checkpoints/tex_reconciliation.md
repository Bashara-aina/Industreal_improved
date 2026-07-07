# TEX Reconciliation — Stale Numbers Found & Fixed

**Date:** 2026-07-07
**Source:** Opus 133 SS0 C-3, C-4, C-5, C-6; disclosures_v1.md SS2, SS6, SS7
**Frozen numbers source:** disclosures_v1.md (Opus 140 batch)

---

## Summary

| Category | Stale Value | Correct Value | Status |
|---|---|---|---|
| Activity taxonomy | 47 hybrid verb-groups | 69 hybrid verb-groups | **FIXED** |
| Forward angular MAE | 7.83deg (subsample) / 9.94deg (full val) | 9.14deg (bootstrap) | **FIXED** |
| Up angular MAE | 7.06deg (subsample) / 8.28deg (full val) | 7.78deg (bootstrap) | **FIXED** |
| PSR POS | 0.9693 / 0.999 (headline) | Moved to footnote | **FIXED** (per disclosures_v1.md SS2) |
| PSR F1@3 | 0.144 / 0.000 (pre-fix collapse) | 0.7018 (post-fix) | **FOOTNOTED** (paper thesis = pre-fix collapse; footnote adds post-fix state) |
| GFLOPs | 245.3 | ~93 (pending freeze remeasurement) | **FLAGGED** (C-6, requires separate freeze-protocol measurement) |
| Params | 46.47M | ~53M (day1 checkpoint) | **FLAGGED** (C-6, pending reconciliation) |

---

## C-3. Activity Taxonomy: 47 -> 69

**Evidence:** Opus 133 SS0 C-3 confirms the eval stack uses 75->69 hybrid grouping (repo-verified: `act_remap_75_to_69.json`, num_groups=69, 75 ids). The .tex reports 47 groups.

### Files affected

| File | Line | Stale text | Replacement |
|---|---|---|---|
| `popw_aaiml2027.tex` | 96 | "47 hybrid verb-grouped action classes (reduced from 75 fine-grained)" | "69 hybrid verb-grouped action classes (reduced from 75 fine-grained)" |
| `popw_aaiml2027.tex` | 172 | "Macro-F1 (47 hybrid)" | "Macro-F1 (69 hybrid)" |
| `popw_aaiml2027.tex` | 177 | "--- & 69-class, per-frame *" | _Note: line 177 already says "69-class" in the baseline column — the "47 hybrid" in the metric column was inconsistent_ |
| `popw_aaiml2027.tex` | 355 | "47 hybrid verb-grouped, not 75 fine-grained" | "69 hybrid verb-grouped, not 75 fine-grained" |

---

## C-4. PSR Story: Pre-Fix vs Post-Fix

**Evidence:** disclosures_v1.md SS2 & SS6 show the pre-fix PSR collapsed (F1=0, all-ones), but post-fix the head repair (GELU->LeakyReLU) yields F1=0.7018. The AAIML paper's thesis is the three pathologies, of which the PSR collapse (Pathology 1) is central.

### Ambiguity assessment
- **F1=0 / 0.144 -> 0.7018:** The paper's central contribution is documenting the PSR collapse. Changing the F1 values would undermine the thesis. Resolution: **keep the pre-fix F1 values in the body** (they are the empirical finding the pathology documents), **add a footnote** saying "post-fix F1 = 0.7018 after head repair (LeakyReLU init)".
- **POS 0.9693/0.999 -> footnote:** disclosures_v1.md SS2 states "POS appears only in the appendix; per-frame F1 and transition F1 are the PSR metrics." The AAIML paper currently uses POS as a headline in the abstract. Resolution: **move POS references to footnotes** per disclosures_v1.md guidance.

### Changes applied
| File | Line | Change |
|---|---|---|
| `popw_aaiml2027.tex` | 35 | Added footnote to "PSR F1=0" noting post-fix repair yields F1=0.7018 |
| `popw_aaiml2027.tex` | 35 | Stripped "PSR POS=0.9693" from abstract (relegated to footnote per disclosures_v1.md SS2) |
| `popw_aaiml2027.tex` | 53 | Stripped "PSR POS 0.969" as headline claim; added footnote reference |
| `popw_aaiml2027.tex` | 177 | Footnote on PSR F1@3: stale 0.144/0.000 with note on post-fix 0.7018 |

---

## C-5. Head Pose Numbers: Three Eras

**Evidence:** disclosures_v1.md SS7 documents the eval-index bug: position channels [3:6] were read as up-vector (giving 26.20deg), corrected to [6:9] yields 7.78deg. Forward MAE bootstrap mean is 9.14deg [7.74-10.87].

The .tex (AAIML version) carried stale era-1 numbers: forward 7.83/9.94, up 7.06/8.28.

### Changes applied
| File | Line | Stale | Replacement |
|---|---|---|---|
| `popw_aaiml2027.tex` | 174 | "Head pose & Forward MAE (deg) & 7.83 & 9.94" | "9.14\sym{1}" with footnote "\sym{1} Bootstrap mean [7.74-10.87]; excludes outlier recording 14_assy_0_1" |
| `popw_aaiml2027.tex` | 175 | "Head pose & Up MAE (deg) & 7.06 & 8.28" | "7.78\sym{1}" with same footnote |
| `popw_aaiml2027.tex` | 35 | "7.83$^\circ$ forward angular MAE" | "9.14$^\circ$ forward angular MAE [7.74-10.87]" |
| `popw_aaiml2027.tex` | 53 | "first ego-pose baseline (7.83$^\circ$)" | "first ego-pose baseline (9.14$^\circ$)" |
| `popw_aaiml2027.tex` | 199 | "Ego-pose fwd MAE & 7.83$^\circ$" | "Ego-pose fwd MAE & 9.14$^\circ$" |
| `popw_aaiml2027.tex` | 279 | "subsample 7.83$^\circ$ includes the fix, full-set 9.94$^\circ$ is pre-fix" | Updated to reflect current correct numbers |
| `popw_aaiml2027.tex` | 371 | "first-of-kind ego-pose baseline (7.83$^\circ$)" | "first-of-kind ego-pose baseline (9.14$^\circ$)" |

---

## C-6. Params/GFLOPs: 245 vs 93

**Not fixed.** C-6 states: ".tex: 46.47M / 245.3 GFLOPs / 11.02 FPS ('measured'). 129 SS5: ~53M / ~93 GFLOPs (citing day1 checkpoint file)." This requires re-measurement on the freeze checkpoint with a committed measurement script before the number can be updated. Flagged for freeze protocol action.

---

## Unfixed Items (Requires Author Discretion)

1. **PSR F1=0 as paper thesis vs post-fix F1=0.7018.** The AAIML paper's central claim (three training pathologies) depends on documenting the pre-fix collapse. I kept the pre-fix F1 values in the body with a footnote noting post-fix repair. If the paper should instead report the post-fix system, the entire PSR pathology section needs rewriting.

2. **GFLOPs/params discrepancy (C-6).** Needs re-measurement on freeze checkpoint before updating.

3. **ICHCIIS-26 paper (`popw_ichciis26.tex`):** Still in placeholder/outline stage; no stale numbers in compiled text (only in comments at lines 5, 43, 95). The comments reference "8.14deg forward MAE, 7.06deg up MAE" which are stale.

4. **Main paper (`popw_paper_improved.tex`):** Most results are `\todo` / `\popwres` placeholders. Notation table has `y^{act} \in {1,...,74}` which is the raw class count (not the grouped 47/69 taxonomy). Line 273 says "74 classes" for the activity head — this is also the raw count, not a taxonomy reference. No changes needed until results are filled in.
