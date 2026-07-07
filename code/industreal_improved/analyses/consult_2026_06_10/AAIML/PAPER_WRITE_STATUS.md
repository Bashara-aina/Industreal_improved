# Paper Write-Up Status Assessment (File 155 Readiness)

**Date:** 2026-07-07
**Target:** AAIML 2026 submission
**Agent:** 111 -- FINAL PAPER WRITE-UP STATUS SPECIALIST

---

## 1. Section Inventory (12-Section Check)

File 155 has the following top-level sections:

| # | Section | Status |
|---|---------|--------|
| 1 | Abstract (200 words) | Present, clear |
| 2 | Introduction | Present, 3 pathologies enumerated |
| 3 | Method: Model Architecture | Present |
| 4 | Method: Training Configuration | Present |
| 5 | Method: The 9 Implementation Fixes (Detailed) | Present, numbered 1-9 |
| 6 | Results: Head Pose (First Baseline) | Present, with bootstrap CIs |
| 7 | Results: Detection (D1R single-task vs D3 multi-task) | Present, multi-condition table |
| 8 | Results: PSR (GELU Dead -> LeakyReLU Fix) | Present, with activation diagnostics |
| 9 | Results: Activity (Backbone Wrong Type) | Present, with MViTv2-S probe result |
| 10 | Results: FiLM Analysis | Brief but present |
| 11 | Discussion | Present (5 sub-sections) |
| 12 | Conclusion | Present, with 3 lessons |

**Plus extras:** Reproducibility, File Paths Summary, The 50 Deep Questions for Opus (Reference).

**Verdict: 12 sections present.**

---

## 2. File Path Accuracy

File paths in the document reference source code and scripts within the project structure:

- `src/models/model.py` -- references the PSR head definition
- `config.py` -- training configuration
- `scripts/train_detection_d1r.sh` etc. -- training entry points
- `src/evaluation/eval_pose_kalman.py` -- evaluation code
- `src/runs/rf_stages/checkpoints/` -- checkpoint and results directories

These paths are structurally consistent with the repository layout at `/media/newadmin/master/POPW/working/code/industreal_improved/`.

**Missing Opus analysis file paths:** The text references "Opus 140 and 141 audits" and "file 127_50_DEEP_QUESTIONS_FOR_OPUS.md" but these analysis file paths are not included in the File Paths Summary section. All three referenced files exist on disk (140: 197 lines, 141: 349 lines, 127: 704 lines). Consider adding them to the paths summary for cross-reference.

**Verdict: Source code paths correct. Opus analysis paths absent from summary.**

---

## 3. Eight Numbered Disclosures Check

**NOT PRESENT in file 155.** The 8 numbered disclosures are defined in the Opus V2 analysis (file 140, section 4) and are already written into `popw_aaiml2027.tex` at lines 273-294 (subsection "Honest Disclosures"). But file 155 does not contain any disclosures section or reference to the 8 items.

The 8 disclosures (from 140_OPUS_ANSWERS_V2.md and the .tex file):
1. Detection: three distinct numbers, all contextualized
2. PSR F1=0 is real model collapse, not a bug
3. PSR POS beats SOTA but with caveats
4. Activity is per-frame, NOT comparable to MViTv2
5. Ego-pose is first IndustReal baseline
6. TTA broken, not evaluated
7. COCO-pretrained YOLOv8m does not transfer
8. No cross-dataset validation completed

**Gap severity: HIGH.** The 8 disclosures are a core integrity feature of the paper. File 155 needs a dedicated disclosures section or the disclosures need to be integrated into the discussion.

---

## 4. Three Documented Pathologies Check

**PRESENT.** All three are clearly enumerated in the Introduction and thoroughly discussed:

- **Pathology 1: PSR GELU Dead.** GELU saturation at mean pre-activation -130 with +0.1 bias 1300x too small. 99.7% of activations in dead-zone. Discussed in Results (PSR) and Discussion (section "The Three Pathologies").
- **Pathology 2: Detection Class Collapse.** 5 classes never predicted across 38k frames. 91.9% empty frames. Discussed in Results (Detection) and Discussion.
- **Pathology 3: Activity Backbone Mismatch.** ImageNet ConvNeXt provides zero action signal (linear probe 0.2169 vs majority baseline 0.2217). MViTv2-S probe at 0.3810.

Each has root cause, evidence, and fix described.

---

## 5. Abstract Quality

The abstract is approximately 200 words and covers:
- What was done (4-head multi-task on IndustReal with ConvNeXt-Tiny backbone)
- Headline result (single-task detection BEATS SOTA ceiling: mAP50=0.995 vs WACV 0.95)
- Three failure modes with key numbers
- The contribution (pathology analysis)
- Summary of fixes and outcomes

**Verdict: Clear and complete.**

---

## 6. Paper Structure Completeness

File 155 covers the narrative flow well but has the following structural gaps relative to the LaTeX paper (`popw_aaiml2027.tex`):

**Missing sections (present in .tex, absent in 155):**
- Honest Disclosures section (8 items, `.tex` lines 273-294)
- "What the system is good for / not good for" framing (`.tex` lines 300-304)
- Cross-Dataset Generalization: IKEA ASM Plan (`.tex` lines 315-332, with table)
- Lessons for the MTL Community (`.tex` lines 306-312, enumerated)
- Related Work (not in .tex either -- also a gap)
- References/bibliography (not in .tex rendering, but implied by `\cite{}` calls)

**Missing structural elements:**
- No figures or tables referenced (the .tex has formal table environments)
- No explicit "Limitations" section (limitations are discussed inline)
- No formal "Experimental Setup" (hyperparameters, compute budget, data splits)

**File 155 also does not match the .tex on PSR framing:**
- `.tex`: "PSR F1=0 is real model collapse" -- emphasizes the pre-fix broken state
- `155`: "PSR achieves 0.7018 F1" -- emphasizes the post-fix recovery

These are different narrative phases. The .tex was written when PSR was known to be F1=0. File 155 reflects newer understanding with the fix applied. This discrepancy needs reconciliation.

---

## 7. Gaps Requiring Ablation Results

The following items in file 155 are marked as "in flight" or pending:

| Item | Status | Impact |
|------|--------|--------|
| V3 PSR repair training | In flight | Expected F1 > 0.78, would upgrade PSR verdict |
| Single-task ConvNeXt detection training | In flight | Needed for honest multi-task cost denominator |
| MViTv2-S fine-tuning | 2-week investment | Expected 0.45-0.55 activity, approaching SOTA 0.622 |

These are the core ablation experiments. File 155 correctly identifies them as pending but the paper cannot be finalized until they land.

---

## 8. Summary Assessment

| Criterion | Status |
|-----------|--------|
| 12 sections present | PASS |
| 3 pathologies documented | PASS |
| Abstract clear | PASS |
| Source code paths correct | PASS |
| 8 numbered disclosures present | **FAIL -- Missing** |
| Opus analysis file paths referenced | **FAIL -- Not in Paths Summary** |
| Structure matches LaTeX paper | **FAIL -- Significant divergence** |
| Ablation results complete | **FAIL -- 3 items in flight** |
| Related Work section | **FAIL -- Not present in either 155 or .tex** |

**Readiness level:** NOT READY for final write-up. File 155 is a strong narrative draft but has three blocking gaps:
1. The 8 Honest Disclosures must be added (adapt from 140_OPUS_ANSWERS_V2.md section 4 or from popw_aaiml2027.tex lines 273-294)
2. The narrative must be reconciled with the LaTeX paper structure (disclosures section, what-the-system-is-good-for, cross-dataset, lessons)
3. Ablation results must land before the narrative can be finalized (V3 PSR, single-task ConvNeXt detection, MViTv2-S fine-tuning)

**Recommended next steps:**
1. Add the 8 Honest Disclosures section to file 155 (1 hour, text already written in .tex)
2. Reconcile PSR framing between 155 (post-fix 0.7018) and .tex (pre-fix F1=0) -- both are true, they describe different states
3. Add Opus analysis file paths to the File Paths Summary
4. After ablation results land: fill in the three pending numbers and write the Related Work section
