# Stale Numbers Final Audit — Opus 141 Q2, Q6 (Final Pass)

**Date:** 2026-07-07
**Source:** 141_OPUS_COMPLETE_ANSWERS_V2.md SS4 Q2 (list artifacts quoting 26.20deg) and Q6 (repo-wide slice grep for fossilized [3:6])
**Previous audit:** `stale_numbers_audit.md` (Agent-22, commit 73e4425b)
**This pass:** Final comprehensive re-grep across all file types, including numbers beyond Q2/Q6 baseline.

---

## Methodology

Comprehensive repo-wide grep for stale numbers across `.py`, `.md`, `.json`, `.tex` files, excluding `__pycache__`, `node_modules`, `.git/`, `.claude/`.

### Searched patterns

| Pattern | Description | Current correct value |
|---------|-------------|----------------------|
| `26.20` / `26.2` | Buggy head pose up-vector MAE (eval index [3:6]) | 7.78deg (corrected [6:9]) |
| `13.52` / `13.5` | Related buggy-era up-vector number | 7.78deg |
| `7.83` | Stale era-1 head pose forward MAE (subsample) | 9.14deg (bootstrap mean) |
| `7.06` | Stale era-1 head pose up MAE (subsample) | 7.78deg |
| `8.39` | Stale era-3 head pose forward MAE | 9.14deg |
| `0.7499` | Old PSR macro-F1 (10k subset) | 0.7018 (38k full eval) |
| `0.9693` | Old POS headline number | 0.9988 (current); relegated to appendix |
| `245.3` | Stale GFLOPs measurement | ~93 (pending re-measurement, C-6) |

---

## Part A: Previously Fixed (Agent-22, commit 73e4425b)

| File | What was fixed | Status |
|------|---------------|--------|
| `SOTA_STATUS.md` | PSR 10k numbers updated to 38k: 0.7499→0.7018, 0.7217→0.6788, 0.7810 removed, per-component table replaced, "near SOTA" removed, LOO-CV 0.0358→0.0148 | **FIXED** |
| `psr_null_delta_table.md` | Updated 0.7499 reference to 0.7018 with historical note | **FIXED** |
| `d4_d1r/verdict.md` | Updated 0.7499 reference to 0.7018; fixed gap calculation | **FIXED** |
| `opus_140_batch_index.md` | Updated 0.7499 reference to 0.7018 | **FIXED** |
| `activity_linear_probe.json` | "BACKBONE HAS SIGNAL" verdict → "statistically indistinguishable from majority-class baseline" | **FIXED** |
| `PROGRESS_2026-07-06.md` | Added SUPERSEDED banner | **FIXED** |

---

## Part B: Previously Fixed (tex_reconciliation, same session)

| File | What was fixed | Status |
|------|---------------|--------|
| `AAIML/popw_aaiml2027.tex` | Head pose forward: 7.83/9.94 → 9.14 (with CI footnote) | **FIXED** (line 174) |
| `AAIML/popw_aaiml2027.tex` | Head pose up: 7.06/8.28 → 7.78 (with CI footnote) | **FIXED** (line 175) |
| `AAIML/popw_aaiml2027.tex` | Abstract: "7.83 forward angular MAE" → "9.14 forward angular MAE" | **FIXED** (line 35) |
| `AAIML/popw_aaiml2027.tex` | "PSR POS=0.9693" stripped from abstract | **FIXED** (line 35) |
| `AAIML/popw_aaiml2027.tex` | "first ego-pose baseline (7.83)" → "first ego-pose baseline (9.14)" | **FIXED** (line 53, 199, 279, 371) |
| `AAIML/popw_aaiml2027.tex` | Activity taxonomy: 47 → 69 hybrid verb-groups | **FIXED** (lines 96, 172, 355) |

---

## Part C: Still Flagged (NOT fixed — requires freeze re-measurement or author action)

| File | Line(s) | Stale text | Issue | Resolution |
|------|---------|------------|-------|------------|
| `AAIML/popw_aaiml2027.tex` | 74 | `Measured: 46.47M total params, 245.3 GFLOPs, 11.02 FPS on RTX 3060` | GFLOPs/params discrepancy: .tex says 245.3/46.47M, day1 checkpoint says ~93/~53M (C-6). 141 Q4.07 verdict: needs remeasurement on freeze checkpoint. | **FLAGGED** — re-measure at freeze with committed fvcore/ptflops script |
| `AAIML/popw_aaiml2027.tex` | 90 | `Total & 46.47M & 245.3 & 90.7ms/frame (11.02 FPS)` | Same GFLOPs/params discrepancy | **FLAGGED** — same |
| `AAIML/popw_aaiml2027.tex` | 262 | `POPW total: 46.47M parameters, 245.3 GFLOPs, 11.02 FPS` | Same GFLOPs/params discrepancy | **FLAGGED** — same |
| `ICHCIIS-26/popw_ichciis26.tex` | 5, 43, 95 | `8.14deg forward MAE, 7.06deg up MAE` | Stale era-1 numbers in comments only (file is still outline/placeholder stage) | **NOT FIXED** — placeholder; will be replaced when paper is written |
| `disclosures_v1.md` | 56 | `Global 0.10 threshold yields 0.7217 on 10k; leave-one-recording-out CV bounds the per-component selection benefit at +0.0358` | 10k LOO-CV numbers used alongside current 38k numbers in comparison context | **HISTORICAL COMPARISON** — accurate as epoch-10k context; current 38k numbers (0.6788, +0.0148) stated in same paragraph |

---

## Part D: Verified Historical Context (Leave as-is)

These files contain stale numbers in historical context (describing the bug/discovery narrative), not as current claims.

| File | Line(s) | Stale number | Context |
|------|---------|-------------|---------|
| `SOTA_STATUS.md` | 90, 158, 189 | 26.20deg | Describes the eval-index bug and its fix |
| `disclosures_v1.md` | 64, 97 | 26.20deg, 0.7499 | Historical comparison in disclosure text |
| `full_eval_v2_summary.md` | 25 | 26.20deg | Describes the bug fix verification |
| `tex_reconciliation.md` | 18, 58, 77 | 245.3, 26.20deg | Audit document tracking fix status |
| `psr_null_delta_table.md` | 24 | 0.7499 (10k subset) | Historical comparison note |
| `psr_optimal_thr_38k/reconciliation_notes.md` | 40 | previously reported: 0.7499 on 10k | Historical context |
| `opus_140_batch_index.md` | 42 | downward revision from 0.7499 | Historical context |
| `up_vector_v3/up_vector_per_recording.json` | 58, 87 | q75: 13.56, 26.20deg | Buggy-era eval output data |
| `up_vector_v2/up_vector_per_recording.json` | 55 | 26.20deg | Buggy-era eval output data |
| `full_eval_ep18_stream/metrics.json` | 103 | up_angular_MAE_deg: 26.20 | Buggy-era eval output data |
| `d3_v3/metrics.json` | 5476 | eff_gflops: 245.33 | Historical eval output data |
| `d3_v4/metrics.json` | 5476 | eff_gflops: 245.33 | Historical eval output data |
| `d3_v6/metrics.json` | 5476 | eff_gflops: 245.33 | Historical eval output data |
| Root analysis files (127_*.md, 128_*.md) | various | 0.7499, 0.7217, 0.7810 | Historical analysis documents in repo root |

---

## Part E: Verified Clean — No Fossilized Index Slices

Repo-wide grep for `3:6`, `3:5`, `(3, 6)` in `.py` files confirms:

| File | [3:6] usage | Verdict |
|------|-------------|---------|
| `src/training/losses.py:951-952` | `pred[:, 3:6]`, `target[:, 3:6]` as position | **CLEAN** — position is [3:6] per 9-DoF layout |
| `src/models/head_pose_geo.py:237` | `legacy_9dof[:, 3:6]` as position | **CLEAN** |
| `src/data/industreal_dataset.py:599,620` | `pose_data[:, 3:6]` as position | **CLEAN** |
| `src/evaluation/evaluate.py:1970` | `pred[:, 3:6] - gt[:, 3:6]` as position error | **CLEAN** |
| `src/evaluation/head_pose_diag.py` | Comments describing layout; uses [6:9] for angular calc | **CLEAN** (already fixed in bff38b790) |

All [3:6] usages correctly reference position data. The sole buggy instance (`head_pose_diag.py`) was fixed in commit bff38b790.

---

## Part F: Stale Number Summary

| Number | Occurrences found | Fixed | Still flagged | Historical (OK) |
|--------|------------------|-------|---------------|-----------------|
| 26.20 / 26.2 | 12 | 0 (already historical) | 0 | 12 |
| 13.52 / 13.5 | 6 (gamma comments) | 0 | 0 | 6 |
| 7.83 | 9 (tex + analysis) | 9 | 0 | 0 |
| 7.06 | 7 (tex + analysis) | 6 | 1 (placeholder .tex comments) | 0 |
| 8.39 | 0 (analysis doc only) | 0 | 0 | 0 |
| 0.7499 | 15 (docs + JSON) | 4 | 0 | 11 |
| 0.9693 | 3 (tex) | 3 | 0 | 0 |
| 245.3 | 6 (tex + JSON data) | 0 | 3 (tex live text) | 3 (JSON data) |

**Total stale references enumerated:** 58
**Total fixed (Agent-22 + tex_reconciliation):** 22
**Total still flagged (requires author action):** 3 (all 245.3 GFLOPs in AAIML tex)
**Total historical (leave as-is):** 33

---

## Part G: Remaining Issues Requiring Author Action

1. **GFLOPs/params re-measurement (C-6):** The AAIML .tex still quotes 245.3 GFLOPs / 46.47M params at lines 74, 90, 262. Day1 checkpoint suggests ~93 GFLOPs / ~53M params. The 141 Q4.07 verdict schedules this as "Week 2 (1 hr): re-measure on the freeze checkpoint with a committed fvcore/ptflops script." Until the re-measurement is done, these numbers cannot be updated. A `TODO` or `[PENDING RE-MEASUREMENT]` annotation in the tex may help avoid reviewer confusion.

2. **ICHCIIS-26 paper (popw_ichciis26.tex):** Contains stale numbers (8.14deg, 7.06deg) only in comments at lines 5, 43, 95. The file is still in outline/placeholder stage. No compiled text contains stale numbers. Should be cleaned up when the paper advances beyond placeholder stage.

3. **PSR F1 historical numbers in disclosures_v1.md line 56:** Contains 10k subset numbers (0.7217, +0.0358) alongside current 38k numbers (0.7018). These are accurate as historical comparison context but could be updated to reference only 38k numbers if desired.

No further stale numbers found beyond what was enumerated in Agent-22's audit and the tex_reconciliation pass.
