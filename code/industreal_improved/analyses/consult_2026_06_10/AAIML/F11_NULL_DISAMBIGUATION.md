# F-11: Null Baseline Disambiguation (File-157 Audit)

**Date:** 2026-07-07
**Source:** File-157 F-11 finding — two different "null" baselines conflated across documents 150-156.

## The Two Null Baselines

### 1. Persistence Null (copy-prev)
- **Symbol:** `null_copy_prev`
- **Formula:** Predict the previous frame's state (copy-forward)
- **Value:** F1 = 0.9997 on POS metric
- **Interpretation:** Measures how well the model performs relative to a trivial persistence heuristic. Since most frames have no state change, copying the previous prediction is nearly always correct. The model (0.7018) is 29.7% worse than this baseline.
- **Context:** PSR evaluation metric is structurally inflated by temporal auto-correlation.

### 2. Prevalence Null (always-positive)
- **Symbol:** prevalence null (used in `psr_null_delta_table.md`)
- **Formula:** F1_null = 2p/(1+p) where p = component prevalence
- **Value:** Per-component; e.g., comp4 F1_null = 0.249 (p=0.142), comp10 F1_null = 0.310 (p=0.183)
- **Interpretation:** Measures improvement over an always-positive predictor that knows prevalence. Positive delta = genuine learned signal.
- **Context:** Documented deltas: comp4 +0.097, comp10 +0.093, comp9 -0.000, comp8 +0.053.

## Usage Rules for Tables and Text

| Context | Use "persistence null (copy-prev)" | Use "prevalence null (always-positive)" |
|---|---|---|
| F1 comparison to 0.9997 | Yes | No |
| Per-component null-delta table | No | Yes |
| "Model 29.7% worse than..." | Yes | No |
| "Null-delta +0.097/+0.093" | No | Yes |

Always name which null is being used. Never say "null baseline" without qualification.

## Files Modified

| File | Changes |
|---|---|
| 150_MASTER_SYNTHESIS.md | 6 references disambiguated |
| 151_PER_HEAD_DEEP_ANALYSIS.md | 1 reference disambiguated |
| 152_IMPLEMENTATION_BUG_CATALOG.md | 1 reference disambiguated |
| 154_SOTA_COMPARISON.md | 3 references disambiguated |
| 155_FINAL_PAPER_NARRATIVE.md | 1 reference disambiguated (with both nulls distinguished) |
| 156_100_DEEP_QUESTIONS.md | 2 references disambiguated |
