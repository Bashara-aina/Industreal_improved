# Opus 140/141 Final Verification — 2026-07-07

**Verification specialist:** Agent 44 (OOM-safe, read-only)
**Session span:** 50+ commits over Opus 140/141
**Current HEAD:** 693b119b54b5672be73c9917fa9cbeaf86f1094d

---

## SOTA_STATUS.md consistency

### 1. Head pose forward: 9.14 degrees
- SOTA_STATUS.md: 9.14 degrees (95% CI 7.74-10.87)
- bootstrap_ci.json: 9.1355 degrees (95% CI 7.7412-10.8704)
- **PASS** — values match within rounding

### 2. Head pose up: 7.78 degrees
- SOTA_STATUS.md: 7.78 degrees (95% CI 6.89-8.81)
- bootstrap_ci.json: 7.7842 degrees (95% CI 6.8918-8.8137)
- **PASS** — values match within rounding

### 3. PSR F1: 0.7018
- SOTA_STATUS.md: 0.7018 (95% CI 0.6436-0.7321)
- bootstrap_ci.json: 0.701763 (95% CI 0.643573-0.732119)
- psr_optimal_thr_38k/optimal_thresholds.json: optimal_macro_f1 = 0.701763
- **PASS** — values match within rounding

### 4. Detection mAP_pc: 0.573
- SOTA_STATUS.md: 0.573 present-class mAP
- detection_zero_gt_count.json: present_class_mAP_det_mAP50_pc = "~0.573"
- 6 zero-GT classes confirmed (not 9 as earlier subsample suggested)
- **PASS** — values match

### 5. Prohibited phrases absent from SOTA_STATUS.md
- "BEATS SOTA": not found
- "BACKBONE HAS SIGNAL": not found
- "near SOTA": not found
- stale_numbers_audit.md records these as historical issues that were fixed
- **PASS** — no prohibited phrases in SOTA_STATUS.md

### 6. Confidence intervals present
- Head pose forward CI: 7.74-10.87 degrees — present
- Head pose up CI: 6.89-8.81 degrees — present
- PSR F1 CI: 0.6436-0.7321 — present
- PSR LOO-CV: +-0.0158 — present
- Activity linear probe: +-0.0046 — present
- **PASS** — all key CIs present

---

## disclosures_v1.md verification

### 7. Eight numbered disclosures present
- #1: Backbone-swap transfer (D4)
- #2: POS is structurally inflated
- #3: Per-frame action classification is a floor baseline
- #4: Multi-task detection
- #5: PSR per-component gradient starvation
- #6: PSR thresholds are validation-selected
- #7: Three-and-a-half-month evaluation-index bug
- #8: Position is unreported
- **PASS** — all 8 numbered disclosures present

### 8. Current numbers (not stale)
- All numbers reflect epoch_18 best.pth with Opus 140 batch updates
- **PASS** — current numbers used

### 9. D4+D1R F1 = 0.6364 in disclosure #1
- disclosures_v1.md line 12: "F1 reaches 0.6364"
- d4_d1r/retune/verdict.json: f1_at_t_best_global = 0.6363636
- SOTA_STATUS.md: "0.636"
- **PASS** — matches confirming source

---

## Key file existence

### 10. disclosures_v1.md
- `src/runs/rf_stages/checkpoints/disclosures_v1.md`
- **PASS** — exists, well-formed

### 11. up_vector_v3/up_vector_per_recording.json
- **PASS** — exists, 9 recordings, median-of-medians = 5.82 degrees

### 12. full_eval_ep18_v2/metrics.json
- **PASS** — exists, 38,036 frames per-component PSR data

### 13. d4_retuned/sweep_results.json
- **PASS** — exists, best global F1 = 0.3466 (threshold retuned)

### 14. null_model_pos/null_model_pos.json
- **PASS** — exists, 3 recordings, 5000 frames

### 15. psr_null_delta_table.md
- **PASS** — exists, all 11 components with deltas

### 16. bootstrap_ci.json
- **PASS** — exists, 1000 bootstrap replicates, seed 42

### 17. pose_kalman_eval/pose_kalman_results.json
- **PASS** — exists, 16 recordings, 38,036 frames

### 18. decoder_oracle_bound/oracle_f1.json
- **PASS** — exists, oracle macro F1 = 0.5947 (procedure-ordered), 0.875 (relaxed)

---

## Cross-validation of data values

### 19. D4 retuned F1 = 0.347
- d4_retuned/verdict.json: 0.3466
- SOTA_STATUS.md: 0.347
- **PASS**

### 20. D4+D1R decisive F1 = 0.636
- d4_d1r/retune/verdict.json: 0.6364
- SOTA_STATUS.md: 0.636
- disclosures_v1.md: 0.6364
- **PASS**

### 21. Null model POS values
- null_model_pos.json: all-zeros = 0.9995, copy-prev = 0.9984, ours = 0.9988
- SOTA_STATUS.md: all-zeros = 0.9995, copy-prev = 0.9984, ours = 0.9988
- **PASS**

### 22. PSR global 0.10 threshold F1 = 0.6788
- psr_optimal_thr_38k/optimal_thresholds.json: global_0.10_macro_f1 = 0.678753
- SOTA_STATUS.md: 0.6788
- **PASS**

### 23. Kalman smoothing results
- Forward single-frame: 9.14 degrees (9.1355 data) — matches bootstrap
- Forward smoothed: 9.00 degrees (8.9957 data)
- Up single-frame: 7.78 degrees (7.7842 data) — matches bootstrap
- Up smoothed: 7.58 degrees (7.5780 data)
- Improvement: forward +0.14 degrees (+1.5%), up +0.21 degrees (+2.7%)
- SOTA_STATUS.md lines 153-155: all match
- **PASS**

### 24. Activity linear probe verdict fixed
- activity_linear_probe.json: "NO DETECTABLE FRAME-LEVEL SIGNAL"
- Old value was "BACKBONE HAS SIGNAL" — corrected per stale_numbers_audit.md
- **PASS** — verdict reflects CI-overlap finding

### 25. PROGRESS_2026-07-06.md marked superseded
- First line: "SUPERSEDED by 2026-07-07 38k eval. See SOTA_STATUS.md..."
- stale_numbers_audit.md A5 required this — completed
- **PASS**

### 26. detection_zero_gt_count: 6 classes
- File confirms 6 zero-GT non-background channels (channels 1,2,3,14,15,23)
- Not 9 as earlier subsample suggested
- SOTA_STATUS.md line 13 and disclosure #4 reflect corrected count
- **PASS**

### 27. PSR per-component null-deltas
- psr_null_delta_table.md: comp4 = +0.097, comp10 = +0.093, comp9 = -0.000
- SOTA_STATUS.md line 25: matches
- Disclosure #5: matches
- **PASS**

---

## Summary

| Category | Items | Pass | Fail |
|----------|-------|------|------|
| SOTA_STATUS.md consistency | 6 | 6 | 0 |
| disclosures_v1.md verification | 3 | 3 | 0 |
| Key file existence | 9 | 9 | 0 |
| Cross-validation | 9 | 9 | 0 |
| **Total** | **27** | **27** | **0** |

All 27 verification items PASS. All Opus 140/141 fixes are present, consistent, and cross-validated against their data sources. No discrepancies found.
