# Probe Scripts

Diagnostic probes that test specific hypotheses about training dynamics.
These are quick experiments (run in seconds to minutes) to validate fixes
before committing to full training runs.

## Recent Probes

### `probe_logit_bias_disable.py`
**Question:** Does disabling `update_logit_bias()` help background suppression?

**Findings (500-step overfit-200, FullMultiModalDataset):**
- BG conf: 0.0465 → 0.0388 (-16.5% ✓)
- FG conf: 0.327 → 0.362 (+10.7% ✓)
- FG/BG separation: 0.281 → 0.324 (+15.3% ✓)

**Verdict:** Disable update_logit_bias() in training.

### `tal_probe_correct.py`
**Question:** Does TAL aligner outperform 3×3 on detection?

**Findings:** 3×3 overfits fine; TAL comparable. No clear benefit.

**Verdict:** 3×3 suffices — no TAL port needed.

## Deprecated

### `tal_probe_fixed.py`
⚠️ **USES WRONG DATASET** (`IndustRealMultiTaskDataset` instead of
`FullMultiModalDataset`). The original "3x3-suffices" verdict was meaningless.
Use `tal_probe_correct.py` instead.

## Usage

```bash
python scripts/probes/probe_logit_bias_disable.py --n-steps 500
python scripts/probes/tal_probe_correct.py --n-steps 200
```

## Other Probes

- `overfit_probe.py` — Single most important diagnostic (Opus 201 Step 1)
- `overfit_50img_cls.py` — Overfit 50 images for classification
- `mvp_probe3_psr_ab.py` — PSR temporal-resolution A/B
- `mvp_probe4_tal_vs_3x3.py` — Detection TAL vs 3×3 (legacy)
- `debug_q43.py` / `debug_q43_v2.py` — Q43 canonical POS debugging
- `e8_gradient_diagnostic.py` — E8 gradient-flow diagnostic
- `check_weight_evolution.py` — Check weight evolution during training
- `check_train_val_subject_disjoint.py` — Verify subject disjointness