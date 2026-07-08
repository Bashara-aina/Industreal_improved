# PSR Event-F1 Primary Metric Wire-Up (P5 Preflight Fix)

## Before

The PSR headline metric in `full_eval_inprocess.py` was **per-frame macro F1** at a fixed sigmoid threshold of 0.10 (`psr_macro_f1`). This measures frame-level binary classification accuracy for each PSR component.

**Why this was wrong:** Per-frame F1 is not comparable to STORM or B3 evaluation protocols. It conflates transition-timing accuracy with state-holding accuracy. A model that correctly predicts "base plate is placed" but misses the transition moment by hundreds of frames can still score well on per-frame F1. The SOTA literature evaluates on transition events, not per-frame states.

## After

The primary PSR metric is now `event_f1@±3`, computed alongside POS (ordered-pair fraction) and tau (mean signed delay) from the B3/STORM event-matching protocol.

Three new keys in the results dictionary:

| Key | What it measures | Reference |
|-----|-----------------|-----------|
| `psr_event_f1_at_3` | Transition-event F1 with greedy matching within ±3 frames per component | 175 section 7.2 |
| `psr_pos` | Ordered-pair fraction: directional sign agreement of frame-to-frame differences across 11 components. **Null-model sensitive:** all-zeros predicts 0.9995. Always report alongside null-model baseline. | 174 section 3.3 |
| `psr_tau_seconds` | Mean signed delay between matched predicted and GT events, in seconds (frames/30). Positive = lag, negative = anticipation. | 174 section 3.3 |

Per-frame F1 is retained as a secondary metric under `psr_macro_f1` (legacy, appendix-only).

## What event_f1 measures

`event_f1` evaluates how well the model detects **procedure-step transitions** (0-to-1 state changes per assembly component) as discrete events. The algorithm:

1. For each of the 11 components, extract all predicted 0-to-1 transition frames and all ground-truth transition frames.
2. Match predicted transitions greedily to the nearest unmatched GT transition within tolerance (±3 frames, ~100ms at 30fps).
3. Matched = true positive, unmatched predicted = false positive, unmatched GT = false negative.
4. F1 = 2 * precision * recall / (precision + recall).

Tolerance of ±3 frames compensates for decoder hysteresis delay and frame-level GT annotation ambiguity. This matches the B3 (WACV 2024) and STORM (CVIU 2025) protocol.

## Where to find things

- Wire-up logic: `src/evaluation/full_eval_inprocess.py` (transition metric block after per-frame F1 section)
- `event_f1` implementation: `src/evaluation/decoder_oracle_bound.py:252-277` (canonical)
- Helpers `_compute_tau`, `_compute_pos`: `src/evaluation/full_eval_inprocess.py` (module-level, before streaming_eval)
- Standalone reference script: `scripts/eval_psr_transition_f1.py`
- Optimal per-component thresholds: `src/runs/rf_stages/checkpoints/psr_optimal_thr_38k/optimal_thresholds.json`
- Tests: `tests/test_psr_event_f1_wired.py`

## File changes (P5 preflight)

- Modified: `src/evaluation/full_eval_inprocess.py` -- added transition metric computation block, updated print summary
- Created: `tests/test_psr_event_f1_wired.py` -- 21 tests covering event_f1, tau, POS, and wire-up integration
- This file: `EVENT_F1_WIRE_README.md`
