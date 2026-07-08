# PSR Transition Event-F1: Metric Contract

## What event_f1@±3 means

`event_f1` measures how well a model detects **procedure-step transitions** (0-to-1 state changes in 11 assembly components) as discrete events, rather than as per-frame states. A transition event is a frame where a component's binary state flips from 0 (not yet placed) to 1 (placed).

The greedy matching algorithm works per component:

1. For each of the 11 assembly components, find all predicted 0-to-1 transition frames and all ground-truth transition frames.
2. For each predicted transition (in order), match it greedily to the nearest unmatched ground-truth transition within ±tolerance frames.
3. Matched predictions are True Positives. Unmatched predictions are False Positives. Unmatched ground-truth events are False Negatives.
4. Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2 * P * R / (P + R).

Tolerance of ±3 frames (the default, matching the B3/STORM protocol) gives the model a ~100ms window at 30fps to fire a transition near the true event. This compensates for the decoder's hysteresis delay and for frame-level ambiguity in ground-truth annotation.

The implementation lives in `decoder_oracle_bound.py:252-277` and `psr_transition_f1.py:30-53`. Both implement identical logic. The B3/STORM protocol is defined under 174 §3.3 and 175 §7.2.

### Why event_f1 is the primary metric, not per-frame F1

The PSR head produces **11 per-component binary states** at each frame. You can evaluate these as per-frame classification (each frame gets a macro-averaged precision/recall/F1). But per-frame F1 conflates two very different errors:

- A frame where the model correctly says "base plate is placed" but misses the precise moment it happened scores well on per-frame F1 but tells you nothing about transition timing.
- A frame where the model briefly and incorrectly thinks a component was placed (noise) hurts per-frame F1 but is a minor transition error.

The SOTA literature (STORM, B3) evaluates on transition events, not per-frame states. Per-frame F1 is reported as a secondary metric (appendix only, never as the PSR headline).

## How POS differs from event_f1 (and why POS is null-model-sensitive)

**POS** (Ordered-Pair Fraction) measures directional sign agreement between predicted and ground-truth frame-to-frame differences. Concretely:

```
POS = mean( sign(pred[t+1] - pred[t]) == sign(gt[t+1] - gt[t]) )
```

This captures whether the model correctly predicts the *direction* of change at each frame (no change, positive, negative). But because the MonotonicDecoder prevents negative transitions (components can't become "un-placed"), the only non-zero values in practice are 0-to-1 events, which are extremely rare -- typically 3-10 events per recording of 1000-4000 frames.

**The null-model sensitivity problem:** An all-zeros prediction (no transitions anywhere) scores POS = 0.9995, because 99.95% of frame-pairs correctly have no change in either direction. This means any sparse-positive detection system will trivially score near-perfect POS. A model predicting zeros for everything cannot be distinguished from a perfect model by POS alone.

POS is therefore:
- **Appendix-only** with the null-model baseline disclosed alongside (mandatory per 174 §3.3).
- Never used as a "we beat STORM" headline -- STORM's POS = 0.812 is a real number because STORM has fewer false positives and detects more true transitions than the null model. Our POS ~0.998 is an artifact of the null-signal regime.

**event_f1 is robust** to this null-model issue: a model predicting all zeros gets event_f1 = 0.0 (undetected events -> 0 recall), while a model with correct transition timing gets positive F1.

## How delay τ is computed

τ (tau) is the average frame offset between a predicted transition event and its matched ground-truth event. The matching uses the same greedy algorithm as event_f1:

- For each matched pair (pred_frame, gt_frame) within ±tolerance, the signed delay is `pred_frame - gt_frame`.
- Positive τ means the prediction lags behind the ground truth (typical for hysteresis-based decoders).
- Negative τ means the prediction anticipates the ground truth (unusual in practice).

Two variants are reported:

| Variant | Formula | Interpretation |
|---------|---------|----------------|
| τ (signed) | mean(pred_frame - gt_frame) | Shows systematic bias (lag vs. anticipation) |
| τ_abs (absolute) | mean(abs(pred_frame - gt_frame)) | Shows average timing error regardless of direction |

If no events are matched (event_f1 = 0), τ is undefined (NaN).

### Converting frames to seconds

IndustReal is recorded at 30 fps. The temporal stride in the evaluation pipeline is 1 (every frame sampled). So:

```
τ_seconds = τ_frames / 30.0
```

## How to interpret τ < 15.5s vs τ > 22s

The STORM-PSR paper (CVIU 2025, Table 1) reports τ = **15.5 seconds** on IndustReal test split. The prior WACV-2024 B3 baseline reports τ = **22.4 seconds**. These are the published SOTA anchors:

| System | event_f1 | POS | τ (seconds) |
|--------|----------|-----|-------------|
| STORM-PSR (CVIU 2025) | 0.901 | 0.812 | **15.5** |
| B3 (WACV 2024) | 0.883 | 0.797 | **22.4** |
| B2 (WACV 2024) | 0.860 | 0.731 | **22.3** |

STORM's headline contribution is **delay reduction** -- it cuts detection latency from 22s to 15.5s by using a spatio-temporal dual-stream architecture with procedural knowledge.

**Interpretation guide:**
- τ < 15.5s would beat STORM's delay (ambition target).
- τ between 15.5s and 22.3s would be between STORM and B3 -- competitive territory.
- τ > 22.3s would be below the prior baseline -- the decoder is too slow or the predictions are too late.

In frames at 30fps:
- 15.5s ≈ 465 frames
- 22.4s ≈ 672 frames

These are large numbers because the decoder's hysteresis and procedure-order constraints can delay detection by hundreds of frames when the model is uncertain about early components.

## Current state (example output from step 2)

The evaluation was run on two checkpoints using `scripts/eval_psr_transition_f1.py`:

**Checkpoint 1:** `src/runs/rf_stages/checkpoints/best.pth` (epoch 18, GELU head)
```
event_f1@±3:  0.0000
τ (mean):     NaN (no matched events)
POS:          0.9996
Frames:       5000 across 3 recordings
Predicted transitions: 0-9 per recording (none matched GT within ±3)
```

**Checkpoint 2:** `src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/best.pth` (epoch 62, GELU head)
```
event_f1@±3:  0.0000
τ (mean):     NaN (no matched events)
POS:          0.9997
Frames:       5000 across 3 recordings
Predicted transitions: 0 per recording
```

Both checkpoints produce event_f1=0.0 because they were trained with the GELU-saturated PSR head (the GELU activation had mean pre-activation ≈ -130, causing 99.7%+ saturation and near-zero gradient through the PSR head). The LeakyReLU repair at `model.py:1604-1611` has been applied to the architecture but **no training run has been completed with the repair active**.

After retraining with the LeakyReLU head, event_f1 should move toward the decoder oracle bound of 0.595 (the theoretical max of the MonotonicDecoder path when fed perfect GT transition logits).

## Decoder oracle bound context

The `decoder_oracle_bound.py` analysis establishes an upper bound for the MonotonicDecoder path:

| Variant | Macro F1 | Notes |
|---------|----------|-------|
| Oracle (sustained) | 0.595 | Decoder fed ideal transition signals |
| Oracle (relaxed, no procedure-order) | 0.875 | Decoder bottleneck = procedure-order constraint |
| Actual decoder (current) | 0.679 | From `full_eval_inprocess_v2.log` |

The gap between 0.595 (oracle) and 0.679 (actual) is the artifact: the decoder oracle bound is computed differently than the actual decoder path. The important number is that the procedure-order constraint (sequential assembly: comp0→comp1→...→comp10) is the primary bottleneck, reducing the relaxed oracle from 0.875 to 0.595.

## References

- `src/evaluation/decoder_oracle_bound.py:252-277` -- event_f1 implementation
- `src/evaluation/psr_transition_f1.py:30-53` -- transition event_f1 (identical logic)
- `scripts/eval_psr_transition_f1.py` -- standalone eval script
- `src/runs/rf_stages/checkpoints/psr_event_f1_run/metrics.json` -- latest eval output
- `src/runs/rf_stages/checkpoints/decoder_oracle_bound/oracle_f1.json` -- decoder oracle bound
- `code/industreal_improved/analyses/consult_2026_06_10/AAIML/175_ULTIMATE_GUIDE_TIER_F.md` §7.2 -- metric spec
- `code/industreal_improved/analyses/consult_2026_06_10/AAIML/174_SOTA_PROTOCOLS_AND_EVAL_DEFINITIONS.md` §3.3 -- protocol definition
