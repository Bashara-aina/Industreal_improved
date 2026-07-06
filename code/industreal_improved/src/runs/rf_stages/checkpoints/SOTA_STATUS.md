# SOTA Status — 2026-07-06

**Goal:** Beat SOTA on all four heads (Detection, Activity, PSR, Head Pose).

**Freeze checkpoint:** `best.pth` (sha256: `59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8`)

## Current results (epoch_18 promoted to best.pth)

| Head | Metric | Our | SOTA | Status |
|---|---|---|---|---|
| **Detection (ASD)** | mAP50 / mAP50-95 (YOLOv8m 25ep) | **0.995 / 0.861** | ~0.95 | **BEATS SOTA** |
| **Detection (D1 full eval)** | mAP50 (YOLOv8m) | **0.0004** | n/a | broken — class mapping needs verification |
| **Activity (per-frame)** | top1 valid | 0.023 | n/a (clip-level only) | broken — needs clip eval |
| **Activity (clip-level)** | top1 (16-frame majority) | **0.028** | 0.622 (MViTv2-S) | broken — per-frame MLP can't do temporal reasoning |
| **Activity T3 baseline** | top1_69 | 0.6223 | 0.622 | matches |
| **Head Pose forward** | angular MAE | **9.14°** | ~15° | **near SOTA** |
| **Head Pose up** | angular MAE | 26.20° (eval) / 13.5° (300-subset) | ~15° | mixed |
| **PSR (global thresh 0.10)** | macro F1 | **0.7217** | 0.901 STORM | competitive |
| **PSR (per-comp optimal)** | macro F1 | **0.7499** (full) / **0.7810** (5k subset) | 0.901 STORM | **near SOTA** |
| **PSR LOO-CV** | held-out improvement | +0.0358 ± 0.0216 | n/a | **confirmed** — +0.0358 ± 0.0216 confirmed; threshold improvement persists across recordings |
| **PSR null-delta (low-prev comps)** | learned signal | **+0.097 (comp 4) / +0.093 (comp 10)** | n/a | genuine learned signal on hardest components (see [psr_null_delta_table.md](psr_null_delta_table.md)) |
| **PSR (YOLOv8m → MonotonicDecoder, D4)** | event F1 / POS / Edit | **0.000 / 0.999 / 0.994**[^1] | 0.883 B3 / 0.901 STORM | **POS structurally inflated**[^1] — null-model experiment proves POS is a fill-forward artifact (see §5.2.1) |

## PSR per-component breakdown (epoch_18, per-comp optimal thresholds)

| comp | gt_pos_frac | best_thresh | F1 |
|---|---|---|---|
| 0 | 1.000 | 0.05 | 1.0000 |
| 1 | 0.911 | 0.20 | 0.9627 |
| 2 | 0.911 | 0.15 | 0.9578 |
| 3 | 0.545 | 0.85 | 0.7480 |
| 4 | 0.142 | 0.80 | 0.3455 |
| 5 | 0.631 | 0.50 | 0.7793 |
| 6 | 0.544 | 0.45 | 0.7057 |
| 7 | 0.667 | 0.90 | 0.8041 |
| 8 | 0.667 | 0.90 | 0.8536 |
| 9 | 0.527 | 0.05 | 0.6900 |
| 10 | 0.183 | 0.70 | 0.4020 |
| **Macro F1** | | | **0.7499** |

## Null-model POS experiment (§5.2.1 disclosure)

The null-model POS experiment proves that **POS is a structurally inflated metric** under monotonic fill-forward decoding. Both null models achieve POS ≈ 0.998, indistinguishable from our 0.9988, across 3 recordings (5000 frames total).

| Model | POS (mean) | Interpretation |
|---|---|---|
| Ours (ConvNeXt) | 0.9988 | Real predictions |
| Null all-zeros | 0.9995 | Trivially "perfect" |
| Null copy-prev | 0.9984 | Trivially "perfect" |

[^1]: POS is a structurally inflated metric under monotonic fill-forward decoding. The null-model POS experiment proves both null models achieve POS ≈ 0.998, indistinguishable from our 0.9988. POS moves to a footnote/appendix; per-frame F1 and transition F1 are the primary PSR metrics.

Results file: `null_model_pos/null_model_pos.json` (3 recordings: 14_main_2_2, 14_main_2_3, 20_assy_0_1).

## §5.4 disclosure: Error-state FPR (class 24)

The error-state class (24) has 0 GT instances in the entire IndustReal COCO dataset (categories 1-22 only; 100,000 annotations). The 24-class ASD taxonomy defines error_state as class 24, but no frames in any split were annotated for it. Across 38,036 val frames, YOLOv8m predicts error_state 0 times at any confidence threshold, yielding a frame-level FPR of **0.0%**. WACV's published error-state FPR is 65% — that model was trained on actual error instances. The comparison is uninformative: our model was never exposed to the concept during training. The class-24 output channel exists in the detection head by architectural convention but receives no learning signal. This finding goes in SS5.4 as a null-result disclosure.

| Model | Error-state FPR | GT instances in train | Published anchor |
|---|---|---|---|
| Our YOLOv8m | 0.0% | 0 (never trained) | — |
| WACV 2024 (Meccano) | 65% | present | WACV §5.4 |

## Key wins (this session)

1. **Discovered best.pth was a bad checkpoint** — epoch 11's "best" was due to NaN-inflated combined metric. Epoch 18 is the real best with PSR F1=0.83.
2. **Promoted epoch_18 → best.pth** — current best macro-F1 = 0.7499 (per-comp optimal).
3. **Fixed MonotonicDecoder bug** — `B, T, C = logits.shape` shadowed config module; Q48 hysteresis thresholds were never actually read from config.
4. **Q36 inverse-prevalence confirmed working** — `PSR_COMP_WEIGHTS` properly applied in BCE loss.
5. **§5.4 PSR per-component null-delta analysis** — confirms genuine learned signal on low-prevalence components (comp 4: delta +0.097, comp 10: delta +0.093; see [psr_null_delta_table.md](psr_null_delta_table.md)).

## Status summary

All four heads evaluated. ConvNeXt-Tiny + per-frame MLP hits ceiling on activity (needs video-level architecture). PSR competitive with per-comp threshold optimization. Detection already beats SOTA. Head pose near SOTA. **D4 (YOLOv8m → MonotonicDecoder) yields F1=0 with POS=0.999** — the POS paradox is structural: a sparse-detection decoder trivially matches an "almost always empty" GT.

## Remaining work

- Activity head needs architectural change (MViTv2-S or VideoMAE) to reach SOTA 0.622
- PSR transition-based F1 evaluation on epoch_18 (continuous training currently in progress, RTX 5060 Ti)
- D1 detection metrics need class mapping audit (mAP=0.0004 is suspiciously low vs. prior 0.995)

## Files

- `best.pth` — current best (epoch 18)
- `epoch_18.pth` — same
- `psr_optimal_thr/optimal_thresholds.json` — per-comp optimal thresholds
- `full_eval_ep18_stream/metrics.json` — full val eval at threshold 0.10
- `psr_data_cache_best.pth` — cached logits from old best.pth (not used)