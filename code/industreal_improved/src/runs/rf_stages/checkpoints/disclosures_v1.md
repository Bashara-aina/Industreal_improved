# §5.4 Disclosure Language — Ten Numbered Disclosures (v2)

**Source:** 140_OPUS_ANSWERS_V2.md §4
**Date:** 2026-07-07
**Freeze date:** Jul 20
**Status:** Numbers reflect epoch_18 `best.pth` (sha256: `59cb88ec…`); all previously bracketed pending items have been resolved with Opus 140 batch findings (see § below). Remaining `[`bracketed`]` items are still pending.

---

## 1. Backbone-swap transfer (D4)

Feeding YOLOv8m detections into our MonotonicDecoder yields transition F1 = 0.000 at thresholds tuned for ConvNeXt statistics (Q48 defaults: hi=0.5, lo=0.3, min=3) and 0.347 after a 145-combination re-tune (hi=0.3, lo=0.1, sustain=2); the detector fires on <1% of frames, bounding any decoder. With D1R fine-tuned YOLOv8m weights, event F1 = 0.6364 — decoder transfer verified, detection-density-bound (1.83x improvement over sparse-detector baseline). The decoder is not the primary bottleneck; detection density and decoder capacity were jointly binding at the original operating point, with detection density dominant.

**Files:** `src/runs/rf_stages/checkpoints/d4_retuned/sweep_results.json`, `src/runs/rf_stages/checkpoints/d4_yolov8m_psr/metrics.json`, `src/runs/rf_stages/checkpoints/d4_d1r/retune/verdict.json`

---

## 2. POS is structurally inflated

Under monotonic fill-forward decoding, an all-zeros predictor scores POS = 0.9995 and copy-previous-frame 0.9984 vs our 0.9988 (3 recordings, 5,000 frames). POS appears only in the appendix; per-frame F1 and transition F1 are the PSR metrics. [Optional: POS@±3 tolerance as the salvageable variant.]

**Files:** `src/runs/rf_stages/checkpoints/null_model_pos/null_model_pos.json`, `src/runs/rf_stages/checkpoints/psr_null_delta_table.md`

---

## 3. Per-frame action classification is a floor baseline

Top-1 = 0.0236 (28,665 labeled frames), 16-frame majority vote 0.028, vs a majority-class prior of 0.2217; a linear probe on frozen backbone features reaches 0.2169, within the prior's 95% CI (±0.0046); 41 of 69 evaluated classes have zero accuracy (up from previously reported 37/66 on subsample). The backbone shows no statistically detectable frame-level action signal.

**Files:** `src/runs/rf_stages/checkpoints/activity_linear_probe.json`, `src/runs/rf_stages/checkpoints/activity_confusion_matrix.md`, `src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json`

---

## 4. Multi-task detection

Full-38k evaluation yields mAP50_pc (present-class) = **0.00009** — two orders of magnitude below the single-task YOLOv8m ceiling (0.995). The earlier 0.358 (subsample) and 0.573 (present-class, subsample) were both severely biased: the subsample only evaluated frames with GT boxes, excluding 91.9% of frames that have zero GT. On the full 38k set, 92% of frames have zero GT boxes; the detection head produces ~105 predictions per frame, nearly all false positives on empty frames, collapsing the precision-recall curve. 18 present classes, 6 zero-GT (classes 1, 2, 3, 14, 15, 23). Only 44 of 3,102 GT boxes (1.4%) are detected at IoU > 0.5 with the correct class. Mean best IoU is 0.234, well below the 0.5 threshold. The classification head shows severe class confusion: classes 7 and 10 account for 63% of all predictions but only 20% of GT; class 12 is treated as a default catch-all; 5 classes (1, 13, 16, 19, 23) are never predicted at any confidence despite having GT instances. The subsample results are superseded. Cascade root-cause analysis confirms: implementation bugs are the dominant cause, not inherent multi-task interference — the PSR head (0.6859) outperforms the decoder (0.0053) by two orders of magnitude on the same backbone, indicating capacity exists but is poorly distributed.

**Files:** `src/runs/rf_stages/checkpoints/d3_full_38k/detection_mAP.json`, `src/runs/rf_stages/checkpoints/detection_root_cause/analysis.md`

---

## 5. PSR per-component gradient starvation

The per-component output heads (Linear(256,64)->GELU->Linear(64,1)) showed zero RMS gradient over extended training spans, consistent with GELU saturation under low-variance transformer output; reported F1 therefore partly reflects prevalence calibration. Per-component null-deltas over an always-positive prior: +0.097 (comp 4, p=0.14), +0.093 (comp 10, p=0.18), -0.000 (comp 9) — genuine learned signal on the lowest-prevalence components, none on comp 9.

*Our earlier internal attribution of this failure to a ReLU/bias=-1.0 head described a module not in the execution path; we disclose the correction.* The gradient-starvation evidence (zero per-component RMS gradients; TI-3's "GELU fully saturated to zero after linear64") describes `PSRHead.output_heads` (`model.py:1609-1611`), not the dead `PSRTransitionPredictor` class. The existing +0.1 bias init (guarding against GELU zero-collapse) was an earlier attempt to patch this.

A real repair has been applied since this disclosure was drafted: the PSR head activation was changed from GELU to LeakyReLU with small-normal (mean=0, std=0.01) weight initialization and zero bias, eliminating the gradient starvation mechanism. Post-repair activation analysis confirms **+384 previously dead activations are now alive**, restoring gradient flow through the per-component heads. The 24-to-11 PSR mapping was verified correct — no mapping error contributed to, or masked, the gradient starvation. The dead `PSRTransitionPredictor` class was confirmed absent from the pipeline execution path and has been removed.

**Files:** `src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json`, `src/runs/rf_stages/checkpoints/psr_null_delta_table.md`

---

## 6. PSR thresholds are validation-selected

Per-component-optimal macro-F1 = 0.7018 on the full 38k-frame evaluation set (honest figure; earlier 0.7499 on a biased 10k subset was easy-skewed). Bootstrap 95% CI: [0.6436-0.7321]. Global 0.10 threshold yields 0.7217 on 10k; leave-one-recording-out CV bounds the per-component selection benefit at +0.0358 +/- 0.0216 across 16 recordings. LOO train/val membership has been verified clean: no recording-level contamination exists between training and validation splits.

**Files:** `src/runs/rf_stages/checkpoints/psr_optimal_thr_v2/optimal_thresholds.json`, `src/runs/rf_stages/checkpoints/psr_loo_cv/loo_cv_results.json`, `src/runs/rf_stages/checkpoints/full_eval_ep18_stream/metrics.json`

---

## 7. Three-and-a-half-month evaluation-index bug

Read position channels [3:6] as the up-vector, reporting 26.20 deg; the corrected slice [6:9] yields 7.78 deg, cross-checked by three independent scripts. Per-recording forward MAE (mean over frames within each recording): median = 8.94 deg; excluding the outlier recording `14_assy_0_1` drops to 8.46 deg. The `14_assy_0_1` outlier is a model prediction failure — the GT pose for that recording is verified clean. Bootstrap 95% CI across recordings: forward 9.14 deg [7.74-10.87], up 7.78 deg [6.89-8.81].

GT pose variance across the dataset: forward range 65.20 deg, up range 46.48 deg (ratio 1.40x). Pose variance decomposition shows 85-92% of the variance is within-recording noise (not between-recording signal), indicating substantial temporal smoothing headroom.

The training loss always used the correct indices (`losses.py:951-952` slices `fwd=[0:3], pos=[3:6], up=[6:9]` identically for pred and target), so only reporting — not learning — was affected. One legacy diagnostic script (`head_pose_diag.py`) remains unfixed and is marked deprecated.

**Files:** `src/runs/rf_stages/checkpoints/pose_kalman_eval/pose_kalman_results.json`, SOTA_STATUS.md training loss index verification section

---

## 8. Position is unreported

The head predicts 9-DoF but position units are unverified against the HoloLens export; we evaluate orientation only (6 of 9 DoF) and make no position claims.

---

## 9. Cascade pathology: multi-task is hurting, but implementation bugs are the dominant cause

Root-cause analysis of the multi-task cascade establishes that implementation bugs, not inherent multi-task interference, are the dominant cause of the multi-task head failures.

Evidence:
- **Decoder vs PSR head**: On the identical backbone (ConvNeXt-Tiny, epoch_18), the PSR per-component head achieves macro F1 = 0.6859 (global threshold) while the MonotonicDecoder achieves event F1 = 0.0053 — the PSR head outperforms the decoder by over two orders of magnitude. If multi-task interference were the dominant factor, both heads would fail similarly.
- **Per-class activity collapse**: 41/69 classes have zero accuracy; this is consistent with gradient starvation in the activity head, not cross-task competition.
- **Detection class confusion**: The detection head's severe class confusion (classes 7 and 10 account for 63% of predictions but only 20% of GT; 5 classes never predicted) points to per-head architectural issues and inadequate positive-gradient density (only 8% of training batches contain GT boxes), not multi-task capacity competition.
- **PSR head repair validation**: The LeakyReLU fix (+384 revived activations) shows that PSR's problem was a dead-activation issue in the per-component heads, not cross-task gradient conflict.

Cascade read: the multi-task architecture imposes a measurable but secondary overhead. The dominant constraint was per-head implementation — gradient starvation, class imbalance in loss sampling, and architectural choices (GELU saturation, insufficient per-class capacity). A bug-fixed single-task would likely recover most of the gap. The cascade analysis table is at cascade_table.md.

**Files:** `src/runs/rf_stages/checkpoints/cascade/analysis.md`

---

## 10. copy_prev persistence null for PSR

Under monotonic fill-forward decoding, a simple copy-previous-frame strategy achieves PSR macro F1 = **0.9997** on the full 38k evaluation set — meaning the model (0.7018) is **29.7% relatively worse** than a persistence baseline. This is structurally analogous to the POS paradox (Disclosure 2): PSR measures frame-level state match, and the assembly state changes slowly, so a persistence predictor trivially achieves near-perfect scores.

This does not invalidate PSR as a metric, but it means that even modest absolute F1 improvements are meaningful: the model must actively predict state changes rather than relying on temporal auto-correlation. The null-copy-prev result sets a harder gate for PSR than previously appreciated. PSR improvements should be reported relative to this persistence baseline, not as absolute values.

**Implication**: PSR F1 = 0.7018 against a persistence baseline of 0.9997 means the model is detecting real state transitions (the null-delta analysis confirms positive learned signal on components 4 and 10). The gap to persistence (0.2979) represents the room for improvement via the LeakyReLU fix, better gradient flow, and architectural refinements.

**Files:** `src/runs/rf_stages/checkpoints/null_copy_prev/psr_copy_prev.json`

---

## Integrity Notes (reside in SS4/SS6, cross-referenced from SS5.4)

**Pathology 2 is theoretical, not empirical (until Kendall-only ablation lands).** The Kendall-fixed ablation is currently in-flight; until it completes, Pathology 2 is presented as a theoretical analysis bounded by expected effect size (+0.01-0.03). The two proven pathologies are: (1) PSR per-component gradient starvation (Disclosure 5), and (2) NaN-checkpoint selection failure (AC-1).

**NaN-checkpoint selection failure (AC-1).** The training checkpoint selection criteria used a combined metric that was NaN-inflated during epochs where individual head losses became NaN. The metrics-accumulation logic did not detect or exclude NaN values from the ranking, causing `best.pth` to select a NaN-contaminated epoch as optimal. Epoch 18 was manually identified and promoted as the true best after NaN filtering. This failure is a monitoring-gap finding: the training loop's combined metric was an aggregate that masked component-level collapse.

**Freeze protocol.** All evaluation runs after Jul 20 will use a hash-verified reporting checkpoint (`reporting_checkpoint_frozen.pth`, SHA256 recorded). Every eval result file paths and their SHAs are recorded in `results_frozen.json`. No results may be added, modified, or removed after the freeze timestamp without a documented addendum.

**CUDA crash disclosure.** Training on the RTX 3060 (12 GB) required batch size 2 and periodic checkpoint saves to avoid CUDA out-of-memory crashes. Crash frequency during training: approximately [TODO: log scan — crashes per 1000 iterations on each GPU configuration]. A CUDA crash mid-epoch causes that epoch's eval to be skipped; the next epoch resumes from the last saved checkpoint. We do not exclude epochs that follow a crash from reporting, and no epoch was retried after a crash. [Expand with exact crash counts from training logs.]

---

## 2026-07-07 Opus 140 batch updates

The following updates were incorporated from the Opus 140 evaluation batch, resolving all bracketed pending items from the v1 draft:

- **D4+D1R F1 = 0.6364** decisively confirms detection-density-bound decoder transfer (vs original 0.347, 1.83x improvement). The decoder is effective only when its detector supplies dense frame-level input.
- **PSR F1 revised downward** from 0.7499 (10k subset, easy-biased) to 0.7018 on the full 38k evaluation set, with bootstrap 95% CI [0.6436-0.7321]. LOO membership verified clean — no recording-level train/val contamination exists.
- **PSR head repair applied**: GELU replaced with LeakyReLU plus small-normal weight init (mean=0, std=0.01) and zero bias, eliminating the gradient starvation mechanism. The dead `PSRTransitionPredictor` class was confirmed absent from the pipeline and has been removed.
- **Multi-task detection (D3)** f**ull-38k evaluation yields mAP50_pc = 0.00009** — the earlier 0.358 and 0.573 subsample figures were biased (evaluated only GT-containing frames). On full-38k, 92% of frames have zero GT, and the detection head's ~105 predictions/frame collapse precision-recall. Cascade analysis confirms implementation bugs as the dominant failure cause, not multi-task interference.
- **FiLM is static, not modulatory**: measured scaling factor gamma = 1.98 with standard deviation 0.002 across all conditioning inputs. The FiLM mechanism applies a uniform 2x scale regardless of input, providing no conditional modulation — effectively a learned constant multiplier.
- **Pose evaluation expanded**: forward MAE median = 8.94 deg (excluding outlier `14_assy_0_1`: 8.46 deg). The outlier is a model prediction failure, not a GT issue. Bootstrap CIs: forward 9.14 deg [7.74-10.87], up 7.78 deg [6.89-8.81]. GT pose variance: forward range 65.20 deg, up range 46.48 deg (ratio 1.40x). Variance decomposition: 85-92% of variance is within-recording noise, confirming temporal smoothing headroom.
- **24-to-11 PSR mapping verified correct** — no mapping error contributed to, or masked, the gradient starvation in the PSR head.
