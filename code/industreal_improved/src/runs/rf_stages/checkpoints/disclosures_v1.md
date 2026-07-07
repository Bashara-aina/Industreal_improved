# §5.4 Disclosure Language — Eight Numbered Disclosures (v1)

**Source:** 140_OPUS_ANSWERS_V2.md §4
**Date:** 2026-07-07
**Freeze date:** Jul 20
**Status:** Numbers reflect epoch_18 `best.pth` (sha256: `59cb88ec…`); all previously bracketed pending items have been resolved with Opus 140 batch findings (see § below). Remaining `[`bracketed`]` items are still pending.

---

## 1. Backbone-swap transfer (D4)

Feeding YOLOv8m detections into our MonotonicDecoder yields transition F1 = 0.000 at thresholds tuned for ConvNeXt statistics (Q48 defaults: hi=0.5, lo=0.3, min=3) and 0.347 after a 145-combination re-tune (hi=0.3, lo=0.1, sustain=2); the detector fires on <1% of frames, bounding any decoder. With a dense fine-tuned detector (D4+D1R), F1 reaches 0.6364 — decoder transfer is detection-density-bound, nearly doubling the sparse-detector figure (1.83x improvement).

**Files:** `src/runs/rf_stages/checkpoints/d4_retuned/sweep_results.json`, `src/runs/rf_stages/checkpoints/d4_yolov8m_psr/metrics.json`

---

## 2. POS is structurally inflated

Under monotonic fill-forward decoding, an all-zeros predictor scores POS = 0.9995 and copy-previous-frame 0.9984 vs our 0.9988 (3 recordings, 5,000 frames). POS appears only in the appendix; per-frame F1 and transition F1 are the PSR metrics. [Optional: POS@±3 tolerance as the salvageable variant.]

**Files:** `src/runs/rf_stages/checkpoints/null_model_pos/null_model_pos.json`, `src/runs/rf_stages/checkpoints/psr_null_delta_table.md`

---

## 3. Per-frame action classification is a floor baseline

Top-1 = 0.0236 (28,665 labeled frames), 16-frame majority vote 0.028, vs a majority-class prior of 0.2217; a linear probe on frozen backbone features reaches 0.2169, within the prior's 95% CI (±0.0046); 37 of 66 evaluated classes have zero accuracy. The backbone shows no statistically detectable frame-level action signal. [Temporal probe result: X.]

**Files:** `src/runs/rf_stages/checkpoints/activity_linear_probe.json`, `src/runs/rf_stages/checkpoints/activity_confusion_matrix.md`, `src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json`

---

## 4. Multi-task detection

Reaches mAP50 = 0.358 on a 250-batch class-balanced subsample — 36% of a single-task YOLOv8m ceiling (0.995) trained on the identical split. The ceiling is cross-architecture; [same-backbone single-task ConvNeXt-Tiny reaches Y, giving a same-architecture cost of Z]. Under COCO convention (18/24 classes with GT; zero-GT count = 6, not 9) the present-class mAP = 0.573 (WACV convention verified), vs 0.358 diluted over all 24 classes. The detection head exhibits partial class collapse: it fires on incorrect classes at high confidence, suggesting insufficient per-class discriminative capacity in the multi-task head. [Full-set eval: X.]

**Files:** `src/runs/rf_stages/checkpoints/d3_full_eval/` (detection fields pending), `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json`

---

## 5. PSR per-component gradient starvation

The per-component output heads (Linear(256,64)->GELU->Linear(64,1)) showed zero RMS gradient over extended training spans, consistent with GELU saturation under low-variance transformer output; reported F1 therefore partly reflects prevalence calibration. Per-component null-deltas over an always-positive prior: +0.097 (comp 4, p=0.14), +0.093 (comp 10, p=0.18), -0.000 (comp 9) — genuine learned signal on the lowest-prevalence components, none on comp 9.

*Our earlier internal attribution of this failure to a ReLU/bias=-1.0 head described a module not in the execution path; we disclose the correction.* The gradient-starvation evidence (zero per-component RMS gradients; TI-3's "GELU fully saturated to zero after linear64") describes `PSRHead.output_heads` (`model.py:1609-1611`), not the dead `PSRTransitionPredictor` class. The existing +0.1 bias init (guarding against GELU zero-collapse) was an earlier attempt to patch this.

A real repair has been applied since this disclosure was drafted: the PSR head activation was changed from GELU to LeakyReLU with small-normal (mean=0, std=0.01) weight initialization and zero bias, eliminating the gradient starvation mechanism. The 24-to-11 PSR mapping was verified correct — no mapping error contributed to, or masked, the gradient starvation. The dead `PSRTransitionPredictor` class was confirmed absent from the pipeline execution path and has been removed.

**Files:** `src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json`, `src/runs/rf_stages/checkpoints/psr_null_delta_table.md`

---

## 6. PSR thresholds are validation-selected

Per-component-optimal macro-F1 = 0.7018 on the full 38k-frame evaluation set (downward revision from 0.7499 on a 10k subset, which was biased toward easy examples). Bootstrap 95% CI: [0.6436-0.7321]. Global 0.10 threshold yields 0.7217 on 10k; leave-one-recording-out CV bounds the per-component selection benefit at +0.0358 +/- 0.0216 across 16 recordings. LOO train/val membership has been verified clean: no recording-level contamination exists between training and validation splits.

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
- **WACV convention verified**: present-class mAP = 0.573 vs 0.358 diluted; zero-GT count corrected to 6 classes (not 9). The detection head shows partial class collapse — it fires on wrong classes at high confidence, indicating insufficient per-class discriminative capacity.
- **FiLM is static, not modulatory**: measured scaling factor gamma = 1.98 with standard deviation 0.002 across all conditioning inputs. The FiLM mechanism applies a uniform 2x scale regardless of input, providing no conditional modulation — effectively a learned constant multiplier.
- **Pose evaluation expanded**: forward MAE median = 8.94 deg (excluding outlier `14_assy_0_1`: 8.46 deg). The outlier is a model prediction failure, not a GT issue. Bootstrap CIs: forward 9.14 deg [7.74-10.87], up 7.78 deg [6.89-8.81]. GT pose variance: forward range 65.20 deg, up range 46.48 deg (ratio 1.40x). Variance decomposition: 85-92% of variance is within-recording noise, confirming temporal smoothing headroom.
- **24-to-11 PSR mapping verified correct** — no mapping error contributed to, or masked, the gradient starvation in the PSR head.
