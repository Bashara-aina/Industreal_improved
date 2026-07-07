# §5.4 Disclosure Language — Eight Numbered Disclosures (v1)

**Source:** 140_OPUS_ANSWERS_V2.md §4
**Date:** 2026-07-06
**Freeze date:** Jul 20
**Status:** Numbers reflect epoch_18 `best.pth` (sha256: `59cb88ec…`); bracketed items are pending experiments that finalize each sentence.

---

## 1. Backbone-swap transfer (D4)

Feeding YOLOv8m detections into our MonotonicDecoder yields transition F1 = 0.000 at thresholds tuned for ConvNeXt statistics (Q48 defaults: hi=0.5, lo=0.3, min=3) and 0.347 after a 145-combination re-tune (hi=0.3, lo=0.1, sustain=2); the detector fires on <1% of frames, bounding any decoder. [Finalize after D4+D1R: "with a dense fine-tuned detector, F1 = X — decoder transfer {is|is not} detection-density-bound."]

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

Reaches mAP50 = 0.358 on a 250-batch class-balanced subsample — 36% of a single-task YOLOv8m ceiling (0.995) trained on the identical split. The ceiling is cross-architecture; [same-backbone single-task ConvNeXt-Tiny reaches Y, giving a same-architecture cost of Z]. Under COCO convention (15/24 classes with GT) the present-class figure is [0.573, pending convention verification]. [Full-set eval: X.]

**Files:** `src/runs/rf_stages/checkpoints/d3_full_eval/` (detection fields pending), `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json`

---

## 5. PSR per-component gradient starvation

The per-component output heads (Linear(256,64)->GELU->Linear(64,1)) showed zero RMS gradient over extended training spans, consistent with GELU saturation under low-variance transformer output; reported F1 therefore partly reflects prevalence calibration. Per-component null-deltas over an always-positive prior: +0.097 (comp 4, p=0.14), +0.093 (comp 10, p=0.18), -0.000 (comp 9) — genuine learned signal on the lowest-prevalence components, none on comp 9.

*Our earlier internal attribution of this failure to a ReLU/bias=-1.0 head described a module not in the execution path; we disclose the correction.* The gradient-starvation evidence (zero per-component RMS gradients; TI-3's "GELU fully saturated to zero after linear64") describes `PSRHead.output_heads` (`model.py:1609-1611`), not the dead `PSRTransitionPredictor` class. The existing +0.1 bias init (guarding against GELU zero-collapse) was an earlier attempt to patch this.

**Files:** `src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json`, `src/runs/rf_stages/checkpoints/psr_null_delta_table.md`

---

## 6. PSR thresholds are validation-selected

Per-component-optimal macro-F1 = 0.7499 on a 10k-frame subset vs 0.7217 at a global 0.10 threshold; leave-one-recording-out CV bounds the selection benefit at +0.0358 +/- 0.0216 across 16 recordings. [Full-38k per-comp figure: X; LOO caveat: recordings span the model's train/val membership.]

**Files:** `src/runs/rf_stages/checkpoints/psr_optimal_thr_v2/optimal_thresholds.json`, `src/runs/rf_stages/checkpoints/psr_loo_cv/loo_cv_results.json`, `src/runs/rf_stages/checkpoints/full_eval_ep18_stream/metrics.json`

---

## 7. Three-and-a-half-month evaluation-index bug

Read position channels [3:6] as the up-vector, reporting 26.20 deg; the corrected slice [6:9] yields 7.78 deg, cross-checked by three independent scripts. The training loss always used the correct indices (`losses.py:951-952` slices `fwd=[0:3], pos=[3:6], up=[6:9]` identically for pred and target), so only reporting — not learning — was affected. One legacy diagnostic script (`head_pose_diag.py`) remains unfixed and is marked deprecated.

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
