# Preflight Audit — P1–P8 Defects (175 §2)

**Date:** 2026-07-08
**Scope:** Read-only audit against actual repo code at `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved`
**Reference:** `AAIML/175_ULTIMATE_GUIDE_TIER_F.md` §2

---

## P1 — Hash-randomized activity labels (175 §2)

- **Defect (paraphrased):** The class-string-to-integer mapping in `train_v8_multitask.py` uses Python's built-in `hash()`, which is randomized per process (PYTHONHASHSEED). This produces non-deterministic label indices across workers, making activity training unreproducible and effectively random.
- **Reference file:** `scripts/train_v8_multitask.py:216` (175 §2 table says line 216)
- **Actual file:** `code/industreal_improved/scripts/train_v8_multitask.py`

- **Evidence:**
  ```
  $ grep -n "hash" code/industreal_improved/scripts/train_v8_multitask.py
  217:        return hash(cls_str) % self.num_classes
  ```
  Context lines 213-217:
  ```python
  def _class_to_idx(self, cls_str):
      """Map class string to integer index. 69 classes total."""
      if not isinstance(cls_str, str):
          return 0
      return hash(cls_str) % self.num_classes
  ```

- **Status: CONFIRMED** — defect exists at line 217 (off by one from the documented line 216, same code). Python's hash randomization (enabled by default since Python 3.3 via PYTHONHASHSEED) means this mapping gives different results across processes/workers. This is active V8 training code.

- **Action:** Replace with a stable ordered-dict lookup from a frozen sorted class list, ensuring the same string always maps to the same index regardless of Python process.

---

## P2 — Activity zero-gradient / double-ramp (175 §2)

- **Defect (paraphrased):** Activity training loss was effectively 0.0 (zero gradient) due to two compounding bugs: (a) a label masking path that could silence all labels in a batch, and (b) an activity ramp applied TWICE — once at the raw loss level and again at the Kendall precision level, producing effective supervision of only 4% instead of 20% at epoch 0 (ramp-squared).

- **Reference file:** `src/training/losses.py` (activity ramp site)
- **Actual file:** `code/industreal_improved/src/training/losses.py`

- **Evidence:**
  The single-application ramp (remaining canonical site) at lines 1385-1389:
  ```python
  act_ramp = 1.0
  if self.train_act and self._act_epoch_counter >= 0:
      _ramp_ep = self._act_epoch_counter
      act_ramp = min(1.0, (_ramp_ep + 1) / max(self._act_warmup_epochs, 1))
  loss_act = loss_act * act_ramp
  ```
  The F18 double-ramp fix comment at lines 1734-1743:
  ```
  # [F18 2026-07-02 Fable RF4 consult] DOUBLE-RAMP FIX. The activity
  # ramp was applied TWICE: once to the RAW loss in the activity
  # section above (`loss_act = loss_act * act_ramp`, the canonical
  # site — it covers Kendall, fixed-weight, and non-Kendall paths
  # alike), and AGAIN here to the Kendall precision. Effective
  # activity supervision during warmup was ramp^2: 4% (not 20%) at
  # epoch 0, 36% (not 60%) at epoch 2 — a compounding factor in
  # every historical activity-collapse episode. The precision-side
  # multiplication below is removed; the loss-level ramp is the
  # single source of truth.
  ```
  Check that the second ramp was indeed removed (line 1769):
  ```python
  # [F18] Activity ramp handled ONCE at the loss level
  # (activity section above) — the old prec_act *= act_ramp
  # here made staged warmup ramp^2 as well.
  ```
  However, no `assert loss_act > 0` guard exists at step 0:
  ```
  $ grep -n "assert.*loss_act\|loss_act.*assert\|loss_act.*nonzero\|zero.*grad.*activity" \
    code/industreal_improved/src/training/losses.py
  # (no matches)
  ```

- **Status: ALREADY_FIXED** — the double-ramp bug (F18) was fixed on 2026-07-02. The ramp is now applied exactly once at the loss level (line 1389). The precision-side multiplication in the staging block was removed. The label masking path has a guard (line 1344: `if valid_mask.any()`). However, the recommended `assert loss_act > 0` guard at step 0 is not present — this is an additional hardening measure, not the core defect.

- **Action:** Add a step-0 assert that `loss_act > 0` and gradients are non-zero, as recommended in 175 §2. The core double-ramp bug is already resolved.

---

## P3 — Empty-subsample detection eval yields 0.0/NaN (175 §2)

- **Defect (paraphrased):** When the evaluation subsample contained frames with no detection GT boxes, `det_mAP50` was reported as 0.0 and `det_n_present_classes=0`, making it look like the model completely failed. The real mAP was 0.468 when evaluated on the full dataset.

- **Reference file:** `src/evaluation/full_eval_inprocess.py:406` (`det_n_present_classes=0`)
- **Actual file:** `code/industreal_improved/src/evaluation/full_eval_inprocess.py`

- **Evidence:**
  The guard at lines 400-406:
  ```python
  gt_box_total = sum(len(b) for b in dg_boxes)
  if gt_box_total == 0:
      logger.warning("No GT boxes in evaluation split — skipping detection mAP")
      results["det_mAP50"] = 0.0
      results["det_mAP_50_95"] = 0.0
      results["det_mAP50_all_frames"] = 0.0
      results["det_n_present_classes"] = 0
  ```
  The fallback subsample at lines 662-677 (triggered on NaN):
  ```python
  bad_keys = has_nan(metrics)
  if bad_keys:
      logger.warning("FALLING BACK to %d-seed subsample variance.", args.fallback_seeds)
      fallback_result = run_multi_seed_subsample(
          model, device,
          num_seeds=args.fallback_seeds,
          batches_per_seed=args.fallback_batches,  # default 2500
          ...
      )
  ```
  This fallback uses `batches_per_seed=2500` — if those 2500 batches happen to have no GT boxes (possible with rare detection classes), the subsample returns 0.0 mAP.

- **Status: CONFIRMED** — the guard at line 401 handles the edge case of zero GT boxes by returning 0.0, but this is semantically wrong: 0.0 mAP means "model produced no correct predictions" which is different from "no evaluation data available." The subsample fallback (line 671) can also produce misleading zeros when its limited batches contain no GT boxes for certain classes. The `det_n_present_classes=0` sentinel helps detect the issue but doesn't prevent it from being misinterpreted.

- **Action:** Change the zero-GT-box guard to return NaN or a sentinel value (not 0.0 mAP). Ensure the default eval always uses the full, stratified validation set. The subsample fallback should also be stratified to guarantee class coverage.

---

## P4 — Staging zeroes PSR/pose precision until epoch 16 (175 §2)

- **Defect (paraphrased):** When `STAGED_TRAINING=True`, the Kendall staging block zeroes out PSR and head-pose precision during stages 1 and 2 (epochs 0-15), and PSR doesn't activate until epoch 16. This means approximately half the training budget was spent with PSR/pose contributing nothing, producing the "heads dead until epoch 16" artifact.

- **Reference file:** `src/training/losses.py` `_get_kendall_stage`
- **Actual file:** `code/industreal_improved/src/training/losses.py`

- **Evidence:**
  The staging block guard at line 1745:
  ```python
  if bool(getattr(C, 'STAGED_TRAINING', True)) and self._current_epoch >= 1:
      stage = _get_kendall_stage(self._current_epoch)
      if stage == 1 and not _kendall_fixed:
          prec_hp = prec_hp * 0
          lv_hp = lv_hp * 0
          prec_psr = prec_psr * 0
          lv_psr = lv_psr * 0
      elif stage == 2 and not _kendall_fixed:
          prec_psr = prec_psr * 0
          lv_psr = lv_psr * 0
  ```
  Config defaults (lines 887, 113):
  ```
  STAGED_TRAINING = False  # Full production: all 5 heads active from epoch 0
  KENDALL_STAGED_TRAINING = False  # [FIX] was True
  ```

- **Status: ALREADY_FIXED** — Both `STAGED_TRAINING=False` and `KENDALL_STAGED_TRAINING=False` are set in the config. When `STAGED_TRAINING=False`, the staging block is entirely skipped (line 1745 initial condition fails). All heads are trained from epoch 0. The `_get_kendall_stage` function at line 58 correctly delegates to `train.get_stage()` which respects `reinit_epoch_offset`.

- **Action:** Verify that all experiment configs also set `STAGED_TRAINING=False` and `KENDALL_STAGED_TRAINING=False`. Add a config validation assert to catch any run config that accidentally re-enables staging.

---

## P5 — PSR reported as per-frame 0.7018 instead of event-F1@±3 (175 §2)

- **Defect (paraphrased):** The PSR headline metric was reported as per-component per-frame binary F1 (0.7018), which is incomparable to the STORM/B3 literature that uses event-based F1 with ±3 frame tolerance. The correct peer-comparable metric is `event_f1@±3`.

- **Reference file:** `src/evaluation/decoder_oracle_bound.py:253` (event_f1 function)
- **Actual file:** `code/industreal_improved/src/evaluation/decoder_oracle_bound.py`
  Also: `code/industreal_improved/src/evaluation/full_eval_inprocess.py` (primary eval path)

- **Evidence:**
  The `event_f1` function at `decoder_oracle_bound.py:252`:
  ```python
  $ grep -n "def event_f1" code/industreal_improved/src/evaluation/decoder_oracle_bound.py
  252:def event_f1(pred_tr: np.ndarray, gt_tr: np.ndarray, tol: int = 3) -> float:
  ```
  However, the primary eval path in `full_eval_inprocess.py` at lines 458-466 uses per-frame F1:
  ```python
  psr_f1s = []
  for c in range(11):
      prec = psr_tp[c] / max(psr_tp[c] + psr_fp[c], 1)
      rec = psr_tp[c] / max(psr_tp[c] + psr_fn[c], 1)
      f1 = 2 * prec * rec / max(prec + rec, 1e-9)
      psr_f1s.append(f1)
  results["psr_macro_f1"] = float(np.mean(psr_f1s))
  ```
  This computes per-frame binary F1 at a fixed threshold (0.10), NOT event-F1 with tolerance.
  The `event_f1` function exists in `decoder_oracle_bound.py` but is only called from standalone scripts (`psr_transition_f1.py`, `eval_convnext_psr.py`, `sweep_psr_threshold.py`), not from the primary evaluation pipeline.

- **Status: CONFIRMED** — The primary eval path (`full_eval_inprocess.py`) still reports per-frame `psr_macro_f1`, not the event-based `event_f1@±3` required for B3/STORM comparability. The `event_f1` function exists in `decoder_oracle_bound.py` at line 252 but is not wired into the main evaluation pipeline.

- **Action:** Add `event_f1@±3` computation to `full_eval_inprocess.py` as the primary PSR metric. Keep `psr_macro_f1` as a secondary metric. This requires running the monotonic decoder on the accumulated PSR logits, not just per-frame binary thresholding.

---

## P6 — 0.995 detection cited from wrong provenance (175 §2)

- **Defect (paraphrased):** The 0.995 detection mAP figure was cited as a multi-task result, but it came from a fine-tuned single-task YOLOv8m checkpoint (D1R) used as a weight source. The actual multi-task evaluation produces mAP ~0.00043 in the native evaluation, and the correct protocol-matched number should be used instead.

- **Reference files:** `checkpoints/d1_yolov8m_v3/metrics.json` vs `checkpoints/d4_d1r/metrics.json`
- **Actual files:**
  - `code/industreal_improved/src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json`
  - `code/industreal_improved/src/runs/rf_stages/checkpoints/d4_d1r/metrics.json`

- **Evidence:**
  `d1_yolov8m_v3/metrics.json` (actual native eval):
  ```json
  {
      "det_mAP50": 0.00042720194409902557,
      "det_n_present_classes": 18,
      ...
  }
  ```
  `d4_d1r/metrics.json` (weight source metadata showing the 0.995 reference):
  ```json
  {
      "_weight_source": "D1R fine-tuned YOLOv8m (best.pt, mAP=0.995)",
      "_model": "yolov8m (d1r) -> s2 -> MonotonicDecoder",
      ...
  }
  ```

- **Status: CONFIRMED** — The 0.995 figure appears only as descriptive metadata in `_weight_source` within `d4_d1r/metrics.json`, documenting where the weights came from. The native evaluation produces `det_mAP50=0.00043`. These are two completely different evaluation pipelines. The 0.995 should never have been cited as a multi-task detection result.

- **Action:** All detection claims must use protocol-matched mAP from the unified evaluation pipeline. The 0.995 metadata reference should be removed from any citation context. The `efficiency_report.py` (`scripts/training/efficiency_report.py`) exists and uses `fvcore` for proper measurement.

---

## P7 — Val vs test split mismatch (175 §2)

- **Defect (paraphrased):** Headline evaluation numbers were computed on the 5-subject validation set rather than the held-out 10-subject test set, making comparisons to published SOTA potentially unfair (if SOTA used test).

- **Reference:** `bootstrap_ci.json` — confirm it's on val not test
- **Actual file:** `code/industreal_improved/src/runs/rf_stages/checkpoints/bootstrap_ci.json`

- **Evidence:**
  ```json
  {
      "metadata": {
          "n_bootstrap": 1000,
          "random_seed": 42,
          "n_recordings": 16,
          "n_frames": 38036,
          "date": "2026-07-06"
      }
  }
  ```
  The metadata reports 16 recordings and 38036 frames, evaluated as split="val" (confirmed by the eval code path). The 175 §2 prescribes 12 train / 5 val / 10 test subject split. The bootstrap_ci.json was computed on val (16 recordings is consistent with the val set + some additional recordings), not on the test set.
  
  The split mechanism in `industreal_dataset.py` uses directory-based splits (`recordings_root / split`), so the split is determined by which directory contains the data. There is no frozen split config file with explicit subject IDs as recommended.

- **Status: NEEDS_REVIEW** — The bootstrap_ci.json was computed on val (confirmed by eval code reading `split="val"`). The test split (10 subjects) has not been used for any reported metrics. A frozen split config with explicit subject IDs does not appear to exist in the codebase. The split is directory-based, which is fragile.

- **Action:** Create a frozen split config (`SPLIT_CONFIG.yaml` or similar) with explicit subject IDs for train/val/test as prescribed in 175 §7.1. Re-run all SOTA-comparison metrics on the test split after training.

---

## P8 — Fabricated efficiency table in 167/170 (175 §2)

- **Defect (paraphrased):** The 4x/600M efficiency multiplier cited in documents 167 and 170 was an estimate, not a measurement. It claims 4x compute savings and 600M total parameters for single-task baselines, but these were never measured with `fvcore` or similar tooling.

- **Reference files:** `167`/`170` efficiency table
- **Actual files:**
  - `analyses/consult_2026_06_10/AAIML/167_MULTITASK_ARCHITECTURE_STRATEGY.md`
  - `analyses/consult_2026_06_10/AAIML/170_DISCUSSION_CONCLUSION.md`

- **Evidence:**
  From `167_MULTITASK_ARCHITECTURE_STRATEGY.md`:
  ```
  | Parameters | 4x (~150M total) | 1x (~30M) | 1x (~90M) |
  | Storage | 4x | 1x | 1x |
  | Inference time | 4x (sequential) | 1x | 1x |
  "V8 gives 4x efficiency gain over single-task."
  ```
  From `170_DISCUSSION_CONCLUSION.md`:
  ```
  Parameters: 4x ~150M = 600M total
  | GPU hours for training | 4 | 1 | 4x |
  | Parameters | 600M | 90M | 6.7x |
  "V8 multi-task is 4x more efficient than single-task on all dimensions."
  ```
  The `efficiency_report.py` replacement exists at `scripts/training/efficiency_report.py`:
  ```
  $ head -5 code/industreal_improved/scripts/training/efficiency_report.py
  """
  Efficiency Report Generator
  ============================
  Doc 03 B.1: Standard efficiency benchmark for POPW.
  Reports:
  - Total / trainable params (M)
  - GFLOPs at 1280x720 (via fvcore.nn.FlapCountAnalysis)
  ```

- **Status: CONFIRMED** — Documents 167 and 170 contain unmeasured efficiency claims ("4x", "600M", "6.7x") that were never produced by a measurement tool. These are estimates at best and fabricated at worst. The replacement script (`efficiency_report.py`) exists and uses `fvcore` for proper measurement.

- **Action:** Delete the fabricated efficiency claims from documents 167 and 170 (they are planning documents, not source code). Use `efficiency_report.py` with `fvcore.nn.FlapCountAnalysis` to produce real measurements. Populate 175 Table C with actual measured values. Note that the 4x multiplier in 167 is for the **two-backbone V8** (MViTv2-S + YOLOv8m), not the proposed single-backbone Hiera design in 175.

---

## Summary

| Defect | Status | Core Finding |
|--------|--------|-------------|
| **P1** | CONFIRMED | `hash(cls_str) % num_classes` at line 217 of `train_v8_multitask.py` — Python hash randomization breaks label determinism |
| **P2** | ALREADY_FIXED | Double-ramp (F18) fixed 2026-07-02. Ramp applied once at loss level. Missing `assert loss_act > 0` guard. |
| **P3** | CONFIRMED | Zero-GT-box guard returns 0.0 mAP (misleading). `det_n_present_classes=0` sentinel exists but doesn't prevent misinterpretation. |
| **P4** | ALREADY_FIXED | `STAGED_TRAINING=False` and `KENDALL_STAGED_TRAINING=False` in config. Staging block is a no-op. |
| **P5** | CONFIRMED | `full_eval_inprocess.py` still reports per-frame binary F1, not `event_f1@±3`. The `event_f1` function exists at `decoder_oracle_bound.py:252` but is not wired into the primary eval pipeline. |
| **P6** | CONFIRMED | 0.995 appears as metadata `_weight_source` in `d4_d1r/metrics.json`, not as a measured multi-task result. Native eval gives 0.00043. |
| **P7** | NEEDS_REVIEW | Bootstrap CI computed on val (16 recordings). No frozen split config with explicit subject IDs exists. Split is directory-based. |
| **P8** | CONFIRMED | 167/170 contain unmeasured 4x/600M efficiency claims. Replacement script at `efficiency_report.py` uses `fvcore`. |

**Overall preflight status:** 1 already-fixed (P2, P4), 5 confirmed open (P1, P3, P5, P6, P8), 1 needs review (P7). The smoke test (overfit 50 clips to ~0 loss on all 4 heads) should not be attempted until P1 and P5 are resolved.
