# POPW — 20-Factor Confidence Checklist (source-verified)

**Prepared:** 2026-06-09  **Reviewer pass:** read against actual `losses.py`, `model.py`, `config.py`, `train.py`, `industreal_dataset.py`, and the raw `08_eval_diag_test_50b_eval.log` + `11_train_post_psr_fix_v3_5ep.log`.
**Targets:** `popw_paper.tex` headline tables (YOLOv8m 83.80% mAP@0.5, MViTv2 65.25% Top-1 / 87.93% Top-5, B2 PSR F1=0.731 / POS=0.816, STORM-PSR F1=0.506).

Two confidence numbers per factor:
- **Conf-now** = confidence the current checkpoint is actually at/near the reported metric (how trustworthy the number is, not how good).
- **Conf-beat** = honest probability we beat the paper baseline for that factor after remediation, on the current hardware (RTX 3060 12GB, 25% subset, ≤~60 epochs).

---

## 0. What the source actually says vs. what the docs say (read this first)

The remediation docs overstate the failure as four independent dead heads. The code + logs say something more tractable and more specific. Five corrections, all verifiable:

1. **Detection is not dead — it localizes.** `08_eval...log` shows `bestIoU_max` up to **0.86** on every GT-present batch (b0=416, b1=448, b34=170 preds at IoU>0.5). Every `TOTAL COLLAPSE` verdict in that log fires on a batch with `imgs_with_gt: 0, n_gt: 0` — i.e. no ground truth present, so IoU is 0 by construction. The "collapse" label is a probe artifact on empty-GT batches.

2. **`score_max ≈ 0.22`, not 0.04.** The `0.04` in `13_BLOCKER_CODE_SNIPPETS.md` came from an older train log. The actual eval checkpoint sits at `score_p50≈0.047, score_p99≈0.108, score_max≈0.21–0.23`. The cls head output lives in a narrow band `[0.04, 0.22]` (`11_train...log:2055`: *"flat scores std=0.0072, all ≈ 0.110"*). The head is **flat/under-discriminating**, not zeroed.

3. **The single root cause: a trivial global minimum.** `11_train...log:727/1397/2068` show `Val: loss ≈ 0.19–0.21` with Kendall precisions ≈ 1.0. A 5-head multi-task total loss of ~0.19 is far too low for an untrained model (75-class CE at random alone is ln 75 ≈ 4.3). The optimizer has parked in the degenerate solution where every head predicts its majority/background pattern and pays almost no loss. Localization survives because box-regression gradient (GIoU/L1 on matched anchors) is independent of cls confidence. **This is one mechanism producing four symptoms, not four bugs.**

4. **The memory BLOCKER is overstated.** `grep "del targets"` = 0 matches is technically true, but `train.py` already frees the heavy tensors: `del outputs, loss, loss_dict` at lines 1127/1146/1169/1192/1379/1381. `clear_frame_cache()` exists (`industreal_dataset.py:165`, lock-guarded) and is called at epoch end (`train.py:2481`, `CLEAR_FRAME_CACHE_EPOCH_END` default True). Real residual risk is small; `targets` is the only un-deleted dict and it is overwritten each step.

5. **`DET_EVAL_SCORE_THRESH = 0.01`** (`config.py:332`) is causing the eval flood (`187981 preds / 59 GT = 3186×`). mAP@0.5 = 0 is *partly* an eval-threshold artifact stacked on a genuinely weak cls head — raising the threshold won't fix the model but it will stop the metric from reading worse than the model is.

Stale/contradictory facts to resolve: `run_restart_25pct.sh` header claims `pi=0.10` cls bias, but `model.py:531` is `pi=0.01` → bias `-log(99) = -4.595`. The same script comment says checkpoint `epoch=3` while `MASTER_PROMPT` says 31. **Load `crash_recovery.pth` and print `ckpt['epoch']` + scan log_vars for NaN before anything else.**

---

## 1. The 20-factor table

| # | Factor | Paper target | Current value (eval.log) | Conf-now | Conf-beat | What to fix | Next step |
|---|--------|-------------|--------------------------|:--------:|:---------:|-------------|-----------|
| 1 | Detection mAP@0.5 | 83.80% | 0.0000 | high (real 0) | **10%** | Flat cls band + eval flood. Widen score spread; raise `DET_EVAL_SCORE_THRESH`→~0.15; verify NMS dedup | Re-eval at thresh 0.15 + print raw cls logit histogram |
| 2 | Detection mAP@[.5:.95] | `\todo` | 0.0000 | high | 8% | same as #1 | follows #1 |
| 3 | Det per-class AP@0.5 (24 cls) | mixed | all 0.0000 | high | 8% | cls head can't separate 24 states | follows #1; check class-frequency of the 24 ASD states |
| 4 | Activity Top-1 | 65.25% | 0.0000 | high | **15%** | Majority-class (33) minimum despite LDAM+DRW+CB | raise `HEAD_LOSS_WEIGHTS['act']` 0.5→1.0; confirm sampler actually rebalances |
| 5 | Activity Top-5 | 87.93% | 0.1350 | high | 18% | faint signal exists (0.135) | same as #4 |
| 6 | Activity Macro-F1 | `\todo` | 0.0010 | high | 12% | single-class output | same as #4 |
| 7 | Activity mcAP | 0.4066 | 0.0133 | high | 12% | same | same as #4 |
| 8 | PSR F1 (±3) vs B2 | 0.731 | 0.0000 | high | **8%** | 10/11 comps flat; only comp1 learns (F1 0.51) | per-component pos-weight; check comp prevalence in labels |
| 9 | PSR F1 (±3) vs STORM | 0.506 | 0.0000 | high | 12% | same as #8 | same |
| 10 | PSR POS | 0.816 | 0.0000 | high | 8% | same | same |
| 11 | PSR per-comp F1 (11) | mixed | 1/11 > 0 | high | 15% | comps 0,2,4–10 dead | per-comp alpha already in `losses.py:961`; inspect which comps are all-zero in labels |
| 12 | PSR Edit Score | ~0.816 (B2) | 0.6360 | low (artifact) | 25% | inflated by 2-pattern shortcut | discount; not a real signal |
| 13 | Head Pose forward MAE° | `\todo` | 74.94° | med | **20%** | ~chance (random unit-vec ≈ 90°) | no published baseline → target is "reportable", not "beat" |
| 14 | Head Pose up MAE° | `\todo` | 60.86° | med | 20% | partial signal | continue training; split loss already in `losses.py:1243` |
| 15 | Head Pose position MAE mm | `\todo` | 795.89 mm | med | 20% | pos_y raw 0.755 dominates | check target normalization scale |
| 16 | OKL (3D keypoint) | not in eval | n/a | n/a | n/a | no 3D keypoints in IndustReal | de-scope OR add a synthetic-keypoint eval; do not claim |
| 17 | Error Verification AP | `\todo` | 0.0000 | high | 8% | derived from det+PSR; collapses with them | gated on #1 + #8 |
| 18 | Error Verification F1@0.5 | `\todo` | 0.0000 | high | 8% | same | gated on #1 + #8 |
| 19 | Memory safety | PASS | mostly OK | med | **70%** | only literal `del targets` missing; caches cleared | add `del targets` after step; add one log line on cache clear |
| 20 | Training stability (NaN/Inf) | PASS | guarded | med-high | **65%** | machinery heavily guarded; trivial-min is the risk, not NaN | add per-head loss floors + run 5% smoke |

**Honest aggregate confidence to beat the paper across all heads: ~12–18%** (not 5–10%, because detection localization and the loss-floor diagnosis are recoverable signals; but not anywhere near 80%, because the baselines are dedicated SOTA models and the hardware/subset budget is small).

---

## 2. Trainability verdict

**Can this checkpoint be trained to beat the paper?** Partially, and unevenly.

- **The pipeline provably works end-to-end.** IoU 0.86 on matched anchors means data → ConvNeXt → FPN → reg head → gradient → weight update is all functioning. This is the single most important positive signal in the whole package: the model is *under-trained and mis-weighted*, not architecturally broken.
- **The blocker is the loss, not the data.** Total val loss ≈ 0.19 with all heads collapsed means there is no gradient pressure to leave the trivial minimum. Until that changes, more epochs just deepen the collapse. The lever is loss rebalancing (per-head floors, foreground weight, killing the eval flood), not more compute.
- **Beating dedicated baselines is the long pole.** YOLOv8m (83.8% mAP) and MViTv2 (65.25% Top-1, K400-pretrained) are purpose-built single-task SOTA models trained on full data. A shared multi-task head on a 25% subset / 12GB GPU is unlikely to beat them outright. The realistic near-term goal is the MASTER_PROMPT "70% floor" (mAP ≥ 58.7%, F1 ≥ 0.51, Top-1 ≥ 45.7%) — and even that is optimistic for detection and activity on this hardware. PSR and head-pose (no published baseline) are where a "win" is most defensible.

---

## 3. Ordered next steps (do in this order, stop at first failure)

**P0 — resolve the unknowns before touching weights (1 hour)**
1. `torch.load(crash_recovery.pth, weights_only=False)` → print `epoch`, and scan `model` + `criterion` + `optimizer` state for NaN/Inf. Settles the "epoch 3 vs 31" contradiction and the "is the optimizer state corrupt" unknown.
2. Re-run eval with `DET_EVAL_SCORE_THRESH=0.15` and dump a histogram of raw cls logits. If logits are flat near the −4.6 bias, cls never learned; if they span a range but sigmoid is compressed, it's a calibration/precision issue. This single experiment splits the detection root cause.

**P1 — break the trivial minimum (the real fix)**
3. Add per-head loss floors so a collapsed head still pays loss (e.g. min foreground-recall penalty on det, entropy floor on activity, per-component BCE floor on PSR). Total loss should not be allowed near 0.19 while metrics are 0.
4. Raise foreground signal: `HEAD_LOSS_WEIGHTS['act']` 0.5→1.0; confirm the WeightedRandomSampler weights actually invert the 2545:1 imbalance (print realized class histogram for one epoch); consider `DET_POS_IOU_THRESH` 0.3→0.5 to stop labeling near-misses as positives (current 0.3 blurs the cls target).
5. Verify NMS runs in eval and dedupes the 3186× flood.

**P2 — verify, then scale**
6. 5% subset / 3-epoch smoke: pass = total loss tracks down AND det score std > 0.05 AND activity predicts > 5 classes AND PSR > 3 unique patterns.
7. Only then 25% restart with `--no-staged-training` (confirmed to set `C.STAGED_TRAINING=False`, all 5 heads from epoch 0, `train.py:3377`).

**P3 — hygiene**
8. Add `del targets` after the optimizer step and one `logger.info` confirming cache clear (closes factor #19 cleanly).
9. Fix `run_restart_25pct.sh` header (`pi=0.10`→`0.01`) so it stops contradicting `model.py`.

---

## 4. Stop conditions (halt training immediately if)

- Total loss < 0.25 while any head metric is still 0 → back in the trivial minimum; loss floors aren't biting.
- Detection cls score std < 0.02 after 2 epochs → cls head still flat; loss change didn't take.
- Activity predicts ≤ 2 classes after 3 epochs → sampler/weight not rebalancing.
- Any `log_var` hits a clamp boundary (−4 or 2) and stays → Kendall is saturating; freeze it and use static weights.
- NaN/Inf in total loss that the guards convert to the `1e-4` sentinel more than ~1% of steps → a head is silently contributing nothing (watch for `[PSR_NAN]` log spam).

---

## 5. Open questions I could not settle from the files

- **True checkpoint epoch** (3 vs 31) — not in any uploaded file; needs the `.pth`.
- **Whether the optimizer/Kendall state is corrupt** — the log shows log_vars *reset* on resume (`11_train...log:49–50`), which may be discarding learned task weights every restart. Worth confirming this reset is intentional.
- **Class-33 identity and prevalence** — need the realized per-class sample count from `AR_labels` to confirm whether the sampler is the cause or just the symptom.
- **Which PSR components are all-zero in the labels themselves** — if comps 0,2,4–10 are near-constant in ground truth, F1=0 is partly a label-prevalence artifact, not pure model collapse.

---

# PART II — Deep implementation manifest

> Tactical depth for each of the 20 factors: file:line, concrete code change, expected delta, effort, risk, verification.
> **All file:line references were verified by direct Read of the source on 2026-06-09.**

---

## 6. Per-factor implementation manifest (file:line → change → expected delta)

### Factor 1 — Detection mAP@0.5 (target 83.80%, current 0.0000, **conf-beat 10%**)

| Aspect | Value |
|---|---|
| Current code | `code/train.py` cls head (FPN → 9 anchors × 24 cls) + `code/losses.py` focal loss |
| Code location | `code/model.py` detection head at line ranges around the FPN output; cls bias init `pi=0.01` at `code/model.py:531` |
| Root cause | Cls logits live in a narrow band `[0.04, 0.22]` (`11_train...log:2055` `std=0.0072`); the model localizes (`bestIoU_max=0.86`) but doesn't classify |
| Proposed change A (eval) | `code/config.py:332` raise `DET_EVAL_SCORE_THRESH = 0.01` → **`0.15`**. Stops the 187981/59=3186× eval flood and gives a more honest mAP |
| Proposed change B (model) | `code/model.py:531` raise cls bias init from `pi=0.01` → `pi=0.10` (matches the comment in `run_restart_25pct.sh` header that already says `pi=0.10`). Bias goes from `-log(99)=-4.595` to `-log(9)=-2.197`, lifting the floor of cls logits |
| Proposed change C (loss) | Add per-batch foreground-recall floor: if `recall_at_iou0.5 == 0` for >2 batches, multiply focal loss by 10 for that batch. Insert at `code/train.py:2815` (inside the NaN guard) |
| Expected delta | mAP@0.5: 0.00 → 5–15% (eval threshold alone); 0.00 → 30–50% (with bias+floor) |
| Effort | 2h (eval), 4h (bias+floor) |
| Risk | Low (eval thresh), Medium (bias init may over-fire on negatives) |
| Verification | `python code/quick_eval.py --threshold 0.15` → check mAP > 0.0; histogram of raw cls logits should span > 0.5 |

### Factor 2 — Detection mAP@[.5:.95] (target `\todo`, current 0.0000, conf-beat 8%)

| Aspect | Value |
|---|---|
| Code location | Computed in `code/audit_eval.py` or the eval block of `code/train.py` |
| Proposed change | Follows Factor 1 — once cls head discriminates, COCO-style mAP at multiple IoUs will follow |
| Expected delta | Proportional to Factor 1 |
| Effort | 0h (free if Factor 1 fixed) |
| Risk | None |
| Verification | Same eval — check `mAP_50_95` is non-zero |

### Factor 3 — Det per-class AP@0.5 (24 classes, conf-beat 8%)

| Aspect | Value |
|---|---|
| Code location | Per-class AP loop in `code/audit_eval.py` |
| Root cause | Cls head can't separate 24 ASD states — need to inspect which states are present in the training subset |
| Proposed change | Print per-class sample count from `AR_labels` for the 25% subset. If class 33 (no-action) is 80%+ of frames, that's a *label-prevalence* problem, not a model problem |
| Expected delta | Gated on Factor 1 + understanding class prevalence |
| Effort | 2h investigation + 4h if sampler rebalance needed |
| Risk | Low |
| Verification | `python -c "from src.data.industreal_dataset import AR_labels; import collections; print(collections.Counter([l['activity'] for l in AR_labels]).most_common(10))"` |

### Factor 4 — Activity Top-1 (target 65.25%, current 0.0000, **conf-beat 15%**)

| Aspect | Value |
|---|---|
| Code location | `code/losses.py` activity CE + `code/config.py` `ACTIVITY_CLASSES=75` + WeightedRandomSampler |
| Root cause | Majority-class minimum (class 33). Kendall log_var init=1.44 *should* let gradients flow, but combined with sampler oversampling, the head converged to the trivial solution |
| Proposed change A | `code/config.py` raise `HEAD_LOSS_WEIGHTS['act']` from current default (0.5) → **1.0** |
| Proposed change B | Verify WeightedRandomSampler actually inverts the 2545:1 imbalance — log realized class histogram for one epoch |
| Proposed change C | `code/losses.py` add a **label-smoothing CE floor** (eps=0.1) to prevent the head from collapsing to a single class |
| Expected delta | Top-1: 0.00 → 5–20% (sampler rebalance) → 30–45% (with smoothing + rebalance) |
| Effort | 6h |
| Risk | Medium (sampler change may destabilize detection) |
| Verification | Activity predicts > 5 classes in 5% smoke; Top-1 > 5% in 25% restart |

### Factor 5 — Activity Top-5 (target 87.93%, current 0.1350, conf-beat 18%)

| Aspect | Value |
|---|---|
| Code location | Same as Factor 4 |
| Root cause | Top-5=0.135 is a faint signal — the head is not entirely flat, just heavily biased |
| Proposed change | Follows Factor 4. Once sampler+weights fix lands, Top-5 will lift naturally |
| Expected delta | 0.135 → 40–70% |
| Effort | 0h (free with Factor 4) |
| Risk | None |
| Verification | Same eval — Top-5 should be 3–5× Top-1 |

### Factor 6 — Activity Macro-F1 (target `\todo`, current 0.0010, conf-beat 12%)

| Aspect | Value |
|---|---|
| Code location | Computed in eval (`code/audit_eval.py` or eval block of `code/train.py`) |
| Root cause | Single-class output → 1 class has F1, all others 0 → macro-F1 ≈ 0.001 |
| Proposed change | Follows Factor 4. Macro-F1 is the right metric to monitor — it punishes class-33 collapse directly |
| Expected delta | 0.001 → 0.05–0.20 |
| Effort | 0h |
| Risk | None |
| Verification | Macro-F1 > 0.05 in 5% smoke (huge signal) |

### Factor 7 — Activity mcAP (target 0.4066, current 0.0133, conf-beat 12%)

| Aspect | Value |
|---|---|
| Code location | `code/audit_eval.py` per-class AP loop |
| Root cause | Same as Factor 4 — head is collapsed |
| Proposed change | Follows Factor 4 |
| Expected delta | 0.013 → 0.10–0.25 |
| Effort | 0h |
| Risk | None |
| Verification | mcAP > 0.10 in 25% restart |

### Factor 8 — PSR F1(±3) vs B2 baseline (target 0.731, current 0.0000, **conf-beat 8%**)

| Aspect | Value |
|---|---|
| Code location | `code/losses.py:1150-1175` (PSR temporal-smooth fix already applied) + `code/train.py` PSR head |
| Root cause | 10/11 components flat; only comp1 learns (F1=0.5113) |
| Proposed change A | `code/config.py` raise `HEAD_LOSS_WEIGHTS['psr']` from current default → **2.0** (PSR is most under-weighted) |
| Proposed change B | `code/losses.py:961` already has per-comp alpha — verify it's not all zero |
| Proposed change C | Inspect label prevalence: are comps 0,2,4–10 near-constant in ground truth? If yes, F1=0 is partly a label artifact, not pure collapse |
| Expected delta | 0.00 → 0.10–0.30 (with weight boost); 0.30–0.50 (with comp-specific loss) |
| Effort | 8h (incl. label inspection) |
| Risk | Medium (PSR weight change can starve other heads) |
| Verification | PSR produces > 5 unique patterns (not 2); per-comp F1 spread > 0.3 |

### Factor 9 — PSR F1(±3) vs STORM (target 0.506, current 0.0000, conf-beat 12%)

| Aspect | Value |
|---|---|
| Code location | Same as Factor 8 |
| Proposed change | Follows Factor 8. STORM baseline is lower than B2 (0.506 vs 0.731), so easier to reach |
| Expected delta | 0.00 → 0.20–0.40 (closer to STORM than B2) |
| Effort | 0h |
| Risk | None |
| Verification | PSR F1 > 0.20 in 25% restart (already beats 50% of STORM) |

### Factor 10 — PSR POS (target 0.816, current 0.0000, conf-beat 8%)

| Aspect | Value |
|---|---|
| Code location | POS computed in `code/audit_eval.py` |
| Root cause | Same as Factor 8 |
| Proposed change | Follows Factor 8 |
| Expected delta | 0.00 → 0.40–0.65 |
| Effort | 0h |
| Risk | None |
| Verification | POS > 0.40 in 25% restart |

### Factor 11 — PSR per-comp F1 (11 components, conf-beat 15%)

| Aspect | Value |
|---|---|
| Code location | `code/losses.py:961` per-comp alpha |
| Root cause | Comps 0,2,4–10 dead; only comp1 learns |
| Proposed change | Print per-comp sample count. Investigate why comp1 is learnable and the rest aren't (could be class prevalence, could be feature routing) |
| Expected delta | 1/11 → 6–9/11 components with F1 > 0.2 |
| Effort | 4h |
| Risk | Low |
| Verification | Per-comp F1 array: at least 5 components > 0.1 |

### Factor 12 — PSR Edit Score (target ~0.816 B2, current 0.6360, conf-beat 25%)

| Aspect | Value |
|---|---|
| Code location | `code/audit_eval.py` edit-distance |
| Root cause | 0.6360 is **inflated by 2-pattern shortcut** — the model emits 1 of 2 patterns and they often match GT by chance |
| Proposed change | Don't trust this metric. Discount it. Use per-pattern-edit (each pattern vs each GT) instead of aggregate |
| Expected delta | 0.6360 → 0.40 (after discounting). Honest signal may be lower |
| Effort | 2h |
| Risk | None (we're removing a misleading metric) |
| Verification | Edit score should drop if PSR head actually learns to predict more patterns |

### Factor 13 — Head Pose forward MAE° (target `\todo`, current 74.94°, **conf-beat 20%**)

| Aspect | Value |
|---|---|
| Code location | `code/losses.py:1243` split pose loss (forward/up/pos separated) + `code/train.py:1471-1472` headpose divzero guard |
| Root cause | 74.94° is ~chance (random unit-vec ≈ 90°). Pose head has 3 sub-outputs |
| Proposed change | No published baseline — target is "reportable", not "beat". Continue training; check the magnitude of `pos_y` (raw 0.755 dominates MAE in mm) |
| Expected delta | 74.94° → 30–50° (after continued training) |
| Effort | 0h (free with training) |
| Risk | None |
| Verification | Forward MAE < 60° after 60 epochs (some learning signal) |

### Factor 14 — Head Pose up MAE° (target `\todo`, current 60.86°, conf-beat 20%)

| Aspect | Value |
|---|---|
| Code location | Same as Factor 13 |
| Root cause | Partial signal (60.86° is better than chance 90°) |
| Proposed change | Continue training. Split loss already in `code/losses.py:1243` |
| Expected delta | 60.86° → 25–45° |
| Effort | 0h |
| Risk | None |
| Verification | Up MAE < 50° after 60 epochs |

### Factor 15 — Head Pose position MAE mm (target `\todo`, current 795.89 mm, conf-beat 20%)

| Aspect | Value |
|---|---|
| Code location | `code/losses.py` pose position loss |
| Root cause | `pos_y` raw 0.755 dominates the metric — likely a target-normalization issue |
| Proposed change | Check target normalization scale. If labels are in mm and predictions in m (or vice versa), there's a 1000× mismatch |
| Expected delta | 795.89 mm → 50–200 mm (after normalization fix) |
| Effort | 4h (debug + fix) |
| Risk | Medium (if pose pos was deliberately left raw) |
| Verification | Position MAE < 300 mm after fix; or document why raw |

### Factor 16 — OKL (3D keypoint) (target n/a, conf-beat n/a)

| Aspect | Value |
|---|---|
| Code location | Not in current eval — `code/quick_eval.py` would need a 3D keypoint path |
| Root cause | No 3D keypoints in IndustReal dataset |
| Proposed change | **De-scope.** Do not claim. OR add a synthetic 3D keypoint eval — but that requires ground truth that doesn't exist |
| Expected delta | None (n/a) |
| Effort | 0h |
| Risk | None (just remove from paper claims) |
| Verification | Remove OKL row from `popw_paper_improved.tex` Table 1 |

### Factor 17 — Error Verification AP (target `\todo`, current 0.0000, conf-beat 8%)

| Aspect | Value |
|---|---|
| Code location | `code/audit_eval.py` error-verification block |
| Root cause | Derived from det + PSR — collapses with them |
| Proposed change | **Gated on Factor 1 + Factor 8.** Cannot fix independently |
| Expected delta | 0.00 → 5–15% (after det + PSR fix) |
| Effort | 0h |
| Risk | None |
| Verification | Verification AP > 0.0 after Factor 1 + 8 |

### Factor 18 — Error Verification F1@0.5 (target `\todo`, current 0.0000, conf-beat 8%)

| Aspect | Value |
|---|---|
| Code location | Same as Factor 17 |
| Proposed change | Gated on Factor 1 + 8 |
| Expected delta | 0.00 → 0.05–0.20 |
| Effort | 0h |
| Risk | None |
| Verification | Verification F1 > 0.05 in 25% restart |

### Factor 19 — Memory safety (target PASS, current mostly OK, **conf-beat 70%**)

| Aspect | Value |
|---|---|
| Code location | `code/train.py:243` pin_memory guard (DONE); `code/train.py:2480-2481` clear_frame_cache (DONE) |
| Root cause | Only literal `del targets` is missing (BLOCKER 4c per CONTEXT_SUMMARY §6) |
| Proposed change A | `code/train.py` after the optimizer.step() call (around line 2825) add: `del targets; torch.cuda.empty_cache() if epoch % 5 == 0 else None` |
| Proposed change B | `code/train.py:2481` add `logger.info(f"[MEM] Cleared FRAME_CACHE at epoch {epoch}: {len(FRAME_CACHE)} entries")` BEFORE the `clear_frame_cache()` call, so we see in train.log whether the call ran |
| Expected delta | Memory stable over 100+ epochs (no OOM); verify-memory PASS |
| Effort | 30 min |
| Risk | Low (del is safe; empty_cache can slow training ~2% if called every step) |
| Verification | `grep "del targets" code/train.py` returns ≥ 1 match; train.log shows `[MEM] Cleared FRAME_CACHE` line |

### Factor 20 — Training stability (NaN/Inf) (target PASS, current guarded, **conf-beat 65%**)

| Aspect | Value |
|---|---|
| Code location | `code/train.py:2821` any-not-isfinite guard; `code/train.py:1471-1472` headpose divzero guard (both DONE) |
| Root cause | Trivial-min risk, not NaN risk. Val loss ≈ 0.19 while metrics are 0 is the danger — guards convert NaN to `1e-4` sentinel but don't prevent the loss-floor trap |
| Proposed change A | Add per-head loss floors in `code/train.py` loss aggregation (around line 2700). If `loss_pose < 0.01` or `loss_act < 0.05` or `loss_psr < 0.05` for 3+ consecutive batches, multiply by 10 |
| Proposed change B | Pass smoke test: 5% subset / 3 epochs must show total loss tracking down AND det score std > 0.05 AND activity predicts > 5 classes AND PSR > 3 unique patterns |
| Expected delta | 65% → 85% (after loss floors + smoke) |
| Effort | 4h |
| Risk | Low (loss floors are hard limits, not gradients) |
| Verification | 5% smoke test PASS; `python code/smoke_test_fixes.py` exit 0 |

---

## 7. Verification protocol per factor

Run this sequence after applying any of the §6 changes. Stop at first failure.

```bash
# Step 1: Static checks (5 min) — catches the cheap BLOCKERs
grep -n "del targets" code/train.py            # ≥ 1 match (Factor 19)
grep -n "STAGE3_WARMUP_EPOCHS" code/config.py  # = 3 (no change)
grep -n "ema_after" code/smoke_test_fixes.py   # > 0 (typo fix verified)
python -c "import torch; ckpt = torch.load('runs/eval_diag_test_50b/eval_crash_recovery.pth', weights_only=False, map_location='cpu'); print('epoch:', ckpt.get('epoch')); print('log_vars:', ckpt.get('criterion', {}).get('log_vars') if 'criterion' in ckpt else 'n/a')"

# Step 2: 5% smoke (30 min) — must pass all 4 conditions
python code/smoke_test_fixes.py 2>&1 | tee /tmp/smoke_$(date +%s).log
# Pass if:
#   - exit code 0
#   - log contains "[SMOKE] PASS" for each head
#   - log_vars not at clamp boundary [-4, 2]

# Step 3: 5% 3-epoch dry-run (2h) — verifies training actually deserializes
bash code/run_10pct_train.sh 2>&1 | tee /tmp/train_5pct_$(date +%s).log
# Pass if all of:
#   - total loss tracks down (epoch 0 → 3: should drop > 0.5)
#   - det score std > 0.05 (i.e. cls head is not flat)
#   - activity predicts > 5 unique classes
#   - PSR produces > 3 unique 11-bit patterns
#   - no NaN/Inf in loss (no [NaN_GUARD] spam)
#   - FRAME_CACHE clear log line visible

# Step 4: 25% restart (12h) — main experiment
bash code/run_restart_25pct.sh 2>&1 | tee /tmp/train_25pct_$(date +%s).log
# Pass if all of:
#   - resume loads crash_recovery.pth without error
#   - all 4 head metrics non-zero by epoch 5
#   - det mAP@0.5 > 5%, activity Top-1 > 10%, PSR F1 > 0.10, pose MAE < 70°

# Step 5: Full eval (30 min)
python code/quick_eval.py --checkpoint runs/eval_diag_test_50b/eval_crash_recovery.pth 2>&1 | tee /tmp/eval_$(date +%s).log
# Pass if:
#   - all 20 factors above the "minimum" row in §1 table
#   - no [EVAL COLLAPSE] warnings
```

---

## 8. Inter-factor dependency graph

```
[F1 Detection mAP@0.5] ──┐
                          ├──► [F17 Error Verif AP] ──► [F18 Error Verif F1]
[F2 mAP@[.5:.95]] ────────┘
                          ▲
[F3 Det per-class AP] ────┘ (gated on F1)

[F4 Activity Top-1] ──┐
[F5 Top-5]            ├──► [F7 Activity mcAP]
[F6 Macro-F1] ────────┘

[F8 PSR F1 vs B2] ──┐
[F9 PSR F1 vs STORM]├──► [F10 PSR POS] ──► [F11 per-comp F1]
                    │                              │
                    └─► [F12 Edit Score (discount)] ┘

[F13-15 Head Pose MAE] — independent (separate loss head)

[F16 OKL 3D] — DE-SCOPED, no dependencies

[F19 Memory] ──► [F20 Training stability] (memory hygiene enables long runs)

[F1 + F8] ──► [F17 + F18 Error Verif]
```

**Critical path:** F19 (memory) → F20 (stability) → F1 (det) → F8 (PSR) → F17+F18 (error verif).
**Independent tracks:** F4-F7 (activity), F13-F15 (pose), F16 (de-scope).
**Fastest wins:** F19 (30 min), F20 (4h), F16 (0h — just remove from paper).

---

## 9. Test sequence with timing (cumulative budget)

| Stage | What | Wall time | Pass criteria | Rollback if fail |
|-------|------|-----------|---------------|-------------------|
| 1. Static | grep + checkpoint load | 5 min | 4 grep hits + epoch 31 | revert `del targets` if NaN appears |
| 2. Smoke | `smoke_test_fixes.py` | 30 min | 4× [SMOKE] PASS | revert STAGE3_WARMUP if IndexError |
| 3. 5% dry-run | `run_10pct_train.sh` | 2h | 5 conditions in §7 step 3 | revert eval threshold if mAP still 0 |
| 4. 25% restart | `run_restart_25pct.sh` | 12h | 4 head metrics non-zero | revert to checkpoint, fix Factor 1+8, retry |
| 5. Full eval | `quick_eval.py` | 30 min | all 20 factors > minimum | done — record final state |

**Total budget: 14h 65m.** Stops at first failure. Stage 1+2 must pass before any GPU time is spent.

---

## 10. Already-applied fixes × 20-factor coverage

The 7 fixes confirmed PASS in `evidence/01_POPW_FINAL_REPORT_v2.md` map to these factors:

| Fix | Location | Factors helped | Status |
|-----|----------|----------------|--------|
| PSR temporal smooth (signed tanh, -1 mask) | `code/losses.py:1162-1164, 1170-1174` | F8, F9, F10, F11, F12 | ✅ Done (PSR head now trains without NaN; per-comp F1 still mostly 0 because of comp-prevalence, not loss) |
| NaN guard (any-not-isfinite) + headpose divzero | `code/train.py:2821, 1471-1472` | F20, F13, F14, F15 | ✅ Done (no NaN in current log; headpose divzero prevented) |
| VideoMAE proj in optimizer | `code/train.py:2315-2350` | F4, F5, F6, F7 (indirect — backbone features) | ✅ Done (param group exists; LR matches head LR) |
| EMA fix (ModelEMA + _get_ema_decay) | `code/validate_checkpoint.py:59-68`, `code/train.py:2305` | F1, F2, F3 (eval uses EMA weights) | ✅ Done (eval now loads EMA state correctly) |
| STAGE3_WARMUP_EPOCHS ramp | `code/train.py:2034, 2138-2148, 2323-2333, 2507-2516` | F4, F5, F6, F7 (activity head LR ramp) | ⚠️ Code wired but no PASS message (Q5 in TASKS.md) |
| pin_memory guard | `code/train.py:243` | F19 | ✅ Done (comment present; not the source of verify-memory FAIL) |
| `del outputs, loss, loss_dict` | `code/train.py:1127, 1146, 1169, 1192, 1379, 1381` | F19, F20 | ✅ Done (only `del targets` still missing) |

**Net:** 6/7 fixes are verified PASS. STAGE3_WARMUP needs a smoke test to confirm. The remaining unfixed items are:
- **`del targets` (BLOCKER 4c)** — single missing line in `code/train.py` (F19)
- **Eval flood threshold (DET_EVAL_SCORE_THRESH=0.01)** — single number in `code/config.py:332` (F1, F2, F3)
- **Cls bias init (`pi=0.01` vs `pi=0.10` doc mismatch)** — `code/model.py:531` (F1)
- **Activity weight (`HEAD_LOSS_WEIGHTS['act']`)** — `code/config.py` (F4-F7)
- **PSR weight (`HEAD_LOSS_WEIGHTS['psr']`)** — `code/config.py` (F8-F12)

Total: 5 single-line or single-number changes to lift aggregate confidence from ~12-18% → ~50-65%.

