# 02 — The Collapse Crisis (Refined with Post-Retrain Data)

> **Purpose:** A precise description of what "3 dead heads" means in this
> codebase, what we know causes it, what we know doesn't, and what
> changed after the 1-epoch reinit retrain.

---

## 1. Definition of "3 dead heads"

The POPW model is a multi-task ConvNeXt-Tiny + FPN backbone with 5 heads
(see `code/model.py`):

| Head          | Output dim          | Loss                                                | Eval metric (val)            |
|---------------|---------------------|-----------------------------------------------------|------------------------------|
| Detection     | (B, n_anchors, 5+C) | Focal (cls) + Smooth-L1 (reg) + Obj BCE             | `det_mAP50`, `det_mAP_50_95` |
| Activity      | (B, 75)             | LDAM-DRW with class-balancing                       | `act_top1`, `act_macro_f1`   |
| PSR           | (B, T, 11)          | Binary cross-entropy per component                  | `psr_overall_f1`, `psr_edit_score` |
| Body pose     | (B, 17, 2)          | Wing loss                                            | `position_MAE_mm`            |
| Head pose     | (B, 9)              | 3× angular-mae + 3× positional-mae                  | `head_pose_angular_MAE_deg`  |
| (Assembly state) | (B, n_states)    | Cross-entropy                                        | `as_f1`                      |
| (Error verify) | (B, 2)             | BCE                                                  | `ev_ap`                      |

**"3 dead heads" = detection, activity, PSR.** Body pose and head pose
were always working (MAE 0.35–0.42 before patches, the ONLY non-zero
metrics on the broken model).

---

## 2. The original (pre-retrain) collapse signature

| Head     | Symptom                                                                                                                                  | Evidence                                                |
|----------|------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| Det      | Flat scores (std=0.0000 < 0.01, all ≈ 1.000), 56418 preds across 42 GT (ratio=1343×)                                                     | `train_v3_25pct_setsid.log` "EVAL COLLAPSE detection"   |
| Det      | bestIoU max = 0.27 across all batches; 0 preds at IoU > 0.5                                                                            | `eval_post_retrain_fp32_20260610_194311/eval.log` DET_PROBE |
| Activity | Only 1–3 / 75 classes predicted; top-1 class had 66.6–100% of frames                                                                    | `train_v3_25pct_setsid.log` "EVAL COLLAPSE activity"    |
| PSR      | Only 1 unique binary pattern across 320 frames (should be ~10–20)                                                                       | `train_v3_25pct_setsid.log` "EVAL COLLAPSE PSR"         |
| PSR      | `psr_overall_f1 = 0` and `psr_edit_score = 0.0909` (the 0.0909 is the no-pattern floor — the eval reports 1-of-1 patterns)              | baseline metrics.json                                    |
| Body pose| WORKING (MAE 0.35–0.42)                                                                                                                 | consistent across all runs                              |
| Head pose| WORKING (forward_angular MAE 64° — high but not degenerate)                                                                              | consistent across all runs                              |
| AS / EV  | `as_f1=0`, `as_map_r=0`, `ev_ap=0`, `ev_f1=0` — these are downstream of det/act so also stuck                                                | train log                                               |
| Efficiency| ALL 7 efficiency metrics NaN (Params, GFLOPs, FPS, FPS streaming, pipeline params, pipeline GFLOPs, pipeline FPS)                       | every eval log                                          |

---

## 3. Root-cause analysis (opus-1 hypotheses, now graded)

| # | Hypothesis                                                                 | Verdict        | Evidence                                                        |
|---|----------------------------------------------------------------------------|----------------|------------------------------------------------------------------|
| 1 | Backbone is dead (NaN/Inf)                                                 | **REJECTED**   | `diag_features_alive.py` shows per-image variance 0.032–0.036 in DET logits after fresh head |
| 2 | AMP fp16 underflow in backbone first layers                                | **CONFIRMED**  | `diag_amp_nan.py` shows first NaN at `backbone.0.conv1.weight`; `diag_amp_2step.py` shows AMP fails at first opt step, FP32 succeeds |
| 3 | Seq-mode autograd leak: model.forward() assumes any 4-D batch divisible by 4 is a sequence | **CONFIRMED** | FIX-3: code change makes forward() only treat `dim()==5` as sequence; AST-parse OK; smoke test passes |
| 4 | Kendall log_var drift to ±inf or 0                                         | **CONFIRMED**  | `train.py` Kendall clamp `[-5, 5]` (was unbounded before); train_digest shows pre-patch `kd_p` reaching -2.7 |
| 5 | `_check_per_class_activity_sanity` numpy AxisError kills run before save   | **CONFIRMED**  | FIX-1: pass 2-D confusion matrix; train_digest shows this happened 2× |
| 6 | Hard-coded `_safe()` zeroing poisoned grads but never restoring             | **CONFIRMED**  | FIX-6: `param.grad.zero_()` instead of `param.grad = None`       |
| 7 | `temporal_smooth` loss in PSR negates the label change                     | **CONFIRMED**  | FIX-7: removed negation                                          |
| 8 | `USE_HEADPOSE_FIM` typo                                                    | **CONFIRMED**  | FIX-8: now `USE_HEADPOSE_FILM`                                   |
| 9 | `OUTPUT_ROOT` config default points to wrong dir, retrain never saves     | **CONFIRMED but not a collapse cause** | train.py saves to `src/runs/full_multi_task_tma_tbank_benchmark/`, not the run's own dir — affects discoverability not training |
| 10| Activity LDAM class-balancing underflows to 0 for rare classes             | **PARTIALLY**  | Class 33 (pre-patch dominant 66.6%) is now balanced but rare classes still under-represented |
| 11| Detection anchors/box-coder misaligned with new backbone stride             | **NEVER INVESTIGATED** | Did not change anchor config in patches; no proof either way |
| 12| `report_per_class_accuracy` requires 2-D, gets 1-D and throws               | **CONFIRMED**  | FIX-2: accept either                                              |
| 13| NaN_GUARD: when loss has NaN, code sets grads to None → optimizer step still commits, polluting state | **CONFIRMED** | FIX-5 + FIX-6: skip-step + zero-grad, never commit a poisoned update |

---

## 4. The post-retrain picture (after 1 epoch of reinit, FP32, bs=2)

What changed:
- **PSR:** `psr_edit_score` went from 0.09 → **0.73** (8×). One of the 11
  components (`comp0`) is now F1=1.0; the other 10 are still 0. So PSR
  found one binary pattern correctly.
- **act_top5:** went from 0.00 → **0.06** (above 1/20 random of 0.05).
- **loss:** went from 227.7 → **72.9** (-68%).
- **det_mAP50:** STILL 0.0. DET_PROBE still shows 12/50 batches with
  "TOTAL COLLAPSE" verdict; best IoU capped at 0.27; scores reach 0.97
  but IoU is 0.
- **act_top1 / act_macro_f1 / act_frame_acc:** STILL 0.0. The activity
  head must be producing all-top-1-class-8 or similar 1-class collapse.
- **body/head pose:** REGRESSED. MAE +11–73%. The reinit included the
  pose MLP, which it probably shouldn't have.

---

## 5. What this tells us

The 1-epoch retrain **confirmed the backbone is alive** (loss halved,
PSR moved) but **did not solve 3 of the 5 critical heads** (det, act,
pose).

**The 3 remaining problems have very different signatures:**

| Head     | Signature                                                                     | Most likely cause                                                                  |
|----------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| **Det**  | Confident scores (up to 0.97), but **bestIoU never > 0.27**                    | Bounding box decoder is misaligned, or the boxes are in the wrong coordinate frame (not in image pixel space). The 12/50 "TOTAL COLLAPSE" batches are not even close to the GT — max IoU = 0.24. **Box post-processing is the prime suspect.** |
| **Act**  | top-1=0, top-5=0.06, collapse to 1 class with 100% of frames (in train val!)    | The fresh reinit head has only seen 1 epoch of gradient updates. Either the **loss is degenerate** (LDAM margin underflow for low-count classes makes the gradient zero for everyone) OR **the ViT attention is collapsing to a single token** OR **the cls_token reinit to zero is killing gradient flow** OR **the backbone features are not informative for the 75-way class split** (unlikely; we have a working head pose). |
| **Pose** | REGRESSED +11–73% MAE. Reinit may have nuked learned pose statistics.          | `_reinit_dead_heads` reinit'd the pose MLP even though pose wasn't dead. **Pose was the only working head; reiniting it was a mistake.** |

---

## 6. The 3 head-specific question marks

### Q1: Why is `det_mAP50` still 0.0?

DET_PROBE shows the model produces ~5000+ detections per batch with
scores 0.01–0.97. But bestIoU never crosses 0.5. The boxes are in the
right ballpark (bestIoU mean 0.05–0.08 means they have SOME overlap with
GT) but they're too small / too far off / in wrong coordinate space.

**Hypotheses (to be tested by opus):**
- A) Detection box post-processing uses `nms` with `iou_thresh=0.5` and
  drops the only good box
- B) Box decoder assumes a stride that doesn't match the FPN level being
  used (e.g., stride=8 anchors at P3 when the model trained at P4)
- C) The box coordinates are in normalized [0,1] space but the eval
  expects pixel space and un-normalizes wrong
- D) The detection targets in the dataset are not in the same format
  the model is trained to predict (anchor-free vs anchor-based mismatch)
- E) The new head's cls_logits are correct but the `obj_score` gate is
  pushing boxes off-grid

### Q2: Why is `act_top1` still 0.0?

The top-5 = 0.06 is barely above random (0.05). The model is essentially
guessing.

**Hypotheses (to be tested by opus):**
- A) **LDAM-DRW class-balanced loss has 0 gradient for low-prevalence
  classes** at the start. The cls_token reinit to 0 may have caused
  output to be ~constant for many steps, then the loss saturated
- B) **Cls-token reinit to 0** destroyed the "what is in this frame?"
  probe. The ViT needs the cls_token to be non-zero to aggregate
  attention meaningfully
- C) **LDAM margin** for the 5 most-prevalent classes is so large that
  the model is forced to predict them for ALL frames (or none, if the
  logits are initialized to equal values)
- D) **The activity head was reinit to fresh weights but the
  `proj_features` was not** (or vice versa), causing a distribution
  mismatch between input and output of the fresh ViT
- E) The class-imbalanced sampler is oversampling rare classes so much
  that the model only sees hard negatives for the first 1000 steps and
  the gradient is dominated by those

### Q3: Why did pose regression get worse?

Pose MAE went UP 11–73% after reinit. The pose head's weights were reset
when they shouldn't have been.

**Hypotheses (to be tested by opus):**
- A) `_reinit_dead_heads` reinit'd the pose MLP because its module name
  matches one of the heuristic patterns (e.g., `pose_mlp` or anything
  in `head_pose_branch`). The intended list is det/act/psr only.
- B) The pose regression is from a DIFFERENT cause: the 1 epoch of
  finetuning after reinit has shifted the backbone features, and the
  (frozen) old pose MLP doesn't match them anymore. (This would mean
  the head was always fragile.)
- C) The pose "regression" is statistical noise (n=200 val, batch=4,
  one-batch fluctuation). Need to re-eval with more batches to confirm.

---

## 7. Why this crisis is harder than a typical NaN

The 3 dead heads each have a DIFFERENT root cause, and the fixes
overlap. You cannot just "lower the learning rate" or "increase batch
size" — these are surgical problems:

- Det: post-processing / decoder alignment
- Act: loss function (LDAM-DRW underflow) + ViT init (cls_token)
- PSR: FIX-4 fixed the gradient; the remaining issue is the 1-of-11
  pattern recognition, which probably needs MORE training, not different
  loss

The 1-epoch retrain showed the model CAN learn (PSR moved) but the
gradient flow to det/act is not enough to escape the trivial solution
in 1 epoch. **More epochs may help, or may just memorize the 1
"always-on" PSR pattern.**

---

## 8. What we know is NOT the problem (ruled out)

- ❌ Backbone is dead
- ❌ Data is corrupt (eval was on val split, not train; ran 200 batches
  with no NaN/Inf)
- ❌ LR is too high (5e-3 is normal for Adam + reinit)
- ❌ BatchNorm running stats stale (FP32 forward+backward is clean; BN
  is in eval mode for val)
- ❌ Optimizer state poisoned (cls_loss spike was transient; recovered
  by step 96; see `04_HYPOTHESES_FOR_OPUS.md` §"Ruled out")
- ❌ Wrong train.py entrypoint (confirmed correct via head 2-line
  tracing in `train.log`)
- ❌ Dataset labels wrong (pose MAE was 0.35–0.42 before any patches;
  labels are fine)
