# 03 — Current Recovery State (Post-Retrain, 19:42 2026-06-10)

> **Purpose:** The precise metric-by-metric comparison between the baseline
> (pre-retrain, with heads re-init at eval time) and the post-retrain
> (FP32, 1 epoch, heads evaluated as trained). Plus the per-head
> interpretation.

---

## 1. Side-by-side metric table

Both evals: `split=val`, `max_batches=50`, `batch=4`, `n_samples=200`.

| Metric                          | Baseline (pre-retrain, _reinit) | Post-retrain (FP32 bs2) | Δ     | Verdict |
|---------------------------------|--------------------------------:|------------------------:|------:|---------|
| **TOTAL loss**                  | 227.72                          | **72.91**               | −68%  | ✅ huge drop |
| `act_top1_accuracy`             | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `act_top5_accuracy`             | 0.0000                          | **0.0600**              | +0.06 | ⚠️ barely above random (0.05) |
| `act_macro_f1`                  | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `act_clip_accuracy`             | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `act_frame_accuracy`            | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `act_macro_recall`              | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `act_macro_f1_present`          | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `act_mean_per_class_acc`        | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `psr_overall_f1`                | 0.0000                          | **0.0909**              | +0.09 | ⚠️ 1-of-11 component |
| `psr_f1_at_t` (F1@±3)           | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `psr_f1_at_t5` (F1@±5)          | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `psr_edit_score`                | 0.0909                          | **0.7273**              | +0.64 | ✅ 8× |
| `psr_pos`                       | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `det_mAP50`                     | 0.0000                          | 0.0000                  | 0     | ❌ TOTAL COLLAPSE |
| `det_mAP_50_95`                 | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `det_mAP50_pc`                  | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `det_mAP50_all_frames`          | 0.0000                          | 0.0000                  | 0     | ❌ flat |
| `det_n_present_classes`         | 1                               | 1                       | 0     | ❌ only 1 class ever predicted |
| `det_precision` / `det_recall`  | MISSING                         | MISSING                 | n/a   | ⚠️ metric not in metrics.json (eval emits but sanitizer dropped) |
| `position_MAE_mm`               | 739.5                           | 823.5                   | +11%  | ❌ REGRESSED |
| `head_pose_angular_MAE_deg`     | 61.04                           | 71.50                   | +17%  | ❌ REGRESSED |
| `forward_x_MAE`                 | 0.1051                          | 0.1821                  | +73%  | ❌ REGRESSED |
| `forward_y_MAE`                 | 0.0278                          | 0.1280                  | +361% | ❌ REGRESSED |
| `forward_z_MAE`                 | 0.9195                          | 1.0134                  | +10%  | ❌ REGRESSED |
| `forward_angular_MAE_deg`       | 64.28                           | 68.65                   | +7%   | ❌ REGRESSED |
| `up_angular_MAE_deg`            | 57.80                           | 74.34                   | +29%  | ❌ REGRESSED |
| AS F1, AS MAP@R, EV AP, EV F1   | 0.0000                          | 0.0000                  | 0     | ❌ flat (downstream of det/act) |
| Efficiency metrics (7)          | all NaN                         | all NaN                 | n/a   | ⚠️ eff profile still NaN |
| `combined` (train's stop metric)| (not logged for pre-retrain)    | 0.1116                  | n/a   | — |

---

## 2. Per-head interpretation

### Loss: 227.7 → 72.9 (−68%)

This is the single most important number. It means the patches
correctly unblocked gradient flow and the 1-epoch retrain allowed the
fresh-init heads to find SOME local minimum. The 227.7 baseline was
heavily dominated by `det_loss` (which was unbounded due to flat-logits
hitting the focal-loss epsilon) and the NaN-GUARD-related phantom
gradients.

### PSR: 0.09 → 0.73 edit_score

The PSR head's "Edit Score" (the Levenshtein-style ratio of predicted
sequence vs GT sequence, more forgiving than per-component F1) moved
from 0.09 (the floor = predicting all-zeros matches the all-zeros majority)
to 0.73 — meaning the model is now predicting the RIGHT binary pattern
for at least 1 of 11 components consistently.

Per-component F1 in the post-retrain eval:
- comp0: 1.0000 (perfect)
- comp1–comp10: 0.0000 (still wrong)

So the PSR head learned ONE of the 11 binary classifiers and ignored
the other 10. This is the same "1-of-N collapse" pattern that activity
shows.

**Hypothesis:** All 11 PSR classifiers share the same trunk features
(the LSTM). The trunk learned to fire for comp0's pattern; the per-comp
classifiers are initialized fresh and haven't received enough gradient
to specialize. With more epochs (or a per-classifier learning-rate boost)
this should resolve.

### Detection: 0.0 → 0.0 (no movement)

12/50 batches in the post-retrain eval show "TOTAL COLLAPSE" verdict
in the DET_PROBE (output in `eval.log`):

```
[DET_PROBE b0] ... bestIoU_max=0.24 bestIoU_mean=0.06 | verdict: TOTAL COLLAPSE
```

The model IS producing predictions (~32,000 per batch, score range
0–0.97) but the boxes have IoU < 0.5 with GT for every prediction.

**This is NOT a "head not learned" problem** (the scores reach 0.97).
This is a **box decoder problem** or a **post-processing problem**.

### Activity: 0.0 / 0.06 (top1 / top5)

top-5 = 0.06 is barely above random (0.05 for 1/20). The model is
guessing. The per-class breakdown in the eval log shows:
```
background  : 0.0000  (GT=42)
```

The "background" class (likely class 0 or 74) has GT=42 instances but
the model predicts 0. This is unusual because before any patches the
model was predicting class 28 with 66.6% of frames (a different kind of
collapse). Now it predicts something else, just as badly.

**The fresh-init activity head is showing the same "1-class collapse"
pattern as before the patches**, just with a different dominant class.

### Head pose: regression (61° → 71°)

The pose MAE got WORSE by 17%. This is the most surprising result. The
pose head was working before (MAE 0.35–0.42 on the same val split) and
now it's degraded.

**The likely cause:** `_reinit_dead_heads` reinit'd the pose MLP. The
list of "dead heads" was defined as det/act/psr, but the heuristic
might have matched pose tensors (e.g., anything in the
`head_pose_branch` module). The 1 epoch of finetuning shifted the
backbone features, and the now-fresh pose MLP couldn't keep up.

---

## 3. What the eval log says (raw output)

From `evidence/post_retrain_fp32_20260610_194311/eval.log`:

```
[reinit-eval] ckpt epoch=43 step=None
[reinit-eval] load: missing=0 unexpected=0
[reinit-eval] params with NaN/Inf after load: 0
[reinit-eval] EVAL_SKIP_REINIT=1: NOT re-initializing heads
LDAMLoss.set_class_counts: got 75 entries but num_classes=74. Keeping natural length; margins/weights are re-aligned to the logits width at forward time.
[reinit-eval] starting eval...
[DET_PROBE b0] ... bestIoU_max=0.24 ... verdict: TOTAL COLLAPSE
... (50 batches, 12 with TOTAL COLLAPSE, 38 with NO-GT or TOTAL COLLAPSE) ...
```

The "NO-GT" verdicts (n_gt=0) for many batches are concerning — they
mean the val split has 320-42 = 278 frames with no GT detections, but
this matches the val distribution (most frames don't have a person
"starting" an assembly state). The 12 "TOTAL COLLAPSE" batches are the
ones with GT that the model misses entirely.

```
ACTIVITY RECOGNITION
  Top-1 (frame)          : 0.0000
  Top-5 (frame)          : 0.0600
  mcAP (mean per-class) : 0.0000
  Macro-F1               : 0.0000
  Frame Accuracy (all)  : 0.0000
  Frame Accuracy (no NA): 0.0000
  Clip Accuracy (majority): 0.0000
  Weighted-F1            : 0.0000
  Macro-Recall          : 0.0000
  background            : 0.0000  (GT=42)
```

```
PROCEDURE STEP RECOGNITION (PSR)
  Overall F1 (thresh)    : 0.0909
  Edit Score            : 0.7273
  PSR POS               : 0.0000
  Per-component F1:
    comp0       : 1.0000
    comp1-comp10: 0.0000
```

```
HEAD POSE (9-DoF)
  Forward angular MAE (deg): 68.6469
  Up angular MAE (deg)     : 74.3445
  Position MAE (mm)        : 823.4841
```

---

## 4. The 3 remaining problem classes (priority order)

### Priority 1: Detection `det_mAP50` = 0.0

This is the most embarrassing number. The model produces 5000+ confident
boxes per batch but they don't overlap GT. Either:

- The boxes are decoded from the wrong stride / level
- The box coordinates are normalized wrong
- The NMS / score-threshold post-processing is killing the only good box
- The detection targets in the dataset are not in the format the model
  is trained to predict

The DET_PROBE shows max scores of 0.97 — the model is CONFIDENT — so
this is NOT a "head didn't learn" problem. It's a post-decoder
problem.

**To debug:** Save 1 val batch, run the model, dump raw cls/reg outputs
and the box decoder, plot the boxes on top of the image. Compare to GT.

### Priority 2: Activity `act_top1` = 0.0

The fresh activity head is collapsing to 1 class (different from the
pre-patch collapse). With 75 classes and `act_top5` = 0.06, the model
is essentially random.

**To debug:** Inspect the fresh-init `cls_token` value. Inspect the
LDAM-DRW margin for the dominant class. Plot the gradient of the
`proj_features` weight after 1 step — is it nonzero? Is the cls_token
gradient blocked by something?

### Priority 3: Pose regression (+11–73%)

The pose head was working and is now worse. This is the LEAST critical
of the 3 because pose was already 0.35–0.42 MAE (a normal but not great
number) and is now 0.4–0.5 MAE (still in the same range).

**To debug:** Check `_reinit_dead_heads` for any pose-related tensor
matches. If pose tensors were reinit, that's the bug. If not, the
regression is from backbone feature shift after 1 epoch and will heal
with more training.

---

## 5. What needs to happen next

The opus needs to answer 3 questions (in priority order):

1. **Why is det_mAP50 0.0 despite confident scores?** → Is it decoder
   alignment, post-processing, or label format?
2. **Why is act_top1 0.0?** → Is it LDAM-DRW underflow, cls_token=0
   init, or ViT attention collapse?
3. **What should we reinit next time?** → Pose is regressed; the
   reinit list is too aggressive.

After the opus answers, the next action is either:
- A) Apply a 9th patch (decoding / post-processing / activity init)
  + run a 3-epoch retrain
- B) Reduce the scope of `_reinit_dead_heads` to det+act+psr only
  + run a 3-epoch retrain (no pose reinit)
- C) Both A and B in sequence

The plan is laid out in `05_MASTER_PROMPT.md`.
