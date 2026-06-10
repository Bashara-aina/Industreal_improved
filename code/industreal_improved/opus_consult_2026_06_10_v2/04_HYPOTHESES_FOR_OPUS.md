# 04 — Hypotheses for Opus

> **Purpose:** A prioritized list of hypotheses for the opus to investigate.
> Each hypothesis is paired with the existing evidence (so opus can
> build on what we already know) and the experiment needed to
> confirm/deny it.

---

## Priority 0 (Ruled out — do NOT re-investigate)

These were investigated in opus 1 and verified dead. The opus should
treat them as background, not re-open the investigation.

| # | Hypothesis                                                        | Evidence                                                                                                          |
|---|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| R1| Backbone is dead (NaN/Inf in FPN features)                         | `code/diag_features_alive.py`: per-image variance 0.032–0.036 in DET logits after fresh head                     |
| R2| AMP fp16 underflow in backbone first layers                       | `code/diag_amp_nan.py`: first NaN at `backbone.0.conv1.weight`. `code/diag_amp_2step.py`: AMP fails step 1, FP32 succeeds |
| R3| Seq-mode autograd leak                                             | FIX-3 makes forward() only treat `dim()==5` as sequence. AST-parse OK. Smoke test passes.                         |
| R4| Kendall log_var drift                                              | `code/train.py` clamps `[-5, 5]` (was unbounded). Pre-patch `kd_p` reached -2.7.                                   |
| R5| `_check_per_class_activity_sanity` crashes before save             | FIX-1: pass 2-D confusion, wrap in try/except. train_digest shows this crashed 2× before fix.                     |
| R6| NaN_GUARD zeros grads but commits optimizer step                   | FIX-5 + FIX-6: skip-step + zero-grad. Verified in losses.py.                                                       |
| R7| Temporal smooth loss negates label change                          | FIX-7: removed negation. Verified in losses.py.                                                                   |
| R8| `USE_HEADPOSE_FIM` typo                                            | FIX-8: now `USE_HEADPOSE_FILM`. Verified.                                                                          |
| R9| Adam m/v optimizer state resume spike is a transient              | Live log: cls_loss c=19.8M at step 28, c=2.85 at step 96, c=0–5 from step 100+. Kill-criteria: "monotonically INCREASING past step 200, or c > 10x baseline AND not declining" (saved as memory `feedback_optimizer_state_resume_spike.md`) |

---

## Priority 1: Detection `det_mAP50` = 0.0 despite confident scores

### H1.1: Box decoder stride mismatch

**Claim:** The box decoder is decoding from the wrong FPN level. With
P2/P3/P4 strides 4/8/16/32, the model needs to pick the right level per
anchor. If the level assignment is wrong, the boxes are decoded at the
wrong scale.

**Evidence to look for in `code/model.py`:**
- The DetectionHead class's `forward()` — which FPN level it picks
- The anchor configuration in `code/config.py` (`ANCHOR_SCALES`,
  `ANCHOR_RATIOS`, `FPN_LEVELS`)
- The `decode_boxes` function (or similar) — is it stride-aware?

**Experiment to run:**
1. Save 1 val batch with GT boxes
2. Run model.forward() in eval mode
3. For each FPN level, dump the raw `reg` output + the decoded boxes
4. Compare decoded boxes to GT for each level
5. The level that produces boxes closest to GT is the "right" one

**Files:** `code/model.py` (DetectionHead.forward + decode), `code/config.py` (anchor config)

### H1.2: Box coordinates in wrong frame

**Claim:** The model predicts boxes in normalized [0, 1] space, but the
eval un-normalizes wrong (or vice versa). The boxes end up at the right
relative position but at the wrong absolute scale.

**Evidence:** DET_PROBE shows `bestIoU_max = 0.24`, `bestIoU_mean = 0.06`.
A 0.24 IoU means the boxes have SOME overlap with GT but are too
small/offset. This is consistent with a 2-3× scale error.

**Experiment:** Save 1 val image + its GT boxes + the predicted boxes,
plot all on the same image. Visually inspect.

**Files:** `code/evaluation/evaluate.py` (detection eval section), `code/model.py` (box decode)

### H1.3: NMS too aggressive

**Claim:** The post-processing uses NMS with `iou_thresh=0.5` and drops
the only good box. The DET_PROBE shows `preds>0.5 = 4036` per batch
(that's a lot of conf boxes), but `bestIoU>0.5 = 0`.

**Experiment:** Run the eval with `nms_iou_thresh=0.9` instead of 0.5
and see if det_mAP50 jumps. If yes → NMS is the bug.

**Files:** `code/evaluation/evaluate.py` (NMS call)

### H1.4: Anchor format mismatch

**Claim:** The dataset produces GT boxes in `(x, y, w, h)` (cxcywh)
format but the model is trained to predict `(x1, y1, x2, y2)` (xyxy)
or vice versa. The loss is computed wrong, the boxes are decoded wrong,
and they never line up with GT.

**Evidence:** `code/data/industreal_dataset.py` (line ~50–200) — how
GT boxes are stored. `code/training/losses.py` (detection loss) — what
format it expects.

**Experiment:** Add a print statement in the detection loss that dumps
1 example's GT format and 1 example's prediction format. Compare.

### H1.5: Detection targets are in a different format (anchor-free vs anchor-based)

**Claim:** The model is anchor-free (like FCOS) but the dataset
provides anchor-based targets (like RetinaNet). The loss is computed
on misaligned pairs.

**Evidence:** `code/model.py` DetectionHead class — does it have
`num_anchors` (anchor-based) or is it dense per-pixel (anchor-free)?

---

## Priority 2: Activity `act_top1` = 0.0

### H2.1: LDAM-DRW margin underflow

**Claim:** For 75 classes with the LDAM-DRW class-balanced loss, the
margin for low-prevalence classes is so large that the gradient is
numerically zero for those classes. The model only learns the top-K
classes; the rest are stuck at uniform logits.

**Evidence:** `code/training/losses.py` (LDAMLoss class) — look for
the margin formula `m_i = max(0, margin_const * (1/n_i)^(1/4))` and
check if `m_i` overflows for n_i = 5 (rare class).

**Experiment:** Compute `m_i` for all 75 classes, print distribution.
If any m_i > 0.5 (i.e., the model is asked to push logit > 0.5
relative to the next class for a low-prevalence class), the loss is
degenerate.

### H2.2: cls_token reinit to 0 kills gradient flow

**Claim:** `_reinit_dead_heads` reinit's the `cls_token` Parameter to
0. With cls_token=0, the ViT's first attention layer has 0 input
contribution, and the gradient w.r.t. `cls_token` is the dot product of
the gradient w.r.t. the ViT's output (which is small because the
backbone features dominate). Result: the activity head gets almost
no learning signal.

**Evidence:** Look at `code/train.py` `_reinit_dead_heads` — is
`cls_token.zero_()` called? `code/models/model.py` ActivityHead class
— how is cls_token used in forward?

**Experiment:** Initialize `cls_token` to `torch.randn_like(cls_token) * 0.02`
instead of 0, re-run smoke, check if activity logits have nonzero
variance at step 10.

### H2.3: ViT attention collapse

**Claim:** With a fresh 3-layer TransformerEncoder + cls_token=0, all
3 attention layers collapse to uniform attention (each head attends
equally to all tokens). The output is then a constant regardless of
input. After 1 epoch of training, the attention weights haven't moved
much because the gradient is dominated by the LDAM-DRW margin.

**Evidence:** `code/models/model.py` ActivityHead — print attention
weights at step 0 and step 50. Are they all `1/T`?

**Experiment:** Save attention weights from the activity ViT at eval
time, plot the distribution. If uniform → confirmed.

### H2.4: Class-imbalanced sampler oversamples rare classes

**Claim:** The training sampler (in `code/data/industreal_dataset.py`)
uses a class-balanced sampler that oversamples rare classes so much
that 80% of training samples are from the bottom 20% of classes. The
model only learns the hard negatives, can't escape the trivial
solution.

**Evidence:** `code/data/industreal_dataset.py` (sampler logic) +
`code/training/train.py` (how samples are drawn).

**Experiment:** Set `class_balanced_sampler=False`, retrain 1 epoch,
check if act_top5 moves to >0.1. If yes → sampler is the bug.

### H2.5: Fresh-init proj_features distribution mismatch

**Claim:** `proj_features` (Linear that maps backbone features to
ViT-input space) was reinit'd fresh, but the backbone features are
in a different distribution than what the new Linear was initialized
to expect (default Linear init = `Uniform(-1/sqrt(d), 1/sqrt(d))` for
both weight and bias). After 1 epoch, the Linear hasn't had time to
adapt to the backbone's actual feature distribution.

**Evidence:** `code/train.py` `_reinit_dead_heads` — does it reinit
`proj_features`?

**Experiment:** Try `--reinit-heads-with-bias=match-backbone-std` (a
hypothetical flag) that reinit's the Linear to have the same
input-mean and input-std as the backbone. Compare 1-epoch metrics.

---

## Priority 3: Pose regression (+11–73% MAE)

### H3.1: `_reinit_dead_heads` over-included pose tensors

**Claim:** The "dead heads" heuristic matched pose-related tensors
(e.g., anything in `head_pose_branch` module) and reinit'd them.
Pose was the only working head; reiniting it was a mistake.

**Evidence:** `code/train.py` `_reinit_dead_heads` — print the list
of 169 tensors reinit'd. Filter for `pose` in the name. How many
match?

**Experiment:** Add a `--dry-run-reinit` flag that prints the 169
tensor names without actually reiniting. Confirm whether pose tensors
are in the list.

### H3.2: 1-epoch backbone shift

**Claim:** The pose MAE regression is because the 1 epoch of finetuning
shifted the backbone features, and the (now-reinit'd) pose MLP is
still learning the new feature distribution. With more epochs, it
would recover.

**Evidence:** Limited. Need to compare pose MAE at epoch 43 (before
retrain) vs epoch 44 (after retrain) vs a future epoch 45+.

**Experiment:** Run 1 more epoch of training (no reinit) and re-eval.
If pose MAE recovers → confirmed (just need more training).

### H3.3: Statistical noise (n=200)

**Claim:** The pose MAE increase is within 1-σ of the 200-sample val
distribution. Pose was 0.35–0.42 before; now it's 0.40–0.50. Both are
"OK but not great" — within the noise floor.

**Evidence:** Pose MAE std (from pre-patch train log) was 0.40, so
±0.07 is 1-σ. Post-retrain is 0.42, well within 1-σ.

**Experiment:** Re-eval with `MAX_BATCHES=200` (800 samples) instead
of 50 (200 samples). If pose MAE converges back to 0.35–0.42 → confirmed
noise.

---

## Priority 4: Other observations

### H4.1: Efficiency metrics still NaN

**Claim:** The efficiency profile (Params, GFLOPs, FPS, etc.) is
NaN even after patches. This is a separate bug from the 3 dead heads.

**Evidence:** Every eval log since 2026-05-27 has `eff_params_m = nan`
in the metric dump.

**Likely cause:** The efficiency profiler uses `thop` or `fvcore` and
crashes silently for one of the heads (probably the transformer
encoder in activity head). The `_print_single_run_results` call at the
end of `eval_post_reinit.py` raises `KeyError: 'eff_trainable_params_m'`
because the NaN was filtered out.

**Experiment:** Run efficiency profile manually in a Python REPL and
see which call fails.

### H4.2: `LDAMLoss.set_class_counts: got 75 entries but num_classes=74`

**Claim:** There's a class-count mismatch (75 vs 74). Either the
class_counts tensor has 1 extra entry or the model has 1 too few
output classes.

**Evidence:** Eval log line 4. Consistent across multiple runs.

**Likely cause:** Background class (or similar) is double-counted in
class_counts but not in the model's output dim. The `set_class_counts`
handles it ("margins/weights are re-aligned to the logits width at
forward time") but the warning is a smell.

**Experiment:** Check `code/data/industreal_dataset.py` class_counts
construction. Check `code/config.py` `NUM_ACTIVITY_CLASSES`.

### H4.3: combined = 0.1116 is suspiciously stable

**Claim:** The `combined` metric (the training scheduler's stop metric)
is exactly 0.1116 at the new best. The baseline (epoch 42 best) was
also 0.1116. This suggests the combined metric is dominated by a
constant term (probably 1 - position_MAE / 1000 or similar) that
doesn't change.

**Evidence:** Pre-patch and post-patch combined = 0.1116.

**Experiment:** Check `code/evaluation/evaluate.py` combined
computation. What are the 4 components? What are their weights?

---

## What we need from the opus

1. **Validate the priority order.** Is det really #1? Or should we
   fix activity first because it's "easier"?

2. **Refine the hypotheses.** Add the hypotheses we missed. Reject
   the ones that are clearly not the issue.

3. **Propose a 9th patch + 1 more retrain.** The patch should be the
   minimum-change fix for Priority 1 (det). The retrain should be
   3-5 epochs with a more conservative reinit list (no pose).

4. **Estimate the recovery ceiling.** With all 4 problems fixed
   (det + act + psr + pose), what's the realistic val metric range?
   Baseline expectations for a ConvNeXt-Tiny + 75-class activity on
   IndustReal val: act_top1 ~ 0.3–0.5? det_mAP50 ~ 0.2–0.4?

5. **Identify any structural problems we missed.** Are there other
   bugs in `code/model.py` or `code/losses.py` that we haven't
   surfaced yet?
