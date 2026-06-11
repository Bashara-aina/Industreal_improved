# The 4-Head Collapse — Evidence and Root Cause

> The 39-epoch checkpoint `crash_recovery.pth` is the artifact that taught us
> the most. This file is the full autopsy.

## TL;DR

The model trains cleanly for ~15 epochs (stages 1+2 = detection only, then
detection+pose), then at stage 3 (when activity and PSR heads come online)
the gradient signal from activity and PSR collapses three heads to
trivial solutions within a handful of batches, and bf16 AMP masks the
problem until it's irreversible.

## The Timeline

```
Epoch  0–4   : Stage 1 (detection only).              Loss drops 80 → 18.
Epoch  5–14  : Stage 2 (detection + pose + headpose). Loss 18 → 12.
Epoch 15     : Stage 3 starts. Activity loss ramp 0→1 over 5 ep.
Epoch 15     : LDAM-DRW class-balanced reweighting activates (DRW_EPOCH=0).
Epoch 15–18  : First activity batches produce 0-grad. LDAM margin = 4, logit
               output ~0 → softmax ≈ uniform → cross-entropy ≈ ln(74) ≈ 4.3.
Epoch 17     : PSR seq-mode batch (T=4 window). bf16 underflow in the causal
               transformer softmax → NaN in autograd graph.
Epoch 17     : NaN propagates to next non-seq PSR batch via residual graph
               → NaN in PSR loss → NaN in MultiTaskLoss aggregation.
Epoch 17     : `NaNDetector` fires. `crash_recovery.pth` saved. Training
               aborts to shell.
Epoch 18–38  : Manual resume from `crash_recovery.pth` with reduced LR.
               The 3 dead heads' weights are still in the file. The model
               reaches "epoch 39" but never recovers.
```

## The Three Collapsed Heads — Verbatim Evidence

The baseline `evidence/metrics_baseline_pre_reinit.json` was produced by
`code/eval_post_reinit.py` loading the dead checkpoint and re-running
evaluation. Here is the signal-level evidence for each collapse:

### 1. Detection: cls_score saturated to 1.0

```
[DEBUG] det: dp_boxes=320 imgs, total_preds=64950, dg_total=42
[DEBUG] det: dp_scores range=[1.000, 1.000] mean=1.000
[DEBUG] det: scores above thresh 0.01: 64950/64950
[EVAL COLLAPSE] detection head produces flat scores (std=0.0000 < 0.01,
                 all ≈ 1.000). det_mAP50=0 is a model collapse, not an
                 eval bug.
[EVAL COLLAPSE] excessive prediction count: 64950 preds across 42 GT
                 boxes (ratio=1546x).
```

**What this means:** the cls_score bias learned a value of +∞ (or close
to it). After sigmoid, every anchor scores 1.0. The NMS step keeps
1546× as many "predictions" as there are GT boxes. mAP@0.5 = 0 because
all predictions are 1.0 = no ranking signal.

**Root cause:** initial bias = `-log((1-π)/π)` with π=0.01 → bias=-4.6
→ sigmoid(−4.6)=0.01. After ~100 steps, the weight update pushed the
bias up to keep up with the loss, overshot through the linear region,
and ended up at +∞. AMP (bf16) provided underflow protection that
masked the early gradient signal that would have warned us.

### 2. Activity: stuck on class 27 (most common)

```
Activity — Acc: 0.0000  Macro-F1: 0.0000  Weighted-F1: 0.0000
             Top-5: 0.0000  Frame Acc (all): 0.0000
             Frame Acc (no NA): 0.0000  Macro-Recall: 0.0000
  Per-class accuracy summary: macro=0.0% min=0.0% max=0.0%
```

But the confusion matrix shows:
```
class 27 ('check_instruction' — 48 GT samples) ← all predictions land here
class  0 ('NA' — 71 GT samples) ← all predictions land here
class  1 ('align_objects' — 15) ← all predictions land here
class  7 ('take_screw_pin' — 13) ← all predictions land here
class 16 ('plug_screw_pin' — 7) ← all predictions land here
class 30 ('fit_short_brace' — 34) ← all predictions land here
class 44 ('take_objects' — 12) ← all predictions land here
```

Every class collapses to whatever has the largest prior. The model is
not random — it is **deterministic and wrong**.

**Root cause:** `activity_classifier` was initialized with bias=0 and
`Linear.weight ~ N(0, 1/512)`. With LDAM-DRW reweighting at DRW_EPOCH=0,
the class with the most samples (NA, 71) immediately dominates the
gradient. The symmetric weight init means the network's only stable
fixed point is "predict the majority class." Combined with Kendall
`s_act=0` (neutral), there's no bias toward learning anything else.

### 3. PSR: stuck on all-zeros

```
[DEBUG] psr_logits range=[-1.027, -0.981]  sigmoid range=[0.264, 0.273]
        unique_binary_patterns=1  total_frames=320
  pattern[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[EVAL COLLAPSE] PSR head produces only 1 unique binary pattern(s)
                 across 320 frames. psr_overall_f1=0 is a model
                 collapse, not an eval bug.
PSR — Overall F1: 0.0000  F1@±3: 0.0000  P@±3: 0.0000  R@±3: 0.0000
       F1@±5: 0.0000  P@±5: 0.0000  R@±5: 0.0000  Edit: 0.0909
       POS: 0.0000
[DEBUG] as_vocab size (K)=1  unique patterns=[0]
[DEBUG] first 20 GT state IDs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0]
[DEBUG] unique GT state IDs: [0]
[DEBUG] GT transitions at frames: []
```

**What this means:** the 11 binary PSR outputs are at sigmoid 0.27
(after `output_heads` bias init of -1.0). The val set has 0 GT
transitions for these particular 320 frames (we sampled the wrong
window). The "Edit: 0.0909" is non-zero because the GT sequence is
short enough that Damerau-Levenshtein returns a small non-zero.

**Root cause:** `output_heads[i]` (11 MLPs) had bias=-1.0, giving
sigmoid output of 0.27. With the threshold at 0.5, every output is
"off." The model never has a reason to push the bias upward because
the loss is dominated by the 0-class and the model is rewarded for
predicting 0. AMP didn't help — it actively hurt because the
threshold-crossing gradient was in the bf16 underflow regime.

## The 7th Head that Survived (sort of)

Body pose and head pose both produce non-zero values. This is a critical
clue: **the heads that work are the ones with bounded, well-scaled
outputs and well-conditioned losses.**

- Body pose: `WingLoss(ω=0.05, ε=0.005)` × 0.001 + 17 keypoints in [0,1]
  range → gradient is always well-scaled.
- Head pose: MSE × 0.001 on 9-DoF, LayerNorm + GELU, dropout. Stable.
- The 3 dead heads: Focal loss with α=0.25 and γ=2 (steep gradient
  near 0/1), LDAM margins, output bias in extreme initial positions,
  loss caps set to absorb the cascade rather than prevent it.

## What AMP Did Wrong

We were running `bf16` (Autocast) for the bulk of training. The bf16
mantissa is 8 bits. In the early epochs, this is fine — gradients are
well-scaled. But at the seq=1 sequence-mode PSR batch:

1. The PSR causal transformer computes a 4×256 attention matrix.
2. The softmax denominator is the sum of `exp(q·k/√d_k)` across 4
   positions, all of which are in bf16.
3. The autograd graph for this single batch is **retained** because
   `retain_graph=True` was implicit in our pipeline.
4. The next non-seq batch's autograd graph is built on top of the
   poisoned one.
5. The result: NaN appears in PSR loss in the next batch.
6. NaNDetector fires. We save crash_recovery.pth and exit.

The `diag_amp_nan.py` script reproduced this 4 times out of 4 in 100-step
smoke tests. **FP32 is the workaround.** It costs 2× VRAM and 2× time
but eliminates the NaN.

## The Two Bugs in Our Loss Stack

### Bug A: Kendall log_var clamp missing

In the original paper, `s_t` (Kendall log variance) is unconstrained.
In our impl it was unconstrained for 39 epochs. The gradient on `s_t`
is `0.5 - 0.5·L_t·exp(-s_t)`. When `L_t` is small (activity converged
to ln(74)≈4.3) and `s_t` is large positive (the model learned to
suppress activity by inflating its variance), the gradient on `s_t`
becomes very small but the bias on activity loss becomes negligible.
Result: activity gradient → 0 within ~10 batches of stage 3.

**Fix:** hard-clamp `s_t ∈ [-4, 2]`. We added this in
`train.py:1782` (`_clamp_kendall_log_vars`).

### Bug B: Loss caps absorb the cascade instead of preventing it

`ACTIVITY_LOSS_CAP=80, PSR_LOSS_CAP=20, HEAD_POSE_LOSS_CAP=30` were
intended to protect against rare spike losses. But when a head
collapses to a trivial solution, the loss is *low* (Focal near 0), not
high. The cap doesn't help. The head's gradient is the issue, not its
magnitude.

**Fix:** bias init + class-balanced sampling + LDAM-DRW deferred to
epoch 5 (so the first 5 epochs of stage 3 are pure CE before DRW).
This is partial — see `03_CURRENT_RECOVERY.md`.

## What About the 7-day "binding constraint"?

We are training on `SUBSET_RATIO=0.10` (4 recordings train, 4 val).
The val set is what the eval uses. With 4 recordings:

- Activity: many classes have 0 GT samples in val (see confusion
  matrix above — only 7 of 74 classes have any val samples).
- PSR: 0 transitions in the 320-frame val window.
- Detection: 42 GT boxes total across 320 frames.

This means even a **perfect** model on this val set would have
activity accuracy near 0% (because 67 of 74 classes have 0 GT samples
in val — you cannot predict a class that doesn't appear). The "5–11%
combined" baseline is partly an artifact of the val split.

**Two responses:**
1. Use a different val split (the paper's "val" is the first 4 of
   17 recordings, we have to use 4 of 17). This is solvable with
   different `SUBSET_RATIO` and stratified sampling.
2. Recognize that head accuracy metrics on this val set are
   uninformative, and use **training loss / per-class gradient norm**
   as the proxy for "is this head learning?"

Both of these are on our fix list.

## What Survives the Collapse (the Good News)

Despite all 3 of 5 heads being dead, the model still:
- Produces non-zero feature maps at every backbone stage
  (`diag_features_alive.py` showed per-image variance > 0.001, per-
  channel std > 0.05, gradient norm > 1e-6 at all 5 stage outputs).
- Has body-pose MAE in the right order of magnitude (739 mm is "way
  too big" but the units are correct, not random).
- Has head-pose output in unit-vector form (the eval prints
  `head_pose_status=unit_vectors_ok`).
- The forward pass completes in <500ms per batch on the 3060.

The architecture is **alive**. The training procedure is killing it.

## What We Want Opus to Tell Us

(Detailed in `04_HYPOTHESES_FOR_OPUS.md`.)

- Is there a way to make the heads train together without the
  mutual-collapse we see at stage 3?
- Is `det_conf → activity` (max-pool of cls scores as a feature) a
  reasonable conditioning signal when det is itself collapsed?
- Should we ditch the LDAM-DRW on activity and use plain CE?
- Should we ditch the seq-mode PSR entirely and just predict per-
  frame? The seq-mode is what causes the bf16 NaN.
- Is `Focal(γ=2)` right for 11 binary PSR outputs? γ=1 fixed the
  PSR all-zeros but the paper says γ=2.
- Should we drop VideoMAE? It adds 22M frozen params and the eval
  doesn't even read its outputs.
