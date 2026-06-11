# 7 Specific Hypotheses for Opus

> These are the questions that, if Opus answers them with concrete
> code-level guidance, will let us move from "4 heads training" to
> "4 heads training well." Each is grounded in evidence from the
> collapse diagnostics and the recovery retrain.
>
> For each, we state: (a) the hypothesis in one sentence, (b) the
> evidence we have, (c) the code locations it touches, (d) the
> experiment we would run, and (e) what success looks like.
>
> We are not asking Opus to "be creative." We are asking for a
> concrete decision: do this, or do that, with code.

---

## H1. Kendall Reweighting vs GradNorm vs PCGrad vs Plain Sum

**Hypothesis.** Kendall homoscedastic uncertainty (`exp(-s_t)·L_t + s_t`)
is the wrong reweighting for 5 heads at 5% subset, because the
"reweighting" itself is a free parameter that the model can exploit to
silence heads. A norm-based balancer (GradNorm) or a conflict-resolving
balancer (PCGrad) would be more robust.

**Evidence.** With unconstrained `s_t`, after 5 epochs of stage 3, `s_act`
drifted to +2.4 and `s_psr` to -1.8. After we clamped to `[-4, 2]`,
`s_t` is bounded, but the per-step reweighting still has 64× of dynamic
range, and the activity head's gradient is the one most likely to get
silenced because the 74-class cross-entropy has high variance.

**Code locations.**
- `losses.py:816` — `MultiTaskLoss` (Kendall).
- `train.py:920` — `_clamp_kendall_log_vars` is called every batch.
- `losses.py:603` — `ClassBalancedFocalLoss` (LDAM-DRW) — uses its own
  per-class weights, not Kendall.

**Experiment.** Train 2 versions, both at 5% subset, both FP32, both
with reinit:
1. **A:** current Kendall + clamp.
2. **B:** GradNorm (Chen et al. 2018) with `α=0.5` weight on the
   balancing term, no clamp needed.
Run for 5 epochs. Compare per-head gradient norm at epoch 1 and
epoch 5, and per-head eval metric. If GradNorm produces a less
uneven gradient distribution and equal-or-better eval, swap.

**Success metric.** Per-head gradient norm at epoch 5 has ratio
max/min < 5× (currently this ratio is ~50× in favor of pose).

**Opus's call:** do we keep Kendall-with-clamp, swap to GradNorm,
swap to PCGrad, or just use a plain sum with hand-tuned per-head
weights? Show us the implementation of your recommended approach.

---

## H2. The Staged Curriculum is the Wrong Inductive Bias for 5 Heads

**Hypothesis.** The 3-stage curriculum (det only → det+pose → all 5
heads) is what causes the cascade: stages 1 and 2 train the backbone
to optimize a 2-head loss, then at stage 3 we suddenly ask the same
backbone to satisfy a 5-head loss. The backbone's representations are
already shaped for the 2-head regime, and the new heads can't push
back. A joint training from epoch 0 (with lower per-head LR for the
first 2 epochs) would be more stable.

**Evidence.** Stages 1+2 train cleanly for 15 epochs. At stage 3 (when
activity and PSR come online), 3 heads collapse within ~5 batches.
The backbone features are alive throughout (per
`diag_features_alive.py`), so the cascade is not a backbone-capacity
problem — it's a loss-landscape problem. The simplest explanation:
the new heads' gradients push the backbone in directions that
destroy the stage-1/2 heads' solutions.

**Code locations.**
- `train.py:788` — `train_one_epoch`, the staged logic is in
  `main` around `train.py:3000-3200` (stages toggle which heads
  get gradients).
- `config.py` — `TRAIN_DET=True, TRAIN_HEAD_POSE=True, TRAIN_ACT=True,
  TRAIN_PSR=True` flags, plus the stage schedule.

**Experiment.** Train 2 versions, both at 5% subset, both FP32,
both with reinit:
1. **A:** current 3-stage curriculum.
2. **B:** joint training, all heads active from epoch 0, with
   per-head LR ramp:
   - epoch 0: det,pose at 1×LR; act,psr at 0.1×LR.
   - epoch 2: act,psr at 0.5×LR.
   - epoch 5: all at 1×LR.
   No stage boundaries.

Run for 5 epochs. Compare eval metrics. If B is equal-or-better on
det,pose (the survivors) and **better** on act,psr (the dead
heads), swap.

**Success metric.** `act_top5_accuracy` > 0.05 at epoch 5 (vs
Kendall-and-staged baseline, which we expect to be ~0).

**Opus's call:** is the curriculum salvageable by adjusting the
stage boundaries, or should we drop staging entirely? Show the
code.

---

## H3. Drop the Sequence-Mode PSR Causal Transformer

**Hypothesis.** The PSR head's sequence-mode (T=4 window, causal
transformer, 1/10 of batches) is a 2× memory cost, an AMP-NaN
source, and provides < 0.05 F1 improvement. A per-frame MLP with
the 11 binary outputs is sufficient for the PSR task; the
"temporal context" the causal transformer was supposed to learn
is already captured by the TCN and the 2×ViT in the activity head.

**Evidence.** The PSR head's eval metric on the dead checkpoint is
0.0 across the board (F1, POS, Edit). The seq-mode batches are the
ones that caused the bf16 NaN. The detach-fix is a band-aid. The
PSR F1 target (0.731 for PSRT-B2) is not fundamentally a temporal
modeling problem; the components are mostly visible per frame.

**Code locations.**
- `model.py:1399` — `PSRHead` with `PSR_SEQUENCE_LENGTH=4`.
- `losses.py` — `PSRFocalLoss` is per-frame; sequence mode is
  purely architectural.
- `config.py` — `PSR_SEQUENCE_LENGTH`, `PSR_SEQ_BATCH_RATIO`.

**Experiment.** Train 2 versions, both at 5% subset, both FP32,
both with reinit:
1. **A:** current PSR with seq-mode.
2. **B:** PSR with seq-mode disabled, T=1 always, no causal
   transformer. Just the per-frame MLP + 11 binary outputs.

Run for 5 epochs. Compare `psr_overall_f1` at epoch 5. If B is
equal-or-better, the causal transformer was never helping.

**Success metric.** `psr_overall_f1` ≥ 0.05 at epoch 5 in B (vs
~0 in A under the same conditions).

**Opus's call:** is the causal transformer worth keeping? If yes,
show us how to train it without AMP-NaN. If no, show us the
PSRHead simplification.

---

## H4. The VideoMAE Stream is Dead Weight

**Hypothesis.** The 22M-param frozen VideoMAE-Small stream (384-D
output fed into the activity head) is a 22M-param tax that returns
zero F1. The eval does not even read the VideoMAE output. Removing
it would free VRAM, speed up training, and let us run batch=4
in FP32 instead of batch=2.

**Evidence.** `model.py:23-31` shows VideoMAE is loaded but the
eval log shows `eff_params_m = nan` (i.e., the eval doesn't count
it). The activity head's collapse was caused by class imbalance +
LDAM-DRW + bias init, not by a missing motion feature. We have
no ablation showing VideoMAE helps.

**Code locations.**
- `model.py:23-31` — `VideoMAEStream` class.
- `config.py` — `USE_VIDEOMAE=True` flag (or similar).
- `model.py:1192` — `ActivityHead` constructor takes the
  VideoMAE feature.

**Experiment.** Train 2 versions:
1. **A:** current with VideoMAE.
2. **B:** VideoMAE disabled, activity head input dimension drops
   from 1048 to 664 (1048 - 384 = 664).

Run for 5 epochs. Compare `act_top5_accuracy` and VRAM usage.
If B is equal-or-better and uses less VRAM, drop VideoMAE.

**Success metric.** VRAM at batch=4 in FP32: < 11 GB (vs OOM at
12 GB with VideoMAE), or batch=8 stable.

**Opus's call:** drop it, keep it, or replace it with something
that actually adds signal (e.g., a 1-D temporal difference of
RGB)?

---

## H5. Class-Balanced Activity Sampling + Drop LDAM-DRW

**Hypothesis.** LDAM-DRW with `DRW_EPOCH=0` (deferred-reweighting
starts immediately) is the wrong rebalance for activity at 5%
subset. The model has too few samples per class to learn meaningful
LDAM margins, and the reweighting amplifies the gradient on the
classes the model is least likely to predict. Plain
class-balanced sampling (WeightedRandomSampler per epoch) with
plain cross-entropy would be more stable.

**Evidence.** With 74 classes and ~3,000 training frames
(`evidence/metrics_baseline_pre_reinit.json`), we have ~40 samples
per class on average. 7 classes dominate with 100+ samples each.
LDAM margins of 4.0 require accurate class-conditional statistics
that we don't have at 40 samples per class. The activity head
collapses to class 27 (the most common), and the LDAM-DRW
reweighting amplifies this collapse by making class 27's gradient
even larger.

**Code locations.**
- `losses.py:486` — `LDAMLoss`.
- `losses.py:603` — `ClassBalancedFocalLoss` (the LDAM-DRW
  wrapper).
- `config.py` — `LDAM_MARGIN`, `DRW_EPOCH`, `ACT_USE_LDAM`.

**Experiment.** Train 2 versions, both with reinit, both FP32,
both with `--reinit-heads`:
1. **A:** current LDAM-DRW with `DRW_EPOCH=0`.
2. **B:** drop LDAM-DRW, use `WeightedRandomSampler` with
   `weights = 1 / class_count`, and `FocalLoss(α=0.25, γ=2)` for
   the 74-class head.

Run for 5 epochs. Compare per-class accuracy distribution. B
should have lower std (more uniform) and higher macro-F1.

**Success metric.** `act_macro_f1` > 0.05 at epoch 5 in B (vs
~0 in A under the same conditions), AND
`min(per_class_acc) > 0` in B (vs all-zeros in A).

**Opus's call:** is there a smarter rebalance for 74 classes ×
40 samples, or should we drop the long tail entirely (keep top
30 classes, ignore the rest)?

---

## H6. Replace ConvNeXt-Tiny with a Smaller Backbone (EfficientNet-B0 or MobileNet-V3)

**Hypothesis.** ConvNeXt-Tiny is over-parameterized for our 5%
subset. With 28.6M backbone params + 12.4M head params, we have
~41M trainable params and ~3,000 training frames. The parameter-
to-sample ratio is ~14,000:1, which is well into the
overfit-immediately regime. A 5M backbone (EfficientNet-B0) would
have a 7,000:1 ratio and would free enough VRAM to run batch=8
in FP32 with 4× the gradient signal per epoch.

**Evidence.** Per `diag_features_alive.py`, the ConvNeXt-Tiny
features are alive but possibly too rich. The activity head's
collapse is not a backbone-capacity problem; it's a
class-imbalance + reweighting problem. The detection head's
collapse is a bias-init problem, not a backbone problem. The PSR
head's collapse is a bias-init problem, not a backbone problem.

**Code locations.**
- `model.py:164` — `ConvNeXtBackbone` class.
- `config.py` — `BACKBONE='convnext_tiny'`, `PRETRAINED=True`.

**Experiment.** Train 2 versions:
1. **A:** ConvNeXt-Tiny, batch=2 in FP32 (current).
2. **B:** EfficientNet-B0 (`timm.efficientnet_b0`), batch=8 in
   FP32. Adjust FPN channel counts (B0 outputs 24/40/112/192
   vs ConvNeXt's 96/192/384/768).

Run for 5 epochs. Compare per-head eval metrics AND per-step
time. If B is equal-or-better on per-epoch eval but 2-3× faster
per step, swap.

**Success metric.** 5-epoch wall-clock time < 90 min (vs ~3.5h
for A), AND `det_mAP50 > 0` AND `act_top5_accuracy > 0.05` at
epoch 5.

**Opus's call:** is ConvNeXt-Tiny worth the 5% subset cost, or
should we drop to a smaller backbone? Or should we keep
ConvNeXt-Tiny but freeze the first 2 stages (saves VRAM, no
quality loss)?

---

## H7. Drop FiLM Conditioning on Activity (or Drop Activity)

**Hypothesis.** `PoseFiLM(C5)` and `HeadPoseFiLM(C5_mod)` carry
pose information into the activity head. The eval shows body
pose MAE = 739 mm (broken scale) and head pose MAE = 61° (large
but real). The FiLM signals are therefore noisy, and a noisy
conditioner can destabilize the conditioned head. Either (a) drop
FiLM and use plain concatenation of pose as additional input
features, or (b) drop activity entirely and ship a 3-head
model (det + pose + PSR).

**Evidence.** The activity head collapses despite receiving
FiLM-modulated C5 features. The body pose head (which produces
the FiLM input) is in a poor scale regime (MAE 739 mm = 7× the
body's expected 100 mm scale). The FiLM is conditioning on a
noisy signal.

**Code locations.**
- `model.py:607` — `PoseFiLMModule`.
- `model.py:703` — `HeadPoseFiLMModule`.
- `model.py:1192` — `ActivityHead` consumes the FiLM-modulated C5.
- `config.py` — `USE_FILM=True`.

**Experiment.** Train 2 versions:
1. **A:** current with FiLM.
2. **B:** drop PoseFiLM and HeadPoseFiLM, activity head input
   is just GAP(C5) ‖ GAP(P4) = 768 + 256 = 1024 (vs 1048 with
   det_conf).

Run for 5 epochs. Compare `act_top5_accuracy`. If B is equal-
or-better, FiLM was hurting. If A is better, keep FiLM but
fix the pose-head scale first (separate ablation).

**Success metric.** `act_top5_accuracy` improves by ≥ 0.02
absolute in B (or stays within noise if FiLM was a no-op).

**Opus's call:** keep FiLM, drop FiLM, or replace it with
something simpler (cross-attention, additive bias)? Show the
code.

---

## Bonus: The Hardest Question (H8)

**Hypothesis.** The right architecture for "5 heads, 5% subset,
12GB GPU" is not a single shared backbone at all. It is
**3 separate small backbones** with shared low-level features
(only the first 2 ConvNeXt stages are shared, not the FPN).
This is the design that the paper's "single backbone" claim
should be tested against.

**Why this is H8.** This is the "change the whole architecture"
option. It is the most likely path to a working model but
contradicts the paper's headline. We list it for completeness
and ask Opus to confirm or deny.

**Code locations.** N/A (would require a new model class).

**Opus's call:** confirm or deny. If confirmed, sketch the
3-backbone design and tell us which heads share which stages.

---

## How to Read These Hypotheses

Each hypothesis is **independently testable** in 5 epochs at 5%
subset (~3.5h in FP32). The user has stated they are open to
"changing the backbone or anything." These 8 hypotheses cover
the full surface area: loss (H1), curriculum (H2), architecture
(H3, H4, H6, H7, H8), and class balance (H5).

If Opus picks even 3 of these and provides code, we can run
the experiments in 1–2 days and have a working 4-head model.

The 5+ MD files Opus returns should be the implementation
plan for 3+ of these hypotheses. See `05_MASTER_PROMPT.md`
for the format.
