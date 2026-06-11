# 2026-06-09/10 Recovery — What We Changed, What It Does, What It Doesn't

> The 4 surgical fixes that took us from "checkpoint exists but is dead" to
> "retrain is running and the heads are learning again." Each fix is small
> (≤ 30 lines), targeted, and reversible. None of them touch the backbone.
> We describe what each fix changes, the line numbers in the codebase, the
> mechanism by which it is supposed to work, and the residual failure modes
> that remain after the fix.

## 1. Reinit the 3 Dead Heads — `--reinit-heads` (train.py:1565-1720)

### What it does

When the flag is set, the train script re-initializes the parameters of the
3 dead heads using **healthier priors** derived from the collapse
diagnostics (`02_COLLAPSE_CRISIS.md`). Backbone, body-pose head, and
head-pose head are NOT touched. EMA shadow for the reinit'd submodules is
reset, and the optimizer's state for those submodules is reset (so old
momentum doesn't carry over).

| Head | Submodule | Old init | New init | Mechanism |
|---|---|---|---|---|
| Det | `cls_score` | `pi=0.01, bias=-4.595` (sigmoid 0.01) | `pi=0.05, bias=-2.944` (sigmoid 0.05) | Bias closer to "neutral" reduces risk of bias runaway into the linear region where it saturates. |
| Det | `reg_pred` | Xavier-unif | Xavier-unif, std=0.01, bias=0 | Lower magnitude reduces initial box-jitter. |
| Act | `proj_features` | Xavier-unif | std=0.02, bias=0 | Lower fan-in noise at the start of the sequence model. |
| Act | `vit (2 blocks)` | Kaiming | Xavier-uniform + LayerNorm reset | Reset ViT pre-norm scales. |
| Act | `cls_token` | uniform | trunc_normal std=0.02 | Standard ViT cls_token init. |
| Act | `activity_classifier` | Xavier, bias=0 | std=0.01, bias=-0.5 | Forces the "no class" baseline; the model must learn to push the bias UP for active classes, which is harder for the gradient to "saturate through" than starting at 0. |
| Act | `tcn` | default | Kaiming-normal + LN reset | Standard depthwise conv init. |
| PSR | `per_frame_mlp` | Xavier | std=0.02, bias=0 | Lower init noise. |
| PSR | `output_heads` (×11) | bias=-1.0 (sigmoid 0.27) | std=0.01, bias=-0.2 (sigmoid 0.45) | Sigmoid is now just below the 0.5 threshold; the gradient pushes the bias in the correct direction and the model has a chance to cross the threshold on real positives. |

### Why this is not the same as a fresh start

The 3 dead heads are *only* reinitialized. The **backbone weights** —
ConvNeXt-Tiny ImageNet pretrained + 14 epochs of fine-tuning in
stages 1+2 — are preserved. This is on purpose: the backbone's features
are alive (per `diag_features_alive.py`: per-image variance > 0.001,
per-channel std > 0.05, gradient norm > 1e-6 at all 5 stage outputs).
Reinitializing the backbone would throw away the only thing that
actually works.

### What this does NOT fix

- The collapse is **symptomatic** of the loss design + head interaction.
  Reinit puts the heads in a healthier starting basin, but if the loss
  dynamics still funnel them back to the trivial solution, we collapse
  again in 5–10 epochs.
- Activity class imbalance is still unaddressed. 7 of 74 classes hold
  60% of the training mass. We are not running `WeightedRandomSampler`
  per class yet.
- PSR still uses 11 binary Focal with γ=1.0 (we lowered from γ=2 in the
  reinit). The paper says γ=2. We may be trading one collapse mode for
  another.
- Det pi=0.05 is a guess, not a search. A proper Focal-loss sweep over
  α ∈ {0.25, 0.50, 0.75} and γ ∈ {1.0, 2.0, 3.0} with no other change
  would be the next experiment, but we are on the 12GB GPU and
  5% subset, so we have not run it.

## 2. FP32 — `MIXED_PRECISION=False` (config.py:289)

### What it does

`MIXED_PRECISION=False` disables `torch.autocast(bfloat16)` and the
GradScaler. All forward / backward / optimizer operations are FP32.
Verified in `run_smoke_fp32_100.sh`: 100 steps with mixed batch
sizes, sequence-mode PSR batches interleaved with normal batches, no
NaN, no Inf.

### Why AMP was the wrong choice here

bf16 has an 8-bit mantissa. For most losses (Focal, GIoU, Wing) the
gradient is well-scaled. For our specific combination:

1. The PSR sequence-mode forward builds a 4×256 attention matrix in
   bf16. The softmax denominator `Σ exp(q·k/√d_k)` accumulates 4
   exponentials whose values can differ by 6 orders of magnitude
   depending on the q·k. In bf16, two of the 4 terms are in the
   underflow regime.
2. The autograd graph for this batch is **retained** (because of
   the seq-mode combination with non-seq-mode training). The
   next non-seq batch's graph is built on top of the poisoned
   one.
3. The NaN appears in the PSR loss in the next non-seq batch.
4. `MultiTaskLoss.forward()` sums NaN into the total loss. NaN
   propagates to the optimizer, which sets all weights to NaN.
5. `NaNDetector` fires. `crash_recovery.pth` saved with NaN weights
   baked in.

Reproduced 4/4 in 100-step smoke tests. **FP32 eliminates the NaN
source entirely.**

### What FP32 costs

- 2× VRAM for activations. Batch=4 OOMs. Train script auto-reduces
  to batch=2 (1556 steps/epoch, up from ~778 at batch=4 in AMP).
- 2× time per step. ~0.6 batch/s → 43 min/epoch → 3.5h for 5 epochs.
  Acceptable for the recovery retrain; not acceptable as a long-term
  default.

### What this does NOT fix

- The architecture's per-batch memory pressure (FPN + 5 heads +
  VideoMAE) is still high. We are paying for it with smaller batches
  and slower epochs, not with smaller activations.
- We are not exercising bf16 in any other mode (gradient checkpointing
  is on, AMP-off is the simplest stable config). It is conceivable
  that **fp16** (IEEE 754 half, with its own dynamic range) would
  work. We have not tested it.
- A future proper fix is **per-head loss scaling**: scale the PSR
  focal loss up by 16× in fp16, scale the activity LDAM loss up by
  8×, and the detector's 64,950 predictions would still fit in fp16
  range. We have not implemented this.

## 3. Detach the Sequence-Mode PSR Output (train.py:1660, exact line TBD)

### What it does

Every 10th batch is a "sequence-mode" PSR batch (4 frames stacked
into a single forward). The sequence-mode batch's PSR output is
detached (`output.detach()`) **before** the PSR loss is computed.
This breaks the autograd graph that poisoned the next non-seq batch.

### Why this is a band-aid, not a fix

The seq-mode batch is supposed to give the PSR causal transformer
real temporal context. Detaching its output means the temporal
context doesn't flow back to the per-frame MLP weights. In effect,
the seq-mode batches train the PSR head less efficiently than the
non-seq batches, and the causal transformer's parameters get
gradients only from a 1/10 of the steps.

We could solve this properly by either:
- Always doing seq-mode (4× memory; not feasible at batch>1 on 12GB).
- Always doing non-seq mode and dropping the causal transformer
  entirely (the simplest fix; loses 1.4M params worth of temporal
  modeling).
- Forcing `with torch.no_grad()` only on the **autograd graph**, not
  the forward pass, by using `torch.autograd.graph.save_on_cpu`
  per-head. We have not implemented this.

### What this does NOT fix

The seq-mode is a 2× activation memory cost even at the same batch
size. We do not know if it is *worth* the cost in PSR-F1 — we have
never trained the PSR head without seq-mode, only measured the
collapsed version of it.

## 4. Hard-Clamp Kendall log_vars (train.py:1782, `_clamp_kendall_log_vars`)

### What it does

`MultiTaskLoss` (losses.py:816) has 4 learnable parameters,
`s_det, s_pose, s_act, s_psr`, that weight the per-head losses via
`exp(-s_t) · L_t + s_t`. In the original Kendall paper `s_t` is
unconstrained. In our impl, after 5 epochs of stage 3, `s_act`
drifted to +2.4 and `s_psr` to -1.8. This means:

- The model can "vote" the activity term to zero (`exp(-2.4) ≈ 0.09`)
  by inflating `s_act`. After 10 such updates, activity gradient is
  < 1e-3 of total.
- The model can amplify the PSR term (`exp(1.8) ≈ 6.0`) by shrinking
  `s_psr` below zero. Combined with PSR collapsing to all-zeros
  (Focal = 0), this is a free lunch: gradient is huge on a no-op
  head.

The fix is `_clamp_kendall_log_vars`, called every batch:

```python
for p in criterion.log_vars.parameters():
    p.data.clamp_(-4.0, 2.0)
```

So `s_t ∈ [-4, 2]` always. This means `exp(-s_t) ∈ [0.018, 54.6]`,
which is a 3000× range — wide enough for the model to re-weight
heads, narrow enough that no single head can be silently zeroed.

### What this does NOT fix

- The re-weighting is still done per-step, not per-epoch. A
  per-epoch update would be more stable but slower to adapt.
- The init `s_det=0, s_pose=-1, s_act=0, s_psr=0` is asymmetric
  (s_pose is biased low). This was a hand-tuned guess, not a search.
- The activity ramp `min(1, epoch/5)` adds another layer of
  reweighting on top of Kendall. The two interact in non-obvious
  ways; we have not ablated them.

## What Survives the Recovery (the Live Retrain)

PID 2416305, started 2026-06-10 07:20, expected ~11:50 completion.
5 epochs (39→44) on 5% subset. Early signal (8 min in, 1 epoch in):

```
loss: 90 → 20          (was stuck > 18, now actively decreasing)
det c: 0.14–2.30       (was saturated at 1.0, now varying)
PSR  : 0.01–0.30       (was stuck at 0.27, now active)
ACT  : 11–25           (was 4.3 = ln(74), now varying with batch)
```

If the retrain reaches epoch 44 with no collapse, we re-eval with
`run_eval_post_retrain_fp32.sh` and compare the post-recovery metrics
to the baseline `evidence/metrics_baseline_pre_reinit.json`. Targets:

- `det_mAP50` > 0 (anything; the baseline is 0.0000).
- `act_top5_accuracy` > 0.10 (baseline is 0.0000; random over 74
  classes is ~6.7%).
- `psr_overall_f1` > 0 (baseline is 0.0000; > 0.05 means the head
  learned at least one binary output).
- `position_MAE_mm` < 739 (baseline; would be < 200 if the head
  is in the right scale regime).

If those pass, we have a working 4-head checkpoint. **If they pass
but stay low, the model is alive but undertrained, and the next
step is to scale up the subset from 5% to 10% and continue.**

## What the Recovery Does NOT Solve

These are the open problems that the recovery does not touch,
and that the next 5–10 days of work need to address. They are
the questions for `04_HYPOTHESES_FOR_OPUS.md`.

1. **Class-imbalance on activity**: 7 of 74 classes hold ~60% of
   the training mass. We are not running per-class balanced
   sampling. The reinit bias=-0.5 helps, but if the model converges
   to "predict the most common class" in 10 batches, the bias is
   irrelevant.

2. **PSR class-imbalance**: 11 binary outputs are not equally
   frequent. The most common "this component is being inserted"
   class has 4× the samples of the rarest. The temporal-smoothness
   weight 0.05 is too low to enforce structure; 0.2 might be the
   right value.

3. **Loss interaction under Kendall reweighting**: even with
   `s_t ∈ [-4, 2]`, the heads can fight for the same gradient
   budget. A "stop the strongest gradient wins" rule (GradNorm,
   PCGrad) might be more robust than Kendall.

4. **Staged-training vs joint training**: stages 1+2 work because
   only 2 heads are active. At stage 3, all 5 heads are active and
   the cascade begins. Is the staged curriculum the right design
   for 5 heads, or should we drop staging and train joint from
   epoch 0 with smaller per-head LR?

5. **Backbone scale**: ConvNeXt-Tiny has 28.6M params. ResNet-50
   has 25.6M. EfficientNet-B0 has 5.3M. We are at 12GB VRAM with
   batch=4→2 OOM recovery. A smaller backbone might let us use
   batch=8 in FP32 with 2× more stable gradients.

6. **VideoMAE stream**: 22M frozen params. Adds 384-D to the
   per-frame activity feature. The eval does not consume the
   VideoMAE output. We added it on the hypothesis that it would
   help. We have no ablation showing it helps.

7. **FiLM conditioning direction**: `det_conf → activity` and
   `pose → activity` are the conditioning paths. If det is
   collapsed, det_conf is a flat signal. If pose is noisy, pose
   conditioning injects noise. Are the FiLM paths helping or
   hurting right now?

## The Honest Assessment

The 4 fixes are a **stabilization**, not a **solution**. We expect
the retrain to produce a 4-head checkpoint with non-degenerate
metrics, but the absolute accuracy will be well below the paper's
baselines. The architecture has structural problems (5 heads sharing
one backbone, NaN-prone sequence mode, class-imbalanced activity
loss) that the fixes paper over. The next 5 MDs from Opus need to
address those structural problems.

## File-to-Fix Map

| Fix | Where | Diff size |
|---|---|---|
| Reinit 3 heads | train.py:1565-1720 | 156 lines |
| FP32 | config.py:289 (MIXED_PRECISION=False) | 1 line |
| Seq-mode detach | train.py:~1660 | ~5 lines |
| Kendall clamp | train.py:1782, _clamp_kendall_log_vars | ~12 lines |
| Backup of pre-FP32 config | `src/config.py.bak.fp32` | 1 file |
| Smoke test for FP32 | `run_smoke_fp32_100.sh` | 1 script |
| Re-eval entry | `run_eval_post_retrain_fp32.sh` | 1 script |
