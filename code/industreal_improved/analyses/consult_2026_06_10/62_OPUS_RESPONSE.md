# 62 — Opus Response to the RF10 Consult (files 56–61)

Date: 2026-06-30 · Model: Opus 4.8 · Scope: code-only checkout (no data/GPU in this
environment, so every claim below is read from source, not from a re-run).

---

## TL;DR

1. **The "invariant 0.010 activity gradient" is a red herring.** At a fixed
   parameter state the gradient w.r.t. the activity-head weights cannot change
   when you change the learning rate or the backbone blend ratio — that is basic
   autograd, not a bug. You spent attempts 4–6 (3x → 10x → 20x LR) chasing a
   number that is mathematically guaranteed not to move. Stop tuning LR.

2. **The real root cause is structural: the temporal head is fed non-temporal
   data.** Training uses a class-balanced `WeightedRandomSampler` (per-frame,
   shuffled). The `FeatureBank` ring buffer is keyed by `video_id`, so it
   accumulates *random, non-consecutive frames from random videos*. The
   TCN + 2×ViT stack (8.2M params) is therefore learning temporal structure from
   noise, on a 3.7k-frame dataset, while the one live frame is just 1 of 17 tokens
   behind two self-attention blocks — which is exactly why its effective gradient
   is small and the head collapses to the majority class.

3. **Fix shipped in this branch:** `ACTIVITY_HEAD_SIMPLE` (default on) bypasses
   TCN+ViT and classifies the projected per-frame feature with a ~150K-param MLP.
   Short gradient path, far less overfitting capacity, no noise from the fake
   temporal bank. This is the single highest-leverage change for getting *any*
   non-trivial activity signal under the current per-frame regime.

4. **Be honest about the paper.** Matching dedicated single-task SOTA
   (YOLOv8m det_mAP50 = 0.838; B2 psr_f1 = 0.731) with a from-scratch 5-task
   model on consumer GPUs and a 50% subset is not a realistic claim, and no
   hyperparameter will close a 78-point detection gap. There **is** a publishable
   paper here — just not that one. See §5.

---

## 1. Why the 0.010 gradient never moved (the core mystery, dissolved)

The gradient of the loss w.r.t. the activity-head parameters at a given weight
state depends on exactly two things: the forward activations and the loss. It does
**not** depend on:

- **Learning rate.** LR multiplies the *update* (`θ ← θ − lr·g`), not the gradient
  `g`. Measuring `‖g‖` at step 0 across LR 0.5×→20× and getting 0.010 every time is
  the expected result. ("What We've Ruled Out → LR too low" is not a finding.)
- **`ACTIVITY_GRAD_BLEND_RATIO`.** That blend (`model.py:2106`) controls how much
  activity gradient flows *into the backbone* via `c5_mod_blend`. It changes the
  backbone's gradient, not the activity head's own parameter gradients. Invariance
  here is also expected.
- **`ACTIVITY_HEAD_GRAD_CLIP`.** Clip at 1.0 with `‖g‖ = 0.01` is a 100× no-op, as
  you noted.

So the entire "invariant gradient" investigation was measuring a quantity that, by
construction, those knobs cannot affect. The number being *small in absolute terms*
is real and worth explaining (next section); the number being *invariant* is not a
clue.

## 2. What is actually small, and why

The activity contribution to the total loss is
`prec_act · (loss_act · ACTIVITY_LOSS_WEIGHT) + lv_act` (`losses.py:1765`), and at
step 0:

- `act_ramp = (0+1)/5 = 0.2` multiplies `prec_act` (`losses.py:1696`, 1715),
- `ACTIVITY_LOSS_WEIGHT = 0.8`,
- so the effective weight on `loss_act` is ≈ `0.2 · 0.8 = 0.16` early on.

That alone is a mild down-scale. The dominant effect, though, is **architectural
attenuation**: the gradient to `proj_features` has to traverse
`classifier → CLS-token pooling → 2× ViT self-attention → TCN`, where the live
frame is one of 17 tokens and most of the bank is `detach()`-ed
(`FEATURE_BANK_DETACH_GRAD_ENTRIES_ONLY=True`, `model.py:1215`/`1241`). The 8.2M
temporal params soak up a lot of the gradient norm, so the per-parameter signal on
the part that matters (the projection + classifier) is tiny. Hence a small `‖g‖`
*and* a head that prefers the constant-output (majority-class) minimum.

## 3. The data + sampler interaction (the decisive evidence)

- `train.py:340` → `sampler = ds.get_sampler()` → `WeightedRandomSampler`
  (`industreal_dataset.py:1481`), class-balanced over action classes.
- The DataLoader draws frames in shuffled, class-balanced order, so two adjacent
  items in a batch are unrelated frames from possibly different recordings.
- `FeatureBank.forward` (`model.py:1179–1244`) builds the temporal window by
  appending to a per-`video_id` list **in arrival order** — i.e., in shuffled
  order. The resulting `[B, T, 512]` "sequence" is not a temporal sequence.
- `ActivityHead.forward` then runs TCN+ViT over that non-sequence and overwrites
  only slot −1 with the live frame (`model.py:1384`).

Conclusion: under the current sampler the temporal modeling has **no valid input**.
It cannot help; it can only overfit and dilute gradient. This is the smoking gun,
and it is consistent with every symptom you reported (collapse to 1 class,
small-but-nonzero gradient, no response to LR/blend/clip).

## 4. What I changed in this branch

All gated; nothing is destructive. Re-validate on your machine — I could not run it
here.

| File | Change |
|------|--------|
| `src/config.py` | `ACTIVITY_HEAD_SIMPLE = True`, `ACTIVITY_HEAD_SIMPLE_HIDDEN = 256` |
| `src/models/model.py` | `ActivityHead` builds a `simple_classifier` MLP and, when the flag is on, `forward()` returns `simple_classifier(proj_feat)` directly — TCN/ViT/bank unused. |
| `src/training/train.py` | `--reinit-heads` now also reinitializes `simple_classifier` (hidden Xavier; logit layer std=0.01, bias=−0.5 to discourage majority collapse). |

The simple head keeps the exact same projection input
(`det_conf ⊕ GAP(c5_mod_blend) ⊕ GAP(p4)` → `proj_features` → 512), so backbone
gradient flow is still governed by `ACTIVITY_GRAD_BLEND_RATIO` and nothing else in
the pipeline needs to change. Param-group LR multiplier, gradient clip, and
gradient centralization all still apply because the new params live under the
`activity_head.` prefix.

**How to use it.** Start a *fresh* RF run (or resume with `--reinit-heads`) so the
reinitialized simple classifier is what trains. Watch `act_macro_f1` and, more
importantly, the **prediction-entropy / number of distinct predicted classes** in
the first 1–2 epochs: if it stays diverse instead of collapsing to one class, the
fix is working. Expect modest numbers (top-1 in the low 0.1s at best on this data),
not the paper target — see §5.

If/when you train on genuine **sequence batches** (`sequence_mode=True`), set
`ACTIVITY_HEAD_SIMPLE = False` to bring the temporal stack back; that is the only
regime where it is justified.

## 5. Honest paper feasibility (file 60, Q26–31)

| Task | Now | Stage/paper target | Verdict |
|------|----:|----:|---------|
| head_pose MAE | 8.71° | ≤35° / 10° | **Already meets it.** Your one solid result. |
| act_top1 | ~0 | 0.18 / 0.375 | Low 0.1s is plausible with the simple head + balanced sampling. 0.375 is not realistic on 3.7k frames / 72 classes (46 classes <1%). |
| det_mAP50 | 0.053 | 0.30 / 0.838 | 0.838 is a dedicated YOLOv8m on full data. **Unreachable** for a from-scratch joint head on 50% data. 0.2–0.3 is an honest ceiling. |
| psr_f1_at_t | ~0 | 0.16 / 0.731 | Needs the transition objective working on sequence batches; 0.731 is a single-task SOTA, not a multi-task target. |

**Do not frame the paper as "we match SOTA on five tasks."** That framing forces
gaps you cannot close and will not survive review. The defensible AHFE paper is a
**systems / applied** contribution:

> *A single shared-backbone model that performs assembly verification across five
> tasks on consumer hardware; head-pose matches SOTA; and we provide an honest
> analysis of the optimization pathologies of joint training under severe class
> imbalance and limited annotation (the temporal-head/sampler mismatch, Kendall
> precision dynamics, majority-class collapse), with ablations.*

AHFE is an applied human-factors venue. A working multi-task pipeline + the
failure-mode analysis you have already painstakingly produced (files 56–60 are
*good material*) is publishable. Reframing is the highest-value move you can make
this week — it converts 10 days of "failures" into the paper's contribution.

## 6. Direct answers to the six questions in file 61

1. **Why 0.010 everywhere?** Because none of the knobs you varied can move a
   fixed-state parameter gradient (§1). It is *small* because of weighting + deep
   attenuation (§2), not detached.
2. **Rip out ViT+TCN for an MLP?** Yes, for the per-frame regime — done (§4). Keep
   the temporal stack only for true sequence batches.
3. **Validate without CUDA hangs?** Use **subprocess evaluation** (fork a fresh
   process, hard-kill on timeout) rather than `ThreadPoolExecutor`/`SIGALRM` —
   threads and signals cannot interrupt a CUDA kernel, a separate process can be
   `SIGKILL`-ed. Also drop to **every-N-epochs** validation to shrink exposure.
4. **Two GPUs?** No — not now. DDP engineering time is poorly spent while the model
   doesn't learn the task. Fix the head first; revisit throughput later.
5. **Targets realistic?** head_pose yes; the rest no at *paper-SOTA* levels (§5).
   The *stage gates* (act_top1 ≥0.18, det ≥0.30) are reachable; the *paper numbers*
   are not, so stop treating them as the same thing.
6. **First thing in one week?** In order: (a) the simple head (shipped); (b)
   subprocess eval to stop the crashes; (c) confirm activity stops collapsing via
   per-class prediction diversity; (d) **reframe the paper** per §5; (e) lock
   head_pose + a respectable det/act/psr and write. Do **not** spend the week on
   PCGrad/CAGrad/Nash-MTL — advanced MTL optimizers won't rescue a head that's
   being fed noise.

---

## 7. What I could not do here

This is a code-only checkout (8.2 MB, no images, no checkpoints, no GPU), so I did
not run training, measure the real per-class frame counts, or validate the simple
head end-to-end. The changes compile (`py_compile` clean) and are gated. Please run
a short fresh RF4 with `--reinit-heads` and report back the first-2-epoch
prediction diversity + `act_macro_f1`; that single number will confirm or refute
the diagnosis fast.
