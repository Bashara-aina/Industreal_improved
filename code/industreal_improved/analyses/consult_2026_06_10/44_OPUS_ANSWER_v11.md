# 44 — OPUS ANSWER v11: The Fix Is Right; The Experiment and the Data Aren't (2026-06-21)

> Response to `43_OPUS_MASTER_PROMPT_v11.md`. Opus re-read the live code on the
> branch (`claude/wonderful-gates-bd2uco`), not just the summary. Three premises
> in the v11 prompt are contradicted by the repository itself — fixing them
> changes the restart plan.

---

## TL;DR

1. **`detach_reg_fpn=False` for RF2 is already committed** (commit `2ad6cfe`,
   `config.py:1115`). The working tree is clean. The prompt's "applied but NOT
   COMMITTED" is stale. **The only pending action is the restart**, because the
   live PID still holds the pre-`2ad6cfe` config in memory.
2. **Restart with ONLY the diagnosed variable changed.** Revert
   `DET_LR_MULTIPLIER → 1.0` and `DET_BIAS_LR_FACTOR → 1.0`. They are *your*
   additions, they contradict the v8 "bias-acceleration own-goal" finding, and
   bundling them with the detach flip means a win or a loss is uninterpretable.
3. **Do not build the class-6 decision tree on the v11 per-class table — its
   provenance is broken.** Detection per-class AP is *not* written to
   `metrics.jsonl` (v10 said so; the schema confirms it). The GT counts in §5
   contradict the repo's own curated `DET_CLASS_ALPHAS`. Regenerate the table
   from `evaluate.py` with verified indexing *before* reasoning about class 6.
4. **Expected magnitude:** detach=False should lift the ceiling to **~0.25–0.32
   in 3–4 epochs** (moderate case). It is necessary, not sufficient, for 0.40.

---

## Part A — Three Corrections to the v11 Premise (all code-verified)

### C1 — The detach fix is committed, not "in the working tree"

```
$ git log --oneline -1 -- src/config.py
2ad6cfe fix(rf): detach_reg_fpn=False — v10 breakthrough fix for RF2 ...
$ git status → working tree clean
```

`config.py:1115` (`stage_rf2`) = `'detach_reg_fpn': False`, blamed to `2ad6cfe`
(2026-06-21 17:04). The plumbing is correct end-to-end:
`stage preset → config.py:1466 DETACH_REG_FPN → model.py:1742 → model.py:561
reg_feat = feat.detach() if self.detach_reg_fpn else feat`. A process launched
before `2ad6cfe` reads the *old* value at startup; Python won't hot-reload it.
**The fix takes effect on the next launch and only then.** Nothing else to apply
for the flip itself.

### C2 — The per-class AP table has no trustworthy source

The v11 §5 says the table was *"parsed from metrics.jsonl … it was always being
written."* That cannot be true for **detection** per-class AP:

| Check | Result |
|-------|--------|
| `metrics.jsonl` val schema | has `act_per_class_acc`, `psr_per_component_f1`, `ev_ap` — **no `det_per_class_ap`** |
| v10 answer (`42_..._v10.md:19,49`) | *"computed, thrown away — NOT persisted"* |
| `evaluate.py:267,273` | `compute_ap_per_class()` returns `per_class_ap` up the stack |
| `train.py` | **never writes** that dict into `val_metrics` (only activity per-class is persisted, `train.py:2711`) |

So the §5 numbers came from somewhere unlabeled (a manual `evaluate.py` run, or
an estimate). Worse, the **GT counts disagree with the repo's own curated map**,
`DET_CLASS_ALPHAS` (`config.py:471–506`, annotated *"v4 — Corrected model-index
mapping"* precisely because v3 had index errors):

| v11 §5 claim | `DET_CLASS_ALPHAS` says (model-index) | Verdict |
|---|---|---|
| Class 6: **1739 GT**, AP=0 — "more than most working classes" | idx 6 = cat 7, **train=65, val=91**, AP=0.000 | GT count off by ~20× |
| Classes 13, 19: "AP=0 **WITH GT**" | idx 13/19/23 = **Zero train GT (can't learn)** | not a mystery — untrainable |
| Classes 5, 21 AP=1.0 ⇒ "architecture CAN learn perfect classification" | idx 21 = *"AP=1.0 **artifact** (no GT in 0.50 subset val)"* | a metric artifact, not proof |

**Consequence:** the "class 6 litmus test" rests on a GT count the codebase
contradicts, and the "proof the architecture works" rests on AP=1.0 from
near-empty validation. Both must be re-derived before they can gate a decision.

### C3 — The "v10 breakthrough" re-derives a result you shipped on 2026-06-17

`stage_manager.py:107–121` (the recovery strategy) already runs
`reinit_heads: True` **with** `detach_reg_fpn: False`, and its comment is the v10
diagnosis verbatim, four days early:

> *"…that leaves the backbone with ONLY the sparse classification path — features
> never become object-discriminative … the head sticks at the … 'predict-
> background-everywhere' equilibrium (localizes but won't fire). … the detach was
> redundant overkill that starved the trunk."*

This is good news twice over: the fix is **battle-tested**, and the comment
already tells you the correct guard against reinit gradient shock is
`REINIT_REG_WARMUP_STEPS=1000` (`config.py:671`), **not** detach. It reframes the
plateau as *config drift* — the `False` was applied to `stage_rf1` and the
recovery path but never propagated to `stage_rf2…rf10`.

---

## Part B — Answers to the Seven Questions

### Q1 — How much, how fast?

**Moderate case is most likely: 0.25–0.32, visible within 3–4 epochs**, with the
first movement (dead easy/medium classes firing) by **epoch 1–2** after restart.
Reasoning: detach=True was a real handicap (severed the densest detection
gradient from the FPN), but RF2-detach=True already reached 0.204 *above*
RF1-detach=False at 0.184 — so detach is not the *whole* story; v8 fixes + data
were compensating. Removing the handicap should compound, but the residual
ceiling is fine-grained class discrimination (see Q3), which gradient restoration
helps but does not solve. **0.40 in this single restart is the optimistic tail,
not the expectation.**

### Q2 — Revert `DET_LR_MULTIPLIER` and `DET_BIAS_LR_FACTOR` to 1.0? **Yes — both.**

This is the most important operational call in the prompt. You are about to flip
**four** variables at once (detach, IoU-floor, LR×2, bias×4); two of them are
unvalidated and one *contradicts a prior Opus finding you accepted*. The
mechanics confirm the risk:

- `train.py:3217–3218`: `det_head_lr = head_lr × DET_LR_MULTIPLIER`;
  `det_head_bias_lr = head_lr × DET_BIAS_LR_FACTOR`. At 4.0, the cls **bias**
  moves at ~13× the generic-bias rate (generic `BIAS_LR_FACTOR=0.3`). The v8
  finding was that **bias momentum toward the all-background equilibrium is the
  collapse mechanism**. `IOU_FLOOR=0.2` reduces false-positive labels but does
  **not** reverse the direction of the dominant negative gradient under
  173K:1 imbalance — it just slightly raises the positive count. Accelerating the
  bias 4× still pushes the same direction faster.
- `DET_LR_MULTIPLIER=2.0` raises head LR while Focal + unbounded-ish top-k
  produce noisy assignments; higher LR amplifies assignment noise.

**Restart at `1.0 / 1.0`.** If, after detach=False stabilizes (≥2 epochs of
upward trend), the head looks gradient-starved (`det_head` grad ≪ `head_pose`
grad), raise `DET_LR_MULTIPLIER` to 1.5–2.0 **alone** and observe. Never move
both at once again. Keep `DET_BIAS_LR_FACTOR=1.0`; if anything, the RetinaNet
prior is set once via `reinit_pi`, not driven by a high bias LR.

### Q3 — Why is class 6 dead while 5/21 are "perfect"?

First, distrust the framing (C2). Then note **what these classes are**: the
detection labels are 11-bit *assembly-state* codes (`config.py:171–196`), e.g.
cat 7 = `11110010000`, cat 8 = `11110100000` — **adjacent states differ by 1–2
bits = one washer/nut placed, on a near-identical partial assembly.** This is a
**fine-grained classification** problem, not a localization or small-object
problem (the box is the whole partial model, so the anchor-size hypothesis is
*weak* here — `ANCHOR_SIZES=(96…512)` on 1280×720 cover large assemblies fine).

Most-likely causes, ranked:
1. **Data scarcity × fine-grained confusion.** If idx 6 really has ~65 train
   instances (per `DET_CLASS_ALPHAS`), AP=0 is expected, exactly like idx 16
   (26 train, AP=0) and idx 8 (142, AP=0). The model defaults to the
   visually-near-identical high-count neighbor.
2. **Label/index provenance error** (C2) — resolve before anything else.
3. **Confusion-pair**, not random: predictions land on the 1-bit-Hamming
   neighbor. A 24×24 detection confusion matrix settles this in one eval.

Anchor-IoU mismatch and top-k poisoning are **lower** probability for whole-
assembly boxes, but the per-class max-IoU histogram is cheap insurance.

### Q4 — Overfit first, or restart first?

**Restart first; run the 50-image overfit in parallel.** The overfit needs a
separate process on a tiny dataset and does not block the main GPU run
meaningfully (it's minutes of compute, and it's the only clean read on the
*architecture's* ceiling free of multi-task/data confounds). Restarting is a
<5-min config edit + relaunch. Do both; gate nothing on serializing them.

### Q5/Q6 — RF3 (and a global policy)

**Set `detach_reg_fpn=False` for `stage_rf3` through `stage_rf10`**
(`config.py:1152, 1185, 1218, 1251, 1284, 1317, 1350`). Justification is not
speculative — it's the codebase's own established conclusion (C3): the recovery
strategy runs reinit **with** detach=False and uses `REINIT_REG_WARMUP_STEPS` as
the guard. First-time RF3–RF10 runs are **non-reinit continuations**
(`stage_manager.py:3003` only applies `reinit_heads` on *retries*), so they
carry the identical latent handicap RF2 just escaped.

**Better than per-preset hardcoding:** bind detach to the actual event — detach
the reg→FPN path *only for the first `REINIT_REG_WARMUP_STEPS` after a head was
just reinitialized this stage*, then release it. That is the principled rule;
the current preset booleans are a coarse proxy that already drifted once. There
is **no stage where permanent detach=True is correct** given the warmup guard
exists.

### Q7 — The combined metric (`best_metric=0.462`)

Deprioritize, don't delete. `0.667·mAP50 + 0.333·(1/(1+MAE))` with MAE near its
~9° floor pins ~1/3 of the score at a near-constant ~0.036-from-max, so the gate
looks "close" while detection is at 0.20. Keep logging it for continuity, but
**gate and judge on raw `det_mAP50` + per-class AP.** Don't let 0.462 read as
"almost at 0.50."

### Q8 — Decision rule after restart

Keyed to *measurements you must first persist* (per-class AP + per-class positive
count + per-class max-anchor-IoU + 24×24 confusion):

| Observation (epochs 1–4 post-restart) | Conclusion | Next |
|---|---|---|
| mAP → 0.25–0.32 **and** several dead classes fire | detach was the dominant handicap | continue RF2; then propagate detach=False forward |
| mAP climbs but the same low-count classes stay 0, predictions land on Hamming-neighbor | fine-grained data/imbalance ceiling | targeted oversampling / class-balanced sampler / hard-pair mining |
| mAP flat < 0.22 for 3 epochs **and** 50-img overfit also stalls < 0.8 | target/assignment/eval-pipeline bug, not training | audit `evaluate.py` indexing + assignment, not hyperparams |
| Overfit hits > 0.8 but full run stays flat | data-scale/assignment on rare classes | sampler + label audit |

---

## Part C — The Exact Restart Config

| Parameter | File:Line | Set to | Why |
|---|---|---|---|
| `detach_reg_fpn` (stage_rf2) | `config.py:1115` | **False** ✅ already committed | the diagnosed variable |
| `DET_POS_IOU_IOU_FLOOR` | `config.py:304` | **0.2** ✅ keep | bounds top-k label noise (v9) |
| `DET_POS_IOU_THRESH` / `TOP_K` | `config.py:298,303` | **0.4 / 9** ✅ keep | v8, unchanged |
| `DET_LR_MULTIPLIER` | `config.py:55` | **1.0** ⬅ revert from 2.0 | isolate the variable |
| `DET_BIAS_LR_FACTOR` | `config.py:56` | **1.0** ⬅ revert from 4.0 | undo the v8 "own-goal" |
| `reinit_pi` (stage_rf2) | `config.py:1120` | **0.05** keep | warm bias start |
| Focal/OHEM (`α=0.90`, `γ_pos=0/γ_neg=1.5`, OHEM 2:1, min_neg 32) | `config.py:463,528,529,515,518` | unchanged | leave the loss alone this round |
| Resume | — | from current `best.pth`, **keep heads, no reinit** | continuation, not bootstrap |
| Per-class logging | `train.py` (add) | **persist `evaluate.py` `per_class_ap` + positive-count + max-IoU into `val_metrics`** | the one missing measurement (Q8) |

**One-variable discipline:** the only training-dynamics deltas vs. the live run
are `detach_reg_fpn` and the two LR reverts (which return you to the v8 baseline).
That makes the next 4 epochs interpretable.

---

## What I Did Not Touch

The two reverts and the RF3–RF10 propagation are config changes that feed a
6–8h GPU run on your hardware. They are clearly correct per the codebase's own
prior conclusions, but I'm leaving the trigger to you rather than mutating the
training config unilaterally. Say the word and I'll apply: (a) the `1.0/1.0`
reverts, (b) `detach_reg_fpn=False` for `stage_rf3…rf10`, and (c) the per-class
AP persistence patch in `evaluate.py`/`train.py`.

---

*Saved from Opus consultation round 11. Code-grounded: the v10 fix is correct
and committed; the remaining risks are experimental hygiene (four variables at
once) and a per-class dataset whose provenance the repository contradicts.
Restart one-variable-clean, persist per-class AP, and let class 6 be answered by
measurement rather than narrative.*
