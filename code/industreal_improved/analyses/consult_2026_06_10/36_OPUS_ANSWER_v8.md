# 36 — OPUS ANSWER v8: The RF2 Collapse Is One Bug, Not Three (2026-06-20)

> Response to `35_OPUS_MASTER_PROMPT_v8.md`. Every claim below was verified by
> reading the **current** source in `src/` (not the doc snapshots), plus the live
> `src/runs/rf_stage_state.json`. Line numbers refer to the working tree on
> branch `claude/epic-hamilton-6sw452`.

---

## 0. TL;DR — Read This First

You have been chasing **three independent failure modes** (empty-frame
normalization, gradient sparsity, cls-score bias equilibrium). After reading the
code, I believe the RF2 collapse is **one mechanism wearing three masks**:

> **The shared backbone is being optimized for head_pose, not detection. Kendall
> uncertainty weighting mathematically guarantees this once head_pose converges,
> and an internal epoch-based stage switch turns head_pose precision ON at
> epoch 6 — which is exactly where your detection peak (ep8) → collapse (ep13)
> begins. The cls-bias drift to −2.54 and `cls_std→0.0068` are *downstream
> symptoms* of the backbone losing object-discriminative features, not a separate
> "bias equilibrium."**

Three findings from the code change the strategy materially:

1. **The `0.45` is a phantom.** `rf_stage_state.json` says `best_metric: 0.181`
   and `metric_history` tops out at `0.184`, but `stage_history[rf1] =
   0.45`. That `0.45` is *byte-identical to the RF3 gate threshold*
   (`stage_manager.py:164 'det_mAP50': 0.45`). RF1 never hit 0.45. **There is no
   great checkpoint being "eaten."** Drop every hypothesis built on catastrophic
   forgetting of a 0.45 model (Q3/Q10) — the real story is "0.18 detector
   destroyed by head_pose," which is a *much* smaller and more fixable gap.

2. **`pi` is already 0.03, not 0.01.** `model.py:541` → `pi = 0.03`, bias init
   `−3.48`. Your "Option A: pi=0.1" is the *third* iteration of bias-init tuning
   (0.01 → 0.03 → 0.1). It is a symptom knob. Stop turning it.

3. **There are TWO curricula fighting each other.** The RF stage manager sets
   `train_head_pose` per stage, AND `losses.py:1536-1567` runs a *second*,
   epoch-indexed Kendall curriculum that zeros/enables precision on its own
   schedule (stage1=ep1-5 head_pose frozen, stage2=ep6-15 active, stage3=ep16+
   PSR ramps). These were designed for the old all-heads `paper_run` and are now
   silently layered under the RF ladder. This double-curriculum is the engine of
   the "learns-then-collapses" timing.

Net: your five fix proposals (pi=0.1, remove bias, QFL, VFL, bias-LR) all target
the classification head. **The classification head is the victim, not the
culprit.** Fix the multi-task weighting and the anchor supply; the head recovers
on its own.

---

## 1. The Mechanism, In Code

### 1.1 Kendall makes the smallest-loss task the loudest task

`losses.py:1512-1520`:

```python
lv_det = self.log_var_det.clamp(-4.0, 2.0)
lv_hp  = self.log_var_pose.clamp(-4.0, _pose_max)   # _pose_max = KENDALL_LOG_VAR_MAX_POSE = 3.0
...
prec_det = torch.exp(-lv_det)     # range [exp(-2),  exp(4)] = [0.135, 54.6]
prec_hp  = torch.exp(-lv_hp)      # range [exp(-3),  exp(4)] = [0.050, 54.6]
```

For a Kendall task the optimizer's stationary point is `log_var* = log(loss)`, so
`precision* = 1/loss`. From your own logs:

| Task | typical loss (RF2) | Kendall-optimal precision | clamped |
|------|--------------------|---------------------------|---------|
| detection (`det_cls`) | ~0.5–1.0 | ~1–2× | ~1.4× |
| head_pose (`loss_head_pose`) | converges toward ~0.01 (doc 31: 1.60→0.01) | ~100× | **54.6× (floor −4 hit)** |

So head_pose is weighted **~30–40× detection** on the **shared backbone**. The
clamp comments (`losses.py:1505-1508`) only guard against pose/PSR being
*suppressed* (a ceiling on `lv`). **There is no floor protecting detection from
pose domination** — `lv_hp` can run all the way to −4. That asymmetry is the bug.

Crucially: head_pose's loss is small not because it is "solved" but because 9-DoF
MSE on standardized targets (`head_pose_loss_split`, `losses.py:887-905`,
`pos_weight=dir_weight=1.0`) is numerically small. A near-mean predictor
(your Q13 worry — MAE 47.84° is barely inside the useful range) produces a tiny
loss, and tiny loss → giant Kendall precision → backbone domination. **This is a
classic Kendall pathology when losses live on different scales.** You already saw
the same pathology and patched it for activity-vs-PSR
(`ACTIVITY_LOSS_WEIGHT=0.2`, comment at `losses.py:1599-1602`). It is now
happening to detection-vs-head_pose, unpatched.

### 1.2 The epoch-6 trigger lines up with the collapse

`losses.py:1536-1559` + config (`STAGE1_EPOCHS=5`, `STAGE2_EPOCHS=10`):

- **stage 1 = effective epochs 1–5:** `prec_hp = prec_hp * 0` → head_pose frozen.
- **stage 2 = effective epochs 6–15:** head_pose precision live.
- **stage 3 = effective epochs 16+:** PSR ramps in.

Your RF2 trajectory:

```
ep7: 0.007 → ep8: 0.184 (PEAK) → ep9: 0.181 → ep10: 0.159 → ep13: 1e-5 → ep15: 0.001
```

Detection peaks the moment head_pose precision comes online (ep6→8), then decays
monotonically as `prec_hp` climbs toward its 54.6 ceiling over the following
epochs. This is not a coincidence; it is the staging schedule.

(One more smell: at epoch 0 the staging block is skipped — `_current_epoch >= 1`
guard at `losses.py:1536` — so head_pose is ON at ep0, OFF ep1-5, ON ep6+. An
on/off/on auxiliary signal is exactly what you do *not* want under a learned
weighting that re-estimates variance from recent loss.)

### 1.3 Why detection can't defend itself: the supply side

Even without Kendall, detection's gradient to the shared backbone is weak because
positives are scarce — and that part of your gradient-sparsity analysis (file 29)
is correct. But the *root cause of the scarcity* is anchor assignment, not a law
of physics:

`losses.py:91-132` `_match_anchors` is vanilla RetinaNet — IoU≥0.5 positive,
IoU<0.4 negative, plus a single force-match of the best anchor per GT
(`labels[ious[:, gi].argmax()] = ...`). For **small assembly parts**, almost no
anchor clears IoU 0.5, so each GT contributes essentially **one** positive (the
forced one). 4 GT/image → ~4 pos → ~16/batch. RetinaNet works on COCO with the
same loss and the same `pi` because COCO objects are large enough to light up
*many* anchors per GT. **You are not hitting a focal-loss limit; you are hitting
an anchor-coverage limit.** (Your gradient math in file 29 §4 over-attributes
this to `pi`; the lever is positives-per-GT, which assignment controls.)

OHEM then keeps `max(2·n_pos, 32)` negatives (`DET_OHEM_RATIO=2.0`,
`DET_OHEM_MIN_NEG=32`). With `n_pos≈4` that is 8 pos:32 neg *inside* GT frames,
plus 2048 background anchors per empty frame (`DET_EMPTY_SAMPLE=2048`,
`DET_EMPTY_BG_SCALE=0.05`, `losses.py:225-238`). The net bias gradient points
toward "background," and **`DET_BIAS_LR_FACTOR=5.0` (config.py:56) multiplies
that drift by 5×.** You added 5× bias LR to *escape* the equilibrium; if the bias
gradient points toward background (it does, because positives are starved), 5× LR
drives it *into* the equilibrium faster. That is an own-goal.

### 1.4 The symptom chain (why it looks like a "bias equilibrium")

```
head_pose dominates backbone (×40 via Kendall, from ep6)
   └─> shared features drift toward orientation-mean, lose object discriminability
        └─> cls_subnet input becomes uninformative; W·x ≈ const
             └─> weight decay (1e-4) shrinks cls_score.weight toward 0 (no positive
                  gradient strong enough to hold it up)
                  └─> output ≈ bias for every class  →  cls_std → 0.0068
                       └─> bias settles where Σσ(b)≈base-rate ≈ 0.079  →  "−2.54 equilibrium"
                            └─> mAP → 0.001
```

Every number in your DET_PROBE (`score_mean 0.079`, `score_std 0.0068`,
`cls_mean −2.54`, `preds>0.30 = 0`) is explained by "the features feeding the
classifier went dead," not by an intrinsic property of the bias parameter.

**Falsifiable prediction:** log `cls_score.weight.norm()` across ep6→15. It will
*shrink* monotonically while `prec_hp` rises. If it does, the bias-removal /
QFL / pi=0.1 proposals are all treating the wound, not the knife.

---

## 2. Direct Answers to Your 10 Questions (master prompt §6)

**Q1 — Which fix for the "cls bias equilibrium"?**
None of the five, as primary. They target the symptom. Primary fix = stop
head_pose from owning the backbone (§3, Fix 1) + give detection more positives
(§3, Fix 2). If you want a one-liner to buy time today: clamp `lv_hp ≥ lv_det`
(head_pose precision can never exceed detection's). That alone should arrest the
ep6→13 decay.

**Q2 — Is QFL/VFL the right long-term choice?**
Good upgrade, wrong order. QFL/VFL align confidence with localization quality and
remove the hard 0/1 target; they *will* make the head more robust and are
defensible in the paper (they're standard in modern detectors and several recent
egocentric/assembly detection works). But they do **not** fix gradient
starvation or Kendall hijacking — feed a QFL head dead features under a ×40
head_pose and it still collapses. Adopt VFL **after** §3 Fixes 1–2 land, as a
quality improvement, not a rescue.

**Q3 — Why does RF2 "eat" the RF1 checkpoint?**
It doesn't eat a 0.45 — that number is a recording artifact equal to the RF3
gate constant. RF2 starts from a ~0.18 checkpoint and head_pose domination
grinds it to ~0. The forgetting is real; the magnitude you feared is not.

**Q4 — Has PSR ever produced a non-zero gradient?**
Almost certainly never *meaningfully*. PSR is `binary_focal_loss`
(`losses.py:775`) on fill-forward labels where ~20/22 components are 0. Logits
saturate (your −23/+22), so `(1−p_t)^γ ≈ 0` for the dominant negatives → loss
floors at ~1.5e-8. The constant value across runs is the focal loss of a
trivially-correct-on-the-majority-negative predictor, not a frozen graph. It is
also gated off in RF1/RF2 (`train_psr=False`) and only ramps at Kendall stage 3
(ep16+). **Defer it.** Before RF4, prove it can move with a *PSR-only overfit
test* on 50 sequences with `pos_weight`/per-component α already in your code
(`per_component_alpha`, `comp_weights`). If it can't overfit 50 samples, it's an
architecture bug (transformer logit scale), not a training bug. It does not
block the detection paper.

**Q5 — Is HeadPose helping or hurting?**
Both, at different stages. In RF1 (no competing dense task) it *helped* — it gave
the backbone a stable signal (your Kendall-fix evidence is real). In RF2 it
*hurts* because Kendall promotes it to backbone owner. Recommendation: keep it
for stability but **demote it to a read-only consumer** — either fixed small
weight (λ≈0.1–0.3) or detach/scale its gradient into the backbone (you already
use the stop-grad pattern for PoseFiLM/HeadPoseFiLM/det_conf; apply the same
discipline here). Do **not** delete it; a bare detection-only backbone reverts to
the file-29 sparsity problem.

**Q6 — What blind spot persists?**
This document's §1: (a) the Kendall precision asymmetry with no floor protecting
detection, (b) the second, epoch-indexed curriculum inside `losses.py` colliding
with the RF stage manager, and (c) the phantom 0.45 corrupting your gate logic.
"The next hidden failure mode" was never a new mode — it was the multi-task
weighting you fixed *into existence* when you enabled head_pose.

**Q7 — Skip to RF3 (all heads)?**
No. RF3 adds activity — another dense, low-absolute-loss task that Kendall will
*also* amplify (you already had to add `ACTIVITY_LOSS_WEIGHT=0.2` to stop it
dominating PSR). Adding heads before fixing the weighting buries detection
deeper. Fix weighting first; *then* more heads genuinely help.

**Q8 — Gradient leakage from disabled heads?**
Negligible. 0.02–0.05 vs 5+ for active heads is <1%. It's the Kendall log_var
reg + shared-feature forward paths, not a leak worth chasing. Ignore.

**Q9 — Is the 22-agent swarm overkill?**
For finding bugs, it earned its keep. For steady-state, a 5-check set suffices:
(1) PID alive, (2) loss finite, (3) `det_mAP50` not dropping N cycles,
(4) `cls_score.weight.norm` not shrinking, (5) `prec_hp/prec_det` ratio. The
last two would have caught this collapse weeks ago.

**Q10 — Dataset label quality?**
A ceiling factor, not the collapse mechanism. Synthetic projected labels plausibly
cap you near ~0.18 (missed objects, jitter) and could explain Q24's
memorization worry. But 0.18→0.001 in 5 epochs is far too sharp for label noise;
that's a dynamics failure. Audit labels for the *ceiling*, fix weighting for the
*collapse*.

---

## 3. The Fix Plan (prioritized, single-RTX-3060 safe)

> All changes are config/loss-level. None touch the architecture or VRAM budget.
> Do them in order; stop when detection holds past epoch 13.

**Fix 1 — De-fang Kendall for detection-bootstrap stages (highest leverage).**
Pick ONE:
- *Predictable (recommended):* set `use_kendall=False` for RF1–RF2 and use fixed
  weights `total = loss_det + λ_hp·loss_head_pose` with `λ_hp ≈ 0.1–0.3`.
  Detection drives the backbone; head_pose just stabilizes it. Re-enable Kendall
  at RF3+ once detection is real.
- *One-line stopgap:* after `losses.py:1513`, add `lv_hp = torch.maximum(lv_hp,
  lv_det.detach())` so head_pose precision can never exceed detection's.

**Fix 2 — Feed the detector more positives (attack the supply).**
- Lower `DET_POS_IOU_THRESH` 0.5 → 0.4 (or 0.35) and replace the single
  best-anchor force-match with **top-k (k≈9) by IoU per GT**
  (`losses.py:129-130`). Confirm via `MATCH_PROBE` (`losses.py:12`) that
  `pos/GT` rises from ~1 to ~6–10.
- Verify anchor scales actually cover IndustReal part sizes; if parts are small,
  ensure the smallest P3 anchors are small enough. This is the single biggest
  determinant of whether this architecture can detect at all.
- Revert `DET_BIAS_LR_FACTOR` 5.0 → 1.0. It accelerates the wrong drift.

**Fix 3 — Kill the double curriculum.**
The RF stage manager already owns which heads train. Make the epoch-indexed
Kendall staging (`losses.py:1536-1567`) a no-op when the stage manager is driving
(or delete the `prec_* = prec_* * 0` zeroing entirely and rely on the
`train_*` flags). One curriculum, not two.

**Fix 4 — Fix the phantom 0.45.**
Find where `stage_history[*].best_det_mAP50` is written in `stage_manager.py` and
make it record the *observed* best (`best_metric`), not the gate constant.
Otherwise gate/retry decisions compare against a number RF1 never reached.

**Defer:** QFL/VFL (Fix 5, post-recovery quality upgrade), PSR (RF4+, gated
behind a 50-sample overfit sanity test).

---

## 4. Cheap, Decisive Experiments (run before committing to fixes)

| # | Experiment | Cost | Decides |
|---|-----------|------|---------|
| E1 | Overfit 4 GT frames, head_pose OFF, Kendall OFF, 300 steps | minutes | If mAP→0.8: arch is fine, problem is multi-task dynamics (expected). If not: anchor/assignment bug — do Fix 2 first. |
| E2 | Log `prec_det` vs `prec_hp` every epoch | 1 line | Confirms head_pose hits ~54× while detection ~1–2×. |
| E3 | Log `cls_score.weight.norm()` ep6→15 | 1 line | Confirms W shrinks (symptom chain) vs bias-only story. |
| E4 | Backbone grad-norm split by task (file 29 §9, but per-head) | small | Confirms head_pose >> detection on *shared* params. |

E1+E2 together will, in under an hour, tell you whether this is a supply problem
(Fix 2) or a weighting problem (Fix 1) — almost certainly both, in that order of
appearance.

---

## 5. What I'd Tell You If You Only Read One Paragraph

Stop treating the classification head as the patient. The head is flatlining
because you put a ×40 head_pose tourniquet on the shared backbone at epoch 6 via
Kendall, while detection was already on a starvation diet of ~1 positive anchor
per object. Cut the tourniquet (fixed/clamped weights), feed the detector (lower
IoU threshold + top-k matching + right-sized anchors), delete the second hidden
curriculum, and erase the phantom 0.45 that's been telling you a detector existed
when it didn't. Then — and only then — consider VFL and PSR as the quality and
novelty layers for the paper.

---

*Generated 2026-06-20. Verified against `src/training/losses.py`,
`src/models/model.py`, `src/training/stage_manager.py`, `src/config.py`, and
`src/runs/rf_stage_state.json` on branch `claude/epic-hamilton-6sw452`. Where my
reading contradicts an earlier doc, I cite the current line number so you can
check me.*
