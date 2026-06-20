# 39 — OPUS ANSWER v9: You Can't See What You're Steering By (2026-06-21)

> Response to `38_OPUS_MASTER_PROMPT_v9.md`. Every claim below was verified by
> reading the **current** source on branch `claude/sleepy-ride-bn0s3s`
> (`src/training/losses.py`, `src/config.py`, `src/training/train.py`,
> `src/evaluation/evaluate.py`, `src/models/model.py`). Line numbers refer to
> that working tree. Where I am inferring from your reported numbers rather than
> from code I could read, I say so — I cannot see the live RTX-3060 run from
> here, only the code that produced it.

---

## 0. TL;DR — Read This First

The v8 fixes are sound and correctly implemented. But your v9 question — *"are
they working?"* — **cannot be answered from the epoch-17 telemetry you quoted**,
because two of the three signals you're reading structurally cannot see
classification:

1. **`score_p50` is background, by construction.** It is the *median* max-class
   sigmoid over **all ~172K anchors** (`evaluate.py:108-131`), of which
   >99.99% are background. Its expected value is ≈ `sigmoid(bias)` whether or not
   classification works. `0.020–0.072` is the **correct floor**, not evidence of
   a stuck classifier. A perfectly-trained detector would show the same `p50`.

2. **The `DET_PROBE` "LOCALIZING" verdict is IoU-only.** It thresholds boxes at
   `score>0.01`, decodes them, and checks IoU vs GT (`evaluate.py:114-147`). It
   **never checks the predicted class.** "LOCALIZING" means boxes overlap
   objects; it is *silent* on whether the class is right. So it can be green
   while classification is perfect **or** dead.

3. **The one signal that can see it — epoch-end `det_mAP50` at
   `DET_EVAL_SCORE_THRESH=0.001` (`config.py:437`) — hasn't printed yet.** That
   is the only number that collapsed last run (0.184→1e-5), and it is the only
   number that will tell you if the fixes worked. Everything else at epoch 17 is
   consistent with *both* "recovering" and "about to collapse again."

Two more things the code says that the prompt gets wrong:

4. **The "−2.5 bias equilibrium" is essentially the initialization.** RF2 reinits
   `cls_score.bias` from `reinit_pi=0.05` (`config.py:1114` → `train.py:2469`),
   i.e. bias = −log(19) = **−2.94**. A drift to −2.5 is +0.44 — a *small* move.
   The catastrophe v8 named is in the **weights going uniform**, not the bias.
   You still have not added the one probe that proves this
   (`cls_score.weight.norm()`, v8's experiment E3). Add it before the next round.

5. **There is a config split-brain you must resolve before trusting anything.**
   The committed `stage_rf2` preset has **`detach_reg_fpn: True`**
   (`config.py:1109`), but `38 §1` and `38 §10` both state
   `DETACH_REG_FPN=False`. These imply *different mechanisms* (see §2). This is
   the same class of bug that already bit you twice (the config-alias audit; the
   `recovery_det_only` vs `stage_rf1` discrepancy). Print the effective value at
   step 0 and find out which run you actually have.

Net: before the next consultation, make the next epoch-end eval *interpretable*
(§5) instead of asking whether an uninterpretable one looked okay.

---

## 1. Reading your epoch-17 telemetry correctly (Q1, Q3)

### 1.1 `score_p50` derivation — why it is a non-signal

`evaluate.py:107-131`:

```python
sig = 1.0 / (1.0 + np.exp(-cls_preds[i]))   # [N_anchors, 24]
max_scores = sig.max(axis=1)                # [N_anchors]  max over classes
all_max_scores.append(max_scores)
...
"score_p50": pct(max_scores_cat, 50)        # median over ALL anchors, ALL imgs
```

`max_scores_cat` is every anchor in the batch. With ~172K anchors/image and a
handful of GT, the median anchor is background. Its max-over-24-classes score is
an order statistic of 24 values each ≈ `sigmoid(bias) ≈ sigmoid(−2.94) ≈ 0.05`.
So `score_p50 ≈ 0.02–0.07` is **what a healthy detector also looks like**. This
number cannot move when classification recovers, because the *denominator of the
median* is background either way.

**Action:** stop treating `score_p50` as a classification health metric. The
informative fields in the *same* probe are `score_p99`, `preds>0.30`,
`preds>0.50` — and even those are diluted by background. The number you actually
want does not exist yet; add it (§5.1): the sigmoid on the **matched positive
anchors at the correct class**.

### 1.2 The `LOCALIZING` verdict says nothing about classification

`evaluate.py:114-147`: keep boxes with `max_scores>0.01`, decode, compute IoU vs
GT, and the verdict is `LOCALIZING` iff `n_matched = (best_ious>0.5).sum() > 0`.
**Class is never compared.** A model that puts class-7 on a class-6 object at
IoU 0.9 scores `LOCALIZING` and `mAP=0`. So `bestIoU_max=0.86–0.98` confirms the
**regression** subnet works; it is fully consistent with the **classification**
subnet being dead. The "LOCALIZING but not CLASSIFYING" framing in `38 §4` is
half-measured: the LOCALIZING half is real; the "not CLASSIFYING" half is
*inferred from `score_p50`*, which can't see it.

### 1.3 So: is it working?

Honest answer: **unknown from what you quoted.** The decisive read is
epoch-end `det_mAP50` at thresh 0.001 — that measures *ranking* (are correct
class+box ranked above false positives), which is exactly what went to zero last
run and exactly what cannot be faked by a uniform-output classifier. Watch that.
As a cheap *leading* proxy between epoch-ends, watch the positive-anchor score
(§5.1) and `cls_score.weight.norm()` (§5.2) — if the weight norm is holding or
rising while `prec_hp/prec_det` stays bounded, the fixes are biting.

---

## 2. The `detach_reg_fpn` / `reinit_pi` split-brain (Q3, Q6 — new)

`config.py:1075-1115`, `stage_rf2`:

```python
'detach_reg_fpn':  True,    # line 1109   ← prompt 38 §1/§10 say False
'reinit_pi':       0.05,    # line 1114   ← bias = −2.94, not −4.60
```

This must be resolved because it changes the *interpretation of your central
observation* (localizes, doesn't classify):

- **If `detach_reg_fpn=True` (what the committed preset says):** the **regression**
  subnet is detached from the FPN, so *only the classification subnet (and
  head_pose/body_pose) shapes the backbone.* Then your excellent localization
  (bestIoU 0.86–0.98) is produced by the reg subnet riding on features carved by
  **cls + pose**, while the cls subnet — *the one detection task that actually
  touches the backbone* — is the stuck one. That points the finger **away from
  "bad features"** and squarely at the **cls loss / cls targets / labels**
  (§3.2, §3.3). The v8 "head_pose ate the backbone" story gets *weaker* in this
  configuration, because if head_pose had wrecked the features, localization
  would degrade too — and it hasn't.

- **If `detach_reg_fpn=False` (what `38` claims):** both subnets feed the
  backbone and the v8 Kendall-domination story applies straightforwardly.

You cannot diagnose Q3 without knowing which. **Print `C.DETACH_REG_FPN` and
`C.REINIT_PI` at step 0** (§5.4). My working assumption, from the committed
code, is `True` — and under `True`, §3.2/§3.3 are the likeliest culprits, not
Kendall.

---

## 3. Why localization learns but classification stalls (Q3) — three candidates, ranked

### 3.1 Instrumentation (resolve first)
Per §1, you may be over-reading a non-signal. Possible that classification is
fine-ish and only the eval *display* (and the not-yet-printed mAP) is the issue.
Low cost to rule out. Do §5 first.

### 3.2 Label noise on fine-grained classes → Bayes-optimal uniform (Q3 + Q8)
IndustReal labels are synthetic projections. **Box** targets are robust to
class-label noise; **class** targets are not. 24 assembly-part classes, many
near-identical across assembly states, with projection jitter, is precisely the
regime where the **loss-minimizing classifier output is a near-uniform,
low-confidence vector** — i.e., the "uniform ~0.079" you observe is not a bug,
it is the *correct* response to ambiguous/noisy class labels. This unifies your
0.18 ceiling and the drift-to-uniform under one mechanism, and it is *more*
likely if `detach_reg_fpn=True` (regression has clean targets; classification
has noisy ones).
**Falsifiable test:** overfit 50 images, cls-only, head_pose OFF, Kendall OFF
(§6). If it *cannot* memorize 50 images to mAP>0.8, the problem is
loss/assignment (§3.3). If it *can*, the problem is label noise / optimization
at scale, and no amount of Kendall/QFL tuning fixes it — you audit labels.

### 3.3 Top-k force-match with **no IoU floor** poisons the cls head (new — Q6)
Your Fix 2 top-k (`losses.py:138-148`):

```python
_topk = int(getattr(C, 'DET_POS_IOU_TOP_K', 9))
for gi in range(gt_boxes.shape[0]):
    labels[gi_ious.argmax()] = gt_labels[gi]
    if _topk > 1 and gi_ious.numel() > _topk:
        _, topk_idx = gi_ious.topk(min(_topk, gi_ious.numel()), largest=True)
        for idx in topk_idx.tolist():
            if labels[idx] < 0:
                labels[idx] = gt_labels[gi]        # ← no minimum-IoU guard
```

There is **no minimum-IoU floor.** For a GT whose best anchors sit at IoU ~0.2
(plausible for small parts — `ANCHOR_SIZES=(96,160,256,384,512)`, `config.py:300`,
vs parts ≈156px), you now force **9 poorly-localized anchors** to predict class
*c* with target 1.0. Regression tolerates loose anchors (GIoU just learns the
offset); **classification is actively mistaught** that features at a 0.2-IoU
location are class *c*. This can *create* the uniform-output pathology for small
objects precisely *because* of Fix 2 — it trades gradient starvation for label
noise. RetinaNet/ATSS add an adaptive IoU floor (mean+std of candidate IoUs) for
exactly this reason.
**Fix:** gate the top-k by IoU — keep only `topk_idx` with `IoU ≥ ~0.2–0.3`, or
adopt ATSS's per-GT adaptive threshold. Confirm via `MATCH_PROBE`
(`losses.py:12`) that you still get ~6–10 pos/GT on *real* parts, not 9 phantoms.

---

## 4. Direct answers (Q1–Q9)

**Q1 — What signal tells you the fixes work?**
Epoch-end `det_mAP50` at thresh 0.001, holding past ep13–15 (the previous
collapse window). Leading proxies: positive-anchor score (§5.1) rising,
`cls_score.weight.norm()` (§5.2) holding/rising, `prec_hp/prec_det` (§5.3)
bounded. **Not** `score_p50` (can't see it), **not** the LOCALIZING verdict
(IoU-only).

**Q2 — If ep20–25 collapses again, what next? (decision tree)**
1. Confirm it's *ranking* collapse, not display: it will be — at thresh 0.001 a
   uniform classifier genuinely scores ~0 mAP.
2. Run the 50-image cls-only overfit (§6) to bin the cause:
   - **Overfits → multi-task dynamics.** Flip `KENDALL_FIXED_WEIGHTS=True`
     (λ_hp=0.2) for RF2 — you already wired this path (`losses.py:1540`); it is
     strictly more decisive than the `HP_PREC_CAP` clamp for a bootstrap stage.
   - **Doesn't overfit → assignment/labels.** Add the top-k IoU floor (§3.3) and
     audit labels (§3.2). KENDALL won't help.
3. **Do NOT** jump to RF3, QFL, or a new backbone first. RF3 adds activity
   (another dense low-loss task Kendall will amplify — you already needed
   `ACTIVITY_LOSS_WEIGHT` to tame it); QFL feeds the same targets a softer loss;
   a backbone swap is unaffordable on a 3060 and unproven to help. All three
   bury the signal you're trying to read.

**Q3 — Is the LOCALIZING/CLASSIFYING dissociation expected at ep17?**
The *measurement* of it is partly an artifact (§1). The *real* phenomenon —
regression learning faster than classification — is expected and has three
candidate causes ranked in §3. Under the committed `detach_reg_fpn=True`, §3.2
(label noise) and §3.3 (top-k poisoning) are the prime suspects, **not**
features. Resolve §2 to be sure.

**Q4 — Enable `KENDALL_FIXED_WEIGHTS=True` for RF2 now?**
Not yet. Keep it as your clean A/B lever. The `HP_PREC_CAP` clamp
(`losses.py:1586-1587`) is the gentler intervention; let the *first* epoch-end
mAP read under it land. If that read collapses, flip fixed-weights and you get an
unambiguous comparison. Flipping both at once costs you the ability to attribute
the result.

**Q5 — Has PSR ever trained? Care now?**
v8 was right: `psr=1.546e-08` is the binary-focal floor of a predictor that is
trivially correct on the ~20/22 always-zero components — `(1−p_t)^γ→0` on the
dominant negatives (`binary_focal_loss`, `losses.py:805`). It is *not* a frozen
graph; it is a degenerate-but-finite loss, and it's gated off in RF2
(`train_psr=False`). **Don't fix it inside the live run.** Run a **PSR-only
50-sequence overfit** *now*, in parallel — it's fully decoupled. If it can't
overfit 50 sequences, the cause is the transformer logit scale (your −23/+22),
fixable (logit clamp / smaller init / pos_weight) before RF4. This de-risks the
paper's novelty claim without touching detection.

**Q6 — What's the next hidden failure mode?**
Not a new "mode" — same family v8 named (measurement + supply): (a) the
`detach_reg_fpn`/`reinit_pi` split-brain (§2); (b) `score_p50` and the IoU-only
verdict steering you (§1); (c) unbounded top-k injecting positive-label noise
(§3.3); (d) the `score_thresh=0.5` *function default* (`evaluate.py:162`) lurking
in any secondary eval script that forgets to pass `C.DET_EVAL_SCORE_THRESH` —
the *main* eval is safe (0.001), but `quick_eval`/`audit_eval`/diag calls may
silently report near-zero mAP for a fine model. Audit those call sites.

**Q7 — Paper novelty at risk?**
PSR risk is real but **orthogonal** to detection — don't let it block RF2. The
50-seq overfit (Q5) tells you in an hour whether PSR is salvageable. If the
paper's spine is cross-task conditioning + multi-task, you can present PSR
honestly (including "transition modeling required X to train"); a never-trained
PSR reported as working would be the actual integrity problem.

**Q8 — Dataset label quality?**
Likely your true ~0.18 *ceiling* **and** a co-driver of the uniform-output
*collapse* (§3.2). Keep v8's separation: labels explain the ceiling, dynamics +
top-k noise explain the cliff. Audit a few hundred projected boxes for *class*
correctness (not just box overlap) — that's the cheap, decisive read.

**Q9 — Plan RF3 now?**
Plan it, launch it only after RF2 produces one epoch-end `mAP50@0.001` that
**holds for ≥3 consecutive epoch-ends past ep15.** Crisp failure criterion: if
`mAP50@0.001 < 0.10` for 3 consecutive epoch-end evals after ep15, declare RF2
failed, run the 50-image overfit, and branch per Q2. Don't run RF2 to ep36 on
hope — the previous run told you ep13–15 is the verdict zone.

---

## 5. The four probes you're still missing (add before next round)

1. **Positive-anchor score.** In `FocalLoss.forward` after matching
   (`losses.py:~262`), every N steps log `p_t` (sigmoid at the *correct* class)
   on `pos_mask` anchors: mean/median/max. **This is the direct "is the
   classifier learning" signal** that none of your current probes provide.
2. **`cls_score.weight.norm()` per epoch.** v8's E3. Still the single most
   diagnostic line. Confirms the "weights go uniform" mechanism vs a bias-only
   story. You have *never logged this.*
3. **`prec_hp / prec_det` per epoch.** v8's E2. One line; confirms whether
   `HP_PREC_CAP` is actually holding head_pose ≤ detection.
4. **Effective `C.DETACH_REG_FPN` and `C.REINIT_PI` at step 0.** Ends the §2
   split-brain in one print.

---

## 6. The single cheapest decisive experiment

**Overfit 50 images, classification-only** (`train_head_pose=False`,
`use_kendall=False`, 300–500 steps, thresh-0.001 eval on the same 50). Cost:
<30 min, parallel to the live run. Outcome bins the *entire* problem:

| Result | Conclusion | Next |
|--------|-----------|------|
| mAP→0.8+ | Arch + assignment + loss are fine; collapse is **dynamics** | `KENDALL_FIXED_WEIGHTS=True`, leave the rest |
| mAP stalls, boxes localize | **Assignment/label** noise (§3.2/§3.3) | top-k IoU floor + label audit |
| boxes don't even localize | anchor/assignment bug upstream of cls | fix matching before anything else |

Run this *first.* It is worth more than another epoch of the live run, because it
removes the multi-task confound that has cost you four consultation rounds.

---

## 7. What I'd tell you if you only read one paragraph

The v8 fixes are good and correctly coded — but you're asking whether they worked
while reading two gauges (`score_p50`, the LOCALIZING verdict) that physically
cannot measure classification, and the one gauge that can (epoch-end mAP at
thresh 0.001) hasn't printed. Before the next round: (1) settle the
`detach_reg_fpn=True`/`reinit_pi=0.05` split-brain between your code and your
docs (§2) — under `True`, the culprit is the cls loss/labels, **not** the
backbone; (2) add the positive-anchor-score and `cls_weight_norm` probes (§5) so
the next eval is interpretable; (3) put an IoU floor on the new top-k match
(§3.3), which may be poisoning the classifier for small parts; and (4) run the
50-image cls-only overfit (§6), which bins the whole problem in half an hour.
Those four turn the next epoch-end from "ambiguous" into "decisive" — and
decisive is the only thing that ends the cycle.

---

*Generated 2026-06-21. Verified against `src/training/losses.py`,
`src/config.py`, `src/training/train.py`, `src/evaluation/evaluate.py`, and
`src/models/model.py` on branch `claude/sleepy-ride-bn0s3s`. Where my reading
contradicts `38_OPUS_MASTER_PROMPT_v9.md`, I cite the current line number so you
can check me — in particular `config.py:1109` (`detach_reg_fpn: True`) and
`config.py:1114` (`reinit_pi: 0.05`), which disagree with `38 §1/§10`.*
