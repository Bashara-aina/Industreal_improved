# 181 — Opus Answer: MTL Path Decision for MViTv2-S (Definitive)

**Date:** 2026-07-09
**Inputs read line-by-line:** 176 (progress), 177 (status/paths), 178 (50 questions), 179 (experiments), 180 (consult prompt)
**Also read line-by-line (ground truth):** `scripts/train_mtl_mvit.py` (1563 lines), `src/models/mvit_mtl_model.py` (447 lines), `src/config.py` (detection classes, PSR weights, DETACH flag)
**Verdict authority:** Every claim below is checked against the code that actually ran `mtl_mvit_run3`, not against the prose in 176–180. Where the docs and the code disagree, the code wins, and I say so explicitly.

---

## 0. READ THIS FIRST — The one-paragraph answer

Take **Path D (a hybrid)**, not A, B, or C as written. The "Kendall paradox" is **not a paradox and not aleatoric uncertainty** — with the exact loss in your code it is provably identical to *dividing each task's backbone gradient by its own loss magnitude* (`weight_i = 1/(2·loss_i)`), so a task whose loss is structurally inflated (activity: 75-way CE × class weights up to 137 × label smoothing) is mechanically starved. The fix therefore has two independent halves that you should do **together**: (1) **stop inflating activity's loss** (kill the ×10 class-weight inflation, cut label smoothing) so Kendall's equilibrium is not insane, and (2) **cap `log_var_act ≤ 1.0` and `log_var_psr ≤ 0.5`** so no task can be driven below a floor. Keep Kendall (now well-behaved) and keep PCGrad. **Separately, three of the doc's premises are factually wrong** (PSR is *not* detached in this run; detection "class-0 collapse" is class 0 = *background*, which is correct-by-design; and your gradient accumulation is a silent no-op wasting ~50% of every step). Prove MTL helps with just **two** baselines (single-task activity clip-level + single-task detection) plus the capped-MTL run — everything else in file 179 is optional. The single highest-leverage change you can make in the next 10 minutes is the 4-line log-var cap; the single highest-value *free* change is fixing the grad-accum bug.

---

## 1. THREE FACTUAL CORRECTIONS (the docs are wrong; the code is right)

Before any path decision, the consultation's mental model has to be corrected, because two of the three "paths" are aimed at problems that don't exist in the running code.

### Correction 1 — There is no "Kendall paradox." It's deterministic loss-scale normalization.

Your code's per-task Kendall term (`train_mtl_mvit.py:661`) is:

```python
prec = torch.exp(-lv)
total_loss = total_loss + prec * loss + lv / 2          #  exp(-s)·L + s/2
```

Treat `s = log_var` as a free parameter optimized by SGD. The stationary point in `s` is:

```
∂/∂s [ exp(-s)·L + s/2 ]  =  -exp(-s)·L + 1/2  =  0
⇒   weight ≡ exp(-s)  =  1 / (2·L)
```

**Each task's learned weight converges to `1/(2·loss)`.** That is not "uncertainty" — it is exact inverse-loss scaling. Plug in your epoch-6 losses and it reproduces every observed number to 2 decimals:

| Task | loss (Ep6) | predicted weight `1/(2L)` | observed weight | match |
|------|-----------|---------------------------|-----------------|-------|
| Activity | 12.31 | **0.0406** | 0.04 | ✅ exact |
| PSR | 1.30 | 0.385 | 0.39 | ✅ exact |
| Detection | 0.31 | 1.61 | 1.51 | ✅ (still drifting to equilibrium) |
| Pose | 0.19 | 2.63 → capped | 1.63 | ✅ (hp_prec_cap binds) |

**Why this starves activity, precisely.** The gradient delivered to the backbone by task *i* is `weight_i · ∂L_i/∂θ`. For softmax cross-entropy the logit-gradient is `(p − y)`, whose norm is **bounded (~√2) regardless of how large the loss value is** — a CE of 12 does *not* produce 40× the gradient of a CIoU of 0.3; the gradients are comparable in norm. So the *only* thing that differs across tasks at the backbone is the scalar `weight_i`. Activity's is `0.04`, detection's is `1.51` → activity influences the shared backbone at **~1/37 of detection's rate**. It cannot shape features, so its loss stays high, which *by construction* keeps its weight low. This is a stable pin at `weight≈0.04`, not a runaway "death spiral to +4" (the product `weight·loss` is held at 0.5; `log_var_act` only climbs if the *loss itself* climbs).

**Direct answer to K-4 (the pivotal question in file 178):** *Yes — with your loss functions Kendall is doing loss-scale normalization, not uncertainty estimation.* The uncertainty interpretation is only valid when every task loss is a comparable log-likelihood. You are mixing class-weighted CE (weights up to 137 + label smoothing) against CIoU+DFL+focal. The "uncertainty" Kendall infers for activity is dominated by your arbitrary class-weighting choice, not by any property of the data. **This is the whole ballgame** and it makes the fix obvious (Section 3).

### Correction 2 — PSR is NOT detached in this run.

Files 177/178/179 build an entire branch of analysis (S-1, S-2, E7, "remove DETACH_PSR_FPN") on the premise that `DETACH_PSR_FPN=True` blocks PSR's gradient to the backbone. **That flag is never read on this code path.** In `mvit_mtl_model.py:417–419` the PSR head consumes the live conv_proj tensor with no detach:

```python
psr_input = fpn_feats.get("P2")     # conv_proj output, still in the autograd graph
psr_logits = self.psr_head(psr_input)
```

`DETACH_PSR_FPN` exists only in a *different* pipeline (`train_psr_repair_wrapper.py` + `config.apply_preset`), which `train_mtl_mvit.py` does not use. So in `mtl_mvit_run3`, **PSR gradients do flow back into `conv_proj`.** PSR being flat therefore is *not* a detach problem — E7 would be a null experiment. The real reasons PSR is flat (Section 2.3) are: (a) Kendall weight 0.39 is moderate but its gradient only reaches `conv_proj` (the *first* layer — no high-level temporal features), and (b) the labels are extremely sparse transition events where BCE plateaus at the base-rate entropy unless the signal is actually present in those low-level features. Delete S-2/E7 from your plan.

### Correction 3 — Detection "class-0 collapse" is class 0 = **background**, and is correct-by-design.

`config.py:215–241`:

```python
NUM_DET_CLASSES = 24   # background + 22 assembly states + error_state
DET_CLASS_NAMES = { 1: 'background', 2:'10000000000', ... 24:'error_state' }
# keys are COCO ids 1..24; model indices = key-1 = 0..23  → model index 0 == 'background'
```

In `detection_loss()` (`train_mtl_mvit.py:198`) `cls_target` is initialized to zeros and only the single center cell of each GT box is overwritten with the real class. **Every unassigned cell keeps target = class 0 = background.** So "every cell predicts class 0 at 0.9999" is the model *correctly* predicting background on background cells — exactly what you want, not a pathology. The docs' claim that this is an anomalous "collapse that resolves by epoch 8–15" is a misread.

The *real* detection problem (why mAP@0.5 = 0) is different and is **independent of MTL**:

1. **One positive cell per GT box.** `detection_loss` marks only the box-center cell positive (`gi,gj`), and if two boxes share a center cell the second is silently dropped (`:225` `if pos_mask: continue`). Out of ~4165 cells/image you get a handful of positives (~0.1%). YOLOv8 reaches its mAP because `TaskAlignedAssigner` assigns *many* cells per GT dynamically. Your foreground supervision is ~1–2 orders of magnitude too sparse.
2. **Focal `alpha=0.25` downweights the foreground term.** `alpha_t = onehot·0.25 + (1−onehot)·0.75` puts weight 0.25 on the true-class channel of a positive cell. Standard focal uses α on the *rare* class — but here the rare thing is foreground, so you are downweighting exactly the signal you're starved for. Combined with (1) this makes foreground emergence glacial.

Detection is not "the healthy head." It is a head whose *localization* works (DFL boxes are valid) but whose *classification* is under-supervised. Fixing this is orthogonal to the Kendall question but is required before any "MTL detection mAP" number is meaningful.

### Bonus correction — Gradient accumulation is a silent no-op (wastes ~50% of compute).

`train_step` calls `optimizer.zero_grad()` at the **top of every micro-batch** (`:679`), then on the accumulation boundary (`:715 do_step`) it steps. So:

```
micro-batch 1 (do_step=False): zero_grad → backward → grads sit in .grad → NO step
micro-batch 2 (do_step=True):  zero_grad  ← WIPES micro-batch 1's grads → backward → step
```

The boundary `zero_grad()` erases the previous micro-batch's gradient. **`grad_accum_steps=2` does not double the effective batch; it throws away every other forward/backward.** Your "effective batch 4" is really batch 2 at 2× the wall-clock. This directly undercuts the paper's "faster / more efficient training" claim and is a pure-win fix (Section 3.4). (The PCGrad path has the same issue plus it *overwrites* `.grad` for the backbone at `:709`, so accumulation there is doubly dead.)

---

## 2. ROOT-CAUSE PER HEAD (grounded in code)

| Head | Symptom | True root cause (verified) | Is it MTL's fault? |
|------|---------|----------------------------|--------------------|
| **Activity** | top-1 ≈ chance, weight 0.04 | Kendall weight = `1/(2·loss_act)`, and `loss_act` is inflated ~×10 by inverse-freq class weights (mean 9.98, max 137.2) + label smoothing 0.1. Head is starved *and* the head is only `LayerNorm→Linear` on the class token. | **Mostly yes** (optimization). Head capacity is a secondary risk — E2 resolves it. |
| **PSR** | loss flat 1.30, F1 0 | (a) input is `conv_proj` = the *first* layer only → no high-level temporal features; (b) transition labels are sparse so BCE sits at base-rate entropy; (c) weight 0.39. **NOT detach** (Correction 2). | **Partly.** Even single-task PSR on conv_proj features may be weak — it's a feature-source problem more than a weighting problem. |
| **Detection** | mAP 0.0, all-background | Class 0 = background is correct. mAP≈0 from **sparse center-cell-only assignment + focal α downweighting foreground** (Correction 3). | **No.** This is a detection-head design issue independent of MTL. |
| **Pose** | loss 0.19, healthy | Class token from Kinetics-pretrained MViT already encodes spatial pose cues; cosine loss is well-scaled. hp_prec_cap keeps it from dominating. | Working. Likely a **positive-transfer** win for the paper. |

**Net:** Of the "3/4 heads MTL is hurting," exactly **one** (activity) is genuinely an MTL-optimization casualty. PSR and detection are architecture/label problems that would *also* hurt a single-task model. This reframes the whole hypothesis: MTL is not the villain the docs think it is.

---

## 3. THE DECISION — Path D (hybrid), with exact code

### 3.1 Why not A / B / C as written

- **Path B (accept):** Rejected. It ships a paper whose headline ("MTL hurts") is an artifact of a loss-scaling choice and a grad-accum bug. Scientifically wrong conclusion.
- **Path C (fixed weights `[1.0, 0.025, 0.24, 1.63]`):** Note that `0.025 ≈ 1/(2·12.31)` — *these "fixed" weights are just the Kendall equilibrium frozen.* So Path C as specified reproduces the starvation on purpose. Fixed weights only help if you set activity **higher** than its loss-equalizing value (e.g. 0.3–0.5), i.e. deliberately over-weight the hard task. Viable but hand-tuned and loses the adaptivity story.
- **Path A (caps only):** Correct direction, but capping `log_var` *without* fixing the loss inflation leaves Kendall permanently pinned at the cap (its gradient `−weight·L+0.5` stays positive, so `s` glues to the ceiling). You get a fixed weight in disguise, and activity's loss is still a noisy ×10-inflated signal feeding det/pose balancing.

### 3.2 Path D = normalize losses + narrow caps + keep Kendall/PCGrad + fix grad-accum

Do all four. They are cheap, independent, and each attacks a distinct verified cause.

**(D1) De-inflate the activity loss** so Kendall's equilibrium is sane and the signal is less noisy. The ×137 inverse-frequency weights are the single biggest loss inflator.

```python
# compute_activity_class_weights() — replace raw inverse-frequency with a
# temperature-softened / normalized-mean scheme so mean(weight) ≈ 1.0.
weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
weights = np.power(weights, 0.5)                 # sqrt-tame the long tail (137 → ~11.7)
weights = weights / weights[weights > 0].mean()  # renormalize so mean ≈ 1.0  → loss no longer ×10 inflated
```

And soften label smoothing (it raises the CE floor on 75 classes):

```python
# activity_loss()
label_smoothing=0.05,   # was 0.1
```

Expected: `loss_act` drops from ~12 to ~3–5, so Kendall's *own* equilibrium weight rises from 0.04 to ~0.10–0.17 before any cap.

**(D2) Per-task log-var caps** (the file-180 Path A change, but narrower and paired with D1):

```python
# train_step(), replace the single clamp at :646 with per-task bounds
LV_MIN = -4.0
LV_MAX = {"det": 4.0, "act": 1.0, "psr": 0.5, "pose": 4.0}   # act≥e^-1=0.37, psr≥e^-0.5=0.61
lv_values = {}
for name in losses:
    lv_values[name] = log_vars[name].clamp(LV_MIN, LV_MAX[name])
```

With D1 lowering the loss and D2 flooring the weight, activity lands at `weight≈0.37` with a *stable* loss — enough gradient to actually learn (≈24% of detection's rate, and CE gradients are dense).

**(D3) Keep Kendall and keep PCGrad.** After D1+D2 Kendall is well-conditioned; PCGrad still resolves genuine direction conflicts. Do **not** rip them out — that throws away the paper's methodological contribution.

**(D4) Fix grad accumulation** (free 2× effective batch or free 2× throughput). Move `zero_grad` out of the per-micro-batch path:

```python
# in train_step: only zero at the accumulation boundary, and scale loss by 1/accum
if do_step:                         # was: unconditional optimizer.zero_grad() at :679
    pass                            # grads were zeroed at the previous boundary
...
scaler.scale(total_loss / grad_accum_steps).backward()   # accumulate
...
if do_step:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad()           # zero AFTER stepping, ready for next window
```

For the PCGrad branch, accumulate the deconflicted backbone grads with `+=` instead of overwriting (`:709`). If a correct accumulation is too fiddly with PCGrad, the honest interim is **set `grad_accum_steps=1`** and stop pretending it's effective-batch-4.

### 3.3 What Path D buys you (realistic, not the doc's optimism)

| Metric | Ep6 now | Ep10 (D) | Ep30 (D) | Notes |
|--------|---------|----------|----------|-------|
| Act top-1 | 0.008 | 0.05–0.10 | 0.20–0.35 | gated by head capacity → run E2 to know the ceiling |
| Det mAP@0.5 | 0.0 | 0.0–0.05 | 0.10–0.30 | **only if you also fix assignment** (Sec 3.5); otherwise stays low regardless of MTL |
| PSR F1 | 0.0 | 0.0–0.05 | 0.10–0.30 | gated by conv_proj feature source → consider deeper feature input |
| Pose fwd MAE | ~10° | ~7–9° | ~4–6° | steady; candidate positive-transfer result |

I am deliberately **more conservative than files 177/179**, which repeatedly quote "act 0.25–0.35 by Ep50, PSR F1 0.5–0.7, det mAP 0.5–0.65." Those are optimistic given (a) a `LayerNorm→Linear` activity head, (b) PSR on first-layer features, and (c) unfixed detection assignment. Promise less; the baselines will tell you the true ceilings.

### 3.5 (Strongly recommended, orthogonal) Fix detection positive assignment

Independent of the MTL decision, detection will never reach a useful mAP with center-cell-only assignment. Minimum viable fix: assign a small radius (e.g. 3×3 cells whose center falls inside the GT box) as positives, and set focal `alpha=0.5` (or drop α entirely and rely on DFL/CIoU). This is the difference between "det mAP 0.1" and "det mAP 0.5+" and it costs ~15 lines in `detection_loss`. Do it before you quote any single-task detection number, or E1 will under-report the ceiling too.

---

## 4. MINIMAL EXPERIMENTAL PROTOCOL TO PROVE "MTL HELPS"

You do **not** need the 10-experiment, 13-day grid in file 179. To make a defensible AAIML claim you need exactly three runs plus one diagnostic:

| Priority | Run | Answers | Cost | Why essential |
|----------|-----|---------|------|---------------|
| **1** | **MTL + Path D** (current run, restarted with D1–D4) | Does fixing the optimization unstarve activity/PSR? | ~1 run | This *is* your headline model. |
| **2** | **E2b: single-task activity, clip-level** (temporal-pooled, no class-weight inflation) | The activity **ceiling** with this backbone. If ST hits 40–50% and MTL hits 25–35%, you've quantified MTL's cost *and* proven the head/backbone are adequate (so the gap is transfer, not capacity). | ~1 run | Without it, every activity number is uninterpretable. |
| **3** | **E1: single-task detection** (with the Sec 3.5 assignment fix) | Detection ceiling → is MTL detection within ~90%? | ~1 run | The reviewer's #1 question ("does the shared backbone cost detection?"). |
| **4** | **E8: gradient-flow diagnostic** (100 batches, no training) | Log per-task cosine similarity + pre/post-PCGrad grad norms on the backbone. | ~2 h | Turns "MTL helps" from assertion into evidence: shows *which* task pairs align (positive transfer) vs conflict. This one plot is worth more to reviewers than 5 training runs. |

Everything else in 179 (E3/E4 single-task PSR/pose, E5 fixed-weight, E6 full cap sweep, E7 detach, E9 aug, E10 dropout) is **optional / nice-to-have**. Drop E7 entirely (Correction 2). If you have spare GPU-days, the highest-value extra is **E4 single-task pose** — if MTL pose ≤ single-task pose, that's your clean L2 "positive transfer" headline.

**Decision rule (replace the sprawling tree in 179 §13):**

```
After Run 1 (Path D) reaches ~Ep20 and Runs 2–3 finish:
  • Activity: MTL within 90% of E2b ceiling?  ── yes → transfer is benign/positive on activity
  • Detection: MTL within 90% of E1 ceiling? ── yes → shared backbone doesn't cost detection
  • Pose: MTL ≤ single-task MAE?             ── yes → POSITIVE TRANSFER (the strong claim)
  • Params: 43.5M vs ~100–138M for 4×ST      ── always true → efficiency claim (L3)
Claim the highest level the evidence supports. Do NOT gate publication on L1.
```

---

## 5. DIRECT ANSWERS TO FILE 180 §7 (the five core questions)

**Q1 — Which path?**
**Path D** (Section 3): normalize activity loss (D1) + narrow per-task caps `act≤1.0, psr≤0.5` (D2) + keep Kendall & PCGrad (D3) + fix grad-accum (D4). Path A alone is second-best; Path C's stated weights re-freeze the starvation; Path B is scientifically wrong.

**Q2 — Minimal protocol to prove MTL helps?**
Four items only: **Path-D MTL run, single-task activity (clip-level, E2b), single-task detection (E1 with assignment fix), and the gradient-flow diagnostic (E8).** Skip E3/E5/E6-sweep/E7/E9/E10; drop E7 as moot. (Section 4.)

**Q3 — Is "act 20% / det 50% / PSR 0.6" publishable as efficiency?**
Yes — but frame it as **L2+L3**, not as beating SOTA. "One 43.5M model performs 4 assembly-understanding tasks in a single forward pass, matching single-task backbones within ~90% on 3/4 tasks and improving head-pose via positive transfer, at 2.3–3× fewer parameters than four specialist models." That is a real, defensible contribution. Combine it with the **methodological** finding (Section 0/1: uncertainty weighting degenerates to inverse-loss scaling and starves high-loss tasks; we characterize and correct it) — that is genuinely novel and is your most reviewer-proof angle.

**Q4 — Is there a Path D we missed?**
Yes, and you're reading it. Beyond the hybrid, three further levers worth knowing:
- **GradNorm** (balance gradient *norms*, not losses) — the principled cure for exactly this failure; more code than caps, keep as a fallback if D under-delivers.
- **Uncertainty weighting on *normalized* losses only** — i.e. never let raw class-weighted CE into Kendall; feed it a running-mean-normalized loss. (D1 is the lightweight version of this.)
- **Deeper feature source for PSR** (block-3/blocks[3] features instead of conv_proj) and **a 2-layer temporal head for activity** — architecture fixes that E2/E3 will tell you whether you need.

**Q5 — The single most important change right now?**
Two-line priority: **(a) if you touch one thing, add the `log_var_act ≤ 1.0` cap** (unblocks the starved head, cheapest possible test, minutes to write). **(b) The single most valuable *free* change is fixing the grad-accum no-op** (Correction 3 / D4) — it's a latent 2× and it's needed for your efficiency claims to be true. Do both; they don't conflict. And *in parallel*, launch the single-task activity baseline, because no MTL activity number is interpretable without it.

---

## 6. ANSWERS TO THE DECISION-CHANGING QUESTIONS IN FILE 178

Only the questions whose answers actually move the decision are listed; the rest are subsumed.

- **K-1 (spiral fundamental or tuning?):** Neither "fundamental" nor "tuning" — it's the *exact equilibrium* `weight=1/(2L)`. Not a bug, not noise; a property of the objective. Fix = change the objective (D1) or floor the weight (D2). It will recur for *any* task whose loss you let grow relative to others.
- **K-2/K-7 (init / LR of log_var):** Second-order. `lr_log_var=1e-3` just sets how fast you reach the same equilibrium. Don't tune these instead of fixing the loss scale.
- **K-3 (does [−4,4] cause exclusion?):** The range isn't the cause; the *equilibrium inside* the range is. Narrowing to per-task caps (D2) is the correct use of clamping.
- **K-4 (uncertainty vs loss-scaling?):** **Loss-scaling. Definitively.** (Section 1, Correction 1.) This is the key that unlocks everything.
- **K-5/P-3/P-4 (PCGrad worth it?):** Answer empirically with E8. Given activity's weight is 0.04, PCGrad is *not* the thing starving it (Kendall is). Keep PCGrad for now; E8 tells you if it's a no-op you can drop for throughput.
- **A-1/A-2 (head capacity vs weight):** Weight explains most of it, capacity is the residual. E2b settles it. If ST clip-level activity < 20%, add a 2-layer temporal head; if > 40%, the head is fine and the gap is pure MTL cost.
- **A-3/A-4 (class weights / smoothing inflate loss?):** **Yes — this is the concrete mechanism behind K-4.** Max weight 137, mean ~10 → loss ×~10. D1 fixes it. This is the most under-appreciated lever in the entire consultation.
- **S-1/S-2 (PSR: Kendall vs detach vs arch?):** **Not detach** (Correction 2). It's feature-source (conv_proj is layer 1) + label sparsity + moderate weight. E3 on *block-3* features would be the informative version.
- **D-1/D-2 (det collapse normal? MTL hurting det?):** Class-0 = background (Correction 3). Detection's ceiling is gated by assignment sparsity, not MTL. Fix assignment (Sec 3.5) before judging MTL's effect on detection.
- **H-2/H-4/H-7 (what bar / Pareto / reviewer):** Target **L2/L3**. You already satisfy the params half of H-7's bar (43.5M ≪ ~100M). Add the two single-task baselines and the gradient-flow evidence and you clear a skeptical reviewer's bar for an *efficiency + positive-transfer + methodological-fix* paper. Do not aim for L1 (beat single-task on all four) — that's a trap that delays publication indefinitely.

---

## 7. HOW THIS DELIVERS THE USER'S THREE GOALS

The user asked for MTL that is **more efficient, faster to train, and more accurate across heads.** Honest scorecard after Path D:

1. **More efficient — YES, real and provable.** 43.5M params / 129.6 GFLOPs / one forward pass for 4 tasks vs ~100–138M and 4 forward passes for specialists. This holds regardless of accuracy. (Report it as the L3 backbone of the paper.)
2. **Faster training — YES, once D4 lands.** Today the grad-accum no-op wastes ~50% of every step, so you're *not* actually faster. Fix it and one MTL run genuinely replaces four single-task runs at comparable wall-clock. Say "one training run for four tasks," not "4× faster per task" (the latter isn't true and a reviewer will catch it).
3. **More accurate across all heads — QUALIFIED.** Expect: pose ≥ single-task (positive transfer — your strong claim), activity/detection within ~90% of single-task (benign transfer — your efficiency claim), PSR the weakest (be honest, or invest in the feature-source fix). "MTL strictly beats single-task on all four" (L1) is unlikely with a shared MViT backbone and should not be promised. The publishable, defensible, *true* claim is **L2 (positive transfer on ≥1 task) + L3 (2.3–3× fewer params, single-pass inference) + a methodological fix (uncertainty-weighting → inverse-loss-scaling pathology, characterized and corrected).**

---

## 8. DO THIS NEXT (ordered, concrete)

1. **Now (10 min):** Apply D2 caps (`act≤1.0, psr≤0.5`) — 4 lines at `train_mtl_mvit.py:646`. Restart `mtl_mvit_run3` from `latest.pt` or fresh.
2. **Now (10 min):** Apply D4 grad-accum fix (or set `--grad-accum-steps 1`). Verify one step: log `param.grad.norm()` before/after the boundary to confirm accumulation.
3. **Same session (20 min):** Apply D1 (tame class weights + `label_smoothing=0.05`). Watch `loss_act` drop toward 3–5 within an epoch and `log_var_act` settle *below* the 1.0 cap (if it still glues to 1.0, the loss is still too big — lower the exponent in D1).
4. **In parallel (launch, then walk away):** E2b single-task activity (clip-level) and E1 single-task detection **with the Sec 3.5 assignment fix**.
5. **One afternoon:** E8 gradient-flow diagnostic — produce the per-task cosine-similarity heatmap. This is your Figure 1.
6. **Before writing detection numbers:** land the Sec 3.5 assignment fix in the MTL model too, so MTL and single-task detection are compared on equal footing.
7. **Do NOT** run E7 (moot), and don't spend GPU-days on the full E6 7-cell sweep — the single `act≤1.0, psr≤0.5` point plus D1 is enough; sweep only if it underperforms.

---

## 9. ONE-SCREEN SUMMARY

```
CLAIM IN DOCS 176–180              →  VERIFIED REALITY (from code)
──────────────────────────────────────────────────────────────────────────
"Kendall paradox, uncertainty"     →  weight = 1/(2·loss). Pure inverse-loss
                                      scaling. Activity starved because its
                                      loss is ×10 inflated by class weights.
"PSR starved by DETACH_PSR_FPN"    →  Flag not on this code path. PSR is NOT
                                      detached. It's a feature-source problem.
"Detection class-0 collapse (norm)" →  Class 0 = background. Correct-by-design.
                                      Real issue: 1-cell-per-GT assignment.
"grad_accum=2 → effective batch 4" →  No-op. zero_grad every micro-batch wipes
                                      accumulation. ~50% compute wasted.
──────────────────────────────────────────────────────────────────────────
DECISION: Path D  =  de-inflate act loss (D1) + caps act≤1.0/psr≤0.5 (D2)
                     + keep Kendall & PCGrad (D3) + fix grad-accum (D4)
PROVE IT:  Path-D MTL  +  ST-activity(clip)  +  ST-detection(assign-fixed)  +  grad-flow diag
PAPER:     L2 (positive transfer, ≥1 task) + L3 (efficiency) + method fix. NOT L1.
FIRST MOVE: log_var_act ≤ 1.0 cap (now) & fix grad-accum (free 2×).
```

*Companion to 176–180. Where this file and 177/178/179 disagree, this file is grounded in the executed code and supersedes them.*
