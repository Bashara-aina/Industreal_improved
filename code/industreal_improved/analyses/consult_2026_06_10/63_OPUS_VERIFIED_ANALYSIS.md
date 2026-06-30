# 63 — Opus Verified Analysis of Files 56–61 (line-by-line, code-checked)

Date: 2026-06-30 · Model: Opus 4.8 · Method: read 56–61 in full, then verified every
load-bearing claim against `src/` in this checkout. Each finding below cites the code
that proves it. Where I could not verify (no data/GPU here), I say so.

---

## PART 0 — The headline correction (this changes everything)

> **The "0.010 invariant activity gradient → 312× starvation" story — the spine of
> files 56, 57, 59, and 61 — is a measurement artifact. It is not real.**

### 0.1 What the probe actually measures

The numbers come from `_log_per_head_grad_norm` (`train.py:2345`). Read its loop
(`train.py:2362–2383`): for each head prefix it walks `named_parameters()` and records

- `first_grad` = `‖grad‖` of the **first** parameter tensor it sees, and
- `last_grad` = `‖grad‖` of the **last** parameter tensor it sees,

then prints `prefix:ALIVE[first]/ALIVE[last]`. It **never sums** the head. So:

- `activity_head: ALIVE[0.0102]/ALIVE[0.00255]` =
  `‖grad of activity_head.proj_features.weight‖ = 0.0102` (first param, a 512×1048
  matrix) and `‖grad of the classifier's last param‖ = 0.00255`.
- `psr_head: ALIVE[3.180]` = `‖grad of psr_head's first param‖` — a different tensor
  with a different shape, fed by a T=8 sequence over 11 components.

Comparing `0.0102` (activity's first-layer weight) to `3.180` (PSR's first-layer
weight) and calling it a "312× gradient gap between heads" (file 57 §"Gradient
Ratios", file 59 line 48–54) is comparing two unrelated tensors. **The ratio is
meaningless.** There is no measured "total activity-head gradient" anywhere in the
logs — so "the activity gradient is 312× weaker" was never actually observed.

### 0.2 The same number disproves the feature-bank-detach hypothesis

Files 56 (Q2, Q8), 57 (Q3), 60 (Q2) all ask: *does `feature_bank` detach the
gradient before it reaches `proj_features`?* The probe already answers it: **no.**
`activity_head.proj_features.weight` reports `ALIVE[0.0102]`. If the bank severed the
graph, that weight's `.grad` would be `None`/`DEAD`. It is alive. Confirmed by the
code path:

- `FEATURE_BANK_DETACH=True` but `FEATURE_BANK_DETACH_GRAD_ENTRIES_ONLY=True`
  (`config.py:814–815`).
- In that mode the stored bank entries are detached, **but the live current frame is
  written back into slot −1 with gradient intact**: `bank_i[-1] = feat_i`
  (`model.py:1241`) and again `bank_seq[:, -1, :] = proj_feat` in the head
  (`model.py` ActivityHead.forward, the `FEATURE_BANK_SLOT_OVERWRITE` branch).
- So gradient flows: CE → classifier → CLS pooling → 2×ViT → TCN → slot −1 →
  `proj_feat` → `proj_features.weight`. The 0.0102 is the proof it arrives.

**Close Q2, Q3, Q8 (file 56/57) and Q2 (file 60): the bank is not the problem.**

### 0.3 Why the number is "invariant" (the part that spooked everyone)

File 60 Q4 calls the invariance "the most concerning observation." It is fully
explained, no bug required:

1. **Invariant across LR (0.5×→20×):** the gradient at a fixed weight state does not
   depend on the learning rate. LR scales the *update* `θ←θ−lr·g`, not `g`. Measuring
   `‖g‖` and varying LR cannot move it. (File 61 "What We've Ruled Out → LR too low"
   is a tautology, not a finding.)
2. **Invariant across blend ratio (0.05→1.0):** `ACTIVITY_GRAD_BLEND_RATIO`
   (`model.py:2105`, `c5_mod_blend = blend*c5_mod + (1-blend)*c5_mod.detach()`)
   changes the gradient flowing **into the backbone**, not the head's own parameter
   gradients. Invariance of `proj_features.grad` is expected.
3. **Invariant across "classifier reinit":** the reinit (file 56 line 42–46) touched
   only `activity_classifier`. `proj_features`, the ViT, and the TCN were untouched —
   so the *first*-param grad it measures was never reset.
4. **Invariant across epochs:** the head is in a collapsed equilibrium — near-constant
   logits → near-uniform softmax → a near-constant CE gradient w.r.t. logits → a
   near-constant signal backpropagated to `proj_features`. Self-consistent. The
   constancy is a *symptom of collapse*, not its cause.

**Net:** 5 of the 6 attempts (the LR/blend/clip/reinit escalation) were optimizing
against a diagnostic artifact. Stop. None of those knobs can move that probe.

---

## PART 1 — What is really happening (verified mechanism)

The collapse is the ordinary, expected failure of **a high-capacity head + tiny,
viciously long-tailed data + no real temporal signal**, in that order of importance.

### 1.1 The temporal head is fed non-temporal data (verified)

- Training calls the model per-frame with real ids: `video_ids = [m['recording_id']
  …]` then `model(images, video_ids=video_ids, …)` (`train.py:1354–1355`).
- The sampler is a class-balanced `WeightedRandomSampler` (`train.py:340` →
  `industreal_dataset.py:get_sampler`, `WeightedRandomSampler` at line ~1481), so
  consecutive items are unrelated frames from arbitrary recordings.
- `FeatureBank.forward` builds the window by appending to a per-`recording_id` list in
  **arrival order** (`model.py:1179–1244`). Under a shuffled sampler that arrival
  order is not temporal. The resulting `[B,16,512]` "sequence" is shuffled frames.
- The TCN (depthwise conv over the time axis, `model.py:1031–`) and the 2×ViT then
  model "motion" over noise. 15 of 16 slots are detached; only slot −1 carries
  gradient. So the 8.2M-param temporal stack is **pure overfitting capacity with no
  usable input** — exactly what makes a 72-way head collapse to the majority class on
  3.7k frames.

(Note: the *specific* ViT-collapse bug from attention mis-scaling is already fixed —
`model.py:1117–1122` now uses `qkᵀ·d^−0.5`, and `pos_embed` is correctly sliced
`[:, :T, :]` at `model.py:1104`, so file 60 Q1's "1024-pos embed on 1 token" concern
is moot: there are 17 tokens and the embed is sliced. The problem is the *input*, not
the ViT internals.)

### 1.2 The data is the binding constraint (from the docs, consistent w/ code)

3,667 train frames, 72 classes, **46 of 72 < 1%** (file 57 lines 144–156). Frame-level
72-way classification at that scale, with a model that has enough capacity to memorize,
has a degenerate global optimum: emit the majority direction. CB-weighted CE + label
smoothing (`USE_LDAM_DRW=False` everywhere, `config.py` presets; CE rebuilt with class
weights at `losses.py:1108–1116`; `CB_LABEL_SMOOTHING=0.1`, `config.py:557`) reshapes
the optimum a little but cannot manufacture signal that isn't in 33 frames of a class.

### 1.3 So the effective fix is: shrink the head, kill the fake temporal path

This is what I already shipped (see PART 3). It is the single highest-leverage move and
it is the correct reading of file 56 Q6, file 57 Q1, file 60 Q5 — all of which propose
exactly this and are all answered **yes** by the analysis above.

---

## PART 2 — Direct answers to all 31 questions (file 60)

Legend: ✔ verified in code · ◑ reasoned (no data here) · ✘ rejected.

**Activity collapse (1–12)**
- **Q1 ViT/TCN broken for single-frame?** ◑ Not "broken" — *misapplied*. It's fed
  shuffled frames (§1.1), so it adds capacity without signal. pos_embed is sliced and
  attention scaling is already fixed (§1.1), so the internals are fine.
- **Q2 feature_bank detaches grad?** ✘ No — `proj_features.weight` is ALIVE[0.0102]
  (§0.2). Disproven by your own probe.
- **Q3 NaN-guard zeroing grad?** ◑ Unlikely to be the driver. The guard
  (`model.py:2117` region) only fires when `proj_feat` is non-finite; if it fired every
  step you'd see DEAD, not ALIVE. It's a tail-risk safety net, not the mechanism.
- **Q4 why 0.010 everywhere?** ✔ Measurement artifact + collapsed equilibrium (§0.1,
  §0.3). Not starvation.
- **Q5 simple Linear(512→75) fixes it?** ✔ Yes — shipped as `ACTIVITY_HEAD_SIMPLE`
  (PART 3). Best single move.
- **Q6 long-tail incompatible with CE+LS?** ◑ The long tail is the core problem, but
  the loss isn't "incompatible" — it's that 33-frame classes can't be learned. Balanced
  *sampling* (you have it) + a low-capacity head + reporting macro-F1 over the *head*
  (frequent) classes is the realistic posture.
- **Q7 focal instead of CE?** ◑ Low priority. Focal helps easy/hard imbalance, not
  missing-data. CE+LS+CB-weights is fine; don't churn it now.
- **Q8 train activity in isolation 2 epochs?** ◑ Worth one cheap run to set a ceiling —
  but do it with the **simple head**, else you're just measuring overfitting.
- **Q9 ACTIVITY_LOSS_CAP killing grad?** ✘ No. `act_cap=80` (`config.py:584`) and the
  cap is a *smooth* log-cap that preserves gradient above the cap (`losses.py:1388–1394`);
  with `loss_act≈0.6` it never engages.
- **Q10 class-12 dominance a label bug?** ◑ Unverifiable here, but 5-minute check on
  your box: print the val histogram of `targets['activity']`. Collapse-to-majority
  predicts whichever class the constant output best matches; if class 12 isn't the
  modal label, suspect a label/threshold artifact in eval. Cheap, do it.
- **Q11 blend=1.0 really flows grad?** ✔ Yes; `1.0*x + 0.0*x.detach()` keeps the graph
  (autograd doesn't constant-fold the detached term out of existence). Not your issue.
- **Q12 set grad-clip=0?** ◑ Irrelevant to learning; clip at 1.0 vs ‖·‖≈0.01 is a
  no-op (you said so). Leave it.

**Multi-task architecture (13–19)**
- **Q13 raise PSR_SEQ_LOSS_SCALE?** ◑ The PSR ALIVE/DEAD oscillation is the seq/det
  batch alternation (file 57 §PSR) — the probe goes DEAD on det-only batches where PSR
  seq-loss isn't present. It's another probe-reading artifact, not instability. Don't
  chase it.
- **Q14 DETACH_PSR_FPN hurting PSR?** ◑ Possibly limits PSR feature learning, but PSR
  is not your bottleneck task; defer.
- **Q15 DETACH_REG_FPN in RF4?** ◑ Defer; detection is improving (§below).
- **Q16 FPN paths correct?** ✔ Each head taps FPN; no double/zero-count found in the
  forward. Fine.
- **Q17 PCGrad/CAGrad?** ✘ Not now. Gradient surgery resolves *conflicting* gradients;
  it cannot create activity signal from absent data. Premature.
- **Q18 Nash-MTL?** ✘ Same as Q17. Don't spend the week here.
- **Q19 ConvNeXt-Tiny too small for 5 tasks?** ◑ Not the binding constraint — data is.
  Tiny is plenty for 3.7k frames; a bigger backbone would overfit faster.

**Infrastructure (20–25)**
- **Q20 DDP dual-GPU?** ✘ Not now — don't parallelize a model that doesn't learn the
  task yet. Revisit after the head works.
- **Q21 evaluate_all hang fix?** ✔ Use **subprocess eval** (option c): fork a fresh
  process, `SIGKILL` on timeout. Threads/`SIGALRM` provably cannot interrupt a CUDA
  kernel (file 58 §99–105 is correct). This is the right infra fix and worth doing.
- **Q22/Q23 SSD / load into RAM?** ✔ The dataset is ~1.8 GB / 3.7k frames — **load it
  all into RAM** (you already have `RAM_CACHE`; raise the cap so it fills). Eliminates
  the HDD bottleneck without SSD. Higher value than DDP.
- **Q24 auto-load crash_recovery.pth?** ◑ Yes, low-effort, saves ~100 steps/crash.
- **Q25 validate every N epochs?** ✔ With subprocess eval the hang risk drops; keep
  every-epoch *gate* eval (200 batches) but move *full* eval to every 2–3.

**Paper (26–31)**
- **Q26/Q27 meet 0.375 act / schedule realistic?** ✘ No. 0.375 top-1 on 72 classes /
  3.7k frames is not reachable; the schedule assumes clean epochs you've never had.
- **Q28/Q31 narrow scope acceptable?** ✔ Yes — see PART 4. A 2–3 task honest system +
  failure analysis is a legitimate AHFE paper.
- **Q29 stop comparing to single-task SOTA?** ✔ **This is the key realization.** Make
  the multi-task-on-consumer-GPU result the *baseline you establish*, not a number you
  must match. Reframes every "gap" as a contribution.
- **Q30 swap to ViT-B + linear probes?** ✘ No — no time, and it trades a known system
  for an unknown one two weeks out.

---

## PART 3 — What I changed (shipped this branch)

| File | Change | Why |
|------|--------|-----|
| `src/config.py` | `ACTIVITY_HEAD_SIMPLE=True`, `ACTIVITY_HEAD_SIMPLE_HIDDEN=256` | gate the simple head |
| `src/models/model.py` | `ActivityHead` builds a ~150K-param MLP; `forward()` returns `simple_classifier(proj_feat)` when on, bypassing bank/TCN/ViT | short gradient path, no overfitting capacity, no noise input |
| `src/training/train.py` | `--reinit-heads` also reinitializes `simple_classifier` (logit bias=−0.5) | discourage majority-class collapse at init |

Same projection input, same backbone blend path, same `activity_head.` prefix (so LR
multiplier / clip / centralization still apply). Set `ACTIVITY_HEAD_SIMPLE=False` only
when you actually train on `sequence_mode` batches.

**Validation signal to watch (first 1–2 epochs, fresh RF4 + `--reinit-heads`):** number
of *distinct predicted classes* and prediction entropy on val. If it stays diverse
instead of collapsing to one class, the diagnosis is confirmed. Expect act_top1 in the
low 0.1s at best on this data — not the paper number.

---

## PART 4 — Honest paper verdict

Verified-against-reality target table:

| Task | Now | Stage gate (RF10) | Paper/SOTA | Realistic ceiling here |
|------|----:|----:|----:|----|
| head_pose MAE | 8.71° | ≤35° | 10° | **Met** (but see caveat) |
| act_top1 | ~0 | 0.18 | 0.375 | ~0.10–0.20 *with* the simple head + balanced eval |
| det_mAP50 | 0.053→ | 0.30 | 0.838 | 0.20–0.30 honest; 0.838 unreachable |
| psr_f1_at_t | ~0 | 0.16 | 0.731 | needs transition objective on seq batches first |

**head_pose caveat (verified concern, file 57 line 138–142, 59 line 89):** the pose.csv
forward vectors have norms 0.014–0.030 instead of 1.0 — they are **not unit-normalized**.
That makes the absolute angular MAE artificially small. Before you put "8.71°" in a
paper, normalize the GT vectors and recompute, or a reviewer will. This is your one
"working" number and it currently rests on an un-normalized target — fix it first.

**The paper that is actually defensible at AHFE (applied HFE venue):**

> A single shared-backbone model performing multi-task assembly verification in
> real time on consumer GPUs; we report multi-task trade-offs, establish the first
> consumer-hardware multi-task baseline on IndustReal, demonstrate head-pose tracking
> at parity with SOTA, and provide a rigorous analysis of joint-training failure modes
> under severe class imbalance and limited annotation — including a cautionary result on
> how a per-frame sampler silently defeats a temporal head, and how a per-parameter
> liveness probe can be misread as gradient starvation.

That last clause turns 10 days of "failure" into a genuine methods contribution. Files
56–60 are the raw material for it.

---

## PART 5 — Discrepancies found between the docs and the code (for your records)

1. **Probe semantics** (§0.1) — docs treat first/last-param norms as head totals.
2. **Architecture order** — files 56/59 draw `proj→ViT→TCN`; code is `proj→bank→**TCN→ViT**→CLS→classifier` (`model.py` ActivityHead.forward). Minor, but your mental model had the order reversed.
3. **Input dims** — file 56 line 78 labels the 1048-D input as `c5(512)+det(512)+p4(24)`; it is actually `det_conf(24)+c5(768)+p4(256)` (`model.py:1312`). Total matches; labels were wrong.
4. **"sequence_length=1"** (files 57/59/60) — not true at runtime; the bank emits T=16 (`window_size=16`), just filled with shuffled frames. The defect is *content*, not *length*.
5. **ViT attention scaling / pos_embed** — already fixed (`model.py:1104`, `1117–1122`); not a live bug.

None of these change the bottom line; they're listed so the next consult starts from the verified picture, not the inherited one.
