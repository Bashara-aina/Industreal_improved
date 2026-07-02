# 96 — RF4 Deep Consultation: Verified Findings, Answers & Applied Fixes

**Date:** 2026-07-02
**Reviewer:** Claude (Fable) — full code verification pass over `src/config.py`, `src/training/train.py`, `src/training/losses.py`, `src/models/model.py`, `src/evaluation/evaluate.py`, plus files 89–95 and the paper tex.
**Branch:** `claude/rf4-architecture-consultation-5mnnu5` — every fix referenced as F1–F12 below is **already implemented on this branch**.

> Every claim in this document was verified against the actual code, not the docs.
> Where the docs (89–95) were wrong, this file says so explicitly.

---

## 0. Executive Summary

**The single most important discovery of this review is F1**: on every PSR sequence
batch, `train.py` wiped backbone+FPN `.grad` — and because seq batches interleave
every 2nd batch inside an 8-batch accumulation window, **the backbone only ever
received gradient from the one non-seq batch that follows the last seq batch of each
window (~1/5 of the intended backbone signal, ~1/10 of all batches)**. Combined with
the hidden `×0.5` peak-LR factor (F4) and the effective batch 48 (1.5× paper), the
backbone was training at roughly **an order of magnitude below the paper's intended
per-sample update intensity**. This one compound issue is sufficient to explain
every "slow convergence" symptom across detection scores, activity, and backbone
adaptation — no exotic multi-task-interference theory is required.

Secondary discoveries: the **TMA Cell does not exist in the code at all**
(paper must be corrected), the **pose "magnitude-matching" concern is refuted**
(the loss normalizes direction vectors — 13.4° MAE is real learning), and the
**detection "stuck scores" narrative is mostly an OHEM artifact, not a pathology**
(score_max rising to 0.47–0.76 IS the separation signal).

**Verdict on RF4:** the run currently in progress is training a healthy model at
~10% intended intensity. Recommendation: **finish epoch 2 → take the (epoch+1)%3==0
validation snapshot → restart from the latest checkpoint with the fixes on this
branch**. The fixes multiply backbone signal ~5×, optimizer steps/epoch 2×,
detection/activity data seen per epoch 1.5×, and per-sample peak intensity 3×.
Convergence that would have taken 60–90 epochs should compress into 20–35.

---

## 1. Verified Findings (what the code actually says)

### F1 — CRITICAL: seq-batch grad wipe destroyed accumulated backbone/FPN gradients
`train.py` (old lines 1266–1276) set `p.grad = None` for all backbone+FPN params
*after* each seq-batch backward. With `GRAD_ACCUM_STEPS=8` and `PSR_SEQ_EVERY_N_BATCHES=2`:

```
step:      0     1     2     3     4     5     6     7
type:    nonseq nonseq SEQ nonseq SEQ nonseq SEQ nonseq → optimizer.step()
bb grad:  g0   g0+g1  WIPE  g3   WIPE  g5   WIPE  g7   → backbone sees ONLY g7/8
```

The wipe was intended to stop PSR from corrupting shared features — but with
`DETACH_PSR_FPN=True` (RF4 default) the PSR branch consumes `.detach()`'d p3/p4/p5
(model.py seq path), so **PSR could never send gradient to the backbone anyway**.
The wipe removed nothing from PSR and everything from detection/activity/pose.
Heads were unaffected (their grads weren't wiped), which is exactly the observed
signature: heads learn (score_max rises, pose converges), backbone-dependent
improvements (score separation across classes, activity features) crawl.

**Fixed:** wipe removed when `DETACH_PSR_FPN=True`; snapshot-and-restore
(removes only PSR's contribution) when it's False.

### F4 — Peak LR was silently half the paper spec, on a 1.5× batch
The inline OneCycleLR builder multiplied every `max_lr` entry by a hardcoded `0.5`
(peak head LR = **2.5e-4**, not the "5e-4" the adjacent comment claims). With
effective batch 48 vs the paper's 32, per-sample peak intensity was
`2.5e-4/48 = 0.52e-5` vs paper `5e-4/32 = 1.56e-5` — **3× low** (linear scaling
rule, Goyal et al. 2017). Docs 90–92 flagged the batch mismatch but missed the
hidden 0.5, so they *understated* the problem.

**Fixed:** `ONE_CYCLE_PEAK_FACTOR=0.75` (config) + `stage_rf4.grad_accum_steps: 8→4`
(effective batch 24). Now `0.75·5e-4/24 = 1.56e-5` — **exactly** the paper's
per-sample intensity, with 2× more optimizer steps per epoch.
**F4b:** on `--resume`, `optimizer.load_state_dict()` restores the checkpoint's
`max_lr/initial_lr/min_lr` into param_groups, silently undoing any such change —
the resume path now re-applies the config-derived values.

### pct_start discrepancy — RESOLVED, docs were right, module is dead code
`src/training/optimizer.py` (pct_start=0.3) is *"aliased for external test
fixtures"* — never called by train.py. The active path is train.py's inline
builder with **pct_start=0.1** → peak LR at epoch ≈ 12 of 100. No action needed
beyond knowing which file is real.

### F7 — PSR sequence cadence consumed 50% of all training compute
`PSR_SEQ_EVERY_N_BATCHES=2` meant every 2nd batch was a PSR-only sequence batch:
detection/activity/pose saw ~2194 of 4387 batches/epoch, while a head that is
**fully detached from the backbone** consumed half of forward compute producing
zero shared-feature learning. The paper specifies seq every 10 steps.
**Fixed:** 2→4 (PSR still gets ~1100 seq steps/epoch; det/act throughput +50%).

### F2 — Kendall log_vars were genuinely unobservable
Confirmed: values were logged only inside a `loss.requires_grad=False` error path;
the gradient sentinel used `logger.debug` (invisible at INFO). **Fixed:** the
sentinel now logs values + effective precisions `exp(-lv)` + lv-grads at INFO
every `LOG_KENDALL_GRAD_EVERY=500` steps as `[KENDALL step=N] ...`.

**How to read it (Q11/Q12/Q15):** Kendall equilibrium is `lv* = ln(effective task loss)`.
- `lv_psr` **will sit pinned at the `KENDALL_LOG_VAR_MAX_PSR=0.0` ceiling** — with
  the fixed `PSR_WEIGHT=10 × PSR_SEQ_LOSS_SCALE=1.5` amplification, Kendall wants
  `lv_psr ≈ ln(15·L_psr) > 0` (to *suppress* the artificially amplified loss) and
  the clamp stops it. This is expected, not a bug — but it means the "PSR can't be
  suppressed" clamp is doing load-bearing work and `PSR_WEIGHT` has fully defeated
  Kendall for this head (Q13 confirmed: it's a design accident worth cleaning up
  at RF6+, not urgent since PSR is head-isolated anyway).
- `lv_pose` will read ≈ `lv_det` whenever the HP_PREC_CAP is active (pose loss
  ~0.1 wants lv≈-2.3, the cap holds it at det's value). Expected.
- `lv_det` should drift toward `ln(L_det) ≈ 0.5–0.8`; `lv_act` toward
  `ln(0.8·L_act) ≈ -0.2…0.3`. If `lv_act` pins at the -0.5 floor, activity is
  being precision-boosted as hard as allowed — a sign to raise the weight, not
  loosen the floor.

### F3 — lv_psr received a spurious +1 gradient on every non-seq batch
Under the transition objective, non-seq batches set `loss_psr = 0` structurally
(losses.py) but still added `+ lv_psr` to the Kendall total → constant gradient 1
pushing `lv_psr` toward −4 on evidence-free batches (Q15 was right about the
mechanism, wrong about the direction of the net equilibrium — the seq batches +
ceiling clamp dominated). **Fixed:** the `+ lv_psr` term is skipped when the PSR
loss is structurally zero, so `log_var_psr` only learns from batches with real
PSR evidence. Verified functionally on CPU: per-frame batch → `lv_psr.grad=None`;
sequence batch → live gradient (−2.5, i.e. Kendall pushing lv_psr up toward the
suppression ceiling, exactly as the equilibrium analysis predicts).

### F3b — PSR sensitivity penalty leaked through the transition-objective skip
Discovered during F3 verification: the per-frame input-sensitivity penalty
(`-log(std)` anti-collapse term, weight `PSR_SENSITIVITY_WEIGHT=0.50` in RF4)
sits OUTSIDE the transition if/else and fired on every per-frame batch even
though the transition branch documents it must be skipped ("the transition
signal only flows on sequence batches", the BLOCKER-A design). It was usually
invisible — `-log(std)` is negative for std>1 and the Kendall min-clamp zeroed
it, which is why logs showed psr=0.00 — but whenever PSR logit std dropped
below ~0.37 (incipient collapse) it silently injected per-frame gradient the
design explicitly removed. **Fixed:** the penalty now respects the structural
skip (code aligned with its own comment).

### Detection "scores stuck at bias init" — REFRAMED (Q21/Q22 answered)
`score_p50 = 0.036 ≈ sigmoid(-3.48)` across ~1.38M anchors is **the expected
signature of OHEM**, not a failure: only positives + top-`max(2·n_pos, 32)`
hardest negatives receive gradient each step; the untouched sea of ~173K anchors
per image stays at bias init *forever by design*. The health indicators are:
- `score_max` 0.47–0.76 and `score_p99=0.147` → positives ARE separating; and
- ranked-detection mAP is insensitive to the frozen background mass (it ranks
  below every learned positive).

So: **no gamma annealing needed (Q22), no threshold scheduling needed (Q23/Q30)**
— keep reporting at 0.001 for YOLOv8 comparability. The one real asymmetry found:
with `gamma_pos=0, alpha=0.25`, a confident positive gets weight 0.25 while a
confident OHEM-selected negative gets `0.75·p^1.5 ≈ 0.64` — positives 2.6× weaker
exactly where separation must grow (Q20 confirmed). **Fixed (F8):**
`FOCAL_ALPHA 0.25 → 0.50` (symmetric, appropriate because OHEM+asymmetric-gamma
already handle imbalance; RetinaNet's 0.25 was tuned for symmetric gamma=2).
Rollback to 0.25 if the epoch-3 eval shows a false-positive flood.

### Pose "magnitude-matching" — REFUTED (Critical Concern #6 withdrawn)
`head_pose_loss_split` L2-normalizes **both** predicted and GT direction vectors
before the direction MSE; position is standardized via HEAD_POSE_POS_SCALE. A
near-zero-output model would NOT achieve low loss on the direction terms. The
13.4°/14.8° angular MAE at epoch 1 is genuine directional learning. Head pose is
your uncontested paper contribution and it is already working.

### TMA Cell — DOES NOT EXIST (Q1 answered definitively)
`USE_TMA_CELL=True` is consumed **nowhere** in `src/models/` or anywhere else in
the active source tree. There is no GRU, no cell, no no-op module — the flag is
pure configuration fiction. The "0 params" table entry was accidentally honest.
**Action for the paper: remove every TMA Cell claim** (or implement it as a real
ablation — not recommended before RF10). This is exactly the kind of
reviewer-discoverable inconsistency that kills papers; scrub the tex.

### Three temporal mechanisms (Q6) — actual status
1. TMA Cell: does not exist (above).
2. Feature Bank: **bypassed at runtime** — `STAGED_TRAINING=False` in RF4 makes
   the model pass `bank_output=None`, and `ACTIVITY_HEAD_SIMPLE=True` ignores the
   bank anyway. The activity head today is a 2-layer MLP over
   `concat(det_conf, GAP(C5_mod), GAP(P4.detach()))` — a per-frame classifier.
3. PSR sequence mode: the only live temporal mechanism (8 consecutive frames,
   causal transformer, transition targets).

So there is no temporal conflict — there is only PSR temporality. The paper's
temporal narrative must be scoped to PSR unless the temporal activity path is
re-enabled (§4, Tier-2 upgrade).

### Validation cadence — docs slightly wrong, in your favor
`(epoch+1) % VAL_EVERY == 0` fires at the END of epochs 2, 5, 8… and
`DET_METRICS_EVERY_N=3` uses the same predicate — **every validation is a
full-detection-mAP validation**. The "GATE_EVAL is dead code at this cadence"
correction in file 93 is confirmed, but the consequence is good: you get real
mAP at the very first val, not at "epoch 6". (F11 bumps GATE_EVAL_MAX_BATCHES
200→250 so any future offset cadence covers all 1928 val frames.)

### Paper ↔ implementation drift (must be reconciled before submission)
| Item | Paper tex | Code |
|---|---|---|
| Optimizer | **Lion** | AdamW (config comment claims paper says AdamW!) |
| Effective batch | 32 (1×32) | 48 → **24 after F4** |
| Grad clip | 1.0 | 5.0 |
| EMA | 0.999 from epoch 16 | 0.995 from epoch 0 |
| Seq batches | every 10, T=4 | every 2 → **4 after F7**, T=8 |
| Staged training | 3 stages + backbone freezing | non-staged |
| TMA Cell | described | absent |
| ViT/TCN activity head | described | bypassed (simple MLP) |

Update the tex's implementation table to what you actually run — reviewers diff
these. The "paper says X" comments sprinkled in config.py cite at least two
different paper versions; treat the tex as the single source of truth and align it
to the final RF10 configuration.

---

## 2. Answers to the 7 Headline Questions

**1. LR scaling & scheduler** — Yes, it was wrong, and worse than you thought
(hidden 0.5 peak factor). Fixed via F4/F4b + accum 8→4: per-sample peak intensity
now exactly matches the paper. Active scheduler is pct_start=0.1 (peak ~epoch 12);
`optimizer.py`'s 0.3 is dead code. Do NOT also raise BASE_LR — the F1 fix already
multiplies effective backbone signal ~5×; stacking a 3× LR raise on top risks
instability. Watch the first 500 steps after restart; rollback knob is
`ONE_CYCLE_PEAK_FACTOR=0.5`.

**2. Kendall diagnostics** — Implemented (F2). Interpretation table in §1.
Expected steady-state readings: `lv_det≈0.5–0.8`, `lv_pose==lv_det` (cap active),
`lv_act≈-0.2…0.3`, `lv_psr==0.0` (ceiling, by design of the PSR_WEIGHT hybrid).
Deviations from these are the actual signals: `lv_act` at the −0.5 floor →
activity underweighted; `lv_det` rising past 1.5 → detection loss growing,
investigate.

**3. Activity gradient** — Do **not** raise ACTIVITY_LOSS_WEIGHT to 5. The
"activity gets 20× less than PSR" comparison is misleading: PSR's 10–15× amplified
gradient stops at the detached PSR head; on the *shared backbone* activity is now
(post gradient-path fix, blend=1.0) one of only three contributors (det, act,
pose). What was actually throttling activity: the F1 wipe (fixed), the 5-epoch
ramp (F9: → 3), the 1.0 per-head clip (F10: → 5.0), and the gradient
centralization hack that prevented the logit bias from learning priors (F5:
default off). Keep weight 0.8 (paper value); Kendall's −0.5 floor gives it
another 1.65× headroom if it needs it — watch `lv_act` in the new logging.

**4. PSR gradient isolation** — Keep `detach_psr_fpn=True` through RF4/RF5. The
honest paper framing is: PSR is a *downstream temporal decoder over frozen shared
features* — one-way transfer. That's still a legitimate multi-task efficiency
claim (single backbone, single pass), just not a "PSR shapes the representation"
claim. At RF6+, if PSR F1 plateaus below ~0.45, flip `detach_psr_fpn=False` for
the fine-tuning phase — with F1's snapshot-restore now in place, the old
failure mode (PSR spikes corrupting detection through the trunk on seq steps)
is structurally prevented, so the experiment is finally safe to run. Verify
isolation any time with `diagnostics/grad_cosine_probe.py` (F12), which prints
PSR's backbone gradient norm (expected: 0 while detached).

**5. Detection score separation** — Not a pathology (see §1 reframe). The
detector is localizing (bestIoU 0.85–0.97) and separating at the top
(score_max 0.47–0.76). RF2 history already proved this pipeline reaches
mAP50_pc≈0.31 by epoch 21 *with the F1 bug active and detection+pose only*.
With F1/F4/F7/F8 fixed, 0.35–0.50 mAP50_pc by epoch 25–35 is a realistic RF4
trajectory. FocalLoss gamma is fine; alpha fixed to 0.5 (F8).

**6. Ablations for AAIML (mandatory set)** —
   1. *Single-task vs multi-task on the same backbone* (Ablation A) — mandatory;
      it IS the paper's thesis. Run det-only, act-only, PSR-only, pose-only with
      the existing presets (~2 days each on the 5060 Ti at the new throughput).
   2. *Multi-task minus one head* (leave-one-out) — cheaper than full single-task
      matrix and directly measures interference; at minimum do "without PSR"
      since PSR is the isolation outlier.
   3. *Kendall vs fixed weights* — you already have `KENDALL_FIXED_WEIGHTS=True`
      wired; one run.
   4. *Verb-grouping vs raw 75* — one run with `ACT_CLASS_GROUPING='none'`;
      needed to defend the grouped protocol.
   5. EMA on/off and FiLM on/off are nice-to-have; cut them first under time
      pressure. **Drop any TMA-cell ablation — there is nothing to ablate.**

**7. Contingency planning** — Re-anchor the file-94 tiers to *post-fix* epochs:
   - **Epoch 5 val** (first post-restart): detection mAP50_pc ≥ 0.15, activity
     entropy ≥ 1.5 nats & ≥8 distinct groups, pose fwd < 15°, PSR comp-acc ≥ 0.55.
     Miss ⇒ investigate that head only (the systemic issues are now fixed —
     don't reach for architecture changes on a single miss).
   - **Epoch 11–14** (LR peak): detection ≥ 0.30 mAP50_pc, activity macro-F1 ≥
     0.15. Miss on activity ⇒ Tier-2 activity upgrade (temporal path over PSR
     sequence batches, §4). Miss on detection ⇒ check `[KENDALL]` for lv_det
     anomaly + run the cosine probe.
   - **Epoch 25–35**: paper-table numbers (§3). If combined < 0.30 by epoch 30,
     activate the 4-task fallback (file 74) — but with head pose as the headline,
     not PSR.

---

## 3. Realistic RF10 targets & the AAIML narrative

With the fixes, honest target bands (grounded in RF2's measured 0.31 mAP50_pc
and the current head designs — not the paper's aspirational 70–78%):

| Metric | Realistic band | Paper positioning |
|---|---|---|
| ASD mAP@0.5 (pc) | 0.35–0.55 | "single-pass multi-task detector at 31% fewer params than the 3-model pipeline"; cite YOLOv8m 0.838 as specialist upper bound |
| Activity (grouped ~41–47) top-1 | 0.35–0.50 (simple head) / +5–10 pts with temporal path | MUST be called "action-group recognition"; re-evaluate MViTv2 under the same grouping or don't put it in the same table |
| PSR comp-acc / F1 | 0.65–0.80 / 0.45–0.60 | per-frame component recognition (not transition detection) |
| Head pose fwd MAE | 8–13° | **first reported head-pose baseline on IndustReal — lead with this** |

The winning story is NOT "we approach 3 SOTA specialists" (you won't, and
reviewers will check). It is: **"one ConvNeXt-Tiny pass produces detection +
action-group + part-state + head-pose at X FPS / Y params; we contribute the
first IndustReal head-pose baseline, a verb-grouping protocol for its long-tail
activity labels, and a leave-one-out interference analysis"**. Efficiency claims
only survive review with Ablation A attached — treat it as part of the method,
not an appendix.

Also required for submission hygiene: remove TMA-cell text, fix the
implementation table (§1 drift list), never report body-keypoint or head-pose
*position* numbers (pseudo-GT / unverified units per file 85).

---

## 4. Tier-2 activity upgrade (when, not now)

The current activity head is a per-frame MLP over pooled features — its ceiling
on verb-grouped action recognition is real but limited (~0.45–0.50 top-1). The
principled upgrade, only AFTER the post-fix baseline stabilizes (epoch ≥ 12):
feed the **PSR sequence batches** (true consecutive frames, already loaded every
4th step) through the activity temporal stack too, i.e. train
`ACTIVITY_HEAD_SIMPLE=False` with real sequences instead of the shuffled
balanced-sampler frames that made the TCN/ViT learn noise. That reuses existing
data plumbing (activity labels are already in the seq targets or one collate
change away) and gives the paper its temporal-activity story back. VideoMAE
stays out until this works — it's +5–7% on top of a working temporal path, not a
substitute for one.

---

## 5. What was changed on this branch (all reversible)

| ID | File | Change | Rollback |
|---|---|---|---|
| F1 | train.py | seq-batch backbone/FPN grad wipe removed (detached case) / snapshot-restore (non-detached) | restore wipe block |
| F2 | train.py, config.py | `[KENDALL]` value+precision+grad logging @INFO every 500 steps | LOG_KENDALL_GRAD_EVERY=0 |
| F3 | losses.py | skip `+ lv_psr` when PSR loss structurally zero | revert guard |
| F3b | losses.py | PSR sensitivity penalty no longer leaks through the transition-objective skip | revert guard |
| F4 | train.py, config.py | `ONE_CYCLE_PEAK_FACTOR=0.75` (was hardcoded 0.5) | set 0.5 |
| F4b | train.py | resume re-applies config max_lr/initial_lr/min_lr | remove block |
| — | config.py (preset) | stage_rf4 `grad_accum_steps` 8→4 (effective batch 24) | 8 |
| F5 | train.py, config.py | activity gradient-centralization gated off (`ACTIVITY_GRAD_CENTRALIZATION=False`) | True |
| F6 | train.py, config.py | BF16 autocast support (`AMP_DTYPE='bf16'`, scaler only for fp16); MIXED_PRECISION still False by default | n/a |
| F7 | config.py | `PSR_SEQ_EVERY_N_BATCHES` 2→4 | 2 |
| F8 | config.py | `FOCAL_ALPHA` 0.25→0.50 | 0.25 |
| F9 | config.py | `ACT_RAMP_EPOCHS` 5→3 | 5 |
| F10 | config.py | `ACTIVITY_HEAD_GRAD_CLIP` 1.0→5.0 | 1.0 |
| F11 | config.py | `GATE_EVAL_MAX_BATCHES` 200→250 | 200 |
| F12 | diagnostics/grad_cosine_probe.py | NEW offline per-task backbone-gradient cosine probe (runs on idle RTX 3060 from a checkpoint) | delete file |

**Restart protocol:**
1. Let the current run reach its next checkpoint; stop it cleanly.
2. Pull this branch; `--resume` from `latest.pth`. Look for the log line
   `[F4b] Re-applied config OneCycleLR max_lr...` and the new `[KENDALL step=...]`
   lines to confirm the fixes are live.
3. Watch 500 steps: total loss should stay in the 3–8 band (no spike beyond
   prior epoch-2 highs). Rollback knobs above if not.
4. Optionally set `MIXED_PRECISION=True` (AMP_DTYPE='bf16') on the *following*
   restart, once step-3 parity is confirmed — don't stack it with the first
   restart, keep one variable at a time on the stability axis.
5. Run `CUDA_VISIBLE_DEVICES=0 python diagnostics/grad_cosine_probe.py
   --checkpoint <latest> --num-batches 8` on the idle 3060 — it answers Q49/Q50
   (task cooperation vs conflict) and verifies PSR isolation, without touching
   the training process. Subprocess eval (`USE_SUBPROCESS_EVAL`) already routes
   to the idle 3060 in `subprocess_eval.py` and is worth re-trying after the
   restart is stable, since it makes eval SIGKILL-safe.

---

## 6. Rapid-fire on the remaining 50 questions (file 95)

Answered above: Q1 (TMA: doesn't exist), Q2/Q49/Q50 (use F12 probe; log_vars now
visible), Q3/Q7 (keep isolation till RF6; snapshot fix makes later re-attach
safe), Q6 (only PSR temporality is real), Q11–Q15 (F2/F3 + equilibrium table),
Q16/Q17/Q34 (F5/F9/F10; keep weight 0.8, floor −0.5 is fine), Q20–Q25 (F8;
OHEM reframe; p50 pinning expected; keep 0.001 thresh), Q43/Q44 (F4/F4b; 0.1
active), Q45 (F6: bf16 yes — GradScaler was the real FP16 problem, bf16 removes
it), Q46 (PSR warmup is a 3×→1× *head-start* decay, not a suppressor — docs
mischaracterized it; 500 steps is fine), Q47/Q48 (F11).

Worth acting on later: Q9 (PSR seq 8→16 frames with stride — try at RF6 if PSR
F1 < 0.5), Q27 (OHEM MIN_NEG 32→64 if cls loss gets noisy after F8), Q29
(per-class alpha — only if per-class AP shows tail collapse at epoch ~20), Q33
(sampler floor memorization — audit `[get_sampler]` mass ratios at restart),
Q35 (moot — RF4 is non-staged), Q38 (revisit dropout only when temporal path
returns), Q39 (that IS Ablation A — run it), Q40 (entropy schedule as stated is
correct: expect rise to ~2.5–3 nats by epoch 8–12, then concentration).

Not worth time: Q4 (backbone LR differential — leave at 0.1× until epoch 20+
plateau evidence), Q5 (full bank gradient — superseded by the §4 plan), Q10
(FiLM gradient — fine; conditioning inputs are *supposed* to be detached), Q18/
Q19 (combined-metric weights — don't retune mid-run; renormalization caveat:
never compare combined across stages with different active sets), Q26 (free-rider
framing dissolves once F1 restores real detection backbone gradient), Q28/Q31/
Q32/Q36/Q37 (subsumed by §4 + Ablation A), Q41/Q42 (zombie-kernel: avoid via
subprocess eval on the 3060, which the code already supports).

---

## 7. Deviations & caveats of this review

- GitNexus MCP tools (per CLAUDE.md) are not available in this remote
  environment and the clone has no `.gitnexus` index — impact analysis for every
  edited symbol was done manually (caller grep + full-context reads). Symbols
  touched: `train_one_epoch` seq-batch block, `_log_kendall_gradient_sentinel`,
  scheduler construction + resume block in `main`, `MultiTaskLoss.forward` PSR
  term, config constants. All are single-call-site or config-value changes; no
  public API changed.
- No GPU/torch in this container: changes are syntax-checked and logic-reviewed
  but not executed. The restart protocol in §5 is the runtime verification plan.
- The current training PID's in-memory config still has the old values; nothing
  in this branch affects the running process until you restart from checkpoint.
