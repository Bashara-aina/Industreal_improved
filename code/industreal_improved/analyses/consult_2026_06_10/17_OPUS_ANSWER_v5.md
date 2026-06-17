# 17 — Opus Answer v5: The Ultimate Path from `det_mAP50=0.0091` to a Defensible Paper

Response to `16_MASTER_PROMPT_v5.md`. Read against the current tree
(branch `claude/loving-ptolemy-9am9e0`, HEAD `ecd2268`). Code claims below are
grounded in the actual source: `losses.py`, `train.py`, `config.py`,
`evaluate.py`, `model.py`, `psr_transition.py`.

> Environment note: this consultation was produced in a docs/analysis container
> (no `torch`, no GPU), so every number below is either (a) derived from the
> code/docs or (b) a target/estimate. The commands in the Appendix must be run
> on the RTX 3060 box. No GitNexus MCP tools were available here, so
> impact/`detect_changes` were not run; this change only *adds* a doc file and
> edits no code symbols.

---

## 0. Verdict first

**You are no longer debugging — you are harvesting.** `det_mAP50=0.0091` with
`bestIoU=0.94` on GT frames is not "almost zero," it is *"localization works,
classification confidence is uncalibrated and the score threshold is at the
floor (score_p50=0.001)."* That is a completely different, and much easier,
problem than the one you spent two months on. The architecture is no longer the
question. **Throughput and discipline are.**

The single most important reframe for this round:

> **"Fill every `\popwres`" is ~32 cells, and they are NOT equally hard. Most of
> the publishable value is reachable *before* any head beats a specialist.**

A number is **defensible** when (a) it is measured under the stated protocol,
(b) it is *non-degenerate* (not a constant-output artifact), and (c) the
comparison is apples-to-apples or the gap is named. By that bar, head pose,
efficiency, the b-boxed-vs-all-frames detection split, and a transition-based
PSR are defensible *long before* they are SOTA. **Optimize for "no zeros, every
cell honest" first; push the 2–3 most winnable cells toward/over baseline
second.** That ordering is the efficient path.

The three effort tiers (full mapping in §A.0):

| Tier | Cost | Cells | What |
|---|---|---|---|
| **Free now** | hours, **no training** | ~13 | Efficiency table (params/GFLOPs/FPS ×2 modes ×2 datasets), param counts in Tables 4–5 |
| **Recovery ladder** | the bulk, IndustReal | ~14 | Detection, activity, PSR, assembly-state/error-verif, head-pose MAE |
| **Second port (defer/scope)** | a whole second dataset | ~5 | All of IKEA-ASM (Table 2): seg AP, body-pose PCK, activity, localization |

---

## A. The Ultimate Path (concrete staged ladder)

### A.0 The placeholder map (what "done" actually requires)

From `popw_paper_improved.tex` (`\popwres` = `\todo`), the open cells are:

- **Table 2 (IKEA, secondary):** seg AP@0.5, AP(COCO); PCK@10px, PCK@0.2;
  activity Top‑1 (front+pose), Top‑1 (all views); localization mAP@0.5. **→ 7**
- **Table 3 (IndustReal, primary):** mAP (b‑boxed), mAP@0.5 (all frames),
  mAP@[0.5:0.95]; activity Top‑1, Top‑5; PSR F1(±3), F1(±5), POS(±3);
  Assembly‑State F1@1, Error‑Verif AP; head‑pose Forward/Up MAE, Position MAE.
  **→ 13**
- **Table 4 (main consolidated):** 2 rows × 7 metrics (w/ and w/o VideoMAE) —
  same numbers as Table 3, re‑tabulated. **→ derived**
- **Table 5 (activity breakdown):** IKEA Top‑1/5 + IndustReal Top‑1/5. **→ derived**
- **Table 6 (efficiency):** params/GFLOPs/FPS, batched + streaming, both
  datasets. **→ ~12, all free**

**Decision: IndustReal is the paper. IKEA is a generalization section, not a
blocker** (see §G.2). Plan the ladder around Table 3 + Table 6.

### A.1 The ladder

Each stage: preset · epochs · subset · **gate to proceed** · expected range.
This supersedes the staged-training schedule in the paper (§Training); it is the
recovery protocol from Opus v4 (R0–R3) extended through paper-fill.

| Stage | Preset / config | Subset | Epochs | **Gate to advance** | Expected at gate |
|---|---|---|---|---|---|
| **R1** *(running)* | `recovery_det_only` (det+head_pose, FP32, eff‑batch 8) | 0.25 | 3 | `det_mAP50 ≥ 0.05` (b‑boxed) **and** `committed>0, skipped=0` | 0.05–0.15 |
| **R1.5** | same + **anchor calibration** + **synthetic det pretrain** | 0.25→**1.0** | +10–20 | b‑boxed mAP ≥ 0.30, not regressing | 0.30–0.55 |
| **R2** | `recovery` (all heads), resume R1.5 best; **activity = CE+LS (no LDAM)**; **PSR off still** | 0.25 | +4 | activity ≥ 4 classes (`pred_seen`); det not −30% | act_top1 ≥ 0.10 |
| **R2.5** | `recovery` + **PSR on via transition objective** (`psr_transition.py`) + raw‑loss probe | 0.25 | +4 | PSR ≥ 3 unique patterns; PSR raw loss O(0.1–0.3) finite | psr_f1 ≥ 0.30 |
| **R3** | full joint; **subset 1.0**; **clip‑level activity eval**; engage FeatureBank/VideoMAE; EMA on | **1.0** | 30–50 | all 5 heads non‑zero & monotonically improving | see §B/§D ceilings |
| **R4** | multi‑seed (×3) + ablations + efficiency + geo head‑pose | 1.0 | — | ±std computed, tables generated | paper-ready |

**Why this order:** detection is the spine — `det_conf` feeds the activity head
and PSR depends on shared backbone features, so detection must be alive and
*not regressing* before the other heads can use it. Activity before PSR because
PSR's winning design consumes the assembly-state signal. Subset 0.25 buys fast
liveness proofs; **the paper numbers come from 1.0** (25% ≈ 35/75 activity
classes — you cannot report a defensible 75-way Top-1 on a subset; full = 74).

**When to re-enable the deferred machinery:** EMA → R3 only (after metrics move
monotonically; an EMA blend during collapse poisons `best.pth`). Mixup → never,
until it mixes *images* pre-forward (`USE_MIXUP=False` today is correct).
`ZERO_DET_CONF` → never again. VideoMAE → R3, after the GAP-only activity head
shows ≥4-class diversity.

---

## B. The `det_mAP50` gap

**B1 — Feasible to competitive in ~100 epochs? Yes for the b-boxed protocol to
~0.40–0.60; no for 83.80 without synthetic pretrain.** The 0.0091 is an
epoch-0/25%-subset/floor-threshold artifact, not a ceiling. The model already
*localizes* (`bestIoU` up to 0.94, 4,142 preds@IoU>0.5 per batch). What is
killing mAP is two fixable things:

1. **Classification confidence is flat/low** (`score_p50=0.001`) — cls hasn't
   learned to *fire confidently at the right anchors* yet. R1 fixes this: now
   that ~85% empty-frame "fire nowhere" gradient is gone (RC-28), the cls head
   gets clean GT-supervised signal. Expect mAP to climb fast over R1.
2. **Anchor recall.** `ANCHOR_SIZES=(24,48,96,192,384)` (config.py:247) vs GT
   k-means centers 164–404px (w=146–594) — only P6/P7 (≈1.6% of anchors) can
   reach IoU≥0.5 with typical GT. **This is the biggest cheap detection win.**
   Run `calibrate_anchors.py`, set sizes to the GT clusters, retrain. This alone
   should move recall (and mAP) materially.

**B2 — Realistic ceiling for ConvNeXt-Tiny + RetinaNet head on ASD:**

| Config | b-boxed mAP@0.5 estimate |
|---|---|
| R1 as-is (no anchor fix, no synth) | 0.15–0.35 |
| + anchor calibration | 0.30–0.55 |
| + synthetic pretrain (`PRETRAIN_DET_ON_SYNTH`, currently wired-but-unused) | 0.55–0.75 |
| YOLOv8m specialist (COCO + 260K synth + real) | 0.838 (reference) |

83.80 is a *dedicated single-task* detector with massive synthetic pretraining.
A shared-backbone model doing 5 tasks trailing it is **expected and
defensible** — that is the efficiency story (§E), not a failure.

**B3 — Report b-boxed and all-frames separately? Yes — the paper already does
this and it is the correct, honest framing.** Table 3 has both rows; the
protocol section (paper L635, L649) already defines "mAP (b-boxed) = annotated
frames only" as the apples-to-apples comparison to 83.80, and "mAP@0.5 (all
frames)" vs 64.10 (which is *lower by construction* — empty frames dilute it).
**Action:** report both, and add one sentence naming the synthetic-pretrain gap
explicitly. Do not chase 83.80 — frame it.

**B4 — Is 87-min eval/epoch sustainable? No — it's 56% of wall-clock (87 of
~155 min) and it gates iteration speed. Fix it three ways:**

1. **Capped-val gate metric.** Eval the *gate* metric on a fixed representative
   val subset (~2,000 frames) every epoch (~10 min); run the **full** val
   (EVAL_MAX_BATCHES=-1) only at checkpoints / end of stage. The knobs exist:
   `EVAL_MAX_BATCHES` (config.py:290) and `SKIP_DET_METRICS_EVAL`
   (config.py:503, ~87 min savings).
2. **Eval det-mAP every N epochs.** Add a small `DET_METRICS_EVERY_N` cadence
   wrapper around the existing `SKIP_DET_METRICS_EVAL` flag (eval det mAP every
   5 epochs; cheap heads every epoch). `VAL_EVERY` already exists (config.py:289).
3. **Compute @0.5 only during training, [0.5:0.95] at the end.** The COCO
   multi-thresh path (`compute_ap_multi_thresh`, 10 IoU × 24 classes,
   evaluate.py:1240) is most of the cost; you only need mAP@0.5 for the gate.

Net: ~10-min gate every epoch, full eval every 5–10 epochs.

---

## C. When to enable Activity and PSR

**C1 — Activity: at R2, GAP-only, with plain CE + label smoothing — NOT
LDAM(s=30).** The config today is `USE_LDAM_DRW=True, LDAM_S=30,
LDAM_DRW_EPOCH=0` (config.py:424–427). LDAM s=30 is a **30× logit amplifier**
stacked on class-balanced sampling **and** label smoothing — three imbalance
mechanisms compounding → the familiar 1-class collapse. **Set
`USE_LDAM_DRW=False` (CB-Focal/CE path) for the first joint runs.** Re-introduce
LDAM only if long-tail recall is the *single* remaining gap, and then at s=10–15.
VideoMAE stays **off** until the GAP-only head shows ≥4-class diversity (it adds
22M params + VRAM and would confound the experiment).

**C2 — PSR: at R2.5, and only via the transition objective + a raw-loss probe
(do NOT enable per-frame focal as-is).** See §D — per-frame focal on 95%-static
fill-forward labels makes constant output near-optimal. `psr_transition.py` is
ready (`build_transition_targets` Gaussian-smear + `MonotonicDecoder`); wire it
behind `USE_PSR_TRANSITION` and turn PSR on with that, not the legacy path.

**C3 — Switch `recovery_det_only` → `recovery`: immediately after the R1 gate
passes.** The `recovery` preset (config.py:555) already turns on all heads with
`det_conf` live (sigmoid-bounded), FP32, eff-batch 8, EMA/mixup off — it is
self-contained. Bring activity up first (R2), then PSR (R2.5), by toggling
`train_act`/`train_psr` rather than flipping everything at once, so you can
attribute any regression.

**C4 — `--subset-ratio 1.0`: at R3, once R2.5 shows all heads alive** (det not
regressing, activity ≥4 classes, PSR ≥3 patterns). 0.25 is for fast liveness
iteration; 1.0 is for the numbers that go in the table. Activity *requires* full
data for a defensible 75-way report.

---

## D. PSR Floor Diagnosis — confirmed, with the exact code mechanism

**Opus v4 was right, and I can now point at the lines.** The `psr=0.0001000`
you see on ~18,615/18,635 steps is **the `1e-4` NaN-sentinel, not a real loss.**
It is written in three places, and the value is identical because it is the same
hard-coded fallback:

- `losses.py:1041` — first NaN-guard loop, `_fallback = 1e-4` for `psr`.
- `losses.py:1225–1230` — `[PSR_NAN]` guard **before** `_smooth_cap`, replaces
  non-finite `loss_psr` with `1e-4`.
- `losses.py:1258, 1268` — the final `_safe` lambda before Kendall, same `1e-4`.

A *healthy* `binary_focal_loss` at sigmoid≈0.5 with `PSR_FOCAL_GAMMA=1.0`
(config.py:443) and per-component α should be **O(0.1–0.3)**: for a negative
element `p_t=0.5`, `(1−0.5)^1 · −log(0.5) · 0.75 ≈ 0.26`; mean over a
mostly-negative batch ≈ 0.1–0.3. So `1e-4` means **the term is being
zeroed/NaN'd upstream and floored**, exactly as v4 said. The reason it spikes to
0.34–1.0 every ~10 steps is `PSR_SEQ_EVERY_N_BATCHES=10` (config.py:457): the
sequence-batch path (`dim==3`) runs the temporal-smooth branch
(losses.py:1195) and produces a real value; the single-frame path (`dim==2`)
runs the **input-sensitivity penalty** `_sens = -log(per_comp_std + 1e-3)`
(losses.py:1186–1193) — when `psr_logits` are constant or contain an extreme
value, `_sens` goes non-finite → the sentinel fires.

**D1 — Add the raw-loss probe? Yes, and you are 80% there.** The `[PSR_DIAG]`
block already prints the *true* `binary_focal_loss` value before the sentinels
(losses.py:763–784). What's missing is correlating it with *which guard fires*.
Before enabling PSR for real, run a 200-step smoke with `train_psr=True` and:

1. **Set `PSR_SENSITIVITY_WEIGHT=0`** (config.py) to remove the `−log(std)` term.
2. Read `[PSR_DIAG]` + `[PSR_NAN]` in the log. If raw loss is now O(0.1–0.3) and
   finite, the sensitivity penalty was the culprit → re-introduce it **bounded**
   (`_sens = clamp(_sens, 0, 5)`), not removed.
3. The `1041` guard does **not** log — add a one-line warning there too so a
   silent floor can never hide again (same discipline lesson as RC-29).

**D2 — Is the PSR architecture appropriate? Yes; the objective and labels are
the problem, not the head.** Multi-scale GAP → Causal Transformer (3L/4H,
correct causal mask) → 11 per-component MLPs + per-video cache is a sound,
efficient design. What's wrong is **per-frame BCE/focal on fill-forward
(95%-static) labels → constant output is near-optimal** (the `edit_score≈0.477`
constant-pattern artifact). Fix = predict **transition events** (the B2 baseline
is *barely neural* — ASD-confidence accumulation + procedure-order — so a learned
transition model is a strict superset). `psr_transition.py` already implements:
Gaussian-smeared 0→1 transition targets (`build_transition_targets`, σ≈3) and a
`MonotonicDecoder` that enforces monotone fill-forward. Add a procedure-order
prior on top.

**D3 — Competitive with B2 (0.731/0.816) and STORM-PSR (0.506/0.812)? Yes — via
transitions, this is your most beatable benchmark.** Target **F1 ≥ 0.60** (beats
STORM-PSR's 0.506, approaches B2). Per-frame focal as-is will *not* — it tops out
at the constant-pattern artifact. **Do not report any PSR number until the
transition objective is in.** A `psr_f1=0.73`-looking number from the constant
pattern on a skewed eval slice is a trap (it already fooled an earlier round).

---

## E. Efficiency Story — your safest contribution, and the cheapest cells

**E1 — At what metric levels does "5 tasks, one forward pass" become
publishable?** The honest bar (from your own `02_GOALS`): *all 5 heads non-zero
& improving + ≥2 heads match/beat a dedicated baseline + efficiency quantified.*
Your two most winnable "match-or-beat" cells are **PSR** (beat B2 with the
transition model) and **head pose** (no baseline → uncontested). Land those two,
get detection into 0.40–0.60 and activity into 0.30–0.50, and the efficiency
table carries the paper as *"a unified model within striking distance of
specialists at ~1/3 the deployed params and one forward pass."* **The one rule
that makes or breaks this framing: no zeros.** A single collapsed head turns
"efficient generalist" into "broken model."

**E2 — Add GFLOPs/FPS? Yes — it's already in Table 6 and it is free.** Params
are already known: **76.16M total / 53.42M trainable**; the paper rows are
**53.3M (w/o VideoMAE)** and **75.3M (w/ VideoMAE)**; `count_parameters`
(model.py:1996) gives the per-component split. `efficiency_report.py` computes
GFLOPs (fvcore `FlopCountAnalysis`), FPS (200-pass timing), latency p50/95/99,
peak memory. **Run it on day 1 — it fills the entire efficiency table and the
param columns of Tables 4–5 with zero training**, and the headline
*"sum-of-baselines ≈ 75.4M params, 3 forward passes, 3 pipelines"* row writes
itself. Report **both** batched (bs=8) and streaming (bs=1) — streaming is where
the shared backbone + cached temporal heads shine.

---

## F. Remaining Known Issues — triage

| Issue | Verdict / action | Priority |
|---|---|---|
| Activity 1-class collapse | `USE_LDAM_DRW=False` → CE + LS 0.15 for first joint runs; LDAM only later at s=10–15 | **R2 blocker** |
| PSR constant pattern | Transition objective (`USE_PSR_TRANSITION`/`psr_transition.py`); never report per-frame PSR | **R2.5 blocker** |
| Head-pose NaN at val | Was the AMP/RC-29 issue; should be gone under FP32. `combined=0.1107` implies head pose is alive (the `0.15/(1+MAE)` term carries it). Verify in R1; likely already fine | Low |
| Mixup mixes logits | Keep `USE_MIXUP=False`; don't enable until it mixes images pre-forward | Low (off) |
| EMA reset no-op | Fixed (`USE_EMA=False`); re-enable at **R3 only**, after metrics move monotonically | R3 |
| VideoMAE stream | Defer to R3; +22M params/VRAM; worth it for activity +5–7% once GAP head alive. Paper wants both rows | R3 |
| Combined-metric weights (0.30/0.35/0.15/0.20) | It's a *checkpoint-selection* metric, not a paper number. RC-20 (pose-only floor) is dissolving now that det is non-zero. Optional: down-weight pose (saturating, no baseline) so selection tracks benchmarkable heads | Minor |
| Val-line NaN (PSR/act) | Cosmetic — stub dict key mismatch in the Val formatter when eval is skipped. Use `.get(k, float('nan'))`; harmless | Cosmetic |

---

## G. Timeline to Paper-Ready

**G1 — GPU-days.** Using your numbers (~155 min/epoch on 25% at eff-batch 8;
full data ≈ 4× train + the eval fix from §B4):

| Phase | Work | Est. GPU-days |
|---|---|---|
| Day 1 | efficiency table + params + anchor calibration run | ~0.5 (free/cheap) |
| R1 → R1.5 | det bootstrap + anchors + synth pretrain + scale to 1.0 | 3–5 |
| R2 → R2.5 | activity (CE) + PSR (transition) alive | 3–4 |
| R3 | full joint, clip-level, temporal, 30–50 ep @1.0 | 7–12 |
| R4 | multi-seed ×3 + ablations | 5–10 |
| **Total** | **defensible full Table 3 + Table 6 with ±std** | **~20–35** |

A **minimum viable paper** (all heads non-zero, ≥2 competitive, efficiency
quantified) is ~**10–14 GPU-days**; the polished, multi-seed version is ~3–5
weeks of 3060 wall-clock.

**G2 — IndustReal first, IKEA second. Decisively.** IndustReal is the primary
(Table 3), has all 5 tasks, and is where the model already shows life. IKEA
(Table 2) is a *different modality* (third-person), needs body-pose PCK +
segmentation AP + a re-port of the whole pipeline. **Populate IndustReal fully
first.** For IKEA, do a focused port of just **activity + temporal localization**
(the rows with the most baselines) as a one-paragraph *generalization* section,
or mark IKEA as in-progress/future work. Do not let IKEA gate submission.

**G3 — Is the 3060 a bottleneck? For *iteration speed*, yes; for *final
numbers*, no.** Two mitigations that matter:

1. **Embedding-cache two-stage training** (already configured via
   `EMBEDDING_CACHE_DIR`): freeze backbone, cache 512-d embeddings once, train
   activity/PSR temporal heads from cache — "hundreds of epochs/hour." This is
   the single biggest iteration win for the two heads that need the most epochs.
2. **The eval fix (§B4).** Reclaiming 87 min/epoch roughly halves wall-clock.

If a bigger GPU (a cloud A100 for a few days) is available, point it at the
**final full-data R3 + multi-seed** runs only — that compresses the 3–5 week tail
to ~1 week. Not required, but high-leverage for the ±std rows specifically.

---

## Appendix — Do this in the next 72 hours

1. **Let R1 finish; read epoch-1/2 `det_mAP50`.** If ≥0.05 → R2 prep. If still
   constant (std<0.01) *with committed steps confirmed* → det-head LR ×3, +2 ep
   (one knob, per v4).
2. **Fill the efficiency table today (no training):**
   ```bash
   python3 scripts/training/efficiency_report.py --backbone convnext_tiny --batch_size 1   # streaming
   python3 scripts/training/efficiency_report.py --backbone convnext_tiny --batch_size 8   # batched
   python3 scripts/training/efficiency_report.py --use_videomae                            # w/ VideoMAE row
   ```
   → Table 6 + param columns of Tables 4–5 done.
3. **Calibrate anchors (biggest cheap detection win):**
   ```bash
   python3 scripts/training/calibrate_anchors.py   # k-means on GT → new ANCHOR_SIZES
   ```
   Set `ANCHOR_SIZES` (config.py:247) to the clusters; this is R1.5.
4. **Stop burning 87 min/epoch:** add `DET_METRICS_EVERY_N=5` around
   `SKIP_DET_METRICS_EVAL`, and cap the per-epoch gate eval (`EVAL_MAX_BATCHES`
   ~250) — full eval only at checkpoints.
5. **Prep R2/R2.5 toggles:** `USE_LDAM_DRW=False`; `PSR_SENSITIVITY_WEIGHT=0` +
   read `[PSR_DIAG]`/`[PSR_NAN]`; then wire `USE_PSR_TRANSITION=True`.
6. **Discipline (the v4 lesson holds):** any deviation from a prescribed config
   gets its own 200-step smoke with the `[RC-29] optimizer windows` line checked
   before a multi-hour run. A silent floor must never cost a GPU-day again.

**Bottom line:** the model works; the path is throughput, honest framing, and
two targeted wins (transition-PSR, uncontested head-pose) on top of a free
efficiency table. Fill cells by *liveness + honesty* first, chase baselines on
the 2–3 winnable cells second, and keep IKEA out of the critical path.
