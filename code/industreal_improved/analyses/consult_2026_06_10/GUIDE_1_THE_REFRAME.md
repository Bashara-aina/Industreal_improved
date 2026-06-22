# GUIDE 1 — THE REFRAME

*The one document that explains why you were stuck and what changes. Read this first.
Written by Opus after reading config.py, losses.py, model.py, evaluate.py, the dataset
loader, the stage manager, the 48 consult docs, the paper, and the presentation strategy.*

> **These five guides replace the 48 consult docs.** They are forward-looking and
> actionable, not forensic. Use them as your single source of truth:
>
> 1. **GUIDE_1_THE_REFRAME.md** — why you were stuck, the strategy, what "good enough" is *(this file)*
> 2. **GUIDE_2_TRAIN_ALL_HEADS.md** — the concrete plan to make all 5 heads learn
> 3. **GUIDE_3_METRICS_AND_BENCHMARKS.md** — honest evaluation + the numbers that win
> 4. **GUIDE_4_THE_PAPER.md** — how to frame and finish the paper so the idea is *proven*
> 5. **GUIDE_5_RUNBOOK.md** — exact commands, day-by-day, what to stop doing

---

## 0. The one paragraph

You are not blocked by a technical bottleneck. Your model already works well enough to
write a strong paper. You have been blocked by **three things that compound**: (1) you
have been judging detection on a metric that is ~40% measurement artifact, (2) you set
yourself a gate (mAP50 ≥ 0.40 on that artifact metric) that is *mathematically
unreachable*, and (3) you have been doing **forensic analysis instead of decisive
experiments** — 48 documents, eleven "Opus consultations," 313k-line log archaeology —
which feels like progress but produces none. The fix is to **stop optimizing and start
finishing**: report honest numbers, train all heads to "alive and competitive," and
write the paper. Your idea is provable *today* with the model you already have.

---

## 1. What was actually wrong (ranked by importance)

### 1.1 You were chasing a phantom metric — `det_mAP50` is ~40% artifact
`evaluate.py` computes `det_mAP50` by averaging Average Precision over **all 24
channels**, including channel 0 = `background` and the **~8 channels that have zero
ground-truth boxes** in your 50% val subset. Each empty channel contributes AP = 0.0 to
the mean. So even a *perfect* detector caps around 16/24 ≈ 0.67, and realistically much
lower. Your honest present-class number, `det_mAP50_pc` (averaged only over channels that
have GT), is **~0.31–0.35** — roughly your real performance.

You *already computed* `det_mAP50_pc` and even logged a `[DILUTION]` warning about it —
but you kept gating and reporting on the diluted number. **That is the single biggest
self-inflicted wound.** (Fixed now — see §4.)

### 1.2 Your "structural ceiling / OHEM+FocalLoss suppression" hypothesis is wrong
Your own `losses.py` disproves it:
- `DET_GAMMA_POS = 0.0` → positive anchors get the **full** cross-entropy gradient, with
  **zero** focal down-weighting.
- `FOCAL_ALPHA = 0.90` (up to 0.96 per class) → positives are weighted **9–24× more**
  than negatives.
- The loss is normalized by `num_pos`; OHEM keeps the `2×n_pos` hardest negatives.

This loss is tuned to **aggressively favor positives**. It physically cannot be
"suppressing positive gradient." The small detection-head grad norm (~0.02) you flagged
is **not suppression** — it is (a) a tiny head being norm-compared against a backbone with
178 parameter tensors that aggregates every head's gradient (apples-to-oranges), and (b)
a head that has **already converged** on the easy/frequent classes. Your own `LIVENESS`
shows det = 0.62 vs body-pose = 0.94 — same order of magnitude. Detection is not starved.
The "LR restart does nothing" finding is exactly what a converged plateau looks like.

> **Stop running the OHEM ablation. Stop tuning `gamma_neg`, `DET_BIAS_LR_FACTOR`,
> `DET_LR_MULTIPLIER`. They are not your problem.**

### 1.3 The task is fine-grained *state classification*, not object detection
Your 24 detection "classes" are 11-bit assembly-state codes (`'11110110000'` vs
`'11110111100'` — one washer placed, on a near-identical partial assembly). The box is the
whole assembly. So mAP conflates *localization* (easy — one big box) with *ultra-fine-
grained state ID* (hard — 1-bit visual differences). Classes with <100 training instances
(idx 6=65, idx 16=26, idx 8=142) sitting at AP≈0 is **expected**: the model maps them onto
their visual Hamming-twins. This is normal and *not a bug to fix* — it is a property of the
task that you **report and explain**, not grind against.

### 1.4 You are comparing yourself to an unfair baseline
YOLOv8m's 83.8% mAP used **COCO pretraining + 260,000 synthetic images + real
fine-tuning**. Your *own* `02_GOALS` doc says real-data-only detection caps at ~0.50–0.60,
and that your synthetic-pretraining path (`PRETRAIN_DET_ON_SYNTH`, `pretrain_synthetic.py`)
is **wired but never run**. You left your single biggest detection lever switched off and
then blamed OHEM. (See GUIDE_2 §3 and GUIDE_5 for how to use it *if you have the synth data*.)

### 1.5 You are training 5 heads jointly on 12 GB — the interference is the real cost
`config.py` is ~50 dated symptom-patches deep (`[FIX]`, `[RC-xx]`, `[OPUS v8]`, `[REVERT]`).
Almost every one band-aids **multi-task gradient interference**: Kendall caps, log-var
floors/ceilings, per-head grad clips, FPN-detach flags, loss caps, warmup ramps, head
reinits. This is a system fighting itself. You cannot tune your way out of it — you have
to **stop the heads from fighting**, which is what GUIDE_2's decoupled plan does (and the
code for it, `embedding_cache.py`, **already exists in your repo, unused**).

### 1.6 The research process itself is the disease
48 numbered docs that contradict each other and the code (doc 45 says the backbone is
ResNet-50; the code says `convnext_tiny`). Eleven consultation rounds. A whole work-cycle
spent proving "Run 1 and Run 2 are identical" — which only re-confirmed a plateau you
already knew about. **Meta-analysis has replaced experiments.** The cure is in GUIDE_5:
delete the loop, run small experiments, write the paper.

---

## 2. The mental shift

| You have been thinking… | Think instead… |
|---|---|
| "The model is broken — mAP is stuck at 0.207." | "The model is at ~0.33 present-class. The 0.207 was a diluted metric. It's fine." |
| "I must break the ceiling before I can advance." | "There is no ceiling to break. Advance, train all heads, write the paper." |
| "I need one more diagnostic / consultation." | "I have enough information. I need to *run* and *write*." |
| "I must beat YOLOv8m (0.838)." | "I must show a unified model is *competitive at a fraction of the compute*, real-data-only." |
| "Detection failure blocks everything." | "Detection is my hardest task and a *stated limitation*, not a blocker." |
| "Gate = mAP50 ≥ 0.40." | "Gate = each head is alive and credibly learning. Done > perfect." |

The paper's contribution **is the architecture** (one model, one forward pass, 5 tasks,
two-stage FiLM, Kendall). That is real and publishable *without* beating any specialist.
Your job is to get clean, honest, complete numbers for all heads — not SOTA on any one.

---

## 3. What "good enough" means (your real target table)

These are the numbers that make a strong paper. They are **achievable with the model you
have**. Full justification in GUIDE_3.

| Task | Honest metric | "Good enough" target | Baseline (context) |
|------|---------------|----------------------|--------------------|
| Detection (ASD) | `det_mAP50_pc` (present-class) | **0.33–0.45** | YOLOv8m 0.838 *(COCO+260k synth)* |
| Activity | clip-level Top-1 | **0.35–0.45** | MViTv2 0.6525 *(K400)* |
| Activity | clip-level Top-5 | **0.70+** | MViTv2 0.879 |
| PSR | F1 (±3 frame) | **0.50–0.62** | B2 0.731, STORM-PSR 0.506 |
| Head pose | forward angular MAE | **≤ 15°** *(you're at 9°)* | no published baseline → free win |
| Body pose | PCK@0.2 / MAE | report it | — |
| Efficiency | params, GFLOPs, FPS | **53M / 1 pass vs ~81M / 3** | the cleanest win |

If you hit these, you have **two clear wins** (head pose = uncontested; efficiency = by
construction), **one competitive result** (PSR within range of STORM-PSR/B2), and **two
honest "competitive at fraction of compute, real-data-only" results** (detection,
activity). That is a complete, defensible paper.

---

## 4. What I already changed in your code (so the above is real, not advice)

All changes byte-compile (`python -m py_compile`). They are minimal and reversible.

| File | Change | Effect |
|------|--------|--------|
| `src/training/train.py` | `best.pth` selection + the `combined` metric now use **`det_mAP50_pc`** (present-class), not diluted `det_mAP50`. `det_mAP50` is still logged as the paper number. | Your checkpoints are now selected on real performance; the headline you watch is honest. |
| `src/training/train.py` | The `Val:` log line now prints `det_mAP50_pc` and `det_n_present` too. | The stage manager can read the honest metric; you can see it every epoch. |
| `src/training/stage_manager.py` | All 10 stage gates (rf1–rf10) now gate on **`det_mAP50_pc`** at achievable, monotonic thresholds (0.22→0.35), and the unreachable `det_mAP50_95` sub-gates were removed. | The curriculum can actually advance, so **all heads get trained**. |

> I could **not** run training here (this is a fresh cloud container with no GPU, no
> torch, and your dataset isn't mounted). These edits are verified by reading + syntax
> compile. You run them on your box. GUIDE_5 tells you exactly how and what to watch.

---

## 5. The three rules from here on

1. **No new analysis documents.** If you have an idea, it becomes an *experiment* (a
   command in GUIDE_5), not a markdown file. The consult-doc loop is over.
2. **Judge everything on honest, per-task metrics** (GUIDE_3). Never report or gate on
   diluted `det_mAP50` again.
3. **"Done" beats "perfect."** A complete, honest, imperfect results table is a finished
   paper. An eternally-tuned rf2 is not. Drive to the Definition of Done in GUIDE_5 §1.

➡ **Next:** GUIDE_2 — how to make all five heads learn correctly.
