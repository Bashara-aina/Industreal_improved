# 54 — Day-by-Day Execution Plan (Grounded, Runnable)

> **Generated:** 2026-06-22 (Opus). Answers doc 50 §12, §14.1. Every command references a **verified** preset/script. Confirm exact CLI flags against `train.py`'s argparse and `GUIDE_5_RUNBOOK.md` before launching (forms shown match doc 49 + the script docstrings).
> **Premise (from `51`):** the rf2 gate is already cleared (0.304 ≥ 0.28, 9.13° ≤ 60°). The plan is built around *advancing*, not waiting.

---

## Phase 0 — Free numbers + diagnostics (TODAY, ~0–1 GPU-h, runs alongside rf2)

These need almost no GPU and fill real paper cells immediately. Do them first.

**0.1 Efficiency table (paper §5, Tables 5/8) — fully fillable now.**
The code exists (`evaluate.py:2881` `thop.profile`; `eff_fps`, `eff_fps_streaming` at 2951-2953). Run the efficiency path on `best.pth`:
```bash
# form: invoke evaluate.py's efficiency measurement on the current best checkpoint
python3 src/evaluation/evaluate.py --ckpt src/runs/rf_stages/checkpoints/best.pth --efficiency-only
#   → writes eff_params_m, eff_gflops, eff_fps, eff_fps_streaming
```
If `--efficiency-only` isn't a flag, call the efficiency function directly (it's a standalone in evaluate.py). **Deliverable:** params (M), GFLOPs, batched FPS, streaming FPS, and the param-reduction factor vs N specialists. *Expect ~53M params per the arch; GFLOPs/FPS unknown until measured — that's the point.*

**0.2 Per-class detection diagnostic (`52` §5).**
```bash
python3 src/diag_per_class_truth.py --run src/runs/rf_stages    # once a det eval has written per-class AP
```
Also render/inspect the 24×24 confusion matrix (already produced by `evaluate.py`'s `compute_det_confusion_matrix`). **Deliverable:** which present classes are stuck, and whether their error mass is on one-bit-adjacent states.

**0.3 Plot-from-logs figures (`53` §8).** Kendall `log_var_*` curves, per-task val curves — all already in `metrics.jsonl`.

**0.4 Draft prose** that needs no numbers: limitations, the detection reframe (`53` §3), conclusion skeleton (`53` §6).

---

## Phase 1 — Let rf2 finish + full eval (Day 0–1, ~13 GPU-h, already running)

Do **not** intervene in detection. Let the running rf2 reach epoch 36, then run the full-test eval.
```bash
# after rf2 completes:
python3 src/evaluation/evaluate.py --ckpt src/runs/rf_stages/checkpoints/best.pth --split test
```
**Deliverables (fills 🟢 rows in `53` §5):** `det_mAP50`, `det_mAP50_pc`, `det_mAP_50_95`, head-pose forward/up/position MAE, det confusion matrix PNG.
**Expect:** `mAP50_pc ≈ 0.30–0.38`, `mAP50 ≈ 0.20–0.25`, head-pose forward ≈ 9–12°.

> If the stage_manager auto-advances rf2→rf3 *before* epoch 36 (because the gate now reads as passed), that is **fine and desirable** — let it. You can run the rf2 full eval from the saved `best.pth` afterward.

---

## Phase 2 — PSR go/no-go (Day 1, ~1 GPU-h) — DECISIVE

PSR has never produced a real signal (`51` F4: `psr_f1_at_t = 0.0` in every snapshot). Settle it with one cheap test **before** committing rf4 time.

**Test:** overfit the PSR head alone on ~50 sequences, fully decoupled from detection (Phase-B style from cached or on-the-fly features), 200 steps.
- **If `psr_f1_at_t` rises above ~0.3 on the 50-seq train set** → PSR *can* learn; schedule it in rf4 and report honestly.
- **If it stays at 0.0 / `comp0=1.0`-only** → PSR is structurally stuck. **Drop it** to a one-paragraph negative result (`53` §7.3). Do not spend rf4 budget on it.

This is a go/no-go, not an optimization. One hour buys a binary decision.

---

## Phase 3 — Activity: advance to rf3 and train (Day 1–2, ~11 GPU-h)

The gate is passed (`51` F1). Advance and train activity — this is the **next real milestone** and the next gate (`act_top1 ≥ 0.22`, `stage_manager.py:172`).

```bash
# the stage manager will auto-launch this on advance; or launch directly:
python3 src/training/train.py --preset stage_rf3 --resume src/runs/rf_stages/checkpoints/best.pth
#   stage_rf3: subset 0.35, 15 epochs, train_act=True, det+pose+act  (config.py:1136)
```

**Two fixes before/at launch (from evidence):**
1. **LDAM-DRW schedule** — the DRW reweighting switches at epoch ~60 (50 §3.2/Q3.5); rf3 is only 15 epochs, so DRW never activates. Either move the switch to ~epoch 8, or use class-balanced weights from epoch 1, so the long tail (support 6 → 6572, `metrics.jsonl`) gets *some* signal. (Note `paper_run` has `use_ldam_drw=False`, `config.py` — confirm which loss rf3 actually uses and set deliberately.)
2. **Watch the collapse mode** — at init, activity collapses to one majority class (`51` F7). Confirm the confusion matrix spreads off the diagonal within a few epochs; if it stays single-column, lower act LR or check the act ramp (`act_ramp = min(1, epoch/5)`, losses.py).

**Deliverables:** `act_accuracy` (Top-1), `act_top5_accuracy`, activity confusion matrix.
**Expect:** Top-1 **10–30%** (vs 1.3% chance). ≥15% = "demonstrates multi-task transfer" (paper-usable). ≥22% = clears rf3 gate.

---

## Phase 4 — Ablation A (the actual contribution) (Day 2–3, ~8 GPU-h)

This is the paper's scientific core (`55` §1). You already have **multi-task** detection (rf2). You need the **single-task** half on the *same backbone*:

```bash
# single-task detection baseline (det + head_pose only) — identical backbone:
python3 src/training/train.py --preset recovery_det_only
#   train_det=True, train_act=False, train_psr=False  (config.py:932)
```
Compare `det_mAP50_pc(single-task)` vs `det_mAP50_pc(rf2 multi-task)`. **That delta IS the "no catastrophic interference" result.** Do detection first (most important comparison); add single-task activity/head-pose **only if** the embedding cache proves out (`55` §3, it's untested).

**Deliverable:** the `tab:abl-heads` single-vs-multi row(s). **Expect:** |Δ| small (a few mAP points) → supports the thesis. A *large* negative Δ is also publishable (it would mean interference is real — a finding, not a failure).

---

## Phase 5 — Ablation B (FiLM), contingent (Day 3, ~6 GPU-h)

Only meaningful if activity trains (Phase 3 ≥ ~15%). Run the FiLM ladder on activity: no-FiLM → PoseFiLM → HeadPoseFiLM → both. If activity is too weak to modulate, **report B as inconclusive** and keep the architecture description (`53` §7.4).

---

## Phase 6 — Write & assemble (Day 3–4, overlaps compute)

Fill tables as runs land, generate the 4 plot-from-logs figures, write the conclusion with real `[bracket]` values (`53` §6). Submission-ready draft = honest detection + head pose + efficiency + Ablation A + activity-if-it-works.

---

## The optimal ordering, condensed (answering 50 Q10.2 / Q14.1)

```
NOW   (0 GPU):  efficiency table + per-class diagnostic + log-figures + draft prose
Day0-1 (13h):   rf2 → epoch36 → full test eval   [det + head-pose numbers]
Day1   (1h):    PSR go/no-go                      [keep or drop decision]
Day1-2 (11h):   rf3 activity (gate already open)  [activity Top-1/Top-5]
Day2-3 (8h):    recovery_det_only single-task     [Ablation A — the contribution]
Day3   (6h):    FiLM ladder if activity works     [Ablation B]
Day3-4:         write, figures, fill tables, submit
```

**Single-GPU wall-clock:** ~33–39h of sequential training + ~10h writing (writing overlaps). The Phase-0 work and writing run on CPU/in-parallel, so they're "free" against the GPU timeline.

---

## Triage if you have only 48h (answering 50 Q12.3)

1. Phase 0 (free numbers) — **always do.**
2. Let rf2 finish + eval — detection + head pose.
3. **Advance to rf3, train activity** even partially — get *a* Top-1 number.
4. **recovery_det_only** for Ablation A detection — the one ablation that defines the paper.
5. Submit with: detection (honest), head pose, efficiency, Ablation A, activity-as-preliminary. PSR/FiLM/IKEA → "ongoing / camera-ready."

This is a viable workshop/BMVC submission. The irreducible core (`53` §7) is reachable inside 48h because **detection is already done and the gate is already open.**

---

## What to expect (so you know when something is *wrong*, 50 Q14.1.5)

| Signal | Healthy range | Alarm |
|---|---|---|
| rf2 final `det_mAP50_pc` | 0.30–0.38 | < 0.25 → check eval split |
| Head-pose forward MAE | 8–13° | > 30° → unit-vector normalization bug (saw 68° in old eval) |
| rf3 activity Top-1 | 10–30% | < 3% → still collapsed; fix LDAM/ramp/LR |
| Ablation A |Δ mAP50_pc| | 0–0.05 | > 0.10 → real interference (report it, don't hide) |
| PSR go/no-go f1_at_t | >0.3 (keep) or 0.0 (drop) | anything between → re-run, ambiguous |
| Efficiency params | ~53M | wildly off → wrong model variant loaded |
