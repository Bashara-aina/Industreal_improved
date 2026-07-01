# 82 — Opus Final Verification Response (files 77–81)

Date: 2026-07-01 · Model: Opus 4.8 · Method: read 77–81 line by line, verified every
claim against `main` source (post-merge). Two code fixes shipped this round; answers to
all 6 questions below, each grounded in the code. **Verdict: GO for the 50-step probe →
100-epoch run, with 2 caveats (PSR framing, sampling monitor).**

First, a note: **file 80 §5/§2.2 are stale** — they show `WEIGHT_DECAY=5e-2` and
`GRAD_CLIP_NORM=1.0`, but the actual config is `1e-3` / `5.0` (the round-4 fixes landed;
verified `config.py:512,521`). The MD wasn't updated; the code is correct.

---

## Q1 — Sampling distribution distortion (highest priority): REAL, bounded, now monitored

Confirmed there are **three stacked reweightings** in `get_sampler`
(`industreal_dataset.py`): (1) `ACT_SAMPLER_MODE='balanced'` → equal per-class mass;
(2) `DET_GT_FRAME_FRACTION=0.40` → forces 40% of mass onto GT-bearing frames; (3)
`USE_TASK_AWARE_SAMPLING=True`, `DET_BOOST=2.0`. Your worry is correct: activity balance
is preserved **within** the GT and non-GT pools, but a class whose frames rarely co-occur
with detection boxes is confined to the 60% non-GT pool, while a class that co-occurs
with GT draws from both. So the *realized* per-class rate is distorted by the
class↔GT-presence correlation.

**Is it acceptable?** For RF3+ (`det_frac=0.40`) yes — the distortion is bounded (neither
pool starves a present class to zero). But it must be *watched*, not assumed. **Shipped:**
a one-time **effective per-class sampling-mass log** in `get_sampler` (prints #present
classes, max/min mass ratio, top-5 sampled ids). Read it at epoch 0: if the ratio is a
few× it's fine; if it's ≫10× some verb groups are being over-sampled via their detection
correlation, and you'd drop `DET_GT_FRAME_FRACTION` to ~0.25 or `DET_BOOST` to ~1.3.

## Q2 — GIoU range: your understanding is correct, AND round-4 fix #6 is a no-op

`reg_loss` is torchvision `generalized_box_iou_loss` (`losses.py:391`), which returns
**1 − GIoU ∈ [0, 2]** — always non-negative. Therefore `loss_det = cls + 2·reg ≥ 0`
always, and the `NEG_SLOPE` floor (`losses.py:1247`) **never executes**. Confirmed: no
other term can make `loss_det` negative.

**Correction the team needs:** the code comment at `losses.py:1237` ("reg_loss can be
negative, GIoU ∈ [-1,1]") is **wrong**, and round-4 **fix #6 (NEG_SLOPE 0.0→0.01) did
nothing** — it cannot have "restored regression gradient for non-overlapping boxes,"
because (a) the floor branch never runs, and (b) the gradient for a non-overlapping box
comes from `1 − GIoU` (which is ~2 with a real gradient), entirely upstream of the floor.
Harmless, but don't attribute any behavior change to it. (The floor is fine as dead
defensive code; just delete the misleading comment when convenient.)

## Q3 — PSR double-weighting: stable; but the real PSR issue is temporal, not weighting

Focal alpha (per-component, from prevalence, `clamp(min=0.1)` and `binary_focal_loss`
`max=1.0`) and `_psr_comp_weights` (uniform per-component) serve different roles
(pos/neg asymmetry vs component scaling) and are both **bounded by clamps** → no
runaway over-emphasis. The `-log(mean(std))` sensitivity penalty is a real backstop
against collapsed logits. **No instability expected. Weighting is fine.**

The PSR concern that actually matters (file 79 §5, which you flagged honestly): with
`STAGED_TRAINING=False` + the shuffled sampler, **PSR runs per-frame (T=1)** — the causal
transformer's mask is a no-op and the temporal-smoothing term computes over T=1
transitions. So this head is a **per-frame component-state classifier, not transition
detection.** That is a perfectly good task and will give a real number (binary accuracy
~0.75–0.85), but **the paper must call it per-frame procedure-component recognition, not
transition/PSR-order detection** — otherwise it's a claim/setup mismatch a reviewer will
catch. If you need true transition detection, that requires the sequence-mode dataloader
(non-shuffled), which is a separate build.

## Q4 — LR warmup vs detection: the premise is overstated; leave WARMUP_EPOCHS=2

The doc says "detection is at full strength for only ~0.9 epochs." That's misleading:
OneCycleLR **peaks** at epoch 2 then **cosine-decays over 98 epochs** — LR stays high for
~10–20 epochs, so detection gets ample high-LR time well past epoch 2. Detection's own
250-step internal warmup + the 1000-step reg warmup fit inside that window. **I did not
change `WARMUP_EPOCHS`** (it's your paper-spec value, `config.py:516`). Recommendation:
leave it at 2; check `mAP50_probe` at epoch 5 — only if it's stuck <0.005 is raising to 5
worth trying. Raising it is low-risk but not indicated yet.

## Q5 — Segment eval protocol bug: CONFIRMED and FIXED

Verified in `compute_activity_segment_metrics` (`evaluate.py:851`): `label` is a **raw
action_id** from `build_activity_segments()`, but `pred = act_logits.argmax()` is in
**grouped output space** (`NUM_ACT_OUTPUTS` channels). With `ACT_CLASS_GROUPING='hybrid'`
they're different index spaces → `act_seg_top1/top5` are meaningless. Since this *is* the
MViTv2-comparable clip metric you'd report, it had to be fixed.

**Shipped:** remap `label` through `remap_activity_label()` (gated on grouping ≠ 'none';
NA-skip stays on the raw id, before remap) and fixed the hardcoded `torch.zeros(1,75)`
fallback to `NUM_ACT_OUTPUTS`. Now both sides share group space.

## Q6 — Anything else? Three things, ranked

1. **PSR temporal no-op (Q3)** — biggest. Not a bug, but frame the paper honestly
   (per-frame component recognition). Highest risk of a reviewer objection.
2. **`act_accuracy_no_na` vs group 0** (file 77 §6, "High cosmetic"): group 0 is the
   reserved `other` group, so `_no_na` excludes a real group. **Don't report
   `act_accuracy_no_na`** — report `act_accuracy` (clip-level, from the now-fixed segment
   eval) and `act_macro_f1`. Already your plan; just don't slip the `_no_na` number into a
   table.
3. **`set_class_counts` re-creating the CE module** (file 77 §3.3 flagged as "check"):
   **not a problem.** `nn.CrossEntropyLoss.weight` is a buffer, not an optimized
   parameter — the optimizer holds the model params + the 4 Kendall `log_var`s, none of
   which live on `act_loss_fn`. Re-creating it each epoch cannot disrupt optimizer state.
   Verified.

No other convergence-blocking mechanism found. The Q1 monitor + the two fixes close the
gaps that were verifiable in code.

---

## Readiness verdict

**GO** — run the 50-step `SIMPLIFY_LOSS=True` probe, and if the 6 step-50 signals pass,
launch the 100-epoch run. The go/no-go tables in 77–80 are sensible. Realistic
expectations, stated honestly:

- **Detection**: on track for the 0.2–0.5 range over training; 0.50–0.65 is optimistic
  but not impossible. Fair to report next to YOLOv8m 0.838 **as long as it's the same
  IndustReal ASD protocol/classes** (see file 82-round-3 fairness note).
- **Head pose (forward-gaze)**: 8.71° holds; report forward/gaze direction only (up
  vector ~95° is unlearned — round-3 finding).
- **Activity (grouped, clip-level)**: the segment fix makes this number real; low-0.x
  top-1 plausible. Remember the round-3 fairness rule — the grouped number is a *different
  task* from IndustReal's 74-class AR; establish your own baseline for it, don't drop it
  next to MViTv2's 0.653.
- **PSR**: a per-frame component-recognition number (~0.75–0.85 binary acc), **not**
  transition detection — frame accordingly.

**Confidence the run completes and produces reportable numbers for 3–4 heads: high.**
**Confidence all epoch-100 target *ranges* are hit: moderate** — the ranges in 77–80 are
optimistic ceilings, not expectations. Launch, watch the epoch-2 and epoch-5 gates, and
adjust `DET_GT_FRAME_FRACTION` / `DET_BOOST` only if the Q1 sampling log shows a large
distortion.

## What I shipped this round
| Change | File | Purpose |
|--------|------|---------|
| Segment-label remap to group space + fallback width | `evaluation/evaluate.py` | Q5 fix — makes `act_seg_top1/top5` valid under grouping |
| Per-class effective sampling-mass log | `data/industreal_dataset.py` | Q1 — surfaces the DET_GT/task-aware distortion before the run |

Both compile clean. No training-dynamics values changed (I did not touch
`WARMUP_EPOCHS`, weights, or LR) — only a correctness fix and a diagnostic.
