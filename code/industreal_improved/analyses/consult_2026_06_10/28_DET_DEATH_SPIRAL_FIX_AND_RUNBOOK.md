# RF1 Death-Spiral Fix + RF1→RF10 Runbook (Opus v6 answer)

> Generated 2026-06-16. Concrete fix for the detection class-imbalance death
> spiral, with a verify-before-you-burn-GPU protocol. Companion to
> `26_RF1_RF10_COMPREHENSIVE_STATUS.md`.

---

## TL;DR

The death spiral is a **data-sampling** bug, not an optimisation bug. You proved
this yourselves: 5 LR strategies gave *identical* trajectories (a deterministic
data problem). The detector sees a GT box in **<1% of steps**, so its positive
logits decay between the rare GT batches.

The real fix is upstream of the loss: **guarantee a fixed fraction of every batch
carries GT boxes.** The loss has been patched ~6 times (RC-28 skip→bounded-bg,
OHEM, asymmetric gamma, DET_EMPTY_SAMPLE, clamps, GIoU guards) — all of these are
compensating for the wrong batch composition. Fix the batch and most of those
band-aids become unnecessary.

**Why the existing `TASK_AWARE_DET_BOOST=2.0` doesn't work:** a *constant*
multiplier scales with the base density. `2 × 0.7% ≈ 1.4%` — still starved. The
new knob targets an **absolute** GT fraction, independent of base density.

---

## The contradiction you must resolve first (1 command, no GPU)

`config.py:503` says the activity sampler yields **~24% GT batches**.
`26_...STATUS.md` says **0.7%**. These can't both be true for RF1. The likely
cause: `subset_ratio=0.2` + the greedy **activity-coverage** subset selection in
`_scan_and_index` ignores OD-label availability, so the RF1 subset may contain
recordings with almost no OD labels.

Run this — it reports the true per-frame AND per-recording OD coverage for the
exact RF1 subset, and simulates the sampler before/after the fix:

```bash
python diag_gt_coverage.py --preset stage_rf1 --subset-ratio 0.2
```

- If **frames-with-boxes** is a few % but legacy per-batch GT is <2% → the
  sampler is the bug; the fix below lifts it to ~90%.
- If **many recordings have ZERO OD** → recording-level sparsity. The sampler
  can't invent GT frames that aren't in the subset → see "Watch item B".

---

## What I changed (3 edits, all flag-gated, legacy-safe)

| File | Change |
|------|--------|
| `src/config.py` | New `DET_GT_FRAME_FRACTION` (default `0.0` = legacy). Derived per-stage in `apply_preset()` from active heads, logged every run (non-silent). |
| `src/data/industreal_dataset.py` | `get_sampler()` redistributes sampling mass so exactly `DET_GT_FRAME_FRACTION` of each batch is GT-bearing, preserving activity balance *within* the GT/non-GT groups. No-op when `0.0`. |
| (new) `diag_gt_coverage.py`, `overfit_one_batch.py` | Diagnostics (below). |

Per-stage values (auto-derived; override by adding `det_gt_frame_fraction` to a preset):

| Stage | Heads | DET_GT_FRAME_FRACTION |
|-------|-------|----------------------|
| RF1, RF2, recovery_det_only | det (±pose) | **0.9** |
| RF3–RF10, paper_run | det + act/psr | **0.4** |
| no detection | — | 0.0 (off) |

Expected effect at effective batch = `BATCH_SIZE(4) × GRAD_ACCUM(8) = 32`:
`0.9 × 32 ≈ 29` GT frames/step (was ≈0.2). ~60–150× more positive signal — the
death spiral cannot form.

---

## Verify-before-you-burn-GPU (do these IN ORDER)

**1. Coverage (seconds, CPU):**
```bash
python diag_gt_coverage.py --preset stage_rf1 --subset-ratio 0.2
```
Confirm the FIX row shows ≫1 GT frames per effective batch.

**2. Can the architecture even learn? (minutes, 1 GPU):**
```bash
python overfit_one_batch.py --steps 400 --lr 1e-3
```
- `det_cls → ~0` and `max(+logit)` climbs past +2 → **architecture is fine**; the
  spiral was purely data. Proceed to step 3.
- Loss plateaus / logits stuck → **architecture bug** (not sampling). Debug
  `losses.py:_match_anchors` (the [0,1] IoU normalisation) and `_decode_boxes`
  before anything else. Use `detection_collapse_probe.py` for best-IoU-vs-GT.

This single test would have saved weeks: it isolates "can it learn" from "is it
being fed correctly."

**3. Short RF1 (1–2 epochs):** launch RF1 as usual. Watch the log for
`[get_sampler] DET_GT_FRAME_FRACTION=0.90: N/M frames carry GT -> ~90% ...` and
for `cls_max` staying high (no decay to +0.05). The first completed epoch should
emit the first-ever `det_mAP50`.

---

## RF1→RF10 protocol

Your staged ladder is **sound** — progressive head activation is exactly the
right cure for the old monolithic collapse (PSR never escaping zero-loss,
Kendall pinned). You're not stuck because of staging; you're stuck on rung 1
because of sampling. Per stage:

1. **Resume** from previous stage `best.pth` (already wired).
2. **Sampler**: `DET_GT_FRAME_FRACTION` auto-set (0.9 early → 0.4 once act/psr on).
3. **Gate**: only advance on a *real* mAP from a *completed* epoch. Do not let a
   capped 200-batch gate-eval (few GT frames → noisy mAP) gate you — see Watch A.
4. **New head turns on** (e.g. RF2 pose, RF3 act, RF4 psr): expect that head's
   loss to be the only one moving for ~1–2 epochs while detection holds. If
   detection mAP *drops* >5% when a head activates, lower that head's loss weight
   (you already have `ACTIVITY_LOSS_WEIGHT`, Kendall) rather than disabling it.

---

## Answers to the 6 questions

1. **GT oversampling impl** — done: `DET_GT_FRAME_FRACTION` + `get_sampler()`
   redistribution. Absolute-fraction targeting, not a constant boost.
2. **RF2 transition** — resume RF1 `best.pth`; pose turns on; frac stays 0.9
   (pose is dense, oversampling GT frames doesn't starve it). Gate on det mAP
   *and* `forward_angular_MAE`.
3. **PSR zero-loss equilibrium** — same imbalance *family*, different axis. PSR
   positives are rare per component; per-frame focal sits at all-negative. Note
   `apply_preset` force-disables `USE_PSR_TRANSITION` when not in sequence mode
   (config.py ~1217) → PSR falls back to per-frame focal → equilibrium. Fix:
   train PSR in **sequence mode** (RF4+) so the transition objective is active,
   and/or per-component positive weighting. GT-frame oversampling alone won't fix
   PSR.
4. **Dead non-det heads in RF1** — *expected, not a bug*: RF1 sets
   `train_act/psr/head_pose=False`, so they have no loss → zero grad. They are
   not dying from FPN collapse in RF1; they're simply off by design. (In the old
   R2.5 monolithic run they died from Kendall domination — staging fixes that.)
5. **164K anchors / switch to roi_detector?** — **No.** The anchor count is not
   the bottleneck (memory is dominated by feature maps, ~63 MB for the cls tensor
   at B=4). Switching detectors mid-crisis is exactly the "add more machinery"
   move that's kept you stuck. Fix the sampler first; revisit only if the overfit
   test (step 2) fails.
6. **Minimum viable experiment** — `overfit_one_batch.py`. If it can't overfit
   one batch, nothing downstream matters.

---

## Watch items

**A. Gate-eval GT coverage.** The gate eval is capped at `GATE_EVAL_MAX_BATCHES=200`
with NO sampler (sequential val). At ~0.7% density that's only a handful of GT
boxes → mAP is noise and the gate may read ~0 even when training is healthy. Do
**not** fix this by oversampling val (that inflates mAP and is invalid). Instead,
on gate epochs run a *full* det eval (you already have `_det_every_n` for "full
det mAP every N epochs", train.py ~3753) or build a detection-val subset of
OD-bearing recordings. Verify a completed-epoch mAP is non-trivial before trusting
the gate.

**B. Recording-level sparsity.** If `diag_gt_coverage.py` shows most recordings
have ZERO OD, the subset itself is starved. The sampler fix can't help. Options:
(i) raise `--subset-ratio` for RF1, or (ii) make `_scan_and_index`'s greedy
selection OD-aware for detection stages (prefer recordings with `num_dets>0`).
This is a small, well-scoped follow-up — run the diagnostic first to see if it's
needed.

---

## Honest caveats

- I **could not execute** the training/diagnostics here (no GPU, no IndustReal
  data, no torch/numpy in this environment). The code edits are written against
  the verified live APIs (`get_sampler`, `apply_preset`, `POPWMultiTaskModel`,
  `MultiTaskLoss`, `_prepare_images`) and the config change is runtime-verified
  to import and derive the right per-stage values. The two scripts are written
  against those same APIs — run step 1 first; if an import path differs on your
  tree, it'll surface immediately and cheaply.
- The **GitNexus MCP tools** mandated by `CLAUDE.md` are **not connected** in this
  session (only GitHub + Vercel MCP are available), so I did manual impact
  analysis instead: `get_sampler`'s only caller is the DataLoader builder
  (`train.py:239`); the config change is additive. Re-run `npx gitnexus analyze`
  locally to refresh the index after pulling.
- The fix is necessary but may not be *sufficient* alone if Watch A/B bite. Run
  the diagnostic; it tells you which regime you're in.
