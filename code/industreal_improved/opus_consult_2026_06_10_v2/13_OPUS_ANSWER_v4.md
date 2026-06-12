# 13 — Opus Answer v4: RC-28 Verified in Part, RC-29 Is the Headline (2026-06-12)

Response to `12_MASTER_PROMPT_v4.md`. All code changes described in §4 are
**applied on this branch** (losses.py, train.py, config.py).

---

## 0. Verdict first

**Your two recovery runs almost certainly committed (close to) zero optimizer
steps. This was not an equilibrium — it was a frozen model. RC-29.**

The decisive evidence is your own: **4 validation cycles identical to 4
decimal places, including `loss=132.4092`,** across 2 more epochs of
"training." A model sitting at a genuine focal-loss equilibrium still moves —
the activity, PSR, pose, and Kendall parameters all receive nonzero gradients
every step, and val loss would jitter in the 3rd–4th decimal. Bit-identical
val across cycles means **bit-identical weights**, and the mechanism is in
your own journey doc §6: the recovery preset as actually run was hand-edited
to **`'mixed_precision': True'`** (comment: "[TEMP: AMP for 12GB VRAM]") —
re-enabling fp16, the **confirmed R2 failure mode** (first NaN at
`backbone.0.conv1.weight`, `diag_amp_nan.py`). Under fp16, `GradScaler`
**silently skips `optimizer.step()`** whenever the unscaled grads contain
inf/NaN — no warning, no error, the progress bar keeps showing healthy
per-batch losses because the *forward* pass is fine. Every symptom fits:

| Observation | RC-29 explanation |
|---|---|
| Train loss "non-zero, non-NaN, 25–160" | Forward pass works in fp16; losses vary because *batches* differ, not weights |
| 4 val cycles identical to 4 decimals | Zero (or near-zero) committed steps after the first val |
| Run 2 (no reinit, +2 epochs) identical | Same skip on every window |
| Step-0 assertion PASSED | It's a forward-only probe — it cannot see skipped steps |
| det scores at 0.154, not at the init prior 0.05 | A handful of early windows committed while the scaler was still calibrating its scale downward, drifting the constant; then skipping became universal |

**One run, one decision: FP32.** And so a frozen run can never again hide for
8 GPU-hours, this branch adds step-commit telemetry (§4.2) that prints a
per-epoch `committed/skipped` verdict and an ERROR line when an epoch commits
zero steps.

---

## 1. Q1 — Is RC-28 (3-way deadlock) correct?

**Partially. One leg is real but mis-attributed, one is overstated, one is
secondary. And none of the three can explain bit-identical validation — that
required RC-29.**

### Det leg — REAL, but the mechanism is normalization, not "Focal at π=0.05"

The killer is one line: `num_pos = max(pos_mask.sum().item(), 1)`
(losses.py:223, pre-fix). An image **with no GT** falls through the matcher
with `pos=0`, gets `num_pos=1`, and contributes its **entire summed negative
focal loss — ~173K anchors × 24 classes ≈ 4.15M elements ≈ 130–200 loss
units — divided by 1**, while a GT-bearing image contributes ~3–4. Your own
log line proves it: "GT batches: loss 5–30; No-GT batches: 120–160." With
~85% of sampled frames empty (the `WeightedRandomSampler` balances ACTIVITY
classes, not GT-box presence — train.py:227), the detection gradient is
~30:1 dominated by a uniform "push every score down everywhere" signal. The
cheapest solution for the optimizer is the one you observed: shrink the final
conv weights toward zero and park the bias at the constant-loss optimum
(scores flat at 0.154, std 0.0095 ≈ pure bias output).

Focal loss itself is fine; π=0.05 is fine; α=0.75 (your config already gives
positives 3× weight) is fine. **Standard detector practice excludes
annotation-free images from detection training entirely** (COCO training
does not contain empty images). Fixed in §4.1.

### Activity leg — OVERSTATED

`det_conf` is **24 of 1048 input dims**. The other 1024 dims (GAP(C5_mod) +
GAP(P4)) vary per frame, and your own metric proves the activity head sees
signal: `act_top5 = 0.2425` ≫ random (0.067). Zeroing det_conf removes one
conditioning signal; it does not blind the head. The 1/75 collapse is the
familiar LDAM(s=30)-under-near-zero-effective-training signature — and under
RC-29 the head received ~0 updates anyway. That said, your instinct is right
for a different reason: **the zeroing's reason-for-being is gone.** It
guarded against *saturated raw* det_conf (O(10–100), L2 243.39 constant);
that's now fixed at the source (sigmoid bound + healthy logits, step-0
median |z| = 2.95). It is pure cost now. **Disabled in the preset** (§4.3) —
and note this also retires the train/eval `ZERO_DET_CONF=1` consistency
burden for future checkpoints.

### PSR leg — SECONDARY

"Backbone gradient dominated by detection's negative focal mass" was true
pre-fix and is addressed by the same normalization fix. But PSR's constant
pattern is primarily its own known issue: fill-forward labels are
near-constant within recordings, so per-frame focal on 95%-static targets
makes a constant output near-optimal (`psr_edit=0.4773` is the
constant-pattern artifact, as in the June 10 round). PSR needs the
transition-weighted objective (you've already stubbed `psr_transition.py`)
and epochs — after the model actually trains.

---

## 2. Q2/Q5 — Escape strategy and the focal-loss question

Minimum set, in order of necessity:

1. **FP32. Non-negotiable for recovery** (RC-29). There is no bf16 plumbing
   in this tree (`autocast` defaults to fp16); FP32 at batch 1 fit on the
   3060 in every earlier run. If you later want AMP for speed, add
   `dtype=torch.bfloat16` plumbing — bf16 has fp32's exponent range and
   doesn't scaler-skip — and re-validate with the new telemetry line.
2. **Skip empty frames in the det loss** (§4.1). This converts detection
   from "97% predict-nothing pressure" to purely GT-supervised.
3. **`zero_det_conf: False`** (§4.3).
4. **Keep π=0.05, keep α=0.75, keep LR.** Don't turn three knobs blind
   while two structural bugs are being removed — you'd be unable to
   attribute the outcome. Raise det LR only if Stage R1 (below) stalls
   *with telemetry showing committed steps*.
5. **OHEM/Varifocal/GHM/ATSS: not now.** Focal + empty-frame exclusion +
   force-assigned best anchor per GT (already in `_match_anchors`) is a
   sound baseline. ATSS is a genuine upgrade to consider *after* the
   pipeline demonstrably learns — it changes assignment, which would
   confound the current experiment.

---

## 3. Q3/Q4 — Staged recovery protocol, and fresh vs salvage

**Fresh ImageNet init. Stop fighting the epoch-43 lineage.** Reasoning: under
RC-29 your 4 epochs were (near-)null, so the lineage has STILL never been
trained with healthy mechanics — its only proven property is that it keeps
costing GPU-days. A fresh init removes every residual doubt (optimizer state,
Kendall state, partial-load GroupNorm keys, D9 ambiguity) for the price of
the det-only bootstrap stage, which is cheap because empty frames no longer
contribute det loss. Salvage is acceptable *only* if you have a strong
external reason; nothing in the evidence supports one.

### Stage R0 — freeze-proof smoke (~20 min GPU)

```bash
TRAIN_MAX_STEPS=200 python3 src/training/train.py --preset recovery_det_only \
  --subset-ratio 0.25 --max-epochs 1 --seed 42        # fresh init: NO --resume
```
Gates: epoch summary line shows `committed > 0, skipped = 0`
(FP32 ⇒ scaler inert); det cls loss varies *and trends down* across GT
windows; no `[RC-29]` warnings. If `committed=0` in FP32, the freeze has a
second cause — stop and send me the log.

### Stage R1 — detection bootstrap (det + head-pose only)

```bash
python3 src/training/train.py --preset recovery_det_only \
  --subset-ratio 0.25 --max-epochs 3 --seed 42
```
- New preset (this branch): `train_act=False, train_psr=False`, FP32,
  effective batch 8, EMA/mixup off, det_conf irrelevant (activity off).
- **Gate: `det_mAP50 ≥ 0.05`** and DET_PROBE showing `bestIoU>0.5` matches.
  Your GIoU is already 0.8–0.9 on positives — localization works; once cls
  learns to *fire at the right places* (now that 97% of its gradient isn't
  "fire nowhere"), mAP must leave zero. If after 3 epochs scores are still
  constant (std < 0.01) **with committed steps confirmed**, then and only
  then: det-head LR ×3 (one knob), 2 more epochs.

### Stage R2 — joint recovery

```bash
python3 src/training/train.py --preset recovery \
  --resume <R1 best.pth> --max-epochs <+4> --subset-ratio 0.25 --seed 42
```
- Preset now: all heads on, det_conf live (sigmoid-bounded), FP32, eff.
  batch 8.
- Gates after 4 epochs: activity predicts ≥ 4 classes (watch the
  `pred_seen` debug line), PSR ≥ 3 unique patterns, det mAP not regressing
  by more than ~30% from R1.
- If activity is still 1-class *with healthy det_conf and committed steps*:
  the next single knob is LDAM → plain CE + label-smoothing 0.1 (one
  config flag, `USE_LDAM_DRW=False` — CB-Focal path) — s=30 margin scaling
  is the remaining suspect there, not the architecture.

### Stage R3 — scale

Subset 0.25 → 1.0, epochs per the original schedule, re-enable EMA
(decay 0.999) once val metrics move monotonically, keep mixup off until the
implementation mixes images. PSR transition-objective and the
ROI/video-stream work you've begun (`roi_detector.py`, `video_stream.py`,
`psr_transition.py`) belong here, not earlier.

**When to re-enable things:** EMA — Stage R3. Mixup — never, until rewritten.
ZERO_DET_CONF — never again (delete after R2 proves stable). Staged-training
schedule — superseded by this protocol.

---

## 4. Code changes applied on this branch

1. **`src/training/losses.py`** (FocalLoss 4-arg mode): images with
   `gt_boxes.shape[0]==0` are skipped; normalization changed from `/B` to
   `/max(n_img_with_gt, 1)` so the det gradient isn't shrunk by the
   empty-frame fraction. Full mechanism documented in-line.
2. **`src/training/train.py`**: RC-29 step-commit telemetry at both
   optimizer-step sites (main + PSR-seq paths). Detection idiom:
   `scaler.update()` reduces the scale iff the step was skipped. Logs the
   1st/10th/50th/every-200th skip, plus a per-epoch summary
   (`optimizer windows / committed / skipped / scale`) and an ERROR when an
   epoch commits zero steps. Inert under FP32.
3. **`src/config.py`**: `recovery` preset — `zero_det_conf: False`,
   `grad_accum_steps: 8`, hardened `mixed_precision: False` comment naming
   RC-29 and the hand-edit that caused it; explicit `train_*: True` keys.
   New **`recovery_det_only`** preset (det + head_pose only). `apply_preset`
   now handles `train_det/act/psr/head_pose` (cached into `CFG_TRAIN_*` via
   the existing `_refresh_runtime_cfg()` call and passed into
   `MultiTaskLoss`/model construction).

Verified: py_compile clean; `apply_preset('recovery_det_only')` →
`TRAIN_ACT=False, TRAIN_PSR=False, MP=False, EFF_BATCH=8`; presets are
self-contained across consecutive applications.

---

## 5. The discipline point (gently, but it matters)

The June 11 audit prescribed `mixed_precision: False` with the R2 evidence
cited in the comment; the run was launched with it hand-flipped to `True`
"[TEMP: AMP for 12GB VRAM]", and 8 GPU-hours evaporated into a frozen model —
while the audited FP32-batch-1 configuration had already been proven to fit
on this GPU. The same pattern lost round 3 (the tree that silently reverted
FIX-4/P5/P7). Rule going forward: **any deviation from a prescribed config
gets its own 200-step smoke with the telemetry line checked before a
multi-hour run.** The new `[RC-29] optimizer windows` epoch line exists
precisely so the next deviation costs 20 minutes instead of a day.
