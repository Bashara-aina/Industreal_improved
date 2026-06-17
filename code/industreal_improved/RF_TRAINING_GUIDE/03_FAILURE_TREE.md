# Failure Decision Tree — Symptom → Cause → Exact Action

*When something looks wrong, find the symptom here. Each entry: how to confirm it, the real
cause, the one action, and how to verify the fix. Do one thing at a time.*

> **First, the meta-rule.** Reduce LR **only** for instability (loss spikes, NaN). For
> "won't learn / won't fire / trunk starved", reducing LR makes it *worse*. The auto-retry
> ladder cuts LR — so for learning-failures you must override it (see `§Doom-loop`).

---

## §Background-equilibrium — "localizes but won't fire" (RF1's classic death)

**Confirm:**
```bash
grep -E "backbone:|DET_HEALTH|DET_PROBE" src/runs/rf_stages/logs/train.log | tail
# backbone:STARVED  +  cls_mean pinned ~-4.7  +  preds>0.30=0  +  bestIoU>0.5 high
```
**Cause:** the shared trunk gets no feature-shaping gradient, so the classifier can't
separate fg/bg. In RF1 this is caused by `--detach-reg-fpn` (regression gradient blocked
from the backbone) combined with reinit + all other heads off.

**Action (in priority order):**
1. **Ensure reg gradient reaches the trunk.** RF1 must launch **without** `--detach-reg-fpn`.
   The fix sets `'detach_reg_fpn': False` on the RF1 stage in `stage_manager.py`. Confirm:
   ```bash
   grep -- "--detach-reg-fpn" src/runs/rf_stages/logs/subprocess.log   # should print NOTHING for RF1
   ```
2. **Keep a dense aux head on.** RF1 now trains `head_pose` (`train_head_pose: True`) — a
   feature-diverse trunk signal. Leave it on.
3. **Do NOT reduce LR.** If anything keep `BASE_LR=5e-4` (default strategy).

**Verify fixed:** within ~200 steps `backbone:ALIVE`, `cls_mean` rising, `score_p50>0.02`,
`preds>0.30` non-zero by epoch 1.

---

## §Reg-shock — `cls_mean` crashes toward −16

**Confirm:** `DET_HEALTH cls_mean` drops fast (−2 → −16) within a few hundred steps after
reinit, classification collapses.

**Cause:** the freshly-reinitialized **regression** head emits garbage boxes on first GT,
producing a large GIoU gradient that shocks shared FPN features.

**Action:** this is what `REINIT_REG_WARMUP` is for — it ramps reg-loss `1% → 100%` over
`REINIT_REG_WARMUP_STEPS` (default 1000). If shock still appears with reg attached:
1. **Lengthen the warmup**, don't re-detach: `REINIT_REG_WARMUP_STEPS = 2000` (config.py).
2. Optionally freeze the backbone for the first ~300 steps (head-only warmup), then
   unfreeze with reg attached.

**Do NOT** "fix" this by setting `detach_reg_fpn=True` permanently — that re-creates
`§Background-equilibrium`. Detach is a transient shock guard at most, never a whole-run
setting.

---

## §Det-dead — `detection_head:DEAD` on frame-batches

**Confirm:** `LIVENESS detection_head:DEAD` on **frame** (not seq) batches, det loss ~0.

**Cause & action:**
- **Seq-batch confusion:** DEAD on `[SEQ-BATCH]` is normal (det loss skipped). Only worry
  about **frame**-batches. For a detection-only stage, half the steps being seq-batches is
  wasted compute — consider raising `PSR_SEQ_EVERY_N_BATCHES` (config.py) when `train_psr`
  is False so detection trains every step.
- **No GT in batch:** if frame-batches have no boxes, `DET_GT_FRAME_FRACTION` is mis-set.
  It should resolve to **0.9** for det-only stages, **0.4** with activity/PSR. Confirm:
  ```bash
  grep "DET_GT_FRAME_FRACTION" src/runs/rf_stages/logs/train.log | head
  ```
- **Loss flag off:** confirm `train_det=True` resolved for the preset.

---

## §Frozen — constant outputs / identical eval cycles

**Confirm:** `[RC-29] committed=0` or eval metrics identical to 4 decimals across epochs.

**Cause:** `mixed_precision=True` somewhere → AMP GradScaler silently skips
`optimizer.step()` on inf/NaN grads. **This is a config bug, not a model bug.**

**Action:** set `mixed_precision: False` in the active preset (it already is in all RF
presets — check nobody hand-edited it on the box). Re-launch. `skipped` must be 0.

---

## §NaN — `nan`/`inf` in a loss

**Confirm:**
```bash
grep -iE "nan|inf|not finite|fallback" src/runs/rf_stages/logs/train.log | tail
```
The loss has per-head NaN guards that substitute `1e-4` and log it — find **which** head.

**Cause & action by head:**
- **head_pose:** usually a scale/normalization issue. Geo head uses 6D→orthonormal rotation;
  ensure `use_geo_head_pose=True`. Direction term is L2-normalized (scale-invariant).
- **PSR:** historically `std(correction=0)` NaN; guarded now. Keep
  `psr_sensitivity_weight=0.01` (not 0.0 with the old code path).
- **detection/GIoU:** degenerate zero-area boxes — guarded by box clamping + `isfinite`
  check; if it still fires, inspect the offending GT boxes.
- **activity/LDAM:** `s=30` overflow — keep `USE_LDAM_DRW=False` for early joint stages.

A single NaN batch is auto-recovered; **repeated** NaN on one head = real bug → isolate by
disabling that head's training flag, confirm the rest is stable, then fix that head alone.

---

## §OOM — CUDA out of memory

**Confirm:** `CUDA out of memory` traceback; or `gpu_mem` climbing toward 12 GB.

**Cause & action (cheapest first):**
1. **Other processes hold VRAM** (this killed RF1 retry #0). `nvidia-smi`; kill strays.
2. **Frame cache growth** at higher data % (RF6–RF10). `CLEAR_FRAME_CACHE_EPOCH_END=True`
   frees ~5–7 GB between epochs — confirm it's on.
3. **Lower micro-batch, keep effective batch:** `batch_size 4→2`, `grad_accum 8→16`
   (effective batch stays 32). Proven safe (R2.5 ran batch=2).
4. Last resort: turn off VideoMAE / reduce sequence length for the seq path.

---

## §Activity-collapse — predicts 1–2 classes (top-1 looks OK but is fake)

**Confirm:** activity `top1` ~= majority-class frequency; val predictions span <5 classes.

**Cause:** 75-class long-tail; LDAM `s=30` over-amplifies → 1-class collapse; or activity
loss dominates the backbone and then collapses.

**Action (already defaulted, verify them):**
- `USE_LDAM_DRW=False` (plain CE + label smoothing for early joint stages).
- `ACTIVITY_LOSS_WEIGHT=0.2`, `ACTIVITY_HEAD_GRAD_CLIP=0.1` (keep activity from dominating).
- `KENDALL_LOG_VAR_MIN_ACT=-0.5` (cap activity precision boost).
- Confirm the activity sampler is class-balanced and the subset actually contains tail
  classes (`§Gate-NaN` cousin: missing labels look like collapse).

**Verify fixed:** val predictions span many classes; top-1 rises above the majority baseline.

---

## §PSR-dead — PSR head DEAD or saturated (RF4+)

**Confirm:** `psr_heads:[h0..h10]` all DEAD even on `[SEQ-BATCH]` steps; `psr_f1` flat.

**Cause:** PSR is sparse (11 binary states, transition-only signal) and dies quietly when
its gradient is starved or saturated.

**Action:**
- Confirm PSR actually gets sequence batches: `PSR_SEQ_EVERY_N_BATCHES=2` means every other
  batch. The `psr_bias_gradient_check` health gate (RF4) verifies bias grad ≠ 0.
- `PSR_WARMUP_INIT_MULT=2.0` gives PSR a head start; `PSR_FOCAL_GAMMA=1.0` (gentler) keeps
  gradient on hard examples.
- Keep `use_psr_transition=True`, `use_psr_order_prior=True` for RF4+.
- If still dead: `STAGE3_WARMUP_EPOCHS=3` ramps the psr_head LR at entry — make sure you're
  not double-ramping (loss-side `PSR_WARMUP_EPOCHS` is disabled on purpose).

---

## §Head-domination — one head starves the others (Kendall imbalance)

**Confirm:** one `log_var` pinned at the clamp floor, another at the ceiling; one metric
races while others stall.

**Action:** the per-task clamps exist for this (`KENDALL_LOG_VAR_*`). If one head still
dominates, lower its loss weight / grad clip (e.g. activity knobs above) rather than
boosting the others. Re-balance one head per relaunch.

---

## §Gate-NaN — gate metric is NaN / missing → infinite retries

**Confirm:** validation log shows `det_mAP50=NaN` (or the metric absent), yet training looks
healthy. The orchestrator keeps retrying with lower LR.

**Cause:** the **evaluator** isn't producing the metric — eval skipped, zero predictions
above threshold, or an eval-side crash. **The model may be fine.**

**Action:**
1. Force an eval on the latest checkpoint and read the real number:
   ```bash
   python -m src.evaluation.evaluate --checkpoint <latest.pth> --preset <stage_preset>
   ```
2. If predictions exist but mAP is NaN → eval-side bug (array length mismatch / empty
   detections). Fix the evaluator, not the model.
3. Until the metric is a real number, **do not let the LR-cutting retry run** (`§Doom-loop`).

---

## §Gate-too-strict — healthy, converged, plateaus just under a gate

**Confirm:** model healthy (all heads alive, no spikes), metric improved then plateaued
within a small margin of the gate for `patience_epochs`.

**This is not a death spiral.** The gate is aspirational (file `01` reality check). Decide:
- **Accept and advance** if the miss is small and other heads are strong (edit the stage's
  gate threshold to the achieved-plus-margin, or advance manually). A multi-task model
  within a defensible margin of single-task baselines is a legitimate paper result.
- **Or buy accuracy with the next data-scaling stage** rather than grinding this one.

Do **not** trigger reinit/LR-cut retries to chase the last 0.02 of a plateaued, healthy
model — you'll trade a small miss for a collapse.

---

## §Doom-loop — auto-retries keep cutting LR and re-initing

**Confirm:** `retry_count` rising; each launch uses a smaller `lr_mult`
(0.2 → 0.1 → 0.05); the failure is a *learning* failure, not instability.

**Cause:** `RETRY_STRATEGIES` only reduce LR + reinit — correct for instability, wrong for
starvation/equilibrium/measurement failures.

**Action — operator override:** stop the orchestrator, fix the **root** per the relevant
section above, then relaunch the single stage at **base LR**:
```bash
kill <PID>
# reset the stage state: retry_count=0, status="pending"; clear stale stage checkpoints
# launch fresh at default strategy (lr_mult=1.0)
python3 -m src.training.stage_manager --start
```
(If you want the orchestrator to never cut LR for the bootstrap stage, that's the
`RETRY_STRATEGIES` change recommended in the consult — ask before applying, it affects all
10 stages.)

---

## §Overfit — val diverges from train (RF9/RF10)

**Confirm:** train loss falling while val metrics flatten/regress over the last epochs.

**Action:** prefer **early-stop on val** (best.pth is already val-selected); reduce epochs
for the final stage; rely on EMA weights (smoother). Confirm you **evaluate the EMA
checkpoint**, not the raw one (`04 §EMA`). More data (the point of RF6–RF10) is the
structural fix; don't over-train a fixed model on a fixed subset.

---

## The 30-second triage flow

```
Is there a traceback?              → §OOM / §NaN / §Crash
RC-29 committed=0?                 → §Frozen
backbone:STARVED?                 → §Background-equilibrium
detection_head DEAD on frame?      → §Det-dead
cls_mean → -16?                   → §Reg-shock
a required head DEAD?              → §Head-dead / §Activity-collapse / §PSR-dead
gate metric = NaN?                → §Gate-NaN
healthy but under gate?            → §Gate-too-strict
retries cutting LR?               → §Doom-loop
none of the above, metrics rising  → you're fine, keep monitoring (00 §4)
```
