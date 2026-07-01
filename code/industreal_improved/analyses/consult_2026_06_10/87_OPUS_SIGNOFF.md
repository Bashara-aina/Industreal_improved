# 87 — Opus Pre-Flight Sign-Off (files 83–86)

Date: 2026-07-01 · Model: Opus 4.8 · Method: verified the 3 NEW "critical fixes"
(commit `2e69b1e`) and the sign-off claims against `main` source — not the summaries.
Every load-bearing item was traced to code.

## Sign-off table

| Question | Verdict | Basis |
|----------|:------:|-------|
| Q1 — Detection pipeline correct & collapse-free? | **YES** | 4 mechanisms wired (OHEM, asym-γ, GT-frac, empty-frame) + reg-warmup + detach_reg_fpn=False in presets — all confirmed earlier + file 84 refs check out. |
| Q2 — Activity pipeline correct end-to-end? | **YES** | Remap at 4 sites, no double-remap, balanced sampler, simple MLP, CE+CB weights, segment-eval remap (my Q5), sampling log (my Q1) — all present. |
| Q3 — PSR pipeline correct (per-frame caveat)? | **YES** | Focal γ=0.5 active, per-comp α + comp_weights bounded, warmups, DETACH_PSR_FPN, `psr_comp_acc` wired (`evaluate.py:3761`). Per-frame framing caveat holds. |
| Q4 — Multi-task Kendall + scheduler/WD correct? | **YES** *(2 notes)* | Kendall bounds/cap/NaN-guards verified; scheduler & bias-WD fixes verified sound — see notes below. |
| Q5 — Go/no-go metrics logged? | **YES** | `det_gt_fraction` (`train.py:1469`), `psr_comp_acc` (`evaluate.py:3761-3763`), diversity, grad-norm all present. |

**All 5 = YES → cleared to launch the 50-step probe, then RF4.** Two precise notes on
Q4 (neither blocking) and one honesty caveat below.

## Verification of the 3 NEW critical fixes (commit 2e69b1e)

**Fix 1 — OneCycleLR `steps_per_epoch=1`: CORRECT, won't crash.**
- Verified `scheduler.step()` is **epoch-level** (`train.py:4309`, inside the epoch loop,
  guarded by the `_train_failed` continue) — not per-batch. So `total_steps=C.EPOCHS`
  matches the call cadence; the "stuck-in-warmup for the whole run" P0 bug is genuinely
  fixed.
- `max_lr` is an explicit **per-group list** (8 entries +1 for loss, `train.py:3635-3646`)
  matching the param-group count — no length-mismatch crash. (The log string
  `max_lr=[5e-5,5e-4]` is cosmetic; ignore it.)
- OneCycleLR is stepped only after the SequentialLR milestone, i.e. `C.EPOCHS − W` times
  (< `total_steps`) — no step-count overflow.

  **Note 4a (not blocking):** the LinearLR warmup **and** OneCycleLR's own `pct_start=0.1`
  warmup are **redundant**. Net effect: LR ramps for epochs 0–1 (LinearLR), then *dips*
  at epoch 2 to `max_lr/25` when OneCycleLR takes over, then re-rises to peak by ~epoch
  12, then decays. Functional, but a double-warmup dip and a slightly truncated final
  decay. If the probe ever throws `Tried to step N times…`, the robust one-line fix is to
  drop the `SequentialLR`/`LinearLR` and use **OneCycleLR alone** (it already warms up via
  `pct_start`). Optional; only if the probe complains. Confirm at runtime (file 83's
  checklist): LR ≈ base/25 at epoch 2, ≈ peak by epoch ~12, decaying after.

**Fix 2 — bias WD=0: CORRECT partition, but the "norm weights" claim is inaccurate.**
- The param categorization (`train.py:3503-3534`) is a clean if/elif/else over
  `named_parameters()` — every trainable param lands in **exactly one** group (none
  dropped, none duplicated; AdamW would have raised on a duplicate). Bias groups carry
  `weight_decay: 0.0`. Good.
- **Correction:** the exclusion filter is `'bias' in name`, so it excludes **biases only**.
  **LayerNorm/BatchNorm *weights* are NOT excluded** — they have no 'bias' in their name,
  so they still receive `WD=1e-3`. File 83's claim "LayerNorm weights are no longer decayed
  toward zero" is wrong. Impact is small (WD=1e-3 is mild), so it's not blocking — but if
  you want true best practice, change the filter to `param.ndim <= 1` (excludes biases AND
  all 1-D norm weights). Optional.

**Fix 3 — metrics: PRESENT and wired.** `det_gt_fraction` logged in `[DET-HEALTH]`
(`train.py:1469`); `psr_comp_acc` computed into `results` and logged
(`evaluate.py:3761-3763`). Confirmed.

## Carry-over corrections still standing (don't lose these)
- **Round-4 fix #6 (NEG_SLOPE 0→0.01) is a verified no-op** — `reg_loss = 1−GIoU ∈ [0,2]`,
  so `loss_det ≥ 0` always and the floor never fires. File 84 already documents this as
  dead code — good. Don't attribute any behavior to it.
- **PSR is per-frame component recognition, not transition detection** — the paper wording
  must match (Q3 caveat).
- **Don't report `act_accuracy_no_na`** (group 0 = reserved 'other').
- **`WARMUP_EPOCHS=2` left as-is** — the "detection at full strength only 0.9 epochs" premise
  was overstated; OneCycleLR keeps LR high for ~10 epochs post-peak.

## The one honesty caveat on the 85% confidence
"Correct and collapse-free **wiring**" is what I've verified — and it's true. That is not
the same as "will hit the epoch-100 target ranges." The go/no-go gates in 77–85 are
sound; the *target ranges* (det 0.50–0.65, act 0.40–0.60, PSR 0.75–0.85) are optimistic
ceilings, not predictions. The 85%→95% jump after the 50-step probe is the right framing
**for "won't crash / is correctly wired."** Keep a separate, lower expectation for hitting
the numeric targets, and let the epoch-2 / epoch-5 gates (file 85) tell you the truth.

## Verdict
**GO.** Launch `--preset stage_rf4 --reinit-heads`, watch the step-50 probe signals, and
at epoch 0 read the two new logs (`det_gt_fraction` ≈ 0.35–0.45; per-class sampling
max/min ratio < ~10×). Nothing I found blocks the run. The two Q4 notes are optional
polish, safe to apply after a successful probe rather than before it — do **not** re-touch
the scheduler right before launch unless the probe actually errors.
