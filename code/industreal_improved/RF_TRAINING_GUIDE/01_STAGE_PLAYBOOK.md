# RF1 → RF10 Stage Playbook

*One card per stage. Open the card before you launch. All numbers are read directly from
`stage_manager.py` (`RF_STAGES`) and `config.py` (presets) — they are your real gates.*

---

## How to read a card

- **GATE** = all must be met to advance (`metric ≥ target`, except MAE which is `≤`). If
  any gate metric is missing/NaN, the gate is treated as FAIL → see `03 §Gate-NaN`.
- **Realistic note** = honest expectation on a single RTX 3060, multi-task ConvNeXt-T.
- **#1 risk** = the failure this stage is most likely to hit; pre-read its entry in `03`.
- All RF stages: `batch_size=4 × grad_accum=8` → **effective batch 32**, `BASE_LR=5e-4`,
  FP32, EMA on.

> **Reality check on the gates.** The gates ramp toward paper baselines
> (`det_mAP50≈0.838` YOLOv8m, `act_top1≈0.6525` MViTv2, `psr_f1≈0.731`). Those baselines
> are *single-task, dedicated* models. Hitting them with one 12 GB multi-task model is
> ambitious. Treat the gates as **direction and discipline**, not as guarantees of nature.
> If a stage is healthy and improving but plateaus just under a gate, that is a
> **scope/tuning decision** (file `03 §Gate-too-strict`), not a death spiral. Do not let a
> slightly-too-high gate trigger the LR-cutting retry loop.

---

## RF1 — Detection bootstrap

| Field | Value |
|------|------|
| Preset | `stage_rf1` |
| Data | 20% · **20 epochs** |
| Heads training | **detection** (+ **head_pose** as trunk aid, after the fix) |
| Resume from | previous `latest` (or fresh ImageNet init) · **`--reinit-heads` ON** |
| Critical flag | **`--detach-reg-fpn` must be OFF** (the fix). Verify in subprocess.log |
| **GATE** | `det_mAP50 ≥ 0.30`, `det_mAP50_95 ≥ 0.12` |
| Health | `max_consecutive_dead=5`, `loss_spike<10×`, `liveness≥0.7` |

- **Realistic note:** This is the stage that has been dying. With the fix, the success
  signal in the first epoch is simple: **`cls_mean` climbs off −4.7 and `backbone:ALIVE`**.
  mAP@50 of 0.30 from a reinit head in 20 epochs at 20% data is reachable but not
  guaranteed; if healthy-but-short, give it the full 20 epochs before judging.
- **What's special:** the only stage with `--reinit-heads`. The detection head starts from
  random + `pi=0.01` bias. Everything downstream inherits this checkpoint, so do not
  advance on a half-alive detector.
- **#1 risk:** background equilibrium ("localizes, won't fire"). → `03 §Background-equilibrium`.
- **Confirm green (epoch 1):** `backbone:ALIVE`, `cls_mean > -3.5`, `DET_PROBE preds>0.30`
  becomes non-zero, `det_mAP50` is a real number (not NaN).

---

## RF2 — Add body + head pose

| Field | Value |
|------|------|
| Preset | `stage_rf2` |
| Data | 35% · **15 epochs** |
| Heads training | detection + pose (body + head) |
| Resume from | RF1 **`best`** |
| **GATE** | `det_mAP50 ≥ 0.40`, `det_mAP50_95 ≥ 0.18`, `forward_angular_MAE_deg ≤ 60` |
| Health | `liveness≥0.7`, pose grad-norm `≥1e-6` |

- **Realistic note:** pose is a "healthy" head — it usually trains cleanly and *helps*
  detection by enriching the backbone. Expect detection mAP to rise vs RF1, not drop.
- **What's NEW:** pose head joins. Watch that adding it does not destabilize detection in
  the first 200 steps (it shouldn't — pose gradient is dense and smooth).
- **#1 risk:** head-pose loss scale wrong → NaN or domination. → `03 §NaN`, `03 §Head-domination`.
- **Confirm green:** `pose:ALIVE` and `head_pose:ALIVE`, detection mAP not regressing.

---

## RF3 — Add activity

| Field | Value |
|------|------|
| Preset | `stage_rf3` |
| Data | 35% · **15 epochs** |
| Heads training | detection + pose + **activity** (PSR still OFF) |
| Resume from | RF2 **`best`** |
| **GATE** | `det_mAP50 ≥ 0.45`, `det_mAP50_95 ≥ 0.20`, `act_top1 ≥ 0.40`, `fwd_MAE ≤ 55` |
| Health | `liveness≥0.65`, act grad-norm `≥1e-6` |

- **Realistic note:** activity is the **most collapse-prone** head (75 classes, long-tail).
  `USE_LDAM_DRW=False` by default (plain CE + label smoothing) precisely because LDAM s=30
  caused 1-class collapse. `ACTIVITY_LOSS_WEIGHT=0.2` and `ACTIVITY_HEAD_GRAD_CLIP=0.1`
  keep it from dominating the backbone.
- **What's NEW:** activity head. This is the first stage where multi-task balance matters.
- **#1 risk:** activity collapses to 1–2 classes (top-1 looks "OK" but it predicts the
  majority class always). → `03 §Activity-collapse`.
- **Confirm green:** `act:ALIVE`, activity predicts **>5 distinct classes** on val, top-1
  rising past the majority-class baseline.

---

## RF4 — Add PSR (all heads, transition enabled)

| Field | Value |
|------|------|
| Preset | `stage_rf4` |
| Data | 50% · **20 epochs** |
| Heads training | **all 5** (PSR transition ON, order prior ON, sensitivity 0.01) |
| Resume from | RF3 **`best`** |
| **GATE** | `det_mAP50 ≥ 0.50`, `act_top1 ≥ 0.45`, `psr_f1_at_t ≥ 0.25`, `fwd_MAE ≤ 50` |
| Health | `liveness≥0.6`, **`psr_bias_gradient_check=True`** (PSR bias must get grad) |

- **Realistic note:** this is the hardest integration stage (longest at 20 epochs for a
  reason). PSR trains in **sequence mode** (`PSR_SEQ_EVERY_N_BATCHES=2`), so on alternating
  batches detection gets no gradient — that is expected here, not a bug.
- **What's NEW:** PSR head + transition decoder. PSR is sparse and dies quietly.
- **#1 risk:** PSR head goes DEAD / saturates (per-component bias grad → 0). → `03 §PSR-dead`.
- **Confirm green:** the `psr_heads:[h0..h10]` liveness line shows multiple components
  ALIVE on **[SEQ-BATCH]** steps; `psr_f1_at_t` is a real number.

---

## RF5 — Consolidate all heads

| Field | Value |
|------|------|
| Preset | `stage_rf5` |
| Data | 50% · **10 epochs** |
| Heads training | all 5 |
| Resume from | RF4 **`best`** |
| **GATE** | `det_mAP50 ≥ 0.55`, `act_top1 ≥ 0.50`, `psr_f1_at_t ≥ 0.30`, `fwd_MAE ≤ 45` |
| Health | `loss_spike<8×`, `liveness≥0.65` |

- **Realistic note:** no new heads, same data — this stage just lets the 5-way balance
  settle. If RF4 was healthy, RF5 is the calmest stage on the ladder.
- **#1 risk:** one head slowly starves another (Kendall imbalance). → `03 §Head-domination`.
- **Confirm green:** all five heads ALIVE across the epoch; every gate metric trending up.

---

## RF6–RF9 — Scale the data

| Stage | Preset | Data | Ep | GATE (`det_mAP50` / `act_top1` / `psr_f1` / `fwd_MAE≤`) |
|-------|--------|------|----|--------------------------------------------------------|
| RF6 | `stage_rf6` | 65% | 10 | 0.58 / 0.52 / 0.35 / 42 |
| RF7 | `stage_rf7` | 65% | 10 | 0.62 / 0.55 / 0.40 / 40 |
| RF8 | `stage_rf8` | 80% | 10 | 0.65 / 0.58 / 0.45 / 38 |
| RF9 | `stage_rf9` | 90% | 10 | 0.70 / 0.60 / 0.50 / 35 |

- All four: all 5 heads, `max_consecutive_dead=3` (stricter), `liveness≥0.7`.
- **Realistic note:** these stages buy accuracy with **data**, not architecture. The model
  is fixed; you're feeding it more. Gains taper — expect diminishing returns per stage.
- **What's NEW:** more data per stage → epochs are longer in wall-clock even at 10 epochs.
  Re-check VRAM at each step-up (more unique frames → bigger frame cache).
- **#1 risk:** OOM as the frame cache grows (→ `03 §OOM`); and the stricter
  `max_consecutive_dead=3` means a 3-epoch dead streak now kills the run faster.
- **Confirm green:** metrics keep inching toward target; VRAM stable; no new DEAD heads.

---

## RF10 — Final full-data push (paper results)

| Field | Value |
|------|------|
| Preset | `stage_rf10` |
| Data | **100%** · **15 epochs** |
| Heads training | all 5 |
| Resume from | RF9 **`best`** |
| **GATE** | `det_mAP50 ≥ 0.75`, `det_mAP50_95 ≥ 0.35`, `act_top1 ≥ 0.63`, `psr_f1_at_t ≥ 0.55`, `fwd_MAE ≤ 30` |
| Health | `loss_spike<8×`, `liveness≥0.75` (strictest) |

- **Realistic note:** these are near the single-task paper baselines. Reaching all five
  simultaneously on one multi-task 12 GB model is the **stretch goal**, not a given. The
  honest success criterion for a paper is: *healthy, converged, and within a defensible
  margin of the gates* — partial misses on one head with strong others are a legitimate,
  reportable result, not a failed run.
- **#1 risk:** over-fitting on full data in 15 epochs (val diverging from train) and final
  EMA-vs-raw checkpoint confusion. → `03 §Overfit`, `04 §EMA`.
- **Confirm green:** best.pth selected on **val**, EMA weights are the ones you evaluate,
  metrics stable across the last 3 epochs.

---

## The whole ladder at a glance

| Stage | Data | Ep | New head | Gate det_mAP50 | The one thing to watch |
|-------|------|----|----------|----------------|------------------------|
| RF1 | 20% | 20 | det (reinit) | 0.30 | `backbone:ALIVE` + cls_mean climbs |
| RF2 | 35% | 15 | + pose | 0.40 | det doesn't regress when pose joins |
| RF3 | 35% | 15 | + activity | 0.45 | activity predicts >5 classes |
| RF4 | 50% | 20 | + PSR | 0.50 | PSR bias grad ≠ 0 on seq-batches |
| RF5 | 50% | 10 | (settle) | 0.55 | all 5 ALIVE together |
| RF6 | 65% | 10 | (scale) | 0.58 | VRAM stable |
| RF7 | 65% | 10 | (scale) | 0.62 | gains still positive |
| RF8 | 80% | 10 | (scale) | 0.65 | no DEAD streak (limit now 3) |
| RF9 | 90% | 10 | (scale) | 0.70 | overfit watch |
| RF10 | 100% | 15 | (final) | 0.75 | EMA checkpoint = the one you report |

Now keep `02_HEALTH_SIGNALS.md` open in a second pane while the run trains.
