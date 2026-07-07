# Health Signals — Decode Every Log Line

*Keep this open while training. Each diagnostic, what it means, and the exact
healthy-vs-sick numbers. Strings match `train.py` so you can `grep` them live.*

```bash
tail -f src/runs/rf_stages/logs/train.log | grep -E \
  "DET-INIT|DET_HEALTH|DET_PROBE|LIVENESS|RC-29|backbone:|fpn:|log_var"
```

---

## The 200-step GREEN BOARD (memorize this)

If all of these read green at step ~200, the stage will almost certainly reach its gate.
If any is red, **kill and go to `03_FAILURE_TREE.md`** — waiting will not help.

| Signal | GREEN (healthy) | RED (kill it) |
|--------|-----------------|---------------|
| `[RC-29] ... committed` | `committed > 0`, `skipped = 0` | `committed = 0` or skips climbing |
| `backbone:` (new probe) | `backbone:ALIVE[…]` | `backbone:STARVED[…]` |
| `detection_head:` | `ALIVE[…]` on frame-batches | `DEAD` on frame-batches |
| `DET_HEALTH cls_mean` | **moving** off −4.7 toward −3…−2 | pinned at −4.7 for 200+ steps |
| heads you train | each `ALIVE` (>1e-6) | any required head `DEAD` |
| any loss | finite, no `nan/inf` | `nan`/`inf` printed |
| `gpu_mem` | stable, < ~11 GB | climbing toward 12 GB |

---

## 1. `[DET-INIT]` — step-0 sanity (printed once)

```
[DET-INIT] cls_preds: sum=... mean=...   → cls_mean should be ≈ -4.6  (pi=0.01 bias)
[DET-INIT] cls_preds max < 0.01 -- detection head stuck near zero      ← WARNING
```

- **Healthy:** `cls_mean ≈ -4.6` (`ln(0.01/0.99)`). This is the *correct* starting point,
  not a problem — it prevents loss explosion from background at init.
- **Sick:** the WARNING fires AND `cls_mean` never moves over the next 200 steps → the head
  is not learning to fire (background equilibrium). The job of training is to move
  `cls_mean` **up** from here.

---

## 2. `[RC-29]` optimizer windows — did steps actually commit?

```
Epoch N: optimizer windows=W  committed=C  skipped=S
[RC-29] Epoch N: ZERO optimizer steps committed — the model ...        ← FATAL
```

- **Healthy (FP32):** `skipped=0`, `committed=W`. Under FP32 this is always inert — that's
  the point of FP32.
- **Sick:** any `skipped>0` means someone flipped `mixed_precision=True` and the AMP
  GradScaler is silently dropping steps. `committed=0` = the model is **frozen** (this was
  the original "constant outputs" bug). → `03 §Frozen`. **Fix the config, do not debug the
  model.**

---

## 3. `LIVENESS_GRAD` — per-head AND trunk gradient (the most important line)

```
[LIVENESS_GRAD step=1400] detection_head:ALIVE[5.38e+00]/ALIVE[…] | pose_head:… |
  head_pose_head:… | activity_head:… | psr_head:… |
  psr_heads:[h0=…,h1=…,…h10=…] [SEQ-BATCH] |
  backbone:ALIVE[2.7e-01|n=210] | fpn:ALIVE[8.0e-02|n=44] | gpu_mem=1.06GB/6.10GB
```

A head/trunk is **ALIVE iff grad-norm > 1e-6**.

| Token | Healthy | Sick → go to |
|-------|---------|--------------|
| `detection_head:ALIVE` | ALIVE on **frame**-batches | DEAD on frame-batches → `03 §Det-dead` |
| `backbone:ALIVE` | `>1e-4`, ideally `O(0.01–10)` | `backbone:STARVED` → `03 §Background-equilibrium` |
| `fpn:ALIVE` | `>1e-4` | `STARVED` → same as backbone |
| `pose_head` / `head_pose_head` | ALIVE every step | DEAD → `03 §Head-dead` |
| `activity_head` | ALIVE on frame-batches | DEAD → `03 §Activity-collapse` |
| `psr_head` / `psr_heads:[…]` | ALIVE on **[SEQ-BATCH]** | all DEAD on seq-batches → `03 §PSR-dead` |

**Read the batch type.** `detection_head` is *expected* DEAD on `[SEQ-BATCH]` steps (det
loss is skipped there) and ALIVE on frame-batches. PSR is the opposite: ALIVE on
seq-batches. Don't panic at a single DEAD — read which batch type it is. Judge over ~10
steps, not one.

> **`backbone:STARVED` while `detection_head:ALIVE` is THE signature of RF1's classic
> failure** — the head learns its own weights but the trunk's features never change, so
> detection can place boxes but never fires. This line is the one the previous 29 files
> never had.

---

## 4. `DET_HEALTH` — is the classifier differentiating?

```
DET_HEALTH cls_mean=-4.70  cls_std=0.88  near_zero=0.0%
```

| Field | Healthy trajectory | Sick |
|-------|--------------------|------|
| `cls_mean` | rises from −4.6 toward −3 … −2 over epoch 1 | pinned at −4.7 (won't fire) **or** plummeting to −10…−16 (over-suppression collapse) |
| `cls_std` | **widens** (>1.0) as the head differentiates classes | stays tight (~0.8) = no differentiation |
| `near_zero` | small, non-zero as some logits approach firing | stuck at 0.0% forever |

- Pinned `cls_mean=-4.7` + tight `cls_std` = background equilibrium → `03 §Background-equilibrium`.
- `cls_mean` crashing toward −16 = regression/gradient shock or over-strong background →
  `03 §Reg-shock`.

---

## 5. `DET_PROBE` — what the detector actually predicts on val

```
DET_PROBE  score_p50=0.0167  score_max=0.06  preds>0.05=… preds>0.30=0  bestIoU>0.5=900
verdict: LOCALIZING
```

| Field | Healthy (improving) | Sick (stuck) |
|-------|---------------------|--------------|
| `score_p50` | rises above 0.02 and keeps climbing | stuck at ~0.0167 (= `pi=0.01`) |
| `score_max` | climbs past 0.3 within a few epochs | capped at 0.06–0.10 |
| `preds>0.30` | becomes **non-zero**, grows | stays **0** = never fires |
| `bestIoU>0.5` | high (localization is usually fine) | high even when stuck (boxes ok, confidence not) |
| `verdict` | moves toward FIRING/DETECTING | stays `LOCALIZING` forever |

**Key trap:** high `bestIoU` + zero `preds>0.30` is the deceptive "looks like it's working"
state — it places boxes but assigns them ~1% confidence. That is a **failure**, not
progress. The number that matters is `preds>0.30` becoming non-zero and `score_p50` rising.

---

## 6. Kendall `log_var` sentinel — multi-task balance (RF3+)

```
log_var_det grad=…  log_var_pose grad=…  log_var_act grad=…  log_var_psr grad=…
```

- The Kendall uncertainty weights are clamped to **[-4, 2]** every forward pass.
- **Healthy:** all four log_vars stay inside the clamp and drift slowly. No single task's
  precision (`exp(-log_var)`) runs away.
- **Sick:** one task pinned at the floor while another is suppressed at the ceiling = a head
  is dominating the backbone. Config guards exist (`KENDALL_LOG_VAR_MIN_ACT=-0.5`,
  `KENDALL_LOG_VAR_MAX_PSR=0.0`, `KENDALL_LOG_VAR_MAX_POSE=3.0`). → `03 §Head-domination`.

---

## 7. Validation / gate metrics — must be NUMBERS

At epoch end the evaluator prints metrics that feed the gate:
`det_mAP50`, `det_mAP50_95`, `act_top1`, `psr_f1_at_t`, `forward_angular_MAE_deg`.

- **Healthy:** each gate metric for the current stage is a **finite number** and trending
  toward target (file `01`).
- **Sick:** `det_mAP50 = NaN` (or metric absent). The gate then reads `UNKNOWN → FAIL` and
  the orchestrator will retry-with-lower-LR forever. This is a **measurement** failure, not
  a model failure. → `03 §Gate-NaN`.

---

## 8. Quick triage grep cheatsheet

```bash
# Is the trunk learning?  (the truth signal)
grep -E "backbone:|fpn:" src/runs/rf_stages/logs/train.log | tail -20

# Is detection firing yet?
grep "DET_PROBE" src/runs/rf_stages/logs/train.log | tail -20

# Did any optimizer step get skipped? (should be none, FP32)
grep "RC-29" src/runs/rf_stages/logs/train.log | tail

# Any NaN/inf anywhere?
grep -iE "nan|inf|not finite" src/runs/rf_stages/logs/train.log | tail

# Which heads are alive right now?
grep "LIVENESS_GRAD" src/runs/rf_stages/logs/train.log | tail -10
```

If everything here is green, you are not in a death spiral — you are training. Go back to
`00 §4` and let the steady-monitor loop run.
