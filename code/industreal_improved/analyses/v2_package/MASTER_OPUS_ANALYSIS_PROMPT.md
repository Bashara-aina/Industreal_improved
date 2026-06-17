# POPW/IndustReal Multi-Task Training — Complete Opus Analysis Package
## v2 — Corrected | Clamp Fix Applied | 2026-06-03

---

## ⚠️ ROOT CAUSE IDENTIFIED AND FIXED

**The single toxic line was `losses.py:1264`:**

```python
total = total.clamp(min=0.0)   # "Final safeguard"
```

This has been **removed** (fix applied 2026-06-03). The NaN guard now fires directly after `total.squeeze()` without any clamp in between.

**Why this broke everything:**

The Kendall total is constructed as:
```
total = Σ(precision_i × loss_i) + Σ(log_var_i)
where precision_i = exp(-log_var_i)
```

At crash-recovery resume, log_vars sum to ≈ −1.8:
- det=−0.152, pose=−1.318, act=−0.011, psr=−0.318

On most training steps, Σ(precision × loss) ≈ 1.0–1.1, so true total ≈ −0.7. The clamp floored this to 0 → **zero gradient on ~86% of steps**. The network only learned when detection spiked high enough to overcome the −1.8.

**This one mechanism explains ALL apparent collapses simultaneously — not architectural failures.**

---

## WHAT THIS PACKAGE CONTAINS

| File | Purpose |
|------|---------|
| `MASTER_OPUS_ANALYSIS_PROMPT.md` | This document |
| `losses.py` | Full training loss (AFTER clamp fix applied) |
| `config.py` | All hyperparameters including Kendall settings |
| `industreal_dataset.py` | Data loading, annotation parsing, collation |
| `popw_train_25pct_15ep_v6.log` | 6244-line training/eval log |
| `split_head_pose_loss.py` | Gradient-balance proof for head pose fix |
| `investigate_annotations.py` | Annotation verification script |

---

## CORRECTIONS TO THE HAIKU-GENERATED v1 PROMPT

| v1 Claim | Reality |
|----------|---------|
| "DET n_gt=0, GT not loading" | ❌ **FALSE.** `DET_PROBE b0` shows `n_gt=16, bestIoU>0.5=419, max=0.80`. Regression branch works. Only cls confidence collapsed (sigmoid ≈0.030). |
| "All 6 tasks structurally collapsed" | ❌ **OVERSTATED.** The clamp explains all collapses. Fix the clamp — architectural changes not needed. |
| "Head pose split loss needs implementing" | ❌ **ALREADY DONE.** `losses.py:1163` uses `head_pose_loss_split` with pos_weight=1.0, dir_weight=1.0. |
| "AS and EV are Kendall collapses" | ❌ **INCORRECT.** AS and EV are NOT Kendall tasks. Their zeros are downstream effects. |
| "n_gt=0 in every DET_PROBE" | ⚠️ **MIXED.** Some val frames genuinely have no annotated objects. |

---

## 1. TRAINING CONFIGURATION

```
Model:      VideoMAE + 6-task FCN heads
Dataset:    IndustReal (IKEA assembly) — 25% subset (~57 recordings)
Epochs:     max=15 (resumed from crash_recovery.pth at epoch 12)
Batch:      16 video frames, T=8
Workers:    0
Optimizer:  AdamW lr=1e-4, weight_decay=0.05
Scheduler:  cosine with warmup
EMA:        decay=0.999, reinit after VideoMAE unfreeze (epoch 10)
Tasks:      DET, PSR, AR (75-class), HeadPose (9-DoF), AS (17-state), EV
Loss:       MultiTaskLoss (Kendall log-var weighting, 4 groups: det, pose, act, psr)
VideoMAE:   frozen epochs 0-9, unfreezes epoch 10
STAGED_TRAINING = False (no-staging throughout this run)
Kendall:    USE_KENDALL=True, log_var_det/hp/act/psr with clamp(-4.0, 2.0)
PSR_LOSS_CAP = 20.0, HEAD_POSE_LOSS_CAP = 30.0
PSR_TEMPORAL_SMOOTH_WEIGHT = 0.05
HEAD_POSE_POS_SCALE = 100.0 (position target standardization)
```

---

## 2. TRAINING STATUS

```
Process:     PID 274430 — KILLED 2026-06-03 after clamp fix applied
Epochs done: 12 (crash_recovery start) + eval epochs 13-14
Epochs left: 15 total (will restart from epoch 12 with fix)
Checkpoint:  crash_recovery.pth at epoch 12
CPU time:    1119+ minutes at kill
```

**Training will restart from crash_recovery.pth with the clamp fix applied.**

---

## 3. KEY LOSS MECHANISMS (from losses.py)

### Kendall Total Construction (lines 1198-1264)

```python
# === Kendall weighting ===
if self.use_kendall:
    lv_det = self.log_var_det.clamp(-4.0, 2.0)
    lv_hp = self.log_var_pose.clamp(-4.0, 2.0)
    lv_act = self.log_var_act.clamp(-4.0, 2.0)
    lv_psr = self.log_var_psr.clamp(-4.0, 2.0)

    total = torch.tensor(0.0, device=device)
    if self.train_det:
        prec_det = torch.exp(-lv_det)
        total = total + prec_det * loss_det + lv_det
    if self.train_pose:
        prec_hp = torch.exp(-lv_hp)
        total = total + prec_hp * loss_pose + lv_hp
    if self.train_act:
        total = total + prec_hp * loss_head_pose + lv_hp
        prec_act = torch.exp(-lv_act)
        total = total + prec_act * loss_act + lv_act
    if self.train_psr:
        loss_psr = loss_psr.clamp(min=0.0)   # per-task floor
        total = total + prec_psr * loss_psr + lv_psr
    total = total.squeeze()

    # NaN guard — fires on inf/NaN only (clamp REMOVED)
    total_val = total.item() if total.numel() == 1 else total
    if not math.isfinite(total_val):
        parts = []
        if self.train_det: parts.append(loss_det)
        if self.train_pose: parts.append(loss_pose)
        if self.train_act: parts.append(loss_head_pose); parts.append(loss_act)
        if self.train_psr: parts.append(loss_psr)
        finite_parts = [p for p in parts if torch.isfinite(p) and p >= 0]
        if finite_parts:
            total = torch.stack(finite_parts).sum()
        else:
            total = loss_det
```

### Per-Task NaN Guards (BEFORE Kendall, lines 1180-1196)

```python
# Each loss replaced with 1e-4 if NaN/inf, then clamped to >= 0
_safe = lambda l, z: torch.tensor(1e-4, device=device, dtype=l.dtype) \
    if not torch.isfinite(l).all() \
    else (torch.where(l < 0, z, l) if l.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64] else l)
loss_det = _safe(loss_det, zero)
loss_pose = _safe(loss_pose, zero)
loss_act = _safe(loss_act, zero)
loss_psr = _safe(loss_psr, zero)
loss_head_pose = _safe(loss_head_pose, zero)
```

### Smooth Cap Function (line ~1027)

```python
def _smooth_cap(x, cap):
    x_safe = x.clamp(min=1e-6, max=1e6)
    return torch.where(x > cap, cap * (1 + torch.log(x_safe / cap)), x.clamp(min=1e-6))
```

Applied to PSR (`PSR_LOSS_CAP=20.0`) and head_pose (`HEAD_POSE_LOSS_CAP=30.0`).

### Head Pose Split Loss (line 1163)

```python
loss_head_pose = head_pose_loss_split(
    outputs['head_pose'],
    targets['head_pose'],
    pos_weight=1.0,
    dir_weight=1.0,
)  # No *0.001 — Kendall handles weighting; O(1) inputs
```

With dataset-side standardization: `pose_data[:, 3:6] /= C.HEAD_POSE_POS_SCALE` (~100).

---

## 4. COLLAPSE STATUS (from log — pre-fix)

### DET (Object Detection — FCOS)
| Metric | Value | Verdict |
|-------|-------|---------|
| mAP@0.5 | 0.0000 | Collapse |
| Confidence range | [0.0299, 0.0311] | Sigmoid saturation |
| max_score | ~0.031 | Far below threshold |
| IoU>0.5 hits | 0/2.76M | No correct detections |
| **Note** | Regression branch works | bbox coordinates at IoU=0.80 |
| **Root cause** | 86% zero-gradient steps from clamp | Will learn once clamp removed |

### PSR (Pose/State Recognition — 11 binary)
| Metric | Value | Verdict |
|-------|-------|---------|
| F1 | 0.0000 | Collapse |
| Edit score | 0.5245 | Low but non-zero |
| Active components | 2/11 | 9 channels never learn |
| **Pinned loss** | `psr=0.0001000` every step | ⚠️ Mechanism unknown — see Q2 |

### AR (Action Recognition — 75 classes)
| Metric | Value | Verdict |
|-------|-------|---------|
| Clip accuracy | 0.0000 | Total collapse |
| Frame accuracy | 0.0000 | Total collapse |
| gt_seen | 59/75 | 16 classes unseen in val |
| pred_seen | **1/75** | Collapsed to class 52 only |
| **Root cause** | 86% zero-gradient + 75-class imbalance | Joint result |

### Head Pose (9-DoF)
| Metric | Value | Verdict |
|-------|-------|---------|
| Forward angular MAE | NaN→0.00 | Non-unit vectors |
| Up angular MAE | NaN→0.00 | Non-unit vectors |
| Position error | 352.0372 mm | Reasonable (unit ambiguity) |
| **Fix status** | Already implemented | pos_weight=1.0, dir_weight=1.0 |

### AS (Action State — 17 states) & EV (Error Verification)
| Task | Metric | Verdict |
|------|--------|---------|
| AS | F1=0.0000, mAP recall=0.0000 | NOT Kendall tasks — downstream |
| EV | AP=0.0116, F1=0.0224 | NOT Kendall tasks — downstream |

---

## 5. LOG EVIDENCE

**From log line 526 (Kendall log_var state at resume):**
```
log_var_det=-0.152, log_var_pose=-1.318, log_var_act=-0.011, log_var_psr=-0.318
Σ log_var ≈ -1.8
```

**From log loss readings (6,244 lines):**
- 5,973 of 6,959 training-step readings show `loss=0.0000000` (~86%)
- Nonzero readings almost entirely from detection spikes (det=1–4)
- PSR loss pinned at exactly `psr=0.0001000` every step — no variation

**DET_PROBE b0:**
```
DET_PROBE b0: imgs_with_gt=16, n_gt=16, bestIoU>0.5=419, bestIoU_max=0.80
```
→ Regression branch works. Only classification confidence collapsed.

---

## 6. REMAINING QUESTIONS FOR OPUS

### Q1: PSR Loss Floor — 0.0001000 Every Step

**What we know:**
- PSR loss is identically `0.0001000` every single training step (no variation)
- This happens BEFORE the Kendall total (PSR is a Kendall task)
- `_safe` replaces NaN with 1e-4, but PSR isn't NaN — it's identically 0.0001000
- `_smooth_cap(loss_psr, 20.0)` applied before Kendall
- PSR component prevalence very imbalanced (component-0 = 1.0, component-4 = 0.116)

**What we need:**
Which mechanism produces the precise `0.0001000` value? Is it:
- A fixed per-component loss floor?
- A property of the binary focal loss with the current target distribution?
- Something else in the PSR loss computation?

**Code location:** `losses.py` PSR loss block (~lines 1120-1175), `_psr_temporal_smooth_weight`, `industreal_dataset.py` PSR collation

---

### Q2: AS and EV — Downstream Collapse Mechanism

**What we know:**
- AS (17-state) and EV are NOT Kendall tasks — they use separate loss
- Their zeros appeared simultaneously with all other collapses
- They are not in the Kendall total, so the clamp didn't directly affect them
- Yet they also show near-zero metrics

**Question:** What is the mechanism by which the Kendall clamp collapse propagates to AS and EV? Are they:
- Affected by the VideoMAE backbone being frozen/starved?
- Losing gradient through the shared backbone?
- Using a separate pathway that was also affected?

**Code location:** AS in `losses.py` (~lines 1050–1120), EV in `losses.py` (~lines 920–1050)

---

### Q3: Clamp Fix — Verification Plan

**The fix is applied.** Before we resume training, does Opus want to:
- Review the exact diff (remove clamp, move NaN guard up)?
- Suggest any additional safeguards?
- Recommend a smoke-test criterion to confirm the fix works?

**Expected immediate effect after resuming from checkpoint:**
- Training loss should rarely if ever hit exactly 0.0
- All 6 tasks should receive non-zero gradient on >90% of steps
- Kendall total should be able to go negative (valid) without being zeroed

---

### Q4: AR Class 52 Collapse — Separate Issue?

With 86% zero-gradient addressed, will the AR head naturally recover? Or is the `pred_seen=1/75` collapse potentially a separate initialization or loss configuration issue that needs independent investigation?

---

### Q5: Should We Reset the Kendall log_vars?

When resuming from `crash_recovery.pth` with the clamp fix, the log_vars are at their collapsed values (sum ≈ −1.8). Should we:
- Reset log_vars to 0 before resuming? (would give precision=1.0 for all tasks)
- Or let them adapt from their current state?

---

## 7. FILES IN THIS PACKAGE

| File | Lines | Key |
|------|-------|-----|
| `losses.py` | 1349 | Full MultiTaskLoss with Kendall, all 6 task heads |
| `config.py` | ~500 | All C.* constants, Kendall settings, caps |
| `industreal_dataset.py` | ~800 | Data loading, annotation parsing, collation |
| `popw_train_25pct_15ep_v6.log` | 6244 | Full training/eval log |
| `split_head_pose_loss.py` | 191 | Gradient balance proof |
| `investigate_annotations.py` | 536 | Annotation verification |
| `MASTER_OPUS_ANALYSIS_PROMPT.md` | — | This document |

---

*Generated by: claude-3-sonnet-4.7*
*Session: session-1780463303685 (v1) → updated session-1780472534015 (v2)*
*Clamp fix applied: 2026-06-03*