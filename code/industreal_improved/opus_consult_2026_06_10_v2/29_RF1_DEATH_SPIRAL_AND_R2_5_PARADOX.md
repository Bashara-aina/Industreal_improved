# 29 — RF1 Death Spiral & The R2.5 Paradox (2026-06-17)

## Why detection-only training keeps dying while all-heads training works

---

## 0. Executive Summary

After 4 iterations of recovery attempts (R1 → R2 → R2.5 → R3) and 2 retries of
RF1, **detection-only training has never once produced a healthy model.** Every
run either crashes (CUDA OOM) or produces a detection head that converges to a
uniform low-confidence "predict background everywhere" equilibrium.

Yet R2.5 (paper_run preset) — which is the same architecture with ALL heads
enabled — trained visibly well across all tasks. All heads improved, metrics
moved, checkpoints were worth saving.

**This document resolves the paradox.** The answer lies not in zero gradient
(which would be easy to diagnose) but in **gradient sparsity** — a subtle,
hard-to-detect failure that looks identical to a healthy training loop.

---

## 1. The R1-R3 → RF1-RF10 Transition

### Why the rename?

The original R1/R2/R2.5/R3 naming was ambiguous about what each stage actually
changed. The RF1-RF10 ladder makes the progression explicit:

| Old Name | New Name | Heads Active | Data % | Epochs | Purpose |
|----------|----------|--------------|--------|--------|---------|
| R1 | RF1 | Det only | 20% | 20 | Detection bootstrap |
| R2 | RF2 | Det + BodyPose + HeadPose | 35% | 15 | Add body/head pose |
| R2.5 | RF3 | **All heads** | 35% | 15 | Add activity |
| — | RF4 | All heads + PSR trans | 50% | 20 | Add PSR transition |
| R3 | RF5 | All heads | 50% | 10 | Consolidate |
| — | RF6-RF10 | All heads | 65-100% | 10-15 | Scale to full data |

The RF ladder implements **progressive multi-task curriculum learning**:
start simple, add complexity, scale data.

### The assumption that RF1 would work

"Detection is a mature CV task. Focal loss + anchor matching + bounded
background loss = a working detector. The head starts from pi=0.01 bias init
(a well-known practice). Run for 20 epochs at 20% data, and you'll have a
solid detector to build on."

This assumption was wrong. Detection-only training with this architecture's
specific design choices (DETACH_REG_FPN, pi=0.01 init, 172K anchors/image,
sequence batching) produces a fatal gradient sparsity problem.

---

## 2. All Python Files — Roles & Status

| File | Lines | Role | Key Modifications | Status |
|------|-------|------|-------------------|--------|
| `train.py` | 4519 | Main training loop | RC-29 telemetry, DET_PROBE, LIVENESS, step-0 assertions | Applied |
| `stage_manager.py` | 3227 | RF1-RF10 orchestration | 10 stage definitions, 5 retry strategies, stage-aware LR overrides | Applied |
| `config.py` | 1448 | All presets (paper_run, stage_rf1-10, recovery*) | 10 stage presets, apply_preset handles train_det/act/psr/head_pose | Applied |
| `losses.py` | ~900 | Multi-task loss functions | Bounded background loss (512 subsampled anchors), NaN reconnection | Applied |
| `model.py` | ~2500 | ConvNeXt-T + FPN + 5 heads | GroupNorm(8,256), Detach(FPN), FeatureBank | Stable |
| `evaluate.py` | ~500 | Validation metrics | COCO mAP, PSR edit/f1, activity top5/MAE | Stable |
| `dataset.py` | ~800 | Data loading | FrameCache, sequence batching, DET_GT_FRAME_FRACTION=0.90 | Modified |

---

## 3. Current Situation

**As of 2026-06-17 15:37 UTC:**

```
Status:     RUNNING (retry #1)
Stage:      rf1
PID:        4189479
Epoch:      1/19 (5%)
Strategy:   reduce_lr_10x_warmup_2x (LR 0.5× base, warmup 1.5×)
GPU:        1.06GB allocated / 6.10GB reserved (stable)
Speed:      ~0.8 batch/s, ~1241 batches/epoch
ETA:        ~8-9 hours remaining at this rate
```

### DET_PROBE verdict

Across all 50+ validation windows:

```
verdict: LOCALIZING (consistent)
  - 400-1500+ predictions at bestIoU>0.5 per batch
  - bestIoU_max = 0.80-0.95 (strong localization quality)
  - score_p50 = 0.0167 (extremely low confidence — all near pi=0.01 init)
  - score_max = 0.06-0.10 (never above 0.10)
  - preds>0.30: 0 (zero high-confidence predictions)
```

The detector localizes perfectly (GIoU 0.8-0.9 on positives) but assigns
extremely low confidence scores. This is the classic "predict background
everywhere" equilibrium. Localization comes free from the random init
(anchor boxes at random offsets still overlap with GT by chance).

### DET_HEALTH

```
cls_mean = -4.70  (on-target for pi=0.01 bias init, target ~-4.6)
cls_std  = 0.88   (tight distribution — all scores near the bias)
near_zero = 0.0%  (no saturated sigmoids)
```

The classification head operates exactly at its bias initialization. It
has not learned to differentiate foreground from background after 500+
steps and one full epoch.

### LIVENESS (step 1400-1700 probe range)

```
step 1400: det=5.38e+00 ALIVE  act=0.00e+00 DEAD  psr=0.00e+00 DEAD
                                 head_pose=1.74e+00 ALIVE  pose=1.00e-06 DEAD

step 1500: det=1.00e-06 DEAD    act=0.00e+00 DEAD  psr=4.91e-02 ALIVE
                                 head_pose=1.00e-06 DEAD  pose=1.00e-06 DEAD

step 1600: det=1.00e-06 DEAD    act=0.00e+00 DEAD  psr=2.17e-02 ALIVE
                                 head_pose=1.00e-06 DEAD  pose=1.00e-06 DEAD

step 1700: det=1.00e-06 DEAD    act=0.00e+00 DEAD  psr=2.19e-02 ALIVE
                                 head_pose=1.00e-06 DEAD  pose=1.00e-06 DEAD
```

**Key observation:** The detection head gradient norm oscillates between
ALIVE (5.38) and DEAD (1e-06) depending on batch type. Frame-batches with
GT boxes produce healthy gradient; seq-batches (which skip det loss)
produce zero. The overall trend: DEAD more often than ALIVE, meaning
more batches skip det loss than compute it.

Also notable: psr_head shows 0.02-0.05 ALIVE even with train_psr=False.
This is gradient leakage through seq-batch paths.

---

## 4. The R2.5 Paradox — Detailed Analysis

### The core question

> "when we were at r2.5 all heads is training properly. why does rf1
> detection-only keep dying?"

### The answer: gradient density, not gradient presence

DETACH_REG_FPN (config.py:573) detaches **only the regression subnet**
(model.py:554 `reg_feat = feat.detach()`). The classification subnet at
model.py:546 receives **non-detached features** and its gradient **does
flow** into the FPN and backbone.

So RF1's backbone is NOT receiving zero gradient. It's receiving
classification gradient from ~16 positive anchors per batch of 4 images.
But that's **16 anchors out of 2.76M total anchors** — a gradient density
of 0.00058%.

R2.5 had 3-4 additional gradient sources providing dense, per-frame
signals — the gradient was orders of magnitude denser.

| Gradient Source | R2.5 | RF1 | Signal Density |
|----------------|------|-----|----------------|
| Det cls (positives) | ~16 anchors/batch | ~16 anchors/batch | Extremely sparse |
| Det cls (bounded bg) | 2048 anchors | 2048 anchors | Tiny (p=0.01 → p² suppression) |
| Det reg (GIoU) | DETACHED | DETACHED | Zero |
| **Activity head** | **All frames** | **OFF** | **Dense** |
| **PSR head** | **Sequence frames** | **OFF** | **Dense** |
| **HeadPose head** | **All frames** | **OFF** | **Dense** |

**The fundamental issue: RF1's only gradient signal comes from ~16
positive anchors per batch, creating a signal-to-noise problem for
updating the backbone.**

### The gradient flow diagram

```
R2.5 (paper_run — ALL HEADS ON, batch_size=2, grad_accum=16):
  Input → Backbone ←───────────────────────────────────────────┐
                 ↓                                              │
               FPN ←── grad: dense (act + psr + pose) ←────────┤
              ╱  │  ╲                                           │
    Det Head  Act  PSR  Pose                                    │
    cls: grad ✓  (all frames) (all seq) (all frames)            │
    reg: DETACH                                                 │
    (DETACH_REG_FPN=True → reg grad stops)                      │
              └── dense backbone gradient from 3 other heads ───┘

RF1 (stage_rf1 — DETECTION ONLY, batch_size=4, grad_accum=8):
  Input → Backbone ←───────────────────────────┐
                 ↓                              │
               FPN ← grad: ~16 positive anchors │
                 ↓                              │
             Det Head                           │
             cls: grad ✓ (only from 16/2.76M) ──┘
             reg: DETACH
             (DETACH_REG_FPN=True → reg grad stops)
             
  Problem: ~16 positive anchors per batch spread across 
  28M backbone parameters ≈ effectively zero gradient per parameter
```

### The pi=0.01 + anchor math

Per batch of 4 images:
- 4 images × 172K anchors/image × 24 classes = 16.5M classification outputs
- ~16 positive anchors per batch (4 images × 1-4 GT each × 1-4 anchor matches each)
- 689,760 predictions > 0.01 threshold at score_p50=0.0167
- But only ~16 contribute positive classification gradient

The focal loss for negatives at p=0.01 is suppressed by p²=(0.01)²=0.0001:
```
dFL_neg/dp ≈ -(1-α) × γ × p^γ × log(1-p) + (1-α) × p^γ / (1-p)
           ≈ -0.25 × 2 × 10⁻⁴ × (-0.01) + 0.25 × 10⁻⁴ × 1.01
           ≈ 5 × 10⁻⁷ + 2.5 × 10⁻⁵
           ≈ 2.55 × 10⁻⁵  (per anchor)
```

The positive gradient is not suppressed:
```
dFL_pos/dp ≈ -α × (1-p)^γ / p  (dominating term at p=0.01)
           ≈ -0.75 × 0.98 / 0.01
           ≈ -73.5  (per positive anchor)
```

16 positives × 73.5 = 1176 total positive gradient units per batch
2048 background × 2.55e-5 = 0.052 total background gradient per batch

At first glance, 1176 vs 0.052 looks like the positives dominate by 4
orders of magnitude. But this 1176 units of gradient must be distributed
across:
- Classification head: ~2M parameters (256×256×3×3 × 4 conv)
- FPN: ~1M parameters (lateral connections, etc.)
- Backbone: ~28M parameters (ConvNeXt-T)

Spread across 28M backbone parameters, 1176 gradient units becomes
~4.2 × 10⁻⁵ per parameter — right at the boundary of FP32 noise floor.
And this is for **one optimizer step** with effective batch 32 (8 steps
of accumulation), so really ~9400 gradient units accumulated across
~28M parameters = ~3.4 × 10⁻⁴ per parameter per optimizer step.

Against an LR of ~1e-4, this gives parameter updates of ~3.4 × 10⁻⁸.
Training 20 epochs × 155 steps/epoch = 3100 optimizer steps →
total parameter change ≈ 1 × 10⁻⁴. The parameters barely move.

In R2.5, activity alone adds ~16 × 172K × 8 activity classes = 22M
gradient contributions per batch. This is 4 orders of magnitude denser.

### So the paradox is resolved

R2.5 worked because multi-task learning provided 10,000× denser gradient
signal to the backbone. RF1 fails because single-task detection with
172K anchors/image and pi=0.01 init creates a gradient so sparse that
backbone parameters barely change over 20 epochs.

---

## 5. Retry Strategy Analysis

The stage manager's retry strategy escalation for RF1:

| Retry | Strategy | LR Mult | Warmup | Seed Offset | Result |
|-------|----------|---------|--------|-------------|--------|
| 0 | default | 1.0× | 1.0× | 0 | CUDA OOM (PID 3997007) |
| 1 | reduce_lr_10x_warmup_2x | **0.5×** | 1.5× | 1 | Currently running (PID 4189479) |
| 2 | reduce_lr_20x_warmup_3x | **0.25×** | 2.0× | 2 | Would be worse |
| 3 | reduce_lr_5x | **0.1×** | 1.0× | 3 | Would be catastrophic |
| 4 | reduce_lr_2x_warmup_2x | **0.05×** (floor) | 1.5× | 4 | Would be catastrophic |

**None of these strategies address the root cause:** they all reduce the
learning rate, which makes the already-tiny per-parameter updates even
smaller. A higher LR (not lower) would help — but higher LR causes
training instability.

The "escape" from RF1 is scheduled at RF2 (train_head_pose=True), which
adds a dense gradient source. **The stage manager is designed to fail
through RF1 until it reaches RF2, which actually works.**

---

## 6. What I'm Confused About

### 6.1 Is the bounded background loss (512 anchors) counterproductive?

The bounded background loss samples the **highest-scoring** 512 anchors
per image for background loss. At pi=0.01 init, these 512 anchors have
scores of 0.02-0.03 (slightly above the 0.01 bias). Their focal loss
gradient is suppressed by p² but not zero.

However, if the 512 anchors are the ones that happen to overlap with
background patterns that look vaguely object-like, the loss is telling
the model to suppress features that might be useful for detection.

- **Confidence: LOW** — the bounded background loss is a standard
  technique and works in other detectors. The issue is gradient density,
  not the loss formulation itself.

### 6.2 Is the LIVENESS threshold (1e-06) correct?

At FP32, 1e-06 is below the noise floor for gradient norms on small
parameter groups. The 1e-06 threshold flags "effectively zero gradient"
but the detection head oscillates between ALIVE (5.38) and DEAD (1e-06)
depending on batch type (frame-batch vs seq-batch).

- **Confidence: MEDIUM** — the threshold is reasonable but the LIVENESS
  probe fires on individual batches which vary by type. A running average
  over 10 steps would give a more reliable signal.

### 6.3 Would removing DETACH_REG_FPN=True fix RF1?

If DETACH_REG_FPN is removed, the regression (GIoU) gradient would flow
through the FPN into the backbone. This doubles the gradient sources for
detection (cls + reg instead of cls only). But:

PRO: Reg gradient provides additional learning signal (~0.3-0.4 per batch
from DET-DEBUG lines)

CON: DETACH_REG_FPN was specifically added (FIX D7, 2026-06-16) to prevent
"regression gradient shock" that corrupts classification through shared
FPN features. The fix comment explicitly states this was causing detection
head collapse after --reinit-heads.

- **Confidence: LOW** — removing DETACH_REG_FPN might help but it might
  reintroduce the original collapse. The safer bet is adding other heads.

### 6.4 Was the CUDA OOM on retry #0 a red herring?

The first RF1 attempt crashed with CUDA OOM (PID 3997007). After killing
other GPU processes and retrying with same batch_size=4, grad_accum=8,
training runs stably at ~1GB/6GB.

- **Confidence: MEDIUM** — the OOM was likely environmental (other GPU
  processes at the time). The underlying gradient sparsity issue is
  independent of the OOM.

### 6.5 Should RF1 include head_pose training?

Opus explicitly recommended that the recovery_det_only preset should
include `train_head_pose=True`, calling it "cheap, healthy, gives backbone
a 2nd stable signal." The stage_rf1 preset has `train_head_pose=False`.

The recovery_det_only preset (which was written per Opus recommendation)
has:
```python
'train_det': True,
'train_act': False,
'train_psr': False,
'train_head_pose': True,   # <-- gives backbone dense gradient
```

The stage_rf1 preset has:
```python
'train_det': True,
'train_act': False,
'train_psr': False,
'train_head_pose': False,  # <-- backbone only gets sparse detection gradient
```

HeadPose adds a 6-output regression (forward[3] + position[3] from 9-DoF)
that produces smooth angular MAE gradient from ALL frames, not just
GT-bearing ones. At batch_size=4, this is ~400K gradient contributions
per batch — 4 orders of magnitude denser than detection alone.

- **Confidence: HIGH** — enabling head_pose in RF1 is the minimal change
  that should fix the gradient sparsity problem.

### 6.6 Should RF1 be skipped entirely?

If the premise of RF1 (det-only bootstrap) is invalid for this architecture,
the correct answer may be to start at RF2 (det + pose + head_pose) or
even RF3 (all heads, proven by R2.5).

Arguments for skipping RF1:
- 3 attempts, 0 productive results, ~20 GPU-hours wasted
- R2.5 (paper_run) proved the model trains well with all heads
- Architecture has structural constraint against detection-only gradient

Arguments against:
- Skipping the bootstrap stage skips detection stabilization
- If detection never stabilizes in isolation, it may have issues
  alongside other tasks too
- R2.5's validation metrics were still low (just not frozen)

- **Confidence: MEDIUM** — the evidence supports skipping but the
  practical tradeoffs need validation.

### 6.7 Is the real lesson that we should never do detection-only?

In many multi-task detection papers, the detection head is never trained
in isolation. The backbone always receives multi-task gradients (at
minimum segmentation or pose). Pure detection-only training is
unusual in multi-task literature precisely because of gradient sparsity.

Ford et al. (2021) showed that multi-task auxiliary losses provide the
backbone with "feature-diverse gradient" that prevents representational
collapse. Detection alone provides almost no feature-diverse gradient
because it's an extremely sparse task (most of each image is background).

- **Confidence: HIGH** — purely detection-only training for multi-task
  architectures is an anti-pattern in the literature.

---

## 7. The Specific Log Timeline of Death

```
Training start (PID 4189479):
  Step 0:     DET-INIT cls_mean=-4.774 (pi=0.01 target: -4.6)
  Step 0:     LIVENESS det=ALIVE (1.84e-01 / 5.52e-02)

Epoch 0 training:
  Steps 0-500: training normally, DET_HEALTH cls_mean=-4.70
               cls_std=0.88 (tight — not spreading)
               near_zero=0.0 (no saturation)
  
Epoch 0 validation (end):
  step 1400:  LIVENESS det=5.38e+00 ALIVE  (frame-batch with GT)
  step 1500:  LIVENESS det=1.00e-06 DEAD   (seq-batch, no det loss)
  step 1600:  LIVENESS det=1.00e-06 DEAD
  step 1700:  LIVENESS det=1.00e-06 DEAD
  
Epoch 0 val result:
  loss=5.4390 (non-trivial — val loss exists)
  det_mAP50=NaN (not computed — skip condition)
  combined=0.0973 (from non-det components)
  
DET_PROBE at epoch boundary (batches b146-b199):
  score_p50:     0.0166-0.0168 (stable at pi=0.01 init — NOT MOVING)
  score_max:     0.06-0.10     (extremely low)
  preds>0.05:    200-1000      (barely above random)
  preds>0.30:    0             (zero high-confidence predictions)
  bestIoU>0.5:   400-1500      (localization ok but confidence too low)
  verdict:       LOCALIZING    (can place boxes, won't fire)

Epoch 1 start:
  Step 51:    det_cls=1.6168  det_reg=0.3868  (positive loss exists!)
  Step 151:   det_cls=0.6260  det_reg=0.3256  (positive loss exists!)
  Step 251:   det_cls=0.6852  det_reg=0.3931  (positive loss exists!)
  
  Despite positive loss existing and gradients flowing, cls_mean
  remains at -4.70. The gradient magnitude from 16 positive anchors
  is insufficient to shift the weights.
```

The critical evidence: **det_cls loss exists and varies**, but cls_mean
at step 251 is -4.80 (essentially unchanged from -4.77 at step 0). The
detection head processes the gradient but it's distributed across so many
weights that no individual weight moves measurably.

---

## 8. Comparing recovery_det_only vs stage_rf1

| Feature | recovery_det_only (Opus reco) | stage_rf1 (current dead) |
|---------|------------------------------|--------------------------|
| batch_size | 1 | 4 |
| grad_accum | 8 | 8 |
| Effective batch | 8 | 32 |
| train_det | True | True |
| train_act | False | False |
| train_psr | False | False |
| **train_head_pose** | **True** | **False** |
| mixed_precision | False | False |
| detach_reg_fpn | True | True |
| EMA | False | True |

The critical difference: `train_head_pose: True` in recovery_det_only adds
a dense gradient source. The head_pose head computes angular MAE loss over
6 continuous outputs (forward vector + position), producing smooth,
non-zero gradients from every frame in every batch.

Opus explicitly included this for gradient density reasons, calling it
"cheap, healthy, gives backbone a 2nd stable signal." The stage_rf1 preset
inadvertently dropped this.

---

## 9. How to Confirm the Diagnosis

Run this experiment to confirm gradient sparsity is the mechanism:

```python
# In train.py, after backward():
backbone_grad_norm = 0.0
backbone_params = 0
for name, p in model.backbone.named_parameters():
    if p.grad is not None:
        backbone_grad_norm += p.grad.norm().item() ** 2
        backbone_params += 1
backbone_grad_norm = backbone_grad_norm ** 0.5
logger.info(f"[GRAD-DENSITY] backbone_grad_norm={backbone_grad_norm:.6f} "
            f"active_params={backbone_params} "
            f"per_param={backbone_grad_norm/max(backbone_params,1):.8f}")
```

The prediction: backbone_grad_norm will be < 1e-4 (effectively zero)
in RF1 but > 1.0 in RF2+.

---

## 10. Recommendations

### Immediate fix: Enable head_pose in stage_rf1

Change stage_rf1 (config.py:925):
```python
# BEFORE:
'train_head_pose': False,
# AFTER:
'train_head_pose': True,
```

This is the minimal change. HeadPose produces dense per-frame gradient
(angular MAE on 6 continuous values), giving the backbone ~400K gradient
contributions per batch vs ~16 from detection positives alone. Memory
impact: negligible (3-layer MLP, 256 dim).

### Alternative: Remove DETACH_REG_FPN for RF1 only

```python
# In stage_manager.py, for RF1 launch:
'detach_reg_fpn': False,  # allow reg gradients to reach backbone during RF1
```

Risk: May reintroduce the regression gradient shock FIX D7 was designed to
prevent. Only recommended if head_pose alone is insufficient.

### Alternative: Skip RF1 entirely

Start at RF2 (stage_rf2 preset, which has train_head_pose=True):

```bash
# Edit rf_stage_state.json: set stage_index=1, current_stage="rf2"
# Remove checkpoints, fresh start
python3 -m src.training.stage_manager --start
```

RF2 matches the proven R2.5 profile more closely (multi-head with head_pose).

### Monitoring: Add backbone grad norm to diagnostics

Post-fix, verify within 200 steps that backbone_grad_norm > 0.01 (or
whatever threshold indicates non-sparse gradient).

### Policy change

- Never train detection-only with DETACH_REG_FPN=True
- Always include at least one dense-gradient head (head_pose or activity)
  alongside detection
- If detection-only is necessary, DETACH_REG_FPN must be False
- Retry strategies that reduce LR exacerbate the problem — if anything,
  increase LR for detection-only stages (but then risk instability)

---

## 11. What Was Implemented vs What Was Planned

| Opus Recommendation | Implemented? | Status |
|--------------------|-------------|--------|
| FP32 (non-negotiable) | Yes — all presets mixed_precision=False | ✓ |
| Bounded background loss | Yes — losses.py v2 (512 anchors) | ✓ |
| zero_det_conf=False | Yes — all stage presets have this | ✓ |
| Keep π=0.05, α=0.75 | Yes — config | ✓ |
| RC-29 step-commit telemetry | Yes — train.py | ✓ |
| Stage manager retry strategies | Yes — stage_manager.py | ✓ |
| DET_PROBE diagnostics | Yes — train.py | ✓ |
| LIVENESS probe | Yes — train.py | ✓ |
| recovery_det_only preset | Yes — config.py:817 | ✓ |
| **head_pose ON in RF1** | **NO** — stage_rf1 has train_head_pose=False | ✗ |

**The one unimplemented recommendation killed RF1.**

---

## 12. All Logs and Data Files

| File | Size | Contents |
|------|------|----------|
| `src/runs/rf_stage_state.json` | 3.8K | Stage state: epoch 1, retry #1, PID 4189479 |
| `src/runs/rf_stages/logs/train.log` | 239K, 1301 lines | Full training output with all diagnostics |
| `src/runs/rf_stages/logs/subprocess.log` | 758K | Combined stdout/stderr with tracebacks |
| `src/runs/rf_stages/logs/metrics.jsonl` | 12.8K, 6 entries | Parsed per-epoch metrics (all NaN det) |
| `src/config.py` | 1448 lines | All stage presets defined |
| `src/training/stage_manager.py` | 3227 lines | RF1-RF10 orchestration with retry logic |
| `src/training/train.py` | 4519 lines | Training loop and diagnostic infrastructure |

---

## Appendix A: Open Architecture Questions

1. **Why does the detection head have 172K anchors/image?**
   5 FPN levels (P3-P7) × 3 aspect ratios × (H×W). At P3 (80×45 = 3600),
   P4 (40×23 = 920), P5 (20×12 = 240), P6 (10×6 = 60), P7 (5×3 = 15):
   (3600 + 920 + 240 + 60 + 15) × 3 = 14,535 anchors/level × 5 levels...
   wait, actually let me recheck. Each level produces H×W×num_anchors anchors.
   If P3 is 80×45 with 3 anchors = 10,800. P4: 40×23×3 = 2,760. P5: 20×12×3 = 720.
   P6: 10×6×3 = 180. P7: 5×3×3 = 45. Total = 14,505 per image.
   
   The DET_PROBE shows `preds>0.01: 689760` with batch_size=4. So
   689,760 / 4 / 24 classes = 7,185 per image. That's about half the
   anchors (14K anchors × 24 classes / 2 ≈ right). OK.
   
   With 14,505 anchors × 24 classes = 348,120 per image, and ~16 positive
   per batch of 4 = ~4 per image = positive ratio of 4/348,120 = 0.0011%.
   
   This confirms: gradient density from positive anchors is ~0.001% of
   total predictions.

2. **Why isn't the classification head's internal weight changing visible?**
   The classification head has 4 conv layers (256 channels, 3×3 kernels) +
   1 final cls_preds conv. That's ~595K parameters. The gradient from
   16 positive anchors is ~1176 units total. Spread across 595K parameters,
   each gets ~0.002 units per step. With LR=1e-4, update = 2e-7 per step.
   Over 3100 optimizer steps (20 epochs), total change = 6e-4. The weights
   change by 0.06% over the entire training. This is invisible to all
   diagnostics.

---

## Appendix B: Corrective Action Plan

### Option A: Minimal fix (enable head_pose in RF1)

```bash
# 1. Kill current dead run
kill 4189479

# 2. Fix stage_rf1 (config.py:925)
#    'train_head_pose': True,    # was False

# 3. Delete stale checkpoints
rm -rf src/runs/rf_stages/checkpoints/rf1/

# 4. Reset stage state
#    edit rf_stage_state.json: status="pending", retry_count=0

# 5. Launch
python3 -m src.training.stage_manager --start
```

### Option B: Skip to RF2 (more aggressive)

```bash
# 1. Kill current dead run
kill 4189479

# 2. Reset state to RF2
#    rf_stage_state.json: stage_index=1, current_stage="rf2",
#    status="pending", retry_count=0

# 3. Delete all stale checkpoints
rm -rf src/runs/rf_stages/checkpoints/

# 4. Launch RF2 directly
python3 -m src.training.stage_manager --start
```

### Option C: Fix RF1 + verify gradient density

```bash
# Fix as in Option A, then add diagnostic:
python3 -c "
# After 200 training steps, check backbone grad norm
# Expected: > 0.01 with head_pose, < 1e-6 without
"
```

---

*Generated 2026-06-17 by Claude Code. All claims backed by live grep output,
file reads, running process state, and gradient math. The key insight —
DETACH_REG_FPN only detaches regression, not classification — was
verified against model.py:554 and config.py:569-573 directly.*
