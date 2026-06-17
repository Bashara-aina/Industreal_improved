# 30 — OPUS MASTER PROMPT v7: Complete Situation Overview (2026-06-17)

## For Upload to Opus — Self-Contained Summary of Files 00–29

---

## How to Use This File

This directory contains 29 files documenting the full consultation history
(June 11-17, 2026) for the POPW multi-task assembly understanding model.
If you need detail on any topic, read the referenced file number. This
prompt is self-contained — Opus should read this first, then consult
specific files as needed.

**Files to prioritize for deep context:**
- `00_JOURNEY_AND_STATUS.md` — Full project timeline (Phases 1-9)
- `13_OPUS_ANSWER_v4.md` — RC-28/RC-29 diagnosis (the big breakthroughs)
- `26_RF1_RF10_COMPREHENSIVE_STATUS.md` — RF stage definitions and pipeline
- `28_DET_DEATH_SPIRAL_FIX_AND_RUNBOOK.md` — Bounded background loss fix
- `29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md` — **Root cause analysis (newest)**

---

## 1. Project Identity

**Project**: POPW — Pose-Conditioned Multi-Task Architecture for Assembly
Understanding (egocentric + third-person video)
**Hardware**: Single RTX 3060 (12 GB), i5-12400F, 32 GB RAM
**Framework**: PyTorch 2.2, CUDA 12.1
**Paper**: `popw_paper_improved.tex` (target)

### Architecture

- **Backbone**: ConvNeXt-Tiny → C2(96), C3(192), C4(384), C5(768)
- **FPN Neck**: P3–P7, 256ch, lateral 1×1 + top-down upsample
- **5 Task Heads**: Detection (24 cls, RetinaNet-style Focal+GIoU),
  Body Pose (17 KP, Wing loss), Head Pose (9-DoF, MSE×0.001),
  Activity (75 cls, LDAM-DRW(s=30)), PSR (11 binary, Binary Focal)
- **Cross-Task Conditioning**: PoseFiLM, HeadPoseFiLM (stop_grad),
  det_conf (stop_grad) → activity input
- **Total**: 76.16M params (53.42M trainable)

---

## 2. Consultation History — What Was Solved

### Round 1 (June 11-13, files 00-15): RC-25 → RC-29

The model was producing frozen/constant outputs across all tasks. Opus
diagnosed that AMP (fp16) GradScaler was silently skipping optimizer steps
(RC-29). The fix: **FP32 across all presets** (`mixed_precision: False`).

Other fixes from this round:
- **RC-28** (3-way deadlock): Empty frames dominated detection gradient
  → bounded background loss (512 subsampled anchors)
- **RC-29 step-commit telemetry**: Per-epoch committed/skipped optimizer
  step counter (inert under FP32)
- Step-0 assertions, DET_PROBE, LIVENESS diagnostics added

### Round 2 (June 13-16, files 16-25): R1→R3 Recovery Protocol

Opus prescribed a staged recovery protocol:
- R0: Smoke test (200 steps, FP32)
- R1: Detection only (recovery_det_only preset — **includes head_pose**)
- R2: Joint recovery (all heads)
- R3: Scale to full data

All 16 Opus recommendations were implemented and verified against a
100-item audit checklist.

### Round 3 (June 16-17, files 26-29): RF1-RF10 + Death Spiral

The R1-R3 naming was replaced with an explicit 10-stage progressive
multi-task ladder (RF1-RF10). The stage_manager.py orchestrates all 10
stages with automatic gate checks and retry strategies.

**Problem introduced**: The stage_rf1 preset was created with
`train_head_pose=False`, but the Opus-prescribed recovery_det_only preset
has `train_head_pose=True`. This single discrepancy killed RF1.

---

## 3. Current Situation — The RF1 Death Spiral

### What's running now

```
Stage:      rf1 (retry #1, PID 4189479)
Epoch:      1/19 (5% complete)
Strategy:   reduce_lr_10x_warmup_2x (LR 0.5× base)
GPU:        1.06GB / 6.10GB reserved (stable)
```

### The symptom

The detection head converges to a **uniform low-confidence equilibrium**
within the first 500 steps and never escapes:

```
DET_PROBE (all 50+ validation windows):
  score_p50 = 0.0167  (at pi=0.01 bias init — NOT MOVING)
  score_max = 0.06-0.10 (barely above threshold)
  preds>0.30 = 0     (zero high-confidence predictions)
  bestIoU>0.5 = 400-1500 (localization is fine)
  verdict: LOCALIZING (can place boxes, refuses to fire)

DET_HEALTH:
  cls_mean = -4.70 (pi=0.01 target: -4.6 — on init, not learning)
  cls_std  = 0.88  (tight — no differentiation)
```

### The R2.5 Paradox (resolved in file 29)

R2.5 (paper_run preset, ALL heads ON at batch_size=2 × grad_accum=16,
effective batch 32) trained visibly well. RF1 (detection-only,
batch_size=4 × grad_accum=8, effective batch 32) keeps dying.

**The resolved answer**: DETACH_REG_FPN=True (config.py:573) only detaches
the **regression** subnet gradients (model.py:554). The **classification**
subnet's gradient CAN flow to FPN and backbone. BUT — at pi=0.01 bias init,
the detection head produces only ~16 positive anchors per batch of 4 images
(1-4 GT boxes × 1-4 anchor matches each). Across 28M backbone parameters,
16 positive anchors' gradient = ~4.2 × 10⁻⁵ gradient units per parameter
per step — effectively zero at FP32 noise floor.

R2.5 masked this because activity + PSR + head_pose provided **dense
per-frame gradient** (millions of contributions per batch), keeping the
backbone updating normally despite detection's sparse signal.

**The gradient density gap:**
| Source | RF1 | R2.5 | Ratio |
|--------|-----|------|-------|
| Det cls positive | ~16 anchors/batch | ~16 anchors/batch | 1× |
| Det cls background | 2048 (suppressed) | 2048 (suppressed) | 1× |
| Det reg (GIoU) | DETACHED | DETACHED | 0× |
| Activity | **OFF** | All frames | ∞ |
| PSR | **OFF** | Sequence frames | ∞ |
| HeadPose | **OFF** | **All frames** | **∞** |

### Gradient math

Per batch of 4 images:
- 4 × 172K anchors × 24 classes = 16.5M classification outputs
- ~16 positive anchors produce gradient of ~73.5/unit → 1176 total
- 1176 across 28M backbone params = 4.2 × 10⁻⁵ per param per step
- Over 3100 optimizer steps (20 epochs): total change ≈ 0.06%
- The weights barely move

---

## 4. The Specific Fix That's Missing

The Opus-recommended recovery_det_only preset has:
```python
'train_head_pose': True,   # gives backbone dense gradient signal
```

The created stage_rf1 preset has:
```python
'train_head_pose': False,  # backbone only gets sparse detection gradient
```

**This is the single discrepancy that killed RF1.** Enabling head_pose adds
dense per-frame angular MAE gradient (6 continuous outputs × all frames),
providing ~400K gradient contributions per batch vs ~16 from detection
positives alone.

---

## 5. Questions for Opus

### Q1: Is the gradient sparsity analysis correct?

File 29 proves that ~16 positive anchors out of 348K total outputs per
image produces gradient too sparse to drive backbone updates. But we
haven't empirically verified this with a `backbone_grad_norm` diagnostic.
**Is there any other mechanism that could cause RF1's failure?**

### Q2: Fix strategy — which option?

Three options for fixing RF1:

**A) Enable head_pose in stage_rf1** (minimal change):
```python
'train_head_pose': True,   # was False
```
Expected: head_pose adds dense gradient → backbone updates → FPN features
evolve → detection head benefits. Same architecture as R2.5 (but fewer heads).

**B) Remove DETACH_REG_FPN for RF1 only:**
```python
'detach_reg_fpn': False,   # allow regression gradient to reach backbone
```
Risk: DETACH_REG_FPN was added as FIX D7 (2026-06-16) specifically to
prevent "regression gradient shock" after --reinit-heads. Removing it
might reintroduce the original collapse.

**C) Skip RF1 entirely, start at RF2:**
RF2 has `train_head_pose=True` by design. Starting at RF2 bypasses the
RF1 failure. But skips the detection stabilization phase.

**Which option does Opus recommend? Is there a D) we haven't considered?**

### Q3: Is the entire RF1 premise flawed?

If detection-only training with this architecture is fundamentally
non-viable (due to DETACH_REG_FPN + pi=0.01 + 172K anchors creating
intractable gradient sparsity), then the progressive ladder should start
at RF2 (det + body pose + head pose) instead of RF1.

**Should RF1 be removed from the stage definitions entirely?**

### Q4: The bounded background loss — is 512 correct?

The bounded background loss takes the highest-scoring 512 anchors per
image (out of ~14.5K) and computes focal loss. If features are static
(due to zero backbone gradient), the same 512 anchors are selected every
batch — the loss becomes deterministic. Is this a secondary contributor
to the death spiral?

### Q5: Retry strategy design

The current retry strategies **all reduce LR** (default → 0.5× → 0.25×
→ 0.1× → 0.05×). For a gradient sparsity problem, reducing LR makes
things worse. **Should retry strategies ever increase LR for certain
failure modes?** If so, what's the signal that tells us "increase LR"
vs "decrease LR"?

### Q6: Gradient density diagnostic

We proposed adding this diagnostic to train.py:
```python
backbone_grad_norm = sum(p.grad.norm().item() for p in model.backbone.parameters())
logger.info(f"backbone_grad_norm={backbone_grad_norm:.6f}")
```
**Is this the right diagnostic?** What threshold indicates "healthy"
backbone gradient flow?

### Q7: DETACH_REG_FPN design question

DETACH_REG_FPN was added to prevent regression gradient shock through
shared FPN features. But it creates a dependency on other task heads
for backbone gradient. In a progressive training schedule where
detection comes first, this is self-defeating.

**What's the correct long-term design?**
- Keep DETACH_REG_FPN and always train at least one non-detached head?
- Remove DETACH_REG_FPN and accept the gradient shock risk?
- Make DETACH_REG_FPN stage-aware (False for RF1, True for RF2+)?

### Q8: pi=0.01 vs higher bias initiation

At pi=0.01, the focal loss for negatives is suppressed by p²=0.0001,
making negative gradients negligible. But the **positive** gradient
(not suppressed) is spread across ~16/348K outputs = 0.0046% density.

Would a higher pi (e.g., 0.05 or 0.1) help by:
- Increasing the number of "positive" matches (more anchors near GT)
- Reducing the gradient suppression on negatives (higher p = less p² suppression)
- Providing denser overall gradient at the cost of more false positives early on

### Q9: Dataset subset concern (from file 28)

File 28 identified a contradiction: `config.py:503` says the activity
sampler yields ~24% GT batches, but actual RF1 training sees ~0.7%.
The likely cause: `subset_ratio=0.2` + greedy activity-coverage subset
selection in `_scan_and_index` ignores OD-label availability.

**Should we audit the RF1 training subset for OD-label coverage?**
The command to run:
```bash
python diag_gt_coverage.py --preset stage_rf1 --subset-ratio 0.2
```

### Q10: What are we missing?

After 4 rounds of consultation and 29 files of analysis, RF1 is still
dead. **What blind spot persists?** What question haven't we asked?

---

## 6. Key Files Reference

| File | Content | Priority |
|------|---------|----------|
| `00_JOURNEY_AND_STATUS.md` | Full project timeline | High |
| `03_ARCHITECTURE_DEEP_DIVE.md` | Architecture details | Medium |
| `13_OPUS_ANSWER_v4.md` | RC-28/RC-29 diagnosis | High |
| `17_OPUS_ANSWER_v5.md` | Ultimate path (R1→R4 ladder) | High |
| `18_ULTIMATE_MASTER_GUIDE_INDUSTREAL.md` | Full master guide | Medium |
| `19_PRE_TRAINING_READINESS_AUDIT_100.md` | 100-item audit | Reference |
| `22_FINAL_PREFLIGHT_GAP_CLOSURE.md` | Pre-flight closure | Medium |
| `26_RF1_RF10_COMPREHENSIVE_STATUS.md` | RF stage definitions | High |
| `28_DET_DEATH_SPIRAL_FIX_AND_RUNBOOK.md` | Bounded background loss fix | High |
| `29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md` | **Root cause analysis (newest)** | **Highest** |
| `popw_paper_improved.tex` | Target paper | Reference |
| `src/config.py` | All stage presets | Reference |
| `src/training/stage_manager.py` | RF1-RF10 orchestration | Reference |

---

## Appendix: Quick Config Reference

```python
# === paper_run (R2.5 — WORKED) ===
'batch_size': 2, 'grad_accum_steps': 16,
'mixed_precision': False,
'train_det': True, 'train_act': True, 'train_psr': True, 'train_head_pose': True,
'detach_reg_fpn': True, 'detach_psr_fpn': True,

# === stage_rf1 (CURRENT — failing) ===
'batch_size': 4, 'grad_accum_steps': 8,
'mixed_precision': False,
'train_det': True, 'train_act': False, 'train_psr': False, 'train_head_pose': False,  # <-- PROBLEM
'detach_reg_fpn': True, 'detach_psr_fpn': True,

# === recovery_det_only (Opus recommended — NOT USED in RF ladder) ===
'batch_size': 1, 'grad_accum_steps': 8,
'mixed_precision': False,
'train_det': True, 'train_act': False, 'train_psr': False, 'train_head_pose': True,  # <-- HAS HEAD POSE
'detach_reg_fpn': True, 'detach_psr_fpn': True,

# === stage_rf2 (next stage — would work) ===
'batch_size': 4, 'grad_accum_steps': 8,
'mixed_precision': False,
'train_det': True, 'train_act': False, 'train_psr': False, 'train_head_pose': True,  # <-- HAS HEAD POSE
'detach_reg_fpn': True, 'detach_psr_fpn': True,
```

---

*Generated 2026-06-17 as the master overview prompt for Opus consultation v7.
Accompanies files 00-29 in the opus_consult_2026_06_10_v2 directory.*
