# 35 — OPUS MASTER PROMPT v8: Complete Situation Overview (2026-06-20)

## For Upload to Opus — Self-Contained Summary of Files 00–34

---

## How to Use This File

This directory contains 34 files documenting the full consultation history
(June 11–20, 2026) for the POPW multi-task assembly understanding model.
This prompt is **self-contained** — Opus should read this first, then consult
specific files as needed for deep context.

**Files to prioritize for deep context:**
- `00_JOURNEY_AND_STATUS.md` — Full project timeline (Phases 1–12)
- `26_RF1_RF10_COMPREHENSIVE_STATUS.md` — RF stage definitions, RF2 epoch-by-epoch data, 5 fix proposals
- `29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md` — Gradient sparsity proof (correct) + RF2 postscript
- `31_KENDALL_BUG_DISCOVERY_AND_FIX.md` — Kendall bug + RF2 cross-validation
- `33_OPEN_QUESTIONS.md` — 24 open questions organized by severity
- `34_RF2_SWARM_MONITOR.md` — 20-agent monitoring swarm documentation

---

## 1. Project Identity

**Project**: POPW — Pose-Conditioned Multi-Task Architecture for Assembly
Understanding (egocentric + third-person video)
**Hardware**: Single RTX 3060 (12 GB), i5-12400F, 32 GB RAM
**Framework**: PyTorch 2.2, CUDA 12.1
**Paper**: `popw_paper_improved.tex` (target)

### Architecture

- **Backbone**: ConvNeXt-Tiny → C2(96), C3(192), C4(384), C5(768), 28M params
- **FPN Neck**: P3–P7, 256ch, lateral 1×1 + top-down upsample, ~1M params
- **5 Task Heads**: Detection (24 cls, RetinaNet-style Focal+GIoU, 172K anchors/image),
  Body Pose (17 KP, Wing loss, NO keypoint annotations — loss=0 always),
  Head Pose (9-DoF, MSE), Activity (75 cls, LDAM-DRW(s=30)),
  PSR (11 binary, Binary Focal, causal transformer 3L/4H/T=2)
- **Cross-Task Conditioning**: PoseFiLM, HeadPoseFiLM (stop_grad),
  det_conf (stop_grad) → activity input
- **Total**: 76.16M params (53.42M trainable)
- **Key quirk**: `DETACH_REG_FPN=True` detaches **regression subnet only** — classification gradient CAN flow to FPN/backbone

### Training Framework

- **RF ladder**: 10-stage progressive multi-task curriculum (RF1–RF10)
- **Stage manager**: `stage_manager.py` (3247 lines) orchestrates all stages with
  automatic gate checks and retry strategies
- **Monitoring**: 20-agent swarm (22 agents, 134 checks/cycle, 5-min interval)
- **Current state**: RF2 epoch 16 (PID 1043628), detection classifier collapsed
- **Heartbeat**: `_write_stage_heartbeat()` every 50 batches for swarm

---

## 2. Consultation History — What Was Solved

### Round 1 (June 11–13): RC-25 → RC-29

The model was producing frozen/constant outputs across all tasks. AMP (fp16)
GradScaler was silently skipping optimizer steps (RC-29). Fix: **FP32 across
all presets**. Other fixes: RC-28 (empty frames dominated detection gradient →
bounded background loss, 512 subsampled anchors), step-commit telemetry,
DET_PROBE, LIVENESS diagnostics, 5 eval guards for single-head training modes.

### Round 2 (June 13–16): R1→R3 Recovery Protocol

Opus prescribed a staged recovery protocol (R0 smoke test → R1 detection +
head_pose → R2 joint → R3 scale). All 16 Opus recommendations implemented,
verified against 100-item audit.
**Key finding**: `recovery_det_only` preset has `train_head_pose=True`,
but `stage_rf1` preset was created with `train_head_pose=False` — a fatal
discrepancy.

### Round 3 (June 16–17): RF1 Death Spiral + Kendall Bug

**Gradient sparsity proof** (`29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md`):
Detection-only training produces ~16 positive anchors per batch out of 2.76M
total anchors (0.00058%). Gradient per backbone param = ~4×10⁻⁵ per step —
effectively zero. R2.5 (all heads) masked this with 10,000× denser gradient
from activity + PSR + head_pose.

**Kendall bug** (`31_KENDALL_BUG_DISCOVERY_AND_FIX.md`): The `train_head_pose=True`
fix was silently neutralized by `losses.py:1589` — when `train_pose=True AND
train_act=False`, the `loss_head_pose` (~1.7) was computed but excluded from
total loss. Head pose received ZERO gradient for 7+ epochs. Fix confirmed:
head_pose gradient went from NO_GRAD → ALIVE at step 0, cls_std 0.88 → 1.37
(1.6× broader), cls_max 0.47 → 2.78 (5.9× higher).

### Round 4 (June 17–20): RF1 Completion → RF2 Epoch 15 Collapse

**RF1 completed** with Kendall fix. Per stage_history, `best_det_mAP50=0.45`.
But metric_history shows max 0.184 — **2.4× discrepancy** (Q02).

**RF2 launched** with: head_pose ALIVE, DET_GT_FRAME_FRACTION=0.90,
DETACH_REG_FPN=False, 35% data.

**Epoch 15 collapse**: det_mAP50 = 0.184 (ep 8) → 0.159 (ep 10) → 0.000010 (ep 13)
→ 0.001 (ep 15). Head pose MAE continued improving (71.67° → 47.84°).

**20-agent monitoring swarm deployed** — 6 bugs found and fixed in first hours.

---

## 3. Current Situation — The cls_score Bias Equilibrium

### The Central Problem

Despite ALL fixes applied, the detection classifier at RF2 epoch 15 produces
uniform ~0.079 scores:

```
DET_PROBE:
  score_p50  = 0.019   (median at bias floor)
  score_mean = 0.079   (all classes ~same)
  score_std  = 0.0068  (near-zero variance — 24 classes indistinguishable)
  cls_mean   = -2.54   (bias drifted from -4.6 pi=0.01 init)
  preds>0.30 = 0       (zero confident predictions)
  
EVAL COLLAPSE: 56 occurrences at epoch 15
det_mAP50  = 0.001    (near zero)
det_mAP    = 0.000
```

**The dissociation**: Head pose MAE 47.84° (improving, gradient flowing)
while detection mAP = 0.001 (collapsed). The backbone IS receiving gradient
but the classification head's internal weights converge to a fixed point.

### The Mathematical Model

```
cls_logit = W·x + b    (bias b in final cls_preds conv)
score = sigmoid(cls_logit)

At equilibrium: Σ sigmoid(b + W·x_i) ≈ Σ target_i for all anchors i
Σ target_i ≈ 0.0011% of total anchors (positive anchors)
→ Σ sigmoid(b + W·x_i) ≈ 0.0011% × 348K per image

This forces b to a value where sigmoid(b) ≈ mean(score) ≈ 0.079

When b = -2.54: sigmoid(-2.54) = 0.073 (bulk of scores)
When b = -2.54 AND W·x > 5: sigmoid(2.46) = 0.92 (rare confident predictions)

The bias is the single point of failure: it determines the baseline score
for ALL classes. The per-class weights W are secondary.
```

### Why This Is Different from the RF1 Failure

| Dimension | RF1 Gradient Sparsity (SOLVED) | RF2 Bias Equilibrium (UNSOLVED) |
|-----------|--------------------------------|----------------------------------|
| Root cause | 16 positive / 2.76M anchors → no backbone update | Bias drifts to background equilibrium |
| Gradient flow | Backbone receives ~0 gradient | Backbone gradient healthy (from head_pose) |
| cls_std | 0.88 (never learned differentiation) | **0.0068** (learned THEN lost differentiation) |
| Head pose | DEAD (Kendall bug) | ALIVE, MAE 71.67°→47.84° |
| Bias trajectory | Stable at -4.6 (init) | Drifts from -4.6 → -2.54 |
| Time to collapse | Immediate (step 0) | Gradual (10+ epochs of apparent progress) |

### The 5 Fix Proposals Currently Ranked

| Option | Approach | Risk | Code Change |
|--------|----------|------|-------------|
| A | pi=0.1 bias init (was 0.01) | Low — RetinaNet literature supports | 1 line in config |
| B | Remove classification bias | Low — forces weight-based differentiation | 1 line in model.py |
| C | Quality Focal Loss | Medium — eliminates bias entirely | Code change in losses.py |
| D | Varifocal Loss | Medium — asymmetric learning | Code change in losses.py |
| E | Dedicated bias LR (DET_BIAS_LR_FACTOR=5.0 already in code) | Low — keeps current training | Already in config |

---

## 4. Three Distinct Failure Modes Discovered

The project has progressively revealed **3 independent failure modes**:

### Failure 1: Empty Frame Normalization (SOLVED — RC-28)

Empty frames contributed their full ~4.15M-element negative focal mass divided
by `num_pos=1` (~85% of frames, 30:1 gradient domination). Fix: skip empty
frames, normalize by GT-bearing image count.

### Failure 2: Gradient Sparsity (SOLVED — head_pose + Kendall fix)

Detection-only training with 172K anchors/image and pi=0.01 init produces
~4×10⁻⁵ gradient per backbone parameter per step — below FP32 noise floor.
Fix: enable head_pose (dense per-frame angular MAE gradient, confirmed by
Kendall bug fix).

### Failure 3: cls_score Bias Equilibrium (UNSOLVED)

Even with healthy backbone gradient, the classification head's bias parameter
drifts to a value (~-2.54) where sigmoid produces ~0.076-0.079 for most classes.
This is a **thin-head collapse** — the 595K-parameter classification subnet
converges to a fixed point where the bias dominates and per-class weights are
irrelevant.

---

## 5. What We Know vs What We Were Wrong About

### Proven Hypotheses (5)

1. **Gradient sparsity kills detection-only training** — 16/2.76M positive
   anchors produce insufficient gradient for 28M backbone params
2. **Head_pose dense gradient enables backbone updates** — Kendall fix
   confirmed: head_pose went from NO_GRAD to ALIVE, cls_std 1.6× broader
3. **R2.5 Paradox resolved** — multi-task gradient density, not architecture,
   enabled R2.5 to train while RF1 died
4. **Focal Loss can train this architecture** — R2.5 produced healthy
   training, proving the loss formulation is workable
5. **Eval pipeline needs 5 guards for single-head training modes** —
   TRAIN_ACT/PSR guards prevent 3 crash types

### Refuted Hypotheses (4)

1. "Run 8 proved architectural collapse" — Wrong. It proved unfixed
   normalization (RC-28) reproduces the same failure
2. "LR reduction = fix" — All retry strategies reducing LR make gradient
   sparsity WORSE. Higher LR might help but risks instability
3. "Head_pose + Kendall fix = complete solution" — Wrong. RF2 epoch 15
   collapse proved the bias equilibrium is independent of gradient supply
4. "DETACH_REG_FPN = main cause of RF1 failure" — Wrong. Classification
   gradient already flows; removing DETACH_REG_FPN adds regression gradient
   but doesn't resolve classification head internal dynamics

### Still Unknown (Core Open Questions)

- **Q01**: Why collapse AGAIN at epoch 15 with everything fixed?
- **Q02**: stage_history 0.45 vs metric_history 0.184 — which is real?
- **Q03**: Has PSR EVER trained? (Loss=1.546e-08 constant across ALL runs)
- **Q04**: Is cls_score bias the single point of failure?
- **Q05**: Is Focal Loss fundamentally wrong for 172K anchors?
- **Q13**: Is head_pose actually learning anything useful (MAE 47.84° ≈ near-random)?

Full list: 24 questions in `33_OPEN_QUESTIONS.md`.

---

## 6. Key Questions for Opus

### Q1: Which Fix for the cls_score Bias Equilibrium?

We have 5 ranked proposals in Appendix A of `26_RF1_RF10_COMPREHENSIVE_STATUS.md`:
pi=0.1 init, remove bias, QFL, Varifocal Loss, bias-specific LR (already coded
as DET_BIAS_LR_FACTOR=5.0). **Which one does Opus recommend?**

**Key constraint**: The fix must work on a single RTX 3060 with 12GB VRAM.
The model has 53M trainable params. We can't afford architectural overhauls.

### Q2: Is QFL/VFL the Right Long-Term Choice?

Quality Focal Loss (QFL) eliminates the bias parameter by predicting IoU score
instead of binary class presence. This would fundamentally change the detection
head's output space. **Is this the right direction for the paper?** The paper
targets SOTA on IndustReal — does QFL or VFL appear in comparable assembly
understanding papers?

### Q3: Why Does RF2 Eat the RF1 Checkpoint?

If RF1 stage_history says `best_det_mAP50=0.45`, then RF2 continuing from
`rf1/best.pth` should START near 0.45 and improve. Instead RF2 peaks at 0.184
and collapses to 0.001. **Does adding head_pose training in RF2 cause
catastrophic forgetting of detection?** Or is the 0.45 value an artifact?

### Q4: Has PSR Ever Produced a Non-Zero Gradient?

PSR loss = 1.546e-08 across ALL configurations. The causal transformer
produces extreme logits (min=-23, max=+22). Sigmoid saturates → gradient = 0.
**Should we investigate PSR architecture or defer to RF4+ stage?** If PSR
is fundamentally broken, it undermines the entire paper's novelty claim.

### Q5: Is HeadPose Actually Helping or Hurting?

Head pose MAE improved from 71.67° to 47.84° — but 47.84° on 9-DoF predictions
is near-random (random sphere has ~57° MAE). **If head_pose is just predicting
empirical mean, its gradient may be feature-smoothing — washing out variance
that detection needs.** Should we disable head_pose for detection stages?

### Q6: What Are We Missing?

After 5 rounds of consultation (v1–v5 Opus answers) and 34 files of analysis,
**what blind spot persists?** The pattern: each round reveals a deeper failure
mode. RF1: gradient sparsity. RF2 epoch 15: bias equilibrium. **What's the
next hidden failure mode?**

### Q7: Should We Skip to RF3 (All Heads)?

If RF2 is stuck at the bias equilibrium, adding activity head (dense per-frame
75-class gradient) might provide the feature diversity needed to break the
classifier out of its fixed point. **Or does RF3 inherit the RF2 checkpoint's
collapsed detection head and just make everything worse?**

### Q8: The Gradient Leakage Question

Disabled heads (train_head=False) show small gradient norms (0.02-0.05).
Is this leakage consuming gradient that should go to enabled heads? Or is
it negligible at <1% of active head gradient?

### Q9: Is the 20-Agent Swarm Overkill?

The swarm found 6 bugs immediately, proving its value. But is 22 agents/134
checks sustainable? **What's the minimum viable monitoring set?**

### Q10: Dataset Label Quality

The IndustReal detection labels are synthetic projections, not hand-annotated.
**Could label noise explain both the 0.184 ceiling and the drift toward
uniform predictions?** If the synthetic labels have systematic biases (missing
objects, class confusion), the classifier might converge to a "safe" uniform
prediction that minimizes expected loss under label noise.

---

## 7. Key Files Reference (34 files)

| File | Content | Priority |
|------|---------|----------|
| `00_JOURNEY_AND_STATUS.md` | Full project timeline Phases 1–12 | High |
| `03_ARCHITECTURE_DEEP_DIVE.md` | Architecture details | Medium |
| `13_OPUS_ANSWER_v4.md` | RC-28/RC-29 diagnosis | High |
| `14_POST_OPUS_V4_IMPLEMENTATION.md` | Phase 10–12: RF1 death spiral, Kendall fix, RF2 collapse | High |
| `17_OPUS_ANSWER_v5.md` | Ultimate path (R1→R4 ladder) | High |
| `18_ULTIMATE_MASTER_GUIDE_INDUSTREAL.md` | Full master guide | Medium |
| `19_PRE_TRAINING_READINESS_AUDIT_100.md` | 100-item audit | Reference |
| `22_FINAL_PREFLIGHT_GAP_CLOSURE.md` | Pre-flight closure | Medium |
| `26_RF1_RF10_COMPREHENSIVE_STATUS.md` | Stage definitions, RF2 metrics, 5 fix proposals | **Highest** |
| `28_DET_DEATH_SPIRAL_FIX_AND_RUNBOOK.md` | Bounded background loss fix | High |
| `29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md` | Gradient sparsity proof + RF2 postscript | **Highest** |
| `30_OPUS_MASTER_PROMPT_v7.md` | Previous master prompt (v7) | Reference |
| `31_KENDALL_BUG_DISCOVERY_AND_FIX.md` | Kendall bug + RF2 cross-validation | **Highest** |
| `33_OPEN_QUESTIONS.md` | 24 open questions by severity | **Highest** |
| `34_RF2_SWARM_MONITOR.md` | 20-agent monitoring swarm | High |
| `popw_paper_improved.tex` | Target paper | Reference |

---

## 8. Critical Config Reference

### Current RF2 Training (running now, PID 1043628)

```python
# === RF2 config (stage_rf2) ===
'batch_size': 4, 'grad_accum_steps': 8, 'effective_batch': 32,
'mixed_precision': False,
'train_det': True, 'train_act': False, 'train_psr': False,
'train_head_pose': True, 'train_pose': True,
'detach_reg_fpn': False,  # was True — regression gradient flows
'detach_psr_fpn': True,
'subset_ratio': 0.35, 'max_epochs': 30,  # was 15
'DET_GT_FRAME_FRACTION': 0.90,
'DET_OHEM_RATIO': 2.0, 'DET_OHEM_MIN_NEG': 32, 'DET_GAMMA_NEG': 1.5,
'DET_BIAS_LR_FACTOR': 5.0,  # bias-specific LR (coded but not verified effective)
'SOFT_ARGMAX_TEMPERATURE': 0.1,
'SOFT_ARGMAX_TEMP_TRAIN': 1.0,  # gradient flow fix
'POSE_LOSS_WEIGHT': 5.0,
'use_randaugment': False,  # disabled for RF1; should re-evaluate for RF2+
'use_spatial_aug': False,
```

### Key Config Changes Since v7

| Parameter | Old Value | New Value | Why |
|-----------|-----------|-----------|-----|
| `DET_OHEM_RATIO` | 1.0 → 5.0 → **2.0** | 2.0 | 5:1 was too aggressive → suppressed all predictions |
| `DET_OHEM_MIN_NEG` | 16 → 128 → **32** | 32 | 128 dominated gradient in low-pos batches |
| `DET_GAMMA_NEG` | 2.0 → 1.0 → **1.5** | 1.5 | 1.0 gave 13.5× negative increase — excessive |
| `DET_BIAS_LR_FACTOR` | — | 5.0 | Bias-specific LR to escape equilibrium (new, untested) |
| `POSE_LOSS_WEIGHT` | 0.01 | 5.0 | Compensated for [0,1] coordinate normalization |
| `SOFT_ARGMAX_TEMP_TRAIN` | — | 1.0 | Prevent soft-argmax gradient vanishing |
| `DET_METRICS_EVERY_N` | 5 | 1 | Eval every epoch |
| RF1 aug | True | False | RandAugment + spatial aug disabled for RF1/2 |

---

## 9. Current Training Snapshot

```
PID:     1043628
Stage:   rf2 (stage_index=1)
Epoch:   15/30 (50%)
Status:  RUNNING (detection collapsed, head_pose still improving)
GPU:     ~8.2GB / 12GB (stable)
Speed:   ~0.9 batch/s, ~1241 batches/epoch
Swarm:   22 agents active, 134 checks/cycle, 5-min interval
Heartbeat: every 50 batches via _write_stage_heartbeat()

det_mAP50 trajectory:
  ep 7: 0.007 → ep 8: 0.184 (PEAK) → ep 9: 0.181 →
  ep 10: 0.159 → ep 13: 0.000010 → ep 15: 0.001 (COLLAPSED)

Head pose MAE trajectory:
  ep 7: 71.67° → ep 9: 63.65° → ep 11: 56.61° →
  ep 13: 55.73° → ep 15: 47.84° (IMPROVING)

EVAL COLLAPSE: 56 occurrences at epoch 15
Checklist status: ALL 5 FAILED (gate, health, convergence, validation, stability)
Best checkpoint: epoch 8 (det_mAP50=0.184)
```

---

## Appendix: Quick Dict of File Numbers

| File | Topic |
|------|-------|
| 00 | Full timeline Phases 1–12 |
| 10 | Opus v2 answer |
| 12 | Master prompt v4 |
| 13 | Opus v4 answer |
| 14 | Post-Opus v4 implementation (extended through RF2) |
| 15 | Git diff summary |
| 16 | Master prompt v5 |
| 17 | Opus v5 answer |
| 18 | Ultimate master guide |
| 19 | 100-item audit |
| 22 | Pre-flight gap closure |
| 26 | RF1-RF10 stage definitions + RF2 data + fix proposals |
| 28 | Death spiral fix + runbook |
| 29 | Gradient sparsity proof + RF2 postscript |
| 30 | Master prompt v7 |
| 31 | Kendall bug + RF2 cross-validation |
| 33 | 24 open questions |
| 34 | 20-agent swarm documentation |
| 35 | **This file** |

---

*Generated 2026-06-20 as the master overview prompt for Opus consultation v8.
Accompanies files 00–35 in the analyses/consult_2026_06_10 directory.
Send this file alongside the full directory for fastest Opus onboarding.*
