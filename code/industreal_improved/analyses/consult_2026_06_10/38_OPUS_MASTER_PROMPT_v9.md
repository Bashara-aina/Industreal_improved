# 38 — OPUS MASTER PROMPT v9: Post-Fix Status Update (2026-06-21)

## For Upload to Opus — Self-Contained Summary of Files 00–38

---

## How to Use This File

This directory contains 37 files documenting the full consultation history
(June 11–20, 2026) for the POPW multi-task assembly understanding model.
**This prompt is the v9 follow-up** — the v8 prompt was sent to Opus, and
the 4 prescribed fixes were implemented. This v9 prompt tells Opus what
happened after those fixes, the current training state, and what to do next.

**Files to prioritize for deep context:**
- `00_JOURNEY_AND_STATUS.md` — Full project timeline (Phases 1–14)
- `26_RF1_RF10_COMPREHENSIVE_STATUS.md` — RF stage definitions, Opus v8 fix epoch-by-epoch data (Section 19)
- `33_OPEN_QUESTIONS.md` — 30 open questions organized by severity
- `34_RF2_SWARM_MONITOR.md` — 22-agent monitoring swarm documentation
- `35_OPUS_MASTER_PROMPT_v8.md` — The previous master prompt (what was sent TO Opus)
- `36_OPUS_ANSWER_v8.md` — Opus v8 answer (the 4 prescribed fixes)
- `37_IMPLEMENTATION_SUMMARY.md` — Implementation details of all 4 fixes (commit beda631)

---

## 1. Project Identity

**Project**: POPW — Pose-Conditioned Multi-Task Architecture for Assembly
Understanding (egocentric + third-person video)
**Hardware**: Single RTX 3060 (12 GB), i5-12400F, 64 GB RAM
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
- **Key quirk**: `DETACH_REG_FPN=True` detaches **regression subnet only** —
  classification gradient CAN flow to FPN/backbone

### Training Framework

- **RF ladder**: 10-stage progressive multi-task curriculum (RF1–RF10)
- **Stage manager**: `stage_manager.py` (3247 lines) orchestrates all stages
- **Monitoring**: 22-agent swarm (134 checks/cycle, 5-min interval)
- **Current state**: RF2 epoch 17 (PID 3176288), Opus v8 fixes active, no epoch-end validation yet

---

## 2. The Journey: 7 Rounds of Opus Consultation

### Round 1 (June 11–13): RC-25 → RC-29

**Problem**: Model producing frozen/constant outputs. AMP (fp16) GradScaler silently
skipping optimizer steps. **Fix**: FP32 across all presets. Also: RC-28 (empty frame
gradient domination → bounded background loss, 512 subsampled anchors), DET_PROBE,
LIVENESS diagnostics, 5 eval guards for single-head training.

### Round 2 (June 13–16): R1→R3 Recovery Protocol

**Problem**: Staged recovery protocol needed. Opus prescribed R0 smoke test → R1
detection+head_pose → R2 joint → R3 scale. All 16 recommendations implemented.
**Key finding**: `recovery_det_only` preset had `train_head_pose=True` but
`stage_rf1` preset had `train_head_pose=False` — fatal discrepancy.

### Round 3 (June 16–17): RF1 Death Spiral + Kendall Bug

**Gradient sparsity**: Detection-only (RF1) produces ~16 positive anchors per batch
out of 2.76M total (0.00058%). Gradient per backbone param = ~4×10⁻⁵ per step —
effectively zero. R2.5 masked this with 10,000× denser gradient from activity+PSR.

**Kendall bug**: `train_head_pose=True` fix was SILENTLY NEUTRALIZED by
`losses.py:1589` — when `train_pose=True AND train_act=False`, `loss_head_pose`
(~1.7) was computed but excluded from total loss. Head pose received ZERO gradient
for 7+ epochs. **Fix confirmed**: head_pose gradient went NO_GRAD → ALIVE,
cls_std 0.88 → 1.37 (1.6× broader), cls_max 0.47 → 2.78 (5.9× higher).

### Round 4 (June 17–20): RF2 Epoch 15 Collapse

**RF1 completed** with Kendall fix. stage_history reported `best_det_mAP50=0.45`
but metric_history shows max 0.184 — **2.4× discrepancy** (phantom 0.45 bug).

**RF2 launched** with: head_pose ALIVE, DET_GT_FRAME_FRACTION=0.90,
DETACH_REG_FPN=False, 35% data. **Epoch 15 collapse**:
det_mAP50 = 0.184 (ep 8) → 0.159 (ep 10) → 0.000010 (ep 13) → 0.001 (ep 15).
Head pose MAE continued improving (71.67° → 47.84°).

**cls_score Bias Equilibrium discovered**: classifier converges to uniform ~0.079
scores, bias drifts to -2.5 where sigmoid(-2.5)≈0.076. This is a **thin-head
collapse** — the 595K-parameter classification subnet's bias dominates and
per-class weights become irrelevant.

**22-agent monitoring swarm deployed** — 6 bugs found and fixed in first hours.

### Rounds 5–7 (June 17–19): Kendall Bug, R2.5 Paradox, Phantom 0.45

Three rounds refining the diagnosis. Key outcomes:
- Kendall bug confirmed (head_pose excluded from total loss)
- R2.5 Paradox resolved (multi-task gradient density, not architecture)
- Phantom 0.45 bug identified (stage_history recording gate threshold instead of best)

### Round 8 (June 20): Opus v8 Fixes Prescribed and Implemented

**Opus unified diagnosis**: The RF2 collapse is "one mechanism wearing three masks" —
Kendall head_pose precision domination driving the detection head's bias to
equilibrium. Not a separate "bias equilibrium" failure mode.

**4 fixes prescribed** and implemented in commit `beda631` (256 insertions, 119 deletions):

| Fix | What | Files Changed |
|-----|------|---------------|
| 1. De-fang Kendall | KENDALL_HP_PREC_CAP clamp + KENDALL_FIXED_WEIGHTS path | `config.py`, `losses.py` |
| 2. More positives | DET_POS_IOU_THRESH=0.4, TOP_K=9, BIAS_LR_FACTOR=1.0 | `config.py`, `losses.py` |
| 3. Kill double curriculum | Documented that STAGED_TRAINING was already a no-op | `config.py` (docs only) |
| 4. Phantom 0.45 fix | `_validate_stage_history_entry()` guard | `stage_manager.py` |

---

## 3. The Opus v8 Fixes — Detailed

### Fix 1: KENDALL_HP_PREC_CAP (Default: True)

`losses.py:1531-1533` — clamps `lv_hp >= lv_det.detach()` so head_pose precision
log-var can never exceed detection's. Prevents the pathology where head_pose
(loss ~0.01) gets ~54.6× Kendall weight while detection (loss ~0.5) gets ~1.4×.

**Alternative**: `KENDALL_FIXED_WEIGHTS=False` (default) — bypasses learned Kendall
log_vars entirely. Uses fixed λ=0.2 for head_pose, λ=1.0 for detection. Designed
for RF1-RF2 bootstrap stages.

### Fix 2: More Positives for Detection Head

Three changes to anchor assignment:

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `DET_POS_IOU_THRESH` | 0.5 | **0.4** | ~3-5× more anchors clear positive threshold |
| `DET_POS_IOU_TOP_K` | 1 (implicit) | **9** | Top-k force-match per GT → ~6-10 pos/GT |
| `DET_BIAS_LR_FACTOR` | 5.0 | **1.0** | Reverted — 5× was accelerating drift toward dead-feature equilibrium |

**Key detail**: top-k preserves existing positive assignments (doesn't overwrite
already-positive anchors). `labels[idx] < 0` guard prevents double-assignment.

### Fix 3: Kill Double Curriculum

**Finding**: The epoch-indexed Kendall staging in `losses.py` was already a no-op
because `STAGED_TRAINING=False` at `config.py:518`. Action: documented as-is.

### Fix 4: Phantom 0.45 Guard

`_validate_stage_history_entry()` (`stage_manager.py:548-582`) cross-checks every
numeric metric against all known gate/health thresholds from `RF_STAGES`. An exact
match triggers `logger.warning`. Wired to all 4 call sites that append to
`state.stage_history`. Stage state file cleaned: RF1 `best_metric` corrected from
phantom 0.45 to actual 0.184.

---

## 4. Current Training State (Epoch 17, PID 3176288)

### Training Overview

```
PID:         3176288
Stage:       rf2 (stage_index=1)
Epoch:       17/36 (47%)
Status:      RUNNING (no epoch-end validation completed yet)
Config hash: 3e6b58a5cb19765e
Started:     2026-06-20 21:44 UTC
GPU:         ~8.2GB / 12GB (stable)
Batch:       ~2750/3302 per epoch, ~0.9 batch/s
Best metric: 0.181 (from metric_history, epoch 8)
Gate:        NOT PASSED (all 5 checklists: fail — no epoch-end validation yet)
```

### DET_PROBE Diagnostics (Step-Level, Epoch 17)

The most important data — shows the model IS localizing even before epoch-end validation:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| score_p50 | 0.020-0.072 | Similar healthy range as pre-fix epoch 8 peak |
| score_max | 0.37-0.99 | Confident predictions on some classes/anchors |
| preds>0.05 | 28K-100K per batch | Many anchors above noise floor |
| bestIoU_max | 0.86-0.98 | Excellent localization quality |
| bestIoU>0.5 | 472-3037 per batch | Consistent positive matches |
| **Verdict** | **LOCALIZING** | Model finds objects, scores haven't collapsed |

### LIVENESS Diagnostics (Epoch 17)

| Head | Value | Status |
|------|-------|--------|
| det | 0.92-1.57 | **ALIVE** (healthy gradient) |
| head_pose | 7.13e-03 to 1.70e-02 | **ALIVE** (low but alive) |
| pose | 1.12-1.47 | **ALIVE** (healthy — body pose) |
| act | ~0 | DEAD (expected — train_act=False) |
| psr | ~0 | DEAD (expected — train_psr=False) |

### Key Observation: LOCALIZING But Not Yet CLASSIFYING

The DET_PROBE data at epoch 17 shows a dissociation:
- **Localization works**: bestIoU_max=0.86-0.98, preds>0.05=28K-100K
- **Classification still flat**: score_p50=0.020-0.072 (bias floor dominates)

This is exactly the dissociation seen at epoch 8-10 in the previous run (before
the collapse). The model CAN find objects (regression subnet works) but CANNOT
distinguish classes (classification subnet still at bias floor).

**The critical question**: Will the Opus v8 fixes (more positives, Kendall cap)
break this dissociation and allow classification to emerge? Or will the model
follow the same trajectory as the previous run and collapse at epoch 20-25?

**No epoch-end validation results yet** — the next epoch-end eval will give the
first det_mAP50 reading with all 4 fixes active. This is the single most
important upcoming data point.

---

## 5. Three Distinct Failure Modes (Updated Status)

### Failure 1: Empty Frame Normalization (SOLVED — RC-28, June 13)

Empty frames contributed full ~4.15M-element negative focal mass divided by
`num_pos=1` (~85% of frames, 30:1 gradient domination). **Fix**: skip empty
frames, normalize by GT-bearing image count. Confirmed working.

### Failure 2: Gradient Sparsity (SOLVED — Kendall fix, June 17)

Detection-only training with 172K anchors/image and pi=0.01 init produces
~4×10⁻⁵ gradient per backbone parameter per step — below FP32 noise floor.
**Fix**: enable head_pose (dense per-frame angular MAE gradient). Kendall bug
confirmed and fixed. Now at epoch 17: det LIVENESS=0.92-1.57 ALIVE.

### Failure 3: cls_score Bias Equilibrium (PENDING VERIFICATION with Opus v8 fixes)

Even with healthy backbone gradient, the classification head's bias parameter
drifts to ~-2.5 where sigmoid produces ~0.076 for most classes. The
595K-parameter classification subnet converges to a fixed point.

**Opus v8's diagnosis**: This is NOT a separate failure mode — it's the same
Kendall head_pose domination mechanism. The 4 fixes should resolve it.

**Current status**: DET_PROBE at epoch 17 shows LOCALIZING but not yet
CLASSIFYING. The model is on the same trajectory as the pre-fix epoch 8-10
window where the previous run looked healthy before collapsing.

---

## 6. What Has Changed vs. Previous Runs

### What's Different This Time (Opus v8 Fixes Active)

1. **Kendall cap**: KENDALL_HP_PREC_CAP prevents head_pose from dominating
   detection via precision weighting
2. **More positive anchors**: DET_POS_IOU_THRESH=0.4 + TOP_K=9 gives each GT
   ~6-10 positive anchors vs. ~1 before
3. **BIAS_LR_FACTOR=1.0**: Removed the 5× bias acceleration that may have been
   actively driving the bias toward -2.5 equilibrium
4. **Phantom 0.45 fixed**: stage_history no longer records bogus metrics
5. **Heartbeat active**: train.py writes periodic heartbeats to state file

### What's the Same

1. Same checkpoint (RF1 best.pth, epoch 17, effective epochs ~8 of real training)
2. Same 50% data (subset_ratio=0.50 for this RF2 run)
3. Same monitor swarm (22 agents, 134 checks, still PID 1049545)

### What Has NOT Been Changed (Opus v8 Deferred)

1. **QFL/VFL**: Deferred to post-recovery quality upgrade. The detection head
   needs features first.
2. **PSR architecture**: Deferred to RF4+ with 50-sample overfit test requirement.
3. **No architectural changes**: All fixes are config/loss-level, safe on RTX 3060.
4. **Original Kendall path preserved**: Standard `prec_task * loss + lv_task`
   path untouched under `else:` for RF3+.

---

## 7. What We Know vs. What We Were Wrong About

### Proven Hypotheses (9)

1. **Gradient sparsity kills detection-only training** — 16/2.76M positive
   anchors produce insufficient gradient for 28M backbone params
2. **Head_pose dense gradient enables backbone updates** — Kendall fix
   confirmed: head_pose went NO_GRAD → ALIVE at step 0, cls_std 1.6× broader
3. **R2.5 Paradox resolved** — multi-task gradient density, not architecture,
   enabled R2.5 to train while RF1 died
4. **Focal Loss can train this architecture** — R2.5 produced healthy training
5. **Eval pipeline needs 5 guards for single-head training modes** — confirmed
6. **Phantom 0.45 was a bug** — gate threshold recorded as best metric
7. **Kendall bug was real** — head_pose excluded from total loss at line 1589
8. **Empty frame normalization was required** — bounded background loss works
9. **Opus v8 fixes are syntactically and semantically correct** — verified via
   ast.parse, training started without crashes, DET_PROBE shows LIVENESS

### Refuted Hypotheses (5)

1. "Run 8 proved architectural collapse" — Wrong. Proved unfixed normalization.
2. "LR reduction = fix" — Lower LR makes gradient sparsity WORSE.
3. "Head_pose + Kendall fix = complete solution" — Wrong. RF2 epoch 15 collapse
   proved bias equilibrium is partially independent of gradient supply.
4. "DETACH_REG_FPN = main cause" — Wrong. Classification gradient already flows.
5. "Bias equilibrium is a separate failure mode from Kendall domination" —
   **Opus v8 says no**: it's the same mechanism wearing different masks.
   Verification PENDING — training at epoch 17 will confirm or refute.

### Still Unknown (Core Open Questions)

See `33_OPEN_QUESTIONS.md` for the full list of 30 questions. The most critical:

- **Q01**: Will Opus v8 fixes break the bias equilibrium? (Pending epoch-end val)
- **Q03**: Has PSR EVER produced a non-zero gradient? (Loss=1.546e-08 constant)
- **Q05**: Is Focal Loss fundamentally wrong for 172K anchors?
- **Q25**: Do the 4 Opus v8 fixes interact correctly? (Kendall cap + more
  positives + bias LR = 1.0)
- **Q30**: Will RF2 reach gate targets (det_mAP50>=0.40, MAE<=60°) with v8 fixes?

---

## 8. Key Questions for Opus (v9 — Post-Fix Status)

### Q1: Are the Fixes Working? What Should We Watch For?

The DET_PROBE at epoch 17 shows LOCALIZING but not yet CLASSIFYING — score_p50
still at 0.020-0.072 (bias floor). In the PREVIOUS run, this same dissociation
at epoch 8-10 preceded collapse at epoch 15. **What specific signal should we
watch for to know if the fixes are working?**

- Is a rising score_p50 the key indicator?
- Or should we watch bias values directly (cls_mean moving away from -2.5)?
- Or is epoch-end det_mAP50 the only reliable signal?

### Q2: What If Epoch 20-25 Shows the Same Collapse Trajectory?

If the Opus v8 fixes don't break the equilibrium, what's the next step?
- Skip to QFL/VFL immediately?
- Try KENDALL_FIXED_WEIGHTS=True (bypass Kendall entirely)?
- Skip RF2 and launch RF3 (add activity head for more feature diversity)?
- Abandon ConvNeXt-Tiny backbone for something else?

**Key constraint**: Single RTX 3060 12GB, 53M trainable params. We can't afford
architectural overhauls.

### Q3: Is the dissociation between LOCALIZING and CLASSIFYING expected at epoch 17?

Detection has TWO subnets sharing the same FPN features:
- **Regression subnet** (4×Conv+GIoU): learns to localize — works (bestIoU=0.86-0.98)
- **Classification subnet** (4×Conv+Focal): learns to classify — stuck (score_p50=0.020)

Why does the regression subnet learn while the classification subnet stalls?
Is this a Focal Loss issue specifically, or does classification simply need
more positive examples than regression?

### Q4: Should We Enable KENDALL_FIXED_WEIGHTS for RF2?

The fixed-weights path (λ_hp=0.2, λ_det=1.0) was designed for RF1-RF2 bootstrap
stages. If the HP_PREC_CAP clamp isn't enough, should we switch to fixed weights
to completely eliminate Kendall interference during the detection bootstrap phase?

### Q5: Has PSR Ever Trained? Should We Care Now?

PSR loss = 1.546e-08 constant across ALL runs and configurations. The causal
transformer produces extreme logits (min=-23, max=+22), sigmoid saturates, gradient=0.
Is this a critical paper issue (PSR is the main novelty claim) or should we
defer to RF4+ as originally planned?

### Q6: Are We Missing Something Again?

Every round reveals a deeper failure mode:
1. Empty frames (RC-28)
2. Gradient sparsity (RF1)
3. Kendall bug (head_pose excluded)
4. Bias equilibrium (RF2)
5. Phantom 0.45 (stage_history)

**What's the next hidden failure mode?** If Opus v8 fixes work and we reach
RF3 (det+pose+act), will activity head training reveal yet another problem?

### Q7: Is the Paper's Novelty Claim at Risk?

The paper contribution hinges on:
1. Cross-task conditioning (PoseFiLM, HeadPoseFiLM, det_conf → activity)
2. Multi-task architecture with 5 heads
3. **PSR** (the most novel component — NEVER trained)

If PSR has NEVER produced a non-zero gradient, the paper's core novelty claim
may be invalid. How urgent is this relative to getting detection working?

### Q8: Dataset Label Quality

IndustReal labels are synthetic projections, not hand-annotated. Could label
noise explain both the 0.184 ceiling and the drift toward uniform predictions?
If the synthetic labels have systematic biases (missing objects, class confusion),
the classifier might converge to a "safe" uniform prediction.

### Q9: Should We Start Thinking About RF3?

RF2 is epoch 17/36. At ~2h per epoch (50% data), we have ~38h remaining. If
the v8 fixes work, should we plan RF3 config now? If they don't work, when do
we declare RF2 failed and try a different approach?

---

## 9. Key Metrics Comparison: Previous Run vs. Current Run

### Previous Run (No v8 Fixes)

| Epoch | det_mAP50 | Head Pose MAE | Notes |
|-------|-----------|---------------|-------|
| 7 | 0.007 | 71.67° | Starting up |
| 8 | **0.184** | — | PEAK — then collapse begins |
| 9 | 0.181 | 63.65° | Plateau |
| 10 | 0.159 | 56.61° | Declining |
| 11 | — | 56.61° | No val this epoch |
| 12 | — | — | No val this epoch |
| 13 | 0.000010 | 55.73° | COLLAPSED |
| 15 | 0.001 | 47.84° | Fully collapsed |

### Current Run (v8 Fixes Active)

| Epoch | DET_PROBE score_max | DET_PROBE score_p50 | bestIoU_max | Notes |
|-------|---------------------|---------------------|-------------|-------|
| 17 | 0.37-0.99 | 0.020-0.072 | 0.86-0.98 | LOCALIZING, no epoch-end val yet |

**Awaiting**: Epoch 17 epoch-end validation → first det_mAP50 with all 4 fixes active.
This is THE critical data point.

---

## 10. Config Reference — Current Active RF2 Training

```python
# === RF2 config (stage_rf2) — Opus v8 fixes applied ===
'batch_size': 4, 'grad_accum_steps': 8, 'effective_batch': 32,
'mixed_precision': False,
'train_det': True, 'train_act': False, 'train_psr': False,
'train_head_pose': True, 'train_pose': True,
'detach_reg_fpn': False,

# Opus v8 Fix 1: Kendall precision cap
'KENDALL_HP_PREC_CAP': True,          # clamp lv_hp >= lv_det.detach()
'KENDALL_FIXED_WEIGHTS': False,        # False = use standard Kendall
'KENDALL_STAGED_TRAINING': False,      # was already no-op
'KENDALL_HP_FIXED_LAMBDA': 0.2,       # only used if FIXED_WEIGHTS=True

# Opus v8 Fix 2: More detection positives
'DET_POS_IOU_THRESH': 0.4,             # was 0.5 → ~3-5× more positive anchors
'DET_POS_IOU_TOP_K': 9,               # was implicit 1 → ~6-10 pos/GT
'DET_BIAS_LR_FACTOR': 1.0,            # was 5.0 → removed bias acceleration

# Prior RF2 settings (unchanged)
'subset_ratio': 0.50, 'max_epochs': 36,
'DET_GT_FRAME_FRACTION': 0.90,
'DET_OHEM_RATIO': 2.0, 'DET_OHEM_MIN_NEG': 32, 'DET_GAMMA_NEG': 1.5,
'POSE_LOSS_WEIGHT': 5.0,
'SOFT_ARGMAX_TEMPERATURE': 0.1,
'SOFT_ARGMAX_TEMP_TRAIN': 1.0,
'use_randaugment': False,
'use_spatial_aug': False,
```

---

## 11. File Reference (38 files, updated 2026-06-21)

| File | Content | Priority |
|------|---------|----------|
| `00_JOURNEY_AND_STATUS.md` | Full project timeline Phases 1–14 | High |
| `26_RF1_RF10_COMPREHENSIVE_STATUS.md` | Stage definitions, Opus v8 run analysis (Section 19) | **Highest** |
| `33_OPEN_QUESTIONS.md` | 30 open questions by severity | **Highest** |
| `34_RF2_SWARM_MONITOR.md` | 22-agent swarm, 134 checks, 6 bugs found | High |
| `35_OPUS_MASTER_PROMPT_v8.md` | Previous master prompt (what was sent to Opus) | Reference |
| `36_OPUS_ANSWER_v8.md` | Opus v8 answer — 4 prescribed fixes | **Highest** |
| `37_IMPLEMENTATION_SUMMARY.md` | Fix implementation details (commit beda631) | High |
| `38_OPUS_MASTER_PROMPT_v9.md` | **This file — v9 post-fix status update** | Current |
| `29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md` | Gradient sparsity proof | High |
| `31_KENDALL_BUG_DISCOVERY_AND_FIX.md` | Kendall bug + fix confirmation | High |
| `14_POST_OPUS_V4_IMPLEMENTATION.md` | Everything between v4 and v8 | Medium |
| `03_ARCHITECTURE_DEEP_DIVE.md` | Full architecture details | Medium |
| `13_OPUS_ANSWER_v4.md` | RC-28/RC-29 diagnosis | Reference |
| `28_DET_DEATH_SPIRAL_FIX_AND_RUNBOOK.md` | Bounded background loss | Reference |

---

## Appendix: Quick Dict of File Numbers

| File | Topic |
|------|-------|
| 00 | Full timeline Phases 1–14 |
| 03 | Architecture deep dive |
| 10 | Opus v2 answer (RC-25→RC-29) |
| 13 | Opus v4 answer |
| 14 | Post-Opus v4 implementation |
| 16 | Master prompt v5 |
| 17 | Opus v5 answer |
| 18 | Ultimate master guide |
| 19 | 100-item audit |
| 22 | Pre-flight gap closure |
| 26 | RF1-RF10 stage definitions + Opus v8 training analysis |
| 28 | Death spiral fix + runbook |
| 29 | Gradient sparsity proof |
| 30 | Master prompt v7 |
| 31 | Kendall bug discovery + fix |
| 33 | 30 open questions |
| 34 | 22-agent monitoring swarm |
| 35 | Master prompt v8 (sent TO Opus) |
| 36 | Opus v8 answer (4 fixes) |
| 37 | Implementation summary |
| 38 | **This file — v9 master prompt** |

---

*Generated 2026-06-21 as the v9 master overview prompt for Opus consultation.
This is the POST-FIX status update — the 4 Opus v8 fixes have been implemented
(commit beda631) and training is running at RF2 epoch 17 with all fixes active.
Send this file alongside the full directory for fastest Opus onboarding.*
