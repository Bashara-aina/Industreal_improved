# 61: Overview Prompt for Opus [2026-06-30]

## Companion Files (56-60) — Read These First for Full Context

This overview is a high-level summary. The detailed analysis lives in 5 companion files that must be read alongside this one:

| # | File | What It Covers |
|:-|------|----------------|
| **56** | `56_ACTIVITY_HEAD_COLLAPSE_ROOT_CAUSE.md` | Deep root cause of the activity head collapse — 6 failed fix attempts, invariant gradient norm (0.010), architecture graph, gradient path tracing through ViT → TCN → feature_bank, 8 diagnostic questions for Opus |
| **57** | `57_MULTI_TASK_GRADIENT_IMBALANCE.md` | The 312x gradient gap between PSR and activity heads, head-by-head gradient ratios, Kendall weighting limits, PSR oscillation (500-step DEAD/ALIVE cycle), per-head architecture analysis, data long-tail (46/72 classes with <1%), 7 questions |
| **58** | `58_INFRASTRUCTURE_STABILITY_AND_VALIDATION.md` | 5 process crashes in 12 hours, all 6 safety fixes deployed, remaining CUDA hang vulnerability (ThreadPoolExecutor cannot interrupt kernels), GPU 0 (RTX 3060) completely idle at 0% util, HDD bottleneck at 1.2 batch/s, 8 questions |
| **59** | `59_MASTER_PROMPT_V12_FOR_OPUS.md` | The master prompt: complete structured context including all hyperparameter changes, stage history, gate targets, paper target gaps, training state, and what we need from Opus |
| **60** | `60_STRUCTURAL_QUESTIONS_FOR_OPUS.md` | **31 questions across 4 domains**: (1-12) activity head collapse, (13-19) multi-task architecture, (20-25) infrastructure, (26-31) paper targets and feasibility |

**Reading order for Opus:** Start with this overview (61), then 56 (root cause), 57 (gradient imbalance), 58 (infrastructure), 59 (master prompt), and conclude with the 31 questions in 60.

---

## Who We Are

Training a multi-task assembly verification model on the IndustReal dataset for an AHFE 2026 Hawaii paper. 5 tasks (detection, activity recognition, PSR, head pose, body pose) on a single ConvNeXt-Tiny backbone with FPN, running on consumer GPUs (RTX 3060 + RTX 5060 Ti).

## The Problem in One Sentence

After 10 days of training across 6 full attempts with escalating fixes, the activity recognition head has NEVER produced non-trivial validation metrics (best act_macro_f1 = 0.0022, typically collapses to 1/75 classes), and we cannot complete 3 consecutive training epochs without a CUDA hang or process crash.

## Python Files Involved

| File | Lines | Role |
|------|------:|------|
| `src/config.py` | 1,749 | All hyperparameters including ACTIVITY_LR_MULTIPLIER, blend ratios, LR configs |
| `src/training/train.py` | 5,159 | Main training loop, optimizer setup, gradient clipping, heartbeat, watchdog, checkpointing |
| `src/training/losses.py` | 1,878 | MultiTaskLoss with Kendall uncertainty weighting, CB-balanced CE for activity, focal for det/PSR |
| `src/models/model.py` | 2,262 | Model architecture: ConvNeXt-Tiny backbone, FPN, 5 task heads, gradient blending |
| `src/training/stage_manager.py` | 3,274 | RF1-RF10 progressive stage pipeline, gate evaluation, retry strategies |
| `src/training/training_supervisor.py` | 868 | Cron-based monitor, symptom detection, auto-intervention |
| `src/data/industreal_dataset.py` | 1,688 | Dataset loader, frame cache, GT-frame oversampling |
| `src/evaluation/evaluate.py` | 4,489 | Validation pipeline, per-task metric computation, DET_PROBE |

## What We've Tried (Summary of 6 Failed Attempts)

All config changes are in `src/config.py`. All training loop changes are in `src/training/train.py`.

| Attempt | Key Changes | Result |
|---------|------------|--------|
| 1 (Jun 20) | Default config, 1x LR, blend=0.05 | act_macro_f1=0.0 (activity not trained in RF1) |
| 2 (Jun 28) | RF3, blend=0.10, clip=0.3 | act_macro_f1=0.0022 (effectively random) |
| 3 (Jun 29) | RF4, blend=0.70, clip=0.3 | Collapse: 4/75 classes |
| 4 (Jun 30) | blend=1.0, 3x LR, GC, separate param group | Collapse: 4/75 classes, class 12=87.5% |
| 5 (Jun 30) | blend=1.0, 10x LR, GC | Collapse: 4/75 classes, class 12=98% |
| 6 (Jun 30, NOW) | blend=1.0, **20x LR**, GC, **classifier reinit** | **AWAITING VALIDATION** |

## Current Training Run

```
PID:      3369770
Stage:    RF4 (50% data, all 5 heads)
Epoch:    3/23 (batch 3380/3469, ~2 min to epoch end)
GPU:      RTX 5060 Ti 16GB, 100% util, ~6GB VRAM
Speed:    1.3 batch/s, ~44 min/epoch
Optim:    AdamW, act=20x (1.0e-2 LR for activity_head)
Checkpoint: best.pth has reinitialized activity classifier weights
```

## Key Files Changed This Session (Jun 30)

### `src/config.py` changes
- `NUM_WORKERS = 0` — eliminated DataLoader deadlocks
- `VAL_NUM_WORKERS = 0` — same for validation
- `ACTIVITY_HEAD_GRAD_CLIP = 1.0` (was 0.3)
- `ACTIVITY_LR_MULTIPLIER = 20.0` (new, was 3.0 then 10.0)
- `ACTIVITY_GRAD_BLEND_RATIO = 1.00` (was 0.70)

### `src/training/train.py` changes
- Pre-val checkpoint: saves `latest.pth` after training, before validation
- Activity head split into its own param group (was shared with PSR)
- Gradient centralization for activity head (both AMP and FP32 paths)
- Training watchdog thread (kills process on 600s GPU heartbeat stall, PID-verified)
- CSV header fix for pandas read (`names=[...]`)
- OneCycleLR max_lr for all 8 param groups

### Checkpoint changes
- `best.pth` reinitialized: activity_head classifier LayerNorm→default, weight→xavier uniform, bias→0

## What We Need From You

### 1. Why does the activity head gradient norm = 0.010 across ALL configs?
This is the core mystery. Every head's gradient norm changes with LR and architecture choices — except activity. 0.010 is invariant. Is the gradient vanishing in ViT+TCN (designed for temporal sequences, fed single frames)? Or is feature_bank detaching the gradient?

### 2. Should we rip out ViT+TCN and use a simple MLP classifier?
8.2M params for proj_features → ViT → TCN → LayerNorm → Linear(512→75). If we replace with proj_features → Linear(512→75), we get ~0.5M params and a shorter gradient path. Risk: losing temporal modeling that we may need after Stage 3-10 with sequence data.

### 3. Can we validate without CUDA hangs?
5 crashes in 12 hours. ThreadPoolExecutor can't interrupt CUDA kernels. Options:
- Subprocess-based evaluation (fork+kill)
- Per-batch CUDA sync with timeout
- Every-N-epochs validation instead of every epoch

### 4. Should we use 2 GPUs?
GPU 0 (RTX 3060) sits idle. DDP could double throughput. Worth engineering time given other problems?

### 5. Are the paper targets realistic?
act_top1 target = 0.375 (paper §3.7.1). Current = 0.0. Gap = 37.5 points.
det_mAP50 target = 0.838 (YOLOv8m). Current = 0.053. Gap = 78.5 points.
psr_f1_at_t target = 0.731 (B2). Current = 0.0. Gap = 73.1 points.
head_pose target = 10° (est SOTA). Current = 8.71°. ONLY ONE ON TRACK.

### 6. What would you change first if you had 1 week?
Architecture? Loss formulation? Training procedure? Data augmentation? Multi-task optimization (PCGrad, CAGrad, Nash-MTL)?

## Key Metrics Snapshot

```
Best validation ever (epoch 2, old config):
  det_mAP50 = 0.053    det_mAP50_pc = 0.079
  act_macro_f1 = 0.002  (collapsed to 1 class)
  psr_f1_at_t = 0.000   (no signal)
  forward_MAE = 8.71°   (exceeds RF10 target)

Training loss (epoch 2):
  total=8.59  det=1.12  pose=1.70  act=0.60  psr=1.15

Gradient norms (ALL runs, step 0):
  activity_head = 0.010  ← INVARIANT
  detection     = 0.48   (48x)
  head_pose     = 0.56   (56x)
  pose          = 0.44   (44x)
  psr           = 3.18   (318x)

Dataset:
  36 train recordings, 16 val, 3,667 frames
  72 activity classes, 46 with <1% of data (long-tail)
  24 detection classes, 11 PSR components, 36 PSR steps
```

## Stage Gate Targets

| Stage | det_mAP50_pc | act_top1 | psr_f1_at_t | head_pose_MAE |
|-------|:-:|:-:|:-:|:-:|
| RF4 (current) | ≥0.20 | ≥0.06 | ≥0.05 | ≤65° |
| RF5 | ≥0.22 | ≥0.08 | ≥0.06 | ≤60° |
| RF6 | ≥0.24 | ≥0.10 | ≥0.08 | ≤55° |
| RF7 | ≥0.24 | ≥0.12 | ≥0.10 | ≤50° |
| RF8 | ≥0.26 | ≥0.14 | ≥0.12 | ≤45° |
| RF9 | ≥0.28 | ≥0.16 | ≥0.14 | ≤40° |
| RF10 | ≥0.30 | ≥0.18 | ≥0.16 | ≤35° |

Paper target act_top1 = 0.375 (not 0.18). Paper target det_mAP50 = 0.838 (not 0.30).
The gate targets are MILESTONES on the path to paper numbers, not the paper numbers themselves.

## What We've Ruled Out

- **Gradient clipping too aggressive**: activity gradient = 0.010, clip = 1.0 (100x too high to matter)
- **LR too low**: tested 0.5x, 1x, 3x, 10x, 20x — gradient norm unchanged at 0.010
- **Backbone gradient blocked**: blend_ratio tested at 0.05, 0.10, 0.30, 0.70, 1.00 — no effect
- **Kendall weights suppressing activity**: bounds set, weights equal at initialization
- **Data loader deadlocks**: NUM_WORKERS=0 fixed this
- **Bad checkpoint resume**: weights verified, classifier reinitialized this run
