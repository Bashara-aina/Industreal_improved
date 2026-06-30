# 68: Master Prompt — Final Push to RF10 and AAIML Paper [2026-06-30]

## Who We Are

Training a multi-task assembly verification model (5 tasks, ConvNeXt-Tiny backbone,
FPN neck) on the IndustReal dataset for an AAIML 2027 Tokyo paper (IEEE Xplore,
deadline Oct 10, 2026). Running on RTX 5060 Ti 16GB (GPU 1) with a second
RTX 3060 12GB (GPU 0) sitting idle.

## What's Changed Since v12 (File 59)

### What Opus Proved Wrong (files 62-63, committed + pushed)

1. **"312x gradient gap" was a measurement artifact.** The probe
   (`_log_per_head_grad_norm`, `train.py:2345-2383`) logs FIRST and LAST individual
   parameter grad-norms, not head totals. "activity=0.010" is
   `proj_features.weight` only. "psr=3.180" is psr_head's first param. Not comparable.
2. **Gradient "invariance" is not a bug.** LR can't move a fixed-state gradient;
   blend ratio only affects backbone gradient; reinit only touched classifier.
   None of attempts 2-6 (10 days) could change that number.
3. **Feature bank is NOT detaching the gradient.** `proj_features.weight` reads
   ALIVE[0.0102] — if the bank severed the graph, that would be DEAD. Confirmed
   by `model.py:1241` (slot -1 keeps live gradient).
4. **Real root cause:** Shuffled sampler + FeatureBank = non-temporal sequences →
   TCN+ViT learns noise over 3.7k frames → majority-class collapse. The data
   (46/72 classes <1%) is the binding constraint.

### What We Fixed This Session

| Fix | File | What Changed |
|-----|------|-------------|
| ACTIVITY_HEAD_SIMPLE | config.py:687 | Bypass TCN+ViT, use 150K MLP: LayerNorm→Linear(512→256)→GELU→Dropout→Linear(256→75) |
| Logit bias=-0.5 | model.py:1391 | Init logit bias to -0.5 to discourage majority collapse |
| FEATURE_BANK bypass | model.py:2193-2198 | When STAGED_TRAINING=False, pass None (use expand path) |
| ACTIVITY_LR_MULTIPLIER=1.0 | config.py:668 | Reset from 20x — gradient flows now, no multiplier needed |
| RAM_CACHE_MAX_IMAGES=8000 | config.py:404 | Full dataset (5,595 frames) fits in RAM at ~2.2 GB |
| NUM_WORKERS=0 | config.py:400 | Eliminates DataLoader deadlocks |
| Gradient centralization | train.py:1280,1730 | Removes common-mode gradient from activity head |
| Watchdog with PID check | train.py:3927 | Kills only on our PID's stale heartbeat |
| Pre-val checkpoint | train.py:4343 | latest.pth saved after training, before validation |

## Current Training

```
PID: 3618126
Stage: RF4 (50% data, all 5 heads, simple head active)
Epoch: 3/23 (batch ~150/3469, ~48 min/epoch)
GPU: RTX 5060 Ti 16GB, 100% util, ~6GB VRAM
Speed: 1.2 batch/s → ~48 min/epoch
Optimizer: AdamW, act=1x (5e-4), OneCycleLR
Checkpoint: best.pth epoch=2 (pre-simple-head), simple_classifier freshly init'd
First validation with simple head: ~40 min from now
```

## Three Critical Problems That Are NOT Solved

### Problem 1: CUDA Kernel Hangs During Validation (HIGHEST PRIORITY)

5 crashes in 12 hours. ThreadPoolExecutor (1200s timeout, `train.py:4404-4416`)
cannot interrupt CUDA kernels. SIGALRM in evaluate_all (`evaluate.py:3568`)
silently fails when called from threads.

**We need Opus's design for subprocess evaluation (SIGKILL).** File 64 details
three options. Which architecture is best?

### Problem 2: Head Pose GT Not Normalized (MEDIUM PRIORITY)

Our best paper number (8.71° angular MAE) uses GT forward vectors with norm
0.014-0.030 instead of 1.0 (`_parse_pose` warning every run). The eval normalizes
before computing angle (so 8.71° IS valid) but training MSE on short vectors is
suboptimal. File 65 details the fix — normalize at data-load time.

### Problem 3: Paper Framing (HIGHEST PRIORITY FOR WRITING)

Opus says: **Stop comparing to single-task SOTA.** Reframe as "multi-task system
on consumer GPU + failure analysis." File 66 has a complete abstract draft and
structure. We need Opus to validate the new framing before we write.

## What We Need From Opus: 10 Final Decisions

### Architecture & Training (1-4)
1. **Is the simple head (LayerNorm→Linear→GELU→Dropout→Linear) the correct
   architecture for per-frame activity?** Or should we also try: (a) deeper MLP,
   (b) residual connections, (c) adding det_conf as a separate path?
2. **What validation metric tells us "simple head works"?** act_macro_f1 > 0.01?
   Distinct predicted classes > 10? Prediction entropy > threshold? We need a
   go/no-go criterion we can evaluate in the first epoch.
3. **Should we raise DET_GT_FRAME_FRACTION from 0.4 to 0.6 for RF4?** With the
   simple head, detection no longer competes with TCN+ViT gradient. More GT
   frames per batch could accelerate detection.
4. **What is the optimal OneCycleLR pct_start for the simple head?** Currently
   0.1 (10% of training). With a freshly init'd MLP (Xavier, bias=-0.5), should
   it be 0.3 (longer warmup)?

### Infrastructure (5-7)
5. **Subprocess eval design:** Option A (full subprocess), B (per-batch sync),
   or C (hybrid)? We need concrete code we can implement in 2 hours.
6. **Auto-load crash_recovery.pth:** Yes/no? If yes, should we prefer it on
   resume when mtime is newer than best.pth? Risk: crash_recovery saves
   mid-epoch (pre-validation), so loaded weights haven't been validated.
7. **GPU 0 (RTX 3060):** Opus says skip DDP. But could we use a simple
   `torch.cuda.device` split for batch parallelism without DDP?
   Model uses 6GB on GPU 1. GPU 0 has 12GB free.

### Paper (8-10)
8. **Does the reframed abstract in file 66 work for AAIML EIC/PC track?** The ML
   hook is: consumer GPU → on-device processing → no cloud dependency → worker
   privacy (IEEE 7005-2021). Is that sufficient for the EIC track?
9. **Head pose disclosure:** If normalization validates 8.71°, do we need to
   disclose the pre-normalization value? Or just report the validated number?
10. **Single-task comparison:** Do we run YOLOv8m or ConvNeXt+det-only for
    2 epochs as baseline? This strengthens "multi-task trade-offs" claim but
    takes 2 hours of GPU time. Worth it?

## File Summary for Opus

Read this one first (68), then the specific file for each problem:

| File | Topic | Opus Needs To |
|------|-------|---------------|
| 64 | Subprocess evaluation design | Design the SIGKILL architecture |
| 65 | Head pose data normalization | Verify the fix approach |
| 66 | Paper reframing for AAIML | Validate abstract + structure |
| 67 | RF10 training roadmap | 10 strategic decisions on schedule/tuning |
| **68 (this)** | Master prompt | All of the above |

## Git History (files 62-68 committed to main)

```
18e0160 fix: apply Opus-recommended simple_classifier init in __init__ (logit bias=-0.5)
8207632 fix: root cause — bypass feature bank in non-staged mode, fix in-place gradient severing
ea325d2 chore: raise RAM_CACHE_MAX_IMAGES to 8000 (full dataset fits in RAM)
(62, 63 merged from claude/rf10-opus-overview-xdj395)
```

## Hardware Available

- GPU 0: RTX 3060 12GB — IDLE (can be used for ablations or batch splitting)
- GPU 1: RTX 5060 Ti 16GB — TRAINING (100% util, ~6GB VRAM)
- CPU: 12 cores
- RAM: 32GB (20GB free during training)
- Disk: 1.2TB free on HDD
- Dataset: ~1.8 GB (3,667 train frames + 1,928 val frames)
