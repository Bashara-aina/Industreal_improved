# Preflight Checklist — 50 Items, 5 Layers

**Purpose:** Verify every load-bearing assumption before starting a training run.
**Target:** RF4+ (simple head, 5 tasks, no staged training subset)
**Audience:** Human operator running the launch command.

---

## Layer 1 — Hardware & Environment (Items 1–10)

### 1.1 GPU allocation
- [ ] **1. GPU 1 (RTX 5060 Ti 16GB) is free** — `nvidia-smi` shows <100 MB used, no stale training PIDs
- [ ] **2. GPU 0 (RTX 3060 12GB) is free for subprocess eval** — only needed if `USE_SUBPROCESS_EVAL=1`; otherwise may be used for det-only ablation in parallel
- [ ] **3. No stale PID lock** — check `OUTPUT_ROOT/logs/.train.pid`; if exists and process is dead, remove it
- [ ] **4. VRAM budget confirmed** — model + optim + data = ~6 GB on 5060 Ti. With gradient accumulation (batch 4 × accum 8 = effective 32), peak is ~7.5 GB. 16 GB is safe.

### 1.2 Environment variables
- [ ] **5. Do NOT set `CUDA_VISIBLE_DEVICES`** — CUDA reorders GPUs by compute capability. Index 0 = 5060 Ti (faster, training), index 1 = 3060 (slower, subprocess eval). train.py defaults to `cuda:0` which is the 5060 Ti. Subprocess eval sets `CUDA_VISIBLE_DEVICES=1` internally to isolate the 3060.
- [ ] **6. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — set in train.py line 6, verify no override
- [ ] **7. `CUBLAS_WORKSPACE_CONFIG=:4096:8`** — set in train.py line 11, verify no override
- [ ] **8. `OMP_NUM_THREADS=12`** — matches CPU core count; prevents thread oversubscription in DataLoader
- [ ] **9. `TORCH_NUM_THREADS=12`** — caps intra-op parallelism; too high wastes CPU cycles on 12-core system
- [ ] **10. `USE_SUBPROCESS_EVAL`** — set to `1` if GPU 0 is available and you want SIGKILL-safe validation; leave unset/`0` to use ThreadPoolExecutor (backward-compat path)

### 1.3 Disk & filesystem
- [ ] **11. Disk space ≥ 50 GB free** — `df -h .`. Each checkpoint is ~500 MB; with per-stage subdirs and crash_recovery, 50 GB is safe for 23+ epochs

---

## Layer 2 — Data Pipeline (Items 11–20)

### 2.1 Dataset files
- [ ] **12. Training CSV has no header row** — `head -1 train.csv` should be numeric data, not `recording_id,state_id,...`. Config has `names=['recording_id','state_id','activity','start_frame','end_frame']` in `train.py` DataFrame read.
- [ ] **13. Validation CSV has no header row** — same check as above
- [ ] **14. pose.csv forward vectors are un-normalized** — norm ~0.014–0.030. **Do NOT normalize them** — `head_pose_loss_split` already L2-normalizes. Normalization would be a redundant no-op.
- [ ] **15. Frame files exist for every row** — spot-check 3 random rows: `ls -la {root}/rgb/frame_{recording_id}_{state_id}_{frame:06d}.jpg` resolves

### 2.2 DataLoader configuration
- [ ] **16. `NUM_WORKERS=0`** — hard confirmed in config.py:400. Do not raise. Worker deadlocks were causing CUDA kernel hangs during validation.
- [ ] **17. `VAL_NUM_WORKERS=0`** — hard confirmed. Same rationale.
- [ ] **18. `RAM_CACHE_MAX_IMAGES=8000`** — confirmed in config.py:404. Full dataset (5,595 frames) fits in RAM at ~2.2 GB. 8,000 is safe.
- [ ] **19. `DET_GT_FRAME_FRACTION=0.4`** — confirmed for RF3-RF10 in config.py:1652. Only raise to 0.9 for RF1/RF2 (det-only stages). Opus says keep 0.4 — raising helps detection at the expense of activity (fragile).
- [ ] **20. `BATCH_SIZE`** — base config: BATCH_SIZE=2, GRAD_ACCUM_STEPS=8, effective=16. **RF4 preset overrides to batch_size=4**, effective=32. Stage-managed runs use preset values (batch=4, accum=8, effective=32). Standalone runs use base values (batch=2, effective=16). Verify which path is active.

### 2.3 Class balance checks
- [ ] **21. Activity class imbalance understood** — 46/72 classes have <1% support. CB-balanced CE + label smoothing is active. The diversity gate (Item 41) will confirm whether the simple head escapes the majority-class attractor.
- [ ] **22. `_weights[0]` is NOT zeroed** — confirmed in losses.py: action_id=0 is `take_short_brace` (63 frames), not NA. The old line `_weights[0] = 0.0` was removed.

---

## Layer 3 — Model Architecture (Items 21–30)

### 3.1 Backbone & neck
- [ ] **23. `BACKBONE='convnext_tiny'`** — confirmed in config.py:103. Config variable is `BACKBONE`, not `BACKBONE_NAME`. Read by train.py:3353 as `getattr(C, 'BACKBONE', 'resnet50')`. Correct for consumer GPU: 28M params, ~4.6 GFLOPs.
- [ ] **24. FPN neck active** — P3-P7 feature maps at 256 channels. No modifications from base.
- [ ] **25. Feature bank is BYPASSED in non-staged mode** — `model.py:2193-2198` returns `None` when `STAGED_TRAINING=False`. The bank's recording-id-keyed buffer was feeding non-temporal sequences to TCN+ViT, causing majority-class collapse.

### 3.2 Activity head
- [ ] **26. `ACTIVITY_HEAD_SIMPLE=True`** — confirmed in config.py:687. Bypasses TCN+ViT entirely.
- [ ] **27. Simple head architecture correct** — `LayerNorm→Linear(512→256)→GELU→Dropout(0.2)→Linear(256→75)` at `model.py:1377`. Xavier init, logit bias=-0.5. 150K params, short gradient path.
- [ ] **28. Logit bias=-0.5** — confirmed in `model.py:1391`. Discourages majority-class collapse by penalizing the zero-logit initial state.
- [ ] **29. `ACTIVITY_LR_MULTIPLIER=1.0`** — confirmed in config.py:668. No longer needs 20× — gradient flows freely with the simple head.
- [ ] **30. Gradient centralization active for activity head** — both AMP path (`train.py:1280`) and FP32 path (`train.py:1730`). Subtracts mean gradient per parameter.

### 3.3 Detection head
- [ ] **31. `DET_LR_MULTIPLIER=1.0`** — confirmed. Opus says keep at 1.0; gradient contention is reduced by the simple head.

### 3.4 Kendall uncertainty weighting
- [ ] **32. `USE_KENDALL=True`** — confirmed. Four log-var params (det, pose, act, psr) learned per Kendall et al. Bounds prevent complete suppression of any single task.
- [ ] **33. Kendall log-var bounds are per-variable** — config.py:714-716 has `KENDALL_LOG_VAR_MIN_ACT=-0.5`, `KENDALL_LOG_VAR_MAX_PSR=0.0`, `KENDALL_LOG_VAR_MAX_POSE=3.0`. Read by losses.py:1659-1661. NOT a single bounds pair. Each task has independent range.

---

## Layer 4 — Training Loop & Stability (Items 31–40)

### 4.1 Optimizer & scheduler
- [ ] **34. `OneCycleLR` with `pct_start=0.1`** — confirmed in train.py:3593. 10% warmup is correct for a 150K MLP; longer warmup is for large from-scratch modules.
- [ ] **35. `ACTIVITY_PARAM_GROUP_IDX=3, PSR_PARAM_GROUP_IDX=4`** — confirmed. Activity and PSR have separate param groups; `ACTIVITY_LR_MULTIPLIER` controls activity group.
- [ ] **36. `GRAD_CLIP=1.0`** — confirmed in config. Global gradient norm clipping. `ACTIVITY_HEAD_GRAD_CLIP` also = 1.0 (removed the old 0.3 bottleneck).
- [ ] **37. `WARMUP_EPOCHS=2`** — confirmed in config.py:388. Two epochs of linear warmup before OneCycleLR starts.

### 4.2 Checkpointing
- [ ] **38. Pre-val checkpoint active** — `train.py:4343` saves `latest.pth` after training, before validation. A validation crash loses at most the current epoch's training batch (recovered on resume).
- [ ] **39. Per-stage checkpoint subdirectories** — `train.py:3018-3023` creates `checkpoints/{stage_name}/` subdirectory when `_STAGE_NAME` env var is set by stage manager. Ensures RF4_best.pth is not overwritten by RF5.
- [ ] **40. Crash recovery auto-load** — `train.py` now auto-loads `crash_recovery.pth` when its mtime > `latest.pth` mtime. Never promotes to `best.pth`.

### 4.3 Crash hardening
- [ ] **41. Watchdog with PID verification** — `train.py:3927` checks heartbeat file for our PID only. Kills only on our process's stale heartbeat, not another training run's.
- [ ] **42. PID lock file** — `train.py:3071` writes `.train.pid` to log dir. Blocks duplicate training processes on same GPU.
- [ ] **43. `IN_EVALUATION_PHASE` flag** — set to `True` during validation, `False` otherwise. Prevents signal handler (e.g., SIGINT during eval) from corrupting checkpoint save.
- [ ] **44. `_cuda_is_healthy()` check** — `evaluate.py:3011` checks CUDA context health before saving eval crash recovery. CPU-fallback if CUDA is corrupt.

### 4.4 Subprocess evaluation
- [ ] **45. `USE_SUBPROCESS_EVAL` wiring** — If `USE_SUBPROCESS_EVAL=1`: after EMA swap, `val_subprocess.pth` is saved; validation calls `run_val_subprocess()` which forks on `CUDA_VISIBLE_DEVICES=0`. Child can be SIGKILL'd without corrupting parent CUDA context. Falls back to `latest.pth` if `val_subprocess.pth` missing.
- [ ] **46. Subprocess timeout** — `SUBPROCESS_EVAL_TIMEOUT=900` (15 min). Subprocess sends warning at 450 s. After 900 s: SIGKILL.

---

## Layer 5 — Monitoring & Paper Readiness (Items 41–50)

### 5.1 Real-time metrics
- [ ] **47. Diversity/entropy instrumentation active** — `evaluate.py` now logs `[DIVERSITY] pred_distinct={N}/74 entropy={X.XXX} nats gt_distinct={N}/74` every epoch. Go/no-go gate: `pred_distinct ≥ 15` AND `entropy ≥ 1.5 nats` AND `macro-F1 > 0.01`.

### 5.2 Validation cadence
- [ ] **48. `DET_METRICS_EVERY_N=1`** — full detection mAP every epoch. Can raise to 3 for faster epochs, but for RF4-5 we want per-epoch signal.
- [ ] **49. `GATE_EVAL_MAX_BATCHES=200`** — caps fast-gate eval on non-full-det epochs. 200 batches × 4 batch size = 800 frames ~ 10 min on 5060 Ti.

### 5.3 Paper data collection
- [ ] **50. Launch command recorded** — train.py saves `run_command.txt` with full argv + relevant env vars to `OUTPUT_ROOT/logs/`. Verify after 10 batches.

---

## Launch Command Template

### With subprocess eval (both GPUs):
```bash
USE_SUBPROCESS_EVAL=1 \
SUBPROCESS_EVAL_TIMEOUT=900 \
OMP_NUM_THREADS=12 \
TORCH_NUM_THREADS=12 \
python src/training/train.py
```

### Without subprocess eval (GPU 0 used for ablation):
```bash
OMP_NUM_THREADS=12 \
TORCH_NUM_THREADS=12 \
python src/training/train.py
```

### Resuming from crash:
```bash
OMP_NUM_THREADS=12 \
TORCH_NUM_THREADS=12 \
python src/training/train.py --resume OUTPUT_ROOT/checkpoints/latest.pth
```

### Det-only baseline on GPU 0 (for paper §4 ablation):
```bash
CUDA_VISIBLE_DEVICES=0 \
TRAIN_ACT=0 TRAIN_PSR=0 TRAIN_HEAD_POSE=0 \
OMP_NUM_THREADS=12 \
TORCH_NUM_THREADS=12 \
python src/training/train.py --max-epochs 2
```

---

## Quick Go/No-Go After Epoch 1

| Metric | Pass | Fail |
|--------|------|------|
| `pred_distinct` (classes) | ≥ 15 | ≤ 3 |
| `entropy` (nats) | ≥ 1.5 | ≤ 0.5 |
| `act_macro_f1` | > 0.01 | ≤ 0.001 |

If all three pass → training is on track. If all three fail → the simple head also collapsed. In that case, switch activity loss to class-balanced focal loss and/or raise head dropout to 0.3.
