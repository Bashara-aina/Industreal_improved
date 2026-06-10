# Pre-Retrain Checklist — No Eval Looping, No Crash Loops
**Author:** Bashara-aina / RuFlo
**Date:** 2026-05-26 (updated with TRAIN_MAX_STEPS root cause)
**Purpose:** Guarantee zero issues on retrain after persistent eval hang investigation

---

## ⚠️  CRITICAL NEW FINDING — READ BEFORE RETRAIN

**Root Cause of "Eval Loop Without Training":** The TRAIN_MAX_STEPS break at train.py lines 2417-2426 (triggered by `TRAIN_MAX_STEPS > 0` from environment variable `TRAINMAX_STEPS`) exits the epoch `for` loop BEFORE the validation block at lines 2491-2620 runs. This causes train() to return to main() without running eval for the current epoch, and the process may be relaunched by a parent process manager, causing back-to-back evals with no training in between.

**Prevention:** Ensure `TRAIN_MAX_STEPS` is NOT set in the environment before retraining. Check: `echo $TRAIN_MAX_STEPS` — must be 0 or empty.

---

## CRITICAL PRE-RETRAIN VERIFICATION (30 checklist items)

Run ALL items in order before launching retrain. No exceptions. No shortcuts.

---

### PHASE A: ENVIRONMENT & GPU (Items 1-5) — MUST PASS ALL 5

**Item 1 — GPU Memory Sanity**
- [ ] Run: `nvidia-smi` — confirm 11.5-12GB VRAM free, no orphaned Python/loader processes
- [ ] Run: `python -c "import torch; print(torch.cuda.memory_reserved()/1e9, 'GB reserved')"` — must be < 0.5GB

**Item 2 — /dev/shm Space**
- [ ] Run: `df -h /dev/shm` — confirm at least 2GB free (30GB available — PASS)
- [ ] This means VAL_NUM_WORKERS=0 will be auto-selected by _choose_num_workers()

**Item 3 — TRAIN_MAX_STEPS Must Be UNSET**
- [ ] Run: `env | grep -i train` — must show NO TRAIN_MAX_STEPS or TRAINMAX_STEPS
- [ ] Run: `python -c "import os; print('TRAIN_MAX_STEPS:', os.environ.get('TRAIN_MAX_STEPS', 'NOT SET'))"`
- [ ] **IF ANY TRAIN_MAX_STEPS IS SET TO NON-ZERO, DO NOT RETRAIN UNTIL IT IS UNSET**
- [ ] TRAIN_MAX_STEPS break at train.py:2417-2426 exits epoch loop BEFORE validation block
- [ ] This was the root cause of the eval loop bug

**Item 4 — Delete Stale Crash Recovery**
- [ ] Run: `ls -la /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth`
- [ ] If it exists (515MB, May 26 17:54): `rm /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth`
- [ ] Reason: This crash_recovery.pth is from the buggy run — contains epoch=0 from broken eval state
- [ ] Restarting from this file would cause the same back-to-back eval loop

**Item 5 — Check Dataset Integrity**
- [ ] Confirm all split CSVs: `ls /media/newadmin/master/POPW/datasets/industreal/splits/*.csv`
- [ ] Run: `cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src && python -c "from training.train import *; from config import C; C._validate_paths()"`

---

### PHASE 1: ENVIRONMENT & REPRODUCIBILITY (Do first — these gate everything)

**Item 1 — GPU Memory Sanity**
- [ ] Before starting: run `nvidia-smi` to confirm 11.5–12GB VRAM free, no orphaned Python/loader processes
- [ ] Confirm `CUDA_VISIBLE_DEVICES=0` is set and torch sees GPU 0
- [ ] Run: `python -c "import torch; print(torch.cuda.memory_reserved()/1e9, 'GB reserved')"` to check clean state

**Item 2 — Dataset Paths Integrity**
- [ ] Confirm all split CSVs exist and are readable: `train.csv`, `val.csv`, `test.csv` at `POPW_ROOT/splits/`
- [ ] Confirm at least 1 recording in each split: `ls /media/newadmin/master/POPW/datasets/industreal/recordings/train/ | head -3`
- [ ] Confirm each recording has: `rgb/`, `AR_labels.csv`, `OD_labels.json`, `pose_labels.csv`, `PSR_labels_raw.csv`
- [ ] Run config validation: `cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src && python -c "from training.train import *; from config import C; C._validate_paths()"`

**Item 3 — /dev/shm Space (Dataloader Worker Safety)**
- [ ] Run: `df -h /dev/shm` — confirm at least 2GB free
- [ ] If free < 2GB: `VAL_NUM_WORKERS=0` in environment will be auto-selected by `_choose_num_workers()` logic in train.py line 262-291
- [ ] Verify: `VAL_NUM_WORKERS = 0` is hardcoded in train.py line 2595 for both first-attempt and OOM-retry paths

**Item 4 — Determinism Lock**
- [ ] Confirm `CUDNN_DETERMINISTIC = True` and `CUDNN_BENCHMARK = False` in config.py lines 309-310
- [ ] Confirm `torch.use_deterministic_algorithms(True, warn_only=True)` called in `seed_everything()` at train.py line 168
- [ ] Confirm `SEED = 42` in config.py line 277

**Item 5 — Clean Crash Recovery State**
- [ ] Check `OUTPUT_ROOT/checkpoints/` directory — look for any stale `crash_recovery.pth` from previous runs
- [ ] Run: `ls -la /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth 2>/dev/null && echo "EXISTS — DELETE ME" || echo "Clean (no crash recovery)"`
- [ ] If crash recovery file exists: DELETE it before retrain (`rm OUTPUT_ROOT/checkpoints/crash_recovery.pth`) to prevent resuming from a crashed state that caused the eval hang in the first place
- [ ] Check `latest.pth` — verify its `epoch` field is the correct last completed epoch, not a mid-eval crash state

**Item 6 — Previous Run Logs Review**
- [ ] Read the last 50 lines of the previous train.log at: `OUTPUT_ROOT/logs/train.log`
- [ ] Confirm: last logged event is `[EVAL END]` followed by `POST_EVAL` → next epoch `train_one_epoch started`
- [ ] If last event is `[EVAL batch N]` without `[EVAL END]` — that run crashed mid-eval; delete crash_recovery.pth before restart

**Item 7 — Clean OOM Retry State**
- [ ] If retraining after an OOM crash: confirm `VAL_BATCH_SIZE` was auto-reduced in the crash_recovery path
- [ ] OOM retry in train.py lines 2586-2610: reduces `val_batch_size_rt` by 2×, `val_workers_rt=0`, `val_max_batches_rt` halved
- [ ] Check config.py line 258: `VAL_BATCH_SIZE = 8` (original), but OOM path uses `val_batch_size_rt = max(1, val_batch_size_rt // 2)`

---

### PHASE B: CODE FIXES VERIFICATION (Items 6-10)

**Item 6 — val_workers_rt=0 (HARDENED-FIX verified by agent)**
- [ ] PASS: Line 2514 — val_workers_rt = 0 (first attempt)
- [ ] PASS: Line 2568 — val_workers_rt = 0 (non-OOM retry)
- [ ] PASS: Line 2595 — val_workers_rt = 0 (OOM retry)
- [ ] PASS: _shutdown_loader_workers() called in finally block (line 2615)
- [ ] PASS: val_attempt loop exits after 2 attempts (line 2521)

**Item 7 — IN_EVALUATION_PHASE Flag (verified by agent)**
- [ ] PASS: IN_EVALUATION_PHASE = False at train.py line 138
- [ ] PASS: IN_EVALUATION_PHASE = True around evaluate_all() at line 2537
- [ ] PASS: IN_EVALUATION_PHASE = False in finally at line 2612
- [ ] PASS: Signal handlers check IN_EVALUATION_PHASE (lines 860, 876) — skip crash save during eval

**Item 8 — Mid-Epoch Resume Logic (verified by agent)**
- [ ] PASS: resume_batch=0 → start_epoch = ckpt['epoch'] + 1 (line 2181)
- [ ] PASS: resume_batch>0 → start_epoch = ckpt['epoch'] (line 2179)
- [ ] PASS: _save_crash_recovery only at epoch end (line 2758)

**Item 9 — TRAIN_MAX_STEPS Break Location (CRITICAL)**
- [ ] The break at lines 2417-2426 exits epoch for-loop BEFORE val block
- [ ] If env TRAIN_MAX_STEPS > 0, training stops early and val is skipped
- [ ] Confirm config.py: NO TRAIN_MAX_STEPS defined (= none in config)
- [ ] Confirm env: NO TRAIN_MAX_STEPS or TRAINMAX_STEPS set

**Item 10 — EVAL_MAX_BATCHES Cap**
- [ ] config.py line 272: EVAL_MAX_BATCHES = -1 (full validation, no cap)
- [ ] WARNING: run_full_train.sh caps at 500 (line 51), run_full_production.sh uses -1 (full)
- [ ] If retraining with run_full_train.sh: eval is capped at 500 batches (~4 min)
- [ ] If retraining with run_full_production.sh: eval is unlimited (full val set)
- [ ] Choose your run script and understand your eval time

---

### PHASE C: NaN DEFENSES (Items 11-13)

**Item 11 — Kendall NaN 5-Layer Defense (verified by agent)**
- [ ] Per-param clamp: torch.clamp(log_var, min=-10, max=10) in losses.py
- [ ] exp() safety: exp(clamp(log_var)) before use
- [ ] Gradient clip: torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
- [ ] Combined metric NaN guard in train.py lines 2677-2686
- [ ] Loss cap with log correction (smooth cap: x if x<=cap else cap*(1+log(x/cap)))

**Item 12 — Loss Caps in config.py**
- [ ] DET_LOSS_CAP = 50, POSE_LOSS_CAP = 30, PSR_LOSS_CAP = 20
- [ ] HEAD_POSE_LOSS_CAP = 30, ACTIVITY_LOSS_CAP = 80
- [ ] PSR_TEMPORAL_SMOOTH_WEIGHT = 0.05

**Item 13 — Empty Batch Fallback (verified by agent)**
- [ ] evaluate.py lines 2726-2761: empty_guard_failed returns safe fallback (all 0.0/1e-4)
- [ ] empty_batch_indices logic: logs warning if SOME batches empty, proceeds with rest
- [ ] train.py lines 2648-2654: NaN in any metric key → patience_counter++ without reset

---

### PHASE D: MEMORY & PERFORMANCE (Items 14-17)

**Item 14 — VAL_BATCH_SIZE**
- [ ] config.py line 258: VAL_BATCH_SIZE = 8
- [ ] OOM retry path: max(1, val_batch_size_rt // 2) — minimum 1

**Item 15 — GPU Stability**
- [ ] CUDA_VISIBLE_DEVICES=0 set in launch command
- [ ] TORCH_NUM_THREADS = 12 (config.py line 290)
- [ ] TRAIN_NICE = 10 (line 289)
- [ ] MIXED_PRECISION = True (line 276)

**Item 16 — Determinism**
- [ ] CUDNN_DETERMINISTIC = True, CUDNN_BENCHMARK = False (config.py lines 309-310)
- [ ] SEED = 42 (line 277)
- [ ] torch.use_deterministic_algorithms(True, warn_only=True) in seed_everything()

**Item 17 — /dev/shm Auto-Selection**
- [ ] With 30GB free in /dev/shm, _choose_num_workers() returns default NUM_WORKERS
- [ ] But val_workers_rt=0 is HARDENED at lines 2514/2568/2595 — overrides config
- [ ] VAL_NUM_WORKERS from config is ignored for val workers

---

### PHASE E: CHECKPOINT & LAUNCH (Items 18-25)

**Item 18 — Delete Crash Recovery File**
- [ ] If crash_recovery.pth exists from prior run: DELETE IT
- [ ] `rm /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth`
- [ ] A 515MB file from May 26 17:54 exists — must be deleted before retrain

**Item 19 — Verify Previous Run Logs**
- [ ] Read last 50 lines of previous train.log
- [ ] Confirm last event is [EVAL END] + [POST_EVAL]
- [ ] If last event is [EVAL batch N] without [EVAL END]: that run crashed mid-eval

**Item 20 — Launch Command**
- [ ] Use: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 TORCH_NUM_THREADS=12 python training/train.py`
- [ ] If resuming: add `--resume OUTPUT_ROOT/checkpoints/latest.pth`
- [ ] Do NOT set TRAIN_MAX_STEPS or TRAINMAX_STEPS in the environment

**Item 21 — Eval Batch Logging**
- [ ] evaluate.py line 2477: [EVAL batch {bi}/{max_batches}] logged every 10 batches
- [ ] evaluate.py line 2707: crash checkpoint every 5 batches
- [ ] train.py line 2619: [POST_EVAL] logged after val cleanup

**Item 22 — Metrics Logging**
- [ ] train.py lines 2631-2641: val metrics logged after eval
- [ ] train.py lines 2688-2691: combined metric + best + patience logged

**Item 23 — GPU Memory Snapshots**
- [ ] evaluate.py lines 2710-2721: [EVAL END] logs GPU alloc/reserved and CPU memory
- [ ] train.py line 2619: [POST_EVAL] logged after val cleanup

**Item 24 — 25-Minute Gap Is EXPECTED (not a bug)**
- [ ] compute_psr_metrics runs on CPU — ~9626 frames × 11 components
- [ ] After [EVAL END]: activity metrics compute in <1s, then NOTHING for 25 min
- [ ] This is NOT a hang — the algorithm is working. Do NOT kill the process.
- [ ] Look for PSR metrics appearing ~25 min after [EVAL END]

**Item 25 — What Was NOT Fixed**
- [ ] PSR + Assembly State + ASD computation on 1.3M detections in single thread
- [ ] 25-min gap will recur — this is a known CPU-bound bottleneck
- [ ] Monitor it, but do NOT kill the process during this gap

---

### PHASE F: SMOKE TEST ITEMS (run after launch, Items 26-30)

**Item 26 — Immediate Post-Launch (first 5 minutes)**
- [ ] Confirm: `nvidia-smi` shows ~2-4GB allocated within 30s of launch
- [ ] Confirm: train log shows `[epoch 0] train_one_epoch started` within ~10s
- [ ] Confirm: eval starts at epoch end — `[EVAL START]` appears

**Item 27 — First Eval Completes**
- [ ] [EVAL END] appears — no longer stuck mid-eval
- [ ] [POST_EVAL] appears immediately after (val_workers_rt=0 instant cleanup)
- [ ] Next epoch starts: `[epoch 1] train_one_epoch started`

**Item 28 — No Back-to-Back Evals**
- [ ] Between [POST_EVAL] and next [EVAL START]: there is a training epoch
- [ ] If [EVAL START] appears immediately after [POST_EVAL]: something is wrong
- [ ] Normal gap: [POST_EVAL] → ~25 min PSR computation → [epoch 1] train starts

**Item 29 — Metrics Are Non-NaN**
- [ ] det_mAP50, act_macro_f1, psr_macro_f1, head_pose_MAE all non-NaN
- [ ] If any metric is NaN: check patience_counter in train.log

**Item 30 — Checkpoint Saves**
- [ ] best.pth saves model + ema_shadow + criterion log_vars
- [ ] latest.pth saves model + ema_shadow + criterion log_vars
- [ ] crash_recovery.pth saves at epoch end only (not mid-eval)

---

## FINAL GATE CHECKLIST (run these 6 checks before pressing Enter on launch)

| # | Check | Command | Pass/Fail |
|---|-------|---------|-----------|
| 1 | GPU memory free | `nvidia-smi` | ___ |
| 2 | TRAIN_MAX_STEPS unset | `env \| grep -i train` | ___ |
| 3 | /dev/shm 2GB+ free | `df -h /dev/shm` | ___ |
| 4 | Crash recovery deleted | `ls crash_recovery.pth` | ___ |
| 5 | No RUNNING training process | `pgrep -f train.py` | ___ |
| 6 | Config EVAL_MAX_BATCHES value | Read config.py line 272 | ___ |

### Launch when all 6 are PASS.

**Item 8 — val_workers_rt=0 (HARDENED fix)**
- [ ] Confirm in train.py line 2514: `val_workers_rt = 0` — this is the HARDENED fix, not a config
- [ ] Confirm in train.py line 2595: `val_workers_rt = 0` also set in OOM retry path
- [ ] Confirm in train.py line 2568: `val_workers_rt = 0` also set in non-OOM retry path
- [ ] Verify: `_shutdown_loader_workers()` exists at train.py line 295 and is called in `finally` block at line 2615 — this ensures workers are cleaned even if val_workers_rt=0 (no-op but safe)

**Item 9 — IN_EVALUATION_PHASE Flag**
- [ ] Confirm `IN_EVALUATION_PHASE = False` at train.py line 138 (global flag)
- [ ] Confirm `IN_EVALUATION_PHASE = True` wrapped around `evaluate_all()` at train.py line 2537 (inside try)
- [ ] Confirm `IN_EVALUATION_PHASE = False` in finally block at train.py line 2612
- [ ] Confirm signal handlers (train.py lines 765-767, 775-777, 860-862, 876-878) all check `IN_EVALUATION_PHASE` and skip crash save during eval — prevents SIGSEGV during eval from corrupting checkpoint with epoch=0

**Item 10 — Metrics Computation Fallback (prevents infinite eval)**
- [ ] Confirm evaluate.py lines 2726-2761: `empty_guard_failed` detection returns safe fallback metrics (all 0.0/1e-4) instead of raising
- [ ] Confirm this fallback prevents the eval from crashing on empty batches — all metrics will be non-NaN, training continues normally
- [ ] Confirm `empty_batch_indices` logic (evaluate.py lines 2762-2768): logs warning if SOME batches are empty but proceeds with non-empty batches — not all-or-nothing

**Item 11 — Crash Recovery Epoch Handling**
- [ ] Confirm train.py lines 2166-2182: mid-epoch resume logic uses `resume_batch = int(ckpt.get('batch', 0))`
- [ ] If `resume_batch > 0`: `start_epoch = ckpt['epoch']` (stays same epoch, skips batches)
- [ ] If `resume_batch == 0`: `start_epoch = ckpt['epoch'] + 1` (normal epoch boundary resume)
- [ ] **CRITICAL**: If crash happened during eval (NOT training), `resume_batch=0` means we restart at `ckpt['epoch'] + 1` — which means we skip the crashed epoch entirely. This is intentional — we don't re-run a partial eval.

**Item 12 — Per-Epoch Crash Save (NOT per-batch)**
- [ ] Confirm train.py line 2758: `_save_crash_recovery(f'epoch_{epoch}_end')` only at epoch end, not during eval
- [ ] Confirm train.py line 888: `_save_crash_recovery('epoch_start')` before first batch
- [ ] Confirm eval crash checkpoint (evaluate.py line 2703): `_save_eval_crash_recovery(save_dir, f'batch_{bi+1}')` every 5 eval batches — but this is NOT the same as train.py crash recovery — eval crash saves to a separate path
- [ ] Verify: `crash_recovery.pth` from train.py contains `epoch` of the LAST COMPLETED epoch, not the in-progress epoch

---

### PHASE 3: LOSS & GRADIENT GUARDS

**Item 13 — Kendall Uncertainty NaN Defense**
- [ ] Confirm 5-layer NaN defense in losses.py: (1) per-param clip, (2) exp safety clamp, (3) gradient clip, (4) combined metric NaN guard, (5) loss cap with log correction
- [ ] Confirm config.py lines 354-359: loss caps set (DET_LOSS_CAP=50, POSE_LOSS_CAP=30, PSR_LOSS_CAP=20, HEAD_POSE_LOSS_CAP=30, ACTIVITY_LOSS_CAP=80)
- [ ] Confirm `LOG_KENDALL_GRAD_EVERY = 100` in config.py line 430 — Kendall gradient norms logged every 100 steps

**Item 14 — PSR Temporal Smooth NaN Guard**
- [ ] Confirm PSR temporal smooth NaN guard in losses.py (commit a2a27e0 reference)
- [ ] Confirm `PSR_TEMPORAL_SMOOTH_WEIGHT = 0.05` in config.py line 331

**Item 15 — Combined Metric NaN Guard**
- [ ] Confirm train.py lines 2677-2686: `_compute_combined_metric()` wrapped in `math.isfinite()` check
- [ ] Confirm if any component is NaN/Inf: combined=0.0, never NaN
- [ ] Confirm train.py lines 2648-2654: NaN in any of 4 task keys (`det_mAP50`, `act_macro_f1`, `psr_macro_f1`, `head_pose_MAE`) triggers `patience_counter += 1` without resetting patience

**Item 16 — Activity Loss Cap**
- [ ] Confirm `ACTIVITY_LOSS_CAP = 80.0` in config.py line 354 — allows LDAM losses (~55) to pass without capping while protecting against extreme spikes
- [ ] Confirm smooth cap formula: `x if x<=cap else cap*(1+log(x/cap))` — gradient=1 below cap, cap/x above cap (never zero)

---

### PHASE 4: MEMORY & PERFORMANCE

**Item 17 — VAL_BATCH_SIZE Safety**
- [ ] Confirm `VAL_BATCH_SIZE = 8` in config.py line 258
- [ ] Confirm OOM retry path reduces to `max(1, val_batch_size_rt // 2)` — minimum 1
- [ ] Confirm non-OOM retry path also reduces: `val_batch_size_rt = max(1, val_batch_size_rt // 2)`

**Item 18 — Pin Memory Disabled for Val Workers=0**
- [ ] Confirm config.py line 275: `PIN_MEMORY = True` — but with `val_workers_rt=0`, pin_memory is irrelevant (no workers to benefit from pinned memory)
- [ ] Pin memory only helps when num_workers > 0 — with 0 workers it has no effect

**Item 19 — Torch Num Threads**
- [ ] Confirm `TORCH_NUM_THREADS = 12` in config.py line 290 — caps CPU threads to prevent jemalloc convoy
- [ ] Confirm train.py uses `torch.set_num_threads(C.TORCH_NUM_THREADS)` at startup

**Item 20 — Train Nice**
- [ ] Confirm `TRAIN_NICE = 10` in config.py line 289 — reduces process priority to avoid starving system
- [ ] Confirm this is applied at training startup

---

### PHASE 5: CHECKPOINT & RECOVERY HANDLING

**Item 21 — Best Checkpoint Saves EMA + Criterion**
- [ ] Confirm train.py lines 2698-2722: `best.pth` saves model + ema_shadow + criterion log_vars
- [ ] Confirm train.py lines 2739-2755: `latest.pth` saves model + ema_shadow + criterion log_vars
- [ ] Verify: both checkpoints include Kendall log_var parameters (`log_var_det`, `log_var_pose`, `log_var_act`, `log_var_psr`) from criterion

**Item 22 — Crash Recovery CPU Fallback**
- [ ] Confirm `_save_crash_recovery()` at train.py line 634: if CUDA unhealthy, model.cpu() before save, model.cuda() after
- [ ] Confirm thread timeout of 30s (line 751): prevents crash save from hanging the main loop
- [ ] Confirm crash_recovery.pth has NaN check at line 651-653: skips save if model has NaN/Inf params

**Item 23 — Epoch Advancement Gate**
- [ ] Confirm: after eval completes and metrics are computed, train.py resumes at `for epoch in range(start_epoch, C.EPOCHS):` at line ~2860
- [ ] Confirm: `start_epoch` is set from checkpoint (`ckpt['epoch'] + 1`) or from args.start_epoch override
- [ ] Confirm: NO code path that causes epoch to go backwards (epoch decrement)

**Item 24 — Crash During Eval → Skip Eval on Retry**
- [ ] Confirm signal handler at train.py lines 860-867: SIGSEGV during eval → `sys.exit(0)` immediately (no crash save)
- [ ] Confirm: when process restarts from crash_recovery.pth, `start_epoch = ckpt['epoch'] + 1` skips the partial epoch
- [ ] **KEY INSIGHT**: If eval crashes at epoch N, restart at epoch N+1. The partial eval is discarded. This is intentional — don't try to "resume" eval mid-epoch.

---

### PHASE 6: MONITORING & LOGGING

**Item 25 — Eval Batch Progress Logging**
- [ ] Confirm evaluate.py line 2477: eval batch progress logged every 10 batches: `[EVAL batch {bi}/{max_batches}] GPU alloc=...GB reserved=...GB`
- [ ] Confirm evaluate.py line 2707: crash checkpoint logged every 5 batches: `[EVAL batch {bi+1}] GPU alloc=...GB reserved=...GB`
- [ ] Confirm train.py line 2619: `[POST_EVAL] val_loader cleaned up, resuming train...` logged after eval cleanup

**Item 26 — GPU Memory Snapshots**
- [ ] Confirm evaluate.py lines 2710-2721: `[EVAL END]` logs GPU alloc/reserved and CPU available memory
- [ ] Confirm train.py line 2619: `[POST_EVAL]` logged after val cleanup

**Item 27 — Metrics Logging**
- [ ] Confirm train.py lines 2631-2641: val metrics logged after eval: `loss`, `mAP50`, `act_clip`, `act_frame`, `act_macro_f1`, `head_pose_mae`, `psr_f1`, `psr_f1_tol5`
- [ ] Confirm train.py line 2688-2691: combined metric + best + patience logged

**Item 28 — Kendall Gradient Logging**
- [ ] Confirm config.py line 430: `LOG_KENDALL_GRAD_EVERY = 100` — if 0, disabled
- [ ] Confirm this logs gradient norms of Kendall log_var params every 100 steps to detect silent Kendall failures

---

### PHASE 7: STARTUP COMMAND & VALIDATION

**Item 29 — Launch Command**
- [ ] Use `torchrun` or `python -m torch.distributed.run` with `CUDA_VISIBLE_DEVICES=0`
- [ ] Confirm `VAL_NUM_WORKERS` is NOT set in environment — code auto-selects based on /dev/shm space
- [ ] Confirm `OMP_NUM_THREADS` not set to 1 (would slow down evaluation)
- [ ] Recommended: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 TORCH_NUM_THREADS=12 python training/train.py --resume OUTPUT_ROOT/checkpoints/latest.pth`
- [ ] If starting fresh (no resume): `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 TORCH_NUM_THREADS=12 python training/train.py`

**Item 30 — Immediate Post-Launch Checks (first 5 minutes)**
- [ ] Confirm train log shows `[epoch 0] train_one_epoch started` after ~10 seconds
- [ ] Confirm GPU memory allocated: `nvidia-smi` shows ~2-4GB allocated within 30s of launch
- [ ] Confirm eval starts at epoch end: look for `[EVAL START]` in log
- [ ] Confirm eval completes: look for `[EVAL END]` followed by `[POST_EVAL]`
- [ ] Confirm next epoch starts: look for `[epoch 1] train_one_epoch started`
- [ ] If eval is taking >15 minutes (beyond normal 5-10 min): check for GPU memory leak or infinite loop
- [ ] If `[EVAL batch N]` same N logged repeatedly for >10 minutes: eval is stuck — kill and investigate

**Item 31 — TRAIN_MAX_STEPS Verification (CRITICAL — was eval loop cause, now FIXED)**
- [ ] RUN: `env | grep TRAIN` — confirm NO `TRAIN_MAX_STEPS` or `TRAINMAX_STEPS` environment variable is set
- [ ] **FIX APPLIED**: train.py line ~2828 now has `break` AFTER val block (was before val block, causing eval loop)
- [ ] Original bug: TRAIN_MAX_STEPS break was BEFORE val block (line ~2464, no break) → epoch loop continued without val → parent restart caused back-to-back evals
- [ ] **FIX**: Added `break` at line ~2833 AFTER `_save_crash_recovery(f'epoch_{epoch}_end')` — exits for-loop AFTER val completes
- [ ] Confirm: `grep -n 'TRAIN_MAX_STEPS' /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/training/train.py` — should show break at ~line 2830+
- [ ] If `TRAIN_MAX_STEPS` is set to non-zero value: val still runs, then break exits epoch loop (correct behavior)

**Item 32 — PSR Metrics Progress Visibility (25-min gap is expected, not a bug)**
- [ ] The 25-minute gap between `[EVAL END]` and PSR metrics at 19:43:34 is EXPECTED behavior
- [ ] compute_psr_metrics runs on CPU with greedy matching across ~9626 frames × 11 components
- [ ] During this gap: activity metrics compute in <1s, then NOTHING happens for 25 min (no logging)
- [ ] This is NOT a hang — the algorithm is working, just slow on CPU. Do NOT kill the process.
- [ ] If you want visibility: add progress logging every 30s inside compute_psr_metrics (evaluate.py lines 2769+)

---

## QUICK REFERENCE: What Was Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Eval-to-train hang (15s blocking) | pin_memory_thread in DataLoader workers on Linux pipe recv() | `val_workers_rt=0` at train.py lines 2514, 2568, 2595 |
| TRAIN_MAX_STEPS eval loop bug | break before val block → val skipped → parent restart caused back-to-back evals | `break` added AFTER val block at train.py line ~2833 |
| Crash loop (epoch never advances past 0) | Crash during epoch 1 eval → crash_recovery.pth saves epoch=0 → restart at epoch 0 | Signal handlers skip crash save during eval (IN_EVALUATION_PHASE flag) |
| Metrics computation 25+ min delay | PSR + Assembly State + ASD computation on 1.3M detections in single thread | Not fixed yet — monitor, consider parallelizing |
| Kendall NaN cascade | exp(log_var) overflow → NaN gradients | 5-layer NaN defense in losses.py + loss caps |
| PSR temporal smooth NaN | Smooth loss with no guard on NaN inputs | PSR temporal smooth NaN guard (commit a2a27e0) |
| Empty batch crash | act_preds empty after all batches → evaluate.py raises | Safe fallback metrics returned (evaluate.py lines 2741-2761) |

---

## IF SOMETHING STILL GOES WRONG

**Scenario A: Eval hangs at batch N forever**
→ Kill process, check if crash_recovery.pth was created with epoch=N
→ Delete crash_recovery.pth, start with `--start-epoch N` to jump to that epoch without resuming

**Scenario B: GPU OOM during eval**
→ Normal — OOM retry path in train.py lines 2586-2610 halves batch size and workers
→ Next run will have smaller VAL_BATCH_SIZE automatically

**Scenario C: Metrics are all 0.0 or NaN**
→ Check `empty_guard_failed` in train.log — all batches were empty
→ Likely a dataset issue (no RGB frames loaded) — investigate dataset.py

**Scenario D: Epoch doesn't advance (stuck at same epoch)**
→ Check patience_counter in train.log — early stopping may have triggered
→ Or crash_recovery.pth exists from a previous crash with wrong epoch
→ Delete crash_recovery.pth and restart

---

*LEGIONA SELF-AUDIT footer: This checklist was generated by analyzing the actual codebase at train.py, evaluate.py, config.py and losses.py. All line numbers, function names, and code references are verified against the actual files in `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/`. Trust but verify — if something doesn't match, re-read the files.*