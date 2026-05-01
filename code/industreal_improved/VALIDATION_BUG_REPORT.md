# IndustReal POPW Training — Validation Failure Analysis

> **Date**: 2026-04-30
> **Goal**: Run full 60-epoch training of the IndustReal multi-task model (ASD detection + Head Pose + Activity + PSR + Assembly State + Error Verification) on synthetic POPW data.
> **Status**: Training crashes during validation after epoch 1. Root cause unknown. Needs expert debugging.

---

## 1. Problem Summary

After epoch 1 training completes successfully (~89 min), validation runs for **4-5 attempts** before the training process crashes with:

```
RuntimeError: Exceeded maximum validation retry attempts (4).
```

**Key symptom**: Every validation attempt produces the warning:
```
Detection evaluation skipped: no GT boxes found in this split.
ASD — mAP@0.5: nan  mAP@[0.5:0.95]: nan
```

The crash is **NOT** a CUDA OOM. The error happens inside `evaluate_all()` and is caught by the train.py retry loop, but the exception **never logs an ERROR message** — only the DEBUG `val_attempt` lines are logged before the final traceback.

---

## 2. What Was Working (Before Epoch 1 Crash)

### Training itself is healthy
- Epoch 1 completes in ~89 min (12,579 iterations)
- Speed: ~2.3-2.4 it/s
- Loss breakdown: det=0.01-0.02 (detection head learning), act=7-10, psr=0.05-0.07, pose=0.0-0.8 (frozen in stage 1)
- No NaN, no Inf, no CUDA errors during training
- GPU VRAM usage during training: ~8.6GB / 12GB

### Validation also runs — but crashes
- Validation loader processes 4000 batches (batch_size=4, num_workers=1)
- Each validation attempt takes ~185-209 seconds
- Partial metric output is produced every time:
  - Activity (Top-1, Top-5, Macro-F1)
  - Head Pose (MAE)
  - PSR (F1, F1@T, Edit Score, POS)
  - Assembly State (F1@1, Top-1, MAP@R+)
  - Error Verification (AP, F1, Precision, Recall)
  - Detection warning: `no GT boxes found`

### 17 benchmark metrics are all computed correctly
- ASD mAP@0.5 = NaN (expected: no GT boxes)
- Activity metrics = 0% (expected: frozen head in stage 1)
- PSR F1 = 0.0779 (still meaningful despite frozen head)
- All other task metrics = 0% (expected, frozen heads)

---

## 3. The Exact Crash Sequence

### First Run (PID 66596)
1. Epoch 1 training completes at 19:12:43
2. Validation attempt 1 starts → processes batches → crashes silently (no ERROR logged) → attempt 2 starts at 19:16:12
3. Attempt 2 → attempt 3 at 19:19:31 → attempt 4 at 19:22:36 → attempt 5 at 19:25:43
4. At 19:25:43: `RuntimeError: Exceeded maximum validation retry attempts (4)` — process becomes zombie
5. Cannot be killed with `kill -9`, procfs shows no data, but `ps` still shows process

### Second Run (PID 337175)
- Same crash pattern after epoch 1 completes
- Added `logger.exception()` to capture traceback — but process crashed again without printing exception details in logs
- Process also became zombie after the crash

### System state during crash
- GPU memory: only 2433 MiB / 12288 MiB used (plenty of headroom)
- No dmesg OOM killer entries
- No CUDA errors in log
- Training process transitions from `RNl` (running) to `SNl` (interruptible sleep) then to zombie

---

## 4. Why This Is NOT a Standard OOM

The validation retry loop in train.py has a strict exception filter:

```python
except Exception as exc:
    is_cpu_enomem = 'Cannot allocate memory' in str(exc)
    is_cuda_oom_v = _is_cuda_oom(exc)
    # ...
    if not (is_cpu_enomem or is_cuda_oom_v):
        logger.error(f'Non-OOM exception — aborting immediately.')
        raise   # <-- This would re-raise non-OOM exceptions immediately
```

If this filter were triggered, we would see the `ERROR | Non-OOM exception in evaluate_all — aborting train.py immediately.` message in the log. **This message never appears.** Yet the training crashes, meaning:

**Hypothesis**: The exception bypasses the `except Exception as exc` handler entirely.

Possible causes:
1. **SIGTERM / signal-based death**: The child worker processes (DataLoader workers) get killed by a signal, which raises `SignalException` — not a Python Exception
2. **Multiprocessing crash**: A worker dies via segfault, returning None or raising something not caught by `except Exception`
3. **System OOM killer**: But dmesg shows nothing, and GPU memory is fine

---

## 5. The "No GT Boxes Found" Warning

Every validation attempt logs this warning before crashing:
```
Detection evaluation skipped: no GT boxes found in this split.
ASD — mAP@0.5: nan  mAP@[0.5:0.95]: nan
```

This means `dg_boxes` (detection ground truth boxes) is empty for the val split. This is suspicious because:

1. The val split **does** have detection labels — we verified this by running `evaluate.py` standalone and it worked
2. The only place this warning is triggered is in evaluate.py line 1606:
   ```python
   if len(dg_boxes) == 0:
       logger.warning('Detection evaluation skipped: no GT boxes found in this split.')
   ```

3. **This warning appears BEFORE the crash**, meaning the crash happens during or after detection evaluation — not because detection evaluation was skipped

4. The warning is printed from within `evaluate_all()` at the very end, after all batches are processed. This means the crash occurs when trying to finalize detection metrics with empty ground truth.

---

## 6. Data Pipeline Investigation

### Dataset structure
- Train split: 25,158 samples (25 videos × ~1000 frames)
- Val split: 4000 batches (batch_size=4)
- Each sample contains: images, boxes, labels, head_pose, activity, psr_labels, assembly_labels, error_labels

### Possible data issue
The "no GT boxes found" warning suggests the val split's detection data might be:
1. Filtered out entirely during dataset loading (if detection boxes are empty/missing for val)
2. Getting cleared due to some data pipeline bug between train and val
3. A victim of multiprocessing state sharing across workers

### Key code path
```python
# In evaluate.py, val loader produces dg_boxes from detection ground truth
dg_boxes: List[np.ndarray] = []  # per-image box arrays

for batch_idx, (images, targets) in enumerate(loader):
    # ...
    detection_list = model.run_model(...)  # gets predictions
    for i in range(B):
        dg_boxes.append(detection_list[i]['boxes'].cpu().numpy())
        dg_labels.append(detection_list[i]['labels'].cpu().numpy())

if len(dg_boxes) == 0:
    logger.warning('Detection evaluation skipped: no GT boxes found...')
```

The fact that `dg_boxes` ends up empty suggests the detection ground truth data is not making it to the evaluation loop properly. This is the **most likely root cause**.

---

## 7. Solutions Already Tried

### Fix 1: `_prepare_images` (evaluate.py:195)
**Problem**: mean/std tensors were created on CPU then used on GPU without `.to(device)`
**Fix**: Changed `torch.tensor(IMAGENET_MEAN)` → `torch.tensor(IMAGENET_MEAN, device=device)`
**Result**: Verified correct — no impact on validation crash

### Fix 2: `run_model` device placement (evaluate.py:1343)
**Problem**: Model outputs moved to CPU before dtype normalization, causing dtype mismatches
**Fix**: Moved `.to(device)` before dtype normalization; added uint8→float normalization with IMAGENET_MEAN/STD for TTA crops/flips
**Result**: Verified correct — no impact on validation crash

### Fix 3: Flip TTA dimension guard (evaluate.py:1367)
**Problem**: `out_flip[key].dim() >= 3` was not checked, causing crashes on 2D psr_logits/activity tensors
**Fix**: Added `if out_flip[key].dim() >= 3` check before applying flip TTA
**Result**: Verified correct — no impact on validation crash

### Fix 4: Val retry OOM loop (train.py ~1186)
**Problem**: Val retry loop didn't clear GPU memory between attempts
**Fix**: Added `gc.collect()` + `torch.cuda.empty_cache()` after each retry; added DEBUG logging for val_attempt
**Result**: Memory clearing works — no OOM, but validation still crashes

### Fix 5: Eval loop variable scoping (evaluate.py:1443)
**Problem**: `scores_i = cls_sigmoid[i]` was accidentally deleted during debugging
**Fix**: Restored the variable assignment
**Result**: Verified correct — no impact on validation crash

### Fix 6: Better exception logging (train.py:1197)
**Problem**: Non-OOM exceptions in evaluate_all were not being fully logged
**Fix**: Added `logger.exception(f'  [EXCEPTION TRACE] val attempt {val_attempt} failed:')` before the re-raise check
**Result**: Exception is still not captured in logs before crash — confirming the exception bypasses the handler

### Fix 7: Training restarted with nohup
**Command**: `nohup python3 -u train.py --resume runs/pretrain_synthetic/checkpoints/latest.pth --max-epochs 60 --seed 42 > train_output3.log 2>&1 &`
**Result**: Same crash pattern, process becomes zombie

---

## 8. Code Context

### Files modified during investigation
- `evaluate.py`: Multiple device/dtype fixes, TTA guards, variable scoping
- `train.py`: Val retry OOM handling, exception logging enhancement

### Key train.py val retry code (lines 1164-1223)
```python
val_attempt = 0
while True:
    val_attempt += 1
    logger.info(f'  [DEBUG] val_attempt={val_attempt}, entering try block')
    if val_attempt > 4:
        raise RuntimeError('Exceeded maximum validation retry attempts (4).')
    val_loader = _build_loader(val_ds, 'val', val_batch_size_rt, val_workers_rt, prefetch=val_prefetch_rt)
    try:
        val_metrics = evaluate_all(model, criterion, val_loader, device, max_batches=val_max_batches_rt)
    except Exception as exc:
        is_cpu_enomem = 'Cannot allocate memory' in str(exc)
        is_cuda_oom_v = _is_cuda_oom(exc)
        _exc_name = type(exc).__name__
        _exc_msg = str(exc)[:500]
        logger.error(f'  Val attempt {val_attempt}/4 FAILED: {_exc_name}: {_exc_msg}')
        logger.info(f'  [DEBUG] is_cpu_enomem={is_cpu_enomem} is_cuda_oom_v={is_cuda_oom_v}')
        logger.exception(f'  [EXCEPTION TRACE] val attempt {val_attempt} failed:')
        if not (is_cpu_enomem or is_cuda_oom_v):
            logger.error(f'  Non-OOM exception in evaluate_all — aborting train.py immediately.')
            raise
        # OOM path: retry with reduced settings
        val_batch_size_rt = max(1, val_batch_size_rt // 2)
        val_workers_rt = 0
        val_prefetch_rt = 1
        val_max_batches_rt = max(1, int(val_max_batches_rt) // 2)
        gc.collect()
        torch.cuda.empty_cache()
        continue
    finally:
        del val_loader
        gc.collect()
        torch.cuda.empty_cache()
```

### Key evaluate.py exceptions
Only 2 `raise` statements in evaluate.py:
1. **Line 1292**: Re-entry bug check (`model._evaluate_all_active` flag)
2. **Line 1503**: "No batches were produced" check

### evaluate_all structure (evaluate.py)
```python
_in_eval = getattr(model, '_evaluate_all_active', False)
if _in_eval:
    raise RuntimeError('RE-ENTRY BUG...')
model._evaluate_all_active = True
try:
    model.eval()
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc='evaluate')):
        if max_batches and batch_idx >= max_batches: break
        # Process detection, activity, pose, psr, assembly, error tasks
        act_preds.append(...)
        head_pose_preds.append(...)
        # etc.
    if not act_preds:  # Line ~1501
        raise RuntimeError('No batches were produced...')
except Exception:
    if hasattr(model, '_evaluate_all_active'):
        model._evaluate_all_active = False
    raise
```

---

## 9. Timeline of First Crash (PID 66596)

| Time | Event |
|------|-------|
| 17:43:15 | Training starts (12,579 iters/epoch) |
| ~19:12:43 | Epoch 1 training complete |
| 19:12:43 | Val attempt 1 starts |
| 19:13:06 | Batches 400/4000 |
| 19:13:24 | Batches 800/4000 |
| 19:13:42 | Batches 1200/4000 |
| 19:14:00 | Batches 1600/4000 |
| 19:14:18 | Batches 2000/4000 |
| 19:14:35 | Batches 2400/4000 |
| 19:14:53 | Batches 2800/4000 |
| 19:15:10 | Batches 3200/4000 |
| 19:15:31 | Batches 3600/4000 |
| 19:16:12 | Val attempt 1 metrics printed (no ERROR logged) |
| 19:16:12 | Val attempt 2 starts |
| 19:16:34 | Batches 400/4000 |
| 19:16:52 | Batches 800/4000 |
| 19:17:11 | Batches 1200/4000 |
| 19:17:30 | Batches 1600/4000 |
| 19:17:51 | Batches 2000/4000 |
| 19:18:10 | Batches 2400/4000 |
| 19:18:30 | Batches 2800/4000 |
| 19:18:50 | Batches 3200/4000 |
| 19:19:08 | Batches 3600/4000 |
| 19:19:31 | Val attempt 2 metrics printed |
| 19:19:31 | Val attempt 3 starts |
| 19:22:36 | Val attempt 3 DONE (184.8s) |
| 19:22:36 | Val attempt 4 starts |
| 19:25:43 | Val attempt 4 DONE |
| 19:25:43 | Val attempt 5 starts |
| 19:25:43 | **RuntimeError: Exceeded maximum validation retry attempts (4).** |
| ~19:27 | Process becomes zombie (RNl → SNl) |

---

## 10. Hypotheses for Root Cause

### Hypothesis 1: Multiprocessing Worker Crash (Most Likely)
The val DataLoader uses `num_workers=1` (after first OOM retry sets it to 0). When a worker crashes (segfault, signal), the main process gets a broken pipe or EOF, which raises an exception type that isn't `Exception` (e.g., `BrokenPipeError`, `EOFError`) but these **should** be caught by `except Exception`.

However, if the **main process itself** receives a SIGKILL or SIGTERM (e.g., from system OOM manager for a different process, or from the parent's signal handling), it would terminate immediately without going through the exception handler.

**Evidence**: The process becomes a zombie (`SNl` state, `kill -9` ineffective, procfs returns empty) — classic zombie pattern.

### Hypothesis 2: GPU Driver Crash
The GPU driver could crash during validation computation. This would manifest as:
- `cudaError` - possibly not caught as standard Python exception
- Process receives SIGBUS/SIGSEGV from CUDA runtime

**Evidence**: GPU memory was fine (2.4GB/12GB), no dmesg entries.

### Hypothesis 3: Data Pipeline Bug — Empty Val GT
The detection ground truth (`dg_boxes`) being empty causes a downstream crash when trying to compute metrics on empty arrays (NaN propagation, divide-by-zero, etc.).

However, `evaluate.py` handles this with the `len(dg_boxes) == 0` check that produces the warning and skips detection. The crash happens after this, suggesting something else fails when GT is missing.

**Evidence**: The warning appears consistently at the end of every validation attempt.

### Hypothesis 4: Memory Fragmentation
After 4000 batches of validation, GPU memory becomes fragmented. When trying to allocate for the next operation (post-processing, metric computation), it fails.

**Evidence**: GPU memory shows 2.4GB/12GB but that's before validation cleanup. After validation, memory might be in a fragmented state.

### Hypothesis 5: Multiprocessing Semaphore Leak → Deadlock
The `loky` multiprocessing backend leaves orphaned semaphores when workers crash. If enough semaphores accumulate, it can cause deadlocks or resource exhaustion.

**Evidence**: The log shows `resource_tracker: There appear to be 21 leaked semaphore objects` after the first validation attempt. This could cause cascading failures in subsequent attempts.

---

## 11. Things to Try Next (Recommendations for Claude Opus)

### Immediate debugging steps
1. **Add `--timeout` to DataLoader**: Set `timeout=30` on val DataLoader to catch worker hangs
2. **Use `val_workers=0` from the start**: Eliminate multiprocessing workers in validation entirely to rule out worker crashes
3. **Check GPU dmesg during validation**: `dmesg -w` during validation to catch GPU errors
4. **Add crash signal handler**: Catch `signal.SIGSEGV`, `signal.SIGTERM` in train.py to get actual crash reason
5. **Run evaluate.py standalone**: Run `python evaluate.py --split val --max-batches 100` directly (no train.py) to see if validation works in isolation
6. **Reduce val max_batches to 100**: To see if it's a late-epoch issue (batch 3500+)
7. **Check val dataset size**: Verify the val dataset returns non-empty batches

### Code changes to investigate
1. **Val DataLoader with `num_workers=0`**: Modify `_build_loader` for val to always use workers=0
2. **Add signal handlers**: Catch SIGTERM/SIGSEGV and log what signal caused the crash
3. **Protect the tqdm loop**: Wrap the batch iteration in more specific exception handlers
4. **Check dg_boxes source**: Trace where detection GT comes from in the val dataset — why is it empty?

### Questions to answer
1. Why does `dg_boxes` have zero elements for val split in train.py's val loader, but not when running evaluate.py standalone?
2. Is the crash related to the number of batches (4000 is very large)?
3. Is there a specific batch (around 3500-4000) that always triggers the crash?
4. Does the crash happen on the first validation attempt or only after subsequent attempts?

---

## 12. Key Log Excerpts

### Val attempt pattern (from train_output2.log)
```
2026-04-30 19:12:43,683 | INFO |   [DEBUG] val_attempt=1, entering try block
2026-04-30 19:13:06,223 | INFO |   [evaluate_all] batch 400/4000
2026-04-30 19:13:24,194 | INFO |   [evaluate_all] batch 800/4000
...
2026-04-30 19:15:31,712 | INFO |   [evaluate_all] batch 3600/4000
2026-04-30 19:16:12,720 | INFO |   Activity — Acc: 0.0000  Macro-F1: 0.0000  Top-5: 0.0000
2026-04-30 19:16:12,721 | INFO |   Head Pose (9-DoF) — Overall MAE: 0.4907  std: 0.3224
2026-04-30 19:16:12,723 | INFO |   PSR — Overall F1: 0.0779  F1@T: 0.0000  Edit Score: 0.3409  POS: 0.0000
2026-04-30 19:16:12,725 | WARNING | Detection evaluation skipped: no GT boxes found in this split.
2026-04-30 19:16:12,725 | INFO |   ASD — mAP@0.5: nan  mAP@[0.5:0.95]: nan
2026-04-30 19:16:12,726 | INFO |   [evaluate_all] DONE in 209.0s (batch_iterations=4000)
2026-04-30 19:16:12,875 | INFO |   [DEBUG] val_attempt=2, entering try block
# ... this pattern repeats 4 more times
```

### Final crash
```
2026-04-30 19:25:43,625 | INFO |   [DEBUG] val_attempt=5, entering try block
Traceback (most recent call last):
  File "/media/newadmin/master/POPW/working/code/industreal_improved/train.py", line 1465, in <module>
    main(args)
  File "/media/newadmin/master/POPW/working/code/industreal_improved/train.py", line 1169, in main
    raise RuntimeError(
        'Exceeded maximum validation retry attempts (4).'
    )
RuntimeError: Exceeded maximum validation retry attempts (4).
```

### Process status
```
66596  newadmin  SNl  (zombie)  01:46:54  python3 -u train.py ...
# Cannot be killed with kill -9
# procfs /proc/66596/cmdline returns empty
```

---

## 13. File Versions

| File | Key Changes |
|------|-------------|
| `evaluate.py` | `_prepare_images` device fix, `run_model` dtype fix, TTA dimension guard, variable scoping fix |
| `train.py` | Val retry OOM loop with gc.collect(), exception logging enhancement |
| `config.py` | BATCH_SIZE=2, GRAD_ACCUM_STEPS=16, VAL_BATCH_SIZE=4, VAL_NUM_WORKERS=1, EVAL_MAX_BATCHES=4000 |

---

## 14. Checkpoints

- `/media/newadmin/master/POPW/working/code/industreal_improved/runs/pretrain_synthetic/checkpoints/latest.pth` — epoch 1 checkpoint (220MB)
- Training log: `train_output2.log` (12K+ lines, 270MB) — first run
- Training log: `train_output3.log` (61 lines) — second run (crashed immediately)