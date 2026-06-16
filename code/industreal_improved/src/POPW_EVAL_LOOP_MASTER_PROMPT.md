# Master Prompt: POPW Training Eval Looping — Diagnosis & Solution Request

## Context

You are debugging a POPW (multi-task IKEA assembly recognition) training pipeline on RTX 3060 12GB GPU. The training keeps **looping in validation** instead of progressing to the next epoch.

---

## Problem Statement

After Epoch 0 training completes and eval runs, the code prints `[POST_EVAL]` (eval cleanup done) followed by a **new `[EVAL START]` at batch 0** — instead of continuing training into Epoch 1.

**Critical observation**: Training is NOT running between the two evals. The eval runs back-to-back, which means:
- Either `evaluate_all()` returns empty dict and silently fails
- Or the epoch loop is re-triggering eval without training

---

## Observed Log Pattern (THE BUG)

```
[AVG_GUARD] running["log_var_psr"]=-0.901... is invalid — reset to 0.0
[Epoch 0] train completed: 518 batches, steps=569, time=606s
Train: loss=9.7092  det=7.4728  pose=0.8596  act=0.7785  psr=0.0282  lr=1.40e-04
  [PRE_VAL_GUARD] epoch 0 training healthy: batches=518, loss=9.7092
Running validation ...
  [pre-val flush] RSS: 4.37GB -> 4.37GB (freed ~0 MB)
  [EMA] Using exponential-moving-average weights for val
  [EVAL START] GPU alloc=1.09GB  reserved=1.13GB
  ... [eval batches running, full metrics printed] ...
  [EVAL END] GPU alloc=1.36GB  reserved=1.76GB
  [DEBUG] act_gt range=[0, 71]  shifted range=[1, 72]  pred range=[1, 51]
  ... [ALL METRICS PRINTED] ...
  [POST_EVAL] val_loader cleaned up, resuming train...
  [EVAL START] GPU alloc=1.09GB  reserved=1.13GB   ← BUG: new eval starting at batch 0, NO TRAINING between them
  ... [second eval running] ...
```

**Key fact**: All metrics ARE printed after `[EVAL END]`. This means `evaluate_all()` completed successfully and returned a populated dict. Yet immediately after `[POST_EVAL]`, another `[EVAL START]` fires.

---

## Environment

- **GPU**: RTX 3060 12GB
- **Config**: 5% subset, 3 epochs, BATCH_SIZE=6, VAL_BATCH_SIZE=16
- **GRAD_ACCUM_STEPS**: 6
- **EVAL_MAX_BATCHES**: -1 (ALL batches)
- **VAL_EVERY**: 1 (every epoch)
- **Files**: `train.py` (~3058 lines), `evaluate.py` (~3608 lines)

---

## Solutions Already Applied

### Fix 1 — May 26: Eval Loop Bug (TRAIN_MAX_STEPS break)
**Problem**: `TRAIN_MAX_STEPS` break was placed BEFORE validation, causing parent process to restart → back-to-back evals with no training.

**Fix**:
- Moved `break` to AFTER validation (line ~2837)
- Added `PRE_VAL_GUARD` (lines 2531-2559) — raises RuntimeError if train_metrics are suspicious before running eval
- Added `num_batches==0 guard` — raises RuntimeError if train_one_epoch returns zero batches

**Code structure**:
```python
for epoch in range(100):
    train_one_epoch()     # MUST complete first
    [VAL BLOCK]           # structurally AFTER train completes
    break                 # only AFTER val finishes
```

### Fix 2 — May 27: PSR Eval Slowdown
**Problem**: Pure-numpy O(n²) Damerau-Levenshtein double-loop took 65+ minutes for 35K frames.

**Fix**: Added numba JIT compilation (`_get_dl_osa_numba()`) with threshold `max(m,n) >= 5000`. Estimated speedup: 400x (~9s instead of 65min).

### Fix 3 — May 27: VAL_EMPTY Guard (Latest Fix)
**Problem**: Unknown — `evaluate_all()` appears to complete successfully (all metrics printed) yet a second eval starts.

**Fix**: Added diagnostic guard after `finally` block (lines 2691-2709):
```python
if val_metrics:
    logger.info(f'  [VAL_OK] epoch {epoch} val completed, loss={val_metrics.get("loss", -1):.4f}')
else:
    raise RuntimeError(f'VAL_EMPTY: evaluate_all(epoch={epoch}) returned empty metrics.')
```

---

## Questions for Opus (Please Answer)

1. **Why would `[EVAL START]` appear again after `[POST_EVAL]` if `evaluate_all()` completed successfully?** The code path `while True` → `evaluate_all()` → `finally` → `[POST_EVAL]` should exit the `while` loop and continue to post-eval code. What could cause a second iteration of the `while` loop?

2. **Could there be a process restart mechanism we missed?** The training is launched via `nohup python train.py ... &`. Could the parent bash process or systemd be restarting the Python process after it completes or crashes silently?

3. **Is it possible `evaluate_all()` is being called TWICE — once in the main process and once in a worker process?** With `num_workers=0` on val loader, there should be no multiprocessing, but could there be a callback or signal handler that triggers a second eval?

4. **The `finally` block runs AFTER `evaluate_all()` returns — what if the `except` block inside the `try` is catching something unexpected?** Could an exception during `evaluate_all()` cleanup cause control flow to restart the `while True` loop?

5. **Should we add a global counter or log rotation to prevent duplicate log entries?** Or add `assert` statements to verify we're not re-entering the same code path?

---

## Files to Upload for Analysis

Please download and analyze these files from `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/`:

| File | Reason |
|------|--------|
| `training/train.py` | Full epoch loop, validation block (lines 2291-2850), TRAIN_MAX_STEPS logic |
| `evaluation/evaluate.py` | Full `evaluate_all()` function (lines 2418-3600), signal handlers, crash recovery |
| `config.py` | EVAL_MAX_BATCHES, VAL_BATCH_SIZE, VAL_EVERY settings |
| `runs/popw_full_100e/logs/train.log` | Full training log showing the double-eval pattern |

---

## What We Need From You

1. **Identify the root cause** of why eval loops after `[POST_EVAL]` even though `evaluate_all()` appears to complete
2. **Verify our code structure** is correct — is there a path from `[POST_EVAL]` back to `[EVAL START]` that we missed?
3. **Suggest additional diagnostics** we can add to definitively identify whether `evaluate_all()` is being called twice, or whether a separate process is starting eval
4. **Confirm or deny** the possibility of a process restart/signal-based re-execution

---

## Code Sections of Interest

### train.py — eval block (lines ~2561-2709)
```python
val_metrics = {}
if (epoch + 1) % C.VAL_EVERY == 0:        # Always true for epoch=0, VAL_EVERY=1
    val_loader = _build_loader(val_ds, 'val', ...)  # workers=0
    IN_EVALUATION_PHASE = True
    try:
        try:
            val_metrics = evaluate_all(model, criterion, val_loader, device, max_batches=val_max_batches_rt)
            _check_per_class_activity_sanity(val_metrics, epoch)
        except Exception as exc:
            # OOM and recoverable error handling with continue (retry loop)
            continue
    finally:
        IN_EVALUATION_PHASE = False
        _shutdown_loader_workers(val_loader, logger)
        del val_loader; gc.collect(); torch.cuda.empty_cache()
        logger.info('  [POST_EVAL] val_loader cleaned up, resuming train...')

    # [VAL_EMPTY guard — latest fix]
    if val_metrics:
        logger.info(f'  [VAL_OK] epoch {epoch} val completed')
    else:
        raise RuntimeError(f'VAL_EMPTY: evaluate_all returned empty metrics.')

    if ema is not None:
        ema.restore()
    # ... post-eval processing, checkpointing ...
    logger.info(f'Val: loss={_s(val_metrics.get("loss")):.4f} ...')   # Line 2721
```

### evaluate.py — entry point (lines 2418-2515)
```python
def evaluate_all(model, criterion, loader, device, max_batches=0, save_dir=None):
    # Line 2505: [EVAL START] logged here
    total_loss = 0.0
    lc = 0
    act_preds, act_labels = [], []
    # ... lists initialized ...
    for bi, (images, targets) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        # [EVAL batch N] logged every 10 batches at line 2538
        # ... inference and metric accumulation ...
    # Line ~3500+: metrics computed and returned
    return metrics_dict
```

### config.py — relevant settings
```python
VAL_BATCH_SIZE = 16
VAL_EVERY = 1
EVAL_MAX_BATCHES = -1   # -1 means ALL batches
```

---

## Log Evidence of Complete Metrics (First Eval)

The first eval prints ALL metrics (loss, mAP50, activity F1, head pose, PSR, assembly state). This proves `evaluate_all()` returned a populated dict:

```
Activity — Acc: 0.0000  Macro-F1: 0.0023  Weighted-F1: 0.0089
Head Pose — Forward angular: 25.6713 deg  Up angular: 21.3376 deg
PSR — Overall F1: 0.0000  F1@±3: 0.0000  Edit: 0.5249
Assembly State — F1@1: 0.0000  Top-1 Acc: 0.0000  MAP@R(+): 0.0000
ASD — mAP@0.5: 0.0000  mAP@[0.5:0.95]: 0.0000  mAP@0.5 (all frames): 170.1726
```

Yet after `[POST_EVAL]`, a second `[EVAL START]` fires at batch 0.

---

## Suspicion: Process Manager Restart

The training is launched with:
```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src
PYTHONUNBUFFERED=1 nohup python -u training/train.py --no-staged-training --subset-ratio 0.05 --seed 42 --max-epochs 3 > runs/popw_full_100e/logs/train.log 2>&1 &
```

**Is there any possibility the bash `nohup` process, or a process manager watching it, is restarting the Python script after some event (e.g., seeing no output for N seconds, or a signal)?** How could we rule this out?

---

**Thank you for your analysis. Please be thorough — this is blocking our thesis experiments.**