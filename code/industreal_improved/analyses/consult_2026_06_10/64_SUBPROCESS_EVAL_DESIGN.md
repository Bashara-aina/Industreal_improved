# 64: Subprocess Evaluation Design — Solving CUDA Kernel Hangs [2026-06-30]

## The Problem

Training has crashed 5 times in 12 hours because `evaluate_all` hangs in CUDA kernels
during validation. Current mitigation (`ThreadPoolExecutor` with 1200s timeout at
`train.py:4404-4416`) cannot interrupt CUDA — the thread stays alive, VRAM fills,
eventual OOM kills the process. The watchdog (600s) detects the hang but loses
the current training step.

Opus confirmed (62 §6, Q3; 63 §5): **Threads and SIGALRM cannot interrupt a CUDA
kernel. Subprocess + SIGKILL is the only reliable solution.**

## Current Code (the problem)

### Path 1: ThreadPoolExecutor in train.py (epoch-end validation)
```python
# train.py:4404-4416
import concurrent.futures as _cf
_eval_executor = _cf.ThreadPoolExecutor(max_workers=1)
_eval_future = _eval_executor.submit(evaluate_all, model, criterion, val_loader, ...)
try:
    val_metrics = _eval_future.result(timeout=1200)
except _cf.TimeoutError:
    _eval_executor.shutdown(wait=False)  # CANNOT interrupt CUDA kernel
    raise TimeoutError(...)
```

The `shutdown(wait=False)` does NOT kill the thread. The CUDA kernel keeps running.
Each retry creates a new executor + zombie thread. After 2 retries, VRAM fills.

### Path 2: SIGALRM in evaluate_all (segment metrics)
```python
# evaluate.py:3568-3576
try:
    _old_handler = signal.signal(signal.SIGALRM, _seg_alarm)
    signal.alarm(_seg_timeout)
except ValueError:
    logger.warning('[GAP-B] Cannot set SIGALRM timeout (not in main thread)')
```

When called from within ThreadPoolExecutor (thread, not main thread), `signal.signal()`
raises ValueError and the alarm is silently skipped. This is exactly the failing path.

## Key Constraints

1. **Model is ~800MB** — serializing to CPU and loading in subprocess takes 5-10 seconds
2. **Validation data is 1,928 frames** at 50% subset — fitting in ~200MB
3. **CUDA kernel hangs are non-deterministic** — may hang on batch 17 or batch 199
4. **Two independent validation calls per epoch**: (a) 200-batch gate eval and
   (b) full eval every DET_METRICS_EVERY_N=1 epoch
5. **The subprocess must not corrupt the parent's CUDA context** — sharing `cuda:0`
   between parent and child is unsafe unless we use `spawn` start method

## Proposed Architecture

### Option A: Subprocess per validation (recommended by Opus)

```python
import multiprocessing as mp
ctx = mp.get_context('spawn')  # NOT fork — fork shares CUDA context

def _eval_subprocess(ckpt_path, val_csv_path, output_path, config_overrides):
    """Run in child process. Loads model from checkpoint, evaluates, writes JSON."""
    import torch, json, sys
    sys.path.insert(0, 'src')
    from src import config as C
    for k, v in config_overrides.items():
        setattr(C, k, v)
    from models.model import POPWMultiTaskModel
    model = POPWMultiTaskModel(...).to('cuda')
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt['model'])
    model.eval()
    loader = build_val_loader(...)
    metrics = evaluate_all(model, ..., max_batches=200)
    with open(output_path, 'w') as f:
        json.dump(metrics, f)

# In train.py, after training completes:
_ckpt_path = ckpt_dir / 'latest.pth'
_output_path = ckpt_dir / f'val_epoch_{epoch}.json'
p = ctx.Process(target=_eval_subprocess, args=(_ckpt_path, ...))
p.start()
p.join(timeout=600)  # 10 min timeout
if p.is_alive():
    p.kill()  # SIGKILL — CAN interrupt CUDA
    p.join()
    logger.warning('Validation subprocess killed after 600s timeout')
# Read results
if _output_path.exists():
    val_metrics = json.loads(_output_path.read_text())
```

**Tradeoffs:**
- + **CUDA kernel hangs are fully killable** via SIGKILL
- + No zombie threads, no VRAM leak
- + Parent CUDA context is isolated (spawn creates fresh context)
- - ~10 second overhead per validation for model load + CUDA init
- - Cannot reuse model weights in-memory (must write checkpoint to disk)
- - 1,928-frame validation with 0 workers adds ~2 min of serial loading

### Option B: Lighter-weight — Per-batch CUDA sync + early exit

```python
# In evaluate_all loop, between each batch:
for batch_idx, batch in enumerate(loader):
    torch.cuda.synchronize()  # Blocks until ALL pending CUDA ops complete
    # If a kernel hangs, synchronize() hangs HERE, not in the forward pass
    # Wrap synchronize in a signal.alarm:
    signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(TimeoutError(...)))
    signal.alarm(30)  # 30 seconds per batch max
    try:
        torch.cuda.synchronize()
    except TimeoutError:
        logger.warning(f'CUDA hang on batch {batch_idx}')
        return metrics_partial  # Return partial results, don't crash
    signal.alarm(0)
```

**Tradeoffs:**
- + Much simpler to implement (~50 lines vs ~200 lines)
- + No subprocess overhead (model already in GPU memory)
- - SIGALRM still may not interrupt a blocking CUDA sync call
- - More fragile — depends on signal delivery timing
- + Returns partial metrics on hang (better than crash)

### Option C: Hybrid — Subprocess only for full eval, per-batch sync for gate eval

Most pragmatic. The 200-batch gate eval runs every epoch (high risk of hang).
The full eval runs every 1-3 epochs (lower risk, worth the subprocess overhead).

## Questions for Opus

1. **Option A vs C?** Is the subprocess overhead acceptable for the 200-batch gate eval
   (every epoch, ~48 times in RF4-RF10)? Or should we only subprocess the full eval?

2. **Multiprocessing start method:** `spawn` vs `forkserver`? With `spawn`, the child
   must re-import all of PyTorch (5-8 seconds). With `forkserver`, we pre-fork once
   at training start. Does PyTorch CUDA support forkserver safely on Python 3.13?

3. **GPU memory isolation:** After `p.kill()`, does the child's GPU memory (6GB) get
   released immediately? Or do we need `torch.cuda.empty_cache()` + `gc.collect()`
   in the parent? Any risk of CUDA context corruption from killing mid-kernel?

4. **Should we disable validation entirely for some epochs?** With VAL_EVERY=1 and
   DET_METRICS_EVERY_N=1, we validate every epoch. If we set VAL_EVERY=3 and
   DET_METRICS_EVERY_N=3, we reduce hang probability by 3x. Is per-epoch metric
   signal worth the crash risk?

5. **crash_recovery.pth auto-load:** Should we modify the resume logic in
   `train.py:3608-3618` to prefer crash_recovery.pth over best.pth when the
   crash_recovery.pth has a newer mtime? Currently it always loads best.pth
   on stage_manager launch. If crash_recovery.pth is 100 steps newer and we
   load it, we save 100 steps per crash. But it carries untested weights
   (post-training, pre-validation). Is this safe?

6. **Model serialization format for subprocess:** `torch.save(model.state_dict())`
   creates a 758MB file. Writing to disk + reading back + deserializing takes
   ~15 seconds total. Could we use `torch.save(model.state_dict(), io.BytesIO())`
   and pass the bytes through a multiprocessing.Queue instead? This avoids disk
   I/O but requires shared memory.

7. **What if the subprocess itself hangs during model loading (CUDA init)?**
   The 10-minute timeout catches this, but we burn 10 minutes of GPU time.
   Should we have a shorter initial timeout for model load (60s) + extend for eval?

## Implementation Plan (if Option A)

```python
# In train.py, new function:
import multiprocessing as mp
import pickle
import json
from pathlib import Path

# Registered at module level once
_EVAL_CTX = None
def _get_eval_ctx():
    global _EVAL_CTX
    if _EVAL_CTX is None:
        _EVAL_CTX = mp.get_context('spawn')
    return _EVAL_CTX

def _run_val_subprocess(model, val_ds, ckpt_dir, epoch, args):
    """Evaluate in a subprocess. Returns metrics dict or {} on timeout."""
    val_pkl = ckpt_dir / f'_val_state_{epoch}.pkl'
    out_json = ckpt_dir / f'_val_result_{epoch}.json'
    
    # Serialize model weights to temp file
    torch.save(model.state_dict(), val_pkl)
    
    ctx = _get_eval_ctx()
    p = ctx.Process(target=_val_worker, args=(str(val_pkl), str(out_json), args))
    p.start()
    p.join(timeout=600)  # 10 min
    if p.is_alive():
        p.kill()
        p.join()
        logger.warning(f'[VAL] Subprocess killed after 600s (epoch {epoch})')
        result = {}
    else:
        if out_json.exists():
            result = json.loads(out_json.read_text())
        else:
            result = {}
    
    # Cleanup
    val_pkl.unlink(missing_ok=True)
    out_json.unlink(missing_ok=True)
    return result

def _val_worker(pkl_path, out_path, args):
    """Run in child process."""
    import sys
    sys.path.insert(0, 'src')
    # ... load model, evaluate, write JSON ...
```

## Related Code References

- `train.py:4387-4506` — Current validation block with ThreadPoolExecutor
- `train.py:4404-4416` — The timeout that CANNOT kill CUDA
- `evaluate.py:3544-3576` — SIGALRM segment metrics that silently fails in threads
- `evaluate.py:2970` — `evaluate_all()` entry point (4,489 lines of metrics)
- `config.py:904` — DET_METRICS_EVERY_N = 1 (full eval every epoch)
- `config.py:872-873` — GATE_EVAL_MAX_BATCHES = 200
- `train.py:3608-3618` — Resume logic (currently always loads best.pth)
