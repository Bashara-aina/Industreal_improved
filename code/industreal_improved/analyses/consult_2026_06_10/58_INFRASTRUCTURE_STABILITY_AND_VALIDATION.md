# 58: Infrastructure Stability and Validation Failures [2026-06-30]

## The Reliability Crisis

In the last 12 hours of training (Jun 30, 06:00-18:00 UTC), we experienced 5 process
deaths and 1 CUDA hang. Only 1 out of 6 training attempts completed a validation cycle
without incident. This is wasting ~80% of GPU compute.

## Incident Log (Jun 30)

### Incident 1: CUDA Hang during epoch 2 validation (PID 2554562)
- Time: 16:14 UTC
- Duration: 37 minutes before manual kill
- Cause: evaluate_all hung in CUDA kernel (probably DataLoader worker deadlock)
- Recovery: Manual kill, state reset, relaunch. Pre-val checkpoint saved, 0 epoch loss.

### Incident 2: CUDA Hang during epoch 2 validation (PID 2813821)
- Time: ~17:00 UTC
- Duration: Unknown (killed on next session)
- Cause: Same as #1

### Incident 3: DataLoader worker crash (PID 2884976)
- Time: 17:35 UTC
- Duration: Unknown
- Cause: With NUM_WORKERS=4, DataLoader worker crashed silently → process hung

### Incident 4: Watchdog false positive (PID 2928725)
- Time: ~17:50 UTC
- Duration: 0 (immediate detection)
- Cause: Watchdog thread read stale heartbeat from PREVIOUS process's rf4/ directory.
  The heartbeat PID was 2340951 (not 3168268), but the initial watchdog code didn't
  check PID. Fixed by adding PID verification.
- Recovery: Automatic (watchdog killed, supervisor restarted)

### Incident 5: CUDA Hang during epoch 3 validation (PID 3369770)
- Time: ~18:39 UTC
- Duration: Detected by watchdog after 616s, auto-killed
- Note: This had NUM_WORKERS=0, ruling out DataLoader workers as the cause.
  The hang is in the evaluate_all CUDA code itself.

### Incident 6: Watchdog recovered stale heartbeat (PID 3369770)
- Time: ~18:48 UTC
- Duration: 616s (watchdog timeout)
- The new PID-checked watchdog correctly identified the stale heartbeat as belonging
  to process 2340951 (from epoch 1), killed the process, supervisor restarted.

## Timeline of Fixes

| Fix | Applied After Incident | Effect |
|-----|----------------------|--------|
| Pre-val checkpoint (#1) | #1, #2 | Latest.pth saved before val → 0 epoch loss |
| NUM_WORKERS=0 | #3, #4, #5 | Eliminates DataLoader deadlocks |
| Watchdog thread | #3 | Automatic detection of CUDA hangs |
| Watchdog PID check | #4 | Prevents false positives from stale heartbeats |
| VAL_NUM_WORKERS=0 | #5 | Also zero workers for eval |
| Evaluate_all ThreadPoolExecutor | #1 (pre-existing) | Prevents SIGALRM from crashing interpreter |

## Current Safety Measures (ALL ACTIVE)

### Pre-Val Checkpoint (train.py:4343+)
```python
# After training completes, before validation:
_atomic_save({...model, optimizer, scheduler, scaler...}, ckpt_dir / 'latest.pth')
```
Confirmed working: epoch 2 latest.pth saved at 16:08:55 before val at 16:14:13.
After incident #1, resume from epoch 3 (latest.pth epoch=2+1=3). Zero training loss.

### GPU Heartbeat (train.py:2028)
```python
_hb_path.write(f'{time.time()}|{step}|{epoch}|{os.getpid()}\n')
```
Every 100 steps. PID included for watchdog verification.

### Watchdog Thread (train.py:3927)
```python
_watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
# Checks every 30s: if heartbeat > 600s old AND pid matches, os._exit(1)
```
PID verification added to prevent killing on stale heartbeat from prior run.

### Atomic Saves (train.py:241)
```python
torch.save(tmp) → tmp.rename(path)  # POSIX atomic rename
```
Prevents half-written checkpoints from mid-write crashes.

### Evaluate_all Timeout (train.py:4404)
```python
_eval_executor = ThreadPoolExecutor(max_workers=1)
_eval_future = _eval_executor.submit(evaluate_all, ...)
val_metrics = _eval_future.result(timeout=_eval_timeout)
```
1200s timeout (configurable via EVAL_TIMEOUT_SECONDS). Cannot interrupt CUDA kernel
hangs (signal handlers need Python interpreter), but ThreadPoolExecutor + retry
mechanism reduces probability.

## Remaining Vulnerabilities

### 1. CUDA Kernel Hangs Are Not Interruptible
The biggest remaining problem. When evaluate_all hangs in a CUDA kernel:
- ThreadPoolExecutor timeout raises TimeoutError
- But the underlying thread keeps running with the CUDA kernel
- The retry creates a NEW ThreadPoolExecutor (zombie thread stays alive)
- Eventually VRAM fills from zombie threads → OOM → crash
- The 600s watchdog detects the hang and kills the process, but loses the current step

### 2. No Full Epoch Validation Has Succeeded
The epoch 1 and epoch 2 validations only ran 200 batches (GATE_EVAL_MAX_BATCHES).
The FULL validation (DET_METRICS_EVERY_N=1 means every epoch) would process
much more data and has NEVER been verified to complete.

### 3. Multi-GPU Underutilization
GPU 0 (RTX 3060 12GB) has been completely idle across ALL runs. Only GPU 1
(RTX 5060 Ti 16GB) is utilized. CUDA_VISIBLE_DEVICES is not set, so PyTorch
defaults to GPU 0, but the model fits entirely in GPU 1's VRAM so GPU 0 is unused.

Data parallelism (DistributedDataParallel) was never configured. With 2 GPUs
we could double throughput from 1.2 to ~2.0 batch/s.

### 4. Hard Drive Bottleneck
Training data is on an HDD (buffers=570GB, cached=18TB, avail=20GB RAM).
With NUM_WORKERS=0, the DataLoader loads frames in the main process, which blocks
on disk I/O. This limits speed to 1.2 batch/s. With 2 GPUs + SSD, could reach
3-4 batch/s.

### 5. No Crash Recovery on Epoch 4+
The pre-val checkpoint only saves after training completes. If a crash occurs
during training mid-epoch, the latest checkpoint is from the PREVIOUS epoch end.
The crash_recovery.pth is saved every 100 steps but is NOT loaded automatically
on resume — only via manual --resume crash_recovery.pth.

## Questions for Opus

1. **Should we implement DistributedDataParallel (DDP) for dual GPU training?**
   This would utilize the idle RTX 3060 and double throughput to ~2.0 batch/s.
   Cost: 1-2 days of engineering + potential debugging.

2. **Can evaluate_all be rewritten to avoid CUDA kernel hangs?**
   E.g., processing in smaller batches with explicit CUDA synchronization between
   each batch, so a hang only loses 1 batch instead of the entire eval?

3. **Should we implement mid-epoch crash recovery?**
   If crash_recovery.pth is saved every 100 steps, and we automatically load it
   on restart, the maximum loss is 100 steps (~83 seconds at 1.2 batch/s).

4. **Can we use NVLink or PCIe peer-to-peer for the 3060+5060 Ti combination?**
   Different GPU architectures may cause compatibility issues with DDP.

5. **Should we switch to SSD-based training?**
   Moving IndustReal dataset to SSD would eliminate the HDD bottleneck.
   Current HDD: cached=18TB (kernel page cache), but cold misses cost ~10ms per seek.
   With 4 workers, this causes thread convoy. With 0 workers, the main process blocks.

6. **What is causing the evaluate_all CUDA hang specifically?**
   It happens during validation, not training. The validation uses the SAME
   model and similar data. Possible causes: (a) torch.no_grad() interacting badly
   with CUDA graphs, (b) eval-time NMS (non-maximum suppression) creating
   variable-size tensors, (c) DataLoader prefetch in val workers.

7. **Should we validate EVERY epoch or can we skip epochs?**
   Currently VAL_EVERY=1 with GATE_EVAL_MAX_BATCHES=200 for gate-only checks.
   Full eval every DET_METRICS_EVERY_N=1 epoch. With the hang risk, consider
   VAL_EVERY=2 or 3 to reduce failure probability.

8. **Is cudaSetDevice or CUDA_VISIBLE_DEVICES misconfigured?**
   GPU 0 sits idle at 0% while GPU 1 runs at 100%. The subprocess log shows
   "GPU: NVIDIA GeForce RTX 5060 Ti" and "VRAM: 16.6 GB" — confirming training
   is on GPU 1. But CUDA_VISIBLE_DEVICES is not set, which means both GPUs are
   visible. torch.cuda.is_available() returns True and device defaults to cuda:0.
   If the code calls torch.cuda.set_device(0) anywhere, GPU 0's VRAM (12GB) would
   be the training device and would OOM quickly. The fact that GPU 0 is IDLE at
   73MB suggests the model is on GPU 1 via some other mechanism.

## GPU Utilization Data

```
GPU 0 (RTX 3060):    0% util,  73 MiB / 12288 MiB, 33°C  — COMPLETELY IDLE
GPU 1 (RTX 5060 Ti): 83-100% util, 4500-7100 MiB / 16311 MiB, 58-62°C — TRAINING
```

GPU 0 has been idle for the entire session. No process is using it.
