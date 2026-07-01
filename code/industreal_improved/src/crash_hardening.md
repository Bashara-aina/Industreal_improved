# Crash Hardening v2 — 2026-07-01

Do NOT rebuild these fixes or re-investigate these root causes — read this file first.

## Symptom
Training PID 245441 died at 16:40:31 with **SIGABRT (Signal 6)** — no traceback, no log, just
`--- Epoch 6/99 ---` then `[CRASH_RECOVERY] Saved epoch_start crash checkpoint` then silence.
Verified by `coredumpctl info 245441`:
```
Signal: 6 (ABRT)
Control Group: app-org.chromium.Chromium-14946.scope
```

## Root Cause (3 independent failure modes)

### (1) CUDA nondeterministic kernel crash (primary)
- cuDNN benchmark mode (`CUDNN_BENCHMARK=True`) picks the fastest convolution algorithm
  at runtime — some implementations have race conditions on Ampere with expandable_segments.
- Result: illegal memory access in GPU kernel → `abort()` from CUDA async error handler.
- Fix: `CUDA_LAUNCH_BLOCKING=1` always-on + `cudnn.deterministic=True` + `benchmark=False`.
- Tradeoff: ~5-10% throughput loss (crash -> 100% loss, so net win).

### (2) VS Code terminal death (amplifier)
- VS Code 1.109.5 has an Electron bug causing repeated SIGILL/SIGTRAP crashes:
  - 2026-07-01 22:14: 2 VS Code crashes (code PID 15019 TRAP, 15042 ILL)
  - 2026-06-26: 9 consecutive crashes every ~5 minutes
- When VS Code's terminal host dies, SIGHUP kills all terminal child processes silently.
- Fix: `signal.signal(signal.SIGHUP, handler)` that logs and continues, not exits.
- Additionally: `scripts/train_isolated.sh` wraps training in tmux (SIGHUP-proof).

### (3) SystemExit from signal handler bypassed retry loop (fatal gap)
- `except Exception` in the epoch retry loop could NOT catch `SystemExit` from
  `sys.exit(0)` called in the SIGABRT handler.
- Even though the handler saved crash_recovery.pth before exit, the epoch was dead.
- Fix: `except BaseException` catches `SystemExit` + `continue` for retry.

## Fix Inventory (8 layers)

| # | Fix | Mechanism | File |
|---|-----|-----------|------|
| 1 | CUDA_LAUNCH_BLOCKING=1 always-on | Catches CUDA errors as Python RuntimeError | train.py:20 |
| 2 | cudnn.deterministic=True + benchmark=False | No nondeterministic kernel selection | train.py:302-303 |
| 3 | NVIDIA_TF32_OVERRIDE=0 + CUDA_MODULE_LOADING=LAZY | TF32 off, lean driver | train.py:25,29 |
| 4 | SIGHUP handler (no exit) | VS Code crash -> logs + continues | train.py:1078-1098 |
| 5 | except BaseException catches SystemExit | Retry loop catches signal-handler exits | train.py:4284+ |
| 6 | WATCHDOG_TIMEOUT=1200 | No false positive watchdog kills | train.py:4092, config.py:539 |
| 7 | train_isolated.sh (tmux wrapper) | Survives terminal pane death | scripts/train_isolated.sh |
| 8 | WATCHDOG_TIMEOUT config key | 1200s default | config.py:539 |

## How to Launch Safely

```bash
cd /media/newadmin/master/POPW/working/code/industreal_improved
./code/industreal_improved/scripts/train_isolated.sh --preset stage_rf4 --reinit-heads --seed 42 --subset-ratio 1.0
```

Monitor:  `tail -f code/industreal_improved/src/runs/rf_stages/logs/train.log`
Attach:   `tmux attach -t industreal-train`
Kill:     `tmux kill-session -t industreal-train`

## Key Code Locations
- `train.py:1-36` — CULA_LAUNCH_BLOCKING + CUDA hardening env vars
- `train.py:297-310` — seed_everything() with deterministic CUDA
- `train.py:1078-1100` — SIGHUP handler registration
- `train.py:4080-4108` — Watchdog with configurable timeout
- `train.py:4284-4310` — BaseException retry with SystemExit handling
- `config.py:539` — WATCHDOG_TIMEOUT = 1200
- `scripts/train_isolated.sh` — tmux launch wrapper

## Counter-Evidence (what it was NOT)
- NOT OOM: 45GB RAM free, no oom-killer in dmesg, GPUs idle
- NOT VS Code crash: VS Code crashed at 22:14, training died at 16:40 (separate events)
- NOT SIGHUP: training died from SIGABRT (internal), not SIGHUP (terminal)
- NOT disk full: 152GB free on /
- NOT GPU driver: nvidia-smi works (33/34C), no Xid errors in dmesg
- NOT watchdog: no watchdog messages near 16:40
