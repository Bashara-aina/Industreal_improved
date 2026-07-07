# Single-task Detection Training Status v3

**Timestamp**: 2026-07-07 16:59 JST
**Source**: `/tmp/train_singletask_det.log` (21,356 lines, 17.6 MB)

## Process

- **PID**: 1574104 (ALIVE)
- **CPU**: 204% (2 cores), RSS 6.5 GB, VSZ 61 GB
- **Command**: `python3 src/training/train_singletask_detection.py --batch-size 2 --no-staged-training`
- **Process started**: 14:54 JST (running ~2 hours)
- **Log created**: 09:50 JST (previous process; current PID 1574104 appending)

## Training Progress

| Metric | Value |
|--------|-------|
| Total epochs | 99 |
| Started epoch | 34/99 |
| Current epoch | 47/99 (step 460 / 13,161 batches) |
| Epochs completed | 13 (epochs 34-46 fully done) |

## Loss Metrics

| Component | Current value (epoch 47 average) |
|-----------|----------------------------------|
| Total loss | ~1.85-2.10 |
| Detection loss | ~2.20-2.75 |
| Detection classification | ~1.35-1.70 |
| Detection regression | ~0.42-0.50 |
| Pose/action/PSR | 0.0 (no multi-task) |

**Trend**: Detection loss bounces between 1.8 and 3.0 at batch level, with rolling average around 2.3. No clear downward trend yet -- model may still be warming up or near a plateau for this epoch.

## Speed & ETA

| Metric | Value |
|--------|-------|
| Current speed | ~3.0-3.6 batches/sec |
| Per-epoch estimate | ~60 min at current speed |
| Remaining epochs | 52 (47/99 done, 52 left) |
| **Estimated completion** | **~52 hours** (July 9 evening JST) |

Note: If speed improves with data caching (current bottleneck may be disk I/O on first epoch pass), ETA could shrink.

## System Resources

| Resource | Usage |
|----------|-------|
| GPU memory | 1.10 GB allocated, 2.66 GB reserved |
| CPU RAM | 20 GB used, 39 GB available (of 62 GB total) |
| System load | 11.80 (3-day uptime) |

## Health

- **Detection**: ALIVE (Kendall lv=0.616, gradient active)
- **Pose/action/PSR**: DEAD (expected -- single-task mode)
- **OOM/crash events**: None observed
- **Crash recovery saves**: 64 (normal epoch-boundary checkpoint saves, not actual crashes)

## Validation mAP

No validation running -- training-only mode. No mAP metrics available.

## Key Observations

1. Training has progressed 14% through total epochs (34 baseline + 13 completed)
2. Detection loss is stable but noisy -- typical for early-to-mid training
3. System has memory headroom (39 GB free)
4. GPU utilization is low (1.1 GB allocated on what appears to be a larger GPU)
5. 606 liveness checks passed, no errors logged
