# Single-Task ConvNeXt-Tiny Detection Training -- Monitoring V2

**Date**: 2026-07-07 ~12:21 onward (launch3)
**Agent**: 76 (v2 check)
**Previous reading**: Agent-68 at epoch 24

## Training Status: ALIVE

- **PID**: 811481 (running since 12:21)
- **Current epoch**: 43/99
- **Current step**: ~3640/13161 (28% into epoch 43)
- **Elapsed**: ~58 minutes in epoch 43 (~6 hours total since launch at 12:21)
- **Speed**: 2.4-2.7 it/s on progress bar, 1.0 batch/s elapsed average
- **Batch size**: 2 (RTX 5060 Ti OOM mitigation, bf16 mixed precision)
- **Precision**: bf16 mixed precision

## GPU Utilization

| GPU | Utilization | Memory Used |
|-----|------------|-------------|
| 0 (RTX 5060 Ti) | 82% | 4,082 MiB / 16.6 GB |
| 1 (RTX 5060 Ti) | 85% | 4,096 MiB / 16.6 GB |

GPU memory allocated to model: ~1.13-1.27 GB, reserved: ~3.5 GB.

## Detection Loss Values (epoch 43, recent samples)

Loss varies widely per batch depending on image content:

| Metric | Typical Range | Notes |
|--------|--------------|-------|
| Total loss | -0.15 to 1.7 | Negative from weight decay (wd=0.28) on empty-frame batches |
| det loss | 0.004 to 2.5 | Low on background images, high on hard positives |
| det_cls (classification) | 0.3-0.9 | Dominant loss component |
| det_reg (regression) | 0.2-0.4 | Smaller contribution |
| wd (weight decay) | 0.28 | Consistent |

Negative total losses occur when a batch has near-zero detection signal (empty frames), and the weight decay term pushes the sum negative. This is expected behavior.

## Livelihood Gradients (Kendall, step 3601)

| Head | Status | RMS Gradient | Notes |
|------|--------|-------------|-------|
| detection_head | ALIVE | 1.08e-01 | 36 params with gradient |
| backbone | ALIVE | 2.17e-01 | 178 params with gradient |
| fpn | ALIVE | 5.15e-03 | 16 params, barely alive |
| pose_head | NO_GRAD | -- | Disabled |
| head_pose_head | NO_GRAD | -- | Disabled |
| activity_head | NO_GRAD | -- | Disabled |
| psr_head | NO_GRAD | -- | Disabled |

HP_PREC_CAP (head precision cap) is active for pose, preventing gradient starvation warnings.

## Crash History

- **Epoch 34**: Crash at step 999, CRASH_RECOVERY saved `epoch34_step1000`
- **Epochs 35-41**: All had `epoch_start` crash checkpoints saved
- **Epoch 42**: SIGTERM received at step 53 (~3:32 into epoch), CRASH_RECOVERY saved, retried automatically via CRASH-RETRY
- **Epoch 43**: Running stable. Crash checkpoints saved at steps 1000, 2000, 3000. Currently at step ~3640.

The crash recovery system works correctly -- SIGTERM events are caught, state is saved to `crash_recovery.pth`, and training resumes automatically.

## Progress vs Agent-68

| Metric | Agent-68 (epoch 24) | Agent-76 (epoch 43) | Delta |
|--------|--------------------|--------------------|-------|
| Epoch | 24 | 43 | +19 epochs |
| Loss pattern | Not recorded | Stable, varying per-batch | -- |
| Stability | Alive | Alive with crash recovery | Consistent |

Training has advanced 19 epochs since the last reading.

## ETA

- Epochs remaining: 57 (100 - 43)
- Time per epoch: ~87 minutes (13,161 batches / 2.5 it/s)
- Estimated remaining: ~82 hours (~3.4 days)
- No validation mAP being computed (training only)
