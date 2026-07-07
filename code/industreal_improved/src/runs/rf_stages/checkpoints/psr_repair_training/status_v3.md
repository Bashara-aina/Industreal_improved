# PSR Head Repair Training Status v3

**Date:** 2026-07-07  
**Monitor:** Agent-75 (PSR Repair Monitor V3 Specialist)  
**Reference:** Previous check by Agent-66 at epoch 24 step 1770

---

## 1. Training Process

| Metric | Value |
|--------|-------|
| Process PID | 813375 |
| Status | **ALIVE** (running since 12:21 UTC) |
| CPU time | 77+ minutes |
| Memory | 9.2% of system RAM |
| GPU 0 util | 91% |
| GPU 1 util | 83% |
| GPU 0 mem | 6,230 MiB |
| GPU 1 mem | 4,096 MiB |

## 2. Current Progress

| Metric | Agent-66 (earlier) | Agent-75 (now) | Delta |
|--------|-------------------|----------------|-------|
| Epoch | 24 | 24 | Same epoch |
| Step | 1,770 | ~2,607 | **+837 steps** |
| Total epoch steps | 13,161 | 13,161 | 19.8% complete |
| Elapsed time | - | 58 min (3,504s for 2,590 steps) | - |
| Speed | - | **0.7 batch/s** (1.4s/batch) | - |
| Progress | - | ETA ~1h40min remaining in epoch | - |
| Weight decay | - | 0.19-0.20 | - |

## 3. Crash History

- One OOM crash at epoch 23 step ~1000 (`alloc_cpu.cpp:127` - CPU memory allocation failure)
- Crash recovery successfully triggered 4 times (crashes at epoch start + step 1000/2000)
- Training auto-resumed from `crash_recovery.pth` checkpoint
- No active crashes - process is healthy

## 4. Loss Values (Latest)

**Non-PSR frames (normal):**
- Total loss: 3.1 - 8.9 (variable per batch)
- Detection loss: 0.12 - 3.5 (cls + reg)
- Pose loss: 0.04 - 4.5
- Activity loss: 1.2 - 2.1
- **PSR loss: 0.0000** (on all normal frames - expected since PSR fires only on seq=1 frames)

**Seq=1 frames (PSR firing):**
- PSR loss: 5.1 - 24.5 (highly variable)
- All other losses zeroed out (det=0.000 pose=0.000 act=0.000)

## 5. PSR Activation Stats

**Total seq=1 events in log:** 381 (from step 1 to ~2607 of epoch 24)

**PSR loss distribution on seq=1 frames (most frequent values):**
- psr=1.322 (98 occurrences) - also includes KENDALL lv_grad values
- psr=7.022 (72)
- psr=7.874 (60)
- psr=9.909 (52)
- psr=11.214 (52)
- psr=16.845 (50)
- psr=10.039 (50)
- Range: 1.3 to 44.5

**PSR head fires on roughly 15% of frames** (381 / ~2600 steps)

## 6. Activation Diagnostics (PSR_DEBUG)

**Early steps (epoch 23, step 0-10):**
- pre_linear: mean=-2.8 to -1.4, std=41-47 (moderate spread)
- post_linear64: mean=-145 to -126, std=44-53
- post_gelu: mean=-1.4 to -1.0, std=0.5-2.3
- post_dropout: mean=-1.4 to -1.0, std=0.6-2.5
- Output range: min=-3.5, max=24.6
- **Pattern: Strongly negative bias with very little positive signal**

**Sequence branch (seq, steps 100-500):**
- pre_linear: mean=-16.2 to -4.4, std=152-619 (growing variance)
- post_linear64: mean=-964 to +211, std=1256-3840 (wide dynamics)
- post_gelu: mean=384-640, std=816-2672
- Output range: min=-11712, max=17920
- **Pattern: Amplitude grows with steps, positive activations emerging after GELU**

## 7. Gradient Flow Status (CRITICAL CONCERN)

**LIVENESS_GRAD (step 2401):**
| Head | Status | RMS Grad |
|------|--------|----------|
| detection_head | ALIVE | 3.67e-02 |
| pose_head | DEAD | 0.00e+00 |
| head_pose_head | ALIVE | 2.43e-02 |
| activity_head | ALIVE | 1.25e-02 |
| **psr_head** | **DEAD** | **0.00e+00** |
| psr_heads h0-h10 | **ALL DEAD** | **0.00e+00** |
| backbone | ALIVE | 5.74e-04 |
| fpn | ALIVE | 3.21e-04 |

**GRAD-NORM (step 2399):**
- backbone: 4.25e-03
- det: 1.68e-01
- hp: 5.51e-01
- act: 5.82e-02
- **psr: 0.00e+00** (ZERO gradient norm)

**KENDALL metric (latest step 2601):**
- det_lv: 0.554 (healthy)
- act_lv: 0.265 (healthy)
- **psr_lv: 0.000** (no signal)
- psr_lv_grad: 3.5838 (gradient present at loss level but not flowing through)

## 8. Summary

**Good news:**
- Training process is alive and making progress
- PSR head IS firing on seq=1 frames (381 events) with non-zero PSR loss
- The seq branch activations show growing dynamic range (std increasing from 41 to 619+, outputs up to 17920)
- Other heads (detection, activity, head_pose) are healthily alive
- GPU utilization is good (83-91%)

**Bad news (ongoing issue):**
- **Gradient flow to PSR head is ZERO** - all 11 sub-heads remain DEAD with RMS=0
- Despite PSR loss being non-zero on seq frames, gradients do not reach the PSR head parameters
- This suggests a stop-gradient or detach operation somewhere between the loss computation and the PSR head
- pose_head also DEAD (RMS=0.00e+00) - affected by same mechanism?
- The HP_PREC_CAP is active, meaning pose precision is being capped due to grad starvation

**Comparison to Agent-66 reading:**
- Still epoch 24, progressed from step 1770 to 2607 (+837 steps / ~30 minutes of training)
- Activation count (381 seq=1 events) is consistent with earlier rate
- Gradient flow status UNCHANGED - same DEAD status for psr_head
- PSR loss values on seq frames remain in the same range (5-25)

## 9. Recommendation

The core issue remains: the PSR head's gradient path is disconnected. The head fires and produces loss, but the loss signal cannot reach the head parameters. This must be fixed at the code level - either a misplaced `.detach()`, stop-gradient operation, or incorrect loss graph connection for the PSR head's sequence-based frame selection.

---

*Generated by Agent-75 at 2026-07-07T12:58 UTC*
