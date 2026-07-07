# PSR Repair V3 — Gradient Status Report

**Date:** 2026-07-07
**Analyst:** Agent 96 (F-1 V3 Gradient Verification Specialist)
**Commit:** 9198d1c26fe6c2cfc13b11458b2b01efd3f3a547
**Source:** `/tmp/train_psr_v3_real.log` (run started 16:50 JST, active at 2000+ steps)

## DETACH_PSR_FPN Status: CORRECTLY SET TO False

The wrapper correctly exports and logs the override:

```
[wrapper] Post-preset override: DETACH_PSR_FPN=False (per env var, PSR gradient flow to backbone ENABLED)
```

Resolved config confirms `DETACH_PSR_FPN = false` in the saved JSON. This fix from commits 59f84c3d4/ea6ac30c is correctly applied and active.

## PSR Gradient Status: STILL DEAD (No Improvement)

Despite DETACH_PSR_FPN=False, the PSR head receives **zero gradient** on every measurement across >2000 training steps:

| Step | psr_head RMS | n | Status |
|------|-------------|---|--------|
| 1 | NO_GRAD | 0 | N/A (pre-seq) |
| 201 | 0.00e+00 | 88 | DEAD |
| 401 | 0.00e+00 | 88 | DEAD |
| 601 | 0.00e+00 | 88 | DEAD |
| 801 | 0.00e+00 | 88 | DEAD |
| 1001 | 0.00e+00 | 88 | DEAD |
| 1801 | 0.00e+00 | 88 | DEAD |
| 2001 | 0.00e+00 | 88 | DEAD |

All 11 per-component output heads show `h0-h10=0.00e+00[DEAD]` consistently.

GRAD-NORM at optimizer steps 399, 799, 1999 confirms `psr=0.00e+00`.

LIVENESS at criterion step 2000 shows `psr=1.00e-06 DEAD`.

## PSR Activation is Healthy (FIXED)

The post_gelu activations are in a healthy range (~4000-4800 mean), NOT the -130 seen before the LeakyReLU + weight init fix:

| Step | pre_linear mean | post_gelu mean | post_gelu min | post_gelu max |
|------|----------------|----------------|---------------|---------------|
| 0 | 57.7 | 4864 | -516 | 74752 |
| 1 | 51.9 | 4448 | -528 | 73216 |
| 10 | 123.2 | 4608 | -494 | 72704 |
| 100 (seq) | 84.1 | 4704 | -556 | 78336 |
| 200 (seq) | 105.4 | 4480 | -524 | 75264 |
| 500 (seq) | 132.2 | 4480 | -552 | 79360 |

This confirms the LeakyReLU and weight initialization fixes from V3 are working correctly.

## PSR Loss IS Computed on Sequence Batches

Seq batches (every 4 steps, `PSR_SEQ_EVERY_N_BATCHES=4`) produce non-zero PSR loss values. Examples from the log at epoch 27:

- step 1992: `psr=20.453 seq=1`
- step 1996: `psr=12.362 seq=1`
- step 2000: `psr=12.362 seq=1`
- step 2052: `psr=12.362 seq=1`
- step 2056: `psr=7.874 seq=1`
- step 2060: `psr=9.140 seq=1`

PSR loss ranges from ~5.7 to ~23 throughout the run, with `PSR_WEIGHT=10.0` applied. The loss is non-zero and varying across batches, confirming data is flowing.

## Other Heads Training Normally

Detection head: ALIVE (RMS 3-6e-02)
Activity head: ALIVE (RMS 1-3e-02)
Head pose head: ALIVE (RMS 0.5-2e-02)
Pose head: DEAD (RMS 0.00e+00) — this is expected (body-pose branch frozen)
Backbone: STARVED — expected with FREEZE_BACKBONE=True (linear probe mode)
FPN: ALIVE (RMS 0.2-4e-04)

## Key Config Parameters

| Parameter | Value |
|-----------|-------|
| DETACH_PSR_FPN | False |
| PSR_WEIGHT | 10.0 |
| USE_PSR_TRANSITION | True |
| PSR_SEQ_EVERY_N_BATCHES | 4 |
| PSR_SEQUENCE_LENGTH | 8 |
| KENDALL_FIXED_WEIGHTS | True |
| FREEZE_BACKBONE | True |
| MIXED_PRECISION | True |

## Root Cause Analysis

The gradient is consistently zero despite:
1. DETACH_PSR_FPN=False (file-157 F-1 fix, confirming no feature-level detach)
2. Healthy PSR activations (post_gelu ~4000+, not -130)
3. Non-zero PSR loss on seq batches (7-23 range)

The most likely root cause is that the seq batch backward is **silently skipped** at the gradient check in the training loop (line 1359 of train.py):

```python
if not torch.isfinite(loss_seq) or not loss_seq.requires_grad:
    nan_skips += 1
    optimizer.zero_grad(set_to_none=True)  # Zeros PSR grads!
    ...
    continue
```

This check silently increments `nan_skips` (no warning message unlike non-seq NaN skips) and calls `optimizer.zero_grad(set_to_none=True)`, which zeroes ALL gradients including any existing PSR accumulation from previous batches.

The condition `not loss_seq.requires_grad` would be True if the total loss tensor from the criterion somehow loses its gradient chain. This could happen if:
- A `torch.no_grad()` context, `.detach()`, or in-place operation in the criterion's forward method accidentally breaks the autograd graph between `outputs['psr_logits']` and the returned `total`.
- The binary_focal_loss or the Kendall assembly path has a bug that detaches the loss tensor.

Since the PSR loss values DO appear in the progress bar (from `loss_dict_seq['psr'] = loss_seq.item()` at line 1344-1345, which executes in the `else` branch of the finite check BEFORE the requires_grad check), the loss computation IS running but the backward is silently skipped.

**Recommendation:** Add a diagnostic `logger.warning` to check `loss_seq.requires_grad` at line 1359 so we can confirm or rule out this root cause. Also add a `logger.warning` for backwards NaN guard skips to the seq path (similar to the non-seq path's `[BAD_SAMPLE]` message).

## Conclusion

The `DETACH_PSR_FPN=False` fix IS correctly applied and active. The PSR activation is healthy with post_gelu ~4000+ (not -130). However, the gradient still does not flow -- the PSR head remains **DEAD** with RMS=0.00e+00 across all 11 output heads through 2000+ training steps. The root cause appears to be a silent backward skip or autograd graph disconnect in the sequence batch training path, which is independent of the DETACH_PSR_FPN fix.
