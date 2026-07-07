# F-10 V3 Process State Verification (File-157 Audit)

## Summary
V3 process (PID 1901736) **IS running the fixed code**. Both the DETACH_PSR_FPN=False fix and the LeakyReLU head repair are active.

## Timeline (2026-07-07, all times JST)

| Event | Time | Details |
|-------|------|---------|
| Commit 59f84c3d4 | 16:49:43 | DETACH_PSR_FPN env var actually takes effect |
| Commit ea6ac30c | 16:50:18 | V3 wrapper forces DETACH_PSR_FPN=False |
| V3 process start | 16:50:36 | Log says "Tue Jul 7 04:50:36 PM JST" |

V3 started **18 seconds after** the last fix commit (ea6ac30c at 16:50:18, V3 at 16:50:36).

## Evidence of Fix in Log (/tmp/train_psr_v3_real.log)

### 1. DETACH_PSR_FPN=False confirmed
```
[wrapper] Post-preset override: DETACH_PSR_FPN=False (per env var, PSR gradient flow to backbone ENABLED)
```

### 2. LeakyReLU + small-normal init + zero bias repair confirmed
Banner at startup:
```
 Preserves:  LeakyReLU + small-normal init + zero bias repair
```

### 3. PSR_DEBUG shows post_gelu values (LeakyReLU is active)
Negative min values confirm LeakyReLU is passing negative activations through (not clipping at zero like ReLU would):

| Step | post_gelu mean | post_gelu std | post_gelu min | post_gelu max |
|------|----------------|---------------|---------------|---------------|
| 0    | 4864.0         | 14912.0       | -516.0        | 74752.0       |
| 1    | 4448.0         | 13568.0       | -528.0        | 73216.0       |
| 10   | 4608.0         | 14336.0       | -494.0        | 72704.0       |
| 100  | 4704.0         | 14144.0       | -556.0        | 78336.0       |
| 200  | 4480.0         | 13568.0       | -524.0        | 75264.0       |
| 500  | 4480.0         | 14080.0       | -552.0        | 79360.0       |

Each step shows negative min values (e.g., -516, -528, -494). With a standard ReLU (clamp at 0), the min would be 0.0. The negative values prove LeakyReLU is active, allowing a small negative slope to pass through.

## Conclusion
**Verdict: V3 is running the fixed code.** No action required. The process started after both fix commits were pushed, and the running log confirms both the DETACH_PSR_FPN=False override and the LeakyReLU head repair are in effect.
