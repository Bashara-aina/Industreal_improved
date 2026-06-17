# POPW Fix Report V2

**Date:** 2026-06-04
**Project:** POPW (Procedure-Oriented Procedural Workflow) — IndustReal multi-task assembly recognition
**Code root:** `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/`
**Branch / run name:** `full_multi_task_tma_tbank_benchmark`

---

## 1. Summary

The POPW v2 fix series targets five categories of bugs that have historically broken training
or evaluation. All fixes have been applied and verified by a 16-check smoke suite.

| Category | Bug class | Status |
|---|---|---|
| 1. PSR temporal smooth | `tanh(abs(diff))` collapsed signed information | **Fixed** |
| 2. NaN guard in `combined_metric` | Single inf/nan component poisoned final score | **Fixed** |
| 3. VideoMAE projection in optimizer | Unfrozen params never registered to optimizer | **Fixed** |
| 4. Frame cache bounded | Cache grew unboundedly across batches | **Fixed** |
| 5. EMA shadow load from checkpoint | Replaced vs. merged; leaked stale keys | **Fixed** |

Smoke test result: **16 / 16 checks PASS** (live numerical + source-level).
Smoke test script: `src/smoke_test_fixes.py`.

---

## 2. Fix list with file:line references

### Fix 1 — PSR temporal smooth uses signed `tanh`
- **File:** `src/training/losses.py:1162`
- **Change:** `diff_p = (p_i[1:] - p_i[:-1]).mean()` (no `.abs()` before `tanh`).
- **Rationale:** `abs()` destroys sign, so `tanh` always saw positive input and saturated
  to `+1`, producing a constant-ones loss term that no longer tracks label transitions.
- **Verified by:** `test_psr_temporal_smooth_signed_tanh` (source regex + live oscillating
  tensor). Live `smooth_loss = 0.026559`, finite.

### Fix 2 — NaN guard in `combined_metric`
- **File:** `src/evaluation/metrics.py` (source pattern `math.isfinite|isfinite(|np.isfinite`)
- **Change:** Any single component returning inf/nan is caught by a finite-check guard
  and replaced with a safe finite fallback, so the combined score stays finite.
- **Verified by:** `test_combined_metric_nan_guard`:
  - `F1_psr=inf` → `combined=0.5000` (finite, not inf)
  - `F1_psr=nan` → `combined=0.5000` (finite, not nan)
  - source guard present.

### Fix 3 — VideoMAE projection added to optimizer
- **Files:**
  - `src/training/train.py:2357` — `opt_params = model.videomae_stream.unfreeze(lr=videomae_lr); optimizer.add_param_group(opt_params[0])`
  - `src/training/train.py:2371` — `activity_head.videomae_proj` params added via second `add_param_group` at `head_lr`.
- **Rationale:** Without these, the unfrozen VideoMAE encoder and the freshly-activated
  `activity_head.videomae_proj` would pass gradients with `requires_grad=True` but
  receive no optimizer updates.
- **Verified by:** `test_videomae_proj_in_optimizer` — 4 stub encoder params confirmed
  present in `optimizer.param_groups` with `requires_grad=True`.

### Fix 4 — Frame cache bounded by one-shot preload
- **File:** `src/data/industreal_dataset.py`
  - `:159` — `FRAME_CACHE: Dict[str, np.ndarray] = {}` (module-level)
  - `:160` — `_FRAME_CACHE_LOADED = False` (singleton guard)
  - `:182-185` — `if _FRAME_CACHE_LOADED: return` short-circuit
  - `:256, 261, 267` — three write sites, all inside `preload_all_frames`
- **Verified by:** `test_frame_cache_bounded`:
  - module-level dict declared;
  - one-time preload guard present (`if _FRAME_CACHE_LOADED:` + `_FRAME_CACHE_LOADED = True`);
  - 3 write sites all inside preload;
  - 10 000 read-only `cache.get` calls leave size stable at 100.

### Fix 5 — EMA shadow weights load correctly
- **File:** `src/training/train.py:2172-2180`
- **Change:**
  ```python
  ema_key = 'ema_state' if 'ema_state' in ckpt else 'ema_shadow'   # accept both keys
  if ema is not None and ema_key in ckpt and ckpt[ema_key]:
      ema.shadow.update({                                          # MERGE, not replace
          k: v.to(ema.device) if ema.device else v
          for k, v in ckpt[ema_key].items()
          if k in ema.shadow                                        # filter to live keys
      })
  ```
- **Rationale:** Prior implementation replaced `ema.shadow` outright, dropping any
  parameter newly present in the model but absent in the older checkpoint, and pulled
  in stale keys (e.g., from previous ablation runs).
- **Verified by:** `test_ema_shadow_weights_load`:
  - both key names (`ema_shadow` / `ema_state`) referenced in source;
  - `ema.shadow.update()` used (merge);
  - `if k in ema.shadow` filters stale keys;
  - live round-trip restores all 4 shadow weights exactly;
  - `stale_key_that_does_not_exist` does **not** leak into `ema_after.shadow`.

### Supporting fix — STAGE3_WARMUP_EPOCHS
- **File:** `src/config.py:378` — `STAGE3_WARMUP_EPOCHS = 3`
- **Files:** `src/training/train.py:2034` (param-group split), `:2143-2150` (`stage3_warmup_state`),
  `:2323-2334` (activation on Stage 3 entry), `:2507-2525` (per-epoch LR ramp).
- **Effect:** When Stage 3 starts (epoch 16), `activity_head + psr_head` are scaled by
  `(epoch - stage3_start + 1) / STAGE3_WARMUP_EPOCHS` over the next 3 epochs, then settle
  to `head_lr = 5e-4`. Prevents gradient blow-up on newly-unfrozen heads.

---

## 3. Smoke test output (live run, 2026-06-04)

```
POPW Fix Smoke Tests -- 5 fix categories
PROJECT_ROOT: /media/newadmin/master/POPW/working/code/industreal_improved_to_archive
SRC_DIR     : /media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src
======================================================================

  1. PSR temporal smooth -- signed tanh, finite on oscillating labels
    [PASS] PSR temporal smooth: signed-tanh in source
    [PASS] PSR temporal smooth: finite on oscillating labels -- smooth_loss = 0.026559

  2. NaN guard in combined_metric
    [PASS] combined_metric: finite output when F1_psr=inf  -- combined=0.5000
    [PASS] combined_metric: finite output when F1_psr=nan  -- combined=0.5000
    [PASS] combined_metric: source has finite-check         -- matched pattern: math.isfinite

  3. VideoMAE projection in optimizer (videomae_stream unfreeze)
    [PASS] VideoMAE optimizer wiring: source check
    [PASS] VideoMAE optimizer: encoder params in optimizer.param_groups -- 4 encoder params

  4. Frame cache bounded (no unbounded growth)
    [PASS] Frame cache: FRAME_CACHE module-level dict
    [PASS] Frame cache: one-time preload guard
    [PASS] Frame cache: writes scoped to preload -- 3 write site(s)
    [PASS] Frame cache: read-only access does not grow cache -- size stable at 100

  5. EMA shadow weights load correctly from checkpoint
    [PASS] EMA shadow load: source check
    [PASS] EMA shadow load: uses ema.shadow.update()
    [PASS] EMA shadow load: filters by existing keys
    [PASS] EMA shadow load: values restored after load -- 4 keys restored
    [PASS] EMA shadow load: stale keys filtered

======================================================================
16/16 checks passed
ALL CHECKS PASSED
```

Reproduce locally:
```bash
cd /media/newadmin/master/POPW/working/code/industreal_improved_to_archive
python3 src/smoke_test_fixes.py
```

---

## 4. Restart instructions

Run the pre-staged 25% subset, 31-epoch restart from the existing crash-recovery checkpoint:

```bash
cd /media/newadmin/master/POPW/working/code/industreal_improved_to_archive && bash src/run_restart_25pct.sh
```

What the script does:

- `PROJ_DIR = src/`
- Resumes from `src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth`
- Trains on a 25% subset (`--subset-ratio 0.25`)
- 31 max epochs (`--max-epochs 31`)
- `--no-staged-training` — single-stage run (Stage 3 effectively from epoch 0)
- `--num-workers 0` — prevents /dev/shm OOM on small containers
- Logs to `src/runs/full_multi_task_tma_tbank_benchmark/logs/restart_25pct_15ep.log`

The full set of structural fixes active in the resumed run (also documented in
`src/run_restart_25pct.sh:5-10`):

1. Kendall clamp removed (`losses.py:1264`).
2. DET cls bias `pi=0.10` (`model.py:526`).
3. `DET_POS_IOU_THRESH=0.3` via `C.DET_POS_IOU_THRESH` (`losses.py:862`).
4. `CB_BETA=0.99` (`config.py:343`).
5. 50× weight-ratio cap in `ClassBalancedFocalLoss` (`losses.py`, after `forward`).
6. PSR Stage 3 warmup ramp (`losses.py:1101`).

---

## 5. Files touched in V2 series

| File | Lines (approx) | Purpose |
|---|---|---|
| `src/training/losses.py` | 1162, 1101, 1264 | PSR signed smooth, PSR warmup, Kendall clamp |
| `src/evaluation/metrics.py` | math.isfinite guard | combined_metric NaN guard |
| `src/training/train.py` | 2034, 2143-2150, 2323-2334, 2357, 2371, 2507-2525 | Param-group split, stage3 warmup state, videomae unfreeze, EMA shadow load |
| `src/data/industreal_dataset.py` | 159, 160, 182-185, 256/261/267 | FRAME_CACHE bounded |
| `src/config.py` | 378, 343 | STAGE3_WARMUP_EPOCHS=3, CB_BETA=0.99 |
| `src/run_restart_25pct.sh` | full file | Restart driver |

---

## 6. Sign-off

All five V2 fix categories verified by 16/16 smoke checks. Resume run scripted at
`src/run_restart_25pct.sh`. STAGE3_WARMUP_EPOCHS wired through to per-epoch LR ramp
in `train.py:2507-2525`. Ready to relaunch.
