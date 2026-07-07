# DETACH_PSR_FPN Audit

**Date:** 2026-07-07
**Agent:** DETACH_PSR_FPN Audit + Cleanup Specialist
**Context:** File-157 deeper investigation — PSR gradient dead issue

## Summary

`DETACH_PSR_FPN` controls whether PSR head gradients flow into the shared FPN backbone. When True (the default in all 14 presets), PSR features are `.detach()`-ed before entering the PSR head, producing zero backbone gradient from PSR loss. This prevents PSR loss spikes from corrupting detection features, but means PSR never learns shared representations.

## 1. All References to DETACH_PSR_FPN

### 1a. Source code (active)

| File | Line(s) | Role |
|------|---------|------|
| `src/config.py` | 1068-1076 | **Default value**: `True` (env-overridable via `DETACH_PSR_FPN` env var) |
| `src/config.py` | 2120 | `global DETACH_REG_FPN, DETACH_PSR_FPN` — declared global for `apply_preset()` |
| `src/config.py` | 2155 | `DETACH_PSR_FPN = bool(preset.get('detach_psr_fpn', DETACH_PSR_FPN))` — preset override |
| `src/models/model.py` | 2101-2107 | **Sequence path**: `if getattr(C, 'DETACH_PSR_FPN', False): p3_t = p3_t.detach() ...` |
| `src/models/model.py` | 2157-2161 | **Non-sequence path**: `if getattr(C, 'DETACH_PSR_FPN', False): psr_pyramid = {k: v.detach() if k in ('p3','p4','p5') ...}` |
| `src/training/train.py` | 5713-5715 | `--detach-psr-fpn` CLI arg forces `C.DETACH_PSR_FPN = True` |
| `src/training/train_singletask_psr.py` | 132-133 | Same: `--detach-psr-fpn` CLI arg forces `C.DETACH_PSR_FPN = True` |
| `src/training/train_singletask_detection.py` | 116-117 | Same: `--detach-psr-fpn` CLI arg forces `C.DETACH_PSR_FPN = True` |
| `src/training/stage_manager.py` | 1633-1634 | Passes `--detach-psr-fpn` if stage_cfg has `detach_psr_fpn: True` |
| `scripts/train_psr_repair_wrapper.py` | 39-43 | **[FIX F-1]** Post-preset override: forces `C.DETACH_PSR_FPN = False` when env var says False |
| `scripts/train_psr_repair_v3.sh` | 27 | Launches with `DETACH_PSR_FPN=False` env var |
| `tests/test_fable_consult_fixes.py` | 38 | Test asserts snapshot-restore path exists for `DETACH_PSR_FPN=False` |
| `diagnostics/grad_cosine_probe.py` | 16, 118, 171 | Diagnostic: verifies PSR shows zero backbone gradient when True |

### 1b. Presets (all set `detach_psr_fpn: True`)

| Stage Preset | Line in config.py |
|-------------|-------------------|
| `stage_rf1` | 1452 |
| `stage_rf2` | 1515 |
| `stage_rf3` | 1574 |
| `stage_rf4` | 1619 |
| `stage_rf5` | 1648 |
| `stage_rf6` | 1691 |
| `stage_rf7` | 1724 |
| `stage_rf8` | 1757 |
| `stage_rf9` | 1788 |
| `stage_rf10` | 1834 |
| `stage_recovery_det` | 1872 |
| `stage_recovery_all_heads` | 1914 |
| `stage_recovery_pose_psr` | 1954 |
| `stage_psr_only` | 1994 |

**Total: 14 presets, all True.**

### 1c. Archived / historical

| File | Lines | Notes |
|------|-------|-------|
| `analyses/consult_2026_06_10/code/config.py` | 694, 1006-1484 | Previous version of config.py |
| `analyses/consult_2026_06_10/code/model.py` | 2012-2068 | Previous version of model.py |
| `analyses/consult_2026_06_10/code/stage_manager.py` | 1657 | Previous version of stage_manager.py |
| `analyses/consult_2026_06_10/code/train.py` | 4863-4865 | Previous version of train.py |
| `src/runs/full_multi_task_tma_tbank/logs/resolved_config.json` | 354, 568-1090 | Historical run log (all True) |
| `src/runs/full_multi_task_tma_tbank/checkpoints/config.py` | 999-1875 | Historical checkpoint config |

## 2. Is the Value Actually Set to False in V3?

**YES — for the V3 repair training path only.**

The fix works through three layers:
1. **Env var**: `scripts/train_psr_repair_v3.sh` line 27: `DETACH_PSR_FPN=False`
2. **Wrapper**: `scripts/train_psr_repair_wrapper.py` line 39-43 patches `C.apply_preset()` to force False after any preset override
3. **Config default**: `src/config.py` line 1076 reads the env var at module load time

However, the config.py default is **overridden** by every preset (all 14 set `detach_psr_fpn: True`). Without the wrapper patch, the env var is useless because `apply_preset()` runs after module load and overwrites it. The wrapper's patched `apply_preset()` is the essential fix.

**For normal (non-wrapper) training:** `DETACH_PSR_FPN=True` always — all presets set it, and `--detach-psr-fpn` arg also forces True.

## 3. Is There a Hidden detach() in the PSR Path?

### All `.detach()` calls in `src/models/model.py` relating to PSR:

| Line | Code | Context |
|------|------|---------|
| 2105-2107 | `p3_t.detach()`, `p4_t.detach()`, `p5_t.detach()` | **Guarded** by `DETACH_PSR_FPN` — only fires when True (sequence path) |
| 2158 | `v.detach() if k in ('p3','p4','p5')` | **Guarded** by `DETACH_PSR_FPN` — only fires when True (non-sequence path) |

### Other `.detach()` calls in model.py (not PSR-related):

| Line | Code | Context |
|------|------|---------|
| 562 | `feat.detach() if self.detach_reg_fpn` | Regression FPN detach (separate flag) |
| 696 | `confidence.detach()` | Stop gradient on confidence score (always) |
| 1215-1244 | `feat_i.detach().clone()` | Feature bank storage (always detach stored features) |
| 1706 | `feat_i.detach().clone()` | Cache storage |
| 2057 | `keypoints.detach()` | Pose FiLM conditioning (stop grad on keypoints) |
| 2181 | `head_pose.detach()` | HeadPose FiLM (stop grad on head pose) |
| 2189 | `c5_mod.detach()` | Activity gradient blend (controlled by `ACTIVITY_GRAD_BLEND_RATIO`) |
| 2193 | `pyramid['p4'].detach()` | Spatial feature for head pose (always detach) |
| 2324-2326 | `param.data.clone().detach()` | Shadow weights for EMA |

**Verdict: No hidden detach() in the PSR path.** The only `.detach()` calls on FPN features for PSR are the two explicitly guarded by `DETACH_PSR_FPN`.

## 4. PSR Forward Path Analysis

The PSR forward path in `src/models/model.py`:

```
Forward pass (lines 2080-2164):
  ├── T_main > 1  (sequence path, lines 2090-2154)
  │     ├── Reshape pyramid into per-frame [B, T, C, H, W]
  │     ├── For each frame t:
  │     │     ├── Extract p3_t, p4_t, p5_t
  │     │     ├── [IF DETACH_PSR_FPN: p3_t.detach(), p4_t.detach(), p5_t.detach()]  ← lines 2104-2107
  │     │     ├── GAP each → concat → per_frame_mlp → [B, hidden]
  │     │     └── Collect into frame_feats list
  │     ├── stack → [B, T, hidden]
  │     ├── Causal transformer → [B, T, hidden]
  │     ├── flatten → [BT, hidden]
  │     └── output_heads → [BT, 12]
  │
  └── T_main <= 1 (non-sequence path, lines 2156-2164)
        ├── [IF DETACH_PSR_FPN: detach p3,p4,p5 from pyramid]  ← lines 2157-2159
        └── self.psr_head(pyramid) → [B, 12]
```

The FPN features flow through GAP (adaptive average pooling) which is differentiable. When `DETACH_PSR_FPN=True`, the `.detach()` is applied BEFORE GAP, meaning the PSR head sees zero-gradient features and never contributes to FPN/backbone learning.

## 5. Actual Fix Required

### Fix applied (V3):
- **Wrapper patch** (`scripts/train_psr_repair_wrapper.py:39-43`): Post-preset override forces `DETACH_PSR_FPN=False` when env var says False — essential because presets override the config default.
- **Env var default** (`src/config.py:1076`): Reads `os.environ.get('DETACH_PSR_FPN', 'True')` — allows env override at module load, but doesn't help alone since presets override it.

### Remaining issues (unfixed):
1. **The wrapper is a band-aid**: Normal training (no wrapper) always has `DETACH_PSR_FPN=True` because all 14 presets set it. Normal training produces zero PSR backbone gradient.
2. **Config default `True`**: Line 1076 defaults to True. Any training without explicit `DETACH_PSR_FPN=False` env var will detach.
3. **No stage_manager path for False**: Stage manager only passes `--detach-psr-fpn` when the stage has it True. There is no mechanism to deliberately set it False for a stage.

### What a complete fix would need:
- For PSR to learn shared representations, `DETACH_PSR_FPN=False` must be set in the stage_rf4-rf10 presets (or at least for the stages where PSR should converge).
- This must be coordinated with a gradient clipping or loss stabilization strategy to prevent PSR loss spikes from corrupting detection features.
- The `ACTIVITY_GRAD_BLEND_RATIO` pattern (line 2188) could be extended to PSR — a small blended gradient instead of binary detach/allow.

## 6. Count Summary

| Category | Count |
|----------|-------|
| Active source files referencing DETACH_PSR_FPN | 10 files |
| Presets with `detach_psr_fpn: True` | 14 |
| `.detach()` calls on PSR FPN features | 2 (both guarded by DETACH_PSR_FPN) |
| Hidden detach() in PSR path | 0 |
| Fix layers (env + config + wrapper) | 3 |
