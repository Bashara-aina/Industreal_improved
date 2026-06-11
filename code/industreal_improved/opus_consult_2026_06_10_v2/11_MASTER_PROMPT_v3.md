# Master Prompt v3 — Final Opus Audit (2026-06-11)

## Context

This is a follow-up to `10_OPUS_ANSWER_v2.md` (your previous answer). We have implemented every change you prescribed. This document lists what was implemented, where, and with what modifications. The ask: **a final audit of the entire implementation** — what did we miss, what is wrong, what needs improvement.

## What Changed Since v2

All 11 patches (P1-P11) were already applied before v2. The v2 answer identified 3 new root causes (RC-25, RC-26, RC-27) and prescribed concrete code changes. All have been implemented:

### RC-25: Feature-Magnitude Explosion (FIXED)

**FPN reinit in `_reinit_dead_heads()`** — `src/training/train.py:1668-1684`
- Reinitializes 8 FPN Conv2d modules with Kaiming-uniform (a=1): lateral_c3/c4/c5, smooth_p3/p4/p5, p6_conv, p7_conv
- `assert fpn_reinit == 8` prevents silent no-op from renamed modules
- RC-14 fix applied: uses `cls_subnet`/`reg_subnet` (actual attribute names), not `cls_tower`/`reg_tower`

### RC-27: GroupNorm in Detection Subnets (FIXED)

**`src/models/model.py:503`** — `nn.GroupNorm(8, 256)` added inside `make_subnet()`, affects both cls_subnet and reg_subnet.

### det_conf Zeroing During Recovery (FIXED)

- **`src/config.py:252`** — `ZERO_DET_CONF_FOR_RECOVERY = False` module-level flag
- **`src/config.py:535-553`** — `'recovery'` preset with `zero_det_conf: True`, FP32, batch=1, grad_accum=32
- **`src/models/model.py:1808`** — gated zeroing: `if C.ZERO_DET_CONF_FOR_RECOVERY: det_conf = torch.zeros_like(det_conf)`

### Step-0 Assertion (FIXED — 2 layers)

**Layer 1: Pre-training check** — `src/training/train.py:2635-2650`
- Runs BEFORE `train_one_epoch` when `--reinit-heads` is set
- Single forward pass, checks `cls_logits.abs().median() < 8.0`
- Raises `RuntimeError` (not just logs) on failure

**Layer 2: In-training permanent guard** — `src/training/train.py:1096-1116`
- Inside `train_one_epoch`, at step 0, after `criterion(outputs, targets)`
- **Unconditional**: `assert cls_loss < 1e4` — catches trunk explosion on every training run
- **Conditional**: `if _REINIT_HEADS_ACTIVE: assert cls_logits.abs().median() < 8` — extra gate when reinit is active

**Module flag** — `src/training/train.py:140`
- `_REINIT_HEADS_ACTIVE = False` set to `True` at line 2602 when `--reinit-heads` is active

### EMA Re-Anchor After Reinit (FIXED)

**`src/training/train.py:2608-2615`** — After `_reinit_dead_heads()`, clones head/FPN params from model into EMA shadow, resets Kendall log_vars to neutral (0.0).

### `--reinit-heads` CLI Argument (FIXED)

**`src/training/train.py:3570`** — `--reinit-heads` flag, `action='store_true'`

### Recovery Preset (FIXED)

**`src/config.py:535-553`** — `'recovery'` preset: manual_only, batch=1, grad_accum=32, zero_det_conf=True, staged=False, mixed_precision=False
- **`src/config.py:574-586`** — `apply_preset()` updated to handle new globals (`ZERO_DET_CONF_FOR_RECOVERY`, `STAGED_TRAINING`, `MIXED_PRECISION`)

### D7-D9 Diagnostic Scripts (FIXED)

- **`code/diag_feature_magnitude.py`** (D7) — FPN output magnitude comparison (fresh vs ckpt), RMS collapse factor per level
- **`code/diag_step0_logits.py`** (D8) — Step-0 cls_logit percentiles, % saturated, per-class stats
- **`code/diag_weight_norms.py`** (D9) — Per-layer weight-norm ratios grouped by component

## Audit Request

Please do a final comprehensive audit. Specifically:

1. **Trace every change from `10_OPUS_ANSWER_v2.md` section 3 (Revised Recovery Plan) through sections 5-6 against the actual source code.** Read the relevant source files to verify correctness — do NOT rely on this summary alone.

2. **Check the `_reinit_dead_heads()` function** (train.py:1654-1780) — are all reinit parameters correct? Especially:
   - Detection head: pi=0.05 prior, cls_score std=0.01, reg_pred std=0.01
   - Activity head: proj_features std=0.02, cls_token std=0.02, classifier bias=-0.5
   - PSR head: per_frame_mlp, output_heads bias=-0.2
   - FPN: Kaiming-uniform a=1

3. **Check the step-0 assertion implementation** — does the unconditional `cls_loss < 1e4` check cover all code paths? Is `loss_dict['det_cls']` always populated?

4. **Check the GroupNorm integration** — is placing GroupNorm AFTER each Conv3x3+ReLU in `make_subnet()` correct? Should it be BEFORE ReLU?

5. **Check config.py `apply_preset()`** — does the recovery preset actually set all the right globals? Are there any missing preset keys?

6. **Check the EMA re-anchor logic** — is cloning from model after reinit correct, or should EMA be rebuilt from scratch?

7. **Any edge cases or silent failures?** Particularly:
   - What happens if `cls_preds` is absent from outputs at step 0?
   - What if `loss_dict` doesn't contain 'det_cls' or 'cls'?
   - What if the dataloader is empty?

8. **Overall verdict:** Is this implementation safe to run a recovery training? If not, what specifically must change?

## Source Files to Audit

- `src/training/train.py` — primary target (~3600 lines, all changes in the top ~140 lines and ~1090-1120, ~1650-1780, ~2595-2650)
- `src/models/model.py` — GroupNorm at line 503, det_conf zeroing at line 1808
- `src/config.py` — ZERO_DET_CONF_FOR_RECOVERY at 252, recovery preset at 535, apply_preset at 574
- `code/diag_feature_magnitude.py` — D7
- `code/diag_step0_logits.py` — D8
- `code/diag_weight_norms.py` — D9

Read each file before reporting. Report every issue with exact line numbers.
