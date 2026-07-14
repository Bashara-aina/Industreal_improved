# Config Flag Audit: `AAIML_SUBMISSION_CHECKLIST.md` vs `src/config.py`

**Audit date:** 2026-07-14
**Checker:** Claude Code
**Scope:** 33 flag names (20 grouped items) from AAIML submission checklist, verified against `src/config.py` (2409 lines).

## Results Summary

| Status | Count |
|--------|-------|
| Present with correct default | 31 |
| Missing (added by this audit) | 2 |
| **Total** | **33** |

## Flags Present (31)

| Flag | Default | config.py Line |
|------|---------|----------------|
| `USE_KENDALL` | `True` | 54 |
| `KENDALL_FIXED_WEIGHTS` | env-default `False` | 129 |
| `KENDALL_HP_PREC_CAP` | `True` | 122 |
| `USE_LDAM_DRW` | `True` | 1162 |
| `LDAM_DRW_EPOCH` | `50` | 1165 |
| `USE_PSR_TRANSITION` | `True` | 1318 |
| `PSR_TRANSITION_SIGMA` | `3.0` | 1319 |
| `USE_GEO_HEAD_POSE` | `True` | 1334 |
| `KENDALL_LOG_VAR_MIN_ACT` | `-0.5` | 1195 |
| `KENDALL_LOG_VAR_MAX_PSR` | `0.0` | 1198 |
| `KENDALL_LOG_VAR_MAX_POSE` | `3.0` | 1199 |
| `USE_UW_SO` | env-default `False` | 68 |
| `UW_SO_TEMPERATURE` | `1.0` | 71 |
| `USE_ASL_PSR` | `False` | 49 |
| `USE_BALANCED_SOFTMAX_ACT` | `False` | 1001 |
| `PSR_LR_MULTIPLIER` | `0.5` | 1069 |
| `HEAD_POSE_LR_MULTIPLIER` | `0.3` | 1070 |
| `EMA_START_EPOCH` | `5` | 195 |
| `USE_BIFPN` | `False` | 187 |
| `DET_OHEM_ENABLED` | `True` | 942 |
| `USE_DISTILLATION` | `False` | 1411 |
| `USE_FAMO` | env-default `False` | 55 |
| `USE_METABALANCE` | env-default `False` | 73 |
| `USE_ROTOGRAD` | env-default `False` | 169 |
| `USE_TMA_CELL` | `True` | 214 |
| `USE_TEMPORAL_BANK` | `True` | 215 |
| `USE_VIDEOMAE` | `False` | 199 |
| `USE_HAND_FILM` | `True` | 162 |
| `USE_HEADPOSE_FILM` | `True` | 190 |
| `DETECTION_RESOLUTION` | `224` | 660 |
| `FREEZE_BODY_POSE_BRANCH` | `False` | 83 |

## Flags Missing and Added (2)

| Flag | Default | Rationale | Added At |
|------|---------|-----------|----------|
| `TEACHER_CACHE_DIR` | `'runs/teacher_preds'` | Used via `getattr(C, ...)` fallback in `train.py:4044` but never declared as a module-level constant. | config.py:1412 |
| `SR_SEQUENCE_LENGTH` | `8` | Checklist uses SR prefix; codebase uses `PSR_SEQUENCE_LENGTH = 8`. Added as explicit alias for clarity. | config.py:1297 |
