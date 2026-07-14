# Pre-Launch Config Review — 2026-07-14

Per 30_DAY_EXECUTION_PLAN.md Day-1 checklist.

## 1. FREEZE_BACKBONE (config.py:199)

**Current value:** `True`

**Context:** V2 plan specifies default `False` for main MTL (multi-task fine-tune). The 2pct training fix branch (`auto/2pct-training-fix-20260520-202419`) should flip this. A dedicated script `scripts/train_finetune_backbone.sh` already exports `FREEZE_BACKBONE=False` explicitly.

**Verdict:** Config default is still `True` (linear probe mode). Flip to `False` for the MTL training path.

## 2. CUBLAS_WORKSPACE_CONFIG

**Python entry points (correct format `:4096:8`):**
- `src/training/train.py:11` — `setdefault(':4096:8')` (before torch import)
- `src/training/train.py:121` — hard override `= ':4096:8'` (after C imported)
- `src/quick_eval.py:33` — `setdefault(':4096:8')`
- `scripts/train_psr_repair_wrapper.py:18` — `setdefault(':4096:8')`

**Shell scripts (MISSING leading colon — uses `4096:8`):**
- `scripts/run_smoke_fp32.sh:30`
- `scripts/run_reinit_fp32.sh:34`
- `scripts/run_smoke_fp32_100.sh:31`
- `scripts/run_reinit_bf16.sh:34`
- `scripts/run_overfit_50img_cls.sh:24`
- `scripts/run_recovery_retrain_25pct.sh:64`
- `scripts/run_reinit_5pct_3ep.sh:37`
- `scripts/run_eval_latest_p200.sh:61`
- `scripts/run_eval_post_retrain_fp32.sh:62`
- `scripts/run_reinit_fp32_bs2.sh:35`
- `src/run_3060_diagnostics.sh:32` (correct `:4096:8`)
- `src/run_5060ti_training.sh:32` (correct `:4096:8`)

**Issue:** The leading colon is significant for cuBLAS. Most shell scripts use `4096:8` instead of `:4096:8`.

## 3. Split File Path

**Reference:** `data/splits/industreal_final_split.json`

**Status:** File does NOT exist on disk. Referenced in `223_EXPERIMENTAL_PROTOCOL.md` (Doc 223 Section 8) as the serialized single split from `SEED_DATA = 42`. Must be generated before any training run.

## 4. Action Items

| # | Item | Priority |
|---|------|----------|
| 1 | Flip `FREEZE_BACKBONE = False` in `config.py:199` | HIGH — blocks MTL training |
| 2 | Fix leading colon in all 11 shell scripts using bare `4096:8` | MEDIUM — non-determinism risk |
| 3 | Generate `data/splits/industreal_final_split.json` | HIGH — missing, needed by all experiments |
