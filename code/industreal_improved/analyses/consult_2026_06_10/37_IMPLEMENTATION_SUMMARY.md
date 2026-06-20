# 37 — Implementation Summary: Opus v8 Fixes (2026-06-20)

> All four fixes from `36_OPUS_ANSWER_v8.md` implemented across 4 source files.
> Committed as `beda631` on branch `main`, pushed to `industreal/main`.

---

## Fix 1 — De-fang Kendall precision weighting

**Files:** `src/config.py`, `src/training/losses.py`

Two mechanisms, one config toggle:

### Primary: KENDALL_HP_PREC_CAP (default: True)
`losses.py:1531-1533` — clamps `lv_hp >= lv_det.detach()` so head_pose precision can never exceed detection precision. Prevents the Kendall pathology where head_pose (loss ~0.01) gets ~54.6× while detection (loss ~0.5) gets ~1.4×.

### Alternative: KENDALL_FIXED_WEIGHTS (default: False)
`losses.py:1518-1547` — bypasses learned Kendall log_vars entirely. Uses fixed λ=0.2 for head_pose, detection at λ=1.0. Designed for RF1-RF2 bootstrap stages where detection needs to drive the backbone without competition. Toggled per-stage by `stage_manager.py`.

**Verified:** Syntax OK via `ast.parse`. Fixed-weight path is a clean `if/else` branch inside `if self.use_kendall:` — either fixed weights or standard Kendall, never both.

---

## Fix 2 — Feed detection more positives

**Files:** `src/config.py`, `src/training/losses.py`

Three changes targeting anchor coverage:

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `DET_POS_IOU_THRESH` | 0.5 | 0.4 | ~3-5× more anchors clear positive threshold |
| `DET_POS_IOU_TOP_K` | 1 (implicit) | 9 | Top-k force-match per GT → ~6-10 pos/GT |
| `DET_BIAS_LR_FACTOR` | 5.0 | 1.0 | Reverted — 5× was accelerating drift toward dead-feature equilibrium |

**Key detail:** The top-k implementation (`losses.py:129-141`) preserves the original best-anchor assignment and only assigns additional anchors that are still `labels[idx] < 0` (don't overwrite already-positive anchors).

---

## Fix 3 — Kill double curriculum

**Files:** `src/config.py` (documentation only)

**Finding:** The epoch-indexed Kendall staging in `losses.py:1536-1567` was already a no-op because `STAGED_TRAINING=False` at `config.py:518` (guarded by `if bool(getattr(C, 'STAGED_TRAINING', True))` at `losses.py:1558`). The Opus analysis was done on branch `claude/epic-hamilton-6sw452` where this may have been different.

**Action:** Added `KENDALL_STAGED_TRAINING = False` to the new Opus v8 config block as documentation — the actual behavior was already correct.

---

## Fix 4 — Fix phantom 0.45

**Files:** `src/training/stage_manager.py`, `src/runs/rf_stage_state.json`

**Root cause:** `stage_history` for RF1 recorded `best_det_mAP50: 0.45` — byte-identical to the RF3 gate threshold (`stage_manager.py:164`). Actual best metric was 0.184.

**Guard:** `_validate_stage_history_entry()` (`stage_manager.py:548-582`) cross-checks every numeric metric against all known gate/health thresholds from `RF_STAGES`. An exact match triggers a `logger.warning`.

**Wired to all 4 call sites** that append to `state.stage_history`:
- `stage_manager.py:2436` — cmd_check healthy advance
- `stage_manager.py:2633` — near-gate stage completion
- `stage_manager.py:2683` — gate-passed stage completion (primary phantom source)
- `stage_manager.py:2757` — max-epochs stage completion

**State file fix:** `rf_stage_state.json` stage_history entry for RF1 cleaned from phantom 0.45 to actual `best_metric: 0.184` with full metadata.

---

## What Was Not Changed

- **QFL/VFL** — deferred per Opus v8 recommendation (Fix 5, post-recovery quality upgrade). The head needs features first.
- **PSR architecture** — deferred to RF4+ with 50-sample overfit sanity test requirement.
- **No architectural changes** — all fixes are config/loss-level, safe on RTX 3060 12GB.
- **Original Kendall path preserved** — the standard `prec_task * loss_task + lv_task` path is untouched under `else:` for RF3+ use.

---

## Commit

```
beda631 fix(rf): implement Opus v8 fixes for RF2 detection collapse (#9)
 4 files changed, 256 insertions(+), 119 deletions(-)
```
