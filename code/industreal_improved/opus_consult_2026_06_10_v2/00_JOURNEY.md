# 00 — Journey: From the First Opus Analysis to Now

> **Purpose:** This document is the *delta* between the first opus consultation
> (`opus_consult_2026_06_10/`, dated morning of 2026-06-10) and the present
> state (~20:30 2026-06-10 evening). It assumes the reader has NOT read the
> first opus, so it stands alone.

---

## TL;DR of the journey

1. **First opus analysis (morning):** Identified that the trained model had
   *three* dead heads (det, act, psr) producing 0.0 metrics, plus severe
   artifacts (NaN efficiency, AMP fp16 Inf gradients in backbone, seq-mode
   autograd leak, Kendall log_var drift, 1-D/2-D tensor mismatch in
   `_check_per_class_activity_sanity`). 8 surgical patches proposed.

2. **Patches applied + smoke-tested (mid-morning):** All 8 patches written to
   live source via `apply_popw_fixes.py --apply`. AST-parse and semantic
   presence verified. A 100-step FP32 smoke test completed without
   NaN/explosion.

3. **AMP diagnosis + 2-step loss guard (midday):** Discovered `--no-amp` was
   silently being overridden by the runner. Wrote `diag_amp_nan.py` +
   `diag_amp_2step.py` to confirm the AMP→FP32 path is stable. Added a
   2-step NaN/Inf detection in losses (skip the step + retry with
   un-perturbed grad).

4. **5%-subset retrain with re-init heads (afternoon):** 8-patch retrain
   from the epoch-43 crash_recovery.pth, batch=2, FP32, 2 epochs
   (resuming at 43 → 44, 1.5 hours on RTX 3060). New
   `best.pth` (948 MB) written 19:42.

5. **Post-retrain eval (19:43–19:44):** Loaded the new best.pth with
   `EVAL_SKIP_REINIT=1` to evaluate the trained post-retrain weights as-is
   (no fresh head reinit). **Partial recovery** — see `03_CURRENT_RECOVERY.md`.

6. **Second opus analysis (now):** Asking the opus to explain why
   *det* + *act_top1* are still 0.0 after a successful 2-epoch retrain
   with a verified-clean backbone and a 169-tensor re-init.

---

## Timeline (UTC+9)

| Time        | Milestone                                                              | Evidence                                    |
|-------------|------------------------------------------------------------------------|---------------------------------------------|
| 2026-06-09  | Multiple crash_recovery.pth files across 5%-subset smoke runs         | `runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth.*` |
| 2026-06-10 08:14 | `diag_collapse_3heads.py` confirms det/act/psr all dead         | `code/diag_collapse_3heads.py`              |
| 2026-06-10 08:30 | `diag_features_alive.py` confirms backbone FPN features alive  | `code/diag_features_alive.py`               |
| 2026-06-10 09:20 | First reinit retrain attempt (`run_reinit_5pct_3ep.sh`)        | `logs/reinit_5pct_3ep_*/train.log`           |
| 2026-06-10 16:17 | All 8 surgical patches applied (`apply_popw_fixes.py --apply`)  | `code/apply_popw_fixes.py`                  |
| 2026-06-10 17:27 | 100-step smoke test passes (FP32, batch=2)                     | `logs/smoke_test_3patches_*.log`             |
| 2026-06-10 18:59 | Reinit retrain v2 launched (FP32, batch=2, --reinit-heads)    | `scripts/run_reinit_fp32_bs2.sh`            |
| 2026-06-10 19:00–19:42 | Retrain runs (1.5h on RTX 3060), 1 epoch complete           | `logs/retrain_5pct_fp32_bs2/train.log`      |
| 2026-06-10 19:42 | New best.pth written (948 MB, combined=0.1116)                | `runs/full_multi_task_tma_tbank_benchmark/checkpoints/best.pth` |
| 2026-06-10 19:43–19:44 | Post-retrain eval (`eval_post_retrain_fp32_20260610_194311`) | `evidence/post_retrain_fp32_20260610_194311/metrics.json` |
| 2026-06-10 20:30 | This opus folder written                                         | `00_JOURNEY.md` (you are here)              |

---

## What changed between opus 1 and opus 2

### Source code (all under `code/`)

| File          | Lines | Notable changes from opus 1                             |
|---------------|-------|---------------------------------------------------------|
| `train.py`    | 3733  | FIX-1 (1-D/2-D sanity check), FIX-5 (3 early `continue` paths), `_reinit_dead_heads` helper, 2-step NaN guard |
| `model.py`    | 2167  | FIX-3/4 (sequence detection: read T from tensor shape, not `_seq_len` tag) |
| `losses.py`   | 1505  | FIX-6/7 (NaN guard, temporal-smooth no longer negates)   |
| `evaluate.py` | 4004  | FIX-2 (accept 1-D per-class-accuracy), DET_PROBE noise (score per-batch IoU stats), AS/EV/procedure-step sections |
| `eval_post_reinit.py` | 130 | FIX-8 (USE_HEADPOSE_FIM → USE_HEADPOSE_FILM), `EVAL_CKPT` env var, `EVAL_SKIP_REINIT` env var |
| `config.py`   | ~890  | No code changes; `OUTPUT_ROOT` still defaults to `src/runs/full_multi_task_tma_tbank_benchmark` (KEY INVARIANT) |

### Metrics

| Metric (val 200b, batch=4) | Baseline (pre-retrain, _reinit heads) | Post-retrain (FP32 bs2) | Delta |
|----------------------------|---------------------------------------|-------------------------|-------|
| `loss`                     | 227.7                                 | 72.9                    | **−68%** |
| `act_top5_accuracy`        | 0.00                                  | 0.06                    | **+0.06** |
| `act_top1_accuracy`        | 0.00                                  | 0.00                    | 0.00 (still flat) |
| `act_macro_f1`             | 0.00                                  | 0.00                    | 0.00 |
| `psr_overall_f1`           | 0.00                                  | 0.0909                  | **+0.0909** (1 component) |
| `psr_edit_score`           | 0.0909                                | 0.7273                  | **+0.6364** (8×) |
| `det_mAP50`                | 0.00                                  | 0.00                    | 0.00 (still TOTAL COLLAPSE in 12/50 batches) |
| `det_mAP50_all_frames`     | 0.00                                  | 0.00                    | 0.00 |
| `position_MAE_mm`          | 739.5                                 | 823.5                   | **+11% worse** |
| `head_pose_angular_MAE_deg`| 61.04                                 | 71.50                   | **+17% worse** |
| `forward_x_MAE`            | 0.1051                                | 0.1821                  | **+73% worse** |
| `forward_angular_MAE_deg`  | 64.28                                 | 68.65                   | +7% worse |
| `up_angular_MAE_deg`       | 57.80                                 | 74.34                   | +29% worse |

**Net result:** Loss dropped 68%, PSR moved, act_top5 moved. But activity
top-1 + detection mAP50 are still 0.0, and head-pose regression worsened.

---

## What we know is fixed (do NOT re-investigate)

These were investigated in opus 1 and verified fixed. The opus should
NOT re-open them; it should treat them as background:

- ✅ Backbone FPN features alive (per-image variance 0.032–0.036 in DET logits)
- ✅ AMP fp16 Inf gradients in backbone first layers (FP32 retrain proven stable)
- ✅ Seq-mode autograd leak (FIX-3: only treat `dim()==5` inputs as sequences)
- ✅ Kendall log_var drift (clamped to ±5)
- ✅ `_check_per_class_activity_sanity` 1-D/2-D crash (FIX-1: pass confusion matrix, wrap in try/except)
- ✅ NaN_GUARD semantics (FIX-6/7: skip-step, not grad-zero)
- ✅ `USE_HEADPOSE_FIM` typo (FIX-8: now `USE_HEADPOSE_FILM`)
- ✅ Adam m/v optimizer state resume spike (transient, recovers in 20–50 steps — see `04_HYPOTHESES_FOR_OPUS.md` §"What we ruled out")

---

## What we want the opus to investigate

The full list is in `04_HYPOTHESES_FOR_OPUS.md`. The three highest-priority
questions are:

1. **Why is `det_mAP50` still 0.0 after reinit + 1 epoch?** The DET_PROBE
   shows scores up to 0.97 (model IS producing confident predictions) but
   bestIoU never exceeds 0.27. Is the decoder (or box post-processing)
   broken? Is the threshold too high? Is the loss not actually supervising
   localization?

2. **Why is `act_top1` still 0.0 when `act_top5` is 0.06?** Top-5 is barely
   above random (0.05 = 1/20). The activity head must be stuck on a few
   classes. Is it LDAM-DRW underflow? Class-imbalance dominance? Or did
   the reinit not actually reinit all activity-related tensors?

3. **Why did head-pose regression get worse?** Pose MAE went UP 11–73%
   after reinit. Is the new pose head under-trained? Did the reinit include
   the head-pose MLP when it shouldn't have, or *not* include it when it
   should have?

---

## Pointers to other docs in this folder

- `01_WHAT_WE_BUILT.md` — current state of all 8 patches + scripts + env
- `02_COLLAPSE_CRISIS.md` — refined analysis of the 3-dead-head collapse
- `03_CURRENT_RECOVERY.md` — what the retrain recovered vs. what is still broken
- `04_HYPOTHESES_FOR_OPUS.md` — prioritized hypothesis list with the evidence
  we already gathered (so opus can pick up where we left off)
- `05_MASTER_PROMPT.md` — the master opus analysis prompt
