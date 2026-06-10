# 01 — What We Built (Current State)

> **Purpose:** Inventory the current code/scripts/environment as of 2026-06-10
> 20:30. This is the *what is in the repo right now* view, not a historical
> changelog. All files in this folder's `code/`, `scripts/`, and `docs/`
> subdirectories are referenced here by relative path.

---

## 1. The 8 surgical patches (all applied 2026-06-10 16:22)

Patcher: `code/apply_popw_fixes.py`. Exact-string replacements with backup
to `<file>.bak_prefix_20260610`. AST-parse verified; semantic presence
verified.

| ID    | Pri | File             | Symptom it fixes                                                                                                                                  | Anchor location (approx)    |
|-------|-----|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| FIX-1 | P0  | `train.py`       | `_check_per_class_activity_sanity` passed 1-D `act_per_class_acc` to `report_per_class_accuracy` which expected 2-D confusion → numpy AxisError at evaluate.py:869 killed the run BEFORE `latest.pth` was saved (happened 2× in train_digest). | train.py near `_check_per_class_activity_sanity` |
| FIX-2 | P0  | `evaluate.py`    | `report_per_class_accuracy` hardened to accept either 1-D vector or 2-D matrix.                                                                  | evaluate.py:860–890         |
| FIX-3 | P0  | `model.py`       | `forward()` decided "temporal sequence" from persistent `model._seq_len` tag → with `_seq_len=4` set once at startup, ANY 4-D batch divisible by 4 (train bs=4, val bs=16, eval bs=4) was regrouped into FAKE 4-frame sequences of UNRELATED frames. | model.py `forward()` early branch |
| FIX-4 | P0  | `model.py`       | In the sequence path, PSR computes ONE prediction (T→1) but the `expand` broadcast was wrong → gradient was zeroed.                              | model.py PSR forward in seq branch |
| FIX-5 | P1  | `train.py`       | Two of the three early-`continue` paths in the NaN-detection block could skip the rest of the training step but still commit the optimizer state.  | train.py NaN detection block |
| FIX-6 | P1  | `losses.py`      | The final `_safe()` NaN guard set `param.grad = None` for poisoned params — should `param.grad.zero_()` instead so the optimizer step is a no-op. | losses.py `_safe` helper |
| FIX-7 | P1  | `losses.py`      | Temporal-smooth loss negated the label change; harmless only because FIX-4's expand bug zeroed the gradient. After FIX-4 it would actively harm. | losses.py temporal-smooth loss |
| FIX-8 | P1  | `eval_post_reinit.py` | Typo `USE_HEADPOSE_FIM` (missing L) crashed `--train-pose True` runs in eval.                                                              | eval_post_reinit.py constructor call |

### Status

All 8 patches: **APPLIED + AST-PARSE OK + SEMANTICALLY PRESENT** in live
source. No `.bak_prefix_20260610` files were used to roll back any patch.

---

## 2. The patched head re-init (`_reinit_dead_heads`)

Added to `code/train.py` (importable from `training.train`). Re-initializes
**169 head tensors** across:

- **detection head** — all `Conv2d` layers in `cls_head` + `reg_head`, plus
  final `nn.Conv2d` projections
- **activity head** — `proj_features` (Linear) + `vit` (TransformerEncoder,
  3 layers × ~12 tensors) + `cls_token` (Parameter) + final classifier
- **PSR head** — all 11 binary classifiers + the LSTM (2 layers × 8 tensors)
- **head-pose MLP** — final `nn.Linear` layers in pose branch

Re-init strategy: Kaiming-uniform for Conv2d weights, xavier for Linear
weights, zeros for biases. **EMA shadow is also reset** for these 169
tensors so EMA re-tracks the fresh init. **Optimizer state is NOT
reset** — this is the known transient-cls-loss-spike source on resume
(see `04_HYPOTHESES_FOR_OPUS.md` §"Ruled out" for the kill-criteria).

Usage:
```bash
python training/train.py --resume best.pth --reinit-heads --no-amp \
    --max-epochs 44 --subset-ratio 0.05 --batch-size 2 --seed 42
```

---

## 3. New shell scripts (all under `scripts/`)

| Script                                  | Purpose                                                                              | Status   |
|-----------------------------------------|--------------------------------------------------------------------------------------|----------|
| `run_reinit_fp32_bs2.sh`                | 1-epoch reinit retrain, FP32, batch=2 (the SUCCESSFUL one)                            | RUN @ 19:00 2026-06-10, completed 19:42 |
| `run_reinit_fp32.sh`                    | Same but batch=4 (OOM at seq-mode PSR T=4 batch)                                      | ABANDONED |
| `run_reinit_bf16.sh`                    | Reinit retrain in BF16 (didn't help; same collapse after AMP off)                    | ABANDONED |
| `run_reinit_5pct_3ep.sh`                | 3-epoch reinit retrain (predecessor, multiple attempts, all crashed)                 | SUPERSEDED |
| `run_smoke_fp32.sh`                     | 60-step FP32 smoke test, batch=1 (proves NaN/Inf-free)                                | PASSED   |
| `run_smoke_fp32_100.sh`                 | 100-step FP32 smoke test (smoke)                                                     | PASSED   |
| `run_fresh_smoke_patched.sh`            | Fresh smoke after all 8 patches applied                                               | PASSED   |
| `run_smoke_test_3patches.sh`            | 3-patches smoke test (FIX-1/2/3 only)                                                | PASSED   |
| `run_eval_post_retrain_fp32.sh`         | Auto-discover latest retrain ckpt + run eval with `EVAL_SKIP_REINIT=1`                | LIVE, RAN @ 19:43 |

---

## 4. Environment

- **Hardware:** RTX 3060 12GB, 64GB RAM, CUDA_VISIBLE_DEVICES=0
- **OS:** Linux 6.8.0-111-generic, x86_64
- **Python:** 3.13 (miniconda `/home/newadmin/miniconda3`)
- **PyTorch:** 2.x with CUDA 12.x (CUBLAS_WORKSPACE_CONFIG=4096:8 for
  deterministic cuBLAS)
- **Memory tuning:** `MALLOC_ARENA_MAX=4`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `OMP_NUM_THREADS=4` (all 4 BLAS variants)
- **Reproducibility:** `PYTHONHASHSEED=42`, `seed_everything(42)` in train.py
- **Venv:** `/home/newadmin/swarm-bot/.venv` (project's own venv); the
  `code/` files use `/home/newadmin/miniconda3` via PATH

### Key env vars in `run_reinit_fp32_bs2.sh`
```bash
export SUBSET_RATIO=0.05
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=4096:8
export CUDA_LAUNCH_BLOCKING=0   # =1 in eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MAX_STEPS=0        # full run
export EVAL_MAX_BATCHES=20
```

### Key env vars in `run_eval_post_retrain_fp32.sh`
```bash
export EVAL_SPLIT=val
export EVAL_BS=4
export MAX_BATCHES=50
export EVAL_CKPT="$CKPT"        # auto-discovered
export EVAL_SKIP_REINIT=1       # CRITICAL: don't reinit the trained post-retrain heads
export RUN_NAME="eval_post_retrain_fp32_$(date +%Y%m%d_%H%M%S)"
```

---

## 5. The auto-discovery bug we just fixed

`run_eval_post_retrain_fp32.sh` originally auto-discovered the most recent
retrain run via:
```bash
RETRAIN_RUN="$(ls -1d $PROJ/src/runs/reinit_5pct_fp32_* 2>/dev/null | sort | tail -1 | xargs -I{} basename {})"
RETRAIN_CKPT="$PROJ/src/runs/$RETRAIN_RUN/checkpoints/best.pth"
```

**Bug:** The retrain's `$RETRAIN_RUN` is `reinit_5pct_fp32_bs2_20260610_190003/`,
which is **empty** under `checkpoints/` because `train.py` writes to
`OUTPUT_ROOT` (default = `src/runs/full_multi_task_tma_tbank_benchmark/`),
NOT the run's own dir. So the auto-discovery returned an empty string and
the script fell through to hardcoded fallbacks.

**Fix:** Added 3 source-dir fallbacks BEFORE the retrain fallbacks, and
also gave `eval_post_reinit.py` an `EVAL_CKPT` env var to read.

**Final priority:**
1. `$RETRAIN_RUN/checkpoints/best.pth` (in case someone configures OUTPUT_ROOT
   to the run dir in the future)
2. `$SOURCE_CKPT` (full_multi_task_tma_tbank_benchmark/checkpoints/best.pth)
3. `$RETRAIN_RUN/checkpoints/latest.pth`
4. `$SOURCE_LATEST`
5. `$RETRAIN_RUN/checkpoints/crash_recovery.pth`
6. `$SOURCE_CR`

This is documented in `run_eval_post_retrain_fp32.sh` lines 25–47.

---

## 6. The other eval-bug we just fixed

`eval_post_reinit.py` originally:
1. Hardcoded `CKPT = '.../crash_recovery.pth'` (ignored shell auto-discovery)
2. **ALWAYS** called `_reinit_dead_heads(model)` — would destroy the
   trained post-retrain weights!

**Fix:**
- `CKPT = os.environ.get('EVAL_CKPT', '.../crash_recovery.pth')` (default
  is now an env var read)
- `SKIP_REINIT = os.environ.get('EVAL_SKIP_REINIT', '0') == '1'`, and
  the shell sets `EVAL_SKIP_REINIT=1` for the post-retrain eval. The
  reinit branch is still there for fallback / smoke-test use cases.

---

## 7. Diagnostic scripts in `code/`

| Script                          | Purpose                                                                                  | Output                                  |
|---------------------------------|------------------------------------------------------------------------------------------|-----------------------------------------|
| `diag_collapse_3heads.py`       | Quick-eval det/act/psr to confirm collapse pattern.                                       | Per-head metric dump                    |
| `diag_features_alive.py`        | Per-image variance in DET logits on a fresh-init head → proves backbone is alive.       | variance 0.032–0.036 in DET logits     |
| `diag_amp_nan.py`               | Step-by-step AMP vs FP32 with NaN/Inf check at every layer.                              | FP32 stable, AMP fp16 Inf in backbone    |
| `diag_amp_2step.py`             | 2-step forward+backward in AMP then FP32 to find first-failing op.                      | First NaN at backbone.0.conv1.weight   |
| `diag_psr_nan.py`               | Per-component PSR loss / grad NaN check.                                                 | All 11 components non-NaN              |
| `diag_psr_train.py`             | Train-mode PSR forward + grad sanity.                                                    | OK                                      |
| `detection_collapse_probe.py`   | Per-batch DET score/Iou stats (the "DET_PROBE" output).                                   | 12/50 batches TOTAL COLLAPSE verdict    |
| `psr_loss_diagnostic.py`        | Standalone PSR loss with N-component ablation.                                           | All components non-degenerate          |
| `split_head_pose_loss.py`       | Decompose total loss into per-head contributions.                                        | det=85, pose=0.0001, act=37, psr=0.001 |
| `eval_post_reinit.py`           | The actual eval-after-reinit script.                                                     | metrics.json + confusion_matrix.png    |

---

## 8. What's in `evidence/`

| Subdir | Contents |
|--------|----------|
| `baseline_eval_post_reinit_v1/` | Pre-retrain eval (EVAL_SKIP_REINIT=0, so heads were re-init at eval time) — the "best possible from old model" baseline. metrics.json, eval_results.csv, confusion_matrix.png, full raw jsonl. |
| `post_retrain_fp32_20260610_194311/` | The new post-retrain eval (EVAL_SKIP_REINIT=1, so trained post-retrain heads are evaluated as-is). metrics.json, eval_results.csv, confusion_matrix.png, full raw jsonl. |

`baseline_eval_post_reinit_v1` is the LOWER bound; `post_retrain_fp32_*` is
the UPPER bound. Anything in between = remaining gap to fix.

---

## 9. What's in `logs/`

- `retrain_5pct_fp32_bs2/train.log` — full 1.5-hour retrain log (672 KB)
  - HEAD: dataset build, model build, epoch 43 start
  - TAIL: epoch 43 end, combined=0.1116, "New best model" written
- `eval_post_retrain_fp32_20260610_194311/eval.log` — full eval log (28 KB)
  - HEAD: ckpt load + no-reinit branch + class-counts warning
  - MIDDLE: 50× DET_PROBE verdicts
  - TAIL: per-head metric printout
- `smoke_test_3patches_20260607_134812.log` — pre-FIX-4/5/6/7/8 smoke (5.1 MB)
- `smoke_test_3patches_20260607_134047.log` — initial 3-patch smoke (3.4 KB)
- `train_v3_25pct_setsid.log` — v3-launch main run (set SID) (580 KB)
- `restart_25pct_15ep_v2.log` — restart attempt (43 KB)
- `restart_25pct_15ep_v3.log` — restart v3 (1.1 KB; just a few lines)

---

## 10. Where to start reading

If you have 5 minutes, read `00_JOURNEY.md` + `03_CURRENT_RECOVERY.md`.
If you have 30 minutes, read those + `04_HYPOTHESES_FOR_OPUS.md`.
If you have 2 hours, read all the MDs + skim `code/train.py` forward()
+ `code/losses.py` MultiTaskLoss + `code/evaluate.py` evaluate_all.
