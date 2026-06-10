# POPW IndustReal Improved — Opus Round 2 Consultation Folder

> **This folder is the MAIN folder for the `industreal_improved`
> project.** It is a self-contained snapshot of the entire codebase,
> evidence, logs, and documentation as of **2026-06-10 20:45**.
> The numbered MDs (`00_…` through `05_…`) form a complete opus
> analysis brief.

---

## What's in here

```
opus_consult_2026_06_10_v2/
├── README.md                        ← you are here
│
├── 00_JOURNEY.md                    Timeline + what changed since opus 1
├── 01_WHAT_WE_BUILT.md              8 surgical patches + scripts + env
├── 02_COLLAPSE_CRISIS.md            The 3-dead-head collapse, post-retrain
├── 03_CURRENT_RECOVERY.md           Side-by-side metric table
├── 04_HYPOTHESES_FOR_OPUS.md        Prioritized hypothesis list (P0–P4)
├── 05_MASTER_PROMPT.md              Single entry point for the opus
│
├── code/                            All Python source + diagnostics (17 files)
├── evidence/                        Eval outputs (2 runs: baseline + post-retrain)
├── logs/                            Training + eval + smoke logs
├── scripts/                         9 shell scripts
└── docs/                            All project MDs + reports + contracts
```

**Total size:** ~9.1 MB (github-pushable)
**Files:** 93
**No checkpoints included** (those are in `runs/` and are 1–2 GB each)

---

## How to read this folder

### If you are the opus and have 5 minutes

1. Read `00_JOURNEY.md` (the TL;DR of the journey)
2. Read `03_CURRENT_RECOVERY.md` §1 (the side-by-side metric table)
3. Read `04_HYPOTHESES_FOR_OPUS.md` §"Priority 0" (the 9 ruled-out
   hypotheses)
4. Skim `04_HYPOTHESES_FOR_OPUS.md` §"Priority 1" (the det
   hypotheses)
5. Decide: is H1.1 (box decoder stride) the most likely cause of
   `det_mAP50 = 0`? If yes, write the diagnostic and the patch.

### If you are the opus and have 30 minutes

1. All of the above
2. Read `04_HYPOTHESES_FOR_OPUS.md` §"Priority 2" (act) and §"Priority
   3" (pose)
3. Read `01_WHAT_WE_BUILT.md` (so you know the patch layout)
4. Read `02_COLLAPSE_CRISIS.md` (so you know what was already ruled
   out and what wasn't)
5. Skim `code/model.py:DetectionHead.forward()` + `code/model.py:ActivityHead.forward()`

### If you are the opus and have 2 hours

1. All of the above
2. Read `05_MASTER_PROMPT.md` (the master prompt)
3. Skim `code/train.py` forward() + `code/losses.py` MultiTaskLoss +
   `code/evaluate.py` evaluate_all
4. Read `code/eval_post_reinit.py` (the post-retrain eval entrypoint)
5. Look at the `evidence/post_retrain_fp32_20260610_194311/eval.log`
   (28 KB) to see the raw per-batch output

---

## The TL;DR of what's happening

### The model
A multi-task ConvNeXt-Tiny + FPN backbone with 5 heads (det, act, psr,
body pose, head pose). Trained on IndustReal (procedural IKEA assembly
videos). On RTX 3060 12GB.

### The bug
**3 dead heads** (det, act, psr) producing 0.0 metrics. **2 working
heads** (body pose, head pose) producing reasonable MAE (0.35–0.42).

### What we did
1. First opus analysis (morning) → 12 root-cause hypotheses
2. Applied **8 surgical patches** (FIX-1 through FIX-8) via
   `code/apply_popw_fixes.py`
3. Confirmed AMP fp16 was poisoning the backbone → switched to **FP32
   retrain** with `--no-amp`
4. Ran a **1-epoch FP32 retrain** with `_reinit_dead_heads` (169 head
   tensors) from the epoch-43 crash_recovery.pth
5. Eval'd the new best.pth with `EVAL_SKIP_REINIT=1`

### The result
- ✅ Loss: 227.7 → 72.9 (**−68%**)
- ✅ PSR: edit_score 0.09 → **0.73** (8×); 1 of 11 components now
  predicts correctly
- ✅ act_top5: 0.00 → **0.06** (barely above random 0.05)
- ❌ act_top1: still **0.00** (model collapses to 1 class)
- ❌ det_mAP50: still **0.00** (12/50 batches TOTAL COLLAPSE; boxes
  have IoU < 0.5 with GT)
- ❌ Pose: regressed **+11–73% MAE** (the reinit list included pose
  tensors when it shouldn't have)

### What we need
1. The 1-2 surgical patches to fix det_mAP50 and (optionally) act_top1
2. A 3-5 epoch retrain with **pose excluded from the reinit list**
3. A 13th root-cause hypothesis (if there is one)
4. An estimate of the realistic recovery ceiling (what should
   act_top1 / det_mAP50 be on IndustReal val?)

---

## Key invariants (do NOT change without thinking hard)

1. **`OUTPUT_ROOT` config default** = `src/runs/full_multi_task_tma_tbank_benchmark/`
   - This is where train.py saves checkpoints
   - NOT the run's own dir (despite the script's name)
   - This is a known wart, documented in `01_WHAT_WE_BUILT.md` §5

2. **`_reinit_dead_heads` reinit list**
   - Currently includes pose tensors (BAD — that's why pose regressed)
   - Should be det + act + psr ONLY

3. **`EVAL_SKIP_REINIT=1`** is required for post-retrain eval
   - Without it, the eval will reinit the trained post-retrain heads
   - And you'll evaluate a fresh-init head (the same as baseline)

4. **`--no-amp`** is required for any retrain
   - AMP fp16 is broken in backbone first layers (NaN at
     `backbone.0.conv1.weight`)
   - The runner silently overrides `--no-amp` in some configs; verify
     with `code/diag_amp_nan.py`

5. **`--batch-size 2`** is the safe choice
   - bs=4 OOMs at seq-mode PSR T=4
   - bs=2 has 1.5h/epoch on RTX 3060 at subset-ratio 0.05

6. **Adam optimizer state resume spike**
   - The first 20-50 steps after `--reinit-heads` have a cls_loss
     spike (c=19.8M at step 28, c=2.85 at step 96, c=0-5 from step
     100+)
   - This is **transient and recovers** — do NOT kill the run for it
   - See `04_HYPOTHESES_FOR_OPUS.md` §"Ruled out" R9 for the
     kill-criteria

---

## Pointers to the most important files

| File                                          | Why it's important                              |
|-----------------------------------------------|-------------------------------------------------|
| `code/model.py`                               | The 5 heads; FIX-3 (seq detection), FIX-4 (PSR expand) |
| `code/train.py`                               | The training loop; FIX-1, FIX-5, `_reinit_dead_heads` |
| `code/losses.py`                              | Multi-task loss; FIX-6 (NaN guard), FIX-7 (temporal smooth) |
| `code/evaluate.py`                            | Eval; FIX-2 (1-D/2-D), DET_PROBE                |
| `code/eval_post_reinit.py`                    | Post-retrain eval entrypoint; FIX-8             |
| `code/apply_popw_fixes.py`                    | The 8-patch applier (run with `--apply`)        |
| `code/detection_collapse_probe.py`            | The DET_PROBE generator (12/50 TOTAL COLLAPSE)  |
| `evidence/post_retrain_fp32_20260610_194311/` | The post-retrain eval results                   |
| `logs/retrain_5pct_fp32_bs2/train.log`        | The 1.5-hour retrain log (672 KB)               |
| `scripts/run_reinit_fp32_bs2.sh`              | The retrain script (template for the next run)  |
| `scripts/run_eval_post_retrain_fp32.sh`       | The eval script (template for the next eval)   |

---

## What this folder is NOT

- ❌ **Not a complete project snapshot.** It excludes `runs/`
  (checkpoints, 18GB), `src/data/` (datasets, ~5GB), and `.venv/`
  (Python packages, 1GB+). The `code/` is the FULL Python source.
- ❌ **Not the original `opus_consult_2026_06_10/`.** That was the
  first opus analysis (morning). This is the second (evening). The
  v1 folder is in the project root for reference.
- ❌ **Not a working git repo.** This folder is meant to be pushed to
  GitHub (or shared) as a self-contained opus analysis package, not
  used as the live working tree.
