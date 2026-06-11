# POPW Project Journey — Chronological Narrative

**Project:** Pose-Conditioned Multi-Task (POPW) unified architecture for industrial
assembly video understanding.
**Code root:** `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/`
**Target paper:** `/home/newadmin/swarm-bot/project/popw/working/code/popw_paper_improved.tex`
**Current focus:** IndustReal dataset (egocentric toy assembly, 24 ASD classes,
74 activity classes, 36 procedure steps with 11 components).
**Hardware:** Single RTX 3060 12GB, 64GB RAM, CUDA_VISIBLE_DEVICES=0.
**Owner:** Bashara (Babas Agency Swarm bot operator).

---

## Phase 0 — Concept (April–May 2026)

The paper `popw_paper_improved.tex` lays out the central thesis: four specialized
models (YOLOv8m for ASD, MViTv2 for activity, STORM-PSR for procedure steps, a
body-pose network) currently have to be deployed together to understand a single
assembly video. Each one runs its own backbone. POPW proposes a **single
ConvNeXt-Tiny + FPN backbone** that feeds five task-specific heads
(detection, body-pose, head-pose, activity, PSR), with FiLM layers carrying
pose information into the activity branch.

The paper's headline claims (vs dedicated baselines):
- **Detection:** YOLOv8m = 83.80% bbox mAP@0.5 (full IndustReal test).
- **Activity:** MViTv2 = 65.25% Top-1 (RGB-only) / 66.45% Top-1 (RGB+VL+stereo).
- **PSR:** STORM-PSR = POS 0.812, F1 0.506 / PSRT-B2 = POS 0.816, F1 0.731.
- **POPW's value proposition:** single forward pass, shared backbone,
  competitive accuracy, vastly better efficiency.

## Phase 1 — Initial Implementation (early May 2026)

We built `industreal_improved/src/` to match the paper's architecture:
- ConvNeXt-Tiny backbone (28.6M params, ImageNet-pretrained).
- FPN neck with P3–P7 levels.
- 5 heads, each fairly faithful to the paper.
- VideoMAE-Small auxiliary stream (384-D, frozen) added for the activity head
  (departure from paper — we'll need Opus's opinion on this).
- AdamW with differential LR (backbone=0.1×, heads=1×, bias=0.3×).
- Staged training: stage 1 = detection only (5 ep), stage 2 = +pose (5 ep),
  stage 3 = +activity +PSR (90 ep).
- Kendall homoscedastic uncertainty for loss balancing.

`config.py` was authored with aggressive AMP (mixed precision), batch=6,
workers=8, subset_ratio=0.10.

## Phase 2 — First Wall: VRAM (mid-May 2026)

A full multi-task batch=6 forward pass with all 5 heads + VideoMAE OOMs the
12GB RTX 3060 immediately. We had to cut batch=1 with grad-accum=32, then
discovered that even batch=2 dies inside ConvNeXt stage 2 (64 MiB free
after the 5-head decoder allocations).

**Lesson:** the architectural plan in the paper assumes a 24GB+ GPU. Our
12GB budget forces a subset-training regime (`SUBSET_RATIO=0.10`, 4
recordings) and torch.compile was offloaded to inference only.

## Phase 3 — First Long Run: Full Multi-Task (early June 2026)

`full_multi_task_tma_tbank_benchmark` ran for 39 epochs at 10% subset. The
checkpoint `crash_recovery.pth` (303 MB) is the only thing we have to show
for it.

What the log says happened:
- Stages 1 and 2 ran cleanly (detection only, then +pose).
- At epoch ~15 (stage 3 entry, all 5 heads online), a NaN cascade began.
  It started in the PSR loss and propagated to activity and detection.
- We added `NaNDetector` callbacks and a crash-recovery save that snapshots
  the optimizer + scaler + model on the first sign of NaN.
- The training process survived, but the saved checkpoint's predictions
  were already damaged: 3 of 5 heads (det, act, psr) had collapsed to
  trivial solutions.

## Phase 4 — Diagnosis Week (2026-06-08 / 2026-06-09)

We wrote a sequence of focused diagnostic scripts in the project root:
- `diag_collapse_3heads.py` — confirmed the three heads had stopped learning.
- `diag_features_alive.py` — confirmed the backbone features were still
  non-degenerate (per-image variance > 0.001, per-channel variance alive).
- `diag_amp_nan.py` — proved that **bf16 was the source of the NaN cascade**
  at the seq=1 sequence-mode PSR batch.
- `diag_psr_nan.py`, `diag_psr_train.py` — proved PSR sigmoid output was
  stuck at one binary pattern (all zeros) and the gradient was zero.
- `detection_collapse_probe.py` — proved detection cls_score was saturated
  to ~1.0 on all anchors (200k+ predictions per 320-frame batch).
- `psr_loss_diagnostic.py` — confirmed PSR focal loss had collapsed to
  the all-zeros trivial solution.
- `debug_kendall_activity.py` — proved Kendall log_vars had drifted to
  extreme values, suppressing the activity gradient.
- `debug_activity_gradients.py` — confirmed activity logits were uniform
  (output distribution entropy ≈ ln(74) — no learning signal).

**Verdict:** The backbone is alive, but three heads had biases/init that
pushed them into dead-regions of the loss surface from the very first
epoch. AMP (bf16) masked the early symptoms and then amplified them at
the seq=1 transition.

## Phase 5 — The Reinit Fix (2026-06-09 / 2026-06-10)

We wrote `--reinit-heads` (train.py:1565) that re-initializes 3 dead heads
with safer biases:
- `detection_head.cls_score`: prior=0.05, bias=-2.94 (was 0.01/-4.6;
  sigmoid was 0.01 starting but overshot to 1.0 after a few hundred steps).
- `activity_head.activity_classifier`: bias=-0.5 (was 0; was collapsing
  to class 27 — the most common — under symmetric init).
- `psr_head.output_heads` (11 binary heads): bias=-0.2 (was -1.0; sigmoid
  0.27 always below the 0.5 threshold).

We also added a hard-detach around the seq-mode PSR output to break the
autograd graph poisoning that the seq=1 batch was causing. And we turned
AMP off (`MIXED_PRECISION=False`) — FP32 is 2× slower but has zero NaN
in 100-step smoke tests.

The current retrain (`reinit_5pct_fp32_20260610_072040`) is running
right now: PID 2416305, started 2026-06-10 07:20, batch=4 → auto-reduces
to batch=2 on OOM, 5 epochs from epoch 39 to 44, expected completion
~11:50. As of 8 min in: loss dropped from 90 to 20, det c=0.14–2.30
(healthy, not saturated), PSR=0.01–0.30 (active, not stuck), ACT=11–25
(varying, not constant).

## Phase 6 — What We're Asking Opus

We have:
- An architecture that is faithful to the paper but is showing a
  pathological interaction between the four heads in a single forward pass.
- A recovery plan that works in the short term (reinit biases, FP32,
  detaching seq-mode) but is not a long-term solution.
- A user target of 80% confidence on 20 factors (vs current 5–11%
  combined on the eval metric `act_top5_accuracy`).

We do **not** have a working multi-task model. We have a model that
reaches a 4-head checkpoint, then collapses. We need Opus's help to
either:
(a) harden this architecture so the heads stop fighting each other, or
(b) replace pieces of it (backbone, head structure, conditioning) with
    something that actually trains end-to-end on a 12GB GPU.

---

## Files in this folder

| Path | Purpose |
|---|---|
| `00_JOURNEY.md` | This file |
| `01_WHAT_WE_BUILT.md` | Architecture overview (read this before reading code) |
| `02_COLLAPSE_CRISIS.md` | The 4-head collapse, in evidence-grade detail |
| `03_CURRENT_RECOVERY.md` | The 2026-06-09/10 fixes, what they do, what they don't |
| `04_HYPOTHESES_FOR_OPUS.md` | 7 specific questions we want Opus to answer |
| `05_MASTER_PROMPT.md` | The end-to-end master prompt to give Opus |
| `code/` | The 8 most important Python files |
| `evidence/metrics_baseline_pre_reinit.json` | Eval results from the broken 39-epoch run |
| `logs/train_digest.log` | Curated excerpts from the current retrain |

## Key file map

- `code/model.py` (2167 lines) — full architecture, all 5 heads.
- `code/config.py` (699 lines) — every hyperparameter, loss cap, head weight, schedule.
- `code/train.py` (3733 lines) — training loop, EMA, crash recovery, reinit, staged training.
- `code/losses.py` (1505 lines) — Focal, GIoU, Wing, LDAM, PSR-Focal, Kendall-weighted MultiTask.
- `code/evaluate.py` (4004 lines) — 4-head evaluation, top-k, mAP, F1, edit score, POS, error verification.
- `code/industreal_dataset.py` (1383 lines) — frame-level data, sequence windows, label mapping.
- `code/metrics.py` (198 lines) — metric utility functions.
- `code/eval_post_reinit.py` — re-eval script that re-runs the dead heads with fresh init.
