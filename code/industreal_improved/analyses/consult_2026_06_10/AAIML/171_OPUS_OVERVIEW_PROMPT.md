# 171 — Overview Prompt for Opus (entry point to files 166–170)

**Date:** 2026-07-08
**Purpose:** Self-contained context Opus can read in one shot to understand the AAIML paper situation, where files 166–170 fit, and what we have empirically.

---

## Who is reading this

You are Opus (or another senior reviewer) joining the IndustReal AAIML paper effort. The author wants 4 heads (detection, activity, head pose, PSR) with meaningful, comparable-to-SOTA results from a single multi-task training run. We are **not** there yet. Read this file first, then read 166, 167, 168, 169, 170 in that order.

## What we wanted to show, vs. what we can show

| Head | What we wanted (SOTA-comparable) | What we actually have | Source |
|---|---|---|---|
| Detection | Multi-task det ≥ 0.90 mAP50 | 0.995 single-task D1R YOLOv8m (separate training); V5b/V8 multi-task det = 0.0 / NaN | `d1_yolov8m_v3/metrics.json` |
| Activity | Multi-task act ≥ 0.55 top-1 | 0.3810 frozen probe (no fine-tune); V5b/V8 multi-task act = 0.0 | `activity_mvit_probe/results.json` |
| PSR | Multi-task PSR F1 ≥ 0.6 | 0.7018 pre-fix per-comp opt (paradigm mismatch with STORM/B2/B3); V5b KENDALL = 0.0 | `psr_optimal_thr_38k/optimal_thresholds.json` |
| Head pose | Multi-task pose ≤ 8° fwd MAE | **8.52° at V5b epoch 34** (only head learning) — first public baseline | `full_eval_ep18_v2/metrics.json` |

**Bottom line:** Only head pose has a defensible multi-task result. The other three heads either come from separate single-task runs (not multi-task evidence) or have collapsed. The paper's headline claim must shift away from "SOTA-comparable across all 4 heads via one multi-task run" toward "we measure the multi-task cost honestly on IndustReal."

## File map (read in order)

1. **166 — Deep Questions Fuel.** 50+ targeted questions across 9 sections (architecture, multi-task, SOTA, fixes, dynamics, efficiency, paper story). The questions are the questions we are trying to answer. Look here to see what we *don't* know yet.

2. **167 — Multi-Task Architecture Strategy.** Comparison of V5 (ConvNeXt shared), V6 (MViTv2-S shared), V8 (YOLOv8m + MViTv2-S), V9 (unified). Includes the "4× more efficient than single-task" claim **which the 10-agent debate flagged as NOT defensible from current data** — the per-head sharing penalty is real, and architectures V5/V8 produced classification collapse, so a 4-head single-run efficiency claim cannot be made yet.

3. **168 — SOTA Comparison Data Audit.** The WACV numbers (det 0.838 / 0.95, activity 0.6223, PSR STORM 0.506 / B2 0.731 / B3 0.883), our 0.995 D1R number, the 0.3810 frozen probe, the 0.7018 PSR per-comp opt, and the honest comparability analysis. **Caveats required everywhere.** Pose has no SOTA.

4. **169 — Training Progress and Architecture Plan.** All training runs with PIDs, log files, current status. **V5b** (PID 758477, GPU 0, epoch 35 82%, KENDALL rebalance) is the live run. **V8** (PID 843794, GPU 1, epoch ~3, MViTv2-S + YOLOv8m) shows the same classification collapse. Read this for the operational picture.

5. **170 — Discussion and Conclusion.** What we wanted to claim vs. what we can claim. Read this last to see the current paper narrative draft.

## Three claims the paper can defend today

1. **Head pose:** V5b multi-task reaches ~8.5° fwd MAE. No published SOTA on IndustReal head pose. First public baseline.
2. **Detection ceiling:** D1R YOLOv8m single-task reaches 0.995 mAP50 — same backbone, trained only on detection, beats WACV 0.838 by 0.157. This validates the data pipeline and labels; it is not a multi-task result.
3. **Multi-task pathology finding:** Kendall uncertainty weighting with `KENDALL_FIXED_WEIGHTS=0` and `HP_PREC_CAP` does **not** rescue classification heads when the per-head gradient signal is asymmetric. log_vars converged by epoch 25 to lv_pose=-0.987 (precision 2.68, capped to 0.59), lv_psr=0.000 (PSR architecturally dead from ReLU saturation in `src/models/psr_transition.py:216-237`), lv_det=0.538, lv_act=0.291. The empirical evidence is in `src/runs/rf_stages/logs/metrics.jsonl` and the LIVENESS_GRAD checkpoints.

## Three claims the paper must NOT make

1. ❌ "V8 achieves 4× compute efficiency over single-task." Not measured, and the per-head results don't support it.
2. ❌ "Detection 0.995" as a multi-task result. It's single-task D1R; the multi-task det is dead.
3. ❌ "PSR 0.7018 beats STORM 0.506." Different paradigm (per-frame component F1 vs. transition F1); we have not computed transition F1.

## Critical bugs already flagged

- **V8 `_class_to_idx` uses Python `hash()` randomization** (`train_v8_multitask.py:217`). Activity class indices shift per subprocess, so activity training in V8 is impossible by construction. Fix: stable dict-based class mapping indexed by an ordered list, not `hash() % num_classes`.
- **PSR head ReLU saturation.** `Linear -> ReLU(inplace) -> Linear(bias=-1.0)` parks the sigmoid at 0.27 and gates the entire head's gradient to zero (Opus audit, file 132, lines 48-51). A `psr_repair` run with LeakyReLU is in flight (`src/runs/rf_stages/checkpoints/psr_repair_training/status.md`); its results are not yet merged.

## What we are doing right now

- **V5b** continues training (PID 758477), expected to finish ~50 epochs by tomorrow morning (2026-07-09). Pose improving 8.82° → 8.52° between epochs 33–34. det/act/PSR remain collapsed. ETA ~18h end-to-end.
- **V8** continues training (PID 843794) on GPU 1 with the hash() bug intact. Until the bug is patched and the run restarted, V8's activity number is structurally meaningless.
- **D1R** detection (single-task YOLOv8m) is in the repo at 0.995 mAP50 and is the only detection number we can cite.
- **Frozen probe** activity is in the repo at 0.3810 and is the only activity number we can cite (with the "frozen, not fine-tuned" caveat).

## What we need Opus to weigh in on

Given the current evidence, we have two paths:

**Path A: Paper as measurement study.** Write the paper around what we have: pose first baseline, detection ceiling via single-task, Kendall pathology as a finding. Honest, defensible, publishable. Risk: no "multi-task is helping" headline.

**Path B: One more training cycle.** Patch V8 hash() bug, restart V8 with bug-free activity head. In parallel, launch a 1-2 day MViTv2-S fine-tune for activity (single-task or as one head of V8). If both runs produce usable activity numbers, the paper can shift toward "we demonstrate a working multi-task system with honest sharing-penalty accounting." Risk: time-budget pressure (~20h remaining in author's estimate), and the activity fine-tune may not finish in time.

**Opus's job:** tell the author which path is defensible given what is in 166–170, and what additional experiments (if any) are mandatory before drafting. Be specific. The author does not want a survey of options; they want a recommendation.

---

## File path cheat sheet

| File | Path |
|---|---|
| 166 (questions) | `/media/newadmin/master/POPW/working/code/industreal_improved/code/industral_improved/code/industral_improved/analyses/consult_2026_06_10/AAIML/166_DEEP_QUESTIONS_FUEL.md` |
| 167 (architecture) | `/media/newadmin/master/POPW/working/code/industreal_improved/code/industral_improved/code/industral_improved/analyses/consult_2026_06_10/AAIML/167_MULTITASK_ARCHITECTURE_STRATEGY.md` |
| 168 (SOTA) | `/media/newadmin/master/POPW/working/code/industreal_improved/code/industral_improved/code/industral_improved/analyses/consult_2026_06_10/AAIML/168_SOTA_COMPARISON_DATA_AUDIT.md` |
| 169 (training) | `/media/newadmin/master/POPW/working/code/industreal_improved/code/industral_improved/code/industral_improved/analyses/consult_2026_06_10/AAIML/169_TRAINING_PROGRESS_ARCHITECTURE.md` |
| 170 (discussion) | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industral_improved/analyses/consult_2026_06_10/AAIML/170_DISCUSSION_CONCLUSION.md` |
| 171 (this file) | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industral_improved/analyses/consult_2026_06_10/AAIML/171_OPUS_OVERVIEW_PROMPT.md` |
| V5b log | `/tmp/train_v5b.log` |
| V8 log | `/tmp/train_v8.log` |
| V8 script | `scripts/train_v8_multitask.py` (line 217 = hash() bug) |
| V5b launch | `scripts/train_v5b_fresh_kendall_rebalanced.sh` |
| Metrics log | `src/runs/rf_stages/logs/metrics.jsonl` |
| D1R metrics | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` |
| Frozen probe | `src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json` |
| Full eval (V5b ep18) | `src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json` |
| PSR optimal thresholds | `src/runs/rf_stages/checkpoints/psr_optimal_thr_38k/optimal_thresholds.json` |
| PSR head (ReLU bug) | `src/models/psr_transition.py:216-237` |
