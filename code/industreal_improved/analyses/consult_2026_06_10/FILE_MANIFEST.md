# Consultation Folder File Manifest — Updated 2026-06-22 10:30 UTC

> **BREAKTHROUGH**: Run 1 (wrong LR/BIAS=4.0/2.0) and Run 2 (correct LR/BIAS=1.0/1.0) produce IDENTICAL mAP50 trajectories. The ceiling at ~0.207 is STRUCTURAL — caused by OHEM+FocalLoss gradient suppression, not hyperparameter configuration. Epoch 21 validation completed. Training continues at epoch 22. See `47_HYPOTHESES_PROVEN_WRONG_AND_UNANSWERED.md` for the consolidated single source of truth.

## Core Python Source Files (code/) — SYNCED 2026-06-21 FROM RUNNING SOURCE

| File | Source Location | Description |
|------|----------------|-------------|
| `train.py` | `src/training/train.py` | Main training loop — Heartbeat, LIVENESS, POS_ANCHOR_PROBE, DILUTION logging |
| `losses.py` | `src/training/losses.py` | FocalLoss + all loss functions — Opus v8 fixes applied |
| `config.py` | `src/config.py` | All configuration — ba48691 commit fixes included |
| `evaluate.py` | `src/evaluation/evaluate.py` | Full evaluation pipeline |
| `model.py` | `src/models/model.py` | POPW architecture |
| `optimizer.py` | `src/training/optimizer.py` | Optimizer/scheduler builders |
| `checkpoint.py` | `src/training/checkpoint.py` | Checkpoint save/load |
| `distillation.py` | `src/training/distillation.py` | Knowledge distillation |
| `ema.py` | `src/training/ema.py` | EMA shadow weights |
| `embedding_cache.py` | `src/training/embedding_cache.py` | Embedding caching |
| `pretrain_mae.py` | `src/training/pretrain_mae.py` | MAE pretraining |
| `pretrain_synthetic.py` | `src/training/pretrain_synthetic.py` | Synthetic pretraining |
| `stage_manager.py` | `src/training/stage_manager.py` | Stage manager — Opus v8 fixes included |
| `training_supervisor.py` | `src/training/training_supervisor.py` | Training supervisor |
| `industreal_dataset.py` | `src/data/industreal_dataset.py` | Dataset loader |
| `psr_transition.py` | `src/models/psr_transition.py` | PSR transition objective |
| `roi_detector.py` | `src/models/roi_detector.py` | ROI detector variant |
| `video_stream.py` | `src/models/video_stream.py` | VideoMAE stream integration |
| `head_pose_geo.py` | `src/models/head_pose_geo.py` | Head pose geometry utilities |
| `diag_per_class_truth.py` | `src/diag_per_class_truth.py` | Per-class ground truth diagnostic (v11) |
| `run_eval_direct.py` | `src/run_eval_direct.py` | Direct evaluation runner |
| `quick_eval.py` | `src/quick_eval.py` | Quick evaluation |

## Training Logs (logs/) — UPDATED 2026-06-21 WITH LIVE DATA

| File | Description |
|------|-------------|
| `current_rf2_train.log` | **FULL live training log** — 313,000+ lines (Run 1 lines 1-136945 + Run 2 lines 136946+) |
| `current_rf2_train_RUN2_ONLY.log` | **Run 2 only extract** — lines 136946+ from the correct-config run |
| `metrics.jsonl` | **Live metrics** — epoch 17-21 val data now confirms identical Run 1/2 trajectories |
| `rf2_checklist_results.json` | RF2 100-point checklist automated results (26,737 bytes) |
| `rf2_checklist_report.txt` | RF2 checklist report (20,703 bytes) |
| `swarm_loop.log` | Swarm monitoring loop log (1,620,759 bytes) |
| `current_rf1_subprocess.log` | Historical — RF1 subprocess output |
| `current_rf1_train.log` | Historical — RF1 training log |
| `current_rf1_metrics.jsonl` | Historical — RF1 metrics |
| `paper_run_r25_fix_20260615.log` | Historical — Paper run R25 fix |
| `recovery_*.log` | Historical — Recovery runs (R0-R8) |
| `fresh_rf1_*.log` | Historical — Fresh RF1 diagnostics |
| `broken_rf1_*.log` | Historical — Broken RF1 launch |

## Evidence (evidence/)

| File | Description |
|------|-------------|
| `eval_metrics.json` | Evaluation metrics (65KB) |
| `rf_stage_state.json` | **Live stage state** — epoch 21, PID 361404, mAP50=0.2069, MAE=9.21°, max_epochs=36 |
| `overfit_50img_results.json` | 50-image overfit results (200 epochs) — shows OHEM+FL gradient suppression |

## Analysis Files (Consultation Documents)

| # | File | Description | Status |
|---|------|-------------|--------|
| 00 | `JOURNEY_AND_STATUS.md` | Full project timeline (Phases 1-14) | OUTDATED — needs Phase 15 |
| 01 | `PROBLEMS_ROOT_CAUSES.md` | Original problem analysis | HISTORICAL |
| 02 | `GOALS_AND_BENCHMARKS.md` | Goals and benchmarks | HISTORICAL |
| 03 | `ARCHITECTURE_DEEP_DIVE.md` | Architecture deep dive | HISTORICAL |
| 04 | `MASTER_PROMPT_FOR_OPUS.md` | First master prompt | HISTORICAL |
| 10 | `OPUS_ANSWER_v2.md` | Opus v2 answer | HISTORICAL |
| 11 | `MASTER_PROMPT_v3.md` | Master prompt v3 | HISTORICAL |
| 12 | `MASTER_PROMPT_v4.md` | Master prompt v4 | HISTORICAL |
| 13 | `OPUS_ANSWER_v4.md` | Opus v4 answer (RC-28/RC-29) | HISTORICAL |
| 14 | `POST_OPUS_V4_IMPLEMENTATION.md` | Phase 10-13 (gradient sparsity, Kendall bug, RF2 collapse) | UPDATED 2026-06-20 |
| 15 | `GIT_DIFF_SUMMARY.txt` | Git diff c454163..HEAD | HISTORICAL |
| 16 | `MASTER_PROMPT_v5.md` | Opus v5 prompt | HISTORICAL |
| 17 | `OPUS_ANSWER_v5.md` | Opus v5 answer | HISTORICAL |
| 18 | `ULTIMATE_MASTER_GUIDE_INDUSTREAL.md` | Ultimate end-to-end guide | HISTORICAL |
| 18 | `HONEST_FEASIBILITY_AUDIT.md` | Feasibility audit | HISTORICAL |
| 19 | `PRE_TRAINING_READINESS_AUDIT_100.md` | 100-item readiness audit | HISTORICAL |
| 22 | `FINAL_PREFLIGHT_GAP_CLOSURE.md` | Final preflight gap closure | HISTORICAL |
| 23-25 | Various historical docs | Earlier analysis | HISTORICAL |
| 26 | `RF1_RF10_COMPREHENSIVE_STATUS.md` | RF1-RF10 stage definitions (v7) | **UPDATED 2026-06-22** — Section 20 rewritten with epoch 21 data, structural ceiling confirmation |
| 28 | `DET_DEATH_SPIRAL_FIX_AND_RUNBOOK.md` | Detection death spiral fix | HISTORICAL |
| 29 | `RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md` | Root cause analysis | UPDATED 2026-06-20 |
| 31 | `KENDALL_BUG_DISCOVERY_AND_FIX.md` | Kendall bug details | UPDATED 2026-06-20 |
| 33 | `OPEN_QUESTIONS.md` | ALL remaining confusions (42 questions) | **CORRECTED 2026-06-21** — Q30, Q01 updated with Run 1/2 notice |
| 34 | `RF2_SWARM_MONITOR.md` | 20-agent swarm documentation | UPDATED 2026-06-20 |
| 35-36 | Opus v8 prompt/answer | Opus v8 consultation | HISTORICAL |
| 37 | `IMPLEMENTATION_SUMMARY.md` | All 4 v8 fixes applied (beda631) | HISTORICAL |
| 38-39 | Opus v9 prompt/answer | detach fix, POS_ANCHOR_PROBE | HISTORICAL |
| 40 | `DEEP_OPEN_QUESTIONS.md` | Deep open questions (12 chapters) | **CORRECTED 2026-06-21** — §12.9 rewritten. **Now superseded by 47** |
| 41-42 | Opus v10 prompt/answer | detach smoking gun breakthrough | HISTORICAL |
| 43-44 | Opus v11 prompt/answer | Per-class AP, config corrections | HISTORICAL |
| 45 | `CURRENT_TRAINING_STATE.md` | **Single source of truth** | **UPDATED 2026-06-22 10:30 UTC** — epoch 21, structural ceiling confirmed |
| 46 | `DEEP_UNANSWERED_QUESTIONS.md` | 10 genuinely unanswerable questions | **CORRECTED 2026-06-21** — **Now superseded by 47** |
| 47 | `HYPOTHESES_PROVEN_WRONG_AND_UNANSWERED.md` | **COMPREHENSIVE CONSOLIDATION** | **CREATED 2026-06-21 — UPDATED 2026-06-22** |

## Quick Reference

**Current training**: Run 2, epoch 21/36 completed, epoch 22 in progress, mAP50=0.2069 (best, epoch 20), MAE=9.21°
**Config**: ba48691 commit — ALL v8 fixes, correct LR/BIAS=1.0/1.0, detach_reg_fpn=False
**Key diagnostic**: POS_ANCHOR_PROBE shows 500-800 positive anchors/image; LIVENESS_GRAD shows detection_head ~0.02 vs backbone ~3.9 (ratio ~0.007×)
**Breakthrough finding**: Run 1/2 trajectories are IDENTICAL — ceiling at ~0.207 is STRUCTURAL, independent of LR/BIAS
**Primary hypothesis**: OHEM+FocalLoss gradient suppression is the root cause of the structural ceiling
**Single authoritative file**: `47_HYPOTHESES_PROVEN_WRONG_AND_UNANSWERED.md`
**Training state**: `45_CURRENT_TRAINING_STATE.md`
**Next decisive action**: OHEM ablation experiment (planned — see 00_JOURNEY_AND_STATUS.md Phase 19)
