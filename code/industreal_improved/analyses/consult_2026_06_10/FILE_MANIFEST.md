# Consultation Folder File Manifest — Updated 2026-06-21 UTC (Opus v10 Breakthrough)

## Core Python Source Files (code/) — UPDATED 2026-06-21 FROM RUNNING SOURCE

| File | Description | Key Changes |
|------|-------------|-------------|
| `losses.py` | FocalLoss + all loss functions | **Opus v8 fixes**: `DET_POS_IOU_TOP_K=9` top-k anchor matching (6-10 pos/GT vs ~1), `KENDALL_HP_PREC_CAP` clamp (lv_hp >= lv_det), `KENDALL_FIXED_WEIGHTS` fixed-lambda path for RF1-RF2. **Prior**: RC-28 empty frame skip + normalization, Kendall bug fix (include head_pose in total loss) |
| `train.py` | Main training loop | Heartbeat (state file writes at epoch/batch boundaries for swarm), global step fix, stdout logging bootstrap |
| `config.py` | All configuration | **Opus v8 fixes**: `KENDALL_HP_PREC_CAP=True`, `KENDALL_FIXED_WEIGHTS=False`, `KENDALL_STAGED_TRAINING=False`, `KENDALL_HP_FIXED_LAMBDA=0.2`, `DET_POS_IOU_THRESH=0.4`, `DET_POS_IOU_TOP_K=9`, `DET_BIAS_LR_FACTOR=1.0` (was 5.0). Prior: `DET_OHEM_RATIO=2.0/MIN_NEG=32/GAMMA_NEG=1.5`, `POSE_LOSS_WEIGHT=5.0`, `SOFT_ARGMAX_TEMP_TRAIN=1.0` |
| `evaluate.py` | Full evaluation pipeline | **5 guards**: top5 broadcast check, TRAIN_ACT/PSR skip for eval + logging |
| `model.py` | POPW architecture | **Soft-argmax training temp (1.0)** for gradient flow, **keypoint normalization to [0,1]** for Wing loss (~1300× scale fix) |
| `optimizer.py` | Optimizer/scheduler builders | AdamW with differential LR, CosineAnnealingWarmRestarts |
| `checkpoint.py` | Checkpoint save/load | Model/optimizer/scheduler state dict management |
| `distillation.py` | Knowledge distillation | Distillation loss and teacher-student training |
| `ema.py` | EMA model weights | Exponential Moving Average shadow weights |
| `embedding_cache.py` | Embedding cache | Caching embeddings for efficient training |
| `pretrain_mae.py` | MAE pretraining | Masked Autoencoder pretraining loop |
| `pretrain_synthetic.py` | Synthetic pretraining | Synthetic data pretraining loop |
| `stage_manager.py` | Stage manager | **Opus v8 Fix 4**: `_validate_stage_history_entry()` guard against phantom gate-threshold recording. Prior: RF2 params (50% data, 30 epochs, patience 10), reinit_heads only on actual retries |
| `training_supervisor.py` | Training supervisor | Supervisor layer for training monitoring |

## Diagnostic Scripts (code/) — LIVE FROM diagnostics/ TREE

| File | Purpose |
|------|---------|
| `diag_feature_magnitude.py` | FPN feature magnitude check |
| `diag_step0_logits.py` | Step-0 logit health |
| `diag_det_anchor_coverage.py` | Anchor coverage analysis |
| `diag_det_level_scores.py` | Per-FPN-level detection scores |
| `diag_act_input_variance.py` | Activity head input variance |
| `diag_attention_saturation.py` | ViT attention saturation |
| `diag_ema_contamination.py` | EMA shadow contamination check |
| `diag_videomae_zero.py` | VideoMAE stream zero check |
| `diag_weight_norms.py` | Weight norm diagnostics |
| `diag_amp_2step.py` | 2-step AMP diagnostic |
| `diag_amp_nan.py` | AMP NaN detection |
| `diag_collapse_3heads.py` | 3-head collapse check |
| `diag_gt_coverage.py` | GT coverage analysis |
| `diag_features_alive.py` | Activity head feature liveness |
| `diag_psr_nan.py` | PSR NaN diagnostic |
| `diag_psr_train.py` | PSR training diagnostic |

## Data and Model Support Files (code/)

| File | Description |
|------|-------------|
| `industreal_dataset.py` | IndustReal dataset loader |
| `psr_transition.py` | PSR transition-weighted objective |
| `roi_detector.py` | ROI-based detector variant |
| `video_stream.py` | VideoMAE stream integration |
| `head_pose_geo.py` | Head pose geometric utilities |
| `uncommitted_changes_r25_fix_20260615.patch` | Uncommitted patch from R25 fix (2026-06-15) |

## Training Logs (logs/) — 18 files total

| Log | Description |
|-----|-------------|
| `current_rf1_subprocess.log` | **LIVE RF1 training** — 404 lines, latest at epoch 0 step ~723/1241 |
| `current_rf1_launch.log` | RF1 launch config |
| `current_rf1_train.log` | RF1 training log |
| `current_rf1_metrics.jsonl` | RF1 metrics JSONL |
| `recovery_r0_smoke.log` | R0 smoke test (278s, PASSED: committed=55, skipped=0) |
| `recovery_r1_det_bootstrap_latest.log` | R1 v4 (latest, in validation) |
| `recovery_train8_run8.log` | Run 8 (pre-fix, killed — epoch 0 val only) |
| `recovery_train1_run1.log` | Old recovery Run 1 (pre-fix, RC-25 era) |
| `recovery_train2_run2.log` | Old recovery Run 2 (pre-fix, RC-25 era) |
| `train_main.log` | Earlier training run |
| `eval_post_retrain.log` | Post-retraining evaluation |
| `reinit_runner.log` | Reinitialization runner |
| `fresh_rf1_det_debug.log` | Fresh RF1 det debug |
| `fresh_rf1_launch_config.log` | Fresh RF1 launch config |
| `fresh_rf1_liveness_grad.log` | Fresh RF1 liveness grad |
| `fresh_rf1_subprocess_tail.log` | Fresh RF1 subprocess tail |
| `paper_run_r25_fix_20260615.log` | Paper run R25 fix |
| `broken_rf1_launch.log` | Broken RF1 launch |
| `broken_rf1_subprocess.log` | Broken RF1 subprocess |
| `bringup_smoke_g1g3.log` | Bringup smoke test G1/G3 |
| `recovery_r1_det_bootstrap.log` | Recovery R1 det bootstrap |

## Evidence (evidence/)

| File | Description |
|------|-------------|
| `eval_metrics.json` | Evaluation metrics (64KB) |
| `rf_stage_state.json` | **Live stage state** — epoch 21, PID 3791482, Opus v10 findings applied (detach_reg_fpn=False fix in working tree) |

## Documentation

| File | Description |
|------|-------------|
| `00_JOURNEY_AND_STATUS.md` | Full project timeline (Phases 1-14) — UPDATED 2026-06-21 with Phase 13 (Opus v8 fixes) and Phase 14 (New training run epoch 17) |
| `10_OPUS_ANSWER_v2.md` | Opus v2 answer (RC-25, RC-27, recovery plan) |
| `12_MASTER_PROMPT_v4.md` | Master prompt sent to Opus v4 |
| `13_OPUS_ANSWER_v4.md` | **Opus v4 answer** — RC-28/RC-29 diagnosis, recovery protocol R0-R3 |
| `14_POST_OPUS_V4_IMPLEMENTATION.md` | **What we did after Opus v4** — UPDATED 2026-06-21: now covers Phase 10 (RF1 death spiral, gradient sparsity proof, Kendall bug), Phase 11 (RF1 completion, 0.45 vs 0.184 discrepancy), Phase 12 (RF2 epoch 15 collapse, cls_score bias equilibrium, 20-agent swarm, 5 proven/4 refuted hypotheses) |
| `15_GIT_DIFF_SUMMARY.txt` | Complete git diff of all source changes (c454163..HEAD) |
| `16_MASTER_PROMPT_v5.md` | **SEND THIS TO OPUS** — Ultimate path to SOTA-ready model, all SOTA targets, current state, key questions |
| `17_OPUS_ANSWER_v5.md` | **Opus v5 answer** — ultimate path (R1→R4 ladder), det-gap framing, activity/PSR enable order, PSR `1e-4` sentinel diagnosis, efficiency-first cell-fill strategy, timeline |
| `18_ULTIMATE_MASTER_GUIDE_INDUSTREAL.md` | **Ultimate end-to-end master guide** — three invariants (non-NaN/non-zero/non-degenerate); pre-flight liveness probe; data/label fixes (activity segment sampler + NA mask, PSR transitions, subset stratification, IMG_SIZE guard); per-task recipes; R0→R5 ladder; full eval-protocol table; loss/numerical hygiene; **all 50 checklist items answered**; efficiency table; failure-mode playbook; Definition of Done |
| `19_PRE_TRAINING_READINESS_AUDIT_100.md` | **100-item pre-training readiness audit** — go/no-go gates with why-analysis per item, across architecture, backbone/FPN/anchors, cross-task conditioning, data/labels, loss/numerical hygiene, Kendall/gradient balance, optimization/training loop, per-head liveness, and per-task readiness (det/act/psr/headpose/assembly); 6 master gates; usage protocol |
| `22_FINAL_PREFLIGHT_GAP_CLOSURE.md` | **Final pre-flight & gap closure** — closes the 3 verified gaps on `main` with exact patches: GAP-C (`apply_preset` never sets the 4 fix flags → paper_run no-op), GAP-A (PSR 9:1 static gradient + MonotonicDecoder unused at eval), GAP-B (activity eval per-recording not per-segment). Per-gap verification, 6-gate executable pre-flight, paper-run ladder, eval→`\popwres` map, 15-point sign-off |
| `popw_paper_improved.tex` | Target paper with all benchmark tables and SOTA comparisons |
| `26_RF1_RF10_COMPREHENSIVE_STATUS.md` | RF1-RF10 stage definitions — UPDATED 2026-06-21 (v5) with Section 19: Opus v8 training run detailed analysis |
| `28_DET_DEATH_SPIRAL_FIX_AND_RUNBOOK.md` | Detection death spiral bounded background loss fix + runbook (2026-06-16) |
| `29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md` | Root cause analysis — UPDATED 2026-06-21 with Section 13 postscript: RF2 epoch 15 collapse proves cls_score bias equilibrium is a DISTINCT failure mode from gradient sparsity (head_pose ALIVE but classifier still collapsed) |
| `30_OPUS_MASTER_PROMPT_v7.md` | Self-contained overview prompt for sending to Opus. Summarizes all 30 files, current situation, 10 questions for Opus, gradient math proof, config comparison. Upload this file alongside the folder for fastest Opus onboarding. (2026-06-17) |
| `31_KENDALL_BUG_DISCOVERY_AND_FIX.md` | Kendall weighting bug: losses.py line 1589 excluded head_pose from total loss when train_pose=True, train_act=False. Fix applied and confirmed — UPDATED 2026-06-21 with Section 10 RF2 cross-validation postscript |
| `33_OPEN_QUESTIONS.md` | **ALL remaining confusions** — 41 open questions. UPDATED 2026-06-21 (v10 era): Q35 RESOLVED (detach_reg_fpn), Q31 UPDATED (per-class AP parsed with class-6 data), Q41 NEW (will detach flip break plateau?), Q40 REFRAMED (restart decision) |
| `34_RF2_SWARM_MONITOR.md` | **20-agent monitoring swarm documentation** — 22 agents, 134 checks/cycle, 5-min interval, 40-thread ThreadPoolExecutor, auto-restart watchdog, 4-channel alerting, 6 bugs found and fixed. (2026-06-21 — UPDATED) |
| `35_OPUS_MASTER_PROMPT_v8.md` | **Send this to Opus** — Self-contained overview prompt for Opus consultation v8. Covers all 34 files, 3 distinct failure modes (solved + unsolved), cls_score bias equilibrium, 5 fix proposals, 10 key questions for Opus, current config reference. (2026-06-21) |
| `36_OPUS_ANSWER_v8.md` | **Opus v8 answer** — Unified diagnosis: RF2 collapse is one mechanism wearing three masks (Kendall head_pose domination, not separate bias equilibrium). 4 prioritized fixes. Verifies phantom 0.45, double curriculum, pi=0.03. (2026-06-21) |
| `37_IMPLEMENTATION_SUMMARY.md` | **Implementation summary** — All 4 Opus v8 fixes applied to source (commit beda631). Config/loss-level changes only, safe on RTX 3060. (2026-06-21 — 256 insertions, 119 deletions) |
| `38_OPUS_MASTER_PROMPT_v9.md` | **v9 master overview prompt** — Self-contained post-fix status update for Opus. Covers all 4 v8 fixes (commit beda631), current epoch 17 training state, DET_PROBE LOCALIZING, updated questions for Opus. (2026-06-21) |
| `39_OPUS_ANSWER_v9.md` | **Opus v9 answer** — Core finding: `score_p50` is structurally blind (median over background-dominated anchors → cannot see classification). Flags `detach_reg_fpn:True` config-vs-doc split-brain (`config.py:1109`), the unbounded top-k force-match injecting positive-label noise, and a label-noise→uniform-output hypothesis. Prescribes 4 missing probes + 50-image cls-only overfit. (2026-06-21) |
| `40_DEEP_OPEN_QUESTIONS.md` | **Deep open questions** — 11 chapters. UPDATED 2026-06-21 (v10): Ch4 RESOLVED (detach_reg_fpn confirmed True), Ch11 NEW (post-breakdown retrospective — verification pipeline failures, two-ceiling hypothesis) |
| `41_OPUS_MASTER_PROMPT_v10.md` | **v10 master overview prompt** — Self-contained overview for Opus round 10. 8 sections covering full project state, epoch-21 evidence, POS_ANCHOR_PROBE, pseudo-classing, LR restart failure, top-k IoU floor, 12/24 AP=0 analysis. (2026-06-21) |
| `42_OPUS_ANSWER_v10.md` | **Opus v10 answer** — The breakthrough. Unified diagnosis: `detach_reg_fpn=True` for RF2 is the smoking gun (config.py:1117, confirmed via code trace). Plateau is substantially dynamic (config regression), not structural. Includes verification table, 7-tier recommendations (Tier 0–1), decision rules, per-class diagnostic table. (2026-06-21) |
| `43_OPUS_MASTER_PROMPT_v11.md` | **v11 master overview prompt** — Post-breakthrough status. Frames the central v11 question (how much will flipping `detach_reg_fpn=False` move the ceiling?), the config delta in the working tree, the per-class AP findings (class 6 / 1739 GT), and 8 open questions for Opus. (2026-06-21) |
| `44_OPUS_ANSWER_v11.md` | **Opus v11 answer** — Code-grounded. Three corrections to the v11 premise: (C1) detach=False is already committed (`2ad6cfe`), only the restart is pending; (C2) the per-class AP table has no trustworthy source — detection per-class AP is not in `metrics.jsonl`, and §5 GT counts contradict the repo's own `DET_CLASS_ALPHAS` (idx 6 = 65 train, not 1739; idx 21 AP=1.0 is a no-val-GT artifact); (C3) the v10 "breakthrough" re-derives the 2026-06-17 `stage_manager.py:108–121` recovery-strategy conclusion. Prescribes a one-variable-clean restart (revert `DET_LR_MULTIPLIER`/`DET_BIAS_LR_FACTOR` to 1.0), `detach_reg_fpn=False` for RF3–RF10, per-class AP persistence, and a measurement-keyed decision rule. (2026-06-21) |
