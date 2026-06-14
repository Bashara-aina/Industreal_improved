# Consultation Folder File Manifest — Updated 2026-06-15 07:45

## Core Python Source Files (code/) — WITH RC-28/RC-29 FIXES APPLIED

| File | Description | Key Changes |
|------|-------------|-------------|
| `losses.py` | FocalLoss + all loss functions | **RC-28**: Empty frames skipped (lines 215-229), normalization by GT-bearing image count (line 295) |
| `train.py` | Main training loop | **RC-29**: committed/skipped telemetry (lines 849-1560), PRE_VAL_GUARD fixed (isfinite), TRAIN_MAX_STEPS support |
| `config.py` | All configuration | `recovery_det_only` preset, TRAIN_MAX_STEPS from env, SKIP_DET_METRICS_EVAL=False |
| `evaluate.py` | Full evaluation pipeline | **5 guards**: top5 broadcast check, TRAIN_ACT/PSR skip for eval + logging |
| `model.py` | POPW architecture | ConvNeXt-Tiny + FPN + 5 heads (~53M params) |
| `optimizer.py` | Optimizer/scheduler builders | AdamW with differential LR, CosineAnnealingWarmRestarts |

## Diagnostic Scripts (code/)

| File | Purpose |
|------|---------|
| `diag_feature_magnitude.py` (D7) | FPN feature magnitude check |
| `diag_step0_logits.py` (D8) | Step-0 logit health |
| `diag_det_anchor_coverage.py` | Anchor coverage analysis |
| `diag_det_level_scores.py` | Per-FPN-level detection scores |
| `diag_act_input_variance.py` | Activity head input variance |
| `diag_attention_saturation.py` | ViT attention saturation |
| `diag_ema_contamination.py` | EMA shadow contamination check |
| `diag_videomae_zero.py` | VideoMAE stream zero check |
| `diag_weight_norms.py` | Weight norm diagnostics |

## Other Source Files (code/)

| File | Description |
|------|-------------|
| `industreal_dataset.py` | IndustReal dataset loader |
| `psr_transition.py` | PSR transition-weighted objective |
| `roi_detector.py` | ROI-based detector variant |
| `video_stream.py` | VideoMAE stream integration |
| `head_pose_geo.py` | Head pose geometric utilities |

## Training Logs (logs/)

| Log | Description |
|-----|-------------|
| `recovery_r0_smoke.log` | R0 smoke test (278s, PASSED: committed=55, skipped=0) |
| `recovery_r1_det_bootstrap_latest.log` | R1 v4 (latest, in validation) |
| `recovery_train8_run8.log` | Run 8 (pre-fix, killed — epoch 0 val only) |
| `recovery_train1_run1.log` | Old recovery Run 1 (pre-fix, RC-25 era) |
| `recovery_train2_run2.log` | Old recovery Run 2 (pre-fix, RC-25 era) |
| `train_main.log` | Earlier training run |
| `eval_post_retrain.log` | Post-retraining evaluation |
| `reinit_runner.log` | Reinitialization runner |

## Documentation

| File | Description |
|------|-------------|
| `00_JOURNEY_AND_STATUS.md` | Full project timeline (Phases 1-9) |
| `10_OPUS_ANSWER_v2.md` | Opus v2 answer (RC-25, RC-27, recovery plan) |
| `12_MASTER_PROMPT_v4.md` | Master prompt sent to Opus v4 |
| `13_OPUS_ANSWER_v4.md` | **Opus v4 answer** — RC-28/RC-29 diagnosis, recovery protocol R0-R3 |
| `14_POST_OPUS_V4_IMPLEMENTATION.md` | **What we did after Opus v4** — all fixes, crashes, retries, breakthrough |
| `15_GIT_DIFF_SUMMARY.txt` | Complete git diff of all source changes (c454163..HEAD) |
| `16_MASTER_PROMPT_v5.md` | **SEND THIS TO OPUS** — Ultimate path to SOTA-ready model, all SOTA targets, current state, key questions |
| `17_OPUS_ANSWER_v5.md` | **Opus v5 answer** — ultimate path (R1→R4 ladder), det-gap framing, activity/PSR enable order, PSR `1e-4` sentinel diagnosis, efficiency-first cell-fill strategy, timeline |
| `18_ULTIMATE_MASTER_GUIDE_INDUSTREAL.md` | **Ultimate end-to-end master guide** — three invariants (non-NaN/non-zero/non-degenerate); pre-flight liveness probe; data/label fixes (activity segment sampler + NA mask, PSR transitions, subset stratification, IMG_SIZE guard); per-task recipes; R0→R5 ladder; full eval-protocol table; loss/numerical hygiene; **all 50 checklist items answered**; efficiency table; failure-mode playbook; Definition of Done |
| `19_PRE_TRAINING_READINESS_AUDIT_100.md` | **100-item pre-training readiness audit** — go/no-go gates with why-analysis per item, across architecture, backbone/FPN/anchors, cross-task conditioning, data/labels, loss/numerical hygiene, Kendall/gradient balance, optimization/training loop, per-head liveness, and per-task readiness (det/act/psr/headpose/assembly); 6 master gates; usage protocol |
| `22_FINAL_PREFLIGHT_GAP_CLOSURE.md` | **Final pre-flight & gap closure** — closes the 3 verified gaps on `main` with exact patches: GAP-C (`apply_preset` never sets the 4 fix flags → paper_run no-op), GAP-A (PSR 9:1 static gradient + MonotonicDecoder unused at eval), GAP-B (activity eval per-recording not per-segment). Per-gap verification, 6-gate executable pre-flight, paper-run ladder, eval→`\popwres` map, 15-point sign-off |
| `popw_paper_improved.tex` | Target paper with all benchmark tables and SOTA comparisons |
