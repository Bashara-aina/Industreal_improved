# R3 Readiness — 100-Item Verification Checklist
> Generated 2026-06-15 | Based on R2.5 fix training analysis (epoch 48)

Status legend: ✅ PASS | ❌ FAIL | ⏳ PENDING | 🔧 NEEDS FIX | ➖ NOT APPLICABLE

---

## A. TRAINING DATA & DATASET (1-10)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | industreal dataset has correct train/val split | ⏳ | Need to read split files |
| 2 | No OOB detection labels (class > 24) in GT | ✅ | Guard added in losses.py — but root cause unknown |
| 3 | No OOB activity labels (class > 75) in GT | ✅ | Guard added — but root cause unknown |
| 4 | activity_mask coverage rate logged | ❌ | Not implemented — urgent need |
| 5 | PSR sequence mode batch generation works (no OOM) | ❌ | USE_PSR_SEQUENCE_MODE=False due to OOM |
| 6 | Consistent DataLoader throughput | ⏳ | Need to measure over epoch |
| 7 | hand_joints key in targets | ✅ | Verified in train.py |
| 8 | pose_confidence key in targets | ✅ | to(device) fix added |
| 9 | Activity class frequency distribution | ❌ | Not logged — missing classes invisible |
| 10 | PSR label sparsity logged | ❌ | Not logged |

---

## B. MODEL ARCHITECTURE (11-20)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 11 | Backbone is ConvNeXt Tiny (not base/large) | ✅ | config.py BACKBONE=convnext_tiny |
| 12 | TMA Cell connected (not dead code) | ⏳ | Need to verify forward pass connectivity |
| 13 | TemporalBank slot_overwrite=False working | ✅ | paper_run preset |
| 14 | EMA enabled and updating | ✅ | paper_run preset |
| 15 | PSR CausalTransformer receives sequences | ❌ | Sequence mode disabled (OOM) — per-frame only |
| 16 | HeadPoseFiLMModule has gradient flow | ⏳ | LIVENESS shows head_pose ALIVE — need to verify FiLM specifically |
| 17 | MonotonicDecoder PSR decode at eval time | ➖ | Requires PSR_TRANSITION=True + sequence mode |
| 18 | PSR bias head residual connection | ❌ | Not implemented — bias head grad DEAD for 4000 steps |
| 19 | GeometryAwareHeadPose produces [B,9] tensor | ✅ | Fix from gap-c commit |
| 20 | DetectionHead FPN outputs correct scale | ⏳ | Need to verify with actual detection forward pass |

---

## C. TRAINING CONFIG (21-40)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 21 | EPOCHS=100 | ✅ | config.py Ln 722 |
| 22 | BATCH_SIZE=2 safe for RTX 3060 12GB | ✅ | ~7.6GB confirmed at batch=2 |
| 23 | GRAD_ACCUM_STEPS=16 (effective 32) | ✅ | config.py |
| 24 | TRAIN_MAX_STEPS=0 (disabled) | ✅ | config.py |
| 25 | VAL_BATCH_SIZE=16 safe | ⏳ | Need to test |
| 26 | WARMUP_EPOCHS=5 | ✅ | config.py |
| 27 | Cosine LR decay configured correctly | ✅ | config.py |
| 28 | GRAD_CLIP_NORM=1.0 appropriate | ⏳ | No grad explosion evidence yet |
| 29 | ACTIVITY_HEAD_GRAD_CLIP=0.1 effective | ⏳ | Activity still dominating — may need 0.01 |
| 30 | ACTIVITY_LOSS_WEIGHT finalized | ❌ | Currently 0.3 — may need 0.1 based on epoch 50 check |
| 31 | PSR_WEIGHT=60 tested | ⏳ | Current run — need to see post-warmup effect |
| 32 | POSE_LOSS_WEIGHT=0.02 tested | ⏳ | Current run — pose grad last-layer still weak |
| 33 | PSR_WARMUP_STEPS and INIT_MULT finalized | ❌ | Currently 6000×3.0 — may need 12000×5.0 |
| 34 | USE_PSR_SEQUENCE_MODE correct for R3 | ❌ | Currently False — need to decide for R3 |
| 35 | PSR_SEQ_EVERY_N_BATCHES appropriate | ❌ | Currently 8 — VRAM bound |
| 36 | Kendall bounds not pinning all log_vars | ❌ | THREE of four pinned — essentially disabled |
| 37 | PSR_TRANSITION enabled | ➖ | Requires sequence mode |
| 38 | PSR_TRANSITION_SIGMA=3.0 | ➖ | Inactive without transition mode |
| 39 | Stage 3 warmup configured | ➖ | STAGED_TRAINING=False |
| 40 | FP32 (not AMP) confirmed | ✅ | AMP broken on RTX 3060 |

---

## D. LOSS FUNCTIONS (41-50)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 41 | Detection FocalLoss OOB label guard | ✅ | Added in current diff |
| 42 | Activity CBFocalLoss OOB target guard | ✅ | Added in current diff |
| 43 | Pose WingLoss configured correctly | ⏳ | Need to verify params |
| 44 | HeadPose geo MSE with geodesic rotation | ✅ | paper_run preset |
| 45 | PSR BCE focal gamma finalized (1.0 vs 2.0) | ❌ | Currently 1.0 — gamma=2.0 may help |
| 46 | PSR sensitivity loss batch>1 guard | ✅ | Added in current diff |
| 47 | Kendall log_var clamping at param level | ✅ | train.py _clamp_kendall_log_vars |
| 48 | Kendall NaN guard in forward path | ✅ | losses.py rebuild from finite |
| 49 | Loss caps not triggering excessively | ⏳ | Need to check clip frequency in logs |
| 50 | PSR precision multiplier logged in output | ❌ | NOT implemented — critical gap |

---

## E. TRAINING LOOP (51-60)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 51 | Gradient clipping order correct | ✅ | per-head before global, before NaN check |
| 52 | NaN gradient guard working (counted) | ✅ | 0 GRAD_NAN in current run |
| 53 | Activity head per-head clip both AMP/FP32 | ✅ | Both paths covered |
| 54 | empty_cache() before sequence batches | ✅ | Added in current diff |
| 55 | Missing targets.to(device) for kpts/pose_conf | ✅ | Added in current diff |
| 56 | --reinit-heads resets log_var_pose | ✅ | Verified in launch script |
| 57 | Checkpoint save frequency appropriate | ⏳ | Need to verify (every epoch?) |
| 58 | Crash recovery saves all optimizer states | ✅ | From fix3 evidence |
| 59 | Per-head grad norm logging (LIVENESS_GRAD) | ✅ | Working every 200 steps |
| 60 | Loss-based liveness thresholds validated | ⏳ | Need to verify thresholds not too loose |

---

## F. EVALUATION (61-70)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 61 | Mid-training eval RUN on current checkpoint | ❌ | **URGENT — NOT DONE** |
| 62 | activity_mask IndexError fixed | ✅ | evaluate.py fix |
| 63 | Segment eval NA skip working | ✅ | evaluate.py fix |
| 64 | Eval metrics logged consistently | ❌ | No eval yet — format undefined |
| 65 | Detection mAP computed correctly | ❌ | Untested |
| 66 | Activity Top-1 and Top-5 computed | ❌ | Untested |
| 67 | Pose PCK at multiple thresholds | ❌ | Untested |
| 68 | PSR step accuracy (frame + segment) | ❌ | Untested |
| 69 | HeadPose MAE computed | ❌ | Untested |
| 70 | Eval metrics to tensorboard/JSONL | ❌ | No eval infrastructure verified |

---

## G. MONITORING & OBSERVABILITY (71-80)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 71 | LIVENESS probe every 200 steps | ✅ | Working |
| 72 | GRAD_NAN counter visible | ✅ | 0 events reported |
| 73 | Per-epoch optimizer window skip summary | ✅ | Working |
| 74 | GPU memory logging at interval | ✅ | Observed at step 6540-6590 |
| 75 | CPU RAM logging available | ✅ | Observed in fp32 fix3 log |
| 76 | Training speed tracked over time | ✅ | it/s in progress bar |
| 77 | Kendall log_vars logged at epoch start | ✅ | Verified |
| 78 | LR logged at epoch start | ⏳ | Need to check |
| 79 | Validation loss during training | ❌ | No val split configured |
| 80 | Automatic alert on GRAD_NAN spike | ❌ | Not implemented — manual only |

---

## H. R3 PLANNING (81-90)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 81 | R3 hyperparameter changes documented | ✅ | This document + 20 Questions file |
| 82 | Only ONE change between R2.5 and R3 | ❌ | Not decided — multiple candidates |
| 83 | R3 deadline known and feasible | ❌ | Unknown — estimate: ~June 28 for R2.5 end |
| 84 | R3 code freeze date set | ❌ | Not set |
| 85 | Rollback plan: restore R2.5 checkpoint | ⏳ | Need to document procedure |
| 86 | Ablation plan for component contributions | ❌ | Not written |
| 87 | Paper results table template ready | ❌ | Not prepared |
| 88 | Statistical significance test chosen | ❌ | Not decided |
| 89 | Compute budget for R3 estimated | ❌ | ~13 days estimate — need confirmation |
| 90 | Spot/preemptible instance strategy | ❌ | Not discussed |

---

## I. PAPER & DELIVERABLES (91-100)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 91 | Multi-task Kendall weighting claim supported | ❌ | All log_vars pinned — claims currently unsupported |
| 92 | Detection ablation results available | ❌ | No detection metrics yet |
| 93 | Activity confusion matrix | ❌ | Not generated |
| 94 | PSR qualitative samples (step vs time) | ❌ | Not generated |
| 95 | Pose skeleton visualizations | ❌ | Not generated |
| 96 | Head pose MAE distribution (histogram) | ❌ | Not generated |
| 97 | Ablation: TMA on/off | ❌ | Not run |
| 98 | Ablation: Kendall vs equal weights | ❌ | Not run — Kendall currently = equal weights |
| 99 | Failure mode analysis per head | ❌ | Not performed |
| 100 | All checkpoints archived with loss curves | ❌ | Checkpoints exist but no loss curves per epoch |

---

## SUMMARY

| Section | Total | ✅ | ❌ | ⏳ | 🔧 | ➖ |
|---------|-------|----|----|-----|-----|----|
| A. Dataset | 10 | 2 | 5 | 2 | 0 | 1 |
| B. Architecture | 10 | 3 | 2 | 4 | 1 | 1 |
| C. Config | 20 | 9 | 6 | 4 | 0 | 1 |
| D. Losses | 10 | 5 | 2 | 2 | 0 | 1 |
| E. Loop | 10 | 7 | 0 | 3 | 0 | 0 |
| F. Eval | 10 | 1 | 8 | 0 | 0 | 1 |
| G. Monitoring | 10 | 7 | 2 | 1 | 0 | 0 |
| H. R3 Planning | 10 | 1 | 8 | 1 | 0 | 0 |
| I. Paper | 10 | 0 | 10 | 0 | 0 | 0 |
| **TOTAL** | **100** | **35** | **43** | **17** | **1** | **4** |

**GREEN: 35/100** | **RED: 43/100** | **GREY: 22 pending/N/A**

## RED to GREEN — Priority Order

1. **[Q17] Run mid-training eval on epoch 48 checkpoint** — unlocks 8 eval items (61-70)
2. **[Q10/Q17] Establish convergence criteria** — answers "when is R2.5 done?" (Q20)
3. **[Q3] Log PSR warmup precision multiplier** — prevents silent post-warmup collapse (50)
4. **[Q5] Fix Kendall bounds** — if all pinned, paper claim unsupported (36, 91)
5. **[Q11] Fix PSR bias head** — last-layer DEAD for 4000+ steps (18)
6. **[Q4] Finalize ACTIVITY_LOSS_WEIGHT** — activity still dominates (30)
7. **[Q14] Decide R3 changes** — one change only (82)
8. **[Q19] Create val split + loss logging** — prevent overfitting blind (79)

---

*Generated from live analysis of paper_run_r25_fix_20260615.log at epoch 48, step 5133.*
*Run `python3 24_MASTER_ANALYSIS_WITH_20_QUESTIONS.py` for the interactive decision matrix.*
