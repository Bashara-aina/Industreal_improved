# 21 — Final 10× Deep Audit: Complete Status (2026-06-13)

All code on `main` branch, HEAD `a0f1ce7`.

---

## Audit History

| Round | Audit | Items | Code-Implemented | Status |
|-------|-------|-------|-----------------|--------|
| v4 | Opus v4 RC-28/RC-29 diagnosis | 6 items | 6/6 | ✅ |
| v5 | 50-item checklist (18_HONEST_FEASIBILITY_AUDIT) | 50 | 40/50 | ✅ |
| v5.1 | 100-item pre-training readiness (19_PRE_TRAINING_READINESS_AUDIT) | 100 | 71/100 | ✅ |
| v5.2 | 10-pass deep audit | 10 | 8/10 | ✅ |

## 10-Pass Deep Audit Results

| Pass | Verdict | Evidence |
|------|---------|----------|
| **P1** Provenance | ✅ | HEAD `a0f1ce7` on `main`, 17 commits |
| **P2** Static existence | ✅ | 20+ fix markers present: psr_transition EXISTS in git, ANCHOR_SIZES=(96,160,256,384,512), DET_EVAL=0.001, USE_GEO_HEAD_POSE, FEATURE_BANK_DETACH, ASSERT_AND_CRASH (4×), liveness probe (8×), activity_mask (3×), stratification (9×), dim gate (4×), IMG_SIZE assert, PSR -1 transient, SIMPLIFY_LOSS |
| **P3** Import smoke | ✅ | Zero ImportError. `build_transition_targets` imports cleanly from `src.models.psr_transition` |
| **P4** Dim gate | ✅ | Transition gated to `self.use_psr_transition and outputs['psr_logits'].dim() == 3` |
| **P5** Liveness | ✅ | Probe code present (losses.py:1307), LIVENESS_EVERY=200. FEATURE_BANK_DETACH flag for temporal gradient |
| **P6** G1/G3 smoke | ✅ | committed=27, skipped=0, nan_skips=0, 139s (bringup_smoke_g1g3.log) |
| **P7** Data audit | 📋 | 1-epoch pass on training box (GPU required) |
| **P8** Eval dry-run | 📋 | Capped val subset on training box |
| **P9** Detection | ✅ | Anchors calibrated, conf 0.001, IMG_SIZE guard, bestIoU up to 0.94 confirmed in R1 v4 |
| **P10** Dress rehearsal | 📋 | Full epoch subset 1.0 on training box |

**8/10 passes verified. 2 pending GPU execution.**

## 3 Blockers — Resolution

| Blocker | Issue | Resolution | Commit |
|---------|-------|-----------|--------|
| **A** | `psr_transition.py` not in git → import fails → silent fallback | Added to git: `psr_transition.py`, `head_pose_geo.py`, `roi_detector.py`, `video_stream.py` | `da2bd60` |
| **B** | Activity eval per-recording, not per-action-segment → MViTv2-incomparable | Deferred to R2 — requires new `ActivityClipDataset` class | 📋 |
| **C** | Winnable-task fixes OFF by default | `paper_run` preset created. User toggles USE_PSR_TRANSITION / USE_GEO_HEAD_POSE / FEATURE_BANK_DETACH=False at R2/R2.5/R3 | `a0b87d8` |

## All Config Flags (17 total)

| Flag | Default | Purpose |
|------|---------|---------|
| `ASSERT_AND_CRASH` | False (env) | NaN → crash, not silent 1e-4 |
| `SIMPLIFY_LOSS` | False (env) | Bypass caps/ramps for diagnosis |
| `LIVENESS_EVERY` | 200 | Per-head ALIVE/DEAD probe |
| `USE_LDAM_DRW` | False | CE+LS instead of LDAM s=30 |
| `PSR_SENSITIVITY_WEIGHT` | 0.0 | Remove -log(std) penalty |
| `USE_PSR_TRANSITION` | False | Gaussian-smeared transitions |
| `PSR_TRANSITION_SIGMA` | 3.0 | Transition smearing width |
| `USE_PSR_ORDER_PRIOR` | False | B2-style assembly order constraints |
| `USE_GEO_HEAD_POSE` | False | 6D rotation + geodesic loss |
| `FEATURE_BANK_DETACH` | True | Gradient through temporal bank |
| `FEATURE_BANK_SLOT_OVERWRITE` | True | Live frame in last bank slot |
| `DET_EVAL_SCORE_THRESH` | 0.001 | YOLOv8-comparable |
| `ANCHOR_SIZES` | (96,160,256,384,512) | Calibrated to GT 146-594px |
| `DET_METRICS_EVERY_N` | 5 | Full mAP cadence |
| `GATE_EVAL_MAX_BATCHES` | 200 | Fast gate eval |
| `TRAIN_MAX_STEPS` | 0 (env) | Early stop for smokes |
| `IMG_SIZE` guard | assert | Prevents silent detection zeroing |

## Training Ladder

| Stage | Preset | Subset | Epochs | Key Toggles |
|-------|--------|--------|--------|-------------|
| R0 | recovery_det_only | 0.25 | smoke | ASSERT_AND_CRASH=1 SIMPLIFY_LOSS=1 |
| R1 | recovery_det_only | 0.25 | 3-5 | det+head_pose only |
| R1.5 | recovery_det_only | 1.0 | 20 | + PRETRAIN_DET_ON_SYNTH |
| R2 | recovery | 0.25 | 4 | USE_LDAM_DRW=False, FEATURE_BANK_DETACH=False |
| R2.5 | recovery | 0.25 | 4 | USE_PSR_TRANSITION=True, USE_PSR_ORDER_PRIOR=True |
| R3 | paper_run | 1.0 | 50 | USE_GEO_HEAD_POSE=True, USE_EMA=True |
| R4 | paper_run | 1.0 | — | Assembly/Error-Verif derivations |
| R5 | paper_run | 1.0 | — | Multi-seed ×3 + efficiency |
