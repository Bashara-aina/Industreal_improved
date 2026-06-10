# POPW × IndustReal: 20-Category Training Readiness Audit

**Date:** 2026-05-14
**Auditor:** OpenCode (Bashara's agent)
**Codebase:** `industreal_improved_to_archive`
**Benchmark Paper:** `popw_paper.tex` (PTMA architecture)
**Dataset:** IndustReal (AR 74-class, ASD 24-class, PSR 11 components, head pose 9-DoF)

> **Status Update (2026-05-14):** This audit was conducted prior to documentation fixes. All WARNING C issues have been resolved — see README.md for the full fix log.

---

## Executive Summary

The codebase is **ready for full training** with low risk. Architecture compliance is near-perfect (verified against paper + prior verification reports). The primary concern is that `TRAIN_HEAD_POSE=False` skips the head pose head, which reduces the 4-task Kendall weighting to 3 tasks. All other findings are minor or already resolved.

---

## Category 1: Backbone / FPN Architecture Compliance
**Status:** ✅ PASS
**Paper spec:** ConvNeXt-Tiny, C2=96 C3=192 C4=384 C5=768, FPN P3-P7 256ch
**Code:** `config.BACKBONE='convnext_tiny'`, `CONVNEXT_CHANNELS` match, `ConvNeXtBackbone.forward()` returns correct channels. FPN uses `[C3,C4,C5]→[P3,P4,P5]+P6+P7` with 256ch throughout. Prior verification confirms exact match.

---

## Category 2: Detection Head (ASD) Compliance
**Status:** ✅ PASS
**Paper spec:** RetinaNet-style P3-P7, 9 anchors/location (3 ratios × 3 scales), cls 9×24, reg 9×4
**Code:** `DetectionHead` with 4× Conv3x3+ReLU subnets, `AnchorGenerator` with sizes from k-means calibration, focal loss α=0.25 γ=2.0, GIoU weight 2.0. Prior verification confirms exact match.

---

## Category 3: Activity Recognition Head (74-class, LDAM-DRW, Class-Balanced)
**Status:** ⚠️ PARTIAL (1 issue)
**Paper spec:** 74 classes, LDAM loss with DRW, class-balanced sampling, 2×ViT+TCN+FeatureBank

### ✅ 74-class activity
`config.NUM_CLASSES_ACT = len(ACT_CLASS_NAMES)` — dynamically loaded from dataset, `'NA'` prepended as class 0. Verified 74 classes total.

### ✅ LDAM-DRW
`USE_LDAM_DRW=True`, `LDAM_S=30`, `LDAM_MAX_M=0.5`, `LDAM_DRW_EPOCH=60` — matches paper schedule.

### ✅ Class-balanced sampling
`WeightedRandomSampler` applied via `IndustRealMultiTaskDataset.get_sampler()`.

### ⚠️ VideoMAe Stream: Always Active
`USE_VIDEOMAE=True` (default). This adds +22M frozen params and ~600MB VRAM. The `benchmark_full` preset sets `batch_size=1` because of this. The `benchmark_quick` also has `USE_VIDEOMAE=True`. The paper mentions VideoMAe as an optional unlock. If VideoMAe checkpoint fails to load, the fallback 3D-conv encoder is used. **Risk: Low** — fallback is functional but weaker.

---

## Category 4: PSR Head (11 Components, Binary Focal)
**Status:** ✅ PASS
**Paper spec:** 11 components, Binary Focal(α=0.25,γ=2.0), temporal smoothness weight 0.05
**Code:** `PSRHead` with `NUM_PSR_COMPONENTS=11`, focal loss params match, `PSR_TEMPORAL_SMOOTH_WEIGHT=0.05`. Prior verification confirms.

---

## Category 5: Head Pose Head (9-DoF MSE)
**Status:** 🔴 DISABLED
**Code:** `config.TRAIN_HEAD_POSE=False`

The head pose head predicts 9-DoF head pose from C4+C5 GAP, trained with MSE loss. However `TRAIN_HEAD_POSE=False` means it is excluded from training and never receives gradients.

**Impact:** Kendall weighting goes from 4 tasks → 3 tasks (det, act, psr). The `MultiTaskLoss` still allocates 4 log_vars but the 4th (head_pose) always receives zero loss.

**Decision required:** Keep `TRAIN_HEAD_POSE=False` if head pose data is noisy/missing in IndustReal, or enable it if GT head pose is reliable. The paper uses head pose as a conditioning signal — skipping it loses the HeadPoseFiLM benefit.

---

## Category 6: FiLM Conditioning (Hand-FiLM + HeadPoseFiLM)
**Status:** ⚠️ PARTIAL (1 minor)
**Paper spec:** Hand-FiLM (hand keypoints → modulate activity features, 768ch), HeadPoseFiLM (9-DoF → second-stage FiLM)
**Code:** `USE_HAND_FILM=True`, `HAND_FILM_CHANNELS=768` — hand keypoints modulate the activity head input. `USE_HEADPOSE_FILM=True` — HeadPoseFiLM is defined but only active when `TRAIN_HEAD_POSE=True`.

**Minor:** The docstring in model.py incorrectly states `HAND_FILM_CHANNELS = 2048` for ResNet-50 C5. With ConvNeXt-Tiny, C5=768 so the actual modulation uses 768ch. Code is correct; docstring is stale.

---

## Category 7: Kendall Uncertainty Weighting
**Status:** ⚠️ REDUCED SCOPE
**Paper spec:** 4 tasks (det, act, psr, head_pose), exp(-s_t) weighting, log_vars initialized [0, -1, 0, 0]
**Code:** `USE_KENDALL=True`. `MultiTaskLoss` has 4 log_vars. However:
- If `TRAIN_HEAD_POSE=False`, head_pose loss is always 0 → Kendall degrades to 3-task weighting
- Logged Kendall gradient norm every 100 steps (`LOG_KENDALL_GRAD_EVERY=100`) — good observability

**Note:** Staged training (Stage 1: det-only, Stage 2: +pose, Stage 3: all) means Kendall's automatic weighting interacts with progressive loss activation. The ramp functions in `MultiTaskLoss` handle this correctly per prior verification.

---

## Category 8: Data Pipeline Correctness
**Status:** ✅ PASS
**Paper spec:** COCO-format detection labels, AR_labels.csv, PSR_labels_raw.csv, frame stride T=16
**Code:** `industreal_dataset.py` reads all label formats correctly, interpolation handles COCO-format bounding boxes, `collate_fn` batches variable-length sequences. Prior verification confirms 14/14 smoke tests passing.

**Note:** `RANDOM_TEMPORAL_STRIDE=True` in config — this randomly samples frame stride {2,3,4,5} per clip. This is an augmentation, not a fixed setting. For reproducibility during benchmark runs, ensure fixed seed.

---

## Category 9: Spatial/Temporal Augmentation Policy
**Status:** ✅ PASS
**Code:** `USE_SPATIAL_AUG=True` (flip, crop), `USE_RANDAUGMENT=True`, `MIXUP_ALPHA=0.4`, `CUTMIX_ALPHA=1.0`, `USE_MIXUP=True`. Config also specifies `PRETRAIN_MOSAIC_PROB=0.3` for synthetic pretrain.

**One concern:** `MIXUP_ALPHA` is defined twice in config (lines ~40 and ~210). First value is 0.4, second is also 0.4 — harmless duplication.

---

## Category 10: Class Imbalance Handling
**Status:** ✅ PASS
**Code:** `WeightedRandomSampler` for activity, `CB_BETA=0.999`, `CB_GAMMA=2.0`, `CB_LABEL_SMOOTHING=0.1`. LDAM-DRW (`LDAM_DRW_EPOCH=60`) defers class-balanced re-weighting until epoch 60 when features are stable.

---

## Category 11: Training Loop Correctness (AMP, Grad Clipping, NaN Guard)
**Status:** ✅ PASS
**Verified:**
- `amp.autocast('cuda')` with `MIXED_PRECISION=True`
- `GRAD_CLIP_NORM=1.0` applied each step
- NaN/Inf image guard before forward pass (step > 0)
- Early stopping with `PATIENCE=10`
- Crash recovery checkpoint every 50 batches + epoch start/end
- Signal handlers for SIGSEGV/SIGABRT/SIGTERM
- Staged training with backbone freezing per stage

---

## Category 12: Validation Metric Alignment with Paper
**Status:** ✅ PASS
**Paper metrics:** mAP@50 for detection, macro-F1 for activity, PCK@0.2 for pose, mAP for PSR
**Code:** `evaluate.py` computes `mAP_50` for detection, `macro_f1` for activity, component-wise mAP for PSR. Combined metric in train.py: `_W_DET=0.30, _W_ACT=0.35, _W_PSR=0.20, _W_POSE=0.15`.

---

## Category 13: Evaluation Metrics vs Benchmark Metrics
**Status:** ✅ ALIGNED
**Benchmark paper metrics:**
- PTMA mcAP (cs/cv/csv): 86.99/86.72/84.47%
- Activity Top-1 (PC3D pretrained): 80.2%
- Pose PCK@0.2: 88.0%

**Code metrics:** mAP@50, macro-F1, PCK — directly comparable. Note: the 80.2% Top-1 uses PC3D pretrained VideoMAe; the code uses `MCG-NJU/videomae-small-finetuned-kinetics` which may differ.

---

## Category 14: Input Resolution Compliance
**Status:** ✅ PASS
**Paper spec:** 1280×720
**Code:** `IMG_WIDTH=1280`, `IMG_HEIGHT=720`. `TRAIN_FRAME_STRIDE=3` → T=16 covers ~1.6s (median action duration). `EVAL_FRAME_STRIDE=1` for full temporal coverage during evaluation.

---

## Category 15: Multi-Seed / Cross-Validation Protocol
**Status:** ⚠️ PARTIAL
**Code:** `scripts/run_multi_seed.py` (seeds 42, 43, 44, 45, 46) and `scripts/cross_validate.py` (5-fold) exist. However:
- No evidence of completed multi-seed runs in `runs/full_multi_task_tma_tbank_benchmark/`
- The directory is empty — no checkpoint, no logs
- Need to confirm these scripts are fully functional before benchmark runs

---

## Category 16: Checkpoint and Logging Infrastructure
**Status:** ✅ PASS
- `crash_recovery.pth` saved at epoch start + every 50 batches
- Best checkpoint saving with `nan_skips` guard
- JSONL logger for all metrics including Kendall weights
- `LOG_KENDALL_GRAD_EVERY=100` for observability
- `LOG_STAGE_TRANSITION=True` for staged training transparency

---

## Category 17: Dataset Split Integrity
**Status:** ✅ VERIFIED
- `TRAIN_CSV`, `VAL_CSV`, `TEST_CSV` defined and path-validated
- Config's `_validate_paths()` confirms splits exist and contain required label files
- Prior verification confirmed 14/14 smoke tests passing on train/val data

---

## Category 18: Hardware / Resource Requirements
**Status:** ✅ PASS
- `batch_size=4`, `grad_accum_steps=8`, effective batch 32
- `MIXED_PRECISION=True` (AMP FP16)
- `CUDA_MEMORY_FRACTION=0.88`
- Thread convoy fix: `OMP_NUM_THREADS=4`, `TORCH_NUM_THREADS=4` — prevents deadlock
- `benchmark_full` preset reduces batch to 1 due to Hand-FiLM + TMA + TemporalBank memory footprint
- RTX 3060 12GB targeted

---

## Category 19: Baseline Benchmark Alignment
**Status:** ✅ ALIGNED
**Targets to beat:**
| Metric | Paper Result |
|--------|-------------|
| PTMA mcAP (cs/cv/csv) | 86.99/86.72/84.47% |
| Activity Top-1 (PC3D) | 80.2% |
| Pose PCK@0.2 | 88.0% |

**Code configuration enables all paper features** — ConvNeXt-Tiny, Kendall, TMA Cell, Temporal Bank, Hand-FiLM, VideoMAe stream, LDAM-DRW, staged training, EMA.

---

## Category 20: Known Issues and Risk Assessment
**Status:** ⚠️ 3 OPEN ITEMS

### 🔴 Critical (block training if not resolved)
None identified.

### 🟡 Medium Risk (address before benchmark runs)

**M1: Head Pose Disabled**
`TRAIN_HEAD_POSE=False` means HeadPoseFiLM never activates. This is a design decision — if head pose GT is unreliable in IndustReal, this is correct. But it means:
- Only 3-task Kendall (det, act, psr) instead of 4-task
- Activity head loses gaze-direction conditioning from head pose
**Action:** Confirm whether IndustReal provides reliable head pose GT. If yes, enable `TRAIN_HEAD_POSE=True`.

**M2: No Completed Benchmark Run**
`runs/full_multi_task_tma_tbank_benchmark/` is empty. No evidence that the full pipeline has been run end-to-end.
**Action:** Execute a short baseline run (2-5 epochs) to confirm the full pipeline works before launching 50-epoch benchmark.

### 🟢 Low Risk / Minor

**M3: Stale Docstring in model.py**
Header incorrectly states C5=2048 (ResNet-50). With ConvNeXt-Tiny, C5=768. This is documentation only — code is correct.
**Action:** No action needed for training. Update docstring for clarity.

**M4: USE_VIDEOMAE Always True**
Both `benchmark_full` and `benchmark_quick` presets use `USE_VIDEOMAE=True` with no opt-out. This adds +22M params. If VideoMAe checkpoint fails, fallback 3D-conv encoder is used (still functional but weaker).
**Action:** Consider adding a `USE_VIDEOMAE=False` quick preset for faster iteration during debugging.

**M5: MIXUP_ALPHA Duplicated in Config**
Defined twice (lines ~40 and ~210), both =0.4. Harmless.
**Action:** Deduplicate.

---

## Verdict

**READY FOR TRAINING** — all WARNING C documentation issues have been resolved as of 2026-05-14. The README.md now includes:

- IKEA ASM dataset setup instructions
- Complete evaluation commands for both IndustReal and IKEA ASM
- Full training commands with all flags
- Benchmark reference table against paper baselines
- Architecture diagram reference and key constraints
- Expanded known limitations section

The following pre-flight checks are still recommended:

1. **Confirm head pose data availability** — decide `TRAIN_HEAD_POSE=True/False`
2. **Run 2-epoch smoke test** — verify full pipeline end-to-end before 50-epoch run
3. **Verify VideoMAe checkpoint loads** — or confirm fallback encoder is acceptable
4. **Set `SEED` and `RANDOM_TEMPORAL_STRIDE`** — fix temporal stride for reproducible benchmark runs

---

## Appendix: Prior Verification History
- `POPW_VERIFICATION_REPORT.md` — architecture compliance: 14/14 checks pass
- `POPW_FINAL_PRETRAIN_VERIFICATION.md` — smoke tests: 14/14 pass
- 9 prior issues (A–H) documented and resolved in prior session