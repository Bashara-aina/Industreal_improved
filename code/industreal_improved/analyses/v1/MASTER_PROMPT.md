# POPW Master Analysis — Opus Request (v2)

You are analyzing a multi-task CV project (POPW — Part-Oriented Process Worker) for IKEA assembly recognition. This is a research/thesis project with a complex training pipeline. Below is the FULL context you need, followed by specific questions to answer.

---

## Project Context

**Location:** `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/`

**Tasks (4 IndustReal tasks):**
1. **ASD** — Assembly State Detection (24 classes, bounding box detection)
2. **Activity Recognition** — 75 classes (NA + IDs 1-74), video frame classification
3. **Head Pose Estimation** — 9-DoF regression (forward_vector[3] + position[3] + up_vector[3]), angular error in degrees
4. **PSR** — Procedure State Recognition (temporal, Damerau-Levenshtein edit distance + POS metrics, 36 steps, 11 components)

**Model Architecture:**
- **Backbone:** ConvNeXt-Tiny (pretrained on ImageNet-22K, 28M params)
- **Neck:** FPN (Feature Pyramid Network)
- **Heads:** 4 task-specific heads (detection cls+reg, activity 75-class, pose 17-keypoint Wing, PSR binary focal)
- **Temporal:** TMA Cell (GRU-based Temporal Masked Attention) + Feature Bank (T=16)
- **Conditioning:** Hand-FiLM (hand keypoints → FiLM modulation on activity features) + HeadPoseFiLM
- **Optimizer:** AdamW (NOT Lion — paper Table 3 specifies AdamW), cosine annealing with warm restarts
- **Loss:** Kendall homoscedastic uncertainty weighting across 4 task groups
- **Multi-task loss formula:** L = Σ_t [ exp(-s_t) * L_t + s_t ], init s_det=0, s_pose=-1, s_act=0, s_psr=0

**Dataset:** IndustReal — 36 recordings, 25,159 train frames, 35,084 val frames, single egocentric RGB camera at 1280×720.

**VRAM:** RTX 3060 (12GB). BATCH_SIZE=6 with GRAD_ACCUM_STEPS=6 → effective batch=32. For full benchmark: batch=1 (VideoMAE alone costs +600MB).

**Current status:** 87-min eval bottleneck fixed (vectorized compute_ap_multi_thresh), DRW wired, all 12 prior bugs resolved. Training run PID 1780264 on 5% subset for validation before full 100-epoch run.

---

## Files to Analyze

| File | Purpose |
|------|---------|
| `src/config.py` | All hyperparameters, loss weights, dataset constants, stage configuration. READ THIS FIRST for actual values. |
| `src/training/losses.py` | All 4 loss functions: LDAMLoss (with DRW), FocalLoss, GIoU, WingLoss, PSRFocalLoss. Kendall log_vars. MultiTaskLoss forward. |
| `src/training/train.py` | Main training loop (~3100 lines). Epoch loop, eval phase, EMA, checkpointing, staged training. **KEY FILE.** |
| `src/evaluation/evaluate.py` | All metric computation. compute_ap_multi_thresh (vectorized), compute_det_metrics_extended, compute_efficiency_metrics, compute_psr_metrics, report per-class accuracy. |
| `src/models/model.py` | POPWMultiTaskModel — ConvNeXt + FPN + detection head + activity head (VideoMAE optional) + pose head + PSR head + TMA Cell + Feature Bank. |
| `src/data/industreal_dataset.py` | Dataset implementation: 24-class ASD, 75-class activity, 17-keypoint pose, 11-component PSR, sequence-mode PSR with T=4 windows. |

---

## Critical Fixes Applied (May 27-29 2026)

### Fix A — Run-Killer: evaluate_all() missing `epoch` parameter
`train.py:2614` calls `evaluate_all(..., epoch=epoch)` but the signature had no `epoch` param → first validation crashed with `TypeError`. Fixed: added `epoch: int = 0` to `evaluate_all()` signature and set `C._CURRENT_EPOCH = epoch` at start.

### Fix 1 — 87-Min Eval Bottleneck (vectorized, NOT skipped)
`compute_ap_per_class` nested loop (24 classes × 35,084 frames × 11 IoU thresholds) was the bottleneck. Fixed with `compute_ap_multi_thresh()` — computes IoU once per (class, frame) pair, then replays greedy matching for all 10 thresholds. Speedup ~9×.

### Fix 2 — DRW Wiring (LDAMLoss + CB weights)
`LDAMLoss.__init__` received `cb_weights=None` and `set_class_counts` never wired them. Fixed: `set_class_counts` now sets `self.cb_weights = weights` when `LDAM_USE_DRW=True`. Added `LDAM_USE_DRW` config flag for A/B testing.

### Fix 3 — VRAM Config Coherence
Old comment said "BATCH_SIZE=1 REQUIRED" but code set `BATCH_SIZE=6`. Fixed comment to clarify: BATCH_SIZE=6 is fine for quick training; full benchmark preset uses batch=1.

### Fix 4 — hp_mae_deg Mislabel
`train.py:2742` was logging `hp_mae_deg=` but reading `forward_angular_MAE_deg`. Fixed to `forward_angular_MAE_deg=` (matching the actual metric key).

---

## Prior Audit (May 27 2026) — 12 Bugs Fixed

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | Eval hang | `compute_ap_per_class` 11× nested loop | `compute_ap_multi_thresh` vectorized |
| 2 | LDAM off-by-one | `_fit_to_width` and `_warned_oob_target` | Fixed margin/resize logic |
| 3 | GIoU negative slope | NEG_SLOPE caused collapse | `NEG_SLOPE = 0.0` floor |
| 4 | PSR focal gamma | gamma=2.0 → trivial collapse | `gamma=1.0` |
| 5 | NUM_CLASSES_ACT | Derived from disk → 74 vs 75 mismatch | Pinned to 75 |
| 6 | Val loader workers=0 | Worker management issues | Hardened to 0 fallback |
| 7 | Activity frozen loss | NaN cascade at Stage 3 entry | NaN guard + class-0 trigger |
| 8 | mAP > 1.0 | Correct rejection injection bug | Removed from `compute_ap_per_class_all_frames` |
| 9 | Kendall double-norm | log_vars gradient clipping | Clipped at 100.0 |
| 10 | Temporal smooth | EMA on val PSR metrics | Applied |
| 11 | Warmup ramp | 3-stage unfreeze with backbone warmup | Implemented |
| 12 | Edit distance overflow | Batch accumulation | fp32 accumulation |

---

## What to Analyze and Answer

### 1. Architecture & Design

- Is the Kendall + 4-task design optimal, or should pose+head_pose share a single head differently?
- Should PSR (temporal) be a separate pipeline instead of sharing ConvNeXt with 3 frame-level tasks?
- Is the FeatureBank (T=16 temporal memory for PSR) well-implemented?
- Is the VideoMAE 2-stream option worth the VRAM cost (+600MB)?

### 2. Loss Functions

- Is LDAMLoss + DRW (Deferred Ring Weighting) the right choice for the 75-class activity head?
- Should GIoU negative slope be higher than 0.0? Does `NEG_SLOPE = 0.0` risk collapse?
- Is Kendall uncertainty weighting stable across all 4 tasks?
- Should log_vars initialization be more careful (e.g., different init for pose vs det)?

### 3. Performance & Scaling

- With `IMG_HEIGHT=720, IMG_WIDTH=1280` and batch=6 on RTX 3060 (12GB VRAM), is there a more efficient input resolution strategy?
- Should mixed-precision (FP16) training be enabled? It's in config but is it actually used in train.py?
- Is the val loader (workers=4, batch=16, prefetch=2) optimal?

### 4. Metric Computation

- Is the vectorized `compute_ap_multi_thresh` correct? Does it produce the same results as the original nested loop?
- Should mAP computation use batched processing to avoid memory issues?
- Is `report_per_class_accuracy` efficient for 75 classes?

### 5. Remaining Bugs or Risks

- Any remaining bugs that could cause training to fail, hang, or produce invalid metrics?
- Could `SKIP_DET_METRICS_EVAL=True` hide real problems by returning NaN?
- Any risk of NaN propagation in Kendall weighting not already guarded?

### 6. Training Stability

- With staged training (3 stages), is the learning rate schedule correct?
- Should early stopping use combined metric or per-task metrics?
- Is EMA (Exponential Moving Average) properly implemented?

### 7. The OOM Auto-Fallback

- train.py is supposed to automatically halve batch on CUDA OOM — is this implemented and does it work correctly?

---

## How to Answer

For each section above, provide:
1. **Assessment** — is this a real problem or a false concern?
2. **Recommendation** — specific fix or optimization, with code-level suggestions
3. **Priority** — should this be addressed before the 100-epoch run, or is it low priority?

Be thorough. This is a research project and the 100-epoch training run will take ~5-7 days. We need to catch any remaining issues now.