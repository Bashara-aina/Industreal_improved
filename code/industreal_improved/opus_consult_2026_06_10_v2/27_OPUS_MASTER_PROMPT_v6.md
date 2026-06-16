# MASTER PROMPT FOR OPUS v6 — Death Spiral Edition
## POPW IndustReal Consultation — Current State: RF1 STUCK

> **Send this prompt to Claude Opus when asking about the `opus_consult_2026_06_10_v2` folder.**
> Opus should read the referenced files in order and produce concrete implementation guidance.

---

## SITUATION — 30-SECOND SUMMARY

We have a unified multi-task model (ConvNeXt-T + FPN + 5 heads: detection, activity, pose, head pose, PSR) for IndustReal assembly understanding. **The model has NEVER produced a single valid mAP metric.** Every training run since June 11 has been stuck in a class imbalance death spiral at RF1 (20% data, detection bootstrap).

**Current state (2026-06-16 17:45 UTC):** RF1 training cycling at epoch 58, predicted to die at step ~1300. Five retry strategies exhausted. All non-det heads DEAD (zero gradient). Detection head bounces between ALIVE (6.8 liveness) and DEAD (0.047) every ~200 steps.

---

## FOLDER STRUCTURE — WHAT TO READ

### Start Here (most current summary)
| File | Size | Why Read |
|------|------|----------|
| `26_RF1_RF10_COMPREHENSIVE_STATUS.md` | 58 KB | **Primary document.** Complete state: architecture, 4-phase timeline, death spiral analysis, all strategies tried, uncertainties |
| `25_R3_100_CHECKLIST.md` | 10 KB | The 100-item readiness checklist before RF1 launched |
| `23_TRAINING_RUNS_AND_CURRENT_STATUS.md` | 5 KB | Run log from the R3 era |

### Context Documents (read these second)
| File | Size | Why Read |
|------|------|----------|
| `00_JOURNEY_AND_STATUS.md` | 14 KB | Complete project history from R1 to RF10 transition |
| `01_PROBLEMS_ROOT_CAUSES.md` | 12 KB | All 24 root causes identified (RC-1 through RC-24) |
| `02_GOALS_AND_BENCHMARKS.md` | 9 KB | Target metrics, baselines, priority ordering |
| `03_ARCHITECTURE_DEEP_DIVE.md` | 15 KB | Module-by-module: backbone, FPN, 5 heads, FiLM, FeatureBank |
| `previous_opus_answer.md` | 44 KB | Previous Opus v5 analysis (forensic audit) |

### Code Files (in `code/` — read when referenced)
| File | Size | Purpose |
|------|------|---------|
| `train.py` | 179 KB | Training loop: staged training, EMA, optimizer, checkpointing |
| `model.py` | 91 KB | Full model: ConvNeXt-T, FPN, 5 task heads, FiLM, ViT |
| `losses.py` | 75 KB | All losses: Focal, GIoU, Wing, LDAM-DRW, Kendall uncertainty |
| `industreal_dataset.py` | 60 KB | Dataset: 5 tasks, frame cache, subset_ratio, augmentation |
| `config.py` | 44 KB | Full config: Tier 1-3 flags, stage presets, RF1-RF10 settings |
| `evaluate.py` | 174 KB | Evaluation: mAP, Top-1/5, PSR F1/POS, efficiency |
| `psr_transition.py` | 12 KB | PSR transition predictor |
| `roi_detector.py` | 15 KB | Alternative ROI-centric detector |
| `video_stream.py` | 16 KB | K400-pretrained video stream |
| `head_pose_geo.py` | 10 KB | 6D rotation head pose |
| `uncommitted_changes_r25_fix_20260615.patch` | 56 KB | Full diff of all RF1-RF10 changes |

### Logs (read after code)
| File | Size | Why Read |
|------|------|----------|
| `logs/paper_run_r25_fix_20260615.log` | 2.2 MB | The R25 fix run that collapsed at epoch 48 |
| `logs/recovery_r1_det_bootstrap.log` | 6.3 MB | The det bootstrap that first revealed death spiral |
| `logs/recovery_train8_run8.log` | 17.6 MB | Longest run (epoch 84) — shows healthy trajectory before death |
| `logs/train_main.log` | 235 KB | Original collapse log |
| `logs/reinit_runner.log` | 42 KB | Reinitialization process |
| `logs/eval_post_retrain.log` | 28 KB | Post-retrain evaluation showing corrupted metrics |

### Evidence
| File | Size | Why Read |
|------|------|----------|
| `evidence/eval_metrics.json` | 65 KB | Full evaluation metrics — all zeros |

### Paper
| File | Why Read |
|------|----------|
| `popw_paper_improved.tex` | The paper we're targeting (81 KB) |

### Previous Answers
| File | Why Read |
|------|----------|
| `17_OPUS_ANSWER_v5.md` | Previous Opus v5 advice (21 KB) |
| `10_OPUS_ANSWER_v2.md` | Opus v2 advice (23 KB) |

---

## THE CORE PROBLEM: CLASS IMBALANCE DEATH SPIRAL

The detection head (164,544 FCOS-style anchors per frame across 5 FPN levels) enters a **death spiral** at step ~1200-1300:

1. **99.3% empty frames**: With `subset_ratio=0.2`, only 0.7% of batches contain GT boxes
2. **Background dominates**: Model learns "confident background" everywhere
3. **Positive logit decays**: `det_cls_max` drops from +2.909 → +0.055
4. **Focal loss vanishes**: When all logits are "confident background" (p~0.5 each), focal loss gradient goes to ~0
5. **Detection head DEAD**: Liveness drops from 6.8 → 0.047
6. **All other heads DEAD**: Activity, PSR, pose, head_pose show zero gradient (they depend on FPN features that collapse)
7. **Bounce-and-die**: When a GT batch occasionally appears, head revives briefly (step 1151, max=6.56), then dies again by step 1300

### What We've Tried (All 5 Strategies, None Work)

| Strategy | Result |
|----------|--------|
| 1. `default` (pi=0.1) | Collapsed in 15 min, cls_mean -2 → -20 |
| 2. `reduce_lr_5x` | Still collapsed — all heads DEAD |
| 3. `reduce_lr_2x_warmup_2x` | Still collapsed |
| 4. `reduce_lr_10x_warmup_2x` | Still collapsed |
| 5. `reduce_lr_20x_warmup_3x` (current) | Produces IDENTICAL trajectory to strategy 2 at same step numbers |

**Critical finding**: LR reduction does NOT fix the problem. The model trajectory is deterministic from the checkpoint and weight initialization. The death spiral is a data problem, not an optimization problem.

### Other Critical Findings
- **1145+ PSR_DIAG entries**: All showing bit-exact same `loss=1.546e-08` — PSR head stuck in zero-loss equilibrium (predicts mostly zeros for 20/22 components)
- **No mAP metrics ever**: All validation entries have samples but produce zero det_mAP50/95
- **Checkpoint never updated**: `latest.pth` is 202+ min stale — no retry completes a full epoch
- **All non-det heads DEAD**: Act=0.00, PSR=0.00, Pose=1e-6, Head_pose=1e-6

---

## THE FIX WE NEED: GT FRAME OVERSAMPLING

The root fix is to ensure every training batch contains at least some frames with GT boxes. This is NOT an LR issue, NOT a model capacity issue — it's a data sampling issue.

### What GT Oversampling Means
- Force-sample frames that contain detection/pose/head_pose labels
- Guarantee N positive frames per batch (e.g., at least 1-2 frames with boxes out of 8)
- This breaks the "confident background" equilibrium by ensuring positive gradient signal every step

### Questions for Opus
1. **GT oversampling implementation**: Exactly how to modify the dataset/batch sampler to guarantee GT frames in every batch? Concrete code.

2. **Advancing to RF2**: Once RF1 produces stable det metrics, what's the RF2 transition protocol? RF2 adds pose + head_pose at 30% data.

3. **PSR recovery**: PSR shows zero-loss equilibrium (loss=1.546e-08). Is this fixable via GT oversampling too, or does PSR need a fundamentally different loss/target representation?

4. **Dead non-det heads**: Are the activity, pose, head_pose heads dying because they depend on collapsed FPN features, or because they have independent bugs (e.g., log_var pinning, missing gradient paths)?

5. **Are we using the right detection head**: 164K anchors is extremely heavy for a single RTX 3060. Should we switch to the ROI-centric detector (in `roi_detector.py`) which has fewer outputs? Tradeoffs?

6. **Minimum viable experiment**: What's the smallest experiment that proves the architecture CAN learn? Single-batch overfit test protocol?

---

## HARDWARE CONSTRAINT
- **Single RTX 3060 12GB** — no multi-GPU
- **64GB system RAM**
- **PyTorch 2.2, CUDA 12.1**
- **Max 80M parameters**

## WHAT WE NEED OPUS TO PRODUCE
Implementation guides (concrete PyTorch code with line numbers) for:
1. **Fixing the death spiral** (GT oversampling implementation)
2. **Getting RF1 to pass gate** (det_mAP50 >= 10%)
3. **Advancing through RF2-RF10** with clear per-stage protocols
4. **PSR recovery** from zero-loss equilibrium
5. **Multi-task head revival** (activating all 5 heads simultaneously)

---

## VERIFICATION BEFORE WRITING
1. Read `26_RF1_RF10_COMPREHENSIVE_STATUS.md` (the full 58KB — most important)
2. Read `00_JOURNEY_AND_STATUS.md` for history
3. Read `23_TRAINING_RUNS_AND_CURRENT_STATUS.md` for run chronology
4. Read `01_PROBLEMS_ROOT_CAUSES.md` for the 24 root causes
5. Read `03_ARCHITECTURE_DEEP_DIVE.md` for architecture understanding
6. Optionally read the previous Opus answer (`17_OPUS_ANSWER_v5.md`) to see what's already been suggested
7. Reference code files as needed — focus on `train.py`, `industreal_dataset.py`, `config.py`
