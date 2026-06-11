# Goals, Targets & Benchmarks — What We Must Beat
## POPW Opus Consultation v2 (2026-06-11)

---

## 1. The Core Mission

**Build a unified multi-task model that genuinely learns across all heads** — detection, activity, pose, head pose, and PSR — in a single forward pass. The model must not catastrophically fail on any head. It must demonstrate that cross-task feature sharing provides a tangible benefit over separate specialist models.

**We are open to ANY architectural change**: different backbone, different heads, different training flow, different loss functions. The only constraint is: one model, one forward pass, multiple tasks, on the IndustReal dataset.

---

## 2. Benchmark Targets to Beat (IndustReal Dataset)

### 2.1 Assembly State Detection (ASD)

| Metric | Baseline | Method | Our Target | Difficulty |
|--------|----------|--------|------------|------------|
| mAP (b-boxed) | **83.80%** | YOLOv8m (COCO+synth+real) | ≥ 70% (competitive) | HARDEST |
| mAP@0.5 (all frames) | **64.10%** | YOLOv8m (COCO+synth+real) | ≥ 50% | HARD |
| mAP@[0.5:0.95] (b-boxed) | ~55% (est.) | YOLOv8m | ≥ 40% | HARD |

**Key insight**: YOLOv8m's 83.8% used COCO pretraining + 260K synthetic images + real fine-tuning. We have synthetic data wired (`PRETRAIN_DET_ON_SYNTH=True`) but unused. Without synthetic pretrain, realistic ceiling is ~0.50-0.60 mAP.

**Winning claim**: "Within ~3 points of YOLOv8m at 1/3 the deployed compute" is still a winning paper claim if we can show efficiency gains.

### 2.2 Activity Recognition

| Metric | Baseline | Method | Our Target | Difficulty |
|--------|----------|--------|------------|------------|
| Top-1 (RGB-only) | **65.25%** | MViTv2 (Kinetics-400) | ≥ 45% | ACHIEVABLE |
| Top-5 (RGB-only) | **87.93%** | MViTv2 (Kinetics-400) | ≥ 70% | ACHIEVABLE |
| Top-1 (RGB+VL+stereo) | **66.45%** | MViTv2 ensemble | N/A (different modality) | — |

**Key insight**: The baseline uses clip-level evaluation (16 uniform frames per action segment → single prediction). We currently do per-frame evaluation — protocol mismatch costs double-digit points. Must align to clip-level protocol.

**Critical requirement**: Fine-tuned Kinetics-400 pretrained video encoder (VideoMAE-v2 or MViTv2-S), NOT frozen per-frame features.

### 2.3 Procedure Step Recognition (PSR)

| Metric | Baseline | Method | Our Target | Difficulty |
|--------|----------|--------|------------|------------|
| F1 (±3-frame) | **0.731** | B2 (ASD-accumulation + procedure-order) | ≥ 0.60 | MOST BEATABLE |
| POS | **0.816** | B2 (ASD-accumulation + procedure-order) | ≥ 0.75 | BEATABLE |
| F1 (±3-frame) | **0.506** | STORM-PSR (dual-stream transformer) | ≥ 0.55 | BEATABLE |
| POS | **0.812** | STORM-PSR | ≥ 0.75 | BEATABLE |

**Key insight**: B2 (F1=0.731) is a HEURISTIC — ASD confidence accumulation + procedure-order constraints. Barely neural. STORM-PSR (fancy dual-stream transformer) gets only F1=0.506. Our model can learn to dominate B2 by predicting TRANSITIONS with monotonic constraints.

**Winning strategy**: Feed detection state-classifier outputs into a causal transformer that predicts per-component transition events, decoded with monotonic constraint + procedure-order prior. This is a learned strict-superset of B2.

### 2.4 Head Pose (9-DoF)

| Metric | Baseline | Our Target | Difficulty |
|--------|----------|------------|------------|
| Forward angular MAE (°) | **No published baseline** | ≤ 25° | FREE WIN |
| Up angular MAE (°) | **No published baseline** | ≤ 25° | FREE WIN |
| Position MAE (mm) | **No published baseline** | ≤ 50mm | FREE WIN |

**Key insight**: No published supervised baseline exists. This is an uncontested table row. Switch from 9 raw numbers with MSE to 6D continuous rotation representation + normalized position with geodesic loss. Expect MAE to drop from 60-70° to 10-25°.

### 2.5 Efficiency

| Metric | Our Advantage | Notes |
|--------|---------------|-------|
| Total params | ~54M (vs 3×54M for 3 specialists) | Weight sharing |
| GFLOPs | Single backbone pass | vs 3 separate forward passes |
| FPS | ~15-20 FPS (estimated) | Real-time capable |

**Key insight**: Efficiency wins by construction with a streaming design (frozen backbone cache for temporal heads). This is the safest contribution.

---

## 3. Honest Recovery Ceiling Estimates

### 3.1 Full Training Data, All Patches Applied, Heads Alive

| Task | Estimated Ceiling | Reasoning |
|------|-------------------|-----------|
| Activity Top-1 | 0.30–0.50 (frame-level) | Single-RGB ConvNeXt-Tiny multi-task with per-frame head; engaging feature bank/VideoMAE properly pushes toward upper end |
| Detection mAP50 | 0.35–0.60 | RetinaNet-style competitive with YOLOv8-m in principle, but anchor calibration + tiny-GT-density costs; 0.84 not realistic without synthetic pretrain |
| PSR F1 | 0.50–0.80 | Given slowly-varying states |
| Head Pose MAE | 10–25° | With geometry-aware parameterization |

### 3.2 On 25% Subset, 3-Epoch Recovery Run

| Metric | Success Bar | Meaning |
|--------|-------------|---------|
| act_top1 | ≥ 0.10 | Head demonstrably alive |
| det_mAP50 | ≥ 0.05–0.10 | Detection learning |
| PSR unique patterns | > 10 | Not constant output |
| pred_seen | ≥ 15/75 classes | Activity exploring |

### 3.3 On 5% Subset (Current)

**Don't bother measuring ceilings** — 12/75 classes present is a structural cap.

---

## 4. Priority-Ordered Targets

### Tier 1: Must Fix (Zero GPU Cost)
1. Fix EMA measurement chain (RC-13)
2. Fix eval collate (RC-17)
3. Fix detection trunk reinit (RC-14)
4. Disable broken Mixup/CutMix (RC-15)
5. Fix ViT attention scaling (RC-16)
6. Sigmoid-bound det_conf (RC-19)
7. Evaluate `latest.pth` (raw weights) at MAX_BATCHES≥200

### Tier 2: Must Redesign (Structural)
1. Two-stage training via frozen-backbone embedding cache
2. ROI-centric detection (class-agnostic localizer + state classifier)
3. PSR transition prediction (events, not per-frame binaries)
4. Clip-level activity training with K400-pretrained video stream
5. Synthetic data pretraining for detection

### Tier 3: The Extra Points
1. Knowledge distillation from dedicated baselines
2. Fix cross-task FiLM conditioning
3. Geometry-aware head pose (6D rotation)
4. Task-aware sampling (per-task dataloaders)

---

## 5. What "Success" Looks Like

### Minimum Viable Paper
- All 5 heads produce non-zero, improving metrics on validation
- At least 2 heads beat or match dedicated baselines
- Efficiency story quantified (params, GFLOPs, FPS vs running N specialists)
- Ablation table showing cross-task conditioning contribution

### Ideal Paper
- PSR F1 > 0.731 (beat B2 heuristic with a learned model)
- Head pose with no baseline → uncontested contribution
- Activity within 5 points of MViTv2 at fraction of compute
- Detection within 5 points of YOLOv8m at fraction of compute
- Efficiency: 3× fewer params, 2× faster inference than 3 specialists

### Stretch Goal
- Beat ALL dedicated baselines on at least 2 tasks
- Demonstrate that multi-task interference is NET POSITIVE (cross-task conditioning helps)
- Show that the unified model discovers task correlations that specialists miss

---

## 6. Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| GPU | RTX 3060, 12 GB VRAM | Single GPU, no distributed training |
| Batch size | 1 (physical) × 32 (accum) | VRAM limit with VideoMAE + ConvNeXt + 5 heads |
| Training time | < 8h per 100 epochs | Practical iteration speed |
| Model size | < 80M params | Must fit in VRAM |
| Framework | PyTorch 2.x | Existing codebase |
| Dataset | IndustReal (real + synthetic) | Primary benchmark |

---

## 7. What We've Already Prepared (Config Tiers)

The config.py already has flags for all Tier 2-3 improvements:

| Flag | Description | Status |
|------|-------------|--------|
| `USE_SIMPLIFIED_LOSS` | CE+label_smoothing, fixed weights | ✅ Enabled |
| `ASSERT_AND_CRASH` | No NaN guards | ✅ Enabled |
| `USE_EMA` | EMA disabled for recovery | ✅ Disabled |
| `USE_ROI_DETECTOR` | ROI-centric detection | ⏳ Configured, not enabled |
| `USE_PSR_TRANSITION` | Transition-based PSR | ⏳ Configured, not enabled |
| `USE_K400_VIDEO_STREAM` | K400 video encoder | ⏳ Configured, not enabled |
| `USE_GEO_HEAD_POSE` | 6D rotation head pose | ⏳ Configured, not enabled |
| `USE_DISTILLATION` | Knowledge distillation | ⏳ Configured, not enabled |
| `USE_TASK_AWARE_SAMPLING` | Per-task dataloaders | ⏳ Configured, not enabled |
| `EMBEDDING_CACHE_DIR` | Frozen backbone cache | ⏳ Configured, not enabled |

These modules exist as Python files in `code/` — they need Opus's guidance on HOW and WHEN to enable them for maximum effect.
