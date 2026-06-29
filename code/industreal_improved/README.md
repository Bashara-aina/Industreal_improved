# POPW — Multi-Task Assembly Action Recognition

Multi-task learning for assembly action recognition. Backbone: ConvNeXt-Tiny + FPN, conditioned by FiLM layers, with Kendall homoscedastic uncertainty for loss weighting.

**Tasks:** (1) Assembly State Detection (ASD, 24-class), (2) hand pose estimation (heatmap regression), (3) activity classification (75-class: 74 AR + 1 NA padding for unlabelable frames), (4) Procedure Step Recognition (PSR, 11-component binary focal).

**Datasets:** IndustReal (74-class assembly actions, primary) and IKEA ASM (33-class action + 7-class assembly state, alternative).

**Paper baselines ( IndustReal — do not fabricate):**

| Method | ASD mAP | Activity Top-1 | PSR F1 | PSR POS | Source |
|-------|---------|---------------|--------|---------|--------|
| YOLOv8m | **83.80%** | — | — | — | Schoonbeek 2024 |
| MViTv2 RGB-only | — | 65.25% | — | — | Schoonbeek 2024 |
| B2 ASD-accum | — | — | 0.731 | 0.816 | Schoonbeek 2024 |
| STORM-PSR | — | — | 0.506 | 0.812 | Schoonbeek 2025 |
| **POPW (ours)** | `\popwres` | `\popwres` | `\popwres` | `\popwres` | This work |

> **Note:** `\popwres` = placeholder awaiting actual POPW training runs. All other values are from the cited papers and must not be modified.
>
> **Planning GUIDEs** (comprehensive reference): `GUIDE_1_THE_REFRAME.md` through `GUIDE_8_THE_PAPER_TEX.md` are in the `analyses/consult_2026_06_10/` directory.
> - GUIDE 1: Strategic reframe — why honest metrics matter
> - GUIDE 2: Training all heads via decoupled A/B/C plan
> - GUIDE 3: Metrics, benchmarks, and target numbers
> - GUIDE 4: Paper framing and contribution
> - GUIDE 5: Day-by-day runbook with exact commands
> - GUIDE 6: 200-point verification checklist
> - GUIDE 7: Audit answers and code verification
> - GUIDE 8: Paper TeX guide for filling placeholders

---

## Project Structure

```
industreal_improved_to_archive/
├── src/
│   ├── config.py                 # All hyperparameters (do not hardcode)
│   ├── models/
│   │   └── model.py              # POPWMultiTaskModel: ConvNeXt + FPN + FiLM + task heads
│   ├── data/
│   │   └── industreal_dataset.py # IndustReal + IKEA ASM loader, augmentations, multi-view
│   ├── training/
│   │   ├── train.py              # Training loop, Kendall weighting, AMP, grad clipping, staged freezing
│   │   ├── losses.py              # KendallLoss, WingLoss, FocalLoss, LDAM-DRW, PSR temporal smoothness
│   │   ├── pretrain_mae.py       # Masked autoencoder pretraining
│   │   └── pretrain_synthetic.py # Synthetic data pretraining
│   └── evaluation/
│       └── evaluate.py           # PCK, Top-1/Top-5, mAP, PSR F1/POS metrics
├── scripts/
│   ├── calibrate_anchors.py     # Anchor calibration for detection head
│   ├── cross_validate.py         # Leave-one-subject-out CV protocol
│   ├── run_multi_seed.py         # Multi-seed averaging (3+ seeds for final results)
│   ├── smoke_test.py             # Sanity-check test suite
│   ├── test_e2e_training.py      # End-to-end training validation
│   ├── diagnostic_visualization.py
│   ├── efficiency_report.py
│   ├── export_onnx.py
│   ├── generate_ablation_table.py
│   └── visualize_*.py             # Various visualization utilities
├── docs/
│   ├── verification/
│   │   ├── POPW_DEEP_VERIFICATION.md       # Component-by-component paper compliance
│   │   ├── POPW_FINAL_PRETRAIN_VERIFICATION.md  # 14/14 smoke tests passed
│   │   ├── POPW_CLAUDE_REVIEW_MASTER_PROMPT.md  # Full review prompt
│   │   ├── ARCHITECTURE_ANALYSIS_ViT_vs_Mamba3.md
│   │   ├── contract-0[1-7]-*.md           # Contract specs (config/dataset/model/losses/train/evaluate/benchmark)
│   │   └── planner-industreal-2026-04-21.md
│   └── planning/
│       └── <misc old planning docs>
├── archive/
│   ├── processed_docs/           # Consolidated review/planning docs from previous iterations
│   └── old_runs/                # Archived experiment runs from previous versions
└── runs/                        # Experiment outputs (checkpoints, logs) — write here during training
```

---

## Dataset Configuration

This codebase supports two datasets:

| Dataset | Classes | Tasks | Primary Use |
|---------|---------|-------|-------------|
| **IndustReal** | 74 activity + 24 ASD + 11 PSR components | All 4 tasks | Primary benchmark (this README's default) |
| **IKEA ASM** | 33 activity + 7 ASD | Activity + ASD (no PSR) | Ablation / comparison |

**Config.py default:** IndustReal paths (`/home/newadmin/swarm-bot/project/popw/working/data/datasets/industreal`). IKEA ASM is an alternative dataset that requires manual configuration overrides (see below).

---

### IndustReal (default)

```
DATA_ROOT = /home/newadmin/swarm-bot/project/popw/working/data/datasets/industreal
```

Expected structure:
```
industreal/
├── recordings/
│   ├── train/<rec_id>/rgb/, AR_labels.csv, OD_labels.json, PSR_labels_raw.csv
│   ├── val/  <rec_id>/...
│   └── test/ <rec_id>/...
└── splits/
    ├── train.csv, val.csv, test.csv
```

---

### IKEA ASM Dataset Setup (alternative)

> **Download:** https://www.idiap.ch/internal/biometrics/deformable_hand_pose  
> Search: "IKEA ASM dataset Ben-Shabat WACV 2021" — download the official release.

**Path configuration** (set in `src/config.py` or via environment):
```python
DATA_ROOT = "/path/to/IKEA_ASM_ROOT"   # ← set this before training
```

**Environment variable (quick switch):**
```bash
export DATASET_MODE="ikea"   # or "industreal" (default)
```

**Expected directory structure:**
```
IKEA_ASM_ROOT/
├── train/
│   ├── video/                  # RGB frames or video files per recording
│   └── annotations/
│       ├── train_labels.csv   # activity + assembly state labels
│       └── hand_joints/       # COCO-format 21-joint JSON files per frame
├── val/
│   ├── video/
│   └── annotations/
│       ├── val_labels.csv
│       └── hand_joints/
└── test/
    ├── video/
    └── annotations/
        ├── test_labels.csv
        └── hand_joints/
```

**CSV format (per split):**
| Column | Description |
|--------|-------------|
| `recording_id` | Recording identifier |
| `frame_id` | Frame number |
| `activity_label` | Integer class ID (0–32) |
| `assembly_state_label` | Integer class ID (0–6) |
| `joints_file` | Path to corresponding hand_joints/*.json |

**Hand joints JSON format (COCO 21 joints):**
```json
{
  "joints": [[x1, y1, visible], [x2, y2, visible], ...],  // 21 joints
  "torso": [cx, cy],                                        // for PCK normalization
  "head": [cx, cy]
}
```

**IKEA ASM configuration overrides** (set in `src/config.py` before training):

```python
# Dataset
DATA_ROOT = "/path/to/IKEA_ASM_ROOT"
SUBSET_RATIO = 1.0  # full dataset

# IKEA ASM task configuration
NUM_CLASSES_ACT = 33   # 33 activity classes
NUM_DET_CLASSES = 7    # 7 assembly state detection classes (not 24)
NUM_PSR_COMPONENTS = 11  # procedure step components (IKEA-specific)

# Disable IndustReal-only tasks (IKEA ASM has no COCO hand GT, no PSR)
TRAIN_HEAD_POSE = False  # No 9-DoF gaze in IKEA ASM
TRAIN_PSR = False        # PSR not defined for IKEA ASM

# Training — 3-task Kendall weighting: act + asd + pose (no head_pose, no psr)
USE_KENDALL = True
TRAIN_DET = True
TRAIN_ACT = True
TRAIN_POSE = True   # body pose available via hand keypoints

# Detection: 7-class assembly state (not 24-class IndustReal ASD)
DETECTION_CLASSES = 7
```

**Quick switch via environment variable:**
```bash
export DATASET_MODE="ikea"   # or "industreal" (default)
```

---

## Training Commands

All training commands are run from the `src/` directory:
```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src
```

### Full Training (IndustReal — all 4 tasks, staged + Kendall)

```bash
python training/train.py \
  --experiment full_multi_task_tma_tbank \
  --epochs 50 \
  --lr 5e-4 \
  --batch-size 4 \
  --grad-accum 8 \
  --use-kendall \
  --tma-cell \
  --temporal-bank \
  --hand-film \
  --use-videomae \
  --use-ema \
  --warmup-epochs 5 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --mixed-precision
```

### All Available Training Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name — outputs go to `runs/<name>/` |
| `--epochs` | int | from config | Total training epochs |
| `--max-epochs` | int | from config | Override C.EPOCHS |
| `--lr` | float | from config | Initial learning rate |
| `--batch-size` | int | from config | Batch size per GPU |
| `--grad-accum` | int | from config | Gradient accumulation steps (effective batch = batch × accum) |
| `--use-kendall` | flag | enabled | Enable Kendall homoscedastic uncertainty weighting |
| `--tma-cell` | flag | disabled | Enable Temporal Memory Aggregation cell |
| `--temporal-bank` | flag | disabled | Enable temporal bank for sequence modeling |
| `--hand-film` | flag | disabled | Enable FiLM conditioning for hand pose head |
| `--use-videomae` | flag | disabled | Use VideoMAE frozen backbone for activity feature enhancement |
| `--use-ema` | flag | disabled | Exponential Moving Average of weights |
| `--warmup-epochs` | int | 5 | Linear warmup epochs |
| `--weight-decay` | float | 1e-4 | AdamW weight decay |
| `--grad-clip` | float | 1.0 | Gradient clipping threshold (0 = disabled) |
| `--mixed-precision` | flag | enabled | AMP mixed precision (required for RTX 3060) |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--seed` / `-s` | int | from config | Random seed (use 42, 123, 7 for multi-seed runs) |
| `--subset-ratio` | float | 1.0 | Fraction of recordings to use (0.1 = 10%% for smoke test) |
| `--num-workers` | int | from config | DataLoader workers (0 = single-threaded) |
| `--debug` | flag | disabled | Debug mode: small dataset, fast validation |
| `--preset` | str | None | Preset name for config override (backwards compatibility) |

**Effective batch size:** `batch_size × grad_accum`  
Example: `--batch-size 4 --grad-accum 8` → effective batch = 32 (RTX 3060 optimal)

### Resume from Checkpoint

```bash
python training/train.py \
  --resume ../runs/full_multi_task_tma_tbank/checkpoints/latest.pth \
  --epochs 50
```

### Pretraining (optional, run before full training)

```bash
# Stage 1: Synthetic data pretraining (detection warmup)
python training/pretrain_synthetic.py --epochs 20

# Stage 2: VideoMAE pretraining (activity feature enhancement)
python training/pretrain_mae.py --epochs 20 --backbone vit --mask 0.5
```

### Multi-Seed Training (required for final results)

Run the same command with different seeds (3+ seeds for final reporting):

```bash
# Seed 0
python training/train.py --experiment popw_seed0 --seed 0 --epochs 50 ...
# Seed 1
python training/train.py --experiment popw_seed1 --seed 1 --epochs 50 ...
# Seed 2
python training/train.py --experiment popw_seed2 --seed 2 --epochs 50 ...
```

Or use the multi-seed script:

```bash
python scripts/run_multi_seed.py --seeds 0 1 2 --epochs 50 --lr 5e-4
```

**AMP (mixed precision):** Enabled by default — required for RTX 3060 to avoid VRAM overflow at batch=4.  
**Staged training:** Detection-only (epochs 1-5) → +Pose/HeadPose (6-15) → Full multi-task (16+).  
**Kendall weighting:** Active in full multi-task stage (3-task by default: det, act, psr — head_pose disabled via TRAIN_HEAD_POSE=False. Set TRAIN_HEAD_POSE=True to enable 4-task mode).

---

## Evaluation Commands

All evaluation commands are run from the `src/` directory:
```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src
```

### Single-Checkpoint Evaluation

```bash
# Evaluate on IndustReal test set (all 4 tasks)
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --save-dir ../runs/full_multi_task_tma_tbank/eval_outputs

# Evaluate on IKEA ASM (after switching dataset config)
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --dataset ikea \
  --split test

# Limit to first N batches (debug / smoke test)
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --max-batches 15
```

### Test-Time Augmentation (TTA)

```bash
# Horizontal flip TTA
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --flip-tta

# 5-crop TTA (4 corners + center, 224×224)
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --crop-tta

# Combined TTA (flip + crop)
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --flip-tta --crop-tta
```

### Multi-Seed Evaluation (required for final results)

```bash
# Evaluate across 3 seeds — computes mean ± std for each metric
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --seeds 42,2024,777

# Evaluate across 5 seeds (more robust statistics)
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --seeds 0,1,2,3,4

# Multi-seed with TTA
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --seeds 42,2024,777 \
  --flip-tta
```

### Cross-Validation (Leave-One-Subject-Out)

```bash
python scripts/cross_validate.py
```

### Ablation Evaluation

```bash
# Run ablation table: baseline vs each improvement component
python evaluation/evaluate.py \
  --checkpoint ../runs/full_multi_task_tma_tbank/checkpoints/best.pth \
  --split test \
  --ablation
```

### All Available Evaluation Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | str | required | Path to model checkpoint (.pth) |
| `--split` | str | test | Dataset split: `train`, `val`, `test` |
| `--dataset` | str | industreal | Dataset mode: `industreal` or `ikea` |
| `--save-dir` | str | from config | Output directory for eval results JSON |
| `--max-batches` | int | 9999 | Maximum batches to evaluate |
| `--seeds` | str | 42,2024,777 | Comma-separated seed list for multi-seed eval |
| `--flip-tta` | flag | disabled | Horizontal flip TTA |
| `--crop-tta` | flag | disabled | 5-crop TTA (4 corners + center) |
| `--ablation` | flag | disabled | Run ablation table |
| `--profile-efficiency-only` | flag | disabled | Only profile efficiency (params, GFLOPs, FPS) |

### Metrics by Task

| Task | Metric | Paper Target (IndustReal) | IKEA ASM Target |
|------|--------|--------------------------|-----------------|
| Assembly State Detection | ASD mAP@0.5 | 83.80% (YOLOv8m baseline) | — |
| Activity Recognition | Top-1 Accuracy | 65.25% (MViTv2 baseline) | 64.15% (I3D+pose) |
| Procedure Step Recognition | PSR F1 (±3-frame) | 0.731 (B2 baseline) | N/A |
| Pose Estimation | PCK@0.2 | — | 88.0% (IKEA ASM paper) |

### End-to-End Training Validation

```bash
# Smoke-test: 2-step training loop (fast, no checkpoint download)
python scripts/smoke_test.py

# End-to-end training sanity (2 epochs, small subset)
python scripts/test_e2e_training.py
```

> **Important:** Never report single-seed results as final. Always run with 3+ seeds and report mean ± std across seeds.

---

## Benchmark Results

> **Fill in actual POPW numbers after training runs complete.** The table structure below is ready — do not fabricate values. Leave `\popwres` placeholders until real evaluation runs are done.

### POPW Results (IndustReal)

| Method | ASD mAP | Activity Top-1 | PSR F1 | PSR POS | Params | Source |
|-------|---------|---------------|--------|---------|--------|--------|
| YOLOv8m | **83.80%** | — | — | — | — | Schoonbeek 2024 |
| MViTv2 RGB-only | — | 65.25% | — | — | — | Schoonbeek 2024 |
| B2 ASD-accum | — | — | 0.731 | 0.816 | — | Schoonbeek 2024 |
| STORM-PSR | — | — | 0.506 | 0.812 | — | Schoonbeek 2025 |
| **POPW (ours)** | `\popwres` | `\popwres` | `\popwres` | `\popwres` | `\popwres` | This work |

### POPW Results (IKEA ASM)

| Method | Activity Top-1 | Pose PCK@0.2 | Params | Source |
|--------|---------------|-------------|---------|--------|
| I3D combined | 63.09% | — | — | Ben-Shabat 2021 |
| I3D combined+pose | 64.15% | — | — | Ben-Shabat 2021 |
| I3D temporal loc | — | 20.00 mAP@0.5 | — | Ben-Shabat 2021 |
| PC3D (all views) | **80.2%** | — | — | Aganian 2023 |
| PTMA | 86.99% mcAP | — | 12.9M | Xie 2025 |
| **POPW (ours)** | `\popwres` | `\popwres` | `\popwres` | This work |

### Ablation Table (IndustReal)

| Component | ASD mAP | Activity Top-1 | PSR F1 | Delta |
|-----------|---------|---------------|--------|-------|
| Baseline | — | — | — | — |
| + RandAugment | — | — | — | — |
| + CutMix | — | — | — | — |
| + LDAM-DRW | — | — | — | — |
| + GIoU | — | — | — | — |
| + Focal PSR | — | — | — | — |
| + TMA cell | — | — | — | — |
| + Temporal bank | — | — | — | — |
| + VideoMAE | — | — | — | — |
| **Full model** | — | — | — | — |

### Efficiency Comparison

| Model | Params (M) | GFLOPs | FPS | Resolution |
|-------|-----------|--------|-----|------------|
| ActionFormer (IKEA) | 27.70 | 83.28 | ~21 | — |
| PTMA (IKEA) | 12.9 | 1.96 | 291 | — |
| MiniROAD (IKEA) | 10.5 | 1.08 | 325 | — |
| **POPW (ours)** | `\popwres` | `\popwres` | `\popwres` | — |

> **Note:** All baseline values are from the cited papers and must not be modified. `\popwres` = placeholder awaiting actual POPW training runs.

---

## Architecture Diagram

The architecture diagram showing the full multi-task pipeline (ConvNeXt-Tiny → FPN → FiLM conditioning → task heads) is available in:

- `docs/verification/` — architecture compliance verification documents
- Will be generated from `src/models/model.py` via `scripts/export_onnx.py --visualize`

### Key Architecture Constraints

- **Do NOT add batch norm after FiLM layers** — this destroys the conditional modulation
- **ConvNeXt-Tiny** required: C2=96, C3=192, C4=384, C5=768
- **FPN** P3-P7 at 256 channels each
- **Kendall uncertainty** weighting across all active tasks

---

## Known Limitations

### Critical (block training if unfixed)

- **`evaluate.py.bak` must not be used.** The archived `evaluate.py.bak` contains a PCK normalization bug. Always use `src/evaluation/evaluate.py` (current version).
- **Never report single-seed results as final.** Always run `scripts/run_multi_seed.py` with 3+ seeds and report mean ± std.
- **`smoke_test.py` may timeout (~120s).** The VideoMAE checkpoint download is the likely cause. For fast iteration, run `scripts/test_e2e_training.py` instead.
- **Pose loss scale discrepancy.** `losses.py` uses `× 0.01` for body pose and head pose losses, but the paper specifies `× 0.001` (a 10× difference). This affects Kendall uncertainty initialization. See C4/C18 in `docs/verification/POPW_20C_TRAINING_READINESS_AUDIT_FINAL.md`.

### Training

- **Validation loss ≠ evaluation metrics.** Validation loss during training can diverge from `evaluate.py` results due to augmentation mismatch. Always trust `evaluate.py` over training validation loss.
- **VideoMAE adds significant overhead.** `USE_VIDEOMAE=True` adds +22M frozen params and ~600MB VRAM. If the checkpoint fails to load, a weaker 3D-conv fallback encoder is used silently.
- **RTX 3060 VRAM.** Mixed precision (AMP) is required — without it, VRAM overflows at `batch_size=4`.
- **IKEA ASM resolution.** The IKEA ASM paper specifies 640×480, but this has not been independently verified in code. Confirm resolution before comparing against IKEA ASM baselines.
- **PSR sequence mode.** `USE_PSR_SEQUENCE_MODE=True` (enabled by default) draws one sequence batch every 10 normal batches. For pure frame-level training, disable this.

### Architecture Constraints

- **Do NOT add batch norm after FiLM layers** — this destroys the conditional modulation.
- **ConvNeXt-Tiny required** — channel counts: C2=96, C3=192, C4=384, C5=768.
- **Shared `log_var_pose`** — body pose and head pose share one Kendall log variance. The Kendall mechanism cannot independently downweight head pose vs body pose. This is an implicit design choice not documented in the paper.
- **Head pose GT absent in IndustReal.** `TRAIN_HEAD_POSE=False` is the default because IndustReal provides 9-DoF gaze (not COCO hand keypoints). This reduces Kendall weighting to 3 active tasks (det, act, psr) instead of 4.

### Dataset Configuration

- **Switching from IndustReal to IKEA ASM requires manual overrides.** Set `NUM_CLASSES_ACT=33`, `DETECTION_CLASSES=7`, `TRAIN_PSR=False`, and `TRAIN_HEAD_POSE=False` in `src/config.py` before training.
- **IKEA ASM does not define PSR.** Ensure `TRAIN_PSR=False` when training on IKEA ASM.
- **Random temporal stride.** `RANDOM_TEMPORAL_STRIDE=True` (default) randomly samples frame stride {2,3,4,5} per clip as augmentation. For reproducible benchmark runs, set `RANDOM_TEMPORAL_STRIDE=False` and fix `TRAIN_FRAME_STRIDE=3`.

### Checkpointing

- **No periodic interval saves.** Checkpoints are saved at epoch end only (latest.pth). No sub-epoch saves by default.
- **`strict=False` on load.** `train.py` uses `strict=False` when loading checkpoints, which silently swallows key mismatches.
- **Checkpoint directories may be empty.** The archived runs in `archive/` have empty checkpoint directories — no saved .pt files were found.

### Reproducibility

- **CUDA non-determinism.** `CUDNN_DETERMINISTIC=False` by default (for performance). True determinism requires `CUDNN_DETERMINISTIC=True` which slows training.
- **DataLoader shuffle.** Not explicitly flagged as seed-controlled — some variance may exist between runs with the same seed.
- **No multi-GPU (DDP).** Training is single-GPU only. Multi-node training would require DistributedDataParallel implementation.

---

## Config Paths

All paths are relative to `src/config.py`:
```python
DATA_ROOT     → IKEA ASM dataset root
OUTPUT_ROOT   → runs/<experiment_name>/
CHECKPOINT_DIR → OUTPUT_ROOT / 'checkpoints'
LOG_DIR       → OUTPUT_ROOT / 'logs'
EVAL_SAVE_DIR → OUTPUT_ROOT / 'eval_outputs'
```
