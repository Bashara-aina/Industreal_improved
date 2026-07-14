# POPW — Multi-Task Assembly Perception on IndustReal

Multi-task learning for assembly action recognition on the IndustReal dataset (WACV 2024). Backbone: ConvNeXt-Tiny + FPN, conditioned by FiLM layers, with Kendall homoscedastic uncertainty weighting and PCGrad gradient surgery for loss balancing.

**Tasks:**
1. Assembly State Detection (ASD, 24-class bounding boxes)
2. Activity Recognition (AR, 74-class with verb-noun grouping)
3. Procedure Step Recognition (PSR, 11-component multi-label binary focal)
4. Head Pose Estimation (9-DoF from HoloLens 2 sensor)

**Hardware target:** Single GPU with ~12 GB VRAM (e.g., RTX 3060 / RTX 4070). BF16/FP16 mixed precision recommended.

---

## Quick Start

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (tested on RTX 3060 12 GB)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For bit-exact reproducibility, use the frozen versions:

```bash
pip install -r requirements_frozen.txt
```

### Dataset

Download the IndustReal dataset from the official source (Schoonbeek et al., WACV 2024). Arrange the data as follows:

```
<POPW_ROOT>/
├── recordings/
│   ├── train/<rec_id>/
│   │   ├── rgb/                  # RGB frames
│   │   ├── AR_labels.csv         # Activity labels
│   │   ├── OD_labels.json        # Object detection / assembly state labels
│   │   ├── PSR_labels_raw.csv    # Procedure step labels
│   │   └── pose.csv              # Head pose (HoloLens 2)
│   ├── val/<rec_id>/...
│   └── test/<rec_id>/...
├── train.csv                     # Train split metadata
├── val.csv                       # Validation split metadata
└── test.csv                      # Test split metadata
```

Set `POPW_ROOT` in `src/config.py` (line 236) to point to the dataset root, or use the `POPW_ROOT` environment variable:

```bash
export POPW_ROOT=/path/to/industreal
```

---

## Training Commands

All training commands run from the project root directory.

### Preset-Based Training

The recommended way to train is via `--preset`, which loads a full configuration from `config.PRESETS`:

```bash
python src/training/train.py --preset benchmark_full --max-epochs 50 --batch-size 2
```

Available presets include:

| Preset | Description |
|--------|-------------|
| `benchmark_full` | Full multi-task (4 heads), BF16 mixed precision, grad accum 16 |
| `paper_run` | Paper configuration: FP32 (AMP off), batch 2, grad accum 16 |
| `stage_rf4` | Run-specific preset with balanced training schedule |
| `ablation_det_only` | Detection-only baseline |
| `ablation_act_only` | Activity-only baseline |
| `ablation_psr_only` | PSR-only baseline with sequence batches |
| `ablation_pose_only` | Head-pose-only baseline |
| `ablation_kendall_fixed` | Multi-task with fixed loss weights (no Kendall) |
| `ablation_grouping_none` | Multi-task with raw 75-class activity (no verb grouping) |

### Single-Task Baselines

Four convenience scripts patch the configuration to isolate one head at a time, then delegate to `train.py`:

```bash
python src/training/train_singletask_pose.py --seed 103 --max-epochs 50 --batch-size 2
python src/training/train_singletask_detection.py --seed 103 --max-epochs 50 --batch-size 2
python src/training/train_singletask_activity.py --seed 103 --max-epochs 50 --batch-size 2
python src/training/train_singletask_psr.py --seed 103 --max-epochs 50 --batch-size 2
```

All four accept the same CLI arguments as `train.py` (see below).

### Multi-Seed Training

Final results require 3+ seeds with mean +- std reporting:

```bash
# Seed 42
python src/training/train.py --preset benchmark_full --seed 42 --max-epochs 50 --batch-size 2
# Seed 123
python src/training/train.py --preset benchmark_full --seed 123 --max-epochs 50 --batch-size 2
# Seed 7
python src/training/train.py --preset benchmark_full --seed 7 --max-epochs 50 --batch-size 2
```

### Full Reproduction

The `scripts/reproduce.sh` orchestrator runs the entire pipeline:

```bash
bash scripts/reproduce.sh                          # full run
bash scripts/reproduce.sh --dry-run                # preview only
bash scripts/reproduce.sh --data /path/to/industreal
```

It executes: environment check > single-task baselines (4 heads x 5 seeds) > multi-task learning (3 seeds) > ablation suite (6 ablations) > evaluation > metrics aggregation placeholder.

### Ablation Suite

Run individual ablations via:

```bash
bash scripts/run_ablation_suite.sh det
bash scripts/run_ablation_suite.sh act
bash scripts/run_ablation_suite.sh psr
bash scripts/run_ablation_suite.sh pose
bash scripts/run_ablation_suite.sh kendall-fixed
bash scripts/run_ablation_suite.sh grouping-none
```

Set `ABLATION_EPOCHS=25` to override the default 25 epochs (sufficient for ablation convergence).

### All Training Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--preset` | str | None | Config preset name from `config.PRESETS` |
| `--max-epochs` | int | from config | Override total training epochs |
| `--batch-size` | int | from config | Batch size per GPU |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--debug` | flag | disabled | Debug mode: small dataset, fast validation |
| `--seed` / `-s` | int | from config | Random seed |
| `--subset-ratio` | float | 1.0 | Fraction of recordings to use |
| `--num-workers` | int | from config | DataLoader workers |
| `--no-staged-training` | flag | disabled | Disable staged progressive training |
| `--start-epoch` | int | from config | Override starting epoch |
| `--reset-scheduler` | flag | disabled | Reset scheduler state after loading checkpoint |
| `--reinit-heads` | flag | disabled | Reinitialize detection/activity/PSR heads + FPN |
| `--detach-reg-fpn` | flag | disabled | Detach regression FPN from computation graph |
| `--detach-psr-fpn` | flag | disabled | Detach PSR FPN from computation graph |

**Effective batch size:** `batch_size x grad_accum_steps` (configured via preset). With `batch_size=2` and `grad_accum_steps=16`, effective batch = 32.

---

## Evaluation Commands

Evaluate a trained checkpoint on the test set:

```bash
python src/evaluation/evaluate.py \
    --checkpoint src/runs/mtl_seed42/checkpoints/best.pth \
    --split test \
    --save-dir src/runs/mtl_seed42/eval_outputs
```

### Evaluation Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | str | required | Path to model checkpoint |
| `--split` | str | val | Dataset split: train, val, test |
| `--save-dir` | str | None | Output directory for evaluation results |
| `--max-batches` | int | 9999 | Maximum batches to evaluate |
| `--profile-efficiency-only` | flag | disabled | Profile params, GFLOPs, FPS only |
| `--seeds` | str | 42,2024,777 | Comma-separated seeds for multi-seed eval |
| `--ablation` | flag | disabled | Run ablation table |
| `--flip-tta` | flag | disabled | Horizontal flip test-time augmentation |
| `--crop-tta` | flag | disabled | 5-crop test-time augmentation |

### Metrics by Task

| Task | Metric | Description |
|------|--------|-------------|
| Assembly State Detection (ASD) | mAP@0.5 | Mean average precision at IoU 0.5 |
| Activity Recognition | Top-1 Accuracy | Top-1 classification accuracy (74 classes) |
| Procedure Step Recognition | F1@+-3 | Event-based F1 with +-3 frame tolerance |
| Procedure Step Recognition | POS | Precision of Steps (PSR-specific) |
| Head Pose Estimation | Forward MAE | Angular error (degrees) for forward gaze vector |
| Head Pose Estimation | Up MAE | Angular error (degrees) for up vector |
| Procedure Step Recognition | Tau | Temporal tolerance tau metric |

---

## Project Structure

```
industreal_improved/
├── src/
│   ├── config.py                 # All hyperparameters and presets
│   ├── models/
│   │   └── model.py              # POPWMultiTaskModel: ConvNeXt + FPN + FiLM + task heads
│   ├── data/
│   │   └── industreal_dataset.py # IndustReal dataset loader, augmentations
│   ├── training/
│   │   ├── train.py              # Training loop, Kendall weighting, AMP, gradient clipping
│   │   ├── train_singletask_pose.py       # Pose-only training script
│   │   ├── train_singletask_detection.py  # Detection-only training script
│   │   ├── train_singletask_activity.py   # Activity-only training script
│   │   ├── train_singletask_psr.py        # PSR-only training script
│   │   └── losses.py             # KendallLoss, WingLoss, FocalLoss
│   └── evaluation/
│       └── evaluate.py           # Evaluation metrics (mAP, Top-1, F1, pose MAE)
├── scripts/
│   ├── reproduce.sh              # Full reproducibility orchestrator
│   ├── run_ablation_suite.sh     # Ablation suite launcher
│   ├── eval_test_split.py        # Test-split evaluation orchestrator
│   └── ... (analysis and utility scripts)
├── config/
├── tests/
├── requirements.txt              # Package dependencies
├── requirements_frozen.txt       # Exact version pins for reproducibility
├── LICENSE                       # Apache 2.0
└── README.md
```

---

## Architecture

The model uses a **ConvNeXt-Tiny** backbone (C2=96, C3=192, C4=384, C5=768) with a **Feature Pyramid Network (FPN)** producing multi-scale features P3-P7 at 256 channels. Task-specific heads share the FPN features via **FiLM (Feature-wise Linear Modulation)** conditioning layers.

Training uses **Kendall homoscedastic uncertainty weighting** to dynamically balance task losses, with **PCGrad** gradient surgery to mitigate conflicting gradients.

### Key Architecture Constraints

- **Do NOT add batch norm after FiLM layers** -- this destroys the conditional modulation.
- **ConvNeXt-Tiny** required: C2=96, C3=192, C4=384, C5=768.
- **FPN** outputs P3-P7 at 256 channels each.
- **Kendall uncertainty** weights all active tasks at every step.

---

## Configuration

All hyperparameters live in `src/config.py`. Key paths:

```python
POPW_ROOT       = Path("/path/to/industreal")        # Dataset root
RECORDINGS_ROOT = POPW_ROOT / "recordings"            # Train/val/test splits
OUTPUT_ROOT     = Path("src/runs")                    # Experiment output
```

Override at runtime via environment variables:

```bash
export POPW_ROOT=/custom/dataset/path
export OUTPUT_ROOT_OVERRIDE=/custom/output/path
```

---

## Known Limitations

- **CUDA non-determinism.** `CUDNN_DETERMINISTIC=False` by default (for performance). True determinism requires `CUDNN_DETERMINISTIC=True` which slows training.
- **Validation loss vs. evaluation metrics.** Validation loss during training can diverge from `evaluate.py` results due to augmentation mismatch. Always trust `evaluate.py` over training validation loss.
- **No multi-GPU (DDP).** Training is single-GPU only. Distributed training would require DistributedDataParallel implementation.
- **Head pose GT absent for IndustReal.** `TRAIN_HEAD_POSE=False` is the default for the public benchmark; head pose annotations are HoloLens 2 sensor data specific to the recording setup.
- **Resume with `strict=False`.** `train.py` uses `strict=False` when loading checkpoints, which silently swallows key mismatches.

---

## Citation

If you use this code or the IndustReal dataset, please cite the original dataset paper:

```bibtex
@inproceedings{schoonbeek2024industreal,
  title={IndustReal: A Dataset for Procedure Step Recognition
         and Assembly State Detection in Industrial Assembly Tasks},
  author={Schoonbeek, Tim and others},
  booktitle={WACV},
  year={2024}
}
```

And this implementation:

```bibtex
@misc{popw2026,
  title={POPW: Multi-Task Assembly Perception with
         ConvNeXt-Tiny and Kendall Uncertainty Weighting},
  author={Bashara and the POPW contributors},
  year={2026},
  howpublished={\url{https://github.com/Bashara-aina/industreal-improved}}
}
```

---

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

```
Copyright 2026 POPW Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
