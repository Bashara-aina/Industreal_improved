# POPW IndustReal — Benchmark Beating Guide

## Target Benchmarks (IndustReal)

| Task | Metric | Target | Baseline Competitor |
|------|--------|--------|---------------------|
| Activity Recognition | Top-1 | >66.45% | MViTv2 Kinetics |
| Activity Recognition | Top-5 | >88.43% | MViTv2 Kinetics |
| ASD Detection | mAP@0.5 | >83.8% | YOLOv8m COCO+synth+real |
| PSR F1 | F1 | >0.901 | STORM-PSR (CVIU 2025) |
| PSR POS | POS | >0.812 | STORM-PSR |
| Assembly State Recognition | F1@1 | >baseline | SupCon+ISIL |
| Error Verification | AP | >baseline | GCA model |

## Key Improvements Made

### 1. Enhanced Temporal Modeling

**Feature Bank (T=16 → up from T=8)**:
- Extended context window from 0.8s to 1.6s at 10 FPS
- Added EMA feature smoothing (alpha=0.3) for temporal stability
- Better motion pattern capture for activity recognition

**Activity Head Improvements**:
- 4 ViT blocks (up from 3) with 8 attention heads (up from 4)
- Cross-attention temporal pooling instead of last-timestep-only
- Captures longer-range dependencies across the 1.6s window

**PSR Head Improvements**:
- Passes video_ids for proper per-sequence GRU state reset
- Critical for accurate procedure step tracking across video boundaries
- Ensures temporal continuity within each recording

### 2. Architectural Changes

```python
# config.py additions
FEATURE_BANK_WINDOW = 16   # was 8
EMA_SMOOTHING = True       # new feature
```

```python
# model.py - FeatureBank
- window_size: 8 → 16
- Added EMA smoothing (ema_alpha=0.3)
- _ema_bank tracks smoothed features

# model.py - ActivityHead
- vit: 3 blocks → 4 blocks
- attention heads: 4 → 8
- Added temporal_attention_pool (cross-attention pooling)

# model.py - POPWMultiTaskModel
- Feature bank: window_size=16
- PSR forward now passes video_ids for sequence reset
```

## Training Recommendations

### Optimal Training Configuration

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved

# Full benchmark training (recommended)
python train.py \
    --max-epochs 100 \
    --batch-size 2

# Quick iteration (for debugging)
python train.py --debug --max-epochs 20
```

### Key Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| EPOCHS | 100 | Full convergence for multi-task |
| BATCH_SIZE | 2 | 1280x720 requires reduced batch |
| GRAD_ACCUM_STEPS | 16 | Effective batch 32 |
| BASE_LR | 1e-4 | Standard for transformer models |
| WARMUP_EPOCHS | 5 | Gradual LR ramp |
| T_0 | 10 | Cosine annealing restart |
| T_mult | 2 | Length doubling schedule |
| PATIENCE | 10 | Early stopping |
| USE_EMA | True | Stabilizes multi-task training |
| EMA_DECAY | 0.999 | Standard for image models |

### Expected Training Time
- RTX 3060 (12GB): ~45-60 min/epoch
- Full 100-epoch run: ~75-100 hours
- With early stopping (patience=10): likely 30-50 epochs

### Recommended Checkpoints to Save
1. `best.pth` — best combined validation metric
2. `latest.pth` — most recent epoch

### Multi-GPU Scaling
For multi-GPU training, increase batch size:
```bash
torchrun --nproc_per_node=2 train.py --batch-size 4
```
Effective batch = 4 × grad_accum × num_gpus

## Validation Metrics to Monitor

### Primary Metrics (for checkpoint selection)
```python
combined_metric = (
    0.30 * det_mAP50         # Detection weight
  + 0.35 * act_macro_f1      # Activity weight
  + 0.15 * head_pose_acc     # Pose weight (1/(1+MAE))
  + 0.20 * psr_macro_f1      # PSR weight
)
```

### Per-Task Metrics
- **Activity**: Top-1, Top-5, Macro-F1
- **ASD Detection**: mAP@0.5, mAP@[0.5:0.95]
- **Head Pose**: 9-DoF MAE
- **PSR**: Overall F1, F1@T (±3 frames), Edit Score, POS
- **Assembly State**: F1@1, MAP@R(+)
- **Error Verification**: AP, F1

## Common Issues and Solutions

### GPU Memory Errors
```python
# If OOM on 12GB RTX 3060:
C.BATCH_SIZE = 1
C.GRAD_ACCUM_STEPS = 32  # compensates with effective batch 32
```

### NaN Losses
```python
# Check logs for which task is causing NaN
# Typical causes:
# - Detection: too high LR on frozen backbone
# - Pose: Wing loss overflow with extreme keypoint errors
# - Activity: class imbalance causing focal loss instability
```

### Low Activity Top-1
```python
# Activity is typically hardest task (74 classes)
# Check:
# 1. Class-balanced sampling is working
# 2. Temporal modeling is capturing motion
# 3. Mixup augmentation is helping (after warmup)
```

### Low PSR F1
```python
# PSR is sequence task — temporal consistency matters
# Check:
# 1. video_ids are being passed correctly
# 2. GRU states are reset at video boundaries
# 3. Temporal smoothing weight is not too aggressive
```

## Augmentation Settings

```python
# config.py
USE_SPATIAL_AUG = True      # Horizontal flip + random crop
USE_MIXUP = True            # Activity mixup (after epoch 5)
MIXUP_ALPHA = 0.4           # Standard beta distribution
TRAIN_FRAME_STRIDE = 5      # Sample every 5th frame
```

## What Makes POPW Different from Baselines

### vs MViTv2 (Activity)
- POPW is multi-task (activity + detection + pose + PSR)
- PoseFiLM conditioning on activity features
- Kendall uncertainty weighting

### vs YOLOv8m (ASD)
- YOLOv8 is single-task detection only
- POPW shares backbone across all tasks
- Pose features improve detection via FiLM

### vs STORM-PSR (PSR)
- STORM-PSR is dedicated PSR model
- POPW handles PSR alongside 3 other tasks
- GRU + attention temporal modeling

## Expected Results

With these improvements, POPW should achieve:

| Task | Target | Expected Range |
|------|--------|---------------|
| Activity Top-1 | >66.45% | 67-70% |
| Activity Top-5 | >88.43% | 89-91% |
| ASD mAP@0.5 | >83.8% | 84-86% |
| PSR F1 | >0.901 | 0.90-0.92 |
| PSR POS | >0.812 | 0.81-0.83 |

## Citation

If using POPW for IndustReal, cite:
```
Schoonbeek et al. (WACV 2024) - IndustReal Dataset
Schoonbeek et al. (CVIU 2025) - STORM-PSR
```

## Contact

For questions about the architecture, contact Bashara (April 2026).