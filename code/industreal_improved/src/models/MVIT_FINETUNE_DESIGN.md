# MViTv2-S Fine-Tuning Design (Opus 144)

## Motivation

The frozen MViTv2-S linear probe achieved 0.3810 Top-1 accuracy (69 classes),
which is 16.4 points above the 0.30 threshold that justifies a full fine-tuning
investment. The ConvNeXt-Tiny baseline (0.2169) and its majority-class oracle
(0.2217) are both far below this score, confirming MViTv2-S Kinetics-400
pretraining provides a strong feature foundation for IndustReal activity
recognition.

The goal is to close the gap to MViTv2-S SOTA (0.622 Top-1 on Kinetics-400)
by end-to-end fine-tuning on the IndustReal domain.

## Architecture

```
Input clip [B, 3, T=16, 224, 224]  (Kinetics-400 normalized)
    |
MViTv2-S Backbone (torchvision, Kinetics-400 pretrained)
    - 34.5M total params, 768-dim clip embedding
    - 16 MultiscaleBlocks, conv_proj, LayerNorm
    - Classification head removed (nn.Identity)
    - Gradient checkpointing enabled during training
    |
MLP Activity Head
    - Linear(768, 512) -> ReLU -> Dropout(0.15) -> Linear(512, N)
    - N = 69 (remapped groups from 75 raw classes)
    |
Logits [B, N]
```

## Training Strategy

### Two-Stage Fine-Tuning

| Stage | Epochs | Backbone | Head | LR (backbone) | LR (head) |
|-------|--------|----------|------|---------------|-----------|
| 1     | 1-3    | Frozen   | Train | N/A           | 5e-5      |
| 2     | 4-20   | Unfrozen | Train | 1e-5          | 5e-5      |

Stage 1 (warmup) trains only the randomly-initialized MLP head so it learns
meaningful feature combinations before the backbone is unfrozen. This prevents
the backbone gradients from being dominated by random head noise in the first
epochs.

Stage 2 unfreezes the last 4 transformer blocks (blocks 12-15, the highest-level
semantic layers) at a reduced learning rate (1e-5 vs 5e-5 for the head). The
early blocks remain frozen to preserve Kinetics-400 spatial-temporal features
that are universal (edge, motion, texture detectors).

### 75-to-69 Class Mapping

The IndustReal dataset emits 75 raw action IDs (0: NA, 1-74: actions). IDs 37
and 64 are permanently absent, and additional classes with fewer than 100 frames
are verb-grouped (by first underscore token). The resulting `act_remap_75_to_69.json`
maps all 75 raw IDs to 69 output groups. The dataset applies this remap at load
time, so the model's cross-entropy loss is computed directly against the 69
grouped labels.

## Parameter Count

| Component      | Total Params | Trainable (Stage 1) | Trainable (Stage 2) |
|----------------|--------------|---------------------|---------------------|
| Backbone       | 34.5M        | 0 (frozen)          | ~2.5M (4 blocks)    |
| MLP Head       | 0.4M         | 0.4M                | 0.4M                |
| **Total**      | 34.9M        | 0.4M                | 2.9M                |

19.3M of the backbone's 34.5M parameters are kept trainable-capable (the
`VideoBackboneWrapper` design), but this fine-tuning only enables the last 4
blocks (~2.5M) plus the head (0.4M) for a total of ~2.9M trainable params.

## Memory Budget

| Component                 | Memory (FP16) |
|---------------------------|---------------|
| Backbone forward (T=16)   | ~2.8 GB       |
| MLP head + optimizer      | ~0.2 GB       |
| Backbone activations (w/ checkpointing) | ~0.5 GB |
| **Total (estimated)**     | ~3.5 GB       |
| GPU available (RTX 3060)  | 12.0 GB       |

With gradient checkpointing, peak memory is ~3.5 GB, well within the RTX 3060
12 GB budget. The `expandable_segments:True` CUDA alloc config prevents
fragmentation OOM.

### Without Gradient Checkpointing

Full activation storage for 16 transformer blocks at batch=2 is approximately
7.5 GB, which still fits but leaves less headroom for DataLoader workers, CUDA
context, and memory fragmentation. Gradient checkpointing is recommended.

## Implementation Files

- `scripts/train_mvit_finetune.sh` -- Launch script
- `src/training/train_video_finetune.py` -- Training module (dataset, model, loop)
- `src/models/MVIT_FINETUNE_DESIGN.md` -- This design document

## Expected Outcome

Based on the 0.3810 frozen probe result, end-to-end fine-tuning is expected to
reach 0.45-0.55 Top-1 on the 69-class validation set after 20 epochs. The gap
to Kinetics-400 SOTA (0.622) is largely attributable to the smaller IndustReal
dataset (~100K frames vs 650K Kinetics-400 clips) and the narrower action set.
