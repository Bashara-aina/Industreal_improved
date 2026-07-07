# Video Backbone Integration Design

## 1. Problem Statement

The current POPWMultiTaskModel uses ConvNeXt-Tiny (ImageNet-pretrained) as its
shared backbone. While strong for per-frame spatial features, ConvNeXt-Tiny has
no temporal modeling -- each frame is processed independently. Activity
recognition and PSR (assembly state estimation) are inherently temporal tasks,
and the current architecture relies on a separate VideoMAEStream (22M extra
params, frozen, only for activity) to add temporal information.

This design replaces ConvNeXt-Tiny + VideoMAEStream with a single Kinetics-
pretrained video backbone (MViTv2-S, 34.5M params) that feeds all 4 task heads.
Kinetics-400 is the canonical large-scale action dataset (240K clips, 400
classes); backbones pretrained on it carry strong motion and mid-level
spatiotemporal features that transfer directly to assembly activity recognition.

## 2. Proposed Architecture

```
Input clip [B, C=3, T=16, H=224, W=224]
    │
    └── VideoBackboneWrapper (MViTv2-S / VideoMAE-S)
            │                              │
            │  stage_features              │  clip_embed
            │  {c3, c4, c5}                │  [B, 768]
            │  (center frame)              │
            │       │                      │
            ▼       ▼                      │
         VideoFPN(P3-P7)                   │
            │       │                      │
       ┌────┤       ├────┐                 │
       │    │       │    │                 │
       ▼    ▼       ▼    ▼                 │
    Detect Pose  FiLM  PSR                 │
    (P3-P7) (P3)    │   (P3+P4+P5)         │
                    ▼                      │
              HeadPoseFiLM                 │
                    │                      │
                    ▼                      ▼
              c5_mod ──────────────► Activity Head
                                    (fuses embed + spatial)
```

## 3. Backbone Options

| Feature              | MViTv2-S            | VideoMAE-S           | ConvNeXt-Tiny (current) |
|----------------------|---------------------|----------------------|-------------------------|
| Parameters           | 34.5M               | 22.0M                | 28.0M                   |
| Pretraining          | Kinetics-400        | Kinetics-400         | ImageNet-1K             |
| K400 Top-1           | 82.3%               | 83.7% (w/ mask)      | N/A (2D only)           |
| Input format         | [B, 3, T, H, W]    | [B, 3, T, H, W]     | [B, 3, H, W]            |
| Clip frames          | 16                  | 16                   | 1                       |
| Feature dim          | 768                 | 384                  | 768 (C5)                |
| Int. features        | Hierarchical (4x)   | Uniform (12x ViT)    | Hierarchical (4x)       |
| Temporal modeling    | Inherent            | Inherent             | None (per-frame)        |
| HF transformers      | No (proxy via VMAE) | Yes                  | N/A                     |
| torchvision native   | Yes (0.19+)         | No                   | Yes                     |

**Recommendation: MViTv2-S.** It has hierarchical feature maps (natural for FPN
integration), native torchvision support with K400 weights, and its 4-stage
structure mirrors the existing C2-C5 pattern that the heads expect. VideoMAE-S
is a flat ViT with uniform feature dimension across all layers, making
multi-scale spatial feature extraction harder.

## 4. Integration Details

### 4.1 Feature Extraction

MViTv2-S processes the clip through:

1.  **conv_proj**: Conv3d(3, 96, k=(3,7,7), s=(2,4,4)) -> [B, 96, T/2, H/4, W/4]
    - The H/4 x W/4 feature map serves as a C2 surrogate for the spatial heads.
    - Center frame selected at t = T'/2.

2.  **16x MultiscaleBlocks**: Each block applies self-attention + pooling, with
    spatial/temporal downsampling at blocks 4, 8, 12:
    - Block 3  -> D~192, stride 8 relative to input  -> C3 surrogate
    - Block 7  -> D~384, stride 16                   -> C4 surrogate
    - Block 11 -> D~768, stride 32                    -> C5 surrogate
    - Block 15 -> D=768 final output                  -> clip embedding

3.  **Patch-to-spatial reshape**: Block outputs are patch sequences
    [B, 1+THW, D]. The class token is stripped, remaining patches are reshaped
    to [B, D, T', H', W'] and center-frame sliced to [B, D, H', W'].

### 4.2 Head Adaptation

| Head        | Input Source      | Change from POPW        |
|-------------|-------------------|-------------------------|
| Detection   | FPN P3-P7         | None (FPN adapted)      |
| Pose        | FPN P3            | None                    |
| Head Pose   | C4, C5 (GAP)      | Channel dims adapted    |
| PoseFiLM    | C5 + keypoints    | None (C5 dim adapted)   |
| Activity    | clip_embed + c5_mod + P4 | Fuses backbone embed |
| PSR         | FPN P3+P4+P5      | None                    |

The key change: C5 input to PoseFiLM / HeadPoseFiLM / ActivityHead uses
MViTv2's D=768 feature (matching ConvNeXt-Tiny's C5=768). C4=384 matches
ConvNeXt-Tiny's C4=384. The FPN lateral connections are adapted for the video
backbone's channel dimensions.

### 4.3 Gradient Flow

```
clip --> VideoBackbone --> C3/C4/C5 --> FPN --> Detection / Pose
                      |                  |
                      |                  └--> PSR Head (detach_psr_fpn)
                      |
                      +--> PoseFiLM --> HeadPoseFiLM --> c5_mod
                      |                                      |
                      |                                      +--> Activity Head (blend_ratio)
                      |
                      +--> clip_embed --> Activity Head (fused)
```

- Spatial heads (detection, pose) get gradients only through FPN + backbone.
- Activity head gets gradients through both spatial path (c5_mod, blend_ratio)
  and temporal path (clip_embed, full gradient).
- PSR head uses detach on FPN features (controlled by DETACH_PSR_FPN config).
- Pose gradients are detached at the keypoints -> PoseFiLM boundary.

## 5. Memory Budget

### 5.1 Parameter Count

| Component          | ConvNeXt-Tiny | Video Backbone | Delta   |
|--------------------|---------------|----------------|---------|
| Backbone           | 28.0M         | 34.5M          | +6.5M   |
| FPN                | 0.5M          | 0.5M           | 0       |
| Detection Head     | 0.8M          | 0.8M           | 0       |
| Pose Head          | 0.2M          | 0.2M           | 0       |
| PoseFiLM           | 0.8M          | 0.8M           | 0       |
| HeadPoseFiLM       | 0.3M          | 0.3M           | 0       |
| HeadPose Head      | 0.3M          | 0.3M           | 0       |
| Activity Head      | 1.2M          | 1.2M           | 0       |
| PSR Head           | 0.4M          | 0.4M           | 0       |
| VideoMAE Stream    | 22.0M         | REMOVED        | -22.0M  |
| **Total**          | **54.5M**     | **39.0M**      | **-15.5M** |

### 5.2 Activation Memory (batch=2, T=16, FP16, RTX 3060 11 GB)

| Component          | ConvNeXt + VMAE | Video Backbone |
|--------------------|-----------------|----------------|
| ConvNeXt (per-frame)| 0.6 GB          | REMOVED        |
| VideoMAE Stream     | 1.5 GB          | REMOVED        |
| MViTv2-S backbone  | 0.0 GB          | 2.3 GB         |
| FPN + 4 heads       | 1.0 GB          | 1.0 GB         |
| Feature Bank + temp | 0.2 GB          | 0.2 GB         |
| **Total (no ckpt)** | **~5.8 GB**     | **~6.5 GB**    |
| Total w/ ckpt       | ~3.5 GB         | **~4.0 GB**    |

### 5.3 Training Memory

The peak GPU memory comes from activation storage during backprop. With
gradient checkpointing on MViT blocks (recompute activations during backward,
store only input), total drops from ~6.5 GB to ~4.0 GB for batch=2.

Adding a third batch pushes to ~6.5 GB (with checkpointing) or ~10 GB (without).

**Configuration recommendation for RTX 3060 11 GB:**
- batch=2, T=16, gradient checkpointing ON -> ~4.0 GB safe zone
- batch=4, T=16, checkpointing ON -> ~6.5 GB (comfortable)
- batch=2, T=32, checkpointing ON -> ~5.5 GB (for longer clips)

## 6. Training Strategy

### 6.1 Stage 0: Frozen Backbone (warmup, ~1000 steps)

- Backbone frozen, only heads trainable.
- LR: 1e-4 for heads (AdamW).
- This warms up the randomly initialized heads without corrupting backbone
  features.
- Outputs are noisy but the backbone provides good Kinetics features from the
  start.

### 6.2 Stage 1: Last Stage Unfrozen (~5 epochs)

- MViTv2-S blocks 12-15 (last 4 of 16) unfrozen.
- Backbone LR: 1e-5, head LR: 1e-4.
- This allows adaptation of high-level temporal features to assembly tasks.

### 6.3 Stage 2: Full Fine-tune (~10+ epochs)

- All backbone parameters unfrozen.
- Backbone LR: 1e-5 (or 5e-6 for stability).
- Head LR: 1e-4.
- Cosine LR schedule with linear warmup.
- Gradient clipping at 1.0 to prevent backbone gradient shock.

### 6.4 Expected Training Time

| Stage | Components     | Batch | Throughput | Time per epoch |
|-------|---------------|-------|-----------|---------------|
| 0     | Heads only    | 4     | ~45 clips/s| ~2.5 hrs       |
| 1     | + last stage  | 4     | ~30 clips/s| ~3.5 hrs       |
| 2     | Full fine-tune| 4     | ~25 clips/s| ~4.5 hrs       |

Based on: ~200K training frames / T=16 / batch=4 = ~3,125 batches per epoch.
240K training frames is approximate (IndustReal dataset). RTX 3060 measured
throughput for MViTv2-S: ~25 clips/s at batch=4 with checkpointing.

**Total estimated training time: ~75-100 hours** for 15 epochs full pipeline.

## 7. Pros and Cons

### 7.1 Pros

1.  **Unified spatiotemporal backbone**: Single model processes video end-to-end,
    no separate streams or late fusion. This is the standard for modern video
    understanding (TimeSformer, VideoMAE, MViTv2).

2.  **Activity recognition SOTA**: MViTv2-S achieves 82.3% on Kinetics-400.
    Assembly activity recognition is a simpler domain (<100 classes vs 400),
    so we expect strong transfer. Target: +5-12% Top-1 over ConvNeXt + frozen
    VideoMAE.

3.  **Fewer total parameters**: 39M vs 54.5M (removes VideoMAEStream 22M).
    All parameters are trainable (no frozen 22M burden).

4.  **All 4 tasks benefit from Kinetics pretraining**: Detection, pose, head
    pose, and PSR all get richer spatiotemporal features from the video
    backbone's intermediate layers. The current ConvNeXt-Tiny has only
    ImageNet features.

5.  **Simpler pipeline**: No dual-backbone management, no separate VideoMAE
    clip processing, no distinct clip/feature data loading.

### 7.2 Cons

1.  **Larger per-sample memory**: Processing T=16 frames in a single forward
    pass requires ~2.3 GB of activation memory (vs 0.6 GB for ConvNeXt-Tiny
    single frame). Mitigated by gradient checkpointing.

2.  **No per-frame independent processing**: All 4 heads must use the clip's
    center frame for spatial predictions. Per-frame temporal detail is lost
    for detection/pose. This is acceptable for the IKEA/IndustReal use case
    where frame-to-frame variation is small and labels are per-frame.

3.  **Architecture dependency**: MViTv2-S requires torchvision >= 0.19. The
    current environment has 0.27.1, which satisfies this.

4.  **Longer inference latency**: ~2x ConvNeXt-Tiny (40ms vs 20ms per frame on
    RTX 3060), since each inference pass encodes T=16 frames in one go. This
    is acceptable for offline processing; for real-time, the latency can be
    amortized with sliding window inference (process every 8 frames, overlap
    by 8).

5.  **Calibration time**: The Kinetics-400 spatial statistics (object
    distributions, motion patterns) differ from IKEA furniture assembly. Fine-
    tuning may require careful LR scheduling to avoid catastrophic forgetting
    of K400 features while adapting to assembly domain.

## 8. Implementation Plan

### 8.1 Files Created

- `src/models/video_backbone_multitask.py` -- VideoMultiTaskModel, VideoBackboneWrapper,
  VideoFPN, training helpers.
- `src/models/VIDEO_BACKBONE_DESIGN.md` -- This design document.

### 8.2 Files Modified

- `src/training/train.py` or equivalent pipeline -- Add `--video-backbone` flag,
  switch between POPWMultiTaskModel and VideoMultiTaskModel.
- `src/training/losses.py` -- MultiTaskLoss is input-format-compatible (both
  models return the same output dict keys). No changes needed if output dict
  keys match.

### 8.3 Suggested CLI Integration

```bash
python train.py --model video_multitask \
    --video-backbone mvit_v2_s \
    --clip-frames 16 \
    --freeze-backbone \
    --batch-size 4 \
    --lr-head 1e-4 \
    --lr-backbone 1e-5
```

### 8.4 Verification Criteria

1.  **Forward pass**: `VideoMultiTaskModel.forward()` produces the same output
    dict keys as `POPWMultiTaskModel.forward()`.
2.  **Loss compatibility**: `MultiTaskLoss` accepts both output dicts without
    modification (same keys: cls_preds, reg_preds, anchors, heatmaps,
    keypoints, head_pose, act_logits, psr_logits, etc.).
3.  **Memory**: Training at batch=2 fits in RTX 3060 11 GB with gradient
    checkpointing enabled.
4.  **Activity Top-1**: Video backbone improves activity Top-1 by >=5 points
    over ConvNeXt-Tiny baseline within 10 epochs of fine-tuning.
5.  **PSR F1**: Temporal features from video backbone improve PSR per-component
    F1 by >=3 points averaged across all 11 components.

## 9. References

- MViTv2: Fan et al., "Multiscale Vision Transformers", CVPR 2022.
- VideoMAE: Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient
  Learners for Self-Supervised Video Pre-Training", NeurIPS 2022.
- Kinetics-400: Kay et al., "The Kinetics Human Action Video Dataset", 2017.
- POPW paper: "Pose-Conditioned Multi-Task Architecture for IndustReal", 2026.
- Existing model: src/models/model.py (POPWMultiTaskModel, ConvNeXtBackbone,
  VideoMAEStream).
- Existing video stream: src/models/video_stream.py (K400VideoStream).
