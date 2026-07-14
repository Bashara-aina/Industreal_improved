# === V1 FACT-CHECK (REQUIRED READING) ===
# This file was generated as V2 before the codebase was fully audited.
# The V1 documents (208-227) it references describe the MViTv2-S era
# which is NO LONGER the active codebase. Key corrections:
#
# **Active model:** POPWMultiTaskModel in src/models/model.py (2361 lines)
# **Backbone:** convnext_tiny (28.59M, ImageNet-1K pretrain) — NOT MViTv2-S
# **Total params:** 46.47M (measured) — NOT ~48.6M
# **Detection head:** RetinaNet-style 5.31M (9 anchors, 5 levels P3-P7)
# **Activity head:** FeatureBank+TCN+2xViT 0.69M (NOT 3-layer MLP 2M)
# **PSR head:** PSRHead hidden_dim=128 3.08M (NOT 1.8M Causal Transformer)
# **Pose:** body 1.64M + head 1.45M + 2 FiLMs 1.24M (NOT single 0.2M)
# **FPN:** Standard P3-P7 4.48M (NOT BiFPN P2-P5 ~2.5M)
# **Effective batch:** 48 (BATCH=6 x GRAD_ACCUM=8) — NOT 16
# **PSR_FOCAL_GAMMA:** 0.5 — NOT 2.0
# **Kendall caps:** act=[-0.5,2.0], psr=[-4,0.0], pose=[-4,3.0] (NOT [1.5,1.0,0.5,2.0])
# **Recordings:** 36/16/32 (NOT 10/6)
# **Frames:** 207,266 (NOT ~75K)
# **Free GPUs:** RTX 5060 Ti 16GB + RTX 3060 12GB
# **Deadline:** AAIML Oct 10, 2026
#
# See consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md for full audit.
# The legacy src/models/mvit_mtl_model.py (MViTv2-S) is DEAD CODE.
# All architecture numbers in this report should be re-validated against
# the active model before any paper submission.
# === END FACT-CHECK ===

# Agent 7: Neck/Feature Pyramid Analysis

## Codebase References

All source lines cited below are from:
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/mvit_mtl_model.py` (primary)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py` (original ConvNeXt-Tiny model)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbone_multitask.py` (intermediate video backbone MTL)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/rotograd.py` (RotoGrad implementation)

---

## 1. Current Neck Design

The `mvit_mtl_model.py` model uses a class called **`LightweightFPN`** (line 143) that is actually a **full BiFPN** (EfficientDet-style Bidirectional Feature Pyramid Network), not a standard FPN. Despite the "Lightweight" name, it implements the complete two-pass architecture:

- **1x1 lateral projections** from input channel dims to `out_channels=256` (line 159-162)
- **Top-down pathway** with `trilinear` upsampling + learnable weighted fusion (line 206-217)
- **Bottom-up pathway** with `trilinear` downsampling + learnable weighted fusion (line 220-232)
- **Learnable fusion weights** using `nn.ParameterDict` with `_fast_weightsum` (ReLU + normalize, per EfficientDet) (line 176-190, 193-196)
- **Smooth 3D convolutions** (Conv3d, kernel=3) on both pathways (line 165-173)
- **3D (spatiotemporal) convolutions** throughout, preserving temporal dimension T=8

Key evidence from the docstring at line 143-150:
```python
class LightweightFPN(nn.Module):
    """BiFPN — top-down + bottom-up with EfficientDet-style weighted fusion.

    [FIX 207 §2.5] P5 fusion now uses two inputs only (p5_td + max_pool(p4_out)),
    eliminating the duplicate p5_lat term.

    Input: dict {P2: 96ch, P3: 192ch, P4: 384ch, P5: 768ch} each [B, C, T, H, W]
    Output: dict {P2, P3, P4, P5} each [B, 256, T, H, W] with H,W halving per level.
    """
```

The input features come from `MViTFeaturePyramid` (line 40), which registers hooks on the MViTv2-S backbone at:
- `conv_proj` -> P2 (96ch, stride 4, 56x56) -- line 76-78
- `blocks[1]` -> P3 (192ch, stride 8, 28x28) -- line 81-84
- `blocks[3]` -> P4 (384ch, stride 16, 14x14) -- line 81-84
- `blocks[14]` -> P5 (768ch, stride 32, 7x7) -- line 81-84

**Comparison with the original model (`model.py`)**:

The original `POPWMultiTaskModel` (model.py line 1762) uses a **standard 2D FPN** (class `FPN`, line 390) that:
- Takes `[C3, C4, C5]` -> `[P3, P4, P5, P6, P7]` (5 levels, including P6/P7 via stride-2 conv)
- Uses additive fusion (simple sum, no learnable weights)
- Has only a top-down pathway
- Operates in 2D only

The intermediate `VideoMultiTaskModel` (video_backbone_multitask.py line 354) uses `VideoFPN` (line 310) with the same 2D top-only architecture but adapted for video backbone channels.

**The `LightweightFPN` is the most sophisticated neck in the codebase**, being the only one with:
1. Bidirectional feature flow
2. Learnable weighted fusion
3. 3D convolutions for temporal dimension

---

## 2. Feature Pyramid: What Multi-Scale Features Does Detection Receive?

Detection in `MTLMViTModel` receives **3 levels: P3, P4, P5** only. P2 is explicitly excluded.

Evidence from `MTLMViTModel.forward()` at lines 573-580:
```python
        det_outputs = {}
        for level_name, feat in fpn_out.items():
            if level_name == "P2":
                # Skip P2 (raw conv_proj features, no semantics — FC-2)
                continue
            # Temporal-pool T dimension for 2D detection
            pooled = feat.mean(dim=2)  # [B, 256, H, W]
            det_outputs[level_name] = self.det_head(pooled)
```

The strides for detection levels (assuming 224x224 input):
| Level | Spatial Resolution | Stride | Channel | Source |
|-------|-------------------|--------|---------|--------|
| P2 (excluded) | 56x56 | 4 | 256 | conv_proj |
| P3 | 28x28 | 8 | 256 | blocks[1] |
| P4 | 14x14 | 16 | 256 | blocks[3] |
| P5 | 7x7 | 32 | 256 | blocks[14] |

Temporal pooling is mean-over-time (dim=2, line 579), collapsing T=8 to single frame.

**Note**: The original `POPWMultiTaskModel` (model.py) uses P3-P7 (5 levels, strides 8-128) with anchor-based detection. The `MTLMViTModel` uses only P3-P5 (3 levels, strides 8-32) with anchor-free TAL assignment, a much lighter detection pipeline.

---

## 3. BiFPN Feasibility

**BiFPN is already implemented.** The `LightweightFPN` class at line 143 IS a BiFPN. The feasibility question is moot -- it exists and is the default neck.

### Cost analysis of current BiFPN:

**Parameter count** (estimated from architecture):

| Component | Calculation | Params |
|-----------|-------------|--------|
| Lateral P2 | Conv3d(96, 256, 1) | 96*256 + 256 = 24,832 |
| Lateral P3 | Conv3d(192, 256, 1) | 192*256 + 256 = 49,408 |
| Lateral P4 | Conv3d(384, 256, 1) | 384*256 + 256 = 98,560 |
| Lateral P5 | Conv3d(768, 256, 1) | 768*256 + 256 = 196,864 |
| **Lateral total** | | **369,664** |
| 4x TD convs | Conv3d(256, 256, k=3) each = 256*256*27 + 256 | 4 * 1,769,728 = **7,078,912** |
| 4x BU convs | Conv3d(256, 256, k=3) each = 256*256*27 + 256 | 4 * 1,769,728 = **7,078,912** |
| Fusion weights | 8 x nn.Parameter (few scalars) | ~16 |
| **Total BiFPN** | | **~14.5M** |

For comparison, the detection head (`DetectionHead` at line 241):
| Component | Calculation | Params |
|-----------|-------------|--------|
| cls_head | 2x Conv2d(256, 256, 3) + GN + Conv2d(256, 24, 1) | ~597K |
| reg_head | 2x Conv2d(256, 256, 3) + GN + Conv2d(256, 64, 1) | ~607K |
| **Total DetectionHead** | | **~1.2M** |

The neck is **~12x larger than the detection head** (~14.5M vs ~1.2M). This is an inverted ratio versus typical YOLOv8/RetinaNet architectures where the neck is usually 2-4x smaller than the detection head.

### VRAM impact:
Each Conv3d(256, 256, 3) with temporal T=8 processes `[B, 256, 8, H, W]`. At P3 (28x28) this is 256*8*28*28 = 1.6M activations per conv. With 8 such convs in the BiFPN, the activation memory for the neck is substantial. Gradient checkpointing on these convs would save ~3-4x at the cost of ~30% extra compute.

### MViT feature hierarchy support:
The MViTv2-S feature hierarchy is well-suited for the BiFPN:
- P2 (56x56) provides high-resolution spatial detail
- P3 (28x28), P4 (14x14), P5 (7x7) provide progressively semantic features
- The 2x spatial downsampling between adjacent levels matches the BiFPN's expected scale ratio

However, the spatiotemporal (3D) convolutions mean the BiFPN processes T=8 frames simultaneously, which is a design choice -- it could be simplified to 2D by pooling temporally first, saving ~27x compute per conv (the 3x3x3 kernel vs 3x3).

---

## 4. Cross-Scale Connections

The current BiFPN has **full bidirectional cross-scale connections**:

**Top-down pathway** (line 207-217):
```
P5 (latent) -> smooth_conv -> P5_td
    | upsample
P4 + P4_td(latent) -> weighted_fusion -> smooth_conv -> P4_td
    | upsample
P3 + P3_td(latent) -> weighted_fusion -> smooth_conv -> P3_td
    | upsample
P2 + P2_td(latent) -> weighted_fusion -> smooth_conv -> P2_td
```

**Bottom-up pathway** (line 220-232):
```
P2_td -> smooth_conv -> P2_out
    | downsample
P3_td + P3_out(down) -> weighted_fusion -> smooth_conv -> P3_out
    | downsample
P4_td + P4_out(down) -> weighted_fusion -> smooth_conv -> P4_out
    | downsample
P5_td + P5_out(down) -> weighted_fusion -> smooth_conv -> P5_out
```

This is strictly **more connectivity** than:
- Standard FPN (model.py line 390): top-down only, additive fusion
- YOLOv8 PANet: top-down + bottom-up, additive fusion
- Original FPN (Lin et al.): top-down only, additive fusion

The `_fast_weightsum` (line 193-196) uses a ReLU + normalize pattern from EfficientDet:
```python
@staticmethod
def _fast_weightsum(weights: torch.Tensor, terms: List[torch.Tensor]) -> torch.Tensor:
    w = F.relu(weights)
    return sum(w[i] * terms[i] for i in range(len(terms))) / (w.sum() + 1e-4)
```

This provides learnable per-level fusion weights, which is more expressive than the fixed addition used in the original model.py FPN or the VideoFPN.

---

## 5. Temporal Feature Fusion

Temporal features are handled differently per task:

### Detection (mvit_mtl_model.py:579):
```python
pooled = feat.mean(dim=2)  # [B, 256, H, W] -- collapse T via mean
```
- **Mean pooling** over time (T=8 after backbone processing)
- Single frame equivalent for 2D detection head
- No attention, no learned temporal aggregation

### PSR (mvit_mtl_model.py:424-454):
```python
# Pool spatial dims -> [B, 768, T=8, 1, 1] -> [B, T=8, 768]
x = self.spatial_pool(conv_proj_feat).squeeze(-1).squeeze(-1).transpose(1, 2)
# Causal Transformer over T=8 frames
x = self.temporal_encoder(x, mask=mask)  # [B, 8, feat_dim]
return self.projection(x)  # [B, 8, 11]
```
- **Causal Transformer** (2 layers, 4 heads, d=256) with upper-triangular causal mask
- Full temporal modeling with self-attention across the T=8 sequence
- Each of the 8 frames produces 11 per-component logits

### Activity (mvit_mtl_model.py:335-355):
- Uses only the clip-level **cls_token** [B, 768] -- single vector for the entire 16-frame clip
- No per-frame temporal aggregation within the head itself (the backbone provides the temporal encoding)

### Pose (mvit_mtl_model.py:481-490):
- Also uses only the **cls_token** [B, 768]
- Single 6D vector prediction per clip

**Summary of temporal handling:**

| Task | Temporal Aggregation | Temporal Dim |
|------|---------------------|--------------|
| Detection | Mean pool over T=8 | Collapsed to 1 |
| PSR | Causal Transformer over T=8 | Preserved (T=8) |
| Activity | cls_token (backbone pooled) | Collapsed to 1 |
| Pose | cls_token (backbone pooled) | Collapsed to 1 |

The old `POPWMultiTaskModel` (model.py) had a different pattern: PSR used per-frame features from the FPN pyramid (P3, P4, P5 GAP -> MLP), while detection operated on single-frame features. Activity used a Feature Bank (T=16) with TCN + ViT blocks for temporal modeling -- this is strictly more sophisticated than the cls_token-only approach in `MTLMViTModel`.

---

## 6. Neck Parameter Count

### Estimated parameter breakdown for `LightweightFPN` (mvit_mtl_model.py): ~14.5M

| Sub-module | Formula | Params |
|------------|---------|--------|
| lateral["P2"] | Conv3d(96, 256, 1) | 24,832 |
| lateral["P3"] | Conv3d(192, 256, 1) | 49,408 |
| lateral["P4"] | Conv3d(384, 256, 1) | 98,560 |
| lateral["P5"] | Conv3d(768, 256, 1) | 196,864 |
| td_conv["P2"] | Conv3d(256, 256, 3) | 1,769,728 |
| td_conv["P3"] | Conv3d(256, 256, 3) | 1,769,728 |
| td_conv["P4"] | Conv3d(256, 256, 3) | 1,769,728 |
| td_conv["P5"] | Conv3d(256, 256, 3) | 1,769,728 |
| bu_conv["P2"] | Conv3d(256, 256, 3) | 1,769,728 |
| bu_conv["P3"] | Conv3d(256, 256, 3) | 1,769,728 |
| bu_conv["P4"] | Conv3d(256, 256, 3) | 1,769,728 |
| bu_conv["P5"] | Conv3d(256, 256, 3) | 1,769,728 |
| Fusion weights | 8 x nn.Parameter (1-2 scalars) | ~16 |
| **Total** | | **~14,527,504** |

### Detection head (mvit_mtl_model.py:241-264): ~1.2M

| Sub-module | Params |
|------------|--------|
| cls_head[0] Conv2d(256, 256, 3) + bias | 590,080 |
| cls_head[1] GroupNorm(32, 256) | 512 |
| cls_head[3] Conv2d(256, 24, 1) + bias | 6,168 |
| reg_head[0] Conv2d(256, 256, 3) + bias | 590,080 |
| reg_head[1] GroupNorm(32, 256) | 512 |
| reg_head[3] Conv2d(256, 64, 1) + bias | 16,448 |
| **Total** | **~1,203,800** |

### Ratio: neck : detection head = ~14.5M : ~1.2M = ~12:1

The neck dominates the non-backbone parameters. Combined BiFPN + DetectionHead = ~15.7M out of the estimated ~40M total model params, meaning ~39% of total params are in the neck+detection pipeline.

For context, in the original model.py:
- `FPN` (model.py line 390): ~1.0M params (2D convs, 3 lateral + 3 smooth + 2 extra = 8 Conv2d modules)
- `DetectionHead` (model.py line 500): ~2.7M params (4+4 conv tower per subnet + 2 output convs)
- Ratio neck:det = ~1:2.7

The new BiFPN is **~14.5x larger** than the old FPN, making it a significant parameter sink.

---

## 7. YOLOv8 Comparison

### Structural comparison:

| Feature | Our LightweightFPN (mvit_mtl) | Standard YOLOv8 PANet | Difference |
|---------|--------------------------------|----------------------|------------|
| Feature levels | P2-P5 (4 levels) | P3-P5 (3 levels) | ours has higher-res P2 |
| Direction | Bidirectional (TD + BU) | Bidirectional (FPN + PAN) | Equivalent topology |
| Fusion method | Learnable weighted sum (`_fast_weightsum`) | Fixed element-wise addition | Ours is more expressive |
| Convolution type | Conv3d (spatiotemporal, T=8) | Conv2d (spatial only) | Ours handles temporal |
| Lateral projection | Conv3d 1x1 per level | Conv2d 1x1 per level | Equivalent |
| Smooth convs | Conv3d 3x3 per level (8 total) | Conv2d 3x3 per level (3 total) | Ours has more |
| Extra levels (P6/P7) | None | None in YOLOv8 | Equivalent |
| Output channels | 256 all levels | 256 all levels | Equivalent |

### Detection head comparison:

| Feature | Our DetectionHead (mvit_mtl) | YOLOv8 Detect head |
|---------|--------------------------------|-------------------|
| Structure | Decoupled cls + reg | Decoupled cls + reg |
| Conv layers per branch | 1x Conv3x3 + GN + ReLU | 2x Conv3x3 + BN + SiLU |
| Output | cls_logits [B, C, H, W] + reg_preds [B, 4*16, H, W] | cls + reg + dfl |
| DFL | Yes (reg_max=16) | Yes (reg_max=16) |
| Assigner | TAL (TOOD-style, tal_assigner.py) | TAL (TOOD-style port) |

Our detection head is simpler (1 conv vs 2 convs per branch) but functionally equivalent. Both use DFL with reg_max=16.

### TAL assigner:

Both use TaskAlignedAssigner from TOOD (Feng et al., ICCV 2021). Evidence from tal_assigner.py line 1-7:
```python
"""TaskAlignedAssigner for YOLO-style object detection.

[OPUS 192] Citation: TOOD: Task-aligned One-stage Object Detection (Feng et al.,
ICCV 2021). NOT Ultralytics YOLOv8 (which is AGPL-3.0 and has no peer-reviewed
paper). The TAL algorithm itself is in TOOD; YOLOv8 uses a port of TOOD's
align_metric with a different topk. Our implementation follows TOOD with
topk=10 (YOLOv8 default).
```

---

## 8. Multi-Task Feature Sharing

### In `MTLMViTModel` (mvit_mtl_model.py):

All 4 heads share the **same backbone features**, but derive different representations:

```
Input clip [B, 3, T=16, 224, 224]
    |
MViTFeaturePyramid (line 40)
    +---> fpn_feats dict: {P2: [B,96,T,56,56], P3: [B,192,T,28,28], P4: [B,384,T,14,14], P5: [B,768,T,7,7]}
    +---> cls_token: [B, 768]
    |
    +---> LightweightFPN(fpn_feats) -> fpn_out dict: {P2,P3,P4,P5} [B,256,T,H,W]
    |         +---> temporal pool -> DetectionHead -> det_outputs (P3/P4/P5 only)
    |
    +---> ActivityHead(cls_token) -> act_logits [B, 75]
    |
    +---> PSRHead(fpn_feats["P5"]) -> psr_logits [B, T=8, 11]
    |
    +---> PoseHead(cls_token) -> pose_6d [B, 6]
```

All gradients flow back through the single shared backbone. There is **no per-task feature routing** (no RotoGrad rotations, no FiLM modulation). The only "differentiation" is that:
- Detection uses FPN-pyramid features (post-BiFPN)
- PSR uses raw P5 features (pre-BiFPN from fpn_feats, not fpn_out)
- Activity and Pose use the cls_token (post-backbone-pooling)

### RotoGrad exists but is unused (rotograd.py):

The `RotoGradRotation` class at rotograd.py:36-120 provides per-task feature rotation:
```python
class RotoGradRotation(nn.Module):
    """Per-task feature rotation for gradient direction alignment.

    Each task gets a rotation matrix R_k belongs to SO(d) applied to the shared
    cls_token BEFORE the task-specific head.
```
And `RotoGradScale` at rotograd.py:123-168 provides gradient magnitude normalization.

These are **not integrated into `MTLMViTModel`**. They are standalone modules available for integration.

### Comparison with POPWMultiTaskModel (model.py):

The original model has significantly more complex feature routing:

| Aspect | MTLMViTModel (new) | POPWMultiTaskModel (old) |
|--------|-------------------|-------------------------|
| Detection | FPN -> detection head | FPN -> detection head |
| Pose | cls_token -> 2-layer MLP | P3 -> ConvT -> soft-argmax |
| PoseFiLM | None | Keypoints -> gamma/beta -> C5 modulation |
| HeadPoseFiLM | None | Head pose -> gamma/beta -> C5_mod2 |
| Activity | cls_token -> 3-layer MLP | GAP(C5_mod) + GAP(P4) + det_conf -> proj -> Feature Bank -> TCN -> 2x ViT -> CLS |
| PSR | P5 -> spatial pool -> causal Transformer | Multi-scale P3+P4+P5 GAP -> MLP -> causal Transformer |
| Gradient isolation | None explicit | DETACH_REG_FPN, DETACH_PSR_FPN, detach keypoints |
| RotoGrad | Not integrated | Not integrated |

The new model is simpler and parameter-efficient for the activity/pose heads but lacks the FiLM-based conditioning that the original model used for activity recognition.

---

## 9. Missing YOLOv8-Style Components

The `mvit_mtl_model.py` DetectionHead is described as "lightweight YOLO-style decoupled head" (line 238) and does share YOLOv8 characteristics:
- Decoupled cls/reg branches
- DFL distribution output (reg_max=16)
- TAL assigner (from tal_assigner.py)

However, it differs from standard YOLOv8 in:
1. **No PANet neck** -- uses BiFPN instead (BiFPN is actually more sophisticated)
2. **Only 1 conv layer** per branch vs YOLOv8's 2 conv layers per branch
3. **GroupNorm** instead of BatchNorm
4. **ReLU** instead of SiLU activations
5. **3D convolutions** in the neck (YOLOv8 is strictly 2D)

---

## Verdict

### Finding 1 (HIGH confidence): The BiFPN is already implemented

The `LightweightFPN` class at `mvit_mtl_model.py:143` IS a full BiFPN with learnable weighted fusion, bidirectional feature flow, and spatiotemporal convs. The question "could a BiFPN be added?" is moot -- it exists and is the default neck. **Evidence strength: DIRECT -- code at lines 143-234.**

### Finding 2 (HIGH confidence): The neck is disproportionately large

At ~14.5M params, the BiFPN neck is ~12x larger than the ~1.2M detection head and ~14.5x larger than the original 2D FPN (~1.0M). This is an inverted parameter ratio compared to typical detection architectures. The 8x Conv3d(256, 256, 3) modules are the primary cost. **Evidence strength: STRONG -- derived from architecture at lines 153-234; confirmed by module structure.**

### Finding 3 (MEDIUM confidence): Detection uses only 3 of 4 FPN levels

P2 is explicitly excluded from detection (`mvit_mtl_model.py:575-577`) because it carries conv_proj patch embeddings with limited semantics. The BiFPN still computes P2 for the bottom-up pathway, so P2 contributes to P3's fusion but is not directly supervised. This means P2's computation (~1.8M from its td_conv and bu_conv) contributes to the neck but is entirely auxiliary. **Evidence strength: DIRECT -- code at lines 575-577 confirms P2 skip, but no evidence quantifies how much P2 contributes to P3 quality.**

### Finding 4 (MEDIUM confidence): No temporal feature fusion for detection

Detection applies simple mean pooling over time (`mvit_mtl_model.py:579`). For a video-based detection task, there is no learned temporal aggregation. This contrasts with PSR which uses a full causal Transformer over the same T=8 sequence. Adding lightweight temporal attention (e.g., a single transformer block on the temporally-pooled features) could improve detection consistency across frames. **Evidence strength: DIRECT -- line 579 shows mean pooling; no temporal attention modules in DetectionHead.**

### Finding 5 (MEDIUM confidence): Multi-task feature routing is not used

All 4 heads share the same backbone gradients with no per-task routing. The `RotoGrad` module (`rotograd.py`) exists in the codebase but is not integrated into `MTLMViTModel`. The original `POPWMultiTaskModel` had FiLM-based conditioning (PoseFiLM, HeadPoseFiLM) that provided feature routing, but this was removed in the `MTLMViTModel` redesign. **Evidence strength: DIRECT -- rotograd.py lines 1-5 confirm availability; mvit_mtl_model.py forward shows no routing integration.**

### Finding 6 (LOW confidence/LIMITED CHECK): The 3D BiFPN may be overkill for the temporal dimension

The use of Conv3d with kernel=3 means each convolution processes T=8 frames jointly with a 3x3x3 kernel. Since detection temporally pools to 2D immediately after the BiFPN (line 579), there may be an opportunity to simplify to 2D BiFPN + temporal pooling earlier, saving ~96.4% of the activation memory per convolution (27x fewer MACs per kernel application). **Evidence: SPECULATIVE -- no ablation study exists to validate this claim. Requires a controlled experiment to measure detection mAP change.**

### UNVERIFIED claims:

- The YOLOv8-style design: The code correctly describes itself as "YOLO-style" -- it has a decoupled head with DFL and TAL assigner, but uses BiFPN instead of PANet, GroupNorm instead of BN, and ReLU instead of SiLU. The label is directionally correct but technically imprecise. VERIFIED as described in code comments.
- Detection accuracy without temporal fusion: Not validated. No ablation results available in the codebase comparing mean-pool vs attended temporal aggregation for detection.
- Whether P2 removal from detection hurts small-object detection: Not validated. The MViTv2-S feature resolutions are low (P3 = 28x28 for 224px input), so small objects would naturally be hard regardless.
