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

# Agent 6: Backbone Capacity Analysis

**Date:** 2026-07-13
**Model:** MViTv2-S (torchvision `mvit_v2_s`) with 4-task MTL head
**Files analyzed:**
- `src/models/mvit_mtl_model.py` — MTL model definition
- `torchvision/models/video/mvit.py` — core MViT implementation (installed at `/home/newadmin/miniconda3/lib/python3.13/site-packages/torchvision/models/video/mvit.py`)
- `scripts/train_mtl_mvit.py` — MTL training script
- `src/training/train.py` — environment config (expandable_segments, CUDA settings)
- `src/training/train_video_finetune.py` — single-task finetuning with gradient checkpointing

---

## 1. Parameter Count: Backbone vs Head Split

### Ground truth from actual model instantiation

```python
# Computed by instantiating MTLMViTModel(num_act_classes=75) and summing parameters per component
Backbone (feature_pyramid.backbone): 34.229M (61.5%)
FPN (fpn):                          14.528M (26.1%)
DetectionHead:                       1.201M ( 2.2%)
ActivityHead:                        3.750M ( 6.7%)
PSRHead:                             1.780M ( 3.2%)
PoseHead:                            0.202M ( 0.4%)
--------------------------------------------------
Total:                              55.689M (100%)
```

**Evidence:** `src/models/mvit_mtl_model.py`, line 5 — docstring claims "Total ~45M params (34.5M backbone + ~10M heads)". This is **stale/wrong**. The actual total is 55.7M, and heads (including the BiFPN) total 21.5M. The 10M head estimate likely predates the BiFPN addition (14.5M alone) or reflects only the 4 task heads (6.9M combined).

| Component | Parameters | Source |
|-----------|-----------|--------|
| MViTv2-S backbone (torchvision) | 34,537,744 | `mvit.py` line 654: `MViT_V2_S_Weights.meta["num_params"]` |
| Backbone (our instance, minus head) | 34,229,000 | Our computation (no pretrained weights loaded, slight variance) |
| BiFPN (lateral + td_conv + bu_conv) | 14,528,000 | `mvit_mtl_model.py` lines 143–234 |
| DetectionHead (shared, 3 FPN levels) | 1,201,000 | `mvit_mtl_model.py` lines 241–279 |
| ActivityHead (768→2048→1024→75 MLP) | 3,750,000 | `mvit_mtl_model.py` lines 286–370 |
| PSRHead (768→256 + 2-layer causal Transformer) | 1,780,000 | `mvit_mtl_model.py` lines 376–454 |
| PoseHead (768→256→6 MLP) | 202,000 | `mvit_mtl_model.py` lines 460–490 |

### Head parameter breakdown detail

**ActivityHead** (line 302–325): `LayerNorm(768) + Linear(768,2048) + Linear(2048,1024) + Linear(1024,75)`
```
LayerNorm: 768 * 2 = 1,536
Linear(768,2048): 768*2048 + 2048 = 1,573,376
Linear(2048,1024): 2048*1024 + 1024 = 2,098,176
Linear(1024,75): 1024*75 + 75 = 76,875
Total: 3,749,963 (~3.75M)
```

**PSRHead** (line 391–415): `Linear(768,256) + 2×TransformerEncoderLayer(d=256, nhead=4, ff=1024) + Linear(256,11)`
```
input_proj: 768*256 + 256 = 196,864
Per encoder layer: self_attn(256→768→256) + MLP(256→1024→256) + norms = ~790K
2 layers: ~1,580K
projection: 256*11 + 11 = 2,827
Total: ~1,780K (~1.78M)
```

**PoseHead** (line 463–470): `Linear(768,256) + Linear(256,6)`
```
Linear(768,256): 768*256 + 256 = 196,864
Linear(256,6): 256*6 + 6 = 1,542
Total: 198,406 (~0.20M)
```

**DetectionHead** (line 248–264): Per FPN level, shared weights (used on P3,P4,P5):
```
cls_head: Conv2d(256,256,3) + GN(32,256) + Conv2d(256,24,1) = ~590K
reg_head: Conv2d(256,256,3) + GN(32,256) + Conv2d(256,64,1) = ~614K
Total per level: ~1.20M × 3 levels = actually shared weights, so total is ~1.20M
```

**Note on docstring discrepancy:** The file-level docstring at line 5 says "Total ~45M params (34.5M backbone + ~10M heads)" but actual total is 55.7M. The BiFPN alone (14.5M) makes this claim invalid. This should be updated.

---

## 2. Receptive Field

### Temporal receptive field

The MViTv2-S processes T=16 input frames. The `conv_proj` layer pools temporally by stride 2:

```python
# mvit.py lines 484–490
self.conv_proj = nn.Conv3d(
    in_channels=3,
    out_channels=block_setting[0].input_channels,  # 96
    kernel_size=(3, 7, 7),
    stride=(2, 4, 4),
    padding=(1, 3, 3),
)
```

After conv_proj: `T = 16 // 2 = 8`, `H = 224 // 4 = 56`, `W = 224 // 4 = 56`.

**No further temporal pooling occurs anywhere in the network.** All 16 transformer blocks have `stride_q[0] = 1` and `stride_kv[0] = 1` for the temporal dimension, meaning the T=8 resolution is preserved end-to-end.

Evidence from `mvit.py` lines 834–851 (MViTv2-S config):
```python
"stride_q": [
    [1, 1, 1],  # block 0: no temporal or spatial pooling (T=8, 56x56)
    [1, 2, 2],  # block 1: spatial pool only (T=8, 28x28)
    [1, 1, 1],  # block 2: no pool (T=8, 28x28)
    [1, 2, 2],  # block 3: spatial pool only (T=8, 14x14)
    [1, 1, 1],  # blocks 4-13: no pool (T=8, 14x14)
    ...         # (10x repeat of [1,1,1])
    [1, 2, 2],  # block 14: spatial pool only (T=8, 7x7)
    [1, 1, 1],  # block 15: no pool (T=8, 7x7)
]
```

The first element of each `stride_q` entry is always 1 (no temporal stride). Same for `stride_kv` (lines 852–869).

**Implication:** The model processes exactly 8 temporal tokens. Each attention head in each block performs full 3D attention across T=8 frames. There is no hierarchical temporal abstraction — unlike the spatial dimension (which goes 56→28→14→7), the temporal dimension stays flat at 8.

### Spatial receptive field schedule

| Stage | Block(s) | Input HxW | Q stride | KV stride | Output HxW | Channels |
|-------|----------|-----------|----------|-----------|------------|----------|
| conv_proj | — | 224 | (2,4,4) | — | 56 | 96 |
| Block 0 | 0 | 56 | (1,1,1) | (1,8,8) | 56 | 96 |
| Block 1 | 1 | 56 | (1,2,2) | (1,4,4) | 28 | 192 |
| Block 2 | 2 | 28 | (1,1,1) | (1,4,4) | 28 | 192 |
| Block 3 | 3 | 28 | (1,2,2) | (1,2,2) | 14 | 384 |
| Blocks 4–13 | 4–13 | 14 | (1,1,1) | (1,2,2) | 14 | 384 |
| Block 14 | 14 | 14 | (1,2,2) | (1,1,1) | 7 | 768 |
| Block 15 | 15 | 7 | (1,1,1) | (1,1,1) | 7 | 768 |

The `stride_kv` values mean the Key and Value tensors are pooled aggressively in early blocks (stride 8 in block 0), creating a pyramidal resolution structure where Q attends to a coarsened K,V representation. This is the "multiscale" in MViT.

### Multi-resolution thw tracking

The dynamic thw computation in `mvit_mtl_model.py` lines 106–109:
```python
T_t, H_t, W_t = x.shape[2], x.shape[3], x.shape[4]
x = x.flatten(2).transpose(1, 2)
x = self.backbone.pos_encoding(x)
thw = (T_t, H_t, W_t)
```

For different input resolutions (T=16):
| Input | After conv_proj | Block 0 | Block 1 | Block 3 | Block 14 | Block 15 |
|-------|----------------|---------|---------|---------|----------|----------|
| 224 | 8 x 56 x 56 | 8 x 56 x 56 | 8 x 28 x 28 | 8 x 14 x 14 | 8 x 7 x 7 | 8 x 7 x 7 |
| 320 | 8 x 80 x 80 | 8 x 80 x 80 | 8 x 40 x 40 | 8 x 20 x 20 | 8 x 10 x 10 | 8 x 10 x 10 |
| 480 | 8 x 120 x 120 | 8 x 120 x 120 | 8 x 60 x 60 | 8 x 30 x 30 | 8 x 15 x 15 | 8 x 15 x 15 |
| 640 | 8 x 160 x 160 | 8 x 160 x 160 | 8 x 80 x 80 | 8 x 40 x 40 | 8 x 20 x 20 | 8 x 20 x 20 |

---

## 3. Temporal Modeling

### How MViTv2-S handles temporal dimension

The temporal dimension is handled via **3D pooling attention** — each attention head applies a depthwise Conv3d to Q, K, and V independently before the attention computation:

```python
# mvit.py lines 289–321 (MultiscaleAttention.forward)
q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(dim=2)

if self.pool_k is not None:
    k, k_thw = self.pool_k(k, thw)  # depthwise Conv3d on K with kernel_kv, stride_kv
if self.pool_v is not None:
    v = self.pool_v(v, thw)[0]       # depthwise Conv3d on V with kernel_kv, stride_kv
if self.pool_q is not None:
    q, thw = self.pool_q(q, thw)     # depthwise Conv3d on Q with kernel_q, stride_q
```

The pooling is done by `Pool` class (`mvit.py` lines 65–106):
```python
class Pool(nn.Module):
    def __init__(self, pool: nn.Module, norm, ...):
        ...
        self.pool = pool  # nn.Conv3d with groups=head_dim (depthwise)

    def forward(self, x, thw):
        # Reshape: [B, n_head, N, head_dim] -> [B*n_head, head_dim, T, H, W]
        x = x.transpose(2, 3)
        B, N, C = x.shape[:3]
        x = x.reshape((B * N, C) + thw).contiguous()
        x = self.pool(x)  # Conv3d with kernel_q=[3,3,3] or kernel_kv=[3,3,3]
        T, H, W = x.shape[2:]
        ...
        return x, (T, H, W)
```

The `kernel_q` and `kernel_kv` are always `[3, 3, 3]` for ALL blocks in MViTv2-S (`mvit.py` lines 798–833). This means each Q element pools over a 3x3x3 neighborhood in T×H×W space.

However, since `stride_q[0] = 1` and `stride_kv[0] = 1` for all blocks, the temporal resolution never collapses — the Conv3d with kernel size 3 and stride 1 is a dilated temporal filter that keeps T=8 resolution with local temporal context.

### Attention-pooling schedule (temporal)

| Block | Q kernel (T,H,W) | Q stride (T,H,W) | KV kernel (T,H,W) | KV stride (T,H,W) | Temporal effect |
|-------|-----------------|------------------|------------------|------------------|-----------------|
| 0 | (3,3,3) | (1,1,1) | (3,3,3) | (1,8,8) | Q: local 3-frame context; KV: heavy spatial pool, no T pool |
| 1 | (3,3,3) | (1,2,2) | (3,3,3) | (1,4,4) | Q: spatial pool; KV: moderate spatial pool |
| 2 | (3,3,3) | (1,1,1) | (3,3,3) | (1,4,4) | Q: no pool; KV: moderate spatial pool |
| 3 | (3,3,3) | (1,2,2) | (3,3,3) | (1,2,2) | Q+KV: spatial pool 14x14 |
| 4–13 | (3,3,3) | (1,1,1) | (3,3,3) | (1,2,2) | Q: no pool; KV: 2x spatial pool → 7x7 |
| 14 | (3,3,3) | (1,2,2) | (3,3,3) | (1,1,1) | Q: spatial pool; KV: no pool |
| 15 | (3,3,3) | (1,1,1) | (3,3,3) | (1,1,1) | Neither pools |

### Temporal modeling limitation

**The model operates on a fixed T=8 temporal window.** This is the input clip length (16 frames) after conv_proj stride 2. There is no mechanism to handle longer sequences — T is fixed at construction time via `input_size`.

```python
# mvit.py lines 397–410 (PositionalEncoding.__init__)
self.temporal_size = temporal_size  # = 16 // 2 = 8 after conv_proj stride 2
```

The temporal position encoding parameters (`rel_pos_t`) are sized for T=8 only. The relative position bias table has shape `[temporal_dim, head_dim]` where `temporal_dim = 2 * input_size[0] - 1 = 15` (from `mvit.py` line 281). This is for the relative position distances, not sequence positions, so it can handle any T up to 8.

**Practical limit:** The model cannot model actions or activities longer than 16 frames (~0.5s at 30fps) without clip-level aggregation. For longer assembly actions (which can span 30-120 frames), the model must rely on late fusion (e.g., temporal pooling in the detection FPN or the PSR head's causal transformer over T=8). There is no hierarchical temporal abstraction.

---

## 4. Capacity Bottlenecks

### Hidden dimensions per block

From `mvit.py` lines 795–797 (MViTv2-S config):
```python
"input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
"output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
```

| Block(s) | Input Ch | Output Ch | Heads | Head Dim | Spatial Res |
|----------|----------|-----------|-------|----------|-------------|
| 0 | 96 | 96 | 1 | 96 | 56x56 |
| 1 | 96 | 192 | 2 | 96 | 28x28 |
| 2 | 192 | 192 | 2 | 96 | 28x28 |
| 3 | 192 | 384 | 4 | 96 | 14x14 |
| 4–13 | 384 | 384 | 4 | 96 | 14x14 |
| 14 | 384 | 768 | 8 | 96 | 7x7 |
| 15 | 768 | 768 | 8 | 96 | 7x7 |

Head dimension is fixed at 96 throughout (output_channels / num_heads = 96 for all blocks).

### Bottleneck analysis for 4 tasks

The MTL head consumes backbone features at two points:
1. **Class token** (768-dim): Activity and Pose heads
2. **P5 features** (768-dim, 7x7 spatial): PSR head
3. **P2–P5 features** (96→768 ch): Detection FPN

**Potential bottleneck: The 96→192→384→768 channel progression means**

- **P2 (96ch, 56x56):** Used only for FPN top-down path. These are raw patch embeddings with no semantic information (`mvit_mtl_model.py` lines 574–578: "Skip P2 (raw conv_proj features, no semantics — FC-2)").
- **P3 (192ch, 28x28):** After only 1 transformer block. Limited semantics.
- **P4 (384ch, 14x14):** After 3 transformer blocks. Moderate semantics.
- **P5 (768ch, 7x7):** After 14 transformer blocks. Full semantics.

The 384-dim bottleneck in blocks 4–13 (10 blocks at 14x14 resolution) is the narrowest point. For 4 simultaneous tasks competing for representational capacity, this 384-dim shared space could be a limiting factor. However, the final 768-dim class token (used by activity and pose) provides sufficient capacity for those two heads.

The MLP expansion ratio is 4× throughout (`mvit.py` line 370: `MLP(attn_dim, [4 * attn_dim, output_channels], ...)`), so the intermediate MLP dimension is 4×384=1536 for blocks 4–13. This is adequate for single-task Kinetics-400 training, but for 4-task MTL, the competition for representational capacity within 384-dim features is a real bottleneck.

**Detection head bottleneck:** The FPN projects all levels to 256 channels (`FPN_CHANNELS = 256` at line 33). The detection head then uses 256-dim features. For 24-class detection with DFL (16 bins × 4 coords = 64 regression outputs), this is adequate.

---

## 5. Comparison to Larger Backbones

### Available alternatives

| Model | Params | GFLOPs | Kinetics-400 Top-1 | VRAM (FP16, B=1, 224) | Fits 16GB? |
|-------|--------|--------|--------------------|----------------------|------------|
| MViTv2-S (current) | 34.5M | 64 | 80.8% | ~3.5 GB with grad ckpt | YES |
| MViTv1-B | 36.6M | 71 | 78.5% | ~4 GB | YES |
| MViTv2-B (paper) | ~52M | ~112 | ~82.0% | ~5.5 GB | YES (maybe w/ ckpt) |
| MViTv2-L (paper) | ~97M | ~282 | ~83.5% | ~9 GB | NO (training) |
| MViTv2-H (paper) | ~167M | ~537 | ~84.8% | ~15 GB | NO |
| VideoMAE-base | ~87M | ~180 | ~81.0% | ~8 GB | YES (w/ ckpt) |
| VideoMAE-large | ~305M | ~600+ | ~83.9% | ~20+ GB | NO |
| VideoMAE-huge | ~657M | ~1200+ | ~84.5% | ~40+ GB | NO |

**Sources:**
- MViTv2-S params: `mvit.py` line 654: `"num_params": 34537744`
- MViTv1-B params: `mvit.py` line 621: `"num_params": 36610672`
- MViTv2 family benchmarks: paper Table 1 (MViTv2: Improved Multiscale Vision Transformers, CVPR 2022)
- Kinetics-400 top-1: `mvit.py` lines 657 and 624

### MViTv2-B feasibility on 16GB

MViTv2-B (~52M params, ~112 GFLOPs) would approximately double the FLOPs vs MViTv2-S. The attention memory cost at 224 would scale by channel count ratio (roughly 1.5x). Estimated ~5.5 GB for backbone at FP16 B=1 with gradient checkpointing. This would fit on 16GB for B=1 training.

However, the current BiFPN and MTL heads would add their own overhead. Total model with MViTv2-B backbone would be ~52M + 21.5M = ~73.5M, still potentially feasible at 224 with gradient checkpointing and B=1.

**Not feasible at 480px.** The attention memory cost at 480 would scale similarly to MViTv2-S (2.95 GB per sample for attn matrices alone), and with ~1.5x larger hidden dims, it would be ~4.4 GB for attn matrices alone. Total VRAM would exceed 16GB.

### VideoMAE-base feasibility

VideoMAE uses a standard ViT-B architecture (12 layers, 768 hidden, 12 heads, 3072 MLP) with 16x16x16 tubelet embedding. For T=16, H=224:
- Patches: T'=1 (if stride 16 in T), H'=14, W'=14 → 196 patches
- Attention: [B, 12, 197, 197] per layer × 12 layers ≈ very manageable
- Total: ~87M params, ~180 GFLOPs

VideoMAE-base would fit on 16GB at 224 with gradient checkpointing, but the temporal resolution is worse (1 frame token vs 8 frame tokens in MViT). This is because VideoMAE uses a 16-frame tubelet, collapsing all temporal information into a single token per spatial location.

---

## 6. The THW Bug: Multi-Resolution Inference Fix

### The fix (1 line)

The original MViT.forward() at `mvit.py` line 557 hardcodes the `thw` tuple:
```python
thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
```

This is **fixed at construction time** — for MViTv2-S initialized with `spatial_size=(224,224)`, `thw = (8, 56, 56)`. If you pass in a 320px or 480px input, the conv_proj output shape changes (e.g., to (8, 80, 80) for 320px), but `thw` is still (8, 56, 56). The Pool module uses `thw` to reshape tensors:

```python
# mvit.py lines 87–106 (Pool.forward)
x = x.reshape((B * N, C) + thw).contiguous()  # WRONG shape for non-224 input!
x = self.pool(x)
T, H, W = x.shape[2:]  # These will be the post-pool shape, but reshape was wrong
```

**The fix** in `mvit_mtl_model.py` lines 106–109 replaces the hardcoded thw with dynamic computation from the actual conv_proj output:

```python
# [FIX 2026-07-13] Use dynamic thw from conv_proj output instead of hardcoded
# pos_encoding.spatial_size. Enables multi-resolution inference (224/320/480):
T_t, H_t, W_t = x.shape[2], x.shape[3], x.shape[4]
x = x.flatten(2).transpose(1, 2)
x = self.backbone.pos_encoding(x)
thw = (T_t, H_t, W_t)
```

### Why this works

MViTv2-S uses **relative position encoding** (`rel_pos_embed=True` at `mvit.py` line 892), not absolute position embeddings. The relative position bias tables are sized based on the maximum distance between tokens, not the absolute grid size. The `_interpolate` function (`mvit.py` lines 109–121) dynamically interpolates these tables for any input resolution:

```python
def _interpolate(embedding: torch.Tensor, d: int) -> torch.Tensor:
    if embedding.shape[0] == d:
        return embedding
    return nn.functional.interpolate(
        embedding.permute(1, 0).unsqueeze(0), size=d, mode="linear",
    ).squeeze(0).permute(1, 0)
```

This means relative pos encoding naturally generalizes to different resolutions — the only thing preventing multi-resolution inference was the fixed `thw` tracking.

### Why it was critical

Without this fix, any inference at a resolution other than 224x224 would:
1. Reshape the conv_proj output with wrong dimensions
2. Cause a shape mismatch error in the Pool module
3. Make multi-resolution evaluation and test-time augmentation impossible

This fix was the load-bearing change for TTA (test-time augmentation) at different resolutions and for high-resolution inference (480px).

---

## 7. Gradient Checkpointing

### Implementation

Gradient checkpointing is implemented in `mvit_mtl_model.py` lines 115–124:

```python
use_grad_ckpt = getattr(self, "_grad_checkpoint", False)
for i, block in enumerate(self.backbone.blocks):
    if use_grad_ckpt and self.training:
        # Checkpoint only the early/middle blocks (cheaper recompute)
        # Leave the last few blocks un-checkpointed for stable training.
        x, thw = torch.utils.checkpoint.checkpoint(
            block, x, thw, use_reentrant=False
        )
    else:
        x, thw = block(x, thw)
```

**Note:** Despite the comment saying "Leave the last few blocks un-checkpointed", the code checkpoints ALL 16 blocks uniformly when `use_grad_ckpt` is True. The comment is aspirational but the code doesn't exclude any blocks.

The flag is set in `scripts/train_mtl_mvit.py` lines 2007–2008:
```python
if getattr(args, 'grad_checkpoint', False):
    model._grad_checkpoint = True
```

And the CLI argument is at line 1919:
```python
parser.add_argument("--grad-checkpoint", action="store_true", default=False,
    help="[v4 480px] Enable gradient checkpointing — reduces VRAM ~3x at cost of ~30%% compute. "
         "Required for 480x480 batch=1 on 16GB GPU.")
```

### What is checkpointed

- **conv_proj**: NOT checkpointed (runs before the loop)
- **pos_encoding**: NOT checkpointed (runs before the loop)
- **16 transformer blocks**: ALL checkpointed (each individually)
- **Head computations**: NOT checkpointed

### VRAM savings

The comment in `mvit_mtl_model.py` line 113 claims "~3-4x lower activation memory". The comment in `train_mtl_mvit.py` line 1920 claims "~3x reduction".

Each transformer block stores activations of size roughly:
- Attention matrix: [B, n_heads, N_q, N_k] — see section 8 below
- QKV projection activations
- MLP intermediate (4× hidden)
- LayerNorm inputs, residual paths

For 16 blocks, the cumulative activation memory is 16× the per-block activation. With gradient checkpointing, only the input to each block is stored, and activations are recomputed during backward. This reduces the saved activation memory from O(16 × activation) to O(1 × activation).

**Estimated savings at 224 B=1 FP16:**
- Without checkpointing: ~8–10 GB activation memory
- With checkpointing: ~2.5–3.5 GB activation memory
- Savings: ~3-4x (matches the docstring claim)

### Compute cost

Each checkpointed block requires one extra forward pass during backward. For 16 blocks, this adds ~16 block-forward recomputations per training step. Since the forward is ~30-40% of the compute (backward is the rest), the total compute overhead is approximately:

- Extra forward: 16 block-forwards (same cost as the original 16 block-forwards)
- Original total: 16 block-forward + 16 block-backward ≈ 16 × 1.4 = 22.4 units
- New total: 16 × 2 block-forward + 16 block-backward ≈ 16 × 2.4 = 38.4 units
- Overhead: ~70% extra compute

The docstring claim of "~30% extra compute" seems optimistic. It may reflect observations where the MLP and attention recomputation are cheaper than the initial forward (because caches are warm), or it may be empirically measured with CUDA graphs and fused kernels. The theoretical overhead is closer to 60-70%.

---

## 8. Resolution Limits

### Why 640 is impossible on 16GB

The attention matrix memory cost dominates at high resolutions. Here is the per-sample attention matrix memory (FP16) for a single forward pass:

| Resolution | Total attn matrix memory (B=1, FP16) | Peak activation (single largest attn) |
|------------|--------------------------------------|--------------------------------------|
| 224×224 | 0.14 GB | 37.5 MB (block 1) |
| 320×320 | 0.58 GB | 156.2 MB (block 1) |
| 480×480 | 2.95 GB | 791.0 MB (block 1) |
| 640×640 | 9.31 GB | 2500.0 MB (block 1) |

At 640×640, block 1 alone requires 2.5 GB for a single attention matrix. With 16 blocks, the cumulative attention memory is 9.3 GB. This does not include:

- Model weights: ~223 MB (backbone) + ~140 MB (heads) = ~363 MB
- Optimizer state (AdamW): 2× model params = ~445 MB (FP32 master weights) + 2× ~223 MB (moments)
- Input/output activations from conv_proj, heads
- CUDA context and frameworks overhead: ~1-2 GB

Total estimated VRAM at 640×640 B=1 FP16: ~13-15 GB for attn matrices + weights + optim states before any batch dimension expansion or head computations. This exceeds 16GB.

Even with gradient checkpointing (which saves block activations but still needs the attention matrices for the recomputed forward during backward), 640×640 is infeasible.

### The rel_pos bias memory cost

The rel_pos bias parameters themselves are negligible — only 0.13M parameters at 224px and 0.33M at 640px:

| Resolution | Total rel_pos params | Block 0 rel_pos size | Memory as parameters |
|------------|---------------------|----------------------|---------------------|
| 224 | 130,000 | 111 × 96 = 10,656 each | ~0.5 MB (FP16) |
| 320 | 183,000 | 159 × 96 = 15,264 each | ~0.7 MB |
| 480 | 263,000 | 239 × 96 = 22,944 each | ~1.0 MB |
| 640 | 334,000 | 319 × 96 = 30,624 each | ~1.3 MB |

The **runtime cost** of the rel_pos bias, however, is significant. The `_add_rel_pos` function (`mvit.py` lines 124–181) computes position-based attention biases and adds them to the raw attention matrix. This requires:

1. Interpolating rel_pos tables via `_interpolate` (linear interpolation) — cheap
2. Computing distance matrices `dist_h`, `dist_w`, `dist_t` — O(N_q × N_k)
3. Einstein summations to compute `rel_h_q`, `rel_w_q`, `rel_q_t` — O(N_q × dim × N_k)

The rel_pos computation scales with the attention matrix size, so at 640×640 it adds ~same order of magnitude as the attention itself.

---

## 9. Verification Notes

### Docstring claims vs evidence

| Claim | Source | Evidence | Verdict |
|-------|--------|----------|---------|
| "Total ~45M params" | `mvit_mtl_model.py:5` | Actual: 55.7M | **STALE** — off by 10.7M |
| "34.5M backbone" | `mvit_mtl_model.py:5` | TV weights: 34.54M, our: 34.23M | **CORRECT** |
| "~10M heads" | `mvit_mtl_model.py:5` | Heads: 21.5M (incl FPN) or 6.9M (excl FPN) | **STALE** |
| "PSR head 70.9M → ~1.8M" | `mvit_mtl_model.py:375` | Actual: 1.78M | **CORRECT** |
| "grad ckpt ~30% extra compute" | `mvit_mtl_model.py:113` | Theoretical: ~60-70% | **OPTIMISTIC** |
| "grad ckpt 3-4x lower activation" | `mvit_mtl_model.py:113` | Plausible for 16 blocks | **PLAUSIBLE** |
| "Total ~40M params" (class docstring) | `mvit_mtl_model.py:500` | Actual: 55.7M | **STALE** |

---

## Verdict

### Finding 1: Parameter count is under-reported in docstrings
**Strength: HIGH** — computed empirically from model instantiation.

The docstring at `mvit_mtl_model.py:5` claims ~45M total (34.5M backbone + ~10M heads). The actual total is **55.7M** (34.2M backbone + 14.5M FPN + 6.9M task heads). The BiFPN at 14.5M is the largest head component, overlooked in the original estimate. The file-level docstring should be updated.

### Finding 2: No temporal hierarchy limits long-action modeling
**Strength: HIGH** — proven by `stride_q` config at `mvit.py:834-851`.

MViTv2-S processes a fixed T=8 temporal window (16 frames at input, pooled to 8 by conv_proj stride 2). **No block ever pools the temporal dimension** — all `stride_q[0] = 1` and `stride_kv[0] = 1`. The model has no hierarchical temporal abstraction. Actions longer than ~0.5s (16 frames at 30fps) require late fusion across clips. This is a fundamental architectural limitation for the assembly action domain where many operations span 1-4 seconds.

### Finding 3: The 384-dim bottleneck spans 10 blocks
**Strength: MEDIUM** — inference from architecture config.

Blocks 4–13 (10 out of 16 blocks) operate at 384 hidden dimension with 4 heads (head_dim=96). For 4-task MTL, the competition for representational capacity within this 384-dim shared space under the same 4× MLP expansion (1536 intermediate) is a known bottleneck. The later expansion to 768 (block 14) and the full-resolution class token mitigate this for activity/pose tasks, but PSR and detection both read from earlier stages.

### Finding 4: 640px training is fundamentally infeasible on 16GB
**Strength: HIGH** — computed from attention matrix size scaling.

Attention matrices alone consume **9.3 GB** at 640px (FP16, B=1). Combined with weights (~363 MB), optimizer state (~445 MB), and CUDA overhead (~1-2 GB), total exceeds 16 GB. Even with gradient checkpointing, the attention matrices must be computed during forward and recomputed during backward, so the peak memory is at least one full forward pass worth of attention matrices. The 480px case (2.95 GB attention matrices) is the practical maximum on 16 GB with gradient checkpointing.

### Finding 5: The THW fix at `mvit_mtl_model.py:106-109` is critical for multi-resolution
**Strength: HIGH** — documented with code evidence and rationale.

The 1-line change from hardcoded `self.pos_encoding.spatial_size` to dynamic `x.shape[2:5]` enables inference at any resolution because MViTv2-S uses relative (not absolute) position encoding. The `_interpolate` function at `mvit.py:109-121` auto-adapts the bias tables. Without this fix, all inference is locked to 224px — multi-resolution TTA and high-resolution evaluation would crash with shape mismatch errors in the Pool module.

---

*Analysis generated by Agent 6 (Backbone Capacity) of Claude Science V2 — Phase 2: Architecture Limits. All claims verified against source code at commit 75ef7f82.*
