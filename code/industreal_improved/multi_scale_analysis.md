# Multi-Scale Training vs Alternatives for Tiny-Object Detection
## Technical Analysis for MViTv2-S + FPN

---

## 1. Literature Review: Multi-Scale in Object Detection

### 1.1 Canonical Multi-Scale Training Papers

**SNIP -- Scale Normalization for Image Pyramids** (Singh & Davis, BMVC 2018; arXiv:1711.08189)
- First systematic analysis showing CNNs are not robust to scale variation
- Key finding: training on full image pyramid (480-800px) with *selective backpropagation* -- gradients for objects that are "too small" or "too large" for a given scale are suppressed
- COCO single-model: 45.7% mAP; ensemble: 48.3% mAP
- Won Best Student Entry in COCO 2017 challenge
- Core mechanism: for each scale, only backprop through object instances whose size is appropriate for that scale. This prevents the detector from learning to ignore small objects during training on low-resolution crops
- Cost: full image pyramid at training = 3-5x compute per iteration

**SNIPER -- Efficient Multi-Scale Training** (Singh, Najibi, Davis, NeurIPS 2018; arXiv:1805.09300)
- Major efficiency improvement: instead of processing full-scale images, process only *chips* (context regions around ground-truth instances at appropriate scales)
- 46.1% mAP on COCO at roughly same compute cost as single-scale training
- 3x faster than full SNIP with comparable accuracy
- Key insight: you don't need the full image pyramid -- only need to render objects at appropriate scales with local context

**AutoFocus -- Efficient Multi-Scale Inference** (Najibi, Singh, Davis, ICCV 2019; arXiv:1812.01600)
- Addresses inference cost: uses a cheap "attention module" to identify spatial regions where high-resolution processing is needed
- 2-5x speedup over full image pyramid with negligible mAP loss
- Demonstrates that most of the image can be processed at low resolution -- only sparse regions need high-res

**Scale-Aware Trident Networks (TridentNet)** (Li et al., ICCV 2019; arXiv:1901.01892)
- Parallel branches with shared weights but different dilation rates, each specialized for a scale
- 48.4% mAP on COCO
- Elegant because weights are shared -- no parameter count increase
- Training cost: 3x per-iteration cost (three parallel branches), but weights shared so convergence is faster

**Feature Pyramid Networks (FPN)** (Lin et al., CVPR 2017; arXiv:1612.03144)
- Built-in multi-scale feature hierarchy: top-down pathway with lateral connections
- This is already standard in your detection head -- so you already have some multi-scale processing
- FPN helps with scale variation but does NOT solve the "vanishing gradient for tiny objects at low resolution" problem -- the backbone features entering FPN are still computed at native resolution

### 1.2 YOLO Multi-Scale Approach

YOLOv4 (arXiv:2004.10934) introduced Mosaic augmentation and multi-scale training:
- Input size randomized per batch from {320, 352, ..., 640} (in 32px strides)
- Combined with Mosaic: mix of 4 images per batch at randomized scales
- This *does* improve robustness to scale variation but is primarily a regularization technique -- it doesn't specifically target tiny-object recall at the feature level

YOLOv8 (Ultralytics, 2023):
- Default multi-scale: random scale factor in [0.5, 1.5] per batch
- The key difference: YOLO is a fully convolutional detector at native resolution -- it doesn't use FPN or attention in the same way MViT does
- Multi-scale training in YOLO works because it effectively augments the training distribution

---

## 2. Compute Overhead Analysis (MViTv2-S + FPN)

Baseline numbers for MViTv2-S (22.7M params, ~27.5 GFLOPs at 224x224 input, scaling quadratically with resolution):

| Method | Training FLOPs/iter | Inference FLOPs | Memory Overhead | Implementation Effort |
|---|---|---|---|---|
| Baseline (640px) | 224 GFLOPs | 224 GFLOPs | 1x | None |
| **Multi-scale train** (0.5-1.25x) | 189 GFLOPs avg (ranges 56-351) | same as baseline | 2x peak (for 800px batches) | Low (config change) |
| **Progressive resizing** (224->448) | 60 GFLOPs avg (28 early, 110 late) | 224 GFLOPs | 0.5x peak | Low (scheduler change) |
| **TTA at eval only** | 224 GFLOPs (unchanged) | 1120 GFLOPs (5 scales) | 1x train, 3-5x eval | Low |
| **SR preprocessor** (SwinIR) | 234 GFLOPs | 234 GFLOPs | 2x (stored upscaled) | High (new component) |
| **Decoupled backbone/head** | 93 GFLOPs | 93 GFLOPs | 0.5x | High (arch change) |

### Detailed Breakdown

#### 2.1 Multi-Scale Training (per-batch randomization)
- **Scales**: {0.5, 0.75, 1.0, 1.25} x 640 = {320, 480, 640, 800}px
- **Per-iteration cost range**: 56 - 351 GFLOPs
- **Average cost**: 189 GFLOPs (15% lower than fixed 640px! but this is misleading)
- **Real cost**: GPU utilization is poor at 320px (bottlenecked by data loading and kernel launch overhead). At 800px, OOM risk is real if batch size isn't adjusted
- **Memory**: Must provision for worst case (800px), so effective memory footprint is 2x
- **Implementation**: Trivial in most training frameworks (pass random size to data loader)

#### 2.2 Progressive Resizing
- **Phase 1 (224px)**: 27.5 GFLOPs for ~60% of training
- **Phase 2 (448px)**: 110 GFLOPs for ~40% of training
- **Average**: ~60 GFLOPs (73% reduction vs baseline)
- **Benefit**: Much faster iterations early on, so you can do more total iterations in the same wall time
- **Risk**: The distribution shift when switching resolutions can cause a "performance cliff" if not managed carefully (gradual transition, cosine schedule reset)
- **For tiny objects**: 224px inputs make already-small objects even smaller -- this directly contradicts the goal of improving tiny-object detection

#### 2.3 TTA at Eval Only
- **Training cost**: Zero. No changes needed
- **Inference cost**: Nx where N = number of test scales. For {0.6, 0.8, 1.0, 1.2, 1.5} = 5x
- **Optimization**: With NMS-based scale fusion, practical overhead is ~2-3x (batch all scales, fuse outputs)
- **mAP gain**: Typically +1-3% mAP on COCO for detectors that didn't train multi-scale. The gain is largest for small objects
- **Caveat**: TTA improves *detection* of objects that *already have signal* at the base scale. If tiny objects are below the feature resolution floor, TTA won't help (no signal to amplify)

#### 2.4 Super-Resolution Preprocessor (SwinIR)
- **SwinIR-lightweight**: ~900K params, ~10 GFLOPs for 224x224 -> 448x448 upscaling
- **Full chain**: 224px input -> SR network (10 GFLOPS) -> 448px -> detector (110 GFLOPs) = 120 GFLOPs total
- **Alternative chain**: 320px input -> SR network (15 GFLOPS) -> 640px -> detector (224 GFLOPs) = 239 GFLOPs total
- **Memory**: 2x for storing full-res SR output
- **Training complexity**: Two-stage (train SR, freeze, train detector) or end-to-end (joint, harder). SR network needs paired low-res/high-res data
- **Risk**: SR artifacts can confuse the detector. GAN-based SR (ESRGAN) can hallucinate textures that don't exist, causing false positives

#### 2.5 Decoupled Resolution (backbone@224, head@640)
- **Intuition**: Run the expensive backbone at low resolution, then upsample feature maps to high resolution before the detection head
- **Cost**: ~93 GFLOPs (59% reduction vs baseline)
- **Problem**: Backbone features at 224px have token spacing of 7px (after 32x stride). Even with upsampling, fine spatial information is already lost. Tiny objects (say 10x10px in the original 640px image) are 3.5x3.5px in the 224px feature space -- essentially a single token.
- **Literature**: This approach appears in some efficient detection papers but universally underperforms for small objects

---

## 3. Expected mAP Gain (One-Month Effort Budget)

I categorize these by the Pareto frontier of accuracy gain vs implementation cost:

| Method | Expected mAP gain (small objects) | Expected mAP gain (overall) | Confidence | Effort |
|---|---|---|---|---|
| **TTA at eval** | +1.5 to 3.0 | +1.0 to 2.0 | High | 1-2 days |
| **Multi-scale train (YOLO-style)** | +0.5 to 1.5 | +0.5 to 1.0 | Medium | 2-3 days |
| **Progressive resizing** | -0.5 to +0.5 | 0.0 to +0.5 | Low | 2-3 days |
| **SR preprocessor (frozen)** | +1.0 to 2.0 | +0.5 to 1.5 | Medium | 1-2 weeks |
| **Decoupled resolution** | -2.0 to -5.0 | -1.0 to -3.0 | High (negative) | 1-2 weeks |
| **SNIP-style selective BP** | +2.0 to 4.0 | +1.5 to 3.0 | Medium-High | 2-3 weeks |
| **TridentNet-style branches** | +2.0 to 4.0 | +1.5 to 2.5 | Medium | 3-4 weeks |

### Why These Numbers?

1. **TTA gets most of the gain**: Wang et al. (2019, "A Closer Look at TTA") showed that simple multi-scale TTA recovers 60-80% of the gain from full multi-scale training. For detectors that already use FPN, TTA amplifies existing scale coverage.

2. **SNIP/SNIPER has the highest ceiling**: The Singh-Davis line of work directly addresses the root cause -- objects becoming "invalid" at certain scales. The gradient suppression mechanism prevents the network from learning harmful gradients from objects it can't possibly detect at a given scale. This is the cleanest theoretical approach.

3. **Progressive resizing is negative for tiny objects**: Starting at 224px means tiny objects in your training data become impossibly small (a 10px object at 640px is 3.5px at 224px -- essentially a point). Early training iterations learn to ignore them entirely. When resolution doubles, the network must "unlearn" this behavior.

4. **Super-resolution helps but is fragile**: The literature shows consistent gains on remote sensing benchmarks (where objects are genuinely pixel-level), but on natural images the gains are modest. SwinIR on COCO small-object detection: +1.1 AP_s in the best case (Bai et al., 2022). The risk of SR artifacts causing false positives is non-trivial.

---

## 4. Deep Dive: Why Vanishing Gradients at Low Resolution?

The core problem: **feature stride vs object size**.

MViTv2-S at 640px input:
- Patch size: 4x4 at stem, then 2x2 pooling in later stages
- Effective stride at P5: 32px
- A 16x16 pixel object in the input occupies... 0.5 tokens at P5. It's subpixel.

When you train at 320px input:
- Same 16x16 object is now 8x8 pixels in the input
- At P5 (32x stride): 0.25 tokens. It's gone.
- The gradient for this object at P5 is exactly zero. It can never be detected.

SNIP's insight: if you train at 320px, the gradient for this object is not just useless, it's harmful -- it teaches the network that "this object doesn't exist" at P5. Selective gradient suppression prevents this.

MViTv2-S with FPN partially mitigates this because FPN's P3 operates at 8x stride, but the features feeding P3 still come from a backbone that pooled at the full 640px resolution. If the backbone runs at 320px, P3 features are at 40px stride (320/8=40) -- still losing 16px objects.

---

## 5. Recommendation

Given a **one-month effort budget** and the goal of improving tiny-object detection, ranked by ROI:

### Tier 1: Do Now (1-2 days, zero training cost)
**Add TTA at evaluation time** with 3 scales {0.8, 1.0, 1.2}. This gives you +1-2 mAP on small objects with no training changes. Implement scale-aware NMS fusion (take max confidence across scales, box averaging for matched detections). This is the classic "free lunch."

### Tier 2: If you can tolerate training changes (1 week)
**Implement SNIP-style selective gradient masking** during training. This is higher priority than multi-scale randomization because it directly addresses the vanishing gradient problem. The implementation is straightforward:
- During forward pass, compute object scale relative to input size
- If object size/input_size exceeds valid range (e.g., [0.06, 0.6] of image), mask its loss contribution to zero
- This requires access to ground-truth box sizes during training, which you already have

### Tier 3: If you have budget for architectural changes (2-3 weeks)
**Add a dedicated tiny-object detection head** operating on P2 features (4x stride) from FPN. This requires:
- Additional upsampling in FPN to produce P2
- A lightweight detection head (2-3 conv layers) at P2
- Training with higher-resolution inputs (800px) for the P2 head
- This is the approach used by recent works like SAHI and TPH-YOLOv5

### Tier 4: Do NOT do (waste of budget)
- **Progressive resizing**: Will hurt tiny-object performance, not help
- **Decoupled backbone/head**: Loses fine detail permanently
- **GAN-based super-resolution**: Unstable training, hallucination risk, marginal gain on natural images

### What About Waiting?

If the 1-month budget gets pushed to 3 months, the ideal pipeline is:
1. Week 1: TTA (immediate gain)
2. Week 2-3: SNIP-style selective gradient masking
3. Week 4-8: Train dedicated P2 head with 800px crops
4. Week 9-12: AutoFocus-style selective high-resolution inference

This avoids the "throw more resolution at it" trap -- the key innovation in the Singh-Davis line of work is that **more resolution without scale-conditional gradient handling actively harms performance**.
