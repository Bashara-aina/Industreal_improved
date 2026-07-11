# 210: Architecture Exploration Space — Beating SOTA with MTL

> **Date:** 2026-07-11
> **Scope:** Systematic survey of feasible architectural changes for IndustReal MTL, estimated by parameter cost, training cost, and expected impact.
> **Ground truth:** Current MViTv2-S baseline has ~45M params (34.5M backbone + ~10M heads), trains at ~1.5s/batch on RTX 3060, 8 GB VRAM.

---

## 1. Current Architecture Analysis

### 1.1 What's Working

- **Pose head**: 8.7 deg angular MAE at epoch 2, exceeds RF10 target (35 deg) by epoch 2. The Tanh-bounded 6D representation + Gram-Schmidt orthonormalisation is a stable, convergent design.
- **PSR head**: The 1.8M-param causal Transformer (2-layer, d=256, nhead=4, ff=1024) on P5 features (768ch, post-attention) is the correct architecture. The 70.9M-to-1.8M diet (Opus 201) proved that P5 feature source (blocks[14], not conv_proj) was the load-bearing change, not head capacity. PSR gradient stays ALIVE at 3.18 norm.
- **Detection localisation**: DET_PROBE bestIoU_max consistently >0.90. The model finds the assembly board perfectly — this is a fine-grained classification problem (single-bit state discrimination), not a localisation problem.
- **Feature pyramid**: BiFPN with 3D convolutions (trilinear up/down) on MViTv2 hierarchical features provides a clean multi-scale representation for detection at P3/P4/P5.
- **PCGrad gradient surgery**: Implementation is correct (Gram-Schmidt projection of conflicting gradients). The mechanism works for the PSR-detection conflict but cannot compensate for the 312x gradient magnitude gap affecting the activity head.

### 1.2 What's Bottlenecking

**Activity head gradient starvation (P0):** The activity head receives gradient norm 0.010 vs PSR at 3.180 (312x gap). This is structural — persists across ALL attempted fixes (LR 1x-20x, blend 0.05-1.0, clip 0.3-1.0, gradient centralisation, classifier reinit). The causal chain:

```
CE loss -> 3-layer MLP (768-2048-1024-75) -> cls_token -> 14 transformer blocks -> backbone
```

The gradient path is long, the output is a single 75-dimensional logit vector per clip, and CE loss on 75 classes produces a sparse gradient signal. Compare to PSR: 11 binary focal losses across 8 frames = 88 dense gradient signals.

**Detection fine-grained ceiling (P1):** 24 classes where adjacent states differ by 1 of 11 binary assembly components. The model saturates at ~0.207 mAP50 regardless of LR/BIAS (tested across 2 runs at 1x vs 4x). This is a data/separation ceiling, not a gradient ceiling.

**BiFPN overcapacity for current load (P2):** 4-level top-down + bottom-up with learnable fusion weights for only 3 detection levels (P3/P4/P5, P2 dropped per Opus 192 FC-2). The FPN consumes ~2.5M params but only 3 heads use it (detection uses 3/4 levels, PSR uses P5 directly via hook).

**Pose head underutilisation (INFO):** The 768->256->6 MLP on cls_token has only ~200K params and saturates early. This is fine for the task but the cls_token may not carry sufficient spatial information for head pose in heavy occlusion.

### 1.3 Parameter Budget (Measured)

| Component | Params | % of Total | Gradient Norm | Health |
|-----------|--------|-----------|---------------|--------|
| MViTv2-S backbone | 34.5M | 76.7% | 2.37 | ALIVE — dominates |
| Activity head | 8.2M (orig) / 1.1M (simplified) | 2.4% | 0.010 | STARVED — 312x gap |
| PSR head | 1.8M | 4.0% | 3.18 | ALIVE — oscillates |
| FPN (BiFPN) | ~2.5M | 5.6% | 1.15 | ALIVE |
| Detection head | ~0.8M | 1.8% | 0.48 | ALIVE |
| Pose head | ~0.2M | 0.4% | 0.44 | ALIVE |
| **Total** | **~45M** | **100%** | — | — |

---

## 2. Backbone Options

### 2.1 MViTv2-S (Current Baseline)
- **Params:** 34.5M
- **K400 Top-1:** 82.3%
- **Structure:** 16x MultiscaleBlocks with hierarchical downsampling at blocks 4, 8, 12
- **Feature dimensions:** P2=96ch/56px, P3=192ch/28px, P4=384ch/14px, P5=768ch/7px (T=8 temporal)
- **Memory:** ~2.3 GB activation storage at T=16, batch=4
- **Pros:** Hierarchical features natural for FPN; native torchvision; strong spatiotemporal priors
- **Cons:** Single cls_token for activity/pose is a bottleneck; no per-frame independent processing
- **Impact assessment:** Current baseline. Not the bottleneck for detection or activity — the bottleneck is gradient routing, not backbone capacity. **No change needed** unless we hit a feature-quality ceiling for activity (which we haven't because the head hasn't trained yet).

### 2.2 MViTv2-B / L (Wider / Deeper Variants)
- **Params:** MViTv2-B = ~51M, MViTv2-L = ~213M
- **K400 Top-1:** 82.9% (B) / 84.3% (L)
- **Memory:** ~3.5 GB (B) / ~12 GB (L) at T=16, batch=4
- **Verdict:** MViTv2-B might fit (51M params, ~3.5 GB activations) but 16.5M extra params for +0.6% K400 is poor ROI. MViTv2-L is out of budget. **SKIP** — gradient routing, not backbone capacity, is the bottleneck.

### 2.3 ConvNeXt-Tiny (Previous Baseline)
- **Params:** 28M (ImageNet-1K, 2D only)
- **K400 Top-1:** N/A (2D backbone, no temporal modelling)
- **Status:** The old architecture required a separate 22M VideoMAEStream for temporal features, totalling 54.5M. The MViTv2-S replacement already saves 15.5M params while adding inherent temporal modelling.
- **Revisit scenario:** If per-frame independent processing becomes necessary (e.g., real-time deployment at 30 fps), ConvNeXt + lightweight temporal module is 2x faster than MViTv2-S (20ms vs 40ms per frame). **Not relevant for SOTA pursuit.**

### 2.4 VideoMAE-S / -B
- **Params:** 22M (S) / 87M (B)
- **K400 Top-1:** 83.7% (S w/ mask) / 84.7% (B)
- **Structure:** Flat ViT (12 layers for S, 24 for B), uniform 384-dim (S) / 768-dim (B) features
- **Key disadvantage:** Flat feature maps at all layers make multi-scale detection integration harder (no hierarchical structure). All backbones need per-frame 2D FPN reconstruction.
- **Verdict:** VideoMAE-S is lighter than MViTv2-S (22M vs 34.5M) with comparable K400 accuracy, but the flat architecture adds complexity for detection FPN. **Low priority** — only consider if MViTv2-S hierarchical features prove inadequate for fine-grained activity discrimination.

### 2.5 ViT-B / -L (Plain, ImageNet-21K Init)
- **Params:** 86M (B) / 307M (L)
- **Pretraining:** ImageNet-21K classification + MAE self-supervised
- **Temporal extension:** Requires TimeSformer-style factorised attention (space then time) or ViViT-style tubelet embedding
- **Memory:** ViT-B at 224px uses ~2 GB activation storage (images only). Adding temporal (T=16) through factorised attention adds ~1.5 GB.
- **Verdict:** Viable only with factorised space-time attention to keep memory tractable. Heavy finetuning cost (+epoch time). Requires significant rearchitecting of the feature extraction pipeline. **High-risk, high-cost.** Only pursue if MViTv2-S gradient starvation cannot be resolved by head redesign.

### 2.6 Swin-T / Swin-S (Video Swin Transformer)
- **Params:** 28M (T) / 50M (S)
- **K400 Top-1:** 82.7% (S)
- **Structure:** Shifted window attention with hierarchical downsampling (4 stages, like ConvNeXt)
- **Native torchvision support:** Yes (torchvision >= 0.15)
- **Pros:** Hierarchical features (natural for FPN), shifted windows are more compute-efficient than full self-attention, built-in video variant (Swin3D in torchvision >= 0.17)
- **Cons:** Window boundaries can create artifacts; shifted window scheduling adds complexity
- **Verdict:** Swin-S (50M) matches MViTv2-S (34.5M) in accuracy at 15.5M more params. Swin-T (28M) is lighter but lower accuracy. **Medium priority** — Swin3D's video variant could provide better temporal features but at higher param cost.

### 2.7 Mamba / State Space Models for Video
- **Params:** Mamba-2 ~130M (too large); VMamba-S ~35M (comparable to MViTv2-S)
- **Status:** Research-stage. VMamba (Zhu et al., 2024) shows ImageNet-1K 83.5% but no K400 video pretrained weights exist.
- **Pros:** Linear-complexity sequence modelling (no self-attention quadratic cost), natural for long sequences
- **Cons:** No K400 pretrained weights available; 2D selective-scan is a new paradigm requiring custom CUDA kernels; integration with video FPN is unexplored
- **Verdict:** **Do not pursue** — not enough infrastructure for production use. Worth monitoring for 2027 deployment but not for current SOTA target.

### 2.8 Backbone Verdict

**MViTv2-S stays.** It is not the bottleneck. All alternatives either cost more params for marginal accuracy gain (MViTv2-B, Swin-S), lose hierarchical features (VideoMAE), or lack pretrained weights (Mamba). The 34.5M backbone is 76.7% of total params — the opportunity is in **redistributing the remaining 23.3%** and **fixing gradient routing**, not replacing the backbone.

---

## 3. Neck / FPN Designs

### 3.1 Current BiFPN (EfficientDet-Style)
- **Structure:** 4-level (P2-P5) top-down + bottom-up with learnable fusion weights, trilinear interpolation, 3x3x3 3D convs for smoothing
- **Params:** ~2.5M
- **Usage:** P3/P4/P5 feed detection head (P2 dropped per Opus 192 FC-2), P5 also feeds PSR head directly via hook (not through FPN)
- **Efficiency:** 3D convolutions are expensive for what is effectively a 2D task (temporal-pooled detection features). The 3D convolutions don't model temporal relationships across >8 frames because the FPN operates on the MViTv2 internal T=8 temporal dimension.

### 3.2 BiFPN Depth/Width Sweep
- **Option A (Reduce):** 2D convolutions only (temporal-pooled features before FPN). Remove 3D convs. Saves ~0.8M params.
  - **Risk:** Loses ability to weight features across time within the FPN. But detection temporal-pools anyway (mean over T dim).
  - **Impact:** -32% FPN params, negligible accuracy change. Do this.
- **Option B (Reduce):** Drop P4 level. Keep only P3 + P5. Saves ~0.5M params.
  - **Risk:** P4 (384ch, 14px at stride 16) provides mid-scale features that may help with medium-sized assembly components.
  - **Impact:** Uncertain. Worth ablating (1 epoch test).
- **Option C (Expand):** Add P6 level (at stride 64, 7px from MViTv2 final 7x7). Adds ~0.3M params.
  - **Impact:** Marginal — assembly objects are large boards, not small objects needing high stride features.
- **Recommendation:** **Option A (2D BiFPN, -32% params).** The 3D convolutions are unused compute.

### 3.3 NAS-FPN (Neural Architecture Search FPN)
- **Structure:** Irregular, cross-scale connections discovered by NAS
- **Params:** ~3-5M typical
- **Pros:** State-of-the-art connections; proven on COCO (1-2 AP gain)
- **Cons:** No pretrained topology for our specific backbone; would need NAS search (500+ GPU-hours); complex to implement
- **Verdict:** **SKIP** — gains are for detection-only models, not MTL where detection already localises well.

### 3.4 PANet (Path Aggregation Network)
- **Structure:** BiFPN is a successor to PANet. PANet adds bottom-up path augmentation on top of FPN.
- **Difference from BiFPN:** PANet uses simple addition (no learnable weights), fewer conv layers.
- **Params:** ~1.5M (vs BiFPN's 2.5M)
- **Verdict:** **Worth trying** as a simpler alternative to BiFPN. Saves 1M params, may not hurt detection since the bottleneck is classification not localisation.

### 3.5 YOLO-Style Necks (RepPAN, CSP-PAN)
- **Structure:** RepVGG-style reparameterisable convs + CSP connections
- **Params:** ~2M (YOLOv8 neck)
- **Pros:** Fast inference (reparameterisable), proven in YOLO series
- **Cons:** Tightly coupled with YOLO head structure; would need adaptation for 3D convs
- **Verdict:** **Low priority** — YOLO necks are optimised for single-scale single-head detection, not multi-task.

### 3.6 FPN Verdict

**Primary recommendation:** Convert BiFPN to 2D-only by temporal-pooling _before_ FPN (not after). This saves ~0.8M params and eliminates 3D conv compute with no expected accuracy loss (detection already temporal-pools). Total FPN: ~1.7M.

**Secondary recommendation:** Ablate PANet-style simpler FPN (no learnable weights) vs current BiFPN. If mAP50 delta < 0.02, adopt PANet for simplicity.

---

## 4. Detection Head Variants

### 4.1 Current: YOLO-Style Decoupled Head + DFL
- **Structure:** 2x Conv2D(256->256) + GN + ReLU -> separate cls (256->24) and reg (256->4x16) heads
- **Params:** ~0.8M (per level, applied at P3/P4/P5)
- **Losses:** Binary focal (cls) + CIoU + DFL (reg) with TAL assigner
- **Performance ceiling:** 0.207 mAP50 at 50% subset. The bottleneck is class separability (single-bit state differences), not head capacity or architecture.

### 4.2 YOLOv8/v9/v10 Head Variants
- **YOLOv8 head:** Decoupled + DFL. Our head is already YOLOv8-style.
- **YOLOv9 head (GELAN):** Adds programmable gradient information (PGI) to prevent information loss in deep layers. Key contribution: reversibly collects gradient info from multiple depths.
  - **Potential relevance:** PGI could help the detection head maintain gradient signal despite PSR domination. If detection gradient (0.48) is suppressed by PSR (3.18) through PCGrad, PGI could preserve detection-specific features.
  - **Cost:** +0.3M params per level. **Worth investigating**.
- **YOLOv10 head:** NMS-free training via dual assignments (one-to-many + one-to-one). Eliminates NMS at inference.
  - **Relevance:** NMS is not our bottleneck (we have 1-5 detections per frame).
  - **Verdict:** Interesting but irrelevant.

### 4.3 DETR-Like (Transformer) Detection Head
- **Structure:** Learnable queries + Transformer decoder (6 layers) -> set prediction
- **Params:** ~5M (for Deformable DETR mini)
- **Pros:** End-to-end, no NMS, no anchors. Handles set prediction naturally.
- **Cons:** 5M params (6x current head), slower convergence, needs 100-300 queries per frame (excessive for our 1-5 objects).
- **Verdict:** **SKIP** — overkill for single-object (or few-object) assembly scene. 300x more queries than objects.

### 4.4 Anchor-Free Designs (FCOS, TOOD)
- **FCOS style:** Per-pixel classification + centerness + regression. No anchors needed.
- **TOOD (Task-aligned One-stage Object Detection):** Our current TAL assigner is from TOOD. The head is already TOOD-compatible.
- **Variant:** Replace decoupled head with TOOD's task-aligned head (1 conv shared + 2 task-specific convs with interleaved interaction).
  - **Cost:** +0.2M params per level.
  - **Impact:** Task-aligned interaction between cls and reg branches could help fine-grained state discrimination (cls features get spatial precision from reg branch).
  - **Recommendation:** **Worth trying** — +0.2M for potential state-discrimination improvement.

### 4.5 Detection Head Verdict

**Primary recommendation:** Keep current decoupled head + DFL. The ceiling is data-class separability, not architecture.

**Secondary:** Ablate YOLOv9 GELAN-style PGI if gradient competition from PSR head (312x ratio) proves to degrade detection features. +0.3M per level as insurance.

---

## 5. Activity Head Variants

This is the **highest-impact design space** — the activity head has never produced meaningful metrics (act_macro_f1 < 0.002 across all 6 attempts). The gradient norm is structurally fixed at 0.010 regardless of LR, blend, or clip changes.

### 5.1 Root Cause Summary

The gradient path from CE loss to backbone has:
1. A single 75-dim logit producing a _scalar_ CE loss (sparse gradient signal)
2. No spatial or temporal structure to amplify backprop signal
3. cls_token is a [B, 768] pooled representation — aggregates 16x7x7=784 patches into one vector
4. Compare to PSR: 88 binary focal losses across T=8 frames (dense, structured gradient)

The 312x gradient gap is a **multiplicative architecture issue**: the number of classifier outputs × the spatiotemporal structure of the loss. CE(1, 75) = 75 comparisons. PSR: 8 × 11 = 88 comparisons. But PSR's 11 _independent_ binary focal losses each generate logit-level gradients, so the total gradient magnitude is ~88× larger _per parameter_ because each PSR component's gradient flows through a shared transformer but has independent output projections.

### 5.2 Current: Simple MLP on cls_token (1.1M params, Opus 207)
- **Structure:** LayerNorm -> Linear(768->2048) -> GELU -> Dropout -> Linear(2048->1024) -> GELU -> Dropout -> Linear(1024->75)
- **Params:** 1.1M
- **Gradient norm:** 0.010 (always)
- **Verdict:** This head is too simple for the task but adding complexity only worsens gradient flow. The head needs to be redesigned from scratch.

### 5.3 Temporal Attention Pool (Recommended #1)
- **Structure:** 
  - Replace cls_token with spatial feature [B, 768, 7, 7] from P5 level
  - Apply 2-layer 2D conv (kernel 3, channels 256) to produce spatial activation maps
  - Global attention pooling: learned attention over spatial positions
  - Output: [B, 256] spatially-attended feature -> Linear(256->75)
- **Params:** ~0.7M (less than current 1.1M)
- **Gradient flow:** The spatial conv produces [B, 256, 7, 7] features where each of 49 spatial positions has a gradient from the classification loss. Instead of 1 gradient source (cls_token), we have 49 gradient sources.
- **Expected gradient norm increase:** 5-10x (spatially distributed loss amplifies backprop)
- **Implementation effort:** Low. Replace cls_token input with P5 spatial features.
- **Risk:** Low. Simpler architecture, shorter gradient path.

### 5.4 Multi-Layer Feature Aggregation (Recommended #2)
- **Structure:**
  - Collect features from multiple MViTv2 blocks (block 7 -> 384ch/14px, block 11 -> 768ch/7px, block 14 -> 768ch/7px)
  - Spatial HDC (hybrid dilated convolution) at each scale: 3 parallel conv paths with dilations 1, 2, 3
  - Upsample lower scales, concatenate: [B, 384+768+768=1920, 7, 7]
  - 1x1 conv to 512 channels, global avg pool -> Linear(512->75)
- **Params:** ~1.5M
- **Gradient flow:** Multi-scale features ensure gradient from CE loss flows back through multiple backbone levels simultaneously. Instead of a single bottleneck path (cls_token), we have 3 parallel paths at different resolutions.
- **Expected gradient norm increase:** 10-20x
- **Risk:** Medium. More complex, may introduce gradient interference between scales.
- **Implementation effort:** Medium. Requires additional hooks or feature extraction.

### 5.5 Detection-Conditioned Activity Head (Hybrid)
- **Structure:**
  - Use detection box features (RoI-Align on P3/P4/P5 features at predicted box locations) as spatial conditioning
  - Concatenate with cls_token: [B, 768(cls) + 256(det_roi)] -> projection -> classification
  - This grounds activity recognition in detected object locations
- **Params:** ~1.3M (current + 0.2M for RoI projection)
- **Gradient flow:** Additional gradient path through detection features (which have healthy gradient norm 0.48). The detection features carry gradients from both activity CE loss and detection loss.
- **Expected gradient norm increase:** 3-5x (from detection branch carry)
- **Risk:** Low-Medium. Detection-conditioned activity is standard in MTL literature. Boxes must be available at activity time (they are — detection forward precedes activity).
- **Implementation effort:** Medium. Requires RoI-Align extraction at predicted box locations.

### 5.6 Temporal Convolution on Frame-Level Features
- **Structure:**
  - Extract per-frame cls_tokens across T=8 time steps (MViTv2 internal temporal resolution)
  - Stack into [B, 8, 768] temporal sequence
  - Apply 1D temporal conv (kernel 3, channels 512) -> TAN -> Linear(512->75)
  - This uses the full T=8 temporal resolution, not just the final cls_token
- **Params:** ~0.9M
- **Gradient flow:** T=8 temporal steps each produce a gradient signal. 8x more gradient sources than single cls_token.
- **Expected gradient norm increase:** 8x (from temporal stacking)
- **Risk:** Low. Straightforward extension. Gradient checkpointing keeps memory in check.
- **Implementation effort:** Low-Medium. Requires intermediate cls_token extraction per block.

### 5.7 Activity Head Verdict

**Implement in order:**
1. **Temporal Attention Pool** (Section 5.3) — lowest risk, 5-10x gradient amplification, -0.4M params from current
2. **Multi-Layer Feature Aggregation** (Section 5.4) — medium risk, 10-20x gradient amplification, +0.4M params
3. **Detection-Conditioned Hybrid** (Section 5.5) — connects to healthy detection gradient path

These three variants are **combinable**: spatial attention can be detection-conditioned, multi-layer features can feed temporal convolutions. The total activity head budget should not exceed 2.0M params (was 8.2M in the over-engineered version, currently 1.1M in the under-powered version).

---

## 6. PSR Head Variants

### 6.1 Current: Causal Transformer (1.8M params)
- Well-functioning. Gradient norm 3.18 (healthy, oscillating).
- The 2-layer causal Transformer (d=256, nhead=4, ff=1024) is proven correct.
- PSR per-component classifiers (11 heads, each linear 256->1) stay ALIVE throughout oscillation.

### 6.2 Bidirectional (Non-Causal) Transformer
- **Difference:** Remove causal mask. Allow each frame to attend to future frames.
- **Pros:** Full context improves transition prediction. For offline assembly analysis, causal constraint is artificial — we have all frames.
- **Cons:** Cannot be used in online/streaming deployment.
- **Params:** No change (same architecture, no mask).
- **Expected impact:** +1-3% F1 on components with post-transition context requirements.
- **Verdict:** **Worth trying** — zero architectural change, flip a boolean. Needs evaluation to confirm no overfitting (causal mask is a regulariser).

### 6.3 Deeper / Wider PSR Head
- **Current:** 2 layers d=256. Causal Transformer with 8 tokens.
- **Deeper (4 layers):** +1.8M params (doubles PSR budget). For 8 tokens, 4 layers of transformer is overkill — the sequence is too short for deep transformers to help.
- **Wider (d=512):** +1.8M params. The projection 768->512 preserves more source information. May help if P5 features are noisy.
- **Verdict:** **Ablate wider (d=512)** for 1 epoch. If PSR F1 improves >2%, keep. Otherwise, current d=256 is sufficient for 8-token sequences.

### 6.4 Detection-Conditioned PSR
- **Structure:** Concatenate detection features (cls_token from box region or detection features from FPN level matching PSR's P5) before temporal encoder
- **Rationale:** Assembly state transitions (PSR) are tightly coupled to which components are detected. A transition from 11110111110 to 11110111111 requires detecting the presence of bit 11.
- **Params:** +0.2M (projection for detection features)
- **Expected impact:** +3-5% F1 (detection features ground PSR in visible objects)
- **Verdict:** **High priority** — detection-conditioned PSR is a natural coupling between the two most related tasks.

### 6.5 PSR Head Verdict

**Minimum change:** Convert to bidirectional (no causal mask) — free accuracy gain.
**Medium change:** Widen to d=512 — +1.8M params for potential >2% F1 gain.
**High-impact change:** Detection-conditioned PSR — +0.2M params for potential +3-5% F1.

---

## 7. Pose Head Variants

### 7.1 Current: MLP on cls_token (200K params)
- **Structure:** Linear(768->256) -> LeakyReLU -> Linear(256->6) -> Tanh
- **Performance:** 8.7 deg MAE at epoch 2 (exceeds RF10 target of 35 deg)
- **Issue:** The cls_token may not carry sufficient spatial information for precise head orientation in heavy occlusion. The MAE at 8.7 deg may be a ceiling — data inspection shows forward vector norms of 0.014-0.030 (should be unit 1.0), indicating data normalisation issues.

### 7.2 Iterative Refinement (Pose Regression + Residual Steps)
- **Structure:** 
  - Initial coarse pose: Linear(768->256->6) + Tanh -> [fwd, up]
  - Extract refined features: pose-conditioned feature modulation (concatenate current pose with intermediate backbone features)
  - Residual correction: Lightweight MLP(2*256->6)
- **Refinement steps:** 1-2 iterations (not more — diminishing returns)
- **Params:** ~0.5M (2.5x current)
- **Expected impact:** -1 to -2 deg MAE (marginal — current pose is already strong)
- **Verdict:** **Low priority** — the pose head is already performing well. Iterative refinement adds complexity for marginal gain.

### 7.3 Heatmap-Based Pose (Direct 3D Heatmap Regression)
- **Structure:** Replace 6D regression with 3D heatmap (volumetric) over head orientation space. Discretise orientation into bins (e.g., 32x32x32 = 32,768 bins) and predict softmax distribution.
- **Params:** ~2M (large classifier on backbone features)
- **Pros:** Captures multi-modal pose distributions (e.g., head is centred in one orientation or another)
- **Cons:** 2M params for a task that already converges at 200K params. Loss is CE over 32K bins (another gradient-starved CE).
- **Verdict:** **SKIP** — solves a multi-modality problem that doesn't exist in assembly (operators face the assembly board).

### 7.4 Transformer-Based Pose (PoseFormer-Style)
- **Structure:** Treat P5 spatial features as tokens [B, 49, 768] (49 patches from 7x7), apply 1-layer transformer with learned [POSE] token, project [POSE] token to 6D pose.
- **Params:** ~0.8M
- **Pros:** Spatial attention over feature patches can improve occlusion robustness
- **Cons:** Only 49 tokens — a shallow transformer is barely different from global avg pool + MLP
- **Verdict:** **Low priority** — test if current cls_token proves insufficient. The 49 patches are at 7x7 resolution; substantial spatial information is already lost by P5 level.

### 7.5 Pose Head Verdict

**No change needed** for the current SOTA target. The pose head outperforms all targets at 200K params. If data normalisation issues (forward vector norms != 1.0) are fixed, MAE should improve further without architectural changes. The parameter budget saved here (up to 1.8M if we were to expand) should be allocated to the **activity head**.

---

## 8. Feature Routing Strategies

This is the **second-highest-impact design space** — how features are routed from backbone to heads determines gradient flow, information specificity, and competition.

### 8.1 Current Routing

```
MViTv2-S backbone
├── cls_token (blocks[15], position 0)  ──→ ActivityHead (MLP 768->75)
│                                         ──→ PoseHead (MLP 768->6)
├── P2 (conv_proj, 96ch, 56px)          ──→ BiFPN (top-down input only)
├── P3 (blocks[1], 192ch, 28px)          ──→ BiFPN ──→ DetectionHead (P3 only)
├── P4 (blocks[3], 384ch, 14px)          ──→ BiFPN ──→ DetectionHead (P4 only)
├── P5 (blocks[14], 768ch, 7px)          ──→ BiFPN ──→ DetectionHead (P5 only)
│                                         ──→ PSRHead (direct hook, not via FPN)
```

**Problems:**
- Activity and Pose share the same cls_token — one [B, 768] vector for two independent regression tasks
- PSR uses P5 via hook but gets it _before_ FPN processing (no top-down context from P4/P3 for semantic enrichment)
- Detection drops P2 for good reason (conv_proj is raw patch embeddings, no semantics) but P2 is still processed by BiFPN top-down — wasted compute

### 8.2 Separate cls_token for Activity vs Pose

- **Change:** Extract cls_token-like features from different backbone depths
  - Activity: blocks[14] features (deeper, more semantic)
  - Pose: blocks[11] features (less semantic but higher spatial resolution, 7x7 vs 14x14 before pooling)
  - Both via global average pool over spatial + cls_token concatenation
- **Params:** No change (same head structure, different input)
- **Rationale:** Pose benefits from spatial resolution (head bounding box needs pixel precision); activity benefits from semantic abstraction (state discrimination)
- **Expected impact:** -1 to -2 deg MAE (pose), +1-3% macro F1 (activity — via reduced gradient competition at cls_token)
- **Verdict:** **Easy win** — zero param cost, architectural change only in forward pass

### 8.3 PSR Gets FPN-Processed Features (Not Raw Hook)

- **Change:** Route P5 from FPN output (after top-down + bottom-up fusion) to PSR head instead of raw blocks[14] features
- **Rationale:** FPN-processed P5 has context from P4 and P3 features (lower-level assembly component features can enrich PSR's transition detection)
- **Risk:** FPN-processed features are temporal-pooled (mean over T), which loses per-frame temporal information that PSR needs
- **Mitigation:** Use temporal-pooled FPN features _in addition to_ raw per-frame P5 features (concatenate: [B, 768 + 256, 7, 7] -> projection)
- **Expected impact:** +1-3% PSR F1 (if lower-level spatial context helps component detection)
- **Verdict:** **Worth trying** — spatial context can only help PSR. Concatenation keeps temporal P5 features intact.

### 8.4 Multi-Head Feature Gate (Routing by Task)

- **Structure:** Learnable gating mechanism per FPN level per head
  - `gate_det_P3(x) = sigmoid(Linear(384->1)) * x`  — detection gets weighted P3 features
  - `gate_psr_P5(x) = sigmoid(Linear(768->1)) * x`  — PSR gets weighted P5 features
- **Params:** Negligible (+0.01M per gate)
- **Rationale:** A gating mechanism can learn to suppress P5 features that are specific to detection (and thus lossy for PSR) and amplify features relevant to transition detection.
- **Verdict:** **Nice-to-have.** Learnable routing is elegant but unlikely to move needle beyond what separate routing already achieves.

### 8.5 Feature Routing Verdict

**Implement in order:**
1. Separate cls_token sources for activity (blocks[14]) vs pose (blocks[11]) — free accuracy gain
2. FPN-processed features as _supplementary_ input to PSR head (+256ch context)
3. Learnable gating if routing conflicts persist

---

## 9. Multi-Scale Feature Design for Multi-Resolution Input

### 9.1 Current: Fixed 224x224 Input, T=16
- MViTv2-S expects [B, 3, 16, 224, 224]
- conv_proj downscales to T=8, H=56, W=56 (spatial stride 4, temporal stride 2)
- All heads operate on this single spatial resolution

### 9.2 Multi-Resolution Input (Patch Size Variant)
- **Idea:** Process clips at multiple resolutions (e.g., 224px + 112px) and fuse features
- Rationale: Large assembly boards are well-represented at 112px; small components (screws, washers) benefit from 224px
- **Cost:** +1.5x compute (second forward pass at 112px), +complex fusion network
- **Impact:** Marginal — assembly components are not small objects (even screws are >32x32 px at 224px)
- **Verdict:** **SKIP** — multi-resolution is for small-object detection in crowded scenes, not assembly state recognition

### 9.3 Variable Window Length
- **Idea:** Process longer clips (T=32 or T=64) and use sliding window attention for temporal coverage
- Current limitation: MViTv2-S memory scales linearly with T. T=32 at batch=4 -> ~4.5 GB activations (fits)
- **Benefit:** More temporal context for PSR transition detection (long assembly steps take 30-60 frames at 2 fps)
- **Cost:** +2x memory, +2x compute
- **Risk:** None — architectural change is input length only (MViTv2-S supports variable T)
- **Verdict:** **Ablate** — T=32 for PSR only. Keep detection/pose at T=16 center frame. Dual temporal branch pattern.

### 9.4 Multi-Scale Verdict

**Only variable window length (T=32) is worth trying**, and only for PSR head. Detection and pose operate on centre frame and don't benefit from longer temporal context. T=32 for PSR can be implemented as a second branch with separate temporal pooling.

---

## 10. Parameter Budget Allocation — Distribution Across Heads

### 10.1 Current Allocation (45M total)

| Component | Current Params | % of Total | Gradient Utilisation | Recommended Delta |
|-----------|---------------|-----------|---------------------|-------------------|
| MViTv2-S backbone | 34.5M | 76.7% | 100% (all tasks) | 0 (keep) |
| BiFPN (current 3D) | 2.5M | 5.6% | 20% (detection only) | -0.8M (→2D) |
| PSR head | 1.8M | 4.0% | 95% (healthy) | +0.2M (det-cond) |
| Activity head | 1.1M | 2.4% | 0.3% (starved) | +0.9M (to 2.0M) |
| Detection head | 0.8M | 1.8% | 15% (healthy) | 0 (keep) |
| Pose head | 0.2M | 0.4% | 14% (healthy) | 0 (keep) |

### 10.2 Target Allocation (45M total, rebalanced)

| Component | Target Params | % of Total | Change | Rationale |
|-----------|--------------|-----------|--------|-----------|
| MViTv2-S backbone | 34.5M | 76.7% | 0 | No change needed |
| BiFPN (2D, simplified) | 1.7M | 3.8% | -0.8M | 3D convs are unused compute |
| PSR head (wider + det-cond) | 2.0M | 4.4% | +0.2M | Detection conditioning |
| Activity head (spatial attn) | 2.0M | 4.4% | +0.9M | Core fix for gradient starvation |
| Detection head | 0.8M | 1.8% | 0 | No change needed |
| Pose head | 0.2M | 0.4% | 0 | Already saturating target |
| **Total** | **41.2M** | **91.5%** | **+0.3M net** | 3.8M headroom for routing |

The headroom (3.8M = 45M - 41.2M) can be allocated to:
- MViTv2-B backbone upgrade (+16.5M — over budget, would need to reduce elsewhere)
- Additional activity head capacity (spatial transformer, Section 5.4) 
- Multi-scale temporal branch for PSR (Section 9.3, T=32 processing)

### 10.3 Principles for Budget Allocation

1. **Gradient utilisation matters more than param count.** The activity head is 1.1M but receives 0.010 gradient norm. Adding params to a starving head only makes the problem worse. **Fix gradient routing first, then scale capacity.**

2. **Head parameter efficiency targets:**
   - Detection head: 0.8M is appropriate (256ch input, 24-class softmax + 4x16reg - small)
   - PSR head: 1.8-2.0M is efficient for 11-component sequence modelling
   - Activity head: 2.0M target is 2x the current MLP but with spatial attention structure (not MLP size)
   - Pose head: 0.2M is optimal. Going larger would violate the principle that a perfect low-param head signals task simplicity, not undercapacity.

3. **Cross-head redundancy elimination:**
   - Activity and Pose both use cls_token. Separation (Section 8.2) adds 0 params but improves gradient specialisation.
   - PSR and Detection both use P5. Shared P5 features means PSR's 3.18 gradient norm fights detection's 0.48 gradient norm. **Routing PSR through a separate backbone level (block 12 vs block 14) could reduce competition.**

---

## 11. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days each, cumulative impact: medium)

1. **PSR bidirectional** (Section 6.2) — flip causal mask boolean. Expected +1-3% PSR F1.
2. **Separate cls_token sources** (Section 8.2) — pose from blocks[11], activity from blocks[14]. Expected -1-2 deg MAE, +1-3% act macro F1.
3. **BiFPN 2D conversion** (Section 3.2 Option A) — temporal-pool before FPN. Expected -0.8M params, no accuracy loss.

### Phase 2: Activity Head Redesign (3-5 days, cumulative impact: critical)

4. **Spatial attention pool for activity** (Section 5.3) — replace cls_token MLP with P5 spatial features + learned attention. Expected 5-10x activity gradient amplification.
5. **Multi-layer feature aggregation** (Section 5.4) — add block 7 + block 11 features. Expected 10-20x gradient amplification (additive with Phase 2.1).
6. **Detection-conditioned activity** (Section 5.5) — RoI-Align detection features into activity. Expected 3-5x gradient amplification from detection gradient carry.

### Phase 3: Cross-Head Conditioning (3-5 days, cumulative impact: high)

7. **Detection-conditioned PSR** (Section 6.4) — feed detection features into PSR temporal encoder. Expected +3-5% PSR F1.
8. **FPN-supplemented PSR** (Section 8.3) — add FPN-processed features to PSR's raw P5 input. Expected +1-3% PSR F1.
9. **T=32 temporal window for PSR** (Section 9.3) — longer temporal context for transition detection. Expected +2-5% PSR F1.

### Phase 4: Ablations and Calibration (3-5 days, cumulative impact: additive)

10. **PSR wider (d=512) ablation** (Section 6.3) — check if increased head capacity helps.
11. **TOOD task-aligned detection head** (Section 4.4) — cls-reg interaction for fine-grained state discrimination. Expected marginal mAP50 gain.
12. **Variable window T=32 PSR** (Section 9.3) — confirm impact.
13. **Final budget rebalance** — measure per-task gradient norms under new architecture, adjust Kendall priors and PCGrad parameters.

### Expected Cumulative Impact

| Task | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 | SOTA Target |
|------|---------|---------|---------|---------|---------|-------------|
| Activity macro F1 | 0.002 | 0.01-0.05 | 0.05-0.15 | 0.15-0.25 | 0.20-0.30 | 0.40 |
| Detection mAP50 | 0.207 | 0.207 | 0.21-0.22 | 0.22-0.24 | 0.24-0.26 | 0.40 |
| PSR F1 (avg) | 0.35-0.45 | 0.40-0.48 | 0.40-0.48 | 0.48-0.55 | 0.50-0.58 | 0.60 |
| Pose MAE (deg) | 8.7 | 7.0-8.0 | 7.0-8.0 | 7.0-8.0 | 6.0-7.0 | 5.0 |

**Key insight:** The activity head going from 0.002 to 0.20-0.30 macro F1 is the single most impactful change in the entire exploration space. Without this, no other architectural change produces SOTA-level combined metrics. With it, the combined metric (act + det + psr + pose) becomes competitive for publication.

---

## 12. Recommendations Not Pursued (Anti-Portfolio)

| Idea | Reason to Skip |
|------|---------------|
| MViTv2-B/L backbone | Gradient routing, not backbone capacity, is the bottleneck |
| VideoMAE flat backbone | Loses hierarchical features for FPN; no advantage over MViTv2-S |
| Mamba/SSM for video | No K400 pretrained weights, unproven for multi-task video |
| NAS-FPN | 500+ GPU-hour search cost for marginal detection improvement |
| DETR detection head | 5M params for 1-5 object scenes (6x overkill) |
| Heatmap-based pose | 2M params for a task that converges at 200K params |
| Multi-resolution input (224+112) | Not helpful for assembly-scale objects |
| GRU/LSTM activity head | LSTMs are harder to train than transformers for 8-token sequences |
| Learnable gating (Section 8.4) | Elegant but low-impact vs separate routing |
| Kendall reweighting to fix 312x gap | Bounds on log_var prevent compensation (KENDALL_LOG_VAR_MIN_ACT = -0.5 gives max 1.65x boost) |
| Freeze PSR to boost activity | PSR oscillation is healthy (ALIVE/DEAD cycling is an optimisation bounce, not gradient bleed) |
| YOLOv10 NMS-free head | NMS is not our bottleneck |

---

## References

- MViTv2: Fan et al., "Multiscale Vision Transformers", CVPR 2022
- BiFPN: Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020
- PCGrad: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
- TOOD: Feng et al., "TOOD: Task-aligned One-stage Object Detection", ICCV 2021
- TAL: Li et al., "Task Aligned Assigner for Object Detection", 2022
- Kendall: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
- VideoMAE: Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training", NeurIPS 2022
- YOLOv9: Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information", 2024
- Current model: `src/models/mvit_mtl_model.py` (45M total params, MViTv2-S backbone)
- Training state: `analyses/consult_2026_06_10/56_ACTIVITY_HEAD_COLLAPSE_ROOT_CAUSE.md`
- Gradient analysis: `analyses/consult_2026_06_10/57_MULTI_TASK_GRADIENT_IMBALANCE.md`
- Detection diagnosis: `analyses/consult_2026_06_10/52_DETECTION_THE_REAL_DIAGNOSIS.md`
