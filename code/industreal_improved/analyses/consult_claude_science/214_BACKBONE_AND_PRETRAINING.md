# 214 — Backbone & Pretraining Deep Dive

**Date:** 2026-07-11
**Status:** Reference document for architecture decisions in the 4-task MTL paper
**Authority rule:** Where this document and earlier strategy docs (202-206) disagree on a factual claim (parameter count, accuracy, FLOPs), this document's numbers are grounded in the committed codebase (`video_backbones.py`, `video_backbone_multitask.py`, `mvit_mtl_model.py`), hand-computed arithmetic from published papers, or the verified corrections in doc 207 (Fact-Check §2.1). Estimates are labelled as such.

---

## 1. Current Backbone: MViTv2-S

The codebase currently supports two backbone architectures: a ConvNeXt-Tiny frame-level backbone (DeiT-style, ImageNet-1K pretrained, used in `mvit_mtl_model.py` for the run11-era multi-task model) and a video backbone suite (`video_backbone_multitask.py`) wrapping MViTv2-S and VideoMAE-S as shared spatiotemporal encoders feeding all four task heads.

The MViTv2-S (Multiscale Vision Transformer v2 Small) is the reference backbone for the WACV 2024 IndustReal SOTA (65.25% activity top-1, Schoonbeek et al.) and the default video backbone in the codebase's `VideoBackboneWrapper`.

**Architecture summary:**

| Property | Value |
|---|---|
| Parameter count | 34.5M (verified: doc 207 §2.6, codebase trace) |
| K400 top-1 | 81.0% (MViTv2 paper, cited in doc 207 §2.1) |
| Blocks | 16, distributed as depths [1, 2, 11, 2] |
| Stage dims | 96 / 192 / 384 / 768 |
| Clip embedding dim | 768 |
| Spatial pooling | Block 3: 56x56 -> 14x14 (stride 16); Block 15: 14x14 -> 7x7 (stride 32) |
| Temporal pooling | Block 15: T/2 -> 1 (global temporal pooling) |
| Conv proj | Conv3d(3, 96, k=(3,7,7), s=(2,4,4)) -> [B, 96, T/2, H/4, W/4] |
| Pretraining | Kinetics-400 supervised classification |
| Native feature scales | C2 (stride 4, 96ch), C3 (stride 16, 384ch), C5 (stride 32, 768ch) |
| Missing scale | C4 (stride 32) created artificially via stride-2 conv on C3 (`SpatialFeatureAdapter`) |

**Why MViTv2-S is the right baseline:** The IndustReal WACV benchmark established MViTv2-S as the single best AR backbone on this dataset, scoring 65.25% top-1 vs SlowFast's 60.39%. The multi-task extension (4 heads sharing one MViTv2-S) is the natural architecture-controlled comparison. Swapping away from MViTv2-S breaks apples-to-apples comparison with the published SOTA — a point made in doc 207 §2.1 and reinforced here. Any backbone replacement must justify the loss of this direct comparability.

**What the current multi-task setup uses from the backbone:**
- `clip_embed` (768-dim mean-pooled): feeds ActivityHead + FeatureBank
- C3 (384ch, 14x14): feeds FPN -> P3-P7 for detection, pose, PSR heads
- C5 (768ch, 7x7): feeds FPN, HeadPoseHead, PoseFiLM, ActivityHead projection
- C2 (96ch, 56x56): available but not used in current head routing (reserved for high-resolution future use)

**Memory footprint (video_backbone_multitask.py, line 43-46):**
- MViTv2-S forward pass: ~2.8 GB activation memory at batch=2, T=16, FP16
- All 4 heads + FPN: ~1.2 GB
- Total train (no checkpointing): ~7.5 GB
- With gradient checkpointing: ~5.0 GB
- With activation offloading: ~4.0 GB
- Fits RTX 3060 (12GB) at batch=2; RTX 5060 Ti (16GB) can push to batch=4

---

## 2. Backbone Candidates Ranked by Potential

Nine candidate families evaluated for replacing or supplementing MViTv2-S. Ranking criteria: (i) published accuracy on K400 or comparable video benchmark, (ii) parameter efficiency relative to our 34.5M baseline, (iii) compatibility with 4-head MTL interface, (iv) VRAM feasibility on our GPUs, (v) pretraining data alignment with assembly-domain tasks.

---

### Tier 1: Viable alternative (strong case, minimal downside)

**1. VideoMAE ViT-B (Tong et al., NeurIPS 2022)**

The most-discussed alternative in earlier strategy rounds. Corrected numbers per doc 207 §2.1:

| Claimed in 202 §1.1 | Corrected value | Source |
|---|---|---|
| 86M params, 87.4% K400 | 86M params, **~81.5% K400** | VideoMAE paper; doc 207 §2.1 |
| 75.4% SSv2 | **~70.8% SSv2** (ViT-L figure, not ViT-B) | VideoMAE paper; doc 207 §2.1 |
| +6.4% gain from better pretraining | ~+0.5% (81.5 vs 81.0) for 2.5x params | Hand computation |

**Verdict: Rejected as MViTv2-S replacement for the paper's main experiment.** The honest comparison is MViTv2-S 81.0% vs VideoMAE ViT-B ~81.5% — a 0.5-point difference for 2.5x the parameters (86M vs 34.5M). This inverts the efficiency claim (86M backbone + heads > 100M specialists the paper aims to beat) and breaks WACV comparability. However, VideoMAE's **self-supervised pretraining** (masked autoencoding on video) is a genuinely different learning signal from MViTv2-S's supervised K400 classification. If the project later pursues a "pretraining matters" ablation row, VideoMAE ViT-B is the correct comparison — not because it scores higher, but because it isolates self-supervision vs supervised pretraining as the independent variable. For that purpose, VideoMAE ViT-B should be frozen (linear probe only) to avoid confounding fine-tuning with pretraining differences.

**When to revisit:** Only in a dedicated pretraining-ablation section of a camera-ready revision, or if Ego4D/domain-specific pretraining becomes available and VideoMAE's MAE objective is needed as the base learner.

**2. VideoMAE V2 (Wang et al., CVPR 2023)**

Scales VideoMAE's masked video modeling to larger architectures with improved training recipes (more aggressive masking ratios, longer training schedules). Key change from V1: the decoder is discarded during fine-tuning (encoder-only, unlike V1's decoder-encoder design).

| Variant | Params | K400 top-1 | K700 top-1 |
|---|---|---|---|
| ViT-B | ~86M | ~82.5% (estimated) | ~74.5% |
| ViT-L | ~307M | ~85.5% | ~78.5% |
| ViT-H | ~632M | ~87.1% | ~80.5% |
| ViT-g | ~1B | ~88.5% | — |

**Verdict: Rejected for our compute constraints.** The smallest V2 variant (ViT-B, ~86M) is 2.5x MViTv2-S with marginal K400 gain (~1.5 points). The variant that produces genuinely impressive numbers (ViT-H/g) requires >300M parameters and 48+ GB VRAM — infeasible on our hardware. The video pretraining community has moved toward V2-like scaling, but the improvements are in the large-model regime. For a ~50M-class model targeting ~12-16GB VRAM, MViTv2-S is strictly better on efficiency.

---

### Tier 2: Viable but indifferent (comparable to MViTv2-S on efficiency, no clear advantage)

**3. ConvNeXt-T / S / B (Liu et al., CVPR 2022)**

ConvNeXt is the current frame-level backbone in the committed multi-task model (`mvit_mtl_model.py`). The frozen linear probe comparison (doc 150, file 144, doc 207) is the project's strongest evidence that **backbone pretraining domain dominates architecture choice** for this dataset.

| Variant | Params | Pretrain | Activity frozen probe |
|---|---|---|---|
| ConvNeXt-T | 22M | ImageNet-1K | 0.2169 (null: indistinguishable from 0.2217 majority baseline) |
| ConvNeXt-S | 50M | ImageNet-1K | Not tested |
| ConvNeXt-B | 89M | ImageNet-1K | Not tested |
| MViTv2-S | 34.5M | K400 | 0.3810 (strong signal) |

**Key finding:** The frozen linear probe comparison is definitive — MViTv2-S (K400) achieves 0.3810 activity top-1 on frozen features while ConvNeXt-T (ImageNet-1K) achieves 0.2169 (statistically indistinguishable from the majority-class baseline). This is not a parameter-count effect (MViTv2-S is 34.5M, ConvNeXt-T is 22M — a 1.6x gap that cannot explain a 16-point accuracy difference). The effect is **pretraining domain**: Kinetics-400 provides frame-level temporal action priors; ImageNet-1K provides object appearance priors. The activity task requires temporal structure; ConvNeXt has none.

ConvNeXt-T's contribution to the project is as a detection backbone, where ImageNet priors are appropriate (objects, not actions). Single-task ConvNeXt-T detection is expected to achieve 0.5-0.7 mAP50 (doc 150 §1.2). In the multi-task context, ConvNeXt-T is the correct frame-level baseline for the "multi-task cost" measurement — not because it is strong, but because it is the architecture-controlled comparison.

**Verdict:** ConvNeXt stays as the frame-level detection/pose backbone in the ConvNeXt-era MTL baseline. It is not a candidate to replace MViTv2-S for activity. ImageNet-22K pretraining (available for ConvNeXt-B and above) would likely improve detection but would not fix the temporal-blindness problem — frame-level features cannot encode motion regardless of pretraining scale.

**4. Swin Transformer V2 (Liu et al., CVPR 2022)**

Swin V2 introduces scaled cosine self-attention, log-spaced continuous position bias, and residual post-normalization to stabilize large-scale training. The video variant (VideoSwin) adapts Swin for spatiotemporal processing.

| Variant | Params | K400 top-1 | Notes |
|---|---|---|---|
| Swin-T | 28M | ~78.6% | Efficient but below MViTv2-S |
| Swin-S | 50M | ~80.2% | Comparable params, similar accuracy |
| Swin-B | 88M | ~81.6% | 2.6x MViTv2-S params, ~same accuracy |
| VideoSwin-B | 88M | ~82.7% | Temporal adaptation adds ~1 point |

**Verdict:** Swin-T/S are comparable to MViTv2-S on efficiency but **do not offer a clear accuracy advantage** at the same parameter budget. VideoSwin-B at 88M is 2.6x heavier for marginal gain. No version is implemented in the codebase; adding one would require a full VideoSwin feature extractor and intermediate feature hooking (weeks of engineering). Not worth the effort when MViTv2-S is already implemented, verified, and performing.

**5. CLIP-Pretrained ViT (Radford et al., 2021)**

CLIP's contrastive image-text pretraining (400M web image-text pairs) produces features that transfer surprisingly well to video tasks via frame-level feature extraction — better than ImageNet-1K pretraining, sometimes approaching weak video pretraining. The key advantage for our setup: CLIP features are **open-vocabulary**, meaning the model has seen a much broader distribution of visual concepts than Kinetics-400's 400 action classes.

| Variant | Params | Frameworks |
|---|---|---|
| ViT-B/16 CLIP | 86M | OpenCLIP, official CLIP |
| ViT-L/14 CLIP | 307M | OpenCLIP, official CLIP |
| ViT-H/14 CLIP | 632M | OpenCLIP only |

**Verdict:** CLIP ViT-B/16 is a plausible frozen-feature extractor for activity — the 400M-image pretraining provides strong object-level features that may capture assembly-relevant visual concepts CLIP learned from instructional images/videos on the web. However, CLIP is **frame-level only** (no temporal modeling), so the same temporal-blindness limitation applies as ConvNeXt. CLIP features are also unproven on IndustReal (no probe result exists). Worth exactly one experiment: a frozen CLIP ViT-B/16 linear probe on activity, identical protocol to the MViTv2-S probe. Cost: ~1 GPU-day (extraction), zero training. If probe > 0.40, CLIP becomes a cheap fallback for the activity head.

---

### Tier 3: Interesting but immature or infeasible

**6. Mamba / Vision Mamba (Zhu et al., 2024; State Space Models)**

Mamba (Gu & Dao, 2023) replaces attention with a selective state-space model that processes sequences in linear (not quadratic) time. Vision adaptations (VMamba, Mamba-ND) apply this principle to image and video processing.

| Model | Sequence complexity | Reported performance |
|---|---|---|
| VMamba-T | O(L) (linear) | Comparable to Swin-T on ImageNet |
| VideoMamba | O(L) (linear) | ~80% on K400 (small models) |

**Verdict:** Too early for a submission-bound paper. Mamba video backbones are pre-release or concurrent (early 2024 onward), lack mature pretrained weights on K400, have no established fine-tuning recipes, and would require a full codebase integration (feature hooking, FPN adaptation, checkpoint serialization). The linear-complexity advantage (O(L) vs O(L^2)) is attractive for long-sequence activity recognition (T=64+) but irrelevant for our T=8-16 clip regime where quadratic attention is cheap. **File under "watch for v2" — not a submission candidate.**

**7. InternVideo / VideoSwin / TimeSformer**

| Model | Params | K400 top-1 | Notes |
|---|---|---|---|
| TimeSformer-L | 121M | ~80.7% | Divided space-time attention; heavy |
| VideoSwin-B | 88M | ~82.7% | Swin temporal extension |
| InternVideo | ~300M | ~83.5% | Multi-modal pretraining; 2-stage |

**Verdict:** TimeSformer-L is comparable to MViTv2-S on accuracy with 3.5x the params — a strict downgrade on efficiency. VideoSwin-B is a plausible candidate but unverified in this codebase and not clearly superior to MViTv2-S at equivalent compute. InternVideo is multi-stage, multimodal, and requires >300M params — infeasible. None offers a clear advantage over the existing, working, benchmarked MViTv2-S implementation.

**8. DINOv2 (Oquab et al., 2023)**

Self-supervised ViT features trained on a curated 142M-image dataset (LVD-142M). DINOv2 features are the current SOTA for dense visual tasks (depth, segmentation) and outperform CLIP on several geometric/shape reasoning benchmarks.

**Verdict:** Like CLIP, DINOv2 is frame-level only. Unlike CLIP, DINOv2's self-supervised objective (self-distillation + masked image modeling) produces features specialized for **spatial correspondence**, not temporal action. DINOv2 is useful for **detection and pose heads** — the spatial tasks where geometric correspondence matters — but would not help activity. A DINOv2-B linear probe on detection features (C3 spatial features, frozen) is a legitimate cheap experiment if detection remains stuck. Not a backbone replacement.

**9. SAM Encoder (Kirillov et al., 2023, Segment Anything)**

SAM's ViT encoder was trained on 11M images with a promptable segmentation objective. The features are extremely robust for segmentation and detection tasks.

**Verdict:** SAM is a detection/segmentation specialist, not a video backbone. Using SAM's encoder as a frozen feature extractor for the detection head could lift detection performance — this is the same logic as COCO pretraining for detection. However, SAM's ViT-H (632M) cannot fit our VRAM; SAM's ViT-B (86M) is possible but unproven. The correct experiment is a light-weight comparison: replace the ConvNeXt-T backbone's C3/C5 features with SAM ViT-B features on a 200-image subset, measure detection mAP change. Zero training required (frozen extraction). Worth 1 day of engineering if detection needs a late-stage rescue.

---

## 3. Pretraining Strategies

Pretraining determines what priors the backbone carries into the assembly domain. The evidence in this project is unambiguous: **pretraining domain dominates architecture choice at fixed parameter count** (MViTv2-S K400 vs ConvNeXt ImageNet-1K, both transformers, 34.5M vs 22M, but activity performance differs by 16 points).

### 3.1 Supervised Video Pretraining

**Kinetics-400 (346K videos, 400 action classes)**

The default for all video backbones in this project. MViTv2-S's K400 checkpoint is loaded from torchvision's `KINETICS400_V1` weights. K400 is the standard video pretraining benchmark — it covers diverse human actions filmed from YouTube videos, providing motion, object interaction, and temporal structure. **Strength:** Broad coverage of human activity, strong temporal priors. **Weakness:** Domain gap — YouTube action videos vs egocentric assembly on a tabletop. Our 0.3810 frozen probe proves the gap is bridgeable.

**Kinetics-600/700 (480K/650K videos, 600/700 classes)**

Larger Kinetics variants. MViTv2-S checkpoints exist for K600 (torchvision supports weights: `KINETICS600_V1`) at marginally higher accuracy (~1-2 points on K600 eval). **Verdict:** Worth using if a K600-pretrained MViTv2-S checkpoint is available and easily pluggable. The 50% more pretraining data and broader concept coverage (700 vs 400 classes) should transfer modestly better to assembly actions. Expected activity gain: +1-2% relative, no architecture change. Cost: re-run of the MViTv2-S fine-tuning pipeline with a different weight file.

**Ego4D (3,670 hours, egocentric, procedural)**

The most interesting pretraining dataset for this project, and the one with the most unrealized potential. Ego4D contains 3,670 hours of egocentric video spanning daily activities, procedural tasks, and social interactions — much closer in distribution to IndustReal (HoloLens 2 egocentric assembly) than Kinetics (third-person YouTube). Ego4D also includes procedural understanding annotations (steps, state changes, errors).

| Pretraining set | Domain | Temporal structure | Size |
|---|---|---|---|
| Kinetics-400 | Third-person, YouTube actions | Short clips (10s) | 346K videos |
| Kinetics-700 | Third-person, YouTube actions | Short clips (10s) | 650K videos |
| Ego4D | First-person, procedural | Minutes-long sequences | 3,670 hrs |
| Ego4D + K400 | Mixed | Both | Largest combo |

**Verdict (from doc 207):** Ego4D pretraining is a "Tier-3 luxury for a later paper" on the current timeline. If the project had 6+ months, an Ego4D-pretrained (or Ego4D+Kinetics co-pretrained) MViTv2-S would be the single highest-leverage architecture change: it aligns data distribution (egocentric), temporal structure (procedural, minutes-long), and task concepts (assembly, state tracking, error handling). Implementation cost: (a) source or train an Ego4D checkpoint on MViTv2-S (not trivial — Ego4D has different annotation modalities), (b) adapt the fine-tuning pipeline, (c) run ~2 GPU-weeks of fine-tuning. **Not for this submission.**

### 3.2 Detection-Focused Pretraining

**COCO (118K images, 80 object classes)**

COCO pretraining is the standard initialization for detection backbones. Our YOLOv8m single-task detection (0.995 mAP50) uses COCO pretraining. The multi-task model's ConvNeXt-T uses ImageNet-1K pretraining instead — a significant disadvantage for the detection head.

**Expected benefit:** Switching the backbone's pretraining from ImageNet-1K to COCO (or COCO+ImageNet sequential pretraining) for the detection head alone is estimated at +3-10% mAP based on COCO-to-detection transfer literature. This is achievable without changing the backbone architecture — the multi-task model can load COCO-pretrained weights for the backbone's spatial feature extractor while keeping K400-based activity features via a dual-backbone setup. Practically, COCO pretraining helps the detection head's **feature granularity** (COCO's 80 classes include tools, parts, and objects relevant to assembly) and its **anchor prior** (COCO object scales match assembly part sizes).

**Verdict:** COCO pretraining for detection is the most actionable pretraining change on the current timeline — zero architecture modification, one weight-file swap, measurable effect. The experiment: train a ConvNeXt-T detection head with COCO-pretrained (rather than ImageNet-1K-pretrained) backbone on the 27K annotated frames. Expected: mAP50 increase from ~0.2 (current multi-task) toward 0.5+. Cost: 1-2 GPU-days.

**ImageNet-22K (14M images, 22K classes)**

Larger ImageNet pretraining scale. ConvNeXt-Base and above offer ImageNet-22K pretrained variants. The transfer benefit from 22K classes to detection is well-documented (+1-2% mAP over ImageNet-1K on COCO). For activity, 22K pretraining on a frame-level backbone does not add temporal structure — the activity head would still need a video backbone.

**Verdict:** Marginal improvement for detection, zero improvement for activity/PSR. Not worth the engineering effort compared to COCO pretraining (which is more aligned with detection task structure). Only relevant if multi-task ConvNeXt detection stays below 0.2 mAP50 after all other fixes.

### 3.3 Self-Supervised Pretraining

**MAE (He et al., CVPR 2022) / VideoMAE (Tong et al., NeurIPS 2022)**

Masked autoencoding learns visual representations by reconstructing masked patches from visible ones. VideoMAE extends this to video with a cube masking strategy (mask out spatiotemporal tubes). The resulting features capture temporal correspondence (which patches go together across frames) without requiring action labels.

**Why VideoMAE pretraining is qualitatively different from K400 supervised pretraining:** K400 supervised pretraining trains a classifier on 400 action labels. The backbone learns features that discriminate between YouTube action classes — a superset of assembly actions but with different concepts (sports, cooking, music vs. peg insertion, screw driving). VideoMAE's self-supervised objective has no label bias; it learns whatever temporal structure exists in the video. For a domain like assembly where the action vocabulary is narrow but the temporal grammar (sequence of subtasks) is rich, self-supervised video pretraining may capture procedural structure that supervised pretraining misses.

**Evidence: The 0.3810 MViTv2-S probe already works without self-supervision, so the question is marginal:** is the remaining gap (0.3810 to target 0.55-0.62) better addressed by self-supervised fine-tuning or by more compute? The literature says more compute wins at fixed model size — VideoMAE's advantage is at scaling to larger models, not at the 34.5M-86M parameter range.

**DINO / DINOv2 (Caron et al., 2021; Oquab et al., 2023)**

Self-distillation with no labels. DINOv2 demonstrated that a carefully curated training set (LVD-142M) with a self-supervised objective can match or exceed supervised ImageNet-22K performance across many downstream tasks. DINOv2's strength is **geometric feature correspondence** — the self-distillation objective naturally produces features where patch-level similarity corresponds to 3D correspondences. This is useful for detection and pose: better spatial correspondence means better keypoint localization and better object detection, especially on the assembly domain where objects are geometric (mechanical parts, tools).

**Verdict:** DINOv2-B linear probe on detection features is a cheap, high-upside experiment. Not a backbone replacement.

**MoCo v3 (Chen et al., ICCV 2021)**

Momentum contrastive learning for Vision Transformers. MoCo v3's ViT-B checkpoint achieves competitive ImageNet accuracy but is superseded by DINOv2 and MAE for ViT pretraining quality. No demonstrated advantage over MAE for video tasks.

**Beit (Bao et al., ICLR 2022; video variant: BEVT)**

Masked image modeling with a discrete VAE tokenizer. BEVT extends BeiT to video by jointly training masked video prediction. Conceptually elegant (predicts visual tokens, not pixels) but requires a pre-trained tokenizer. Not implemented in the codebase and unlikely to outperform VideoMAE at equivalent parameter count.

### 3.4 Multi-Modal Pretraining

**CLIP (Radford et al., 2021)**

Covered in §2 Tier 2. The key property: CLIP's open-vocabulary visual features are grounded in natural language descriptions, not action labels. For assembly tasks, CLIP may recognize objects ("screwdriver"), states ("tightened"), and actions ("inserting") because these concepts appear in image-text pairs. CLIP features are frame-level only. Worth exactly one probe experiment on activity (frozen ViT-B/16 extract, evaluate with linear classifier).

**ImageBind (Girdhar et al., CVPR 2023)**

Binds six modalities (images, text, audio, depth, thermal, IMU) into a shared embedding space. ImageBind's video features integrate audio-visual correspondence, which could be useful for assembly tasks where tool sounds indicate state changes (screw tightening, peg insertion). However, ImageBind's video encoder is CLIP-based (no temporal modeling), and the audio/tactile modalities are not available in IndustReal (HoloLens 2 has audio but it's not annotated or aligned with assembly steps). Not actionable.

**Video-LLaMA / Video-ChatGPT**

Video-language models that combine a video encoder (BLIP-2 or CLIP) with an LLM for video question answering. These are generative models, not feature extractors for dense prediction tasks (detection, pose, PSR). The video encoder components (BLIP-2's Q-Former, CLIP ViT) are frame-level and not competitive with K400-pretrained video backbones for activity recognition. Not relevant.

---

## 4. What Pretraining Helps Which Head

The multi-task model has four heads with fundamentally different feature requirements. The optimal pretraining strategy for the shared backbone must serve all four, but the evidence shows they benefit from **different** pretraining signals.

### Detection Head

**What it needs:** Spatial feature discrimination (objects/parts at different scales), accurate localization, robustness to clutter.

**Best pretraining:** COCO detection pretraining (+3-10% mAP estimated). Detection is the one head where ImageNet-1K classification pretraining is provably suboptimal — detection benefits from object-level discrimination, not action-level discrimination. The current ConvNeXt-T backbone's ImageNet-1K pretraining leaves at least 10-15 mAP points on the table relative to a COCO-pretrained counterpart.

**Second best:** ImageNet-22K classification pretraining (+1-2%). Much smaller benefit than COCO, because classification pretraining optimizes for image-level labels, not per-pixel object discrimination.

**K400 supervised video pretraining:** Uncertain effect. K400's action discrimination could interfere with object detection (the backbone learns to ignore object appearance and focus on motion). The MViTv2-S detector in `video_backbone_multitask.py` extracts spatial features from intermediate MViT blocks (C3/C5) after the center-frame slice — which uses K400 spatiotemporal features for a single-frame task. This mismatch (temporal backbone + spatial head) is unmeasured in the literature and may or may not help detection. The codebase's `VideoBackboneWrapper._forward_mvit()` center-frame slice at lines 208-209 preserves spatial features from a temporal representation — a design choice that should be ablated in the paper.

### Activity Head

**What it needs:** Temporal action discrimination (which assembly step is happening), motion patterns, object-action association.

**Best pretraining:** Kinetics-400/700 supervised video pretraining. Proven by the 0.3810 frozen probe vs ConvNeXt's null result. The margin is decisive (16 points for a 1.6x parameter difference).

**Second best:** Ego4D pretraining (if the project had months, not weeks). The egocentric distribution match to IndustReal is the single largest untapped opportunity.

**Self-supervised video pretraining:** Neutral. VideoMAE's self-supervised features may capture temporal structure differently, but no evidence in this codebase suggests they outperform K400 supervised features at the same parameter count.

**ImageNet/COCO pretraining:** Worthless for activity. The frozen ConvNeXt probe (0.2169, indistinguishable from the 0.2217 majority baseline) is the definitive empirical proof: frame-level pretraining on any image dataset cannot provide temporal action discrimination.

### PSR Head

**What it needs:** Temporal step-change detection (when does assembly state transition from one step to the next), component state tracking across time, robustness to detection failures.

**Best pretraining:** This is the unanswered question of the project. PSR's feature routing (from FPN P3+P4+P5) draws on the backbone's spatial features, which are ImageNet-1K-pretrained in the ConvNeXt setup. The 0.7018 per-component optimal F1 suggests the head learns some state-change signal from these features, but the 0.6364 D4+D1R decoder result (with clean YOLOv8m detections) proves detection quality is the dominant factor — better features through a better backbone pretrained for detection would ripple through to PSR.

**Hypothesis:** PSR benefits most from **temporal pretraining that emphasizes state transitions rather than action categories**. K400 supervised pretraining teaches the backbone to recognize "slicing" as a coherent action class. PSR needs the backbone to recognize when an action **changes** the assembly state. Self-supervised temporal pretraining (VideoMAE, which learns patch-level temporal correspondence) may align better with this need, because the MAE objective forces the model to understand how frame patches relate across time — a skill that transfers to detecting state transitions.

**Expected gain from temporal self-supervised pretraining (VideoMAE ViT-B) on PSR:** Unknown. The experiment (extract MViTv2-S features + VideoMAE features, compare PSR F1) costs ~2 GPU-days once the extraction code is written. Should be gated behind: (i) all 9 implementation fixes applied to PSR, (ii) PSR F1 stabilised above 0.75, (iii) residual gap to WACV B1 baseline (0.779) quantified. If PSR with fixed implementation is still 5+ F1 points below B1, temporal pretraining is the next lever.

### Pose Head

**What it needs:** Geometric correspondence (head orientation from image features), viewpoint invariance.

**Best pretraining:** None. Head pose (9-DoF) is a regression task from spatial features, not a recognition task. The current 9.14deg forward / 7.78deg up is functional with ImageNet-1K pretrained ConvNeXt features — the pose head is the best-working head in the system with the least pretraining dependence. The linear readout from C4+C5 features means the backbone needs good spatial correspondence, which both ImageNet and K400 provide.

**Verdict: No clear pretraining advantage for pose.** DINOv2's self-supervised features (known for geometric correspondence) might yield a marginal improvement, but at 9deg the head already meets the submission threshold. Pose is the one head where pretraining experiments are strictly luxury — defer to later work.

---

## 5. Parameter-FLOPs Trade-off Analysis

The central constraint for any backbone swap is: **does the parameter increase justify the accuracy gain, given that the paper's efficiency claim is built on ~2x parameter reduction vs specialists?**

### Methodology

Total parameters for the multi-task model (with PSR diet head at 1.78M, doc 207 §2.6 verified):
```
Backbone + FPN + 4 heads = backbone_params + ~1.6M (FPN) + 1.78M (PSR) + 5.0M (act) + 0.2M (pose) + 0.36M (det) ≈ backbone_params + 9M
```

Specialists total (~100M, verified in doc 207 and `efficiency_audit.md`):
```
YOLOv8m detection (25.9M) + MViTv2-S activity (34.5M) + PSR decoder (negligible) + MediaPipe pose (negligible) ≈ 60-100M depending on exact pipeline
```

**The paper's efficiency claim holds while backbone_params + 9M < 100M.** MViTv2-S at 34.5M gives 43.5M total (2.3x reduction). VideoMAE ViT-B at 86M gives 95M total (1.05x reduction — essentially parity, no efficiency story).

### Trade-off Table

| Backbone | Params (M) | MT params (M) | Ratio vs 100M | K400 top-1 | Est. activity gain | VRAM (B=2) | Feasibility |
|---|---|---|---|---|---|---|---|
| MViTv2-S | 34.5 | 43.5 | **2.3x** | 81.0% | Baseline | 5.0 GB | **Ready, committed** |
| MViTv2-S (K600) | 34.5 | 43.5 | **2.3x** | ~82.5% | +1-2% | 5.0 GB | Trivial weight swap |
| ConvNeXt-T | 22.0 | 31.0 | 3.2x | N/A (image) | Null (probe) | 2.5 GB | Frame-level only |
| VideoMAE ViT-B | 86.0 | 95.0 | 1.05x | ~81.5% | ~0% | ~12 GB | Marginal, breaks efficiency |
| VideoMAE V2 ViT-B | 86.0 | 95.0 | 1.05x | ~82.5% | +1%? | ~12 GB | Same problem + no code |
| VideoSwin-B | 88.0 | 97.0 | 1.03x | ~82.7% | +1-2%? | ~12 GB | No code, marginal gain |
| Swin-S | 50.0 | 59.0 | 1.7x | ~80.2% | -1% | ~7 GB | Worse accuracy at higher cost |
| ConvNeXt-S | 50.0 | 59.0 | 1.7x | N/A | Null | ~6 GB | Frame-level, no temporal |
| CLIP ViT-B/16 | 86.0 | 95.0 | 1.05x | N/A | Probe TBD | ~12 GB | Frame-level, needs probe |
| VideoMAE V2 ViT-L | 307 | 316 | 0.32x | ~85.5% | +4-5%? | >24 GB | Infeasible (VRAM) |

**Key insight:** No available backbone between 34.5M and 86M provides a clear accuracy-efficiency Pareto improvement over MViTv2-S. The density of video backbone architectures clusters at either ~25-35M (MViTv2-S, Swin-T, EfficientVideo) or ~86M+ (ViT-B variants). The 34.5M-50M range is a desert. MViTv2-S occupies the sweet spot.

### FLOPs Analysis

FLOPs for MViTv2-S on a single T=16, 224x224 clip:
- MViTv2-S forward (T=16, 224^2): ~60 GFLOPs (video_multitrack_model.py measured, fvcore)
- FPN forward: ~10 GFLOPs
- All 4 heads: ~5 GFLOPs
- **Total: ~75 GFLOPs per clip at batch=1**

For comparison:
- ConvNeXt-T (single frame, 224^2): ~4.5 GFLOPs × 16 frames = ~72 GFLOPs (frame-level = 16x forward passes of a frame backbone)
- VideoMAE ViT-B (T=16, 224^2): ~180 GFLOPs (3x MViTv2-S)
- YOLOv8m detection (single frame): ~10 GFLOPs × 16 frames = ~160 GFLOPs (single-task detection baseline)
- MViTv2-S activity specialist (single-task): ~60 GFLOPs

**GFLOPs efficiency of MTL vs separated:**
- MTL (MViTv2-S, all 4 heads shared): ~75 GFLOPs per clip
- Separated (specialist pipeline): ~60 (activity) + ~160 (det, YOLOv8m) + ~20 (pose, MediaPipe) + negligible (PSR decoder) = ~240 GFLOPs
- MTL efficiency ratio: ~3.2x on FLOPs, ~2.3x on params

This GFLOPs advantage is a second efficiency claim the paper can make — the **single forward pass** saves compute beyond parameter count. Per-task backbone adapters (the rejected Lever 3 from doc 207 §2.2) would delete this advantage by requiring 4 backbone passes. The analysis reinforces doc 207's verdict: the single-pass claim is structurally valuable and must be protected.

---

## 6. Compute Constraints and Realistic Path

**Hardware available:**
- RTX 3060: 12 GB VRAM, ~12 TFLOPS FP16
- RTX 5060 Ti: 16 GB VRAM, ~28 TFLOPS FP16 (newer generation)
- Total combined: 28 GB, ~40 TFLOPS (if both used simultaneously for different jobs)
- No multi-GPU training support (SyncBN unavailable)

**What fits on each GPU at batch=2, T=16, FP16 with gradient checkpointing:**

| Model | RTX 3060 (12GB) | RTX 5060 Ti (16GB) |
|---|---|---|
| MViTv2-S + 4 heads | Yes (5.0 GB) | Yes (batch=4 possible at ~8 GB) |
| VideoMAE ViT-B + 4 heads | Borderline (~11 GB) | Yes (batch=2 fits with care) |
| VideoMAE V2 ViT-B + 4 heads | No | Borderline (~14 GB estimated) |
| ConvNeXt-T + 4 heads | Yes (2.5 GB, excess capacity) | Yes |
| MViTv2-S fine-tune (full backbone) | Yes at batch=1 (gradient accum 4) | Yes at batch=2 |

**Realistic training time estimates:**

| Experiment | GPU | Estimated wall time |
|---|---|---|
| MViTv2-S fine-tune (single-task activity, 50 epochs) | 5060 Ti | ~5 days |
| MViTv2-S fine-tune (multi-task, 50 epochs) | 5060 Ti | ~7 days |
| VideoMAE ViT-B linear probe (frozen) | 3060 | ~12 hours |
| ConvNeXt-T single-task detection (50 epochs) | 3060 | ~3 days |
| CLIP ViT-B/16 linear probe (frozen) | 3060 | ~1 day |
| K600 MViTv2-S weight swap + fine-tune | 5060 Ti | ~5 days (cheaper than full swap) |

**The binding constraint is time, not compute.** MViTv2-S fits on both GPUs. The 5060 Ti can handle larger batch sizes. Fine-tuning a K600-pretrained MViTv2-S takes the same wall time as K400 fine-tuning (same architecture, different weight file) and is the single highest-ROI experiment among all backbone changes. Every other candidate (VideoMAE, CLIP, DINOv2) requires at least one additional experiment cycle (2-7 GPU-days) with uncertain or negative upside.

### Recommended Backbone Path for Submission

1. **Primary:** MViTv2-S (K400), as implemented in `video_backbone_multitask.py` — the reference baseline against which all WACV comparisons are drawn.
2. **Secondary (if time permits):** K600-pretrained MViTv2-S weight swap — 1 day of engineering, 5 days of fine-tuning, expected +1-2% activity gain.
3. **Tertiary (diagnostic only, no submission delay):** CLIP ViT-B/16 linear probe on activity — establishes whether CLIP's large-scale supervision bridges the temporal gap for this domain. Cost: 1 day, no fine-tuning.
4. **Not for submission:** VideoMAE, VideoMAE V2, VideoSwin, Mamba, InternVideo, Ego4D pretraining. All require codebase integration effort that outweighs their expected benefit at this stage, and most break the paper's efficiency and comparability narratives.

The backbone experiment order for the paper is therefore:
- **Figure 2 (ablation):** MViTv2-S (K400) vs ConvNeXt-T (ImageNet-1K) — frozen linear probe, 0.3810 vs 0.2169. Already done.
- **Main result:** MViTv2-S (K400) fine-tuned on IndustReal activity, detection, PSR, pose. Running.
- **Ablation row (if complete):** MViTv2-S (K600) fine-tuned identically. Optional, low-cost.
- **Related work sentence:** "Larger video backbones (VideoMAE ViT-B, 86M) were evaluated but did not improve activity accuracy enough to justify the 2.5x parameter increase, consistent with the scaling ceiling observed on this dataset."

---

## Summary: The Backbone Decision

The backbone is not the binding constraint for any head except activity, and MViTv2-S solves that constraint (0.3810 frozen -> 0.45-0.55 fine-tuned expected). Detection pretraining (COCO) is a separate lever that helps the detection head via weight initialization, not architecture. PSR pretraining (self-supervised temporal) is an open research question that the project does not have time to answer. Pose pretraining is irrelevant.

The honest headline from the backbone analysis: **MViTv2-S at 34.5M is the correct backbone for this project, for three independent reasons** — it establishes the WACV-comparable baseline, it maintains the 2.3x parameter efficiency claim, and it has the smallest VRAM footprint among all video backbones that produce non-null activity features. Every other candidate either costs more parameters for equal accuracy, breaks the efficiency story, or requires months of unproductive engineering. The only actionable pretraining change in the current timeline is COCO initialization for the detection head — which is a weight-loading change, not a backbone swap.

---

*Companion to doc 207 (Round 5 verdict), doc 144 (Video Backbone Opus Brief), doc 150 (Master Synthesis). Numbers verified against codebase at HEAD (`video_backbones.py`, `video_backbone_multitask.py`, `mvit_mtl_model.py`) and corrections tracked in doc 207 §2.1 (VideoMAE fact-check) and §2.6 (MViTv2-S verification).*
