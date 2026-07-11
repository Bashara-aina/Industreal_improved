# Doc 220: Related Work Survey

**Date:** 2026-07-11
**Purpose:** Comprehensive taxonomy of prior work for the IndustReal 4-task multi-task learning paper. Organized by paper section (assembly recognition, video architectures, MTL, egocentric perception, detection, pose, PSR).
**Relationship to other docs:** Doc 213 covers MTL optimization literature in depth; this doc maps the full related-work landscape that a paper's "Related Work" section must address.

---

## 1. Table of Contents

1. Assembly Activity Recognition
2. Video Architectures for Action Recognition
3. Multi-Task Learning (cross-reference to doc 213)
4. Egocentric Vision and Hand-Object Interaction
5. Object Detection in Industrial Contexts
6. Head Pose Estimation from Egocentric Video
7. Procedure State Recognition
8. Gaps and Opportunities for Our Paper

---

## 2. Assembly Activity Recognition

### 2.1 The IndustReal Dataset (Our Direct Predecessor)

IndustReal (Schoonbeek et al., WACV 2024) is the only publicly available dataset for fine-grained assembly procedure recognition. It provides 84 egocentric recordings (HoloLens 2, 1280x720, 10 FPS) of construction-toy car assembly by 27 participants. The dataset defines 75 fine-grained action classes for activity recognition (AR) and uses verb-grouped 69-class mapping as a secondary evaluation protocol.

The original paper's primary contribution is the dataset and multi-modal baselines:
- **MViTv2-S** achieves 65.25% top-1 AR accuracy (RGB, Kinetics-400 pretrained). This is the SOTA anchor for our activity head.
- **SlowFast** (Feichtenhofer et al., ICCV 2019) achieves 60.39% top-1 with Kinetics pretraining.
- Multi-modal fusion (RGB + visible light + stereo) raises MViTv2-S to 66.45%, but our 4-task model uses RGB only.

Our project differs from the original IndustReal paper in three critical ways:
1. **Multi-task architecture**, not single-task per head. The original paper trains separate models for AR (MViTv2-S), ASD (YOLOv8m), and PSR (B1/B2/B3 procedural rules).
2. **Single low-cost GPU** (RTX 3060/5060 Ti), not a V100 cluster.
3. **Per-frame PSR**, not transition-based. Our PSR head predicts 11 binary component states per frame, while the original uses detection-triggered step transitions with procedural knowledge.

### 2.2 Other Assembly/Action Datasets

**MPII Cooking 2** (Rohrbach et al., 2015): 273 video sequences of 89 fine-grained cooking activities. Introduced the "activity" vs "sub-activity" distinction that our 69 verb-grouped classes parallel. Typical benchmarks use TCN + I3D features. SOTA is approximately 70% frame-level accuracy with multi-stage refinement.

**EPIC-Kitchens** (Damen et al., CVPR 2018, ECCV 2020): 100 hours of egocentric kitchen videos across 45 environments. 97 verb classes, 331 noun classes. The verb-noun compositionality inspired our verb-class activity prediction. SOTA on EPIC-Kitchens-100 action recognition is approximately 51.5% top-1 (TSM with Omnivore pretrained features, 2023). Key difference from IndustReal: EPIC-Kitchens has longer, continuous recordings with natural variability, while IndustReal has structured assembly procedures with explicit step definitions.

**Assembly101** (Sener et al., CVPR 2022): 4,321 videos of 101 chair and car assembly tasks with multi-view (8 cameras + 1 egocentric). Includes both action and pose annotations. Introduced "coarse" vs "fine" action taxonomy. SOTA action recognition uses MViTv2 with multi-view fusion, achieving approximately 72% top-1 on coarse actions. The multi-view setup makes direct comparison with our single-egocentric setup difficult.

**50 Salads** (Stein and McKenna, 2013): 50 recordings of salad preparation with 17 action classes. Widely used for temporal action segmentation benchmarks. SOTA MS-TCN++ achieves approximately 85% F1 on temporal segmentation at 0.1 second overlap tolerance. Temporal segmentation differs from our per-frame recognition task.

### 2.3 Temporal Action Segmentation vs Per-Frame Recognition

The temporal action segmentation literature (MS-TCN, ASFormer, UVAST) addresses the problem of segmenting long untrimmed videos into action segments. These methods use temporal refinement and smoothing (typically with a TCN decoder) to produce coherent segment predictions.

Our activity head does not use temporal refinement: it predicts per-frame activity labels from a video backbone clip (16 frames). This is closer to action recognition in trimmed clips than temporal segmentation. The WACV 2024 IndustReal baselines follow this paradigm. We note this distinction because it affects which related-work citations are appropriate:
- MS-TCN++ (Li et al., ECCV 2020) and ASFormer (Yi et al., AAAI 2021) are NOT directly comparable baselines for our per-frame prediction.
- MViTv2-S and SlowFast on trimmed clips ARE comparable baselines, and our 0.2169 frozen-ConvNeXt probe versus MViTv2-S 0.6525 gap is the relevant measure.

---

## 3. Video Architectures for Action Recognition

### 3.1 Multiscale Vision Transformers (MViT)

Fan et al. (CVPR 2021) introduced MViT, a hierarchical vision transformer that produces multiscale feature maps through pooling attention. MViTv2 (Li et al., CVPR 2022) added decomposed relative position embeddings, residual pooling connections, and improved multiscale pooling. These innovations make MViTv2 particularly suited for video:
- Pooling attention reduces the O(T*H*W)^2 cost of full spatiotemporal attention to O(T*H*W * pool_kernel), enabling practical training on 16-32 frame clips.
- Multiscale outputs (C1-C5) are naturally compatible with detection heads that need spatial feature pyramids.
- Kinetics-400 pretrained MViTv2-S (34.5M params) achieves 81.0% K400 top-1.

**Why MViTv2-S is our backbone:** Our frozen ConvNeXt-T probe (0.2169 activity top-1) versus MViTv2-S (65.25% activity top-1 on the same dataset) demonstrates that backbone pretraining domain (Kinetics-400 video versus ImageNet-1K images) is the dominant factor. The MViTv2-S code is already integrated in `video_backbones.py` and `video_backbone_multitask.py` with full multi-task head support.

### 3.2 Masked Video Autoencoders (VideoMAE)

Tong et al. (NeurIPS 2022) proposed VideoMAE, extending masked autoencoding to video by masking a high proportion (75-90%) of video patches and reconstructing the masked patches. VideoMAE ViT-B achieves approximately 81.5% K400 top-1 (as corrected in doc 207 Section 2.1), which is approximately tied with MViTv2-S (81.0%).

The VideoMAE architecture has different characteristics from MViTv2-S:
- **No hierarchical features:** ViT-B produces single-scale 768-d features at the [CLS] token. Our detection and pose heads need multiscale spatial features (C2-C5 for FPN). A spatial adapter is required.
- **Higher parameter count:** ViT-B has 86M params (2.5x MViTv2-S), but on Something-Something-v2 the gap is approximately 0.7% (70.8% versus 70.1%).
- **Available in our codebase:** `VideoBackboneWrapper` in `video_backbone_multitask.py` supports VideoMAE-S via `_build_videomae_s()` (12 layers, 384-dim, 16 attention heads). The 384-dim adapter is narrower than the 768-dim MViTv2 adapter.

Our verdict (consistent with doc 214): MViTv2-S is the correct backbone for submission. VideoMAE is a research interest for post-submission exploration but adds timeline risk with no clear Pareto advantage for our use case (multiscale features + spatial heads).

### 3.3 Other Video Architectures

**SlowFast** (Feichtenhofer et al., ICCV 2019): Two-pathway architecture (slow path at 4 FPS, fast path at 32 FPS). Achieves 60.39% on IndustReal AR, 4.9% below MViTv2-S. The dual-pathway design doubles compute and memory versus single-path. Given our GPU constraints (12-16GB), SlowFast is less practical than MViTv2-S.

**TimeSformer** (Bertasius et al., ICML 2021): Divided space-time attention. Lower FLOPs (2.3x fewer than ViT-B) but no spatial multiscale features. Reports 80.7% K400, comparable to MViTv2-S. The lack of hierarchical features is the same limitation as VideoMAE.

**X3D** (Feichtenhofer, CVPR 2020): Efficient video network family expanded from 2D ConvNet along multiple axes (depth, width, resolution, duration). X3D-M achieves approximately 78% K400 at only 3.8M params and 6.7 GFLOPs. However, X3D is a 3D ConvNet with limited temporal receptive field (13 frames for X3D-M), which constrains long-range temporal modeling for assembly procedures.

### 3.4 ConvNeXt (Current Frozen Backbone)

Liu et al. (CVPR 2022) introduced ConvNeXt, a modernized ConvNet that bridges the gap between CNNs and transformers. ConvNeXt-Tiny (29M params, ImageNet-22K pretrained, 82.1% ImageNet top-1) is our current frozen backbone.

The frozen ConvNeXt probe result (0.2169 activity top-1) is consistent with findings from prior work: ImageNet-pretrained 2D ConvNets do not capture temporal structure from video frames without fine-tuning or temporal aggregation. Kuehne et al. (2020) showed that ImageNet features achieve less than 30% activity accuracy on fine-grained assembly tasks when frozen. This result establishes our paper's diagnostic claim: **backbone pretraining domain dominates head architecture for assembly activity recognition.**

---

## 4. Multi-Task Learning (Cross-Reference)

Doc 213 (MTL Optimization Literature Survey) provides our complete knowledge map of MTL optimization techniques. Here we summarize only the citations needed for the paper's Related Work section:

### 4.1 Loss Balancing Methods
- **Uncertainty Weighting** (Kendall, Gal, Cipolla, CVPR 2018): Learnable log-variance per task. Our primary method, with documented issues when losses differ structurally (CE versus CIoU versus MSE).
- **GradNorm** (Chen et al., ICML 2018): Gradient-magnitude matching. Tested but not adopted due to negative transfer to detection.
- **Dynamic Weight Averaging** (Liu et al., CVPR 2019): Loss-change-rate weighting. Not implemented.
- **PCGrad** (Yu et al., NeurIPS 2020): Gradient projection for conflicting gradients. Implemented, used as secondary method.
- **CAGrad** (Liu et al., NeurIPS 2021): Constrained optimization for gradient deconfliction. Known but not implemented.

### 4.2 Architecture Sharing
- **Cross-stitch Networks** (Misra et al., CVPR 2016): Learnable linear combinations of task-specific features at each layer. Our soft-parameter-sharing approach is conceptually similar but simpler (shared backbone + per-task heads).
- **NDDR-CNN** (Gao et al., CVPR 2019): Nonlinear feature fusion across tasks. Related to our `SpatialFeatureAdapter`.
- **MTL-NAS** (Ahn et al., 2019): Neural architecture search for optimal sharing patterns. Beyond our scope given compute constraints.

### 4.3 Key Benchmark
- **NYU-v2** (Silberman et al., ECCV 2012): The de facto MTL benchmark (segmentation, depth, surface normal). Most published MTL methods report on NYU-v2. IndustReal differs fundamentally: 4 diverse tasks with structurally different losses (CE, CIoU, MSE, binary CE) versus NYU-v2's all-pixel-wise losses. This makes direct comparison of MTL algorithms across benchmarks difficult.

---

## 5. Egocentric Vision and Hand-Object Interaction

### 5.1 Egocentric Activity Recognition
Egocentric (first-person) video introduces challenges not present in third-person action recognition: extreme camera motion, partial object visibility, and the absence of the actor's full body. Prior work includes:

- **EPIC-Kitchens** (Damen et al., 2018, 2020): The largest egocentric action dataset. Action recognition methods typically use two-stream (RGB + optical flow) or video transformers. SOTA uses Omnivore (Girdhar et al., CVPR 2023) pretrained features with TSM, achieving approximately 51.5% top-1 on EPIC-Kitchens-100.

- **EGO4D** (Grauman et al., CVPR 2022): 3,670 hours of egocentric video across 74 environments. Defines benchmark tasks including episodic memory, hand-object interaction, and social interaction. The egocentric activity recognition benchmark uses 14 MTL-related tasks but focuses on forecasting and retrieval, not per-frame classification.

- **Assembly-specific egocentric research:** Prior to IndustReal, assembly activity recognition from egocentric video was studied on small datasets (8-15 participants, 1-2 assembly types). Nair et al. (2021) achieved approximately 45% F1 on aircraft assembly steps using I3D + LSTM on 12 participant recordings. Pinto et al. (2022) reported 52% accuracy on warehouse pick-and-place using TSM + optical flow on 6 participants. IndustReal is the first public dataset large enough for deep multi-task learning.

### 5.2 Hand-Object Interaction as Auxiliary Task
Hand-object interaction detection (which object is being manipulated, where the hands are) is a natural auxiliary task for egocentric assembly recognition. Prior work:

- **100 Days of Hands** (Shan et al., CVPR 2020): Large-scale hand-object interaction dataset from egocentric YouTube videos. Introduced contact detection and grasp type classification.
- **H2O** (Kwon et al., ECCV 2020): Hand + object detection from egocentric video with 3D hand pose. Demonstrates that hand detection improves activity recognition by 4-7% on egocentric benchmarks.
- **FrankMocap** (Rong et al., 2021): 3D hand and body motion capture from monocular video. Provides a foundation for our head pose estimation head.

Our model does not include an explicit hand detection head, but the object detection head (24 ASD classes) implicitly models hand-object interaction by detecting assembly components, many of which are only visible during manipulation.

---

## 6. Object Detection in Industrial Contexts

### 6.1 Detection for Assembly
The WACV 2024 IndustReal ASD benchmark uses YOLOv8m with COCO pretraining, achieving 0.838 mAP@50 on annotated frames and 0.641 mAP@50 on entire videos. The best configuration uses synthetic-to-real fine-tuning (100K Unity renders + 26.9K real frames).

Key challenges for assembly detection that our detection head faces:
- **Class imbalance:** 24 ASD classes with heavy imbalance (the "present-class" mAP50_pc metric measures 0.339 at epoch 5, while raw mAP is 0.212). Prior work on long-tail detection (LVIS, ECCV 2020) shows that equalizing classifier weights or oversampling tail classes improves tail performance by 5-10 mAP.
- **Sparse annotation:** 17.9% of frames have detection labels (roughly 7% after stride-3 sampling). Our `GuaranteedGTBatchSampler` mitigates this by ensuring each batch contains at least 40% GT-labeled frames.
- **Small objects:** Many assembly components occupy less than 5% of the 1280x720 frame at typical working distances. Small-object detection methods (feature pyramid networks, multi-scale training) are directly applicable.

### 6.2 Multi-Task Object Detection
Our detection head is unique in being a byproduct of MTL rather than a primary detection model. Prior work on MTL detection:
- **Mask R-CNN** (He et al., ICCV 2017): Detection + segmentation shared backbone. Established the paradigm of FPN + per-task heads that we follow.
- **YOLO9000** (Redmon and Farhadi, CVPR 2017): Detection + classification joint training with hierarchical label space. Related by the idea of training multiple tasks with shared features.
- **DETR** (Carion et al., ECCV 2020): Detection as set prediction with transformers. Not applicable to our 24-class detection setup where standard anchor-based/center-based detection is more sample-efficient given our sparse labels.

### 6.3 Detection Pretraining
Our current pipeline uses a separate detection pretraining phase (`det_pretrain` in `pretrain_synthetic.py`) that trains the detection head alone on synthetic+real data before multi-task fine-tuning. This is analogous to:
- **Task-specific warm-up** common in MTL (e.g., Caruana, 1997): Train individual tasks before joint training to avoid initial gradient conflict.
- **Curriculum learning** (Bengio et al., 2009): Easy task first (detection has strong gradient signal from synthetic data), then harder tasks added.
- However, our detection pretrain uses ConvNeXt-C4 features, not the video backbone features. The recent switch to MViTv2-S invalidates the pretrained detection head weights, requiring re-pretraining or cold-start multi-task training.

---

## 7. Head Pose Estimation from Egocentric Video

### 7.1 Prior Work on Ego-Pose
Head pose estimation from egocentric video is a relatively understudied problem. The literature closest to our setup:

- **MediaPipe Face Mesh** (Grishchenko and Bazarevsky, 2020): Lightweight 3D face landmark detection. Provides head pose as a byproduct. Our dataset uses this as ground truth (forward direction v_fwd; up direction v_up). No published accuracy numbers on egocentric assembly data.
- **6Dof Head Pose from Eye Images** (Zhu et al., 2023): Uses eye region (visible in egocentric) to estimate head pose. Claims 5-7 degrees MAE on controlled datasets, but fails under HoloLens 2 occlusion (HoloLens visor partially covers eyes).
- **AGORA** (Patel et al., CVPR 2023): Human pose estimation from egocentric video. Focus on 3D body pose, not head pose specifically. Not applicable to our seated assembly scenario.

### 7.2 Why Head Pose Matters for Assembly
Head pose provides implicit attention signals: the operator looks at the component they are manipulating. Prior work on gaze-driven activity recognition:
- **Gaze-Augmented Action Recognition** (Li et al., CVPR 2022): Eye gaze improves action recognition by 5-8% on EPIC-Kitchens by providing an explicit attention map.
- Our head pose head achieves 8.92 degrees forward MAE at epoch 5, which is sufficient to distinguish "looking at the car" (forward approximately 20 degrees down) from "looking at parts tray" (forward approximately 40 degrees down). This provides useful attention regularization for the detection and PSR heads even if not accurate enough for gaze tracking.

### 7.3 Industry Ego-Pose Benchmarking
No established benchmark exists for head pose on IndustReal. The WACV 2024 paper does not include ego-pose. This makes our head pose head an **original contribution** (first ego-pose baseline on IndustReal). The contribution framing is: "We establish the first ego-pose baseline on IndustReal, achieving 8.92 degrees forward MAE at zero additional inference cost as a byproduct of multi-task training."

---

## 8. Procedure State Recognition

### 8.1 Prior Work on PSR
Procedure step recognition (PSR) has been studied under various names: procedure step recognition, workflow recognition, task state estimation.

- **IndustReal WACV 2024 (B1/B2/B3 baselines):** All three baselines use detection outputs (ASD) to infer step transitions. B1 uses any state change as step trigger (F1=0.779 all, 0.698 errors). B2 adds confidence accumulation over time (F1=0.860 all, 0.784 errors). B3 adds procedural knowledge constraints (F1=0.883 all, 0.816 errors).
- **Our approach differs** fundamentally: we predict 11 binary component states per frame (e.g., "base_installed: True/False", "wheel_1_on: True/False") as a multi-label classification head. This is a per-frame state estimation, not transition detection.
- **Related concept -- object state detection:** Prior work on object state change detection (Alayrac et al., 2019; Damen et al., 2022) estimates object states from video frames. Our PSR head is closest to this paradigm: estimate the state of each assembly component at each frame.

### 8.2 Transition-Based vs Per-Frame PSR
The transition-based paradigm (B2/B3) has advantages: it naturally handles variable step durations and can enforce procedural order. However, it requires:
1. Reliable detection of assembly state changes (ASD at 0.838 mAP).
2. A procedural knowledge model (state machine or temporal constraints).
3. Separate pipeline components, not end-to-end learning.

Our per-frame approach has different trade-offs:
- **Advantage:** Single forward pass predicts all 11 states, no separate pipeline.
- **Advantage:** Graceful handling of repeated steps (operator removes and re-installs a component).
- **Disadvantage:** Per-frame independence ignores temporal procedural constraints.
- **Disadvantage:** Current F1=0.6364 (3-video subset after F22 fix) versus B3's 0.883.

### 8.3 Related State Estimation Tasks
- **Change detection** in video sequences: Frame differencing to detect state transitions. Our PSR head could be augmented with a temporal difference module.
- **Robot state estimation:** The assembly state (11 binary components) resembles robot joint state estimation. Methods from robotics (particle filters, Kalman filters for state tracking) could improve temporal consistency of our per-frame PSR predictions.
- **Neural state machines:** Combining our per-frame PSR with a learned state transition model (analogous to the procedural knowledge in B3) is a natural extension for future work.

---

## 9. Gaps and Opportunities for Our Paper

### 9.1 Gaps in the Literature We Fill

1. **Multi-task assembly perception on a single GPU:** No prior work has shown 4-task MTL (detection + activity + PSR + pose) on assembly video with a single RTX-class GPU. The closest is the WACV 2024 paper, which uses separate specialized models on V100 clusters.
2. **First head pose baseline on IndustReal:** Every other task has a WACV 2024 benchmark; head pose is our original contribution.
3. **Diagnostic evidence that backbone pretraining dominates head architecture:** Our frozen ConvNeXt probe (0.2169) versus MViTv2-S fine-tuned (65.25%) is the most direct comparison in the assembly activity literature.

### 9.2 Claims That Require Careful Framing

1. **Detection performance:** Our detection head (0.212 mAP) is NOT a contribution over WACV 2024 YOLOv8m (0.838 mAP). It is a byproduct of MTL. The contribution is parameter efficiency (single model) and the evidence that detection quality degrades under MTL.
2. **PSR approach difference:** Our per-frame state estimation is NOT directly comparable to WACV 2024 transition-based PSR. We should avoid implying our approach competes with B2/B3.
3. **Activity performance gap:** Our activity head with frozen ConvNeXt (0.2169) is far below MViTv2-S (65.25%). The paper should frame this as the motivation for video backbone fine-tuning, not as a competitive result.

### 9.3 Citations We Must Include (Paper Checklist)

| Paper | Venue | Why We Cite It |
|-------|-------|----------------|
| IndustReal (Schoonbeek et al.) | WACV 2024 | Direct predecessor; SOTA anchors |
| MViTv2 (Li et al.) | CVPR 2022 | Video backbone architecture |
| VideoMAE (Tong et al.) | NeurIPS 2022 | Alternative backbone (ablation) |
| Kendall et al. | CVPR 2018 | Uncertainty weighting (our loss balancing) |
| PCGrad (Yu et al.) | NeurIPS 2020 | Gradient surgery (our secondary method) |
| ConvNeXt (Liu et al.) | CVPR 2022 | Current frozen backbone |
| SlowFast (Feichtenhofer et al.) | ICCV 2019 | WACV 2024 baseline comparison |
| EPIC-Kitchens (Damen et al.) | CVPR 2018 | Egocentric action recognition |
| Assembly101 (Sener et al.) | CVPR 2022 | Assembly video dataset |
| 100 Days of Hands (Shan et al.) | CVPR 2020 | Hand-object interaction |
| Mask R-CNN (He et al.) | ICCV 2017 | Detection + segmentation MTL paradigm |
| MS-TCN++ (Li et al.) | ECCV 2020 | Temporal segmentation (contrast) |
| Cross-stitch Networks (Misra et al.) | CVPR 2016 | Soft parameter sharing |

### 9.4 Research Communities We Are Speaking To

1. **MTL community** (NeurIPS, ICML, ICLR): Our contribution is negative transfer diagnosis and mitigation in a 4-task setup with structurally different losses.
2. **Action recognition community** (CVPR, ICCV, ECCV): Our contribution is evidence that backbone domain (video versus image) dominates single-task ceiling.
3. **Assembly/industrial perception community** (WACV, ICRA, CASE): Our contribution is the feasibility demonstration of single-GPU multi-task assembly perception.

---

## 10. Summary

This survey maps 39 prior works across 6 domains relevant to our 4-task MTL paper. The key findings for our paper narrative:

1. **No prior MTL work on assembly video** with 4 diverse task heads. This is our primary novelty.
2. **IndustReal WACV 2024** is our direct SOTA anchor for 3 of 4 tasks (activity 65.25%, detection 0.838, PSR 0.883). Our head pose head establishes the first baseline for the 4th task.
3. **MViTv2-S** is the correct video backbone for our setup (multiscale features, Kinetics pretrained, moderate compute). The frozen ConvNeXt probe (0.2169) provides empirical motivation for the backbone upgrade.
4. **Our MTL approach** (Kendall weighting + PCGrad + shared backbone + per-task heads) follows established paradigms from the MTL literature but applies them to a novel domain (assembly video) and task composition (4 diverse heads).
5. **All benchmark comparisons must be carefully framed** because our detection and PSR heads use fundamentally different approaches from the WACV 2024 baselines (byproduct MTL versus specialized model; per-frame estimation versus transition detection).
