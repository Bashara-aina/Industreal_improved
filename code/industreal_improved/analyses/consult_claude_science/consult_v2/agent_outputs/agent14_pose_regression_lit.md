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

# Agent 14: Pose Regression MTL Literature Survey

**Focus:** Rotation representations, regression losses, head pose estimation, multi-task learning (MTL) integration, and egocentric pose estimation.

**Date:** 2026-07-13

---

## 1. Rotation Representations

### 1.1 6D Continuous Representation

**Zhou et al. (CVPR 2019)** -- "On the Continuity of Rotation Representations in Neural Networks"
- arXiv:1812.07035v4 (published 2018-12, CVPR 2019)
- Authors: Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li
- **Key result:** Proved that 3D rotations in R^4 (quaternions) or lower (Euler angles) are discontinuous representations. The paper shows that a 6D representation (two 3D vectors) yields a continuous mapping to SO(3) and is easier for neural networks to learn. Used Gram-Schmidt orthogonalization to project the 6D output to SO(3).
- **Impact:** Foundational for all subsequent 6D rotation regression work. Our current implementation follows this approach.

### 1.2 9D SVD-on-Manifold

**Levinson et al. (2020)** -- "An Analysis of SVD for Deep Rotation Estimation"
- arXiv:2006.14616v1 (2020)
- Authors: Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, Ameesh Makadia
- **Key result:** Shows that SVD orthogonalization is the natural choice for projecting onto rotation groups. A 9D representation (3x3 matrix) + SVD projection achieves SOTA on multiple deep learning rotation tasks. Provides theoretical analysis showing SVD is the geometry-aware way to project onto SO(n).
- **Relation to 6D:** While Zhou uses Gram-Schmidt on 6D, Levinson shows that the full 9D matrix with SVD projection can work better, particularly when the network outputs close-to-orthogonal matrices. The SVD ensures the closest orthogonal matrix in Frobenius norm.

### 1.3 Quaternion and Euler Angle Representations

- **Quaternions** (4D): Known to have a discontinuity in R^4 at the antipodal point (q and -q represent the same rotation). Zhou et al. showed unit quaternions are a discontinuous representation.
- **Euler angles** (3D): Suffer from gimbal lock and discontinuities. Standard representation before 2019, still used by WHENet and some HPE methods.
- **Axis-angle** (3D): Subject to the 360-degree wrapping discontinuity at +pi/-pi (the "continuous" representation at the cost of angular magnitude).

### 1.4 Which Representation Minimizes Error for Head Pose?

**6DRepNet (Hempel et al. 2022)** -- "6D Rotation Representation For Unconstrained Head Pose Estimation"
- arXiv:2202.12555v2 -- IEEE ICIP 2022
- Authors: Thorsten Hempel, Ahmed A. Abdelrahman, Ayoub Al-Hamadi
- **Key result:** First end-to-end head pose estimation method using the continuous 6D rotation matrix representation. Explicitly uses rotation matrix formalism for ground truth labels. Reports significant improvements: "outperforms other state-of-the-art methods by up to 20%" on BIWI and AFLW2000 benchmarks.
- **Loss:** Uses geodesic distance-based loss penalizing the network with respect to SO(3) manifold geometry, combined with a cosine (angular) loss.
- **Verdict on representation:** The 6D representation is currently the best-established approach for head pose, with both theoretical (Zhou 2019) and empirical (Hempel 2022) support.

**Hempel et al. (2023)** -- "Towards Robust and Unconstrained Full Range of Rotation Head Pose Estimation"
- arXiv:2309.07654v1 (2023-09)
- Key: Extension of 6DRepNet to full rotation range, addressing challenges in wide-angle head poses.

---

## 2. Geodesic Loss on SO(3)

### 2.1 Standard Geodesic (Angular) Loss

The geodesic distance on SO(3) for rotation matrices R_1 and R_2 is:

d(R_1, R_2) = arccos((trace(R_1^T R_2) - 1) / 2)

This gives the angle of the residual rotation in degrees (geodesic error). This is used by:
- 6DRepNet (Hempel 2022)
- Multiple pose estimation methods
- Our current implementation

**Advantages:** Mathematically well-founded metric on the SO(3) manifold. Directly corresponds to angular error.
**Disadvantages:** The arccos gradient can explode near 0 and pi. Slow to compute. Numerically unstable near the identity rotation.

### 2.2 Cosine Rotation Loss (Avoiding arccos)

Instead of computing the geodesic distance, many methods use a cosine-based loss:
L_cos = 1 - (trace(R_1^T R_2) - 1) / 4

This is equivalent to the squared geodesic distance in the small-angle regime [cos(theta) ~= 1 - theta^2/2]. It avoids the numerical issues of arccos.

**Our current implementation** uses both geodesic angular error + cosine rotation loss.

### 2.3 Riemannian Optimization

**Becigneul and Ganea (NeurIPS 2018)** -- "Riemannian Adaptive Optimization Methods"
- arXiv:1810.00760v2 (NeurIPS 2018)
- Authors: Gary Becigneul, Octavian-Eugen Ganea
- **Key result:** Generalized Adam, Adagrad, and AMSGrad to Riemannian manifolds (including product of manifolds). Provides convergence proofs for geodesically convex objectives. Riemannian adaptive methods show faster convergence on manifold-valued targets.
- **Relevance:** The rotation space SO(3) is a Riemannian manifold. Using Riemannian-aware optimization can improve pose training. However, most head pose methods (including ours) still use standard Euclidean optimizers on the 6D representation.

### 2.4 Huberised Geodesic Loss

The Huberised geodesic loss combines L2 behavior for small errors (smooth quadratic) with L1 behavior for large errors (robust to outliers):

L_huber_geo(theta) = { theta^2 / (2*delta), if |theta| < delta; |theta| - delta/2, otherwise }

Where theta = geodesic distance in radians.

**Not directly from a single paper** but appears as a practical technique in multiple rotation regression works. The standard Huber loss (from robust statistics, Huber 1964) is applied to the geodesic angle. It is used in some pose refinement pipelines to reduce the impact of outlier predictions.

### 2.5 Probabilistic Rotation Distributions

**Yin et al. (2023)** -- "A Laplace-inspired Distribution on SO(3) for Probabilistic Rotation Estimation"
- arXiv:2303.01743v1 (2023)
- Authors: Yingda Yin, Yang Wang, He Wang, Baoquan Chen
- **Key result:** Proposes a Rotation Laplace distribution on SO(3) that is robust to outliers. The Laplace distribution enforces more gradient in low-error regions compared to Gaussian-like Bingham/matrix Fisher distributions. SOTA on probabilistic rotation regression.
- **Citation count:** 14 (from semantic scholar)

**Peretroukhin et al. (2019)** -- "Probabilistic Regression of Rotations using Quaternion Averaging and a Deep Multi-Headed Network"
- arXiv:1904.03182v2 (2019)
- Authors: Valentin Peretroukhin, Brandon Wagstaff, Matthew Giamou, Jonathan Kelly
- **Key result:** Probabilistic rotation regression using quaternion averaging with a multi-headed network that outputs multiple hypothesis for the rotation, which are then probabilistically combined.

### 2.6 UNVERIFIED: Murphy et al. (2021) "Probabilistic rotation"

Cannot verify this reference. Searched arXiv, semantic scholar, Crossref, DBLP -- no paper matching "Murphy et al. 2021" on probabilistic rotation estimation was found. May be a hallucinated reference. Related verified works: Yin et al. (Rotation Laplace, 2023), Peretroukhin et al. (2019), Kendall et al. (uncertainty weighting, 2017).

---

## 3. Head Pose Estimation Methods

### 3.1 SOTA Methods and Benchmarks

| Method | Venue | Year | Representation | Loss | Notes (verified from papers) |
|--------|-------|------|----------------|------|------|
| **FSA-Net** | CVPR | 2019 | Euler angles | Angular + classification | Feature aggregation at multiple scales; 268 citations (crossref) |
| **WHENet** | BMVC | 2020 | Euler angles | Multi-loss | First full-range head yaw method |
| **6DRepNet** | ICIP | 2022 | 6D rotation matrix | Geodesic + cosine | Up to 20% improvement over SOTA on BIWI/AFLW2000 |
| **img2pose** | CVPR | 2021 | 6DoF pose | Pose regression | Joint detection + pose, no landmarks needed |
| **DirectMHP** | arXiv | 2023 | Flexible (Euler) | Multi-task detection + pose | First end-to-end multi-person HPE |
| **SemiUHPE** | arXiv | 2024 | 6D rotation | Semi-supervised rotation regression | Semi-supervised approach using entropy-based filtering |

#### FSA-Net (Yang et al. CVPR 2019)
- DOI: 10.1109/CVPR.2019.00118
- Authors: Tsun-Yi Yang, Yi-Ting Chen, Yen-Yu Lin, Yung-Yu Chuang
- **Approach:** Learns fine-grained structure aggregation for head pose estimation from a single image. Uses a feature aggregation module combining spatial and channel-wise attention. Predicts Euler angles via classification (binning angles) + regression.
- **Benchmarks:** Reports results on BIWI and AFLW2000. MAE on BIWI reported in single digits.

#### 6DRepNet (Hempel et al. 2022)
- **Approach:** Applies the continuous 6D rotation representation (Zhou et al. 2019) specifically to head pose. Uses rotation matrix ground truth labels (no Euler angle ambiguity). MLP-based head following a ResNet backbone.
- **Loss:** Geodesic distance on SO(3) + mean squared error on rotation matrix components.
- **Benchmarks:** Outperforms prior SOTA on AFLW2000 and BIWI "by up to 20%". Exactly benchmark numbers not verified from the paper source in this analysis, but the 20% relative improvement claim is verified from the paper abstract.

#### WHENet (Zhou & Gregson 2020)
- arXiv:2005.10353v2
- **Approach:** Predicts Euler angles via multi-loss approach. Designed for full-range yaw (0-360 degrees). Uses panoptic dataset annotations for training. Compact and mobile-friendly.

#### DirectMHP (Zhou et al. 2023)
- arXiv:2302.01110v2
- Authors: Huayi Zhou, Fei Jiang, Hongtao Lu
- **Approach:** Direct end-to-end multi-person head pose estimation (MPHPE). Jointly regresses bounding boxes and head orientations. Constructs benchmarks from AGORA and CMU Panoptic datasets for full-range multi-person evaluation.

#### img2pose (Albiero et al. CVPR 2021)
- arXiv:2012.07791v2
- Authors: Vitor Albiero, Xingyu Chen, Xi Yin, Guan Pang, Tal Hassner
- **Approach:** Regresses 6DoF face pose directly from images without face detection or landmark localization. Faster R-CNN based. Also surpasses SOTA on WIDER FACE detection.
- **Benchmarks:** Reports outperforming SOTA on AFLW2000-3D and BIWI.

### 3.2 Standard Benchmarks

| Dataset | Description | Metric | Typical SOTA Error |
|---------|-------------|--------|-------------------|
| **BIWI** | 20 videos of head poses from Kinect | Mean Absolute Error (MAE) | ~3-5 degrees (yaw, pitch, roll) |
| **AFLW2000-3D** | 2000 face images with 3D annotations | MAE | ~4-6 degrees (yaw, pitch, roll) |
| **300W-LP** | Large synthetic pose dataset | MAE | ~3-5 degrees |

---

## 4. Pose Estimation in Multi-Task Learning

### 4.1 HyperFace (Ranjan et al. 2017)

- arXiv:1603.01249v3
- Authors: Rajeev Ranjan, Vishal M. Patel, Rama Chellappa
- **Approach:** Deep CNN fusing intermediate layers + separate CNN, followed by MTL for face detection, landmark localization, pose estimation, and gender recognition simultaneously.
- **Architecture:** AlexNet-based backbone with a fusion CNN that combines features from multiple layers. Pose is one of 4 tasks.
- **Loss handling:** Uses a combined multi-task loss with fixed loss weights for each task. Does NOT address the scale imbalance problem explicitly.

### 4.2 DirectMHP (Multi-Person Detection + Pose)

- As described in Section 3.1, DirectMHP joints detection and head pose in a single network.
- Treats pose as an "auxiliary attribute" of the head bounding box.
- Uses multiple losses: detection loss (classification + regression) + pose loss (regression).
- Does not explicitly discuss loss-weighting strategy but uses joint feature sharing.

### 4.3 img2pose (Detection via Pose)

- Unique approach: replaces bounding box detection labels with 6DoF pose labels.
- Pose serves both as the geometry representation AND the detection mechanism.
- Suggests that 6DoF pose contains more information than bounding boxes.

---

## 5. Pose from Video: Temporal Consistency

### 5.1 Current State

No verified paper specifically addressing temporal consistency in head pose was found in this survey. The literature on temporal head pose refinement is sparse. Approaches that could be applicable:

- **Kalman filtering on SO(3):** Apply a Kalman filter on the Lie algebra (so(3)) for smoothing pose estimates over time. Common in robotics and SLAM.
- **Temporal smoothing:** Simple exponential moving average on the 6D representation or geodesic interpolation between frames.
- **Video-based refinement:** Using temporal context (3D convolutions, LSTMs) to refine per-frame estimates.

**UNVERIFIED:** The user mentioned "temporal filtering" and "video-based pose refinement" -- no specific papers were verified in this category. Crossref search found "Spatial-Temporal Pyramid for 3D Human Head Pose Prediction" (Du & Ikenaga, 2024) but full details not accessible.

### 5.2 Relevant Techniques from Related Fields

- **Pose averaging on SO(3):** Karcher mean (Riemannian center of mass) on SO(3) for averaging pose estimates over a temporal window.
- **Slerp interpolation:** Spherical linear interpolation for quaternions between video frames.

---

## 6. Pose Loss Weighting in Multi-Task Learning

### 6.1 The Scale Imbalance Problem

Pose regression loss values (geodesic error in degrees, or rotation matrix MSE) are typically in the range of 10^0 to 10^3, while classification losses (cross-entropy) are in the range of 10^0 to 10^1. This creates a fundamental imbalance when combining these losses in MTL.

### 6.2 Uncertainty-Based Weighting

**Kendall, Gal, Cipolla (CVPR 2018)** -- "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
- arXiv:1705.07115v3 (CVPR 2018)
- Authors: Alex Kendall, Yarin Gal, Roberto Cipolla
- **Key result:** Proposes weighting multi-task losses using homoscedastic (task-specific) uncertainty. The loss is:
  L_total = sum_t (1 / (2 * sigma_t^2) * L_t + log(sigma_t))
  where sigma_t is a learnable noise parameter for task t.
- **Handling different scales:** The log(sigma_t) term prevents sigma_t from growing unbounded. Higher regression uncertainty sigma automatically downweights the regression loss contribution.
- **Impact:** Canonical approach for MTL loss weighting. Used in scene understanding (depth + segmentation + instance segmentation).
- **Relevance to our case:** Pose regression (scale ~10^3) vs. detection/classification (scale ~1). The uncertainty weighting would automatically learn to balance these.

### 6.3 Practical Approaches from Landmark/Pose MTL Papers

- **Fixed weighting:** HyperFace (Ranjan 2017) uses fixed scalar weights for each task loss, tuned by hand.
- **Learnable weighting:** Following Kendall et al. (2018), many modern MTL approaches use uncertainty weighting or gradient balancing (GradNorm, PCGrad).
- **Gradient normalization:** Normalizing gradient magnitudes across tasks (e.g., GradNorm by Chen et al. 2018) prevents one task from dominating updates.
- **For pose specifically:** Our context has pose loss ~680-785 degrees total (mean ~8-9 degrees per sample). Scaling the loss to match classification scale (e.g., multiply by 0.01-0.05) or using uncertainty weighting is the standard approach.

---

## 7. Uncertainty-Aware Pose

### 7.1 Why Uncertainty for Head Pose?

Head pose from egocentric video has inherent ambiguity due to:
- Self-occlusion (partial views)
- Profile views (yaw near +/- 90 degrees)
- Poor lighting / motion blur
- Gaze direction not always aligned with head orientation

**Figari Tomenotti et al. (2023)** -- "Head Pose Estimation with Uncertainty"
- SSRN DOI: 10.2139/ssrn.4399244 (preprint, 2023)
- Authors: Federico Figari Tomenotti, Nicoletta Noceti, Francesca Odone
- **Details:** Limited availability (SSRN preprint). Explores uncertainty quantification for head pose estimation.

### 7.2 Probabilistic Rotation Regression

The Rotation Laplace distribution (Yin et al. 2023, described above) is the most relevant approach for probabilistic rotation estimation. It provides both a pose estimate AND an uncertainty measure (the concentration parameter of the Laplace distribution on SO(3)).

**Expected error range for egocentric head pose:**
- Typical: 5-15 degrees MAE on yaw for in-the-wild scenarios
- Lower bound: 3-5 degrees MAE for controlled datasets (BIWI, AFLW2000)
- Egocentric: Typically higher error due to motion blur, partial views, wide baselines

---

## 8. Egocentric Head Pose

### 8.1 Ego4D Dataset

**Grauman et al. (2021)** -- "Ego4D: Around the World in 3,000 Hours of Egocentric Video"
- arXiv:2110.07058v3 (2021)
- **Scale:** 3,670 hours, 931 camera wearers, 74 locations, 9 countries.
- **Head pose relevance:** The Ego4D benchmark suite includes tasks related to social interaction (e.g., "looking at me" detection) and head/hand interaction understanding. Head pose is a key cue for understanding camera wearer attention.
- **Egocentric head pose challenges:** Extreme head motion (running, cooking), occluded faces, and non-frontal perspectives.

### 8.2 EPIC-KITCHENS Dataset

**Damen et al. (2018)** -- "Scaling Egocentric Vision: The EPIC-KITCHENS Dataset"
- arXiv:1804.02748v2 (2018)
- **Scale:** 55 hours, 39.6K action segments, 454.3K object bounding boxes.
- **Head pose relevance:** While EPIC-KITCHENS focuses on hand-object interactions, head pose provides contextual information about the camera wearer's focus of attention.
- **Annotation:** Narrated by participants themselves (reflecting true intention). No explicit head pose annotations in the main dataset.

### 8.3 Egocentric Head Pose Challenges

- **Camera perspective:** Head-mounted camera means head pose is inferred from limited visual cues (nose position, facial features when visible).
- **Motion blur:** Head movement during egocentric capture creates significant motion blur.
- **Occlusion:** Hands, objects frequently occlude the face.
- **Dynamic range:** Head orientations span a much wider range than in frontal-face datasets.

---

## 9. Verdict: Actionable Findings

### Finding 1: Stick with 6D representation, consider adding 9D+SVD

The 6D rotation representation (Zhou et al. CVPR 2019, arXiv:1812.07035) is well-established for head pose estimation and is used by the current SOTA methods (6DRepNet, Hempel 2022; SemiUHPE, Zhou 2024). However, the 9D representation with SVD projection (Levinson et al. 2020, arXiv:2006.14616) shows that the full 3x3 matrix with SVD orthogonalization can outperform 6D+Gram-Schmidt, especially when the network outputs near-orthogonal matrices. **Recommendation:** Test 9D+SVD as a drop-in replacement for 6D+Gram-Schmidt in the current MLP head.

### Finding 2: Your current loss (geodesic + cosine) is standard; consider Huberised geodesic

The combination of geodesic angular error + cosine rotation loss is standard in the literature (Hempel 2022). The cosine loss avoids numerical issues of arccos. If outlier predictions (large errors) are problematic, a Huberised version of the geodesic loss can be applied, where errors above a threshold delta transition from L2 to L1. **Recommendation:** Monitor the distribution of per-sample geodesic errors. If there are heavy tails (outliers > 30 degrees), add Huberisation with delta = 10-15 degrees.

### Finding 3: Use uncertainty weighting (Kendall et al. CVPR 2018) for MTL loss balancing

The canonical approach for balancing pose regression (scale ~10^3) with classification losses (scale ~1) is uncertainty-based weighting (arXiv:1705.07115). This automatically learns the appropriate task weights via homoscedastic noise parameters sigma_t, avoiding manual tuning. **Recommendation:** Replace the current fixed loss weight with a learnable uncertainty parameter for the pose regression loss, with the log(sigma) regularizer to prevent unbounded growth. Initialize sigma so that 1/(2*sigma^2) starts at the current manually-set weight.

### Finding 4: Benchmark against 6DRepNet and img2pose on BIWI/AFLW2000

The direct comparison baselines for our pose head should be:
- 6DRepNet (Hempel 2022, arXiv:2202.12555): Uses same 6D representation, ResNet backbone, geodesic + cosine loss
- img2pose (Albiero et al. CVPR 2021, arXiv:2012.07791): 6DoF pose without detection, SOTA results reported on BIWI and AFLW2000-3D

**Recommendation:** Evaluate your model on the standard BIWI and AFLW2000-3D benchmarks and report MAE (yaw, pitch, roll) and geodesic error. Target: < 5 degrees MAE on BIWI for competitive results.

### Finding 5: Uncertainty-aware pose is a differentiator for egocentric applications

For egocentric head pose (our use case), uncertainty estimation is critical because:
- Input quality varies dramatically (motion blur, occlusion, extreme angles)
- The Rotation Laplace distribution (Yin et al. 2023, arXiv:2303.01743) provides both the pose estimate AND uncertainty

**Recommendation:** Extend the current deterministic MLP head to output a distribution on SO(3). The simplest approach is to predict both the 6D rotation parameters AND a concentration parameter (kappa) for a distribution on SO(3). This allows:
- Downweighting uncertain predictions in downstream tasks
- Temporal filtering weighted by per-frame uncertainty
- Providing confidence thresholds for predictions

---

## Appendix: Paper Status Summary

| Paper | Verified | arXiv ID / DOI | Venue | Notes |
|-------|----------|---------------|-------|-------|
| Zhou et al. "6D rotation" | YES | arXiv:1812.07035 | CVPR 2019 | Cornerstone for 6D rep |
| Levinson et al. "SVD rotation" | YES | arXiv:2006.14616 | NeurIPS 2020 | 9D + SVD analysis |
| Hempel et al. "6DRepNet" | YES | arXiv:2202.12555 | ICIP 2022 | 6D rep for head pose |
| Yang et al. "FSA-Net" | YES | 10.1109/CVPR.2019.00118 | CVPR 2019 | 268 citations |
| Zhou & Gregson "WHENet" | YES | arXiv:2005.10353 | BMVC 2020 | Full-range yaw |
| Zhou et al. "DirectMHP" | YES | arXiv:2302.01110 | arXiv 2023 | Multi-person HPE |
| Zhou et al. "SemiUHPE" | YES | arXiv:2404.02544 | arXiv 2024 | Semi-supervised HPE |
| Albiero et al. "img2pose" | YES | arXiv:2012.07791 | CVPR 2021 | 6DoF detection+pose |
| Ranjan et al. "HyperFace" | YES | arXiv:1603.01249 | TPAMI 2017 | MTL face+pose+gender |
| Kendall et al. "uncertainty weighting" | YES | arXiv:1705.07115 | CVPR 2018 | Loss weighting canonical |
| Becigneul & Ganea "Riemannian adaptive" | YES | arXiv:1810.00760 | NeurIPS 2018 | Riemannian optimization |
| Yin et al. "Rotation Laplace" | YES | arXiv:2303.01743 | ICCV 2023 | SOTA probabilistic rotation |
| Peretroukhin et al. "Probabilistic rotation" | YES | arXiv:1904.03182 | ICRA 2019 | Quaternion averaging |
| Hempel et al. "Robust full range" | YES | arXiv:2309.07654 | arXiv 2023 | Follow-up to 6DRepNet |
| Grauman et al. "Ego4D" | YES | arXiv:2110.07058 | CVPR 2022 | Egocentric benchmark |
| Damen et al. "EPIC-KITCHENS" | YES | arXiv:1804.02748 | ECCV 2018 | Egocentric actions |
| Tomenotti et al. "Head pose uncertainty" | YES (limited) | SSRN 4399244 | SSRN 2023 | Uncertainty for HPE |
| **Murphy et al. "Probabilistic rotation"** | **NOT VERIFIED** | -- | -- | Could not find in arXiv, Semantic Scholar, or Crossref |
