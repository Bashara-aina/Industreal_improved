# Agent 9: Head Pose / Regression Specialist Report

**Date**: 2026-07-11
**Context**: Pose is our best head (8.7 deg MAE, MTL/ST ~0.77). 6D+geodesic implemented. Question: do regression tasks benefit from or harm MTL?

---

## Executive Summary

Pose is already our strongest task (MTL/ST ratio ~0.77, 8.7 deg MAE). We use a 6D rotation representation with geodesic loss on SO(3). The evidence below suggests: (1) the 6D + geodesic formulation is well-grounded in SOTA head pose literature, (2) regression tasks face fundamentally different MTL dynamics than classification tasks, and (3) there is published evidence that MTL can *degrade* regression performance while improving classification -- which aligns with the concern that regression tasks may not benefit from MTL the way detection/classification tasks do.

---

## Paper 1: 6DRepNet -- The Foundation

**Title**: 6D Rotation Representation For Unconstrained Head Pose Estimation
**Authors**: Thorsten Hempel, Ahmed A. Abdelrahman, Ayoub Al-Hamadi
**Venue**: ICIP 2022
**arXiv**: 2202.12555 (cited by 242)

**Key contributions:**
- Proposes a continuous 6D rotation matrix representation for direct head pose regression, avoiding discontinuities of Euler angles and quaternions.
- Introduces geodesic distance-based loss on the SO(3) manifold to properly penalize rotation errors.
- Outperforms prior SOTA by up to 20% on AFLW2000 and BIWI datasets.
- Open-source code: github.com/thohemp/6DRepNet.

**Relevance to our context**: This is the canonical reference for 6D + geodesic head pose. Our implementation follows this approach. Baseline MAE: 3.97 deg (AFLW2000), 3.47 deg (BIWI). These numbers are for single-task pose-only models.

---

## Paper 2: Continuity of Rotation Representations in Neural Networks

**Title**: On the Continuity of Rotation Representations in Neural Networks
**Authors**: Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li
**Venue**: CVPR 2019
**PDF**: openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf

**Key contributions:**
- Proves mathematically that 3D rotations have **discontinuous** representations in real Euclidean spaces of 4 or fewer dimensions.
- Shows that quaternions (4D) and Euler angles (3D) are inherently discontinuous and difficult for neural networks to learn.
- Demonstrates that 3D rotations have **continuous** representations in 5D and 6D.
- Empirically shows 6D representation outperforms quaternions and Euler angles on rotation estimation tasks.

**Relevance to our context**: This is the theoretical foundation for why 6D representations work. The paper proves that Euler angles (3D) and quaternions (4D) are fundamentally discontinuous, making them suboptimal for neural network regression. Our 6D approach is theoretically justified by this work. NOTE: Zhou et al. explicitly recommend geodesic loss with the 6D representation (see their Table 2 results where geodesic + 6D outperforms L2 + 6D).

---

## Paper 3: Uncertainty Weighting for MTL with Regression + Classification

**Title**: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
**Authors**: Alex Kendall, Yarin Gal, Roberto Cipolla
**Venue**: CVPR 2018
**arXiv**: 1705.07115 (cited by 5,841)

**Key contributions:**
- Proposes principled approach to weigh multiple loss functions by considering **homoscedastic uncertainty** of each task.
- Derives a single multi-task loss function that handles both regression (L2 for depth) and classification (cross-entropy for semantics).
- Regression loss weighting is derived as: L(W) = (1/2 sigma^2) * ||y - f^W(x)||^2 + log(sigma), where sigma is learned task uncertainty.
- Demonstrates on depth regression + semantic/instance segmentation that the uncertainty-weighted MTL model **outperforms separate single-task models**.

**Key formula (regression-specific)**:
For a regression task with homoscedastic uncertainty sigma, the contribution to the total loss is:
```
L_reg(W, sigma) = (1 / (2 * sigma^2)) * ||y - f^W(x)||^2 + log(sigma)
```
The log(sigma) term prevents sigma from growing unbounded. This formulation lets the model **learn to down-weight noisy regression tasks automatically**.

**Relevance to our context**: This is the canonical method for handling the regression-vs-classification loss balance in MTL. If our pose regression benefits from lower weighting relative to detection/classification losses, Kendall's uncertainty method provides a principled mechanism. Important caveat: Kirchdorfer et al. (2026) found this method can be brittle when task gradients conflict.

---

## Paper 4: Geodesic Loss for Deep Pose Estimation on SO(3)

**Title**: Real-time Deep Pose Estimation with Geodesic Loss for Active Triggering
**Authors**: Seyed S. M. Salehi, Shadrokh Samavi, Nader Karimi, et al.
**Venue**: IEEE Access, 2018
**PMC**: PMC6438698 (cited by 168)

**Key contributions:**
- Formulates pose estimation as direct geodesic distance minimization on SO(3).
- Derives the geodesic loss: d(R1, R2) = arccos((tr(R1 R2^T) - 1) / 2)
- Shows that geodesic loss on rotations is more principled than L2 on Euler angles.
- Demonstrates real-time performance with deep networks.

**Relevance to our context**: This is the earliest and most-cited paper connecting geodesic loss to deep pose estimation. Establishes the SO(3) manifold geometry as the correct space for pose loss. The 6DRepNet paper extends this by combining 6D continuous representation with geodesic loss.

---

## Paper 5: MTL Degrades Regression but Improves Classification (CRITICAL)

**Title**: When Does Multi-Task Learning Fail? Quantifying Data Imbalance and Task Independence in Metal Alloy Property Prediction
**Authors**: Sungwoo Kang et al.
**Venue**: arXiv 2512.22740v2, Jan 2026 (submitted to Computational Materials Science)

**Key findings (directly relevant to our regression concern):**
- **MTL significantly degrades regression performance**: Resistivity R2 drops from 0.897 to 0.844 (p<0.01); Hardness R2 drops from 0.832 to 0.694 (p<0.01).
- **MTL significantly improves classification performance**: Amorphous F1 improves from 0.703 to 0.744 (p<0.05); recall improves by 17%.
- Learned task relation graphs reveal near-zero inter-task weights (~0.006), indicating task independence.
- Regression degradation attributed to: (a) data imbalance causing gradient domination, (b) gradient conflicts (cosine similarity near zero or slightly negative between tasks), (c) functional form mismatch between tasks.
- **PCGrad** improves minority regression R2 by 12.4% by resolving gradient conflicts.
- Recommendation: "Use independent models for precise property regression."

**Relevance to our context**: This paper directly validates the concern that regression tasks may NOT benefit from MTL the same way classification tasks do. It provides published evidence that MTL can actively harm regression while helping classification. The paper's findings suggest that our pose regression may be competing with detection/classification tasks for shared representation capacity, and that independent (or lighter-sharing) pose models may achieve better MAE than hard-shared MTL.

---

## Paper 6: Investigating Uncertainty Weighting for MTL

**Title**: Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Limitations
**Authors**: Lukas Kirchdorfer et al.
**Venue**: International Journal of Computer Vision (IJCV), 2026
**DOI**: 10.1007/s11263-025-02625-x (cited by 4)

**Key contributions:**
- Systematic investigation of Kendall's uncertainty weighting across diverse MTL settings.
- Found that uncertainty weighting works well when tasks have complementary gradients but **can fail when task gradients are in conflict**.
- Heteroscedastic uncertainty (data-dependent) is not always beneficial over simple homoscedastic uncertainty.
- Uncertainty weighting alone does not solve the regression-vs-classification fundamental conflict.
- Proposes diagnostic tools to determine when uncertainty weighting is appropriate.

**Relevance to our context**: This paper is important for Section C (loss balancing). It shows that Kendall's method is not a universal solution and provides diagnostic guidance. If our pose regression gradients conflict with detection/classification, uncertainty weighting may not suffice -- we may need gradient-based methods (like PCGrad, as in Paper 5) or architectural separation.

---

## Paper 7: 9D Rotation Representation with SVD + Geodesic Loss

**Title**: 9D Rotation Representation-SVD Fusion with Deep Learning for Unconstrained Head Pose Estimation
**Authors**: Jiaqi Lyu, Changyuan Wang
**Venue**: International Journal of Advanced Network, Monitoring and Controls, 2024

**Key contributions:**
- Builds on 6DRepNet, extending to a 9D representation with SVD projection back to SO(3).
- Directly compares L2 loss vs. geodesic loss on rotation matrices.
- **Geodesic loss yields better MAE than L2**: on AFLW2000: 3.85 vs. 3.90; on BIWI: 3.73 vs. 3.92; on 70/30 BIWI: 2.50 vs. 2.71.
- Uses EfficientNetV2-S backbone (20.2M params, 2.9G FLOPs) vs 6DRepNet's ResNet50 (43.8M params, 9.8G FLOPs).
- Achieves 3.85 deg MAE on AFLW2000, 3.73 deg on BIWI (cross-dataset eval).

**Relevance to our context**: Confirms geodesic loss is superior to L2 for rotation regression. Also provides MAE baselines for pose-only models that we can compare against our MTL results. The 9D+SVD approach is an alternative to 6D+Gram-Schmidt; both are continuous representations.

---

## Paper 8: Head Pose Estimation Survey

**Title**: Deep Learning for Head Pose Estimation: A Survey
**Authors**: Various (Springer)
**Venue**: SN Computer Science, 2023
**DOI**: 10.1007/s42979-023-01796-z

**Key findings:**
- Comprehensive taxonomy of HPE methods: landmark-based, appearance-based, and hybrid.
- Classifies rotation representations: Euler angles, quaternions, rotation matrices, 6D continuous.
- Documents the shift from classification-based bins to direct regression with continuous representations.
- Notes that joint MTL with face detection, landmark regression, and attribute classification is the dominant paradigm.

**Relevance to our context**: Provides citations for the broader MTL+pose literature and context for our approach. Confirms that 6D continuous representation + geodesic loss is the current SOTA trend in head pose.

---

## Summary: Pose-Specific Implications for the Paper

### Section C (Loss Balancing):
- Kendall et al. (CVPR 2018) provides the canonical uncertainty weighting for regression tasks in MTL.
- Kirchdorfer et al. (IJCV 2026) shows this method has limitations when gradients conflict.
- Kang et al. (2026) demonstrates that PCGrad can resolve gradient conflicts between regression and classification tasks (+12.4% minority task improvement).

### Section E (Pose-Specific):
- 6D+geodesic is theoretically grounded by Zhou et al. (CVPR 2019) and empirically validated by Hempel et al. (ICIP 2022, 242 citations).
- **Key concern validated**: Kang et al. (2026) empirically shows that MTL degrades regression while improving classification -- directly supporting the hypothesis that pose regression may benefit less from MTL than detection/classification tasks.
- Our MTL/ST ratio of ~0.77 for pose is actually quite strong compared to the literature where regression degradation can be >15% relative.
- If pose MAE needs improvement, options include: (1) lighter/shared-only-at-low-levels architecture, (2) PCGrad for gradient conflict resolution, (3) uncertainty weighting with sigma regularization, (4) soft parameter sharing with task-specific backbones.

### Citations to Use:
- Zhou et al. CVPR 2019 -- for 6D continuity proof
- Hempel et al. ICIP 2022 -- for 6DRepNet head pose
- Kendall et al. CVPR 2018 -- for uncertainty weighting
- Kirchdorfer et al. IJCV 2026 -- for limitations of uncertainty weighting
- Kang et al. 2026 -- for regression degradation in MTL (key paper)
- Lyu & Wang 2024 -- for geodesic > L2 on rotations
