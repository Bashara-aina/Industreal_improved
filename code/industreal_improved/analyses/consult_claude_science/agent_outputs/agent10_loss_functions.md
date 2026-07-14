# Agent 10: Loss Function Designer -- Discovery Report

**Date:** 2026-07-11
**Consultation:** Claude Science MTL -- closing per-task MTL-to-ST gaps
**Tasks:** Detection (CIoU+DFL+Focal+TAL), Activity (CE+logit_adjust+class_weights), PSR (BCE+focal+transition_aware), Pose (cosine+geodesic)

---

## Overview

This report surveys 12 published loss-function alternatives across the four tasks, with exact formulations where available. Each candidate was evaluated for (a) published improvement over the current baseline, (b) gradient properties under shared-backbone MTL conditions, and (c) ease of integration into the existing pipeline.

---

## Section 1: Detection -- Box Regression (replacing CIoU)

### 1.1 Wise-IoU (WIoU) -- Tong et al., 2023

**Paper:** "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism" (arXiv:2301.10051)
**Venue:** arXiv 2023; cited 1800+

**Key idea:** Replaces the static CIoU penalty term with a dynamic non-monotonic focusing mechanism that evaluates anchor box quality by *outlier degree* rather than IoU value. Reduces harmful gradients from both very-low-quality and very-high-quality anchors, focusing on medium-quality boxes that benefit most from gradient updates.

**Formulation:**
```
L_WIoU = L_IoU * R_WIoU

where R_WIoU = exp((x - x_gt)^2 + (y - y_gt)^2 / (w_g^2 + h_g^2))

Dynamic FM term: beta = L_IoU* / L_IoU   (outlier degree)
Gradient gain: gamma = beta / (delta * alpha^(beta - delta))
```

**Results on MS-COCO:** With YOLOv7, WIoU improves AP-75 from 53.03% to 54.50%.
**MTL compatibility:** The dynamic focusing naturally smooths gradient magnitudes across varying-quality detections, which stabilises shared-backbone gradient flow. No trainable parameters.

### 1.2 Powerful-IoU (PIoU v2) -- Liu et al., 2024

**Paper:** "Powerful-IoU: More straightforward and faster bounding box regression loss with a nonmonotonic focusing mechanism"
**Venue:** Neural Networks, Vol. 170, 2024 (cited 255+)

**Key idea:** Identifies and solves the "anchor box enlargement" pathology in IoU-based losses. The penalty factor is target-size-adaptive, and the non-monotonic attention layer (single hyperparameter) focuses on medium-quality anchors.

**Formulation:**
```
L_PIoU = 1 - IoU + ( (x - x_gt)^2 + (y - y_gt)^2 ) / (w_c^2 + h_c^2)
        + (w - w_gt)^2 / (w_c^2) + (h - h_gt)^2 / (h_c^2)

PIoU v2: L_PIoUv2 = (IoU^gamma) * L_PIoU
where gamma = beta / (1 + k * (beta - 1)^2)  (non-monotonic)
```

**Results on MS-COCO:** Outperforms CIoU, GIoU, DIoU, EIoU, SIoU across YOLOv8 and DINO. Converges in ~60 epochs vs 80-300 for other IoU losses.
**MTL compatibility:** Faster convergence reduces total training epochs, which can mitigate gradient interference between tasks over long training runs.

### 1.3 Focal-EIoU -- Zhang et al., 2022

**Paper:** "Focal and Efficient IOU Loss for Accurate Bounding Box Regression"
**Venue:** Neurocomputing, 2022

**Key idea:** EIoU decomposes the penalty term into three separate factors (overlap, center distance, side-aspect ratio discrepancies) instead of CIoU's coupled aspect-ratio term. Focal-EIoU blends this with a regression-targeted focal mechanism.

**Formulation:**
```
L_EIoU = L_IoU + L_dis + L_asp
       = 1 - IoU + rho^2(b, b_gt) / c^2 + rho^2(w, w_gt) / w_c^2 + rho^2(h, h_gt) / h_c^2

L_Focal-EIoU = IoU^gamma * L_EIoU
```

**Why over CIoU:** CIoU's aspect-ratio term is coupled and can mis-optimise when w/h ratio is correct but scale is wrong. EIoU separates width and height penalties.
**MTL compatibility:** The focal modulation (IoU^gamma) naturally re-weights gradient contributions, reducing competition with classification-head gradients.

### 1.4 SIoU -- Gevorgyan, 2022

**Paper:** "SIoU Loss: More Powerful Learning for Bounding Box Regression" (arXiv:2205.12740)

**Key idea:** Introduces an angle penalty term between the predicted and ground-truth box vectors, guiding the regression path to be more direct.

**Formulation:**
```
L_SIoU = 1 - IoU + (Delta + Omega) / 2

where Delta = sum of distance terms * (2 - sin^2(angle))
      Omega = shape discrepancy term
```

**Result:** 2-3% AP improvement over CIoU on COCO, particularly beneficial for rotated and small objects.
**MTL compatibility:** The angle-aware term adds a distinct gradient signal that is largely orthogonal to classification gradients, reducing conflict.

---

## Section 2: Detection -- Classification Head (replacing Focal Loss)

### 2.1 Varifocal Loss -- Zhang et al., 2021 (CVPR Oral)

**Paper:** "VarifocalNet: An IoU-aware Dense Object Detector" (arXiv:2008.13367)
**Venue:** CVPR 2021 Oral

**Key idea:** Replaces Focal Loss with an asymmetric loss that treats positive and negative samples differently. Designed to train IoU-aware Classification Scores (IACS) where the target is the soft IoU between predicted and GT box, not a binary 0/1 label.

**Formulation:**
```
L_VFL(p, q) = {
    -q * (q * log(p) + (1 - q) * log(1 - p)),      for q > 0 (positive)
    -alpha * p^gamma * log(1 - p),                  for q == 0 (negative)
}

where q = IoU(box_pred, box_gt) for positives, 0 for negatives
```

**Key difference from Focal Loss:** For positives, instead of `(1-p)^gamma * log(p)`, VFL uses `q * (q * log(p) + (1-q) * log(1-p))`. This makes positive gradients proportional to IoU quality, not just classification difficulty.
**Results on MS-COCO:** VFNet with Res2Net-101-DCN achieves 55.1 AP on COCO test-dev, +2.0 AP over FCOS+ATSS baseline.
**MTL compatibility:** The IoU-aware weighting naturally calibrates the classification loss magnitude to match detection quality, reducing variance in gradients reaching the shared backbone.

### 2.2 ASL (Asymmetric Loss) -- Ridnik et al., 2021

**Paper:** "Asymmetric Loss For Multi-Label Classification" (arXiv:2009.14119)
**Venue:** ICCV 2021

**Key idea:** Applies different focusing mechanisms to positive vs negative samples. For negatives, both hard-thresholding (gamma_minus) and dynamic down-weighting. Designed originally for multi-label but demonstrated on detection.

**Formulation:**
```
L_ASL = {
    L_+ = (1 - p)^gamma_+ * log(p)                     (positive)
    L_- = (p_m)^gamma_- * log(1 - p_m)                  (negative)
}

where p_m = max(p - m, 0)  (hard-thresholding)
      gamma_+ < gamma_-  (asymmetric focusing)
```

**Results:** SOTA on MS-COCO (91.8 mAP), Pascal-VOC, NUS-WIDE, Open Images for multi-label. Demonstrated on detection as well.
**MTL compatibility:** The hard-thresholding (p_m) for negatives reduces the noise from easy-negative gradients, which can otherwise dominate in MTL setups where the detection task has far more negatives than other tasks.

---

## Section 3: Activity Classification (replacing CE + logit_adjust + class_weights)

### 3.1 Balanced Softmax -- Ren et al., 2020

**Paper:** "Balanced Meta-Softmax for Long-Tailed Visual Recognition" (NeurIPS 2020)

**Key idea:** Reformulates softmax to account for label distribution shift between training and testing. The logit adjustment is built into the loss, not applied as a post-hoc correction.

**Formulation:**
```
L_BalancedSoftmax(x, y) = -log( n_y * exp(f_y(x)) / sum_j(n_j * exp(f_j(x))) )

where n_y = number of training examples in class y
      f_y(x) = logit for class y

Equivalently, this is standard softmax with logits shifted by log(class_prior):
    f'_y(x) = f_y(x) + log(n_y / N)
```

**Why over CE+logit_adjust:** Instead of applying logit adjustment as a hand-tuned additive term, Balanced Softmax provides a theoretically grounded correction derived from minimising the Bayesian error under label-shift. No hyperparameter tuning required.
**Results:** Outperforms class-weighted CE, re-sampling, and post-hoc logit adjustment on long-tailed versions of CIFAR, ImageNet, and Places365.
**MTL compatibility:** The built-in logit adjustment eliminates the need for per-class weight tuning, which reduces the MTL hyperparameter search space. Adjacent tasks (detection, PSR) benefit from more stable activity gradients.

### 3.2 Seesaw Loss -- Wang et al., 2021

**Paper:** "Seesaw Loss for Long-Tailed Instance Segmentation" (CVPR 2021)
**Venue:** CVPR 2021 (cited 400+)

**Key idea:** Dynamically re-balances gradients for positive and negative samples per category using two complementary mechanisms: a *mitigation factor* that reduces negative-sample gradient for tail classes, and a *compensation factor* that increases positive-sample gradient when tail-class predictions are suppressed.

**Formulation:**
```
L_Seesaw(i, j) = -log( exp(f_i) / ( exp(f_i) + sum_{j != i} S_{ij} * exp(f_j) ) )

where S_{ij} = M_{ij} * C_{ij}

Mitigation factor M_{ij} = (N_j / N_i)^p    (p >= 0, N = sample count)
Compensation factor C_{ij} = 1 / (1 + k * exp(-(f_j - f_i) / T))   (sigmoid-like)
```

**Why over class_weights:** Static class weights apply a fixed multiplier, ignoring per-sample dynamics. Seesaw Loss adapts the inter-class gradient ratio based on both class priors and current prediction confidence.
**MTL compatibility:** The sigmoid-based compensation factor creates smooth gradient transitions, avoiding the sharp gradient discontinuities that static class weights can cause when switching between easy and hard samples.

### 3.3 LDAM -- Cao et al., 2019

**Paper:** "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)
**Venue:** NeurIPS 2019 (cited 1500+)

**Key idea:** Introduces a class-dependent margin that is inversely proportional to the square root of class frequency, derived from a theoretical generalisation bound.

**Formulation:**
```
L_LDAM(x, y) = -log( exp(f_y(x) - Delta_y) / ( exp(f_y(x) - Delta_y) + sum_{j != y} exp(f_j(x)) ) )

where Delta_y = C / n_y^(1/4)   (C = hyperparameter, n_y = class count)

Combined with DRW (Deferred Re-weighting):
    - Train with LDAM only for epochs 1..T
    - Apply class-balanced re-weighting for epochs T+1..
```

**Results:** Achieves SOTA on long-tailed CIFAR-10/100, ImageNet-LT, and iNaturalist 2018.
**MTL compatibility:** The class-dependent margin acts as a form of gradient normalisation across classes, which can help stabilise joint training with detection (which has many more foreground classes). DRW strategy prevents overfitting to tail classes early in training.

---

## Section 4: PSR -- Binary Classification (replacing BCE + focal + transition_aware)

### 4.1 Unified Focal Loss -- Yeung et al., 2022

**Paper:** "Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation"
**Venue:** Computerized Medical Imaging and Graphics, 2022

**Key idea:** Unifies region-based (Dice/Tversky) and distribution-based (CE/Focal) losses into a single hierarchical framework. For the PSR binary case, the asymmetric Focal variant is most relevant.

**Formulation:**
```
Unified Focal Loss (for binary):
L_UF = lambda * L_asymmetric_Focal + (1-lambda) * L_asymmetric_Focal_Tversky

L_asymmetric_Focal = 
    - (1 - p_t)^(gamma_pos) * log(p_t)         for positive samples
    - (1 - p_t)^(gamma_neg) * log(p_t)          for negative samples
    
where gamma_neg > gamma_pos to down-weight easy negatives
```

**Why over BCE+focal:** The asymmetric focusing (separate gamma for pos/neg) is particularly relevant for PSR where foreground (positive) events are rare. The unified formulation also allows blending with a Tversky component for spatial overlap awareness.
**MTL compatibility:** The smooth, bounded gradient profile avoids the exploding-gradient problem that BCE can exhibit on misclassified hard examples.

### 4.2 ASL (Asymmetric Loss) -- Ridnik et al., 2021

**Paper:** Same as Section 2.2, but repurposed for binary PSR.

**Formulation (for binary case):**
```
L_ASL_binary = -y * (1 - p)^gamma_+ * log(p) 
              - (1 - y) * (p_m)^gamma_- * log(1 - p_m)

where p_m = max(p - m, 0), gamma_+ = 0, gamma_- >= 1 typically
```

**Why over BCE+focal+transition_aware:** The hard-thresholding mechanism (p_m) for negatives is a simpler alternative to the transition-aware scaling. If transition-state probabilities are noisy, ASL's hard threshold provides more robust negative down-weighting.
**Results (multi-label context):** Outperforms BCE, Focal, and weighted CE across four large-scale multi-label datasets.
**MTL compatibility:** The asymmetric focussing creates a natural gradient asymmetry that can help balance the positive-negative ratio disparity between PSR and other tasks.

---

## Section 5: Pose Estimation (replacing cosine + geodesic)

### 5.1 6D Rotation Representation + Geodesic Loss -- Zhou et al., 2019

**Paper:** "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)
**Venue:** CVPR 2019 (cited 2200+)

**Key insight (not a new loss, but a critical representation finding):** Quaternions (4D) and Euler angles (3D) are topologically discontinuous representations for SO(3) -- there exist two rotations arbitrarily close in angle that are arbitrarily far in representational space. The 6D representation is the minimal continuous representation.

**Formulation:**
```
Geodesic Loss (angular distance on SO(3)):
L_geodesic(R_pred, R_gt) = arccos( (trace(R_pred^T * R_gt) - 1) / 2 )

6D representation (from Zhou et al.):
    From rotation matrix R = [r1, r2, r3]^T, keep first two columns and orthogonalise:
    f_6D = [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
    Reconstruction: use Gram-Schmidt on f_6D -> R
    
Cosine loss alternative: L_cos = 1 - cos(theta) = 1 - (trace(R_pred^T * R_gt) - 1) / 2
```

**Why over cosine+geodesic:** Using the 6D representation eliminates the discontinuity that makes quaternion regression difficult. The geodesic loss itself is the true Riemannian metric on SO(3). Cosine loss `(1 - cos(theta))` is a smooth proxy that approximates the squared geodesic distance for small angles but plateaus for large errors.
**MTL compatibility:** The 6D representation provides smooth, continuous gradients back to the shared backbone. The rotation matrix formulation (*not* quaternions) avoids the double-cover issue (q and -q represent the same rotation) that creates gradient conflicts.

### 5.2 Geodesic + CH (Chamfer/Huber on SO(3)) -- Geist et al., 2024

**Paper:** "Learning with 3D Rotations, a Hitchhiker's Guide to SO(3)" (arXiv:2404.11735)
**Venue:** ICML 2024 (cited 58+)

**Key idea:** Comprehensive guide recommending:
1. Use 6D representation (Zhou et al.) for continuous output
2. Use geodesic loss with optional Huber-style clipping for outlier robustness
3. For multi-modal pose, consider Bingham distribution or mixture of von Mises-Fisher

**Formulation:**
```
Huberised geodesic loss:
L_huber_geod(theta) = {
    0.5 * theta^2 / delta,              for |theta| <= delta
    |theta| - delta / 2,                otherwise
}

where theta = arccos( (trace(R_pred^T * R_gt) - S) / 2 ), S in {1, -1}
```

**Why over vanilla geodesic:** The Huberisation prevents extreme gradient values when initial pose estimates are far from ground truth -- common in MTL early epochs where the shared backbone is still learning basic features.
**MTL compatibility:** The clipped gradient profile prevents pose loss gradients from dominating (or being dominated by) other task gradients during early training.

### 5.3 Bingham Loss -- Gilitschenski et al., 2019 / Prokudin et al., 2018

**Paper:** "Deep Bingham Networks: Dealing with Uncertainty and Ambiguity in Pose Estimation" (ECCV 2018)

**Key idea:** Instead of point estimation, predict a distribution over SO(3) using the Bingham distribution (antipodally symmetric, appropriate for quaternions). The loss is the negative log-likelihood under the Bingham.

**Formulation:**
```
Bingham negative log-likelihood:
L_Bingham(q, Z, V) = -log( F_11(1/2, 2, diag(Z)) ) - q^T * V * diag(Z) * V^T * q + const

where q = predicted quaternion
      V = 4x4 orthogonal matrix (orientation modes)
      Z = 3-element concentration vector
      F_11 = confluent hypergeometric function (normalising constant)
```

**Why over cosine+geodesic:** For pose with inherent ambiguity (symmetry, occlusion), a point estimate is insufficient. The Bingham loss captures multi-modal uncertainty and naturally handles the quaternion double-cover issue through antipodal symmetry.
**MTL compatibility:** The uncertainty estimate (through Z parameters) can be used as an MTL weighting signal -- tasks with high uncertainty contribute less gradient. This is analogous to Kendall et al.'s uncertainty weighting but built into the pose loss itself.

---

## Section 6: Cross-Task Interaction & Gradient Management

### 6.1 Uncertainty-Aware Loss Weighting -- Kendall et al., 2018

**Paper:** "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)

**Key idea:** Learn task-specific homoscedastic uncertainty parameters that adaptively weight each task's loss during training, replacing manual grid-search.

**Formulation:**
```
L_total = sum_t ( 1 / (2 * sigma_t^2) ) * L_t + log(sigma_t)

where sigma_t = learnable noise parameter for task t
```

**Why for our setup:** This provides a principled, learned weighting mechanism that automatically adjusts the contribution of each task's loss. Our 4 tasks have vastly different loss magnitudes (CIoU ~0-1, CE ~0-log(C), geodesic ~0-pi).
**Integration path:** Requires adding one scalar sigma per task as a learnable parameter. Simple to implement in PyTorch via `nn.Parameter(torch.zeros(1))`.

### 6.2 PiKE / PCGrad -- Gradient Surgery

**Paper:** "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020); "PiKE: Adaptive Data Mixing for Multi-Task Learning" (ICLR 2025)

**Key idea:** PCGrad projects each task's gradient onto the normal plane of any conflicting gradient, eliminating destructive interference. PiKE extends this to data-level mixing.

**Formulation:**
```
PCGrad:
for each pair of tasks (i, j):
    if g_i * g_j < 0 (conflicting):
        g_i = g_i - (g_i * g_j / ||g_j||^2) * g_j
```

**Why for our setup:** Detection and PSR gradients often conflict in early training (detection wants many positive area proposals, PSR wants tight temporal boundaries). PCGrad-style projection would resolve this.
**Integration path:** Can be applied post-hoc on any collection of per-task loss gradients without modifying individual loss formulations.

---

## Section 7: Recommended Replacement Matrix

| Task | Current Loss | Recommended Alternative | Primary Benefit | Gradient Interaction |
|------|-------------|------------------------|-----------------|---------------------|
| **Detection Box Reg** | CIoU + DFL | **WIoU v3** (dynamic non-monotonic FM) | Outlier-adaptive gradient, faster convergence | Smooth gradient gain, less noise |
| **Detection Cls** | Focal Loss | **Varifocal Loss** (IoU-aware) | IoU-quality-weighted positives | Better gradient calibration with box branch |
| **Activity Cls** | CE + logit_adj + weights | **Balanced Softmax** or **Seesaw Loss** | No hand-tuned weights | Stable gradient across classes |
| **PSR** | BCE + focal + trans.aware | **ASL** (Asymmetric Loss) | Hard-threshold negatives, simple | Asymmetric gradient profile |
| **Pose Reg** | cosine + geodesic | **6D rep + Huberised geodesic** + optional **Bingham** | Continuous representation, robust | Smooth gradient, no discontinuity |

**Recommended MTL-specific additions:**
1. Add **Uncertainty Weighting** (Kendall et al.) -- one sigma per task
2. Add **PCGrad** or **Gradient Normalisation** for detection-vs-PSR gradient conflicts
3. Consider **Progressive Unfreezing** -- train detection+pose first (easier), add activity+PSR later

---

## Section 8: Implementation Notes

- **Varifocal Loss** requires modifying target assignment to compute IoU between each positive anchor and its assigned GT box (TAL already does this partially -- IoU values used in TAL cost can be reused).
- **Balanced Softmax** is a 1-line change to the CE logit computation (add `log(class_prior)`).
- **WIoU** replaces CIoU penalty term with no architectural changes.
- **ASL** replaces BCE+focal with 3 hyperparameters: `gamma_pos`, `gamma_neg`, `m` (margin). Suggested starting values: `gamma_pos=0, gamma_neg=4, m=0.2`.
- **6D rotation** requires changing the final pose head from 4D (quaternion) to 6D output, with Gram-Schmidt orthonormalisation layer. The geodesic loss code stays the same.

---

## References

1. **Zhang, H. et al.** (2021). "VarifocalNet: An IoU-aware Dense Object Detector." CVPR 2021 Oral. arXiv:2008.13367
2. **Ridnik, T. et al.** (2021). "Asymmetric Loss For Multi-Label Classification." ICCV 2021. arXiv:2009.14119
3. **Ren, J. et al.** (2020). "Balanced Meta-Softmax for Long-Tailed Visual Recognition." NeurIPS 2020. arXiv:2007.10740
4. **Wang, J. et al.** (2021). "Seesaw Loss for Long-Tailed Instance Segmentation." CVPR 2021.
5. **Cao, K. et al.** (2019). "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." NeurIPS 2019.
6. **Tong, Z. et al.** (2023). "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism." arXiv:2301.10051
7. **Liu, C. et al.** (2024). "Powerful-IoU: More straightforward and faster bounding box regression loss." Neural Networks, 170:276-284.
8. **Zhang, Y.-F. et al.** (2022). "Focal and Efficient IOU Loss for Accurate Bounding Box Regression." Neurocomputing.
9. **Zhou, Y. et al.** (2019). "On the Continuity of Rotation Representations in Neural Networks." CVPR 2019.
10. **Geist, A.R. et al.** (2024). "Learning with 3D Rotations, a Hitchhiker's Guide to SO(3)." ICML 2024.
11. **Kendall, A. et al.** (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018.
12. **Yeung, M. et al.** (2022). "Unified Focal loss: Generalising Dice and cross entropy-based losses." Computerized Medical Imaging and Graphics.
