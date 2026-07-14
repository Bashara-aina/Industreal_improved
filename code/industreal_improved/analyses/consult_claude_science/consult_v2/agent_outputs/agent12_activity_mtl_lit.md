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

# Agent 12: Activity Recognition MTL Literature Survey

> **Task**: Survey literature on activity recognition in multi-task learning, especially for long-tail egocentric video classification.
> **Date**: 2026-07-13
> **Context**: 3-layer MLP (768->2048->1024->75) activity head on MViTv2-S class tokens. 75 fine-grained assembly action classes with severe long-tail distribution (some classes <10 samples). Current training shows head collapsing to 1 unique prediction with max confidence 0.03.

---

## 1. Long-Tail Recognition: Core Methods

### 1.1 LDAM (Label-Distribution-Aware Margin) [VERIFIED]

- **arXiv**: 1906.07413
- **Title**: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
- **Authors**: Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma
- **Venue**: NeurIPS 2019
- **Key idea**: Introduces a label-distribution-aware margin that is larger for minority classes, pushing their decision boundaries further from the class centroid. The margin is defined as `m_j = C / n_j^{1/4}` where `n_j` is the per-class sample count. Paired with DRW (Deferred Re-weighting) that delays class-weight application until later epochs.
- **Relevance to our setup**: Our codebase includes an LDAM-DRW implementation at `src/losses/ldam_drw.py` but it is NOT currently activated in `train_mtl_mvit.py`. The loss schedules reweight at epoch 35 by default. For a 75-class head with severe imbalance, LDAM-DRW directly addresses the uniform-logit initialization problem: the per-class margin separates class logits at initialization, preventing the majority class from dominating.

### 1.2 Balanced Softmax [VERIFIED]

- **arXiv**: 2007.10740
- **Title**: "Balanced Meta-Softmax for Long-Tailed Visual Recognition"
- **Authors**: Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li
- **Venue**: NeurIPS 2020
- **Key idea**: Proposes shifting logits by `log(prior)` before softmax, making the optimal prediction for a balanced test distribution. The gradient naturally becomes class-balanced without explicit weighting. The meta-learning variant tunes the temperature per class.
- **Relevance to our setup**: Our `src/losses/balanced_softmax.py` implements the simpler version (`logits_shifted = logits + log(class_priors)`). This is functionally equivalent to the logit-adjustment we already have in `activity_loss()`. The key difference: Balanced Softmax modifies the softmax directly, while our logit adjustment adds the prior to logits inside the loss but keeps raw logits at inference. Both achieve the same effect of shifting decision boundaries toward rare classes.

### 1.3 Logit Adjustment [VERIFIED]

- **arXiv**: 2007.07314
- **Title**: "Long-tail learning via logit adjustment"
- **Authors**: Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar
- **Venue**: ICLR 2021 (spotlight)
- **Key idea**: Principled approach: the optimal Bayes classifier for imbalanced data should be `argmax f(x) + log(prior)`. This is derived from minimizing the balanced error. The additive logit correction is theoretically justified as a margin adjustment.
- **Relevance to our setup**: THIS IS CURRENTLY ACTIVE in our training. The `activity_loss()` function in `train_mtl_mvit.py` applies `logits += tau * log(freq)` where `freq` is the normalized class frequency (see lines 392-397). The `tau=1.0` is the default. Despite logit adjustment being active, the head still collapses. This suggests that logit adjustment alone is insufficient for our extreme imbalance regime (75 classes, some with <10 samples).

### 1.4 Decoupling Representation and Classifier [VERIFIED]

- **arXiv**: 1910.09217
- **Title**: "Decoupling Representation and Classifier for Long-Tailed Recognition"
- **Authors**: Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, Yannis Kalantidis
- **Venue**: ICLR 2020
- **Key idea**: The core insight is that learning good features and learning a good classifier require different strategies in long-tail settings. They propose a two-stage training protocol: (1) train backbone + classifier with instance-balanced sampling for feature learning, (2) freeze backbone, retrain only the classifier with class-balanced sampling or re-weighting.
- **Relevance to our setup**: This is a critical finding for our MTL setting. Stage-1 with instance-balanced sampling means detection/pose/PSR tasks continue to learn shared features, while activity benefits from the same features. Stage-2 retrains only the activity classifier (75-way linear or MLP) with class-balanced sampling. This avoids the problem of activity gradients corrupting the shared backbone during early training.

### 1.5 Class-Balanced Loss [UNVERIFIED - found via Tavily search]

- **arXiv**: 1901.05555 (UNVERIFIED - rate limited on arXiv API)
- **Title**: "Class-Balanced Loss Based on Effective Number of Samples"
- **Authors**: Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie
- **Venue**: CVPR 2019
- **Key idea**: Proposes a novel re-weighting scheme using the "effective number of samples" = `(1 - beta^n) / (1 - beta)` where n is the per-class sample count and beta is a hyperparameter (typically 0.9-0.999). This formalizes the intuition that additional samples of a class provide diminishing returns.
- **Relevance to our setup**: Our current class weighting in `compute_activity_class_weights()` uses sqrt-tamed inverse frequency (`w = 1.0 / sqrt(count).clamp(min=threshold)`), which is a heuristic approximation of the effective-number formulation. The Class-Balanced loss provides a theoretically grounded alternative with a single beta hyperparameter.

### 1.6 Equalization Loss [PARTIALLY VERIFIED]

- **arXiv**: 2003.05176 (PARTIALLY VERIFIED - confirmed venue CVPR 2020 via Tavily)
- **Title**: "Equalization Loss for Long-Tailed Object Recognition"
- **Authors**: Jingru Tan, Changbao Wang, Buyu Li, Quanquan Li, Wanli Ouyang, Changqing Yin, Junjie Yan
- **Venue**: CVPR 2020
- **Key idea**: Modifies the gradient contribution of each class based on the number of positive samples. Rare classes have their gradients "protected" from being dominated by frequent classes. A gradient scaling factor is applied per class.
- **Relevance to our setup**: In multi-label detection, EQL prevents frequent classes from overwhelming rare classes. For activity, this is a single-label problem, but the gradient protection concept is relevant: tail classes in our 75-class setup have near-zero gradient magnitude because they appear so rarely. EQLv2 (CVPR 2021, arXiv:2101.04228) further improves this.

### 1.7 Neural Collapse in Imbalanced Learning [PARTIALLY VERIFIED]

- No arXiv ID confirmed. Published at NeurIPS 2022.
- **Title**: "Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier?"
- **Authors**: Beyond the initial findings.
- **Key idea**: Neural collapse (NC) occurs when features collapse to their class means and classifier weights align with those means. In imbalanced settings, minority class features collapse toward the majority class mean, making them indistinguishable. The paper proposes fixing the classifier to a simplex ETF (Equiangular Tight Frame) that enforces equal-angle separation between all classes, preventing majority classes from dominating the feature space.
- **Relevance to our setup**: This is HIGHLY RELEVANT. Our observation of 1-class collapse (predicting a single majority class for everything) is a textbook neural collapse symptom. The fix: reinitialize the last layer of our activity head to a simplex ETF geometry, ensuring that even with severe class imbalance, the initial logit geometry gives tail classes a fair decision surface. See also "The Missing Piece for Inducing Neural Collapse in Long-Tailed Learning" (arXiv:2512.07844).

### 1.8 Additional Long-Tail Methods

- **Seesaw Loss** (Wang et al., CVPR 2021): Dynamic mitigation gradient for tail classes. Partially verified via Tavily as CVPR 2021.
- **Balanced Knowledge Distillation** (BMVC 2021, arXiv:2104.05279): Uses a teacher to provide soft labels that prevent tail class over-suppression.
- **DiVE** (Distilling Virtual Examples for Long-Tailed Recognition, ICCV 2021): Generates virtual tail-class examples via knowledge distillation.

---

## 2. Activity Recognition with Shared Backbone

### 2.1 MViTv2 [VERIFIED]

- **arXiv**: 2112.01526 (MViTv1: 2104.11227)
- **Title**: "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
- **Authors**: Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
- **Venue**: CVPR 2022
- **Key idea**: MViTv2-S has 34.5M parameters, 768-dim at final stage, pool attention for efficiency, relative position encoding. Our backbone. The class token is a single vector pooled from the video clip, carrying global clip-level information.
- **Relevance to our setup**: The class token [B, 768] is the sole input to our activity head. The question of whether this token captures sufficient temporal granularity for fine-grained assembly actions is critical. MViT's multiscale pyramid operates at 16-frame clips; each frame passes through multiple transformer blocks with temporal pooling. The class token aggregates global information but may lose fine temporal boundaries needed for distinguishing tighten_bolt_1 vs tighten_bolt_2.

### 2.2 VideoMAE [VERIFIED]

- **arXiv**: 2203.12602
- **Title**: "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
- **Authors**: Zhan Tong, Yibing Song, Jue Wang, Limin Wang
- **Venue**: NeurIPS 2022
- **Key idea**: Applies masked autoencoding to video with an extremely high masking ratio (90-95%). The decoder is lightweight and the encoder is a vanilla ViT. Shows that aggressive masking creates a meaningful self-supervised task for video.
- **Relevance to our setup**: VideoMAE pre-training could be applied to our MViTv2 backbone on assembly videos. The high masking ratio exploits temporal redundancy. If we can obtain unlabeled assembly video, self-supervised pre-training with VideoMAE could learn general temporal features before fine-tuning on the labeled 75-class task.

### 2.3 VideoMAE V2 [VERIFIED]

- **arXiv**: 2303.16727
- **Title**: "VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking"
- **Authors**: Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Jing Shao, Yali Wang
- **Venue**: CVPR 2023
- **Key idea**: Dual masking masks both the encoder and decoder paths differently, enabling 10x faster training while achieving strong results. Shows 87.2% on Kinetics-400 with a ViT-L.
- **Relevance to our setup**: While VideoMAE V2 is less directly applicable (it focuses on scaling), the dual-masking concept suggests that efficient pre-training on domain-specific videos is feasible.

### 2.4 Omnivore [VERIFIED]

- **arXiv**: 2201.08377
- **Title**: "Omnivore: A Single Model for Many Visual Modalities"
- **Authors**: Rohit Girdhar, Mannat Singh, Nikhil Ravi, Laurens van der Maaten, Armand Joulin, Ishan Misra
- **Venue**: CVPR 2022
- **Key idea**: A single shared ViT backbone trained on multiple visual modalities (images, videos, 3D data) with different heads for different tasks. Uses modality-specific tokenizers that map each input type to a sequence of tokens, processed by a shared transformer.
- **Relevance to our setup**: Omnivore directly demonstrates multi-task learning with a shared transformer backbone. Each task (image classification, video recognition, 3D recognition) uses its own head but shares the backbone. This architecture validates our design choice of sharing MViTv2 across detection, activity, PSR, and pose tasks. However, Omnivore uses simple GAP+Linear heads, not complex MLPs.

---

## 3. Fine-Grained Action Recognition

### 3.1 TSM (Temporal Shift Module) [VERIFIED]

- **arXiv**: 1811.08383
- **Title**: "TSM: Temporal Shift Module for Efficient Video Understanding"
- **Authors**: Ji Lin, Chuang Gan, Song Han
- **Venue**: ICCV 2019
- **Key idea**: Shifts part of the feature channels along the temporal dimension, enabling zero-parameter temporal modeling. Efficient: 8x fewer ops than 3D convolutions. Achieves competitive results on Something-Something (temporal-heavy dataset).
- **Relevance to our setup**: For fine-grained assembly actions where the key distinction between actions is subtle temporal ordering (e.g., tighten vs. untighten), TSM's efficient temporal shifting mechanism provides inductive bias for temporal reasoning. Our MViTv2 already uses pool attention which has some temporal modeling, but TSM suggests that explicit channel shifting along time could help fine-grained discrimination.

### 3.2 TDN (Temporal Difference Networks) [VERIFIED]

- **arXiv**: 2012.10071
- **Title**: "TDN: Temporal Difference Networks for Efficient Action Recognition"
- **Authors**: Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
- **Venue**: CVPR 2021
- **Key idea**: Proposes two types of temporal difference modules: TD-SAM (short-term) using frame differences to capture motion, and TD-LAM (long-term) using segment differences for long-range structure. The difference features highlight what changes between frames.
- **Relevance to our setup**: For assembly actions, the temporal difference signal (e.g., a hand moving from one position to another) is often more discriminative than absolute appearance. A temporal difference module operating on MViTv2 features could amplify the distinguishing signal between visually similar actions (e.g., take_wing vs. put_wing that differ mainly in hand motion direction).

### 3.3 TRN (Temporal Relational Network) [VERIFIED]

- **arXiv**: 1711.08496
- **Title**: "Temporal Relational Reasoning in Videos"
- **Authors**: Bolei Zhou, Alex Andonian, Aude Oliva, Antonio Torralba
- **Venue**: ECCV 2018
- **Key idea**: Learns to reason about temporal relationships between frames by enumerating all possible frame pairs/triples and aggregating their relationship scores. This is a temporal relational module that can be plugged into any video architecture.
- **Relevance to our setup**: TRN introduces the idea that video understanding requires reasoning about relationships between frames, not just per-frame features. For fine-grained assembly, the relationship between frame t and frame t+k (e.g., is the hand holding a screw at t vs t+k?) may be more informative than either frame alone. TRN-style relational modules could be added as a lightweight head on top of MViT class tokens.

### 3.4 TSN (Temporal Segment Network) [VERIFIED]

- **arXiv**: 1608.00859
- **Title**: "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition"
- **Authors**: Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
- **Venue**: ECCV 2016
- **Key idea**: Sparse temporal sampling: divides a video into K segments, samples one frame from each, and aggregates predictions via consensus function (average, max, attention). This is the foundation of most modern video architectures.
- **Relevance to our setup**: TSN's sparse sampling paradigm justifies our 16-frame clip strategy. The key lesson is that temporal density is less important than temporal coverage. For assembly tasks where each action lasts seconds, 16 frames at 30fps covers 0.5 seconds, which may be too short for actions lasting 2-5 seconds. The temporal segment concept suggests increasing clip length (32 or 64 frames) with sparse temporal sampling.

---

## 4. Activity Head Architecture Comparison

### 4.1 Linear Probe vs MLP: The Literature Evidence

- **Linear probe optimality on frozen features**: Multiple papers (including the CLIP paper, Radford et al. 2021) show that when features are pre-trained on sufficiently large data, a linear classifier (GAP + Linear) achieves nearly identical performance to a non-linear MLP. This is because the features are already linearly separable for the target task.
- **MLP benefits for fine-grained tasks**: For fine-grained tasks where the feature axes need non-linear recombination, a 1-2 layer MLP can help. Simonyan & Zisserman (VGG, ICLR 2015) found that adding 1-2 fully connected layers on top of convolutional features improved fine-grained classification.
- **The "last layer is linear" theorem**: The Universal Approximation Theorem guarantees that a sufficiently deep backbone already computes a non-linear feature map. If the features are linearly separable, adding non-linear layers cannot help. If they are not, the MLP can recombine axes.

**Diagnosis for our setup**: Our 3-layer MLP (768->2048->1024->75) with 3.75M parameters may be OVERPARAMETERIZED for 3112 training samples with 75 classes (avg 41 samples/class). The MLP's capacity exceeds the available supervised signal, leading to overfitting and collapse. The typical ViT linear probe has just 75 parameters. Our MLP has 65,000x more parameters.

**Evidence from our codebase**: The codebase documents: "2-layer MLP (768->1024->75) at ep10 = 0.58% top-1 below random. The 1.1M-param head cannot discriminate 75 fine-grained long-tail assembly states from the pooled class token alone." (src/models/mvit_mtl_model.py:289). This observation suggests the problem is NOT insufficient head capacity but rather insufficient feature signal from the class token.

### 4.2 GAP + Linear vs MLP in Video Transformers

- **VideoMAE**: Uses a simple linear classifier on the [CLS] token after pre-training.
- **VideoMAE V2**: Same linear probe approach.
- **Omnivore**: Uses GAP then a linear classifier (no MLP hidden layer).
- **MViTv1/v2 original papers**: Use a single Linear layer as classification head.
- **TimeSformer (Bertasius et al. 2021)**: Linear classifier on [CLS] token.
- **Uniformer (Li et al. 2022)**: Linear classifier.

**Consensus**: The video transformer literature overwhelmingly uses linear classifiers as activity heads. MLP heads are rare and used only when merging features from multiple modalities or scales. Our 3-layer MLP is an outlier in the literature.

### 4.3 Theoretically Grounded Alternative: Fix the Last Layer (Neural Collapse)

The Neural Collapse literature (Papyan, Han, Donoho 2020; Zhu et al. 2021) shows that at convergence of a well-trained network:
1. The last-layer features collapse to their class means (NC1)
2. The class means form a simplex ETF (NC2)
3. The classifier weights align with the means (NC3)
4. The classifier behaves like a nearest-class-mean classifier (NC4)

In imbalanced settings, tail classes do NOT collapse to their own means — they collapse toward the majority class centroid. **The fix**: replace the learned classifier with a fixed simplex ETF (Eq. 1 in Neural Collapse papers) and only learn the backbone features. This guarantees equal geometric separation between all 75 classes, preventing the majority class from dominating.

---

## 5. Class Imbalance Solutions Beyond Reweighting

### 5.1 Focal Loss [VERIFIED]

- **arXiv**: 1708.02002
- **Title**: "Focal Loss for Dense Object Detection"
- **Authors**: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar
- **Venue**: ICCV 2017
- **Key idea**: Modifies cross-entropy by adding a modulating factor `(1 - p_t)^gamma` that down-weights easy examples (high confidence) and focuses training on hard examples. Gamma=2 typically. For class imbalance, gamma reduces the loss contribution from well-classified majority class examples.
- **Relevance to our setup**: Focal loss is designed for multi-label detection (dense prediction), not single-label classification. For our 75-class single-label activity, the primary problem is NOT easy-hard imbalance but rather class-frequency imbalance. Focal loss may not directly help. However, if some classes are "harder" than others (visually ambiguous), focal loss could help focus on those. **Evidence**: Our codebase uses `use_focal=True` for PSR loss but standard CE for activity.

### 5.2 Asymmetric Loss (ASL) [VERIFIED]

- **arXiv**: 2009.14119
- **Title**: "Asymmetric Loss For Multi-Label Classification"
- **Authors**: Emanuel Ben-Baruch, Tal Ridnik, Nadav Zamir, Asaf Noy, Itamar Friedman, Matan Protter, Lihi Zelnik-Manor
- **Venue**: 2021
- **Key idea**: Applies different focusing parameters for positive and negative samples in multi-label classification. For single-label classification, the positive-negative asymmetry is less relevant.
- **Relevance to our setup**: ASL is designed for multi-label where each sample can have multiple positive labels. Our activity task is single-label (each frame has exactly one action class). ASL is not directly applicable but available in our codebase at `src/losses/asymmetric_loss.py`.

### 5.3 Class-Balanced Sampling

- **Square-Root Sampling**: Sampling probability proportional to `sqrt(count)`.
- **Progressive Sampling**: Start with instance-balanced, gradually shift to class-balanced.
- **Class-balanced sampling alone**: Kang et al. (ICLR 2020, Decoupling) showed that class-balanced sampling helps only when the classifier is decoupled from the feature extractor. Training with class-balanced sampling from the start degrades feature quality because rare classes don't have enough variations.

### 5.4 Two-Stage Training (Feature Extractor then Classifier)

**This is the consensus recommendation from the long-tail literature for our setup**:

- **Stage 1**: Train all heads with instance-balanced sampling. The backbone learns general features shared across all 75 classes (this captures variations within head classes, which total the majority of data). The activity head competes fairly with detection/pose/PSR because all use the same sampling distribution.
- **Stage 2**: Freeze backbone. Retrain ONLY the activity classifier with class-balanced sampling (or re-weighting). This addresses the logit scale imbalance without corrupting the shared features.

**Evidence from our codebase**: The `decoupled_act_retrain.py` script already implements this! It freezes the backbone and only trains the activity head. This is likely our best immediate intervention.

### 5.5 Logit Adjustment Theory [VERIFIED]

Our current logit adjustment (Menon et al. 2021) adds `log(prior)` to logits inside the loss. The optimal decision rule for balanced test accuracy is:
```
y_pred = argmax_y f(x)_y + tau * log(prior_y)
```
where prior_y is the training class frequency. With tau=1.0, the decision boundary shifts for rare classes by exactly the log-probability gap. If a rare class has 10 samples vs a head class with 1000 samples, the logit shift is `log(10) - log(1000) = -4.6` for the head class, giving the rare class a 4.6 logit advantage (modulo feature magnitudes).

**Why logit adjustment might still fail in our setup**: The additive correction works when the model has learned discriminative features. If the backbone features haven't learned to separate the classes (because the activity gradients are weak or noisy), logit adjustment alone cannot create separation — it can only shift pre-existing decision boundaries.

---

## 6. Temporal Modeling for Activity with MViT Class Token

### 6.1 Does the MViT Class Token Capture Sufficient Temporal Information?

**Analysis**: The MViTv2-S class token is produced by pool attention, which includes temporal pooling across the 16-frame clip. The class token at the final stage (768-dim) has integrated information across all frames. However:

- **Pool attention aggregates**: The class token interaction with spatial-temporal tokens uses average pooling (not max pooling), which blurs sharp temporal boundaries.
- **Local clip assumption**: MViTv2 processes a 16-frame clip (0.5s at 30fps). For fine-grained assembly actions lasting 2-5 seconds, 0.5s clips may not capture the full action.
- **Temporal position encoding**: MViTv2 uses relative position encoding, which captures temporal ordering within the clip but not long-range dependencies beyond the clip.

**Empirical evidence**: Our head's collapse to 1 class at max confidence 0.03 suggests the class token features are NOT linearly separable for 75 classes. Either (1) the class token aggregates too much information (losing fine discriminative details), or (2) the training signal from 3112 samples is insufficient to learn 75-way separation in 768-dim space (which needs roughly 75 * 768 / (info per sample) dimensional discriminative features).

### 6.2 Temporal Pooling Alternatives

- **TRN (Zhou et al. 2018)**: Temporal relational module aggregates frame-level features by pair-wise comparison. Tested on Something-Something (fine-grained temporal reasoning).
- **TSN (Wang et al. 2016)**: Sparse temporal sampling with segment consensus. Our 16-frame clip is a continuous window; TSN-style sparse sampling could allow longer temporal coverage (e.g., 64 frames sampled sparsely across 2 seconds).
- **Temporal averaging vs max pooling**: Our current approach uses the MViT class token which is a learned aggregation. Explicit temporal pooling (mean, max, or attention over frame features) could supplement the class token.
- **Multi-clip aggregation**: Training with single clips but aggregating multiple clips at inference (Temporal Segment Networks style) often improves accuracy by 3-5%.

### 6.3 Alternative: Frame-Level Features Instead of Clip-Level

Instead of using the class token (one vector per clip), we could:
1. Extract features at each frame time (the intermediate block outputs)
2. Apply a temporal model (e.g., a lightweight Transformer or GRU, similar to our PSR head)
3. Predict activity at frame level

This would decouple temporal granularity from the MViT's clip-level aggregation, allowing the activity head to see finer temporal structure. The PSR head already does this (operates on frame-level P5 features with a causal Transformer).

---

## 7. Knowledge Distillation for Long-Tail

### 7.1 Class-Balanced Distillation [UNVERIFIED - found via Tavily]

- **arXiv**: 2104.05279 (UNVERIFIED)
- **Title**: "Class-Balanced Distillation for Long-Tailed Visual Recognition"
- **Authors**: Ahmet Iscen, Andre Araujo, Boqing Gong, Cordelia Schmid
- **Venue**: BMVC 2021
- **Key idea**: Uses a teacher model pre-trained on the long-tailed data and distills its knowledge to a student, applying class-balanced weighting to the distillation loss. The teacher's soft labels provide richer supervision for tail classes than hard labels alone.
- **Relevance to our setup**: If we train a teacher model with standard CE (which will be biased toward head classes), and use its soft predictions as additional supervision for the student, the student can learn from the teacher's relative confidences between classes. For example, the teacher may confound "tighten_bolt_1" and "tighten_bolt_2" but its soft distribution contains information about which bolt-tightening actions are similar.

### 7.2 Self-Distillation for Tail Classes

Self-distillation (Furlanello et al. 2018, "Born-Again Networks") trains a second model using predictions from the first model as extra supervision, often helping tail classes by providing soft targets that encode class similarity structure.

### 7.3 Using Detection/PSR Knowledge to Inform Activity

Our MTL setup has a unique advantage: the detection head identifies objects (wing, screw, nut) and the PSR head identifies state transitions. These provide complementary signals that can inform activity classification:
- If the detection head outputs "screw" and "screwdriver" in a frame, the activity is likely "tighten_screw" or "untighten_screw"
- If the PSR head detects a transition from one state to another, the activity label at the transition point changes

This cross-task knowledge transfer is underexplored in the literature but directly applicable to our setup.

---

## 8. MixUp/CutMix for Video

### 8.1 MixUp [VERIFIED]

- **arXiv**: 1710.09412
- **Title**: "mixup: Beyond Empirical Risk Minimization"
- **Authors**: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
- **Venue**: ICLR 2018
- **Key idea**: Creates convex combinations of input pairs and their labels: `x' = lambda * x_i + (1 - lambda) * x_j`, `y' = lambda * y_i + (1 - lambda) * y_j`. Acts as a strong regularizer that encourages linear behavior between training samples.
- **Relevance to our setup**: MixUp has been shown to help with long-tail recognition by creating "in-between" samples that interpolate between head and tail classes. For video, MixUp can be applied at the clip level, frame level, or feature level.

### 8.2 Selective Volume MixUp for Video [VERIFIED]

- **arXiv**: 2309.09534
- **Title**: "Selective Volume Mixup for Video Action Recognition"
- **Authors**: Yi Tan, Zhaofan Qiu, Yanbin Hao, Ting Yao, Tao Mei
- **Venue**: 2023
- **Key idea**: Applies MixUp selectively to video volumes (spatio-temporal regions) rather than globally. This preserves the temporal structure of video while still benefiting from MixUp's regularization. Mixing only background regions is less helpful; mixing foreground regions improves action recognition.
- **Relevance to our setup**: For assembly videos, mixing foreground regions (the robotic arm, the workpiece) while preserving background (the workbench) could create realistic augmented samples for tail classes. If we have only 5 frames of "tighten_acorn_nut", mixing its foreground with background from "tighten_bolt_1" (which has 1000 frames) creates a legitimate new sample.

### 8.3 CutMix [VERIFIED]

- **arXiv**: 1905.04899
- **Title**: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
- **Authors**: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo
- **Venue**: ICCV 2019
- **Key idea**: Replaces rectangular regions of one image with patches from another, with labels mixed proportionally to the patch area.
- **Relevance to our setup**: CutMix requires clear spatial structure (bounding boxes or patches). For video clips, CutMix can be applied per-frame or across frames. The spatial structure of assembly video (workpiece at center, hands/tools entering from edges) could make CutMix effective. However, CutMix may destroy the object-context relationships that are critical for fine-grained discrimination.

### 8.4 Remix (Class-Balanced MixUp)

- **arXiv**: 2107.01443
- **Title**: "Remix: Rebalanced Mixup"
- **Key idea**: Rebalances the mixing ratio based on class frequency. When mixing a head class sample with a tail class sample, the tail class label is upweighted: `lambda' = max(lambda, 0.5)`. This ensures the tail class retains a stronger influence in the mixed sample.
- **Relevance to our setup**: This directly addresses our problem. If we mix a tail class sample (5 frames) with a head class sample (1000 frames), standard MixUp with lambda=0.5 would dilute the tail class signal by 50%. Remix preserves tail class signal by ensuring lambda >= 0.5 for tail-head mixtures.

### 8.5 Gen2Balance: Generative Balancing for Long-Tailed Video [VERIFIED]

- **arXiv**: 2606.22416
- **Title**: "Gen2Balance: Generative Balancing for Long-Tailed Video Action Recognition"
- **Authors**: Prajwal Gatti, Simon Jenni, Fabian Caba Heilbron, Dima Damen
- **Venue**: 2026
- **Key idea**: Uses text-to-video generative models to augment the training set. Conditioned on action profiles and training exemplars, the generative model produces synthetic training clips for tail classes. This converts a long-tailed dataset into a more balanced one.
- **Relevance to our setup**: Gen2Balance represents a new paradigm: instead of re-weighting loss terms, generate more data for tail classes. This is computationally expensive but potentially transformative for extreme imbalance. If we can generate even 50-100 synthetic clips for classes with <10 samples, the classification problem becomes tractable.

---

## 9. Diagnosis: Why Our Activity Head Collapses to 1 Class

Based on the literature survey and codebase analysis, here is a multi-factor diagnosis:

### Factor 1: Neural Collapse from Imbalance (PRIMARY)

The literature on neural collapse (Papyan et al. 2020, Zhu et al. 2021, NeurIPS 2022) provides the most direct explanation. In a class-imbalanced setting with cross-entropy loss:

1. **Initialization**: All 75 logits start with similar small values (from Xavier init). The majority class (e.g., class 51 with >1000 samples) appears more frequently, so its logit receives positive gradients more often.
2. **Positive feedback loop**: Each time the majority class is correct, its logit increases while others decrease (cross-entropy pushes incorrect logits down). Over many iterations, the majority class logit grows large while tail class logits shrink.
3. **Collapse point**: When the majority class logit exceeds all others by 3-4 units (in logit space), softmax assigns >95% probability to it for ALL inputs, regardless of the true label. Cross-entropy can no longer recover because the gradient magnitude from a rare correct class (logit of -5) is tiny compared to the majority class gradient (logit of +3).
4. **Max confidence 0.03**: The 0.03 max confidence (below random 1/75 = 0.0133) suggests features are NOT just biased — they are essentially zero-informative. This indicates the backbone features themselves may not be discriminative for the 75-class task.

### Factor 2: MTL Task Conflict

Our MTL setup has detection (with its strong detection loss), PSR (temporal transition loss), and pose (regression loss) all competing for shared backbone features. The detection loss typically produces large gradients (due to dense prediction targets with many anchors). These gradients dominate the backbone update, and the activity head's gradients are comparatively weak. Over time, the backbone specializes to detection features that may not be optimal for activity discrimination.

**Evidence**: The codebase already identifies this: "DET_GT_FRAME_FRACTION=0.40 reweights the sampler to ensure 40% GT-bearing frames, distorting the per-class activity balance" and "activity head predicts only 2/11 classes (top-1 class=7 with 97.5% of frames)".

### Factor 3: Overparameterized Head

Our 3-layer MLP (3.75M parameters) is over-parameterized for 3112 training samples with 75 classes. The literature consensus is that video transformers use linear probes or shallow heads (1 layer). An overparameterized head memorizes majority class patterns and fails to generalize to tail classes.

### Factor 4: Insufficient Temporal Context (SECONDARY)

The 16-frame clip (0.5s at 30fps) may be too short for fine-grained assembly actions that last 2-5 seconds. The MViT class token aggregates information from just 0.5s, which may contain only a fraction of the action. For tail classes with few samples, the model cannot compensate for this information loss.

### Factor 5: Logit Adjustment Without Feature Learning

Logit adjustment (Menon et al. 2021) adds `log(freq)` to shift decision boundaries. BUT this only shifts the existing logit distribution. If the backbone features are not discriminative (Factor 2), logit adjustment cannot create separation — it only biases pre-existing logits. The current implementation applies logit adjustment inside `activity_loss()` but the backbone features may be detection-biased, not activity-informative.

---

## 10. Verdict: Actionable Findings

### Finding 1: Decouple Activity Training from Shared Backbone (HIGHEST IMPACT)
**Paper**: Kang et al. ICLR 2020 (Decoupling, arXiv:1910.09217)
**Papers**: Cao et al. NeurIPS 2019 (LDAM, arXiv:1906.07413)

Train the backbone with all tasks (detection, pose, PSR, activity) using instance-balanced sampling in Stage 1. The activity head learns alongside other tasks but with compatible sampling. In Stage 2, freeze the backbone and retrain ONLY the activity classifier using class-balanced sampling and logit adjustment. Our codebase already has `decoupled_act_retrain.py` for this purpose. The literature consensus is that this two-stage approach is the single most effective intervention for long-tail recognition.

### Finding 2: Replace Learned Classifier with Simplex ETF Geometry (HIGH IMPACT)
**Paper**: Neural Collapse in Imbalanced Learning, NeurIPS 2022
**Paper**: "Mixup meets Neural Collapse in Imbalanced Learning" (OpenReview 2023)

Replace the last linear layer of the activity head (1024-to-75) with a **fixed simplex ETF** classifier. The ETF ensures equal-angle separation (120 degrees for cosine similarity) between all 75 classes, preventing the majority class from dominating the feature space. Only the backbone and feature projector (768->1024) need to be learned. This directly prevents the logit-scale collapse mechanism.

### Finding 3: Reduce Head Complexity or Add Bottleneck (MEDIUM IMPACT)
**Literature consensus**: All major video models (VideoMAE, MViTv1/v2, TimeSformer, Omnivore) use Linear classifiers on pooled features.

Our 3-layer MLP (3.75M params) is 65x larger than typical video classifiers. Options:
- (a) **Replace with Linear probe**: 768->75 (58K params). Trade: may not learn non-linear feature recombination but much lower overfitting risk.
- (b) **Add bottleneck**: 768->256->75 (197K params). The 256-dim bottleneck forces the head to learn a compact representation, reducing overfitting.
- (c) **Add spectral normalization**: On the last layer, to prevent the majority class weight from growing unbounded (the logit scale collapse mechanism).

### Finding 4: Apply Selective Volume MixUp for Tail Class Augmentation (MEDIUM IMPACT)
**Paper**: Tan et al. 2023 (Selective Volume MixUp, arXiv:2309.09534)
**Paper**: Remix (Rebalanced MixUp, arXiv:2107.01443)

Use Selective Volume MixUp on video clips, with Remix-style rebalancing. When mixing a tail class sample with a head class sample, ensure the tail class label dominates the mixed label (lambda >= 0.5 for tail class). This generates augmented tail class samples without requiring a generative model.

### Finding 5: Extend Temporal Context (MEDIUM IMPACT)
**Paper**: TSN (Wang et al. 2016, arXiv:1608.00859)
**Paper**: TDN (Wang et al. 2021, arXiv:2012.10071)

Increase clip length from 16 frames (0.5s at 30fps) to 32 or 64 frames (1-2 seconds) with sparse temporal sampling (TSN-style). Many assembly actions last 2+ seconds, and 0.5s contains only a partial view. Added temporal context provides the backbone with more information about the full action, improving the class token's discriminative power. The current 16-frame clip is a bottleneck for fine-grained assembly action recognition.

### Summary Table

| Finding | Intervention | Paper | Priority | Expected Impact |
|---------|------------|-------|----------|-----------------|
| 1 | Decoupled 2-stage training | Kang ICLR'20 | HIGH | Prevents backbone corruption |
| 2 | Simplex ETF classifier | NC NeurIPS'22 | HIGH | Prevents logit scale collapse |
| 3 | Reduce head complexity | Literature consensus | MEDIUM | Reduces overfitting |
| 4 | Selective Vol MixUp + Remix | Tan'23, Remix'21 | MEDIUM | Augments tail classes |
| 5 | Longer temporal context | TSN ECCV'16 | MEDIUM | Improves feature quality |

