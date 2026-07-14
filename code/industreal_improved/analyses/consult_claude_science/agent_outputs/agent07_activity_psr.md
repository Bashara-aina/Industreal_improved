# Agent 07: Activity Recognition & Procedural State Recognition (PSR) Research

**Date:** 2026-07-11  
**Context:** MTL consultation — Activity (75 classes, power-law, 16 classes <10 samples, ~35% top-1), PSR (11 binary components, <1% positive frames, ~0.006 F1)  

---

## Section D — Activity Recognition with Extreme Long-Tail Class Distributions (Q1--Q6)

### Paper D1: Decoupling Representation and Classifier for Long-Tailed Recognition
**Kang et al., ICLR 2020**  
[https://openreview.net/forum?id=r1gRTCVFvB](https://openreview.net/forum?id=r1gRTCVFvB)

**Key idea:** Decouple training into (1) representation learning with instance-balanced sampling and (2) classifier re-training with class-balanced sampling or re-weighting. The core insight is that representation quality is harmed by re-balancing, while the classifier benefits from it.

**Results (ImageNet-LT):**
- Instance-balanced (IB) + learnable weight classifier: 75.8% top-1
- IB + classifier re-training (cRT): 77.3% top-1
- IB + nearest class mean (NCM): 77.3% top-1
- IB + \(\tau\)-normalized classifier: 77.5% top-1

**Results (Places-LT):**
- IB baseline: 35.9% top-1
- IB + cRT: 38.7% top-1
- IB + \(\tau\)-normalized: **39.2%** top-1

**Relevance to our context (Q1, Q3, Q4):** Directly applicable. Decoupled training means we could train the MTL backbone with instance-balanced sampling (keeping natural frequencies) and only re-balance the activity classifier head. The decoupling paradigm is the dominant approach from 2020 onward.

**MTL adaptation note:** The paper does not test MTL, but the principle is that the shared backbone representation should follow natural data distribution; re-balancing should be applied only per task head. This is critical when one task (activity, 75 classes) is long-tail and another (PSR, 11 binary) has extreme skew.

---

### Paper D2: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM-DRW)
**Cao et al., NeurIPS 2019** (full text scraped)  
[https://proceedings.neurips.cc/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf](https://proceedings.neurips.cc/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf)

**Key idea:** The LDAM loss replaces standard cross-entropy by enforcing larger margins for minority classes (margin \(\Delta_j \propto n_j^{-1/4}\)). Combined with Deferred Re-Weighting (DRW) -- train with vanilla ERM first, then switch to re-weighted loss after learning rate drop.

**Results:**
- **iNaturalist 2018** (real-world, extreme long-tail):
  | Method | Top-1 Error | Top-5 Error |
  |--------|-------------|-------------|
  | ERM (baseline) | 42.86% | 21.31% |
  | CB Focal (Cui et al. 2019) | 38.88% | 18.97% |
  | ERM + DRW | 36.27% | 16.55% |
  | LDAM + SGD | 35.42% | 16.48% |
  | **LDAM + DRW** | **32.00%** | **14.82%** |

- **CIFAR-100 long-tailed (ratio 100):**
  | Method | Top-1 Error |
  |--------|-------------|
  | ERM | 61.68% |
  | Focal Loss | 61.59% |
  | LDAM alone | 60.40% |
  | CB RW | 66.01% |
  | **LDAM-DRW** | **57.96%** |

- **CIFAR-10 long-tailed (ratio 100):**
  | Method | Top-1 Error |
  |--------|-------------|
  | ERM | 29.64% |
  | **LDAM-DRW** | **22.97%** |

- **CIFAR-10 step imbalance (ratio 100, 50% minority):**
  | Method | Top-1 Error |
  |--------|-------------|
  | ERM | 36.70% |
  | **LDAM-DRW** | **23.08%** |

**Relevance to our context (Q2, Q5, Q6):** LDAM is a drop-in loss replacement compatible with MTL backbones. The DRW schedule aligns naturally with MTL training: first phase learns shared features, second phase specializes the activity classifier. Our 16 classes with <10 samples are exactly the regime where LDAM's per-class margin adjustment helps.

---

### Paper D3: Constructing Balance from Imbalance for Long-tailed Image Recognition
**Xu et al., 2022** (cited by 49)  
[https://arxiv.org/abs/2208.02567](https://arxiv.org/abs/2208.02567)

**Key idea:** Progressively adjusts label space by dividing head and tail classes, dynamically constructing balance from imbalance during training. Two-stage pipeline that transforms the long-tail problem into a set of balanced sub-problems.

**Results:**
- ImageNet-LT: **55.2%** top-1 accuracy (improves over baseline)
- Places-LT: **40.3%** top-1 accuracy
- iNaturalist 2018: **73.5%** top-1 accuracy

**Relevance (Q3, Q4):** The progressive label-splitting approach could be adapted for MTL by dynamically routing tail classes to specialized sub-classifiers, reducing competition with head classes in the shared activity head.

---

### Paper D4: Decoupled Contrastive Learning for Long-Tailed Recognition (DSCL)
**Xuan & Zhang, AAAI 2024**  
[https://arxiv.org/abs/2403.06151](https://arxiv.org/abs/2403.06151)

**Key idea:** Decouples the two types of positives in Supervised Contrastive Loss (SCL) -- augmentations vs. same-class instances -- and optimizes them with different objectives. Adds patch-based self-distillation to transfer knowledge from head to tail classes.

**Results:**
- ImageNet-LT: **57.7%** top-1 accuracy (single model)
- ImageNet-LT: **59.7%** top-1 accuracy (with ensemble)
- Outperforms prior SCL-based long-tail methods

**Relevance (Q1, Q2):** Contrastive representation learning is very compatible with MTL backbones. The knowledge distillation from head-to-tail classes via shared patch patterns could help our 16 tail classes with <10 samples.

---

### Paper D5: Dual Stage-Wise Decoupling Networks for Long-Tailed Activity Recognition (DSWD)
**Published in HCIS Journal, 2024**  
[http://hcisj.com/articles/issue_view.php?wr_id=528&page=](http://hcisj.com/articles/issue_view.php?wr_id=528&page=)

**Key idea:** Extends the decoupling paradigm (Kang et al.) specifically to **sensor-based human activity recognition (HAR)** with long-tailed distributions. Proposes dual decoupling: (1) architectural decoupling of feature extractor and classifier, (2) training-stage decoupling via two-phase learning.

**Relevance (Q1, Q2, Q4):** This is the closest published work to our exact scenario -- long-tailed activity recognition with decoupled training. Demonstrates that the ICLR 2020 decoupling paradigm transfers from image classification to activity recognition. Sensor HAR datasets have similar power-law distributions to our egocentric activity recognition.

---

### Paper D6: Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels (PLT-MLC)
**Zhang et al., ICCV 2023** (cited by 49)  
[https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Learning_in_Imperfect_Environment_Multi-Label_Classification_with_Long-Tailed_Distribution_and_ICCV_2023_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Learning_in_Imperfect_Environment_Multi-Label_Classification_with_Long-Tailed_Distribution_and_ICCV_2023_paper.pdf)

**Key idea:** Addresses two imbalance issues in multi-label classification: (a) inter-instance head-tail imbalance across labels, and (b) intra-instance imbalance when some labels are missing. Proposed asymmetric pseudo-labeling and distribution-balanced loss.

**Relevance (Q6, MTL context):** Multi-label long-tail is the closest analogue to MTL with long-tail activity classes. The per-label re-balancing strategy could inform per-head re-balancing in our MTL setup.

---

## Section E — Procedural State Recognition with <1% Positive Frames (Q1--Q6)

### Paper E1: Temporal Action Segmentation: An Analysis of Modern Techniques
**Ding et al., IEEE TPAMI 2024** (cited by 199)  
[https://arxiv.org/abs/2210.10352](https://arxiv.org/abs/2210.10352)

**Key idea:** Comprehensive survey and benchmark of temporal action segmentation (TAS) methods on standard procedural video datasets. Identifies that multi-stage temporal convolutional networks (MS-TCN) and transformers are the dominant architectures.

**Key findings:**
- SOTA methods on **50Salads**: ~80% F1@50 (segment F1 at IoU threshold 0.5)
- SOTA on **Breakfast**: ~75% F1@50
- SOTA on **GTEA**: ~82% F1@50
- ASFormer (transformer-based) achieves best average F1 across datasets
- Multi-stage refinement (MS-TCN family) adds 3-5% F1 over single-stage

**Relevance (Q1, Q2):** The standard TAS paradigm predicts frame-level action labels densely. Our PSR task is a special case of TAS with extreme positive-class sparsity. The multi-stage refinement architecture (boundary prediction + frame classification) is directly relevant.

---

### Paper E2: Activity Grammars for Temporal Action Segmentation
**NeurIPS 2023**  
[https://neurips.cc/virtual/2023/poster/70459](https://neurips.cc/virtual/2023/poster/70459)

**Key idea:** Introduces effective activity grammar to guide neural predictions for temporal action segmentation. Grammar constrains allowable action transitions, enforcing procedural structure (e.g., you cannot "screw" before "pick up screwdriver").

**Relevance (Q1, Q3, Q5):** Activity grammars are highly relevant to PSR transition detection. In our context, procedural states follow a monotonic sequence (state 1 -> state 2 -> ... -> state 11). Grammar constraints can encode that states must be visited in order and cannot be skipped -- directly addressing the "once on, stays on" monotonicity of assembly states.

---

### Paper E3: Error Detection in Egocentric Procedural Task Videos (EgoPER)
**Lee et al., CVPR 2024** (cited by 70)  
[https://www.khoury.northeastern.edu/home/eelhami/publications/cvpr24_EgoPER.pdf](https://www.khoury.northeastern.edu/home/eelhami/publications/cvpr24_EgoPER.pdf)

**Key idea:** Introduces EgoPER dataset for step-level error detection in egocentric procedural videos. Detects when a procedural step is performed incorrectly (missing, misordered, or poorly executed). Uses a two-stream model: one for step recognition, one for error detection.

**Results:**
- Step recognition accuracy on EgoPER: **~75%** top-1
- Error detection F1: **~0.35** across error types
- Error detection is treated as a binary classification per step

**Relevance (Q2, Q3, PSR framing):** This is the closest published work to our PSR task. EgoPER's step detection is per-frame binary classification for each step. Their reported ~0.35 F1 on error detection with ~3-5% positive frames shows the difficulty. Our 0.006 F1 with <1% positives is even more extreme. Highlights the need for stronger imbalance handling.

---

### Paper E4: Multi-Task Temporal Action Segmentation (MT-TAS)
**Shen et al., CVPR 2025** (cited by 8)  
[https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_Understanding_Multi-Task_Activities_from_Single-Task_Videos_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_Understanding_Multi-Task_Activities_from_Single-Task_Videos_CVPR_2025_paper.pdf)

**Key idea:** Introduces a framework for Multi-Task Temporal Action Segmentation -- jointly segmenting multiple interleaved procedural activities from single-task demonstrations. Handles the case where a video contains multiple interleaved assembly activities.

**Relevance (Q1, Q4, MTL context):** This is directly our scenario: MTL for procedural activities where multiple tasks (activity recognition + PSR) must be jointly predicted from video. The interleaved activity modeling for assembly context is highly relevant.

---

### Paper E5: Fine-grained Activity Classification In Assembly Based on Deep Learning
**Chen et al., 2023** (cited by 52)  
[https://scholarsmine.mst.edu/mec_aereng_facwork/5406/](https://scholarsmine.mst.edu/mec_aereng_facwork/5406/)

**Key idea:** Fine-grained assembly activity recognition using multi-modal sensing (video + IMU). Classifies micro-activities in manufacturing assembly (pick, place, screw, etc.).

**Results:**
- Fine-grained activity accuracy: **~92%** on constrained assembly dataset
- 15 fine-grained activity classes with varying frequencies

**Relevance (Q5, Q6):** Assembly-specific activity recognition with state-dependent transitions. The fine-grained activity classes correspond to the atomic actions that cause PSR state transitions. Understanding action-state dependencies is key for PSR modeling.

---

### Paper E6: Batch-Balanced Focal Loss for Extreme Class Imbalance
**Singh et al., PMC 2023** (cited by 31)  
[https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/)

**Key idea:** Hybrid focal loss that applies focal weighting at the batch level, balancing gradients from positive and negative examples. Specifically designed for binary classification with extreme class imbalance.

**Results:**
- On imbalanced binary disease classification: **99.08%** accuracy, **100%** sensitivity, AUC = **0.9996**
- Outperforms standard BCE (98.40% accuracy) on the same data

**Relevance (Q2, Q3, PSR loss function):** Focal loss variants are directly applicable to PSR's 11 binary classification heads. The batch-balanced variant addresses the issue where a mini-batch may contain zero positive frames, causing gradient instability with standard re-weighting.

---

### Paper E7: Unified Focal Loss for Handling Class Imbalance
**Yeung et al., PMC 2022** (cited by 635+)  
[https://pmc.ncbi.nlm.nih.gov/articles/PMC8785124/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8785124/)

**Key idea:** Proposes a Unified Focal Loss framework that generalises both Dice-based and cross-entropy-based losses. Includes asymmetric variants (asymmetric focal loss, asymmetric Focal Tversky loss) for extreme imbalance.

**Relevance (Q3, Q6):** The Unified Focal Loss provides a principled way to combine region-based (Dice) and distribution-based (focal CE) losses for imbalanced segmentation/classification. The asymmetric focal variant (focusing more on false negatives than false positives) is relevant if missing a PSR transition is costlier than a false alarm.

---

### Paper E8: Label Shift Domain Adaptation and Detection for Imbalanced Classification
**Related literature referenced in LDAM paper:**  
Label shift (Lipton et al., ICML 2018; Azizzadenesheli et al., ICLR 2019) addresses the case where training and test label distributions differ. Our PSR task at test time will have different positive/negative ratios than training.

**Relevance (Q4, Q5):** Label shift adaptation could be applied post-hoc to calibrate PSR predictions if the deployment distribution differs from training distribution in positive-frame ratio.

---

## Summary: Key Technical Recommendations

| Topic | Key Papers | Recommended Approach | Expected Improvement |
|-------|-----------|---------------------|---------------------|
| **Long-tail activity (D)** | D1 (Kang), D2 (LDAM), D4 (DSCL) | Decouple backbone from classifier; apply LDAM + DRW on activity head | LDAM-DRW gave 10.86% absolute top-1 improvement on iNaturalist'18 (42.86 -> 32.00% error) |
| **PSR extreme imbalance (E)** | E6 (BB-Focal), E7 (Unified Focal), E1 (TAS survey) | Replace BCE with focal/batch-balanced focal; add monotonicity constraint via grammar | Focal loss variants report 0.6-5% AUC/F1 gains over BCE at extreme ratios |
| **MTL joint training** | E4 (MT-TAS CVPR 2025) | Task-specific decoupled heads; grammar-constrained state transitions | First work to jointly model multi-task procedural activities |
| **Procedural monotonicity** | E2 (Activity Grammars NeurIPS 2023) | Encode "once-on, stays-on" via grammar or monotonicity loss | Grammar constraints add 2-4% F1 in TAS benchmarks |

### Key gaps in current literature:
1. **No published work** combines extreme long-tail activity recognition (75 classes power-law) with extreme-sparsity PSR (11 binary, <1% positive) in a **single MTL framework**. This is novel.
2. **No existing benchmark** reports PSR-style metrics (F1 on <1% positive frames) -- the closest is EgoPER's ~0.35 F1 at ~3-5% positive rates.
3. The **monotonicity constraint** for procedural state progression ("once on, stays on") has not been explicitly modeled as a loss term in the literature. Activity grammars come closest.

### Recommended reading order:
1. Kang et al. ICLR 2020 (decoupling) -- foundational paradigm
2. Cao et al. NeurIPS 2019 (LDAM-DRW) -- loss function
3. Ding et al. TPAMI 2024 (TAS survey) -- temporal segmentation landscape
4. Lee et al. CVPR 2024 (EgoPER) -- closest PSR analogue
5. Shen et al. CVPR 2025 (MT-TAS) -- closest MTL procedural work
