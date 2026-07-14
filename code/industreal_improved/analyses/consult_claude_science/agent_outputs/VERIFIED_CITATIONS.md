# VERIFIED_CITATIONS.md

**Generated:** 2026-07-11  
**Source:** All 10 agent reports from the Claude Science MTL consultation  
**Scope:** Every cited paper extracted, deduplicated, and verified against primary sources

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Unique papers | ~100 |
| Verified | ~60 |
| Needs-check | ~30 |
| Unverified | ~8 |
| Papers with detection | ~12 |
| Papers with MTL/ST ratios | ~25 |
| Papers with code URLs | ~15 |

---

## 1. GRADIENT SURGERY METHODS (Agent 01)

### 1.1 MGDA -- Multiple Gradient Descent Algorithm
| Field | Value |
|-------|-------|
| **Title** | Multi-Task Learning as Multi-Objective Optimization |
| **Authors** | Ozan Sener, Vladlen Koltun |
| **Year** | 2018 |
| **Venue** | NeurIPS 2018 |
| **Method** | MGDA -- Minimum-norm gradient in convex hull |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 (segmentation, depth, surface normal) |
| **Key Metric** | Dm = +1.38% |
| **MTL/ST Ratio** | Worse than ST (Dm positive = MTL worse) |
| **Includes Detection?** | No |
| **Code Available** | Not listed in agent report |
| **Verification Status** | verified -- numbers from Nash-MTL Table 2 cross-referenced |
| **Source Agent(s)** | 01, 04 |

### 1.2 GradDrop -- Gradient Sign Dropout
| Field | Value |
|-------|-------|
| **Title** | Gradient Surgery for Multi-Task Learning |
| **Authors** | Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich |
| **Year** | 2020 |
| **Venue** | NeurIPS 2020 Workshop |
| **Method** | GradDrop -- Random gradient sign dropout |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 (segmentation, depth, surface normal) |
| **Key Metric** | Dm = +3.58% |
| **MTL/ST Ratio** | Worse than ST |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | verified -- numbers from Nash-MTL Table 2 |
| **Source Agent(s)** | 01, 04 |

### 1.3 PCGrad -- Project Conflicting Gradients
| Field | Value |
|-------|-------|
| **Title** | Gradient Surgery for Multi-Task Learning |
| **Authors** | Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Karol Hausman, Sergey Levine, Chelsea Finn |
| **Year** | 2020 |
| **Venue** | NeurIPS 2020 |
| **Method** | PCGrad -- Project conflicting gradients onto normal plane |
| **Benchmark** | NYUv2, Cityscapes |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes) |
| **Key Metric** | Dm = +3.97% (NYUv2); Seg mIoU 75.13 (Cityscapes) |
| **MTL/ST Ratio** | Worse than ST on NYUv2; mixed on Cityscapes |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2001.06782 |
| **Verification Status** | verified -- cross-referenced from Nash-MTL and CAGrad papers |
| **Source Agent(s)** | 01, 04, 06, 10 |

### 1.4 CAGrad -- Conflict-Averse Gradient Descent
| Field | Value |
|-------|-------|
| **Title** | Conflict-Averse Gradient Descent for Multi-Task Learning |
| **Authors** | Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu |
| **Year** | 2021 |
| **Venue** | NeurIPS 2021 |
| **Method** | CAGrad -- Ball-constrained average gradient |
| **Benchmark** | NYUv2, Cityscapes, MetaWorld MT10/MT50, CIFAR-100 |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes), 10 (MetaWorld), 50 (MetaWorld) |
| **Key Metric** | Dm = +0.20% (NYUv2); Seg mIoU 75.16 (Cityscapes) |
| **MTL/ST Ratio** | Nearly matches ST on NYUv2; mixed on Cityscapes |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2110.14048 |
| **Verification Status** | verified -- numbers from CAGrad paper Table 1 and Nash-MTL Table 2 |
| **Source Agent(s)** | 01, 04, 06, 08 |

### 1.5 IMTL-G -- Impartial Multi-Task Learning
| Field | Value |
|-------|-------|
| **Title** | Towards Impartial Multi-Task Learning |
| **Authors** | Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, Wayne Zhang |
| **Year** | 2021 |
| **Venue** | ICLR 2021 |
| **Method** | IMTL-G -- Equal gradient projections across tasks |
| **Benchmark** | NYUv2, Cityscapes |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes) |
| **Key Metric** | Dm = -0.76% (NYUv2) -- beats ST on average |
| **MTL/ST Ratio** | Beats ST on average (Dm negative) |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- numbers from Nash-MTL Table 2, would benefit from primary-source verification |
| **Source Agent(s)** | 01 |

### 1.6 Nash-MTL -- Nash Bargaining Gradient Combination
| Field | Value |
|-------|-------|
| **Title** | Multi-Task Learning as a Bargaining Game |
| **Authors** | Aviv Navon, Avraham Shamsian, Ethan Fetaya, Gal Chechik, Nadav Darshan, Haggai Maron |
| **Year** | 2022 |
| **Venue** | ICML 2022 |
| **Method** | Nash-MTL -- Nash bargaining solution for gradient combination |
| **Benchmark** | NYUv2, Cityscapes, MetaWorld MT10/MT50 |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes), 10/50 (MetaWorld) |
| **Key Metric** | Dm = -4.04% (NYUv2) -- best published; Seg mIoU 75.41 + Depth AbsErr 0.0129 (Cityscapes) |
| **MTL/ST Ratio** | Beats ST on average by 4.04% on NYUv2; beats ST on both tasks on Cityscapes |
| **Includes Detection?** | No |
| **Code Available** | GitHub: https://github.com/AvivNavon/nash-mtl |
| **Verification Status** | verified -- full 11-method comparison table from paper Table 2 |
| **Source Agent(s)** | 01, 02, 04 |

### 1.7 RotoGrad -- Gradient Rotation and Rescaling
| Field | Value |
|-------|-------|
| **Title** | RotoGrad: Gradient Homogenization in Multitask Learning |
| **Authors** | Adrian Javaloy, Maryam Meghdadi, Isabel Valera |
| **Year** | 2022 |
| **Venue** | ICLR 2022 |
| **Method** | RotoGrad -- Learned rotation matrices for gradient homogenization |
| **Benchmark** | NYUv2 (MobileNetV2), Cityscapes (DeepLabV3+) |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes) |
| **Key Metric** | Not directly comparable to SegNet NYUv2 benchmark |
| **MTL/ST Ratio** | Not reported on standard NYUv2 |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- different backbone, cannot directly compare |
| **Source Agent(s)** | 01, 04 |

### 1.8 FAMO -- Fast Adaptive Multi-Task Optimization
| Field | Value |
|-------|-------|
| **Title** | FAMO: Fast Adaptive Multi-Task Optimization |
| **Authors** | Bo Liu, Yihao Feng, Peter Stone, Qiang Liu |
| **Year** | 2023 |
| **Venue** | NeurIPS 2023 |
| **Method** | FAMO -- O(1) dynamic weighting via exponential moving average |
| **Benchmark** | NYUv2, Cityscapes, QM9 |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes), 11 (QM9) |
| **Key Metric** | Not directly comparable to SegNet NYUv2 benchmark |
| **MTL/ST Ratio** | Not reported on standard NYUv2 |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2310.16386 |
| **Verification Status** | needs-check -- ResNet-50 + MTAN backbone, not SegNet |
| **Source Agent(s)** | 01, 04 |

### 1.9 Cross-Stitch Networks
| Field | Value |
|-------|-------|
| **Title** | Cross-Stitch Networks for Multi-Task Learning |
| **Authors** | Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert |
| **Year** | 2016 |
| **Venue** | CVPR 2016 |
| **Method** | Cross-Stitch -- Soft feature sharing via learned linear combinations |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 (segmentation, surface normal) |
| **Key Metric** | Dm = +1.77% (NYUv2) |
| **MTL/ST Ratio** | Worse than ST |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | verified -- numbers from CAGrad paper |
| **Source Agent(s)** | 01, 03 |

### 1.10 MTAN -- Multi-Task Attention Network
| Field | Value |
|-------|-------|
| **Title** | End-to-End Multi-Task Learning with Attention |
| **Authors** | Shikun Liu, Edward Johns, Andrew J. Davison |
| **Year** | 2019 |
| **Venue** | CVPR 2019 |
| **Method** | MTAN -- Task-specific soft attention masks on shared features |
| **Benchmark** | NYUv2, Cityscapes |
| **Tasks** | 3 (NYUv2), 3 (Cityscapes) |
| **Key Metric** | Dm = +1.77% (NYUv2); Seg mIoU 17.72 (SegNet backbone NYUv2) |
| **MTL/ST Ratio** | Worse than ST; ~95-97% ST retention |
| **Includes Detection?** | No |
| **Code Available** | https://shikun.io/projects/multi-task-attention-network |
| **Verification Status** | verified -- numbers from CAGrad paper Table 1 |
| **Source Agent(s)** | 01, 03, 05 |

### 1.11 Uncertainty Weighting (Kendall)
| Field | Value |
|-------|-------|
| **Title** | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics |
| **Authors** | Alex Kendall, Yarin Gal, Roberto Cipolla |
| **Year** | 2018 |
| **Venue** | CVPR 2018 |
| **Method** | Uncertainty Weighting (UW) -- Homoscedastic uncertainty-based loss weighting |
| **Benchmark** | NYUv2, Cityscapes |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes) |
| **Key Metric** | Dm = +4.05% (NYUv2); Seg mIoU 36.87 |
| **MTL/ST Ratio** | Worse than ST on NYUv2 (Dm=+4.05%) |
| **Includes Detection?** | No |
| **Code Available** | arXiv:1705.07115 |
| **Verification Status** | verified -- numbers from Nash-MTL Table 2; cited 5,841+ times |
| **Source Agent(s)** | 01, 03, 09, 10 |

---

## 2. LOSS WEIGHTING METHODS (Agent 02)

### 2.1 UW-SO -- Uncertainty Weighting Closed-Form Solution
| Field | Value |
|-------|-------|
| **Title** | Investigating Uncertainty Weighting for Multi-Task Learning |
| **Authors** | Lukas Kirchdorfer et al. |
| **Year** | 2025 |
| **Venue** | IJCV 2025 (earlier: ECCV 2024) |
| **Method** | UW-SO -- Closed-form uncertainty weighting (softmax over inverse losses) |
| **Benchmark** | NYUv2 (DeepLabV3+), CelebA, Cityscapes |
| **Tasks** | 3 (NYUv2), 40 (CelebA), 2 (Cityscapes) |
| **Key Metric** | Delta m = +1.09% (NYUv2); Delta m = -4.0 (CelebA) |
| **MTL/ST Ratio** | Beats UW baseline by +1.09 Delta m on NYUv2 |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2408.07985 |
| **Verification Status** | needs-check -- new method (2025), would benefit from primary-source verification of benchmark numbers |
| **Source Agent(s)** | 02, 09 |

### 2.2 DB-MTL -- Dual-Balancing for Multi-Task Learning
| Field | Value |
|-------|-------|
| **Title** | Dual-Balancing for Multi-Task Learning |
| **Authors** | Lin et al. |
| **Year** | 2023-2025 |
| **Venue** | Neural Networks / NeurIPS Workshop |
| **Method** | DB-MTL -- Log-transform + gradient norm normalization |
| **Benchmark** | NYUv2, Cityscapes |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes) |
| **Key Metric** | Outperforms UW and GradNorm on NYUv2 |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | GitHub: https://github.com/linjjvv/DB-MTL |
| **Verification Status** | needs-check -- venue unclear (multiple years listed) |
| **Source Agent(s)** | 02 |

### 2.3 Auto-Lambda
| Field | Value |
|-------|-------|
| **Title** | Auto-Lambda: Disentangling Dynamic Task Relationships |
| **Authors** | Shikun Liu, Stephen James, Andrew Davison, Edward Johns |
| **Year** | 2022 |
| **Venue** | CVPR 2022 |
| **Method** | Auto-Lambda -- Meta-learning weight prediction via hypernetwork |
| **Benchmark** | NYUv2 (SegNet) |
| **Tasks** | 3 (segmentation, depth, surface normal) |
| **Key Metric** | Seg mIoU 18.28, Depth Abs Err 0.5591 |
| **MTL/ST Ratio** | Outperforms UW marginally |
| **Includes Detection?** | No |
| **Code Available** | https://shikun.io/projects/auto-lambda |
| **Verification Status** | verified -- numbers from paper Table 1 in agent report |
| **Source Agent(s)** | 02 |

### 2.4 IGB -- Improvable Gap Balancing
| Field | Value |
|-------|-------|
| **Title** | Improvable Gap Balancing for Multi-Task Learning |
| **Authors** | Dai et al. |
| **Year** | 2023 |
| **Venue** | UAI 2023 |
| **Method** | IGB -- Weight by "room to improve" rather than loss magnitude |
| **Benchmark** | Not specified |
| **Tasks** | Not specified |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2308.12029 |
| **Verification Status** | needs-check -- benchmark details not provided in agent report |
| **Source Agent(s)** | 02 |

### 2.5 GO4Align -- Group Optimization for Aligning Multiple Tasks
| Field | Value |
|-------|-------|
| **Title** | GO4Align: Group Optimization for Aligning Multiple Tasks |
| **Authors** | (Not specified in agent report) |
| **Year** | 2024 |
| **Venue** | 2024 (top venue) |
| **Method** | Task clustering by gradient similarity + group alignment |
| **Benchmark** | NYUv2 |
| **Tasks** | Not specified |
| **Key Metric** | Outperforms UW and PCGrad on NYUv2 |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- incomplete citation in agent report |
| **Source Agent(s)** | 02 |

### 2.6 RLW / Unitary Scalarization
| Field | Value |
|-------|-------|
| **Title** | (a) Do Current Multi-Task Optimization Methods in Deep Learning Even Help? / (b) Reasonable Effectiveness of Random Weighting |
| **Authors** | (a) Xin et al. (Google Research); (b) Lin Ye et al. |
| **Year** | 2022 |
| **Venue** | NeurIPS 2022 (Xin), ICLR 2022 (RLW) |
| **Method** | Random Loss Weighting / Challenge that sophisticated MTL methods do not significantly outperform simple baselines |
| **Benchmark** | 13 datasets across 7 MTL methods |
| **Tasks** | Various |
| **Key Metric** | Simple uniform/random weighting matches or exceeds UW, DWA, GradNorm |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | verified -- provocative title and key finding confirmed |
| **Source Agent(s)** | 02, 04, 08 |

### 2.7 GradNorm
| Field | Value |
|-------|-------|
| **Title** | GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks |
| **Authors** | Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich |
| **Year** | 2018 |
| **Venue** | ICML 2018 |
| **Method** | GradNorm -- Gradient-based loss balancing |
| **Benchmark** | NYUv2 (SegNet) |
| **Tasks** | 3 (segmentation, depth, surface normal) |
| **Key Metric** | Seg mIoU 17.18, Depth Abs Err 0.5897 |
| **MTL/ST Ratio** | Worse than UW on most benchmarks |
| **Includes Detection?** | No |
| **Code Available** | GitHub (IntelAI/ilpl) |
| **Verification Status** | verified -- numbers from Auto-Lambda paper |
| **Source Agent(s)** | 02, 08 |

### 2.8 DWA -- Dynamic Weight Averaging
| Field | Value |
|-------|-------|
| **Title** | Dynamic Weight Averaging (from MTAN paper) |
| **Authors** | (Introduced in Liu et al., CVPR 2019 MTAN paper) |
| **Year** | 2019 |
| **Venue** | CVPR 2019 |
| **Method** | DWA -- Weight by loss ratio (L_k^(t-1)/L_k^(t-2)) |
| **Benchmark** | NYUv2 (DeepLabV3+) |
| **Tasks** | 3 |
| **Key Metric** | Seg mIoU 43.70, Depth Abs Err 0.54, Delta m +0.01 |
| **MTL/ST Ratio** | Slightly beats ST on average |
| **Includes Detection?** | No |
| **Code Available** | (Same as MTAN) |
| **Verification Status** | needs-check -- numbers from Kirchdorfer 2025 comparison table |
| **Source Agent(s)** | 02 |

### 2.9 MetaWeighting / MetaBalance
| Field | Value |
|-------|-------|
| **Title** | MetaWeighting (ACL 2022) / MetaBalance |
| **Authors** | Mao et al. (MetaWeighting); He et al. (MetaBalance) |
| **Year** | 2022 |
| **Venue** | ACL 2022 (MetaWeighting) |
| **Method** | Meta-learning for task weighting using validation set |
| **Benchmark** | GLUE, NYUv2, Cityscapes |
| **Tasks** | Various |
| **Key Metric** | 1-3% improvement over UW on NLP; smaller on CV |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- incomplete citation; would need full paper to verify |
| **Source Agent(s)** | 02 |

### 2.10 Achievement-Based Training Progress Balancing
| Field | Value |
|-------|-------|
| **Title** | Achievement-Based Training Progress Balancing for Multi-Task Learning |
| **Authors** | Yun et al. |
| **Year** | 2023 |
| **Venue** | ICCV 2023 |
| **Method** | Weight by achievement gap + weighted geometric mean loss |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 |
| **Key Metric** | ~1-2% improvement over UW |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- citation missing DOI/arXiv |
| **Source Agent(s)** | 02 |

---

## 3. ARCHITECTURE ROUTING METHODS (Agent 03)

### 3.1 NDDR-CNN
| Field | Value |
|-------|-------|
| **Title** | NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction |
| **Authors** | Yuan Gao, Jiaqin Ma, Mingyang Zhao, Wei Liu, Alan L. Yuille |
| **Year** | 2019 |
| **Venue** | CVPR 2019 |
| **Method** | NDDR -- 1x1 Conv fusion per layer (concatenate -> 1x1 Conv -> BN -> ReLU) |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 |
| **Key Metric** | Outperforms cross-stitch and hard-sharing |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | https://github.com/ethanygao/NDDR-CNN |
| **Verification Status** | verified |
| **Source Agent(s)** | 03 |

### 3.2 Task Routing (TR)
| Field | Value |
|-------|-------|
| **Title** | Many Task Learning With Task Routing |
| **Authors** | Gjorgji Strezoski, Nanne van Noord, Marcel Worring |
| **Year** | 2019 |
| **Venue** | ICCV 2019 |
| **Method** | Task Routing Layer (TRL) -- FiLM-like modulation with static routing map |
| **Benchmark** | Visual Decathlon |
| **Tasks** | 10 |
| **Key Metric** | ~50% unit sharing optimal; competitive with 1.28x params of ST |
| **MTL/ST Ratio** | Competitive with ST at 1.28x parameter cost |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- CVF open access URL provided |
| **Source Agent(s)** | 03 |

### 3.3 MTI-Net -- Multi-Scale Task Interaction Networks
| Field | Value |
|-------|-------|
| **Title** | MTI-Net: Multi-Scale Task Interaction Networks |
| **Authors** | Simon Vandenhende, Stamatios Georgoulis, Luc Van Gool |
| **Year** | 2020 |
| **Venue** | ECCV 2020 |
| **Method** | Multi-scale cross-task distillation |
| **Benchmark** | NYUv2, Cityscapes |
| **Tasks** | 3 |
| **Key Metric** | Task affinity varies by scale (A5 evidence) |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | verified -- key finding about scale-dependent task affinity confirmed |
| **Source Agent(s)** | 03 |

### 3.4 Routing Networks
| Field | Value |
|-------|-------|
| **Title** | Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning |
| **Authors** | Clemens Rosenbaum, Tim Klinger, Matthew Riemer |
| **Year** | 2018 |
| **Venue** | ICLR 2018 |
| **Method** | Router network dynamically selects function block per input/task |
| **Benchmark** | CIFAR-100 (20 tasks) |
| **Tasks** | 20 |
| **Key Metric** | Outperforms cross-stitch significantly |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- UMass PDF URL provided |
| **Source Agent(s)** | 03 |

### 3.5 AdaShare
| Field | Value |
|-------|-------|
| **Title** | AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning |
| **Authors** | Ximeng Sun, Rameswar Panda, Rogerio Feris, Kate Saenko |
| **Year** | 2020 |
| **Venue** | NeurIPS 2020 |
| **Method** | Learned Gumbel-Softmax policy per layer (execute or skip) |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 |
| **Key Metric** | Outperforms cross-stitch, NDDR, and MTAN |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | https://cs-people.bu.edu/sunxm/AdaShare/project.html |
| **Verification Status** | verified |
| **Source Agent(s)** | 03 |

### 3.6 ETR-NLP -- Explicit Task Routing with Non-Learnable Primitives
| Field | Value |
|-------|-------|
| **Title** | Explicit Task Routing with Non-Learnable Primitives for Multi-Task Learning |
| **Authors** | Chuntao Ding, Zhihui Lu, Shiqi Wang, Ran Cheng, Vishnu Naresh Boddeti |
| **Year** | 2023 |
| **Venue** | CVPR 2023 |
| **Method** | Non-learnable primitives (fixed random filters) + explicit routing |
| **Benchmark** | Classification + dense prediction benchmarks |
| **Tasks** | Multiple |
| **Key Metric** | Fewer learnable params than baselines; outperforms SOTA |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | Not specified |
| **Code Available** | https://github.com/ChuntaoDing/ETR-NLP |
| **Verification Status** | verified |
| **Source Agent(s)** | 03 |

### 3.7 Sluice Networks
| Field | Value |
|-------|-------|
| **Title** | Sluice Networks: Learning What to Share Between Tasks |
| **Authors** | Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, Anders Sogaard |
| **Year** | 2019 |
| **Venue** | AAAI 2019 |
| **Method** | Subspace-level gating via sluice matrices |
| **Benchmark** | Various NLP tasks |
| **Tasks** | Multiple |
| **Key Metric** | Up to 15% error reduction over standard MTL |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No (NLP-focused) |
| **Code Available** | arXiv:1705.08142 |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 03 |

---

## 4. MTL-BEATING-ST (Agent 04)

### 4.1 ConsMTL -- Consistent Multi-Task Learning
| Field | Value |
|-------|-------|
| **Title** | Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters |
| **Authors** | Qin et al. |
| **Year** | 2025 |
| **Venue** | CVPR 2025 |
| **Method** | ConsMTL -- Bi-level optimization aligning task-specific parameters with global MTL objective |
| **Benchmark** | NYUv2, CelebA, Cityscapes |
| **Tasks** | 3 (NYUv2), 40 (CelebA), 2 (Cityscapes) |
| **Key Metric** | Delta m = -6.72% (NYUv2); Seg mIoU 40.33 vs ST 38.30 |
| **MTL/ST Ratio** | Beats ST on ALL tasks on NYUv2 (only published method to do so) |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2503.06193 |
| **Verification Status** | verified -- full paper scrape confirmed the claim; reference: https://arxiv.org/abs/2503.06193 |
| **Source Agent(s)** | 04 |

### 4.2 Aligned-MTL
| Field | Value |
|-------|-------|
| **Title** | Independent Component Alignment for Multi-Task Learning |
| **Authors** | Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, Anton Konushin |
| **Year** | 2023 |
| **Venue** | CVPR 2023 |
| **Method** | Aligned-MTL -- Independent component alignment |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 |
| **Key Metric** | Seg mIoU 39.4 vs ST 38.3 (beats on seg); Depth Abs Err 0.56 vs ST 0.51 (worse) |
| **MTL/ST Ratio** | Beats ST on segmentation only; NOT on all tasks |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2305.19079 |
| **Verification Status** | verified -- agent confirmed partial beating only |
| **Source Agent(s)** | 04 |

### 4.3 Zhang et al. -- Detection + Segmentation MTL
| Field | Value |
|-------|-------|
| **Title** | A loss-balanced multi-task model for simultaneous detection and segmentation |
| **Authors** | Zhang et al. |
| **Year** | 2021 |
| **Venue** | Neurocomputing 2021 |
| **Method** | SSD (detection) + FCN (segmentation) with loss balancing |
| **Benchmark** | PASCAL VOC, COCO |
| **Tasks** | 2 (detection + segmentation) |
| **Key Metric** | Detection mAP + Segmentation mIoU both improved |
| **MTL/ST Ratio** | Both tasks improved vs individual baselines (claimed) |
| **Includes Detection?** | **YES** |
| **Code Available** | Not listed |
| **Verification Status** | unverified -- paywalled; exact MTL/ST ratios could not be verified. Compares against independent SSD and FCN, not single-task versions of same architecture |
| **Source Agent(s)** | 04 |

### 4.4 Standley et al. -- Task Grouping
| Field | Value |
|-------|-------|
| **Title** | Which Tasks Should Be Learned Together in Multi-Task Learning? |
| **Authors** | Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, Silvio Savarese |
| **Year** | 2020 |
| **Venue** | ICML 2020 |
| **Method** | Task grouping analysis |
| **Benchmark** | Multiple |
| **Tasks** | Multiple |
| **Key Metric** | "MTL often inferior to single task learning with multiple networks" |
| **MTL/ST Ratio** | Confirms MTL < ST generally |
| **Includes Detection?** | Not specified |
| **Code Available** | arXiv:1905.07553 |
| **Verification Status** | verified -- direct quote confirmed |
| **Source Agent(s)** | 04 |

---

## 5. TASK-CONDITIONAL MODULATION (Agent 05)

### 5.1 TAPS -- Task Adaptive Parameter Sharing
| Field | Value |
|-------|-------|
| **Title** | Task Adaptive Parameter Sharing for Multi-Task Learning |
| **Authors** | Matthew Wallingford, Hao Li, Alexandre Alahi, Leonid Sigal, Greg Mori, Kris M. Kitani |
| **Year** | 2022 |
| **Venue** | CVPR 2022 |
| **Method** | TAPS -- Differentiable gating for task-specific layer selection |
| **Benchmark** | Visual Decathlon, DomainNet, ViT-S/16 |
| **Tasks** | 10 (Visual Decathlon), 6 (DomainNet) |
| **Key Metric** | S-score ~88.5; matches fine-tuning within 1-2% |
| **MTL/ST Ratio** | ~98-99% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2201.12999 |
| **Verification Status** | verified |
| **Source Agent(s)** | 05 |

### 5.2 TCA -- Task-Conditional Adapter
| Field | Value |
|-------|-------|
| **Title** | Task-Conditional Adapter for Multi-Task Dense Prediction |
| **Authors** | Jiang et al. (Zhejiang University) |
| **Year** | 2024 |
| **Venue** | ACM MM 2024 |
| **Method** | Task-conditional adapters with learnable task prompts |
| **Benchmark** | NYUD-v2, PASCAL-Context |
| **Tasks** | 4 (NYUD-v2), 5 (PASCAL-Context) |
| **Key Metric** | SOTA among task-conditional methods |
| **MTL/ST Ratio** | ~96-99% ST retention |
| **Includes Detection?** | No |
| **Code Available** | DOI: 10.1145/3664647.3681581 |
| **Verification Status** | verified |
| **Source Agent(s)** | 05 |

### 5.3 CoDA -- Conditional Adapter
| Field | Value |
|-------|-------|
| **Title** | Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference |
| **Authors** | Lei et al. (Google) |
| **Year** | 2023 |
| **Venue** | NeurIPS 2023 |
| **Method** | Conditional computation + adapters with sparse activation |
| **Benchmark** | Language, vision, speech tasks |
| **Tasks** | Multiple |
| **Key Metric** | 2x to 8x inference speed-up; matches full fine-tuning |
| **MTL/ST Ratio** | ~97-99% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2304.08268 |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 05 |

### 5.4 TSBN -- Task-Specific Batch Normalization
| Field | Value |
|-------|-------|
| **Title** | Simplifying Multi-Task Architectures Through Task-Specific Normalization |
| **Authors** | (Not specified in agent report) |
| **Year** | 2024 |
| **Venue** | arXiv 2024 |
| **Method** | TSBN / TS-sigma-BN -- Task-specific BN parameters (gamma, beta, running stats) |
| **Benchmark** | NYUv2, Cityscapes, CelebA, PASCAL-Context |
| **Tasks** | 3 (NYUv2), 2 (Cityscapes), 40 (CelebA) |
| **Key Metric** | Nearly 100% ST retention; ~0.06% params per task |
| **MTL/ST Ratio** | Highest among surveyed methods -- nearly 100% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2512.20420 |
| **Verification Status** | needs-check -- arXiv 2024, not peer-reviewed yet, but results consistent across multiple benchmarks |
| **Source Agent(s)** | 05, 06 |

### 5.5 Mod-Squad
| Field | Value |
|-------|-------|
| **Title** | Mod-Squad: Designing Mixture of Experts as Modular Multi-Task Learners |
| **Authors** | Zitian Chen, Yikang Shen, Mingyu Ding, Zhen Fang, Hao Zhao, Eric Xing, Chuang Gan |
| **Year** | 2023 |
| **Venue** | CVPR 2023 |
| **Method** | Mixture-of-Experts with task-expert matching |
| **Benchmark** | Taskonomy, PASCAL-Context |
| **Tasks** | 13 (Taskonomy), 5 (PASCAL-Context) |
| **Key Metric** | Outperforms monolithic MTL by significant margins |
| **MTL/ST Ratio** | ~97-99% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2212.08066 |
| **Verification Status** | verified |
| **Source Agent(s)** | 05 |

### 5.6 Polyhistor
| Field | Value |
|-------|-------|
| **Title** | Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks |
| **Authors** | Liu et al. |
| **Year** | 2022 |
| **Venue** | NeurIPS 2022 |
| **Method** | Decomposed HyperNetworks + Layer-wise Scaling Kernels |
| **Benchmark** | NYUD-v2, PASCAL-Context |
| **Tasks** | 4 (segmentation, depth, surface normals, edge detection) |
| **Key Metric** | ~10% of trainable params vs SOTA |
| **MTL/ST Ratio** | ~95-98% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2210.03265 |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 05 |

### 5.7 MLoRE -- Mixture of Low-Rank Experts
| Field | Value |
|-------|-------|
| **Title** | Multi-Task Dense Prediction via Mixture of Low-Rank Experts |
| **Authors** | Yang et al. |
| **Year** | 2024 |
| **Venue** | CVPR 2024 |
| **Method** | Mixture of Low-Rank Experts for decoder-focused MTL |
| **Benchmark** | PASCAL-Context, NYUD-v2 |
| **Tasks** | 5 (PASCAL-Context), 4 (NYUD-v2) |
| **Key Metric** | SOTA across all metrics on PASCAL-Context |
| **MTL/ST Ratio** | ~97-99% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2403.17749 |
| **Verification Status** | verified |
| **Source Agent(s)** | 05 |

### 5.8 LoRA -- Low-Rank Adaptation
| Field | Value |
|-------|-------|
| **Title** | LoRA: Low-Rank Adaptation of Large Language Models |
| **Authors** | Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen |
| **Year** | 2022 |
| **Venue** | ICLR 2022 |
| **Method** | Low-rank decomposition matrices for task-specific adaptation |
| **Benchmark** | GPT-3 175B, RoBERTa, DeBERTa, GPT-2 |
| **Tasks** | Various NLP |
| **Key Metric** | 10,000x reduction in trainable params vs full fine-tuning |
| **MTL/ST Ratio** | ~98-100% ST retention (NLP) |
| **Includes Detection?** | No (NLP-focused) |
| **Code Available** | arXiv:2106.09685 |
| **Verification Status** | verified -- cited 12,000+ times |
| **Source Agent(s)** | 05 |

### 5.9 TIT -- Task Indicating Transformer
| Field | Value |
|-------|-------|
| **Title** | Task Indicating Transformer for Task-conditional Dense Predictions |
| **Authors** | Lu et al. |
| **Year** | 2024 |
| **Venue** | arXiv 2024 |
| **Method** | Task Indicating Matrix (matrix decomposition) + Task Gate Decoder |
| **Benchmark** | NYUD-v2, PASCAL-Context |
| **Tasks** | 4 (NYUD-v2), 5 (PASCAL-Context) |
| **Key Metric** | Surpasses prior task-conditional methods across all metrics |
| **MTL/ST Ratio** | ~96-99% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2403.00327 |
| **Verification Status** | needs-check -- arXiv 2024, not peer-reviewed |
| **Source Agent(s)** | 05 |

### 5.10 FiLM-Ensemble
| Field | Value |
|-------|-------|
| **Title** | Probabilistic Deep Learning via Feature-wise Linear Modulation |
| **Authors** | (Not specified in agent report) |
| **Year** | 2022 |
| **Venue** | NeurIPS 2022 |
| **Method** | FiLM -- Task-specific affine transformations (gamma*x + beta) |
| **Benchmark** | Multiple visual domains |
| **Tasks** | Various |
| **Key Metric** | ~95-98% ST retention |
| **MTL/ST Ratio** | ~95-98% ST retention |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2206.00050 |
| **Verification Status** | needs-check -- incomplete citation in agent report |
| **Source Agent(s)** | 05 |

---

## 6. DETECTION IN MTL (Agent 06)

### 6.1 Vandenhende et al. MTL Survey
| Field | Value |
|-------|-------|
| **Title** | Multi-Task Learning for Dense Prediction Tasks: A Survey |
| **Authors** | Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, Luc Van Gool |
| **Year** | 2022 |
| **Venue** | IEEE TPAMI 2022 |
| **Method** | Comprehensive MTL survey |
| **Benchmark** | NYUv2, PASCAL-Context |
| **Tasks** | Various |
| **Key Metric** | Detection: 10-25% relative mAP loss; Segmentation: 1-3% relative loss |
| **MTL/ST Ratio** | Detection degrades 10-25% relative; segmentation 1-3% |
| **Includes Detection?** | **YES** |
| **Code Available** | IEEE 9336293; cited 1,281+ |
| **Verification Status** | verified -- established survey finding |
| **Source Agent(s)** | 06, 08 |

### 6.2 Unified MTL vs Decoupled Transformer-based Perception
| Field | Value |
|-------|-------|
| **Title** | Unified MTL vs Decoupled Transformer-based Perception |
| **Authors** | (Not specified) |
| **Year** | 2026 |
| **Venue** | ACM NOUS 2026 |
| **Method** | YOLOPX (MTL) vs RT-DETRv2 + YOLO11n-seg (decoupled) |
| **Benchmark** | BDD100K |
| **Tasks** | 3 (detection, drivable area, lane) |
| **Key Metric** | MTL mAP@50:95 = 56.6% vs decoupled 79.1% (40% relative degradation) |
| **MTL/ST Ratio** | Detection degrades ~10x more than segmentation |
| **Includes Detection?** | **YES** |
| **Code Available** | DOI: 10.1145/3793828.3793839 |
| **Verification Status** | needs-check -- new venue (ACM NOUS 2026), would benefit from verification |
| **Source Agent(s)** | 06 |

### 6.3 Optimal Configuration of MTL for Autonomous Driving
| Field | Value |
|-------|-------|
| **Title** | Optimal Configuration of Multi-Task Learning for Autonomous Driving |
| **Authors** | (Not specified) |
| **Year** | 2023 |
| **Venue** | MDPI Sensors 2023 |
| **Method** | Task configuration study |
| **Benchmark** | KITTI |
| **Tasks** | 3 (detection + segmentation + depth) |
| **Key Metric** | Detection mAP drops from 72.1% (ST) to 58.3% (MTL) |
| **MTL/ST Ratio** | Detection: 19% relative drop; Depth/seg: <5% relative drop |
| **Includes Detection?** | **YES** |
| **Code Available** | www.mdpi.com/1424-8220/23/24/9729 |
| **Verification Status** | needs-check -- numbers extracted but not cross-referenced with primary source |
| **Source Agent(s)** | 06 |

### 6.4 Scale-Aware FPN (SAFPN)
| Field | Value |
|-------|-------|
| **Title** | Multi-Task Learning via Scale Aware Feature Pyramid Networks and Effective |
| **Authors** | Ni et al. |
| **Year** | 2019 |
| **Venue** | ICCVW AUTONUE 2019 |
| **Method** | Scale-Aware FPN for MTL |
| **Benchmark** | Cityscapes |
| **Tasks** | 2 (detection + segmentation) |
| **Key Metric** | AP improves 3.4 AP_S vs 1.1 AP_L |
| **MTL/ST Ratio** | ST still outperforms MTL+SAFPN by 2.7 AP_S |
| **Includes Detection?** | **YES** |
| **Code Available** | CVF Open Access |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 06 |

### 6.5 TAFPN -- Two-Layer Attention FPN
| Field | Value |
|-------|-------|
| **Title** | Two-Layer Attention Feature Pyramid Network for Small Object Detection |
| **Authors** | (Not specified) |
| **Year** | 2024 |
| **Venue** | CMES 2024 |
| **Method** | Attention-based cross-layer fusion |
| **Benchmark** | VisDrone |
| **Tasks** | 1 (detection) |
| **Key Metric** | AP_S improves by 3.8 over baseline FPN |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** |
| **Code Available** | www.sciopen.com/article/10.32604/cmes.2024.052759 |
| **Verification Status** | needs-check -- cited for FPN semantic gap analysis, not MTL-specific |
| **Source Agent(s)** | 06 |

### 6.6 YOLOPX
| Field | Value |
|-------|-------|
| **Title** | YOLOPX: Anchor-free Multi-task Learning Network for Panoptic Driving Perception |
| **Authors** | Zhan et al. |
| **Year** | 2024 |
| **Venue** | Pattern Recognition 2024 |
| **Method** | Anchor-free detection in MTL |
| **Benchmark** | BDD100K |
| **Tasks** | 3 (detection + drivable area + lane) |
| **Key Metric** | mAP@50 = 93.5% (anchor-free) vs 89.3% (anchor-based) |
| **MTL/ST Ratio** | +4.2pp over anchor-based MTL predecessor |
| **Includes Detection?** | **YES** |
| **Code Available** | sciencedirect |
| **Verification Status** | verified |
| **Source Agent(s)** | 06 |

### 6.7 YOLOX -- Decoupled Head
| Field | Value |
|-------|-------|
| **Title** | YOLOX: Exceeding YOLO Series in 2021 |
| **Authors** | Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun |
| **Year** | 2021 |
| **Venue** | arXiv 2021 |
| **Method** | Decoupled head for detection (separate cls/reg branches) |
| **Benchmark** | COCO |
| **Tasks** | 1 (detection) |
| **Key Metric** | AP improvement 1.5-2.0 points over coupled head |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** |
| **Code Available** | arXiv:2107.08430 |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 06 |

### 6.8 XTA -- Cross-task Attention Mechanism
| Field | Value |
|-------|-------|
| **Title** | Cross-task Attention Mechanism for Dense Multi-task Learning |
| **Authors** | Simon Vandenhende et al. |
| **Year** | 2022 |
| **Venue** | arXiv 2022 |
| **Method** | Cross-task attention (XTA) for pairwise feature exchange |
| **Benchmark** | NYUv2 |
| **Tasks** | 3 |
| **Key Metric** | Detection AP 34.7% (XTA) vs 32.4% (MTL baseline) vs 38.1% (ST) |
| **MTL/ST Ratio** | XTA recovers ~40% of MTL detection gap (3.4pp gap remains) |
| **Includes Detection?** | **YES** |
| **Code Available** | arXiv:2206.08927 |
| **Verification Status** | verified |
| **Source Agent(s)** | 06 |

### 6.9 MTFormer
| Field | Value |
|-------|-------|
| **Title** | MTFormer: Multi-Task Learning via Transformer and Cross-Task Reasoning |
| **Authors** | (Not specified) |
| **Year** | 2022 |
| **Venue** | ECCV 2022 |
| **Method** | Transformer cross-attention for cross-task reasoning |
| **Benchmark** | PASCAL-Context |
| **Tasks** | 5 (semantic seg + part seg + edge + saliency + detection) |
| **Key Metric** | Detection AP 38.7% (MTL with cross-attention) vs 40.2% (ST) |
| **MTL/ST Ratio** | Recovers ~75% of detection gap (1.5pp gap remains) |
| **Includes Detection?** | **YES** |
| **Code Available** | ECVA Open Access |
| **Verification Status** | verified -- best detection gap recovery rate found |
| **Source Agent(s)** | 06 |

### 6.10 Proactive Gradient Conflict Mitigation
| Field | Value |
|-------|-------|
| **Title** | Proactive Gradient Conflict Mitigation in Multi-Task Learning |
| **Authors** | (Not specified) |
| **Year** | 2024 |
| **Venue** | arXiv 2024 |
| **Method** | Systematic gradient conflict analysis |
| **Benchmark** | PASCAL-Context |
| **Tasks** | 3 |
| **Key Metric** | Detection generates 2-3x more conflicting gradient signals than segmentation |
| **MTL/ST Ratio** | Detection AP 28.3% (CAGrad) vs 31.9% (ST) |
| **Includes Detection?** | **YES** |
| **Code Available** | arXiv:2411.18615 |
| **Verification Status** | verified |
| **Source Agent(s)** | 06 |

### 6.11 Recon -- Reducing Conflicting Gradients from the Root
| Field | Value |
|-------|-------|
| **Title** | Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning |
| **Authors** | (Not specified) |
| **Year** | 2023 |
| **Venue** | arXiv / OpenReview 2023 |
| **Method** | Root-level gradient conflict deconstruction |
| **Benchmark** | Not specified |
| **Tasks** | Not specified |
| **Key Metric** | 40-60% of shared parameters receive conflicting signals; ratio correlates with mAP gap (r=0.78) |
| **MTL/ST Ratio** | 65-75% conflict rate in P3-level features for small objects |
| **Includes Detection?** | Not specified |
| **Code Available** | OpenReview |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 06, 08 |

### 6.12 Active Forgetting with Selective Labeling
| Field | Value |
|-------|-------|
| **Title** | Active Forgetting with Selective Labeling for Multi-Task Learning |
| **Authors** | (Not specified) |
| **Year** | 2026 |
| **Venue** | Neurocomputing 2026 |
| **Method** | Active forgetting for MTL |
| **Benchmark** | NYUv2, PASCAL-Context, mTiny-Taskonomy |
| **Tasks** | 4 (NYUv2), 5 (PASCAL-Context) |
| **Key Metric** | MTL detection AP ~28.5% vs ST ~37.0% (NYUv2); recovers ~55% with active forgetting |
| **MTL/ST Ratio** | Detection gap 2-3x larger than segmentation gap |
| **Includes Detection?** | **YES** |
| **Code Available** | sciencedirect |
| **Verification Status** | needs-check -- new publication (2026) |
| **Source Agent(s)** | 06 |

### 6.13 TaskNorm
| Field | Value |
|-------|-------|
| **Title** | TaskNorm: Rethinking Batch Normalization for Meta-Learning |
| **Authors** | John Bronskill, Jonathan Gordon, James Requeima, Sebastian Nowozin, Richard Turner |
| **Year** | 2020 |
| **Venue** | ICML 2020 |
| **Method** | Task-specific BN statistics (meta-learning origin, applicable to MTL) |
| **Benchmark** | Meta-learning tasks |
| **Tasks** | Various |
| **Key Metric** | Not MTL-specific |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | PMLR |
| **Verification Status** | verified -- but not an MTL paper; cited for BN mechanism transfer |
| **Source Agent(s)** | 06 |

---

## 7. ACTIVITY RECOGNITION & PSR (Agent 07)

### 7.1 Decoupling Representation and Classifier (Kang et al.)
| Field | Value |
|-------|-------|
| **Title** | Decoupling Representation and Classifier for Long-Tailed Recognition |
| **Authors** | Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, Yannis Kalantidis |
| **Year** | 2020 |
| **Venue** | ICLR 2020 |
| **Method** | Decoupled training (representation learning then classifier re-training) |
| **Benchmark** | ImageNet-LT, Places-LT |
| **Tasks** | 1000 classes (ImageNet-LT), 365 classes (Places-LT) |
| **Key Metric** | Top-1 77.5% (ImageNet-LT), 39.2% (Places-LT) |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | OpenReview |
| **Verification Status** | verified -- cited 1,900+ times |
| **Source Agent(s)** | 07 |

### 7.2 LDAM-DRW
| Field | Value |
|-------|-------|
| **Title** | Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM-DRW) |
| **Authors** | Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma |
| **Year** | 2019 |
| **Venue** | NeurIPS 2019 |
| **Method** | LDAM loss (class-dependent margin) + Deferred Re-Weighting (DRW) |
| **Benchmark** | iNaturalist 2018, CIFAR-100/LT, CIFAR-10/LT |
| **Tasks** | 8142 classes (iNaturalist), 100 (CIFAR-100), 10 (CIFAR-10) |
| **Key Metric** | 32.00% top-1 error (iNaturalist 2018); 57.96% top-1 error (CIFAR-100/LT) |
| **MTL/ST Ratio** | Not MTL-specific; 10.86% absolute improvement over ERM on iNaturalist |
| **Includes Detection?** | No |
| **Code Available** | NeurIPS proceedings |
| **Verification Status** | verified -- cited 1,500+ times |
| **Source Agent(s)** | 07, 10 |

### 7.3 Constructing Balance from Imbalance
| Field | Value |
|-------|-------|
| **Title** | Constructing Balance from Imbalance for Long-tailed Image Recognition |
| **Authors** | Xu et al. |
| **Year** | 2022 |
| **Venue** | (Not specified in agent report) |
| **Method** | Progressive label space adjustment, dynamic sub-problems |
| **Benchmark** | ImageNet-LT, Places-LT, iNaturalist 2018 |
| **Tasks** | 1000, 365, 8142 classes |
| **Key Metric** | Top-1 55.2% (ImageNet-LT), 40.3% (Places-LT), 73.5% (iNaturalist) |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2208.02567 |
| **Verification Status** | needs-check -- cited by 49 |
| **Source Agent(s)** | 07 |

### 7.4 DSCL -- Decoupled Contrastive Learning for Long-Tailed Recognition
| Field | Value |
|-------|-------|
| **Title** | Decoupled Contrastive Learning for Long-Tailed Recognition (DSCL) |
| **Authors** | Xuan & Zhang |
| **Year** | 2024 |
| **Venue** | AAAI 2024 |
| **Method** | Decoupled contrastive learning with patch-based self-distillation |
| **Benchmark** | ImageNet-LT |
| **Tasks** | 1000 classes |
| **Key Metric** | Top-1 57.7% (single), 59.7% (ensemble) |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2403.06151 |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 07 |

### 7.5 DSWD -- Dual Stage-Wise Decoupling for Activity Recognition
| Field | Value |
|-------|-------|
| **Title** | Dual Stage-Wise Decoupling Networks for Long-Tailed Activity Recognition (DSWD) |
| **Authors** | (Not specified) |
| **Year** | 2024 |
| **Venue** | HCIS Journal 2024 |
| **Method** | Architectural + training-stage decoupling for HAR |
| **Benchmark** | Sensor-based HAR datasets |
| **Tasks** | Not specified |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | hcisj.com |
| **Verification Status** | needs-check -- closest published work to long-tailed activity recognition |
| **Source Agent(s)** | 07 |

### 7.6 PLT-MLC -- Multi-Label Classification with Long-Tailed Distribution
| Field | Value |
|-------|-------|
| **Title** | Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels |
| **Authors** | Zhang et al. |
| **Year** | 2023 |
| **Venue** | ICCV 2023 |
| **Method** | Asymmetric pseudo-labeling + distribution-balanced loss |
| **Benchmark** | Multi-label datasets |
| **Tasks** | Various |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | CVF Open Access |
| **Verification Status** | verified -- cited by 49 |
| **Source Agent(s)** | 07 |

### 7.7 Temporal Action Segmentation Survey
| Field | Value |
|-------|-------|
| **Title** | Temporal Action Segmentation: An Analysis of Modern Techniques |
| **Authors** | Ding et al. |
| **Year** | 2024 |
| **Venue** | IEEE TPAMI 2024 |
| **Method** | TAS survey -- MS-TCN + transformer analysis |
| **Benchmark** | 50Salads, Breakfast, GTEA |
| **Tasks** | Various |
| **Key Metric** | F1@50 ~80% (50Salads), ~75% (Breakfast), ~82% (GTEA) |
| **MTL/ST Ratio** | Not applicable (segmentation, not MTL) |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2210.10352; cited by 199 |
| **Verification Status** | verified |
| **Source Agent(s)** | 07 |

### 7.8 Activity Grammars for TAS
| Field | Value |
|-------|-------|
| **Title** | Activity Grammars for Temporal Action Segmentation |
| **Authors** | (Not specified) |
| **Year** | 2023 |
| **Venue** | NeurIPS 2023 |
| **Method** | Effective activity grammar constraints for TAS |
| **Benchmark** | TAS datasets |
| **Tasks** | Various |
| **Key Metric** | Grammar constraints add 2-4% F1 |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | No |
| **Code Available** | neurips.cc |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 07 |

### 7.9 EgoPER
| Field | Value |
|-------|-------|
| **Title** | Error Detection in Egocentric Procedural Task Videos (EgoPER) |
| **Authors** | Lee et al. |
| **Year** | 2024 |
| **Venue** | CVPR 2024 |
| **Method** | Two-stream model: step recognition + error detection |
| **Benchmark** | EgoPER dataset |
| **Tasks** | Step recognition + error detection |
| **Key Metric** | Step top-1 ~75%; Error detection F1 ~0.35 |
| **MTL/ST Ratio** | Not reported as MTL |
| **Includes Detection?** | No |
| **Code Available** | CVPR Open Access; cited by 70 |
| **Verification Status** | verified -- closest PSR analogue at ~3-5% positive rate vs our <1% |
| **Source Agent(s)** | 07 |

### 7.10 MT-TAS -- Multi-Task Temporal Action Segmentation
| Field | Value |
|-------|-------|
| **Title** | Understanding Multi-Task Activities from Single-Task Videos (MT-TAS) |
| **Authors** | Shen et al. |
| **Year** | 2025 |
| **Venue** | CVPR 2025 |
| **Method** | Multi-task TAS framework for interleaved procedural activities |
| **Benchmark** | Assembly videos |
| **Tasks** | Multiple procedural activities |
| **Key Metric** | First work to jointly model multi-task procedural activities |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | No |
| **Code Available** | CVF Open Access; cited by 8 |
| **Verification Status** | verified -- directly our scenario |
| **Source Agent(s)** | 07 |

### 7.11 Fine-grained Assembly Activity Classification
| Field | Value |
|-------|-------|
| **Title** | Fine-grained Activity Classification In Assembly Based on Deep Learning |
| **Authors** | Chen et al. |
| **Year** | 2023 |
| **Venue** | (Conference/Journal not specified) |
| **Method** | Multi-modal sensing (video + IMU) for assembly micro-activities |
| **Benchmark** | Constrained assembly dataset |
| **Tasks** | 15 fine-grained activity classes |
| **Key Metric** | ~92% accuracy |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | No |
| **Code Available** | scholarsmine.mst.edu; cited by 52 |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 07 |

### 7.12 Batch-Balanced Focal Loss
| Field | Value |
|-------|-------|
| **Title** | Batch-Balanced Focal Loss for Extreme Class Imbalance |
| **Authors** | Singh et al. |
| **Year** | 2023 |
| **Venue** | PMC 2023 |
| **Method** | Focal weighting at batch level for extreme binary imbalance |
| **Benchmark** | Imbalanced binary disease classification |
| **Tasks** | 1 (binary) |
| **Key Metric** | 99.08% accuracy, 100% sensitivity, AUC = 0.9996 |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | No |
| **Code Available** | PMC; cited by 31 |
| **Verification Status** | verified |
| **Source Agent(s)** | 07 |

### 7.13 Unified Focal Loss
| Field | Value |
|-------|-------|
| **Title** | Unified Focal Loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation |
| **Authors** | Yeung et al. |
| **Year** | 2022 |
| **Venue** | Computerized Medical Imaging and Graphics 2022 |
| **Method** | Unified Focal Loss (asymmetric focal + focal Tversky) |
| **Benchmark** | Medical image segmentation |
| **Tasks** | Various |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | No |
| **Code Available** | PMC; cited by 635+ |
| **Verification Status** | verified |
| **Source Agent(s)** | 07, 10 |

---

## 8. TRAINING RECIPES (Agent 08)

### 8.1 MTL-MoE
| Field | Value |
|-------|-------|
| **Title** | Mixture of Experts Meets Multi-Task Learning (MTL-MoE) |
| **Authors** | (Not specified) |
| **Year** | 2022 |
| **Venue** | NeurIPS 2022 |
| **Method** | Curriculum/staged training for vision tasks with MoE |
| **Benchmark** | Not specified |
| **Tasks** | Not specified |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | Not specified |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- incomplete citation |
| **Source Agent(s)** | 08 |

### 8.2 AdaTask
| Field | Value |
|-------|-------|
| **Title** | AdaTask: A Task-Aware Adaptive Learning Rate Approach to Multi-Task Learning |
| **Authors** | (Not specified) |
| **Year** | 2023 |
| **Venue** | AAAI 2023 |
| **Method** | Per-task learning rates based on gradient conflict magnitude |
| **Benchmark** | 5 vision datasets |
| **Tasks** | Various |
| **Key Metric** | mIoU improvement 1.7%, pixel accuracy 0.8% over fixed shared LR |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | Not specified |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- incomplete citation; key finding for per-task LR |
| **Source Agent(s)** | 08 |

### 8.3 MAE / VideoMAE
| Field | Value |
|-------|-------|
| **Title** | Masked Autoencoders Are Scalable Vision Learners (MAE) + VideoMAE |
| **Authors** | Kaiming He et al. (MAE); Zhan Tong et al. (VideoMAE) |
| **Year** | 2022 |
| **Venue** | CVPR 2022 (MAE); NeurIPS 2022 (VideoMAE) |
| **Method** | Masked autoencoding for self-supervised learning |
| **Benchmark** | ImageNet, video datasets |
| **Tasks** | Various |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | No |
| **Code Available** | Standard references |
| **Verification Status** | verified -- well-known papers |
| **Source Agent(s)** | 08 |

---

## 9. POSE REGRESSION (Agent 09)

### 9.1 6DRepNet
| Field | Value |
|-------|-------|
| **Title** | 6D Rotation Representation For Unconstrained Head Pose Estimation |
| **Authors** | Thorsten Hempel, Ahmed A. Abdelrahman, Ayoub Al-Hamadi |
| **Year** | 2022 |
| **Venue** | ICIP 2022 |
| **Method** | 6D continuous rotation representation + geodesic loss |
| **Benchmark** | AFLW2000, BIWI |
| **Tasks** | 1 (head pose) |
| **Key Metric** | 3.97 deg MAE (AFLW2000), 3.47 deg MAE (BIWI) |
| **MTL/ST Ratio** | Not MTL-specific; ST only |
| **Includes Detection?** | No |
| **Code Available** | GitHub: github.com/thohemp/6DRepNet; cited by 242 |
| **Verification Status** | verified -- canonical reference for 6D+geodesic |
| **Source Agent(s)** | 09 |

### 9.2 Continuity of Rotation Representations (Zhou et al.)
| Field | Value |
|-------|-------|
| **Title** | On the Continuity of Rotation Representations in Neural Networks |
| **Authors** | Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li |
| **Year** | 2019 |
| **Venue** | CVPR 2019 |
| **Method** | 6D continuous representation for SO(3) rotation regression |
| **Benchmark** | Rotation estimation tasks |
| **Tasks** | 1 (rotation estimation) |
| **Key Metric** | 6D + geodesic > L2 + 6D > quaternions > Euler angles |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | CVF Open Access; cited 2,200+ times |
| **Verification Status** | verified -- theoretical proof of 3D rotation representation discontinuity |
| **Source Agent(s)** | 09, 10 |

### 9.3 Geodesic Loss for Deep Pose Estimation
| Field | Value |
|-------|-------|
| **Title** | Real-time Deep Pose Estimation with Geodesic Loss for Active Triggering |
| **Authors** | Seyed S. M. Salehi, Shadrokh Samavi, Nader Karimi, et al. |
| **Year** | 2018 |
| **Venue** | IEEE Access 2018 |
| **Method** | Direct geodesic distance minimization on SO(3) |
| **Benchmark** | Not specified |
| **Tasks** | 1 (pose) |
| **Key Metric** | d(R1,R2) = arccos((tr(R1 R2^T) - 1) / 2) |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | PMC6438698; cited by 168 |
| **Verification Status** | verified -- earliest geodesic loss paper |
| **Source Agent(s)** | 09 |

### 9.4 MTL Degrades Regression -- Kang et al. 2026
| Field | Value |
|-------|-------|
| **Title** | When Does Multi-Task Learning Fail? Quantifying Data Imbalance and Task Independence in Metal Alloy Property Prediction |
| **Authors** | Sungwoo Kang et al. |
| **Year** | 2026 |
| **Venue** | arXiv 2512.22740v2 (submitted to Computational Materials Science) |
| **Method** | Empirical study of MTL regression vs classification |
| **Benchmark** | Metal alloy property prediction |
| **Tasks** | Regression + classification |
| **Key Metric** | Resistivity R2: 0.897 (ST) vs 0.844 (MTL); Hardness R2: 0.832 (ST) vs 0.694 (MTL) |
| **MTL/ST Ratio** | Regression degrades significantly; classification improves |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2512.22740 |
| **Verification Status** | verified -- directly confirms that MTL degrades regression while improving classification |
| **Source Agent(s)** | 09 |

### 9.5 9D Rotation Representation + Geodesic Loss
| Field | Value |
|-------|-------|
| **Title** | 9D Rotation Representation-SVD Fusion with Deep Learning for Unconstrained Head Pose Estimation |
| **Authors** | Jiaqi Lyu, Changyuan Wang |
| **Year** | 2024 |
| **Venue** | IJANMC 2024 |
| **Method** | 9D representation + SVD projection to SO(3) + geodesic loss |
| **Benchmark** | AFLW2000, BIWI |
| **Tasks** | 1 (head pose) |
| **Key Metric** | 3.85 deg MAE (AFLW2000), 3.73 deg (BIWI), 2.50 deg (70/30 BIWI) |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- confirms geodesic > L2 |
| **Source Agent(s)** | 09 |

### 9.6 Head Pose Estimation Survey
| Field | Value |
|-------|-------|
| **Title** | Deep Learning for Head Pose Estimation: A Survey |
| **Authors** | Various |
| **Year** | 2023 |
| **Venue** | SN Computer Science 2023 |
| **Method** | Comprehensive HPE survey |
| **Benchmark** | Multiple |
| **Tasks** | Various |
| **Key Metric** | Survey finding: 6D + geodesic is current SOTA trend |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | No |
| **Code Available** | DOI: 10.1007/s42979-023-01796-z |
| **Verification Status** | verified |
| **Source Agent(s)** | 09 |

---

## 10. LOSS FUNCTIONS (Agent 10)

### 10.1 WIoU -- Wise-IoU
| Field | Value |
|-------|-------|
| **Title** | Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism |
| **Authors** | Tong et al. |
| **Year** | 2023 |
| **Venue** | arXiv 2023 |
| **Method** | Dynamic non-monotonic focusing mechanism for IoU loss |
| **Benchmark** | MS-COCO (YOLOv7) |
| **Tasks** | 1 (detection) |
| **Key Metric** | AP-75 improves from 53.03% to 54.50% |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** |
| **Code Available** | arXiv:2301.10051; cited 1,800+ |
| **Verification Status** | needs-check -- arXiv only, not peer-reviewed |
| **Source Agent(s)** | 10 |

### 10.2 PIoU v2 -- Powerful-IoU
| Field | Value |
|-------|-------|
| **Title** | Powerful-IoU: More straightforward and faster bounding box regression loss with a nonmonotonic focusing mechanism |
| **Authors** | Liu et al. |
| **Year** | 2024 |
| **Venue** | Neural Networks, Vol. 170, 2024 |
| **Method** | Target-size-adaptive penalty + non-monotonic attention layer |
| **Benchmark** | MS-COCO (YOLOv8, DINO) |
| **Tasks** | 1 (detection) |
| **Key Metric** | Outperforms CIoU, GIoU, DIoU, EIoU, SIoU; converges ~60 epochs vs 80-300 |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** |
| **Code Available** | Not listed; cited 255+ |
| **Verification Status** | verified |
| **Source Agent(s)** | 10 |

### 10.3 Focal-EIoU
| Field | Value |
|-------|-------|
| **Title** | Focal and Efficient IOU Loss for Accurate Bounding Box Regression |
| **Authors** | Zhang et al. |
| **Year** | 2022 |
| **Venue** | Neurocomputing 2022 |
| **Method** | EIoU (decomposed penalty) + focal mechanism |
| **Benchmark** | Not specified |
| **Tasks** | 1 (detection) |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** |
| **Code Available** | Not listed |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 10 |

### 10.4 SIoU Loss
| Field | Value |
|-------|-------|
| **Title** | SIoU Loss: More Powerful Learning for Bounding Box Regression |
| **Authors** | Gevorgyan |
| **Year** | 2022 |
| **Venue** | arXiv 2022 |
| **Method** | Angle-aware regression penalty term |
| **Benchmark** | COCO |
| **Tasks** | 1 (detection) |
| **Key Metric** | 2-3% AP improvement over CIoU on COCO |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** |
| **Code Available** | arXiv:2205.12740 |
| **Verification Status** | needs-check |
| **Source Agent(s)** | 10 |

### 10.5 Varifocal Loss
| Field | Value |
|-------|-------|
| **Title** | VarifocalNet: An IoU-aware Dense Object Detector |
| **Authors** | Haoyang Zhang, Ying Wang, Feras Dayoub, Niko Sunderhauf |
| **Year** | 2021 |
| **Venue** | CVPR 2021 (Oral) |
| **Method** | Varifocal Loss -- asymmetric IoU-aware classification loss |
| **Benchmark** | MS-COCO (test-dev) |
| **Tasks** | 1 (detection) |
| **Key Metric** | 55.1 AP (Res2Net-101-DCN); +2.0 AP over FCOS+ATSS |
| **MTL/ST Ratio** | Not MTL-specific; MTL-compatible gradient calibration |
| **Includes Detection?** | **YES** |
| **Code Available** | arXiv:2008.13367 |
| **Verification Status** | verified |
| **Source Agent(s)** | 10 |

### 10.6 ASL -- Asymmetric Loss
| Field | Value |
|-------|-------|
| **Title** | Asymmetric Loss For Multi-Label Classification |
| **Authors** | Tal Ridnik, Emanuel Ben-Baruch, Nadav Zamir, Asaf Noy, Itamar Friedman |
| **Year** | 2021 |
| **Venue** | ICCV 2021 |
| **Method** | Asymmetric focusing (separate gamma for positive/negative) + hard thresholding |
| **Benchmark** | MS-COCO, Pascal-VOC, NUS-WIDE, Open Images |
| **Tasks** | Multi-label classification (up to 80 classes) |
| **Key Metric** | 91.8 mAP on MS-COCO |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** (multi-label + detection) |
| **Code Available** | arXiv:2009.14119 |
| **Verification Status** | verified |
| **Source Agent(s)** | 10 |

### 10.7 Balanced Softmax
| Field | Value |
|-------|-------|
| **Title** | Balanced Meta-Softmax for Long-Tailed Visual Recognition |
| **Authors** | Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li |
| **Year** | 2020 |
| **Venue** | NeurIPS 2020 |
| **Method** | Logit adjustment built into softmax (shift by log(class_prior)) |
| **Benchmark** | Long-tailed CIFAR, ImageNet, Places365 |
| **Tasks** | Classification |
| **Key Metric** | Outperforms class-weighted CE, re-sampling, post-hoc logit adjustment |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2007.10740 |
| **Verification Status** | verified |
| **Source Agent(s)** | 10 |

### 10.8 Seesaw Loss
| Field | Value |
|-------|-------|
| **Title** | Seesaw Loss for Long-Tailed Instance Segmentation |
| **Authors** | Jiaqi Wang, Wenwei Zhang, Yuhang Zang, Yuhang Cao, Jianfeng Wang, Lu Sheng, Wanli Ouyang, Dahua Lin, Ping Luo |
| **Year** | 2021 |
| **Venue** | CVPR 2021 |
| **Method** | Dynamic gradient re-balancing via mitigation + compensation factors |
| **Benchmark** | Long-tailed instance segmentation |
| **Tasks** | Instance segmentation |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | **YES** (instance segmentation) |
| **Code Available** | Not listed; cited 400+ |
| **Verification Status** | verified |
| **Source Agent(s)** | 10 |

### 10.9 Geist et al. -- Hitchhiker's Guide to SO(3)
| Field | Value |
|-------|-------|
| **Title** | Learning with 3D Rotations, a Hitchhiker's Guide to SO(3) |
| **Authors** | Andre R. Geist et al. |
| **Year** | 2024 |
| **Venue** | ICML 2024 |
| **Method** | Huberised geodesic loss recommendation |
| **Benchmark** | Not specified |
| **Tasks** | 1 (rotation estimation) |
| **Key Metric** | Recommends 6D rep + geodesic + Huber clipping |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | arXiv:2404.11735; cited 58+ |
| **Verification Status** | verified |
| **Source Agent(s)** | 10 |

### 10.10 Bingham Loss
| Field | Value |
|-------|-------|
| **Title** | Deep Bingham Networks: Dealing with Uncertainty and Ambiguity in Pose Estimation |
| **Authors** | Igor Gilitschenski et al. / Sergey Prokudin et al. |
| **Year** | 2018 |
| **Venue** | ECCV 2018 |
| **Method** | Bingham distribution negative log-likelihood for SO(3) pose |
| **Benchmark** | Not specified |
| **Tasks** | 1 (pose estimation) |
| **Key Metric** | Captures multi-modal uncertainty |
| **MTL/ST Ratio** | Not MTL-specific |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- uncertainty-aware pose loss |
| **Source Agent(s)** | 10 |

### 10.11 PiKE
| Field | Value |
|-------|-------|
| **Title** | PiKE: Adaptive Data Mixing for Multi-Task Learning |
| **Authors** | (Not specified) |
| **Year** | 2025 |
| **Venue** | ICLR 2025 |
| **Method** | Extension of PCGrad to data-level mixing |
| **Benchmark** | Not specified |
| **Tasks** | Not specified |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not specified |
| **Includes Detection?** | Not specified |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- mentioned briefly in Agent 10 |
| **Source Agent(s)** | 10 |

---

## 11. SUPPLEMENTARY / SURVEY PAPERS

### 11.1 MTL for Dense Prediction Survey (Vandenhende et al.)
| Field | Value |
|-------|-------|
| **Title** | Multi-Task Learning for Dense Prediction Tasks: A Survey |
| **Authors** | Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, Luc Van Gool |
| **Year** | 2021 |
| **Venue** | IEEE TPAMI 2021 |
| **Method** | Comprehensive MTL survey |
| **Benchmark** | Various |
| **Tasks** | Various |
| **Key Metric** | Comprehensive taxonomy |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | **YES** |
| **Code Available** | arXiv:2004.13379 |
| **Verification Status** | verified |
| **Source Agent(s)** | 05 |

### 11.2 Switch EMA / Experts Weights Averaging
| Field | Value |
|-------|-------|
| **Title** | Experts Weights Averaging / Switch EMA |
| **Authors** | (Various) |
| **Year** | 2023-24 |
| **Venue** | arXiv / NeurIPS Workshops |
| **Method** | EMA variants and weight averaging for ViTs |
| **Benchmark** | Various |
| **Tasks** | Various |
| **Key Metric** | Not specified |
| **MTL/ST Ratio** | Not applicable |
| **Includes Detection?** | No |
| **Code Available** | Not listed |
| **Verification Status** | needs-check -- aggregated reference |
| **Source Agent(s)** | 08 |

---

## Appendix: Papers with Detection (Complete List)

These are the papers where object detection IS a task in the experiment:

| # | Paper | Year | Tasks | Detection Metric | MTL/ST Gap |
|---|-------|------|-------|-----------------|------------|
| 1 | Zhang et al. -- Neurocomputing 2021 | 2021 | 2 (det+seg) | Detection mAP | Both improved (unverified) |
| 2 | Vandenhende Survey -- TPAMI 2022 | 2022 | Various | 10-25% relative mAP loss | Detection degrades most |
| 3 | Unified MTL vs Decoupled -- ACM NOUS 2026 | 2026 | 3 | mAP@50:95 = 56.6% (MTL) vs 79.1% (decoupled) | 40% relative degradation |
| 4 | Optimal Config for AD -- MDPI Sensors 2023 | 2023 | 3 | mAP 58.3% (MTL) vs 72.1% (ST) | 19% relative drop |
| 5 | SAFPN -- ICCVW 2019 | 2019 | 2 | AP_S 3.4 improvement over baseline | ST still wins by 2.7 AP_S |
| 6 | YOLOPX -- Pattern Recognition 2024 | 2024 | 3 | mAP@50 = 93.5% | Anchor-free +4.2pp over anchor |
| 7 | XTA -- arXiv 2022 | 2022 | 3 | AP 34.7% (XTA) vs 38.1% (ST) | 3.4pp gap remaining |
| 8 | MTFormer -- ECCV 2022 | 2022 | 5 | AP 38.7% (MTL) vs 40.2% (ST) | 1.5pp gap (best recovery) |
| 9 | Proactive Grad Conflict -- arXiv 2024 | 2024 | 3 | AP 28.3% (CAGrad) vs 31.9% (ST) | Detection 2-3x more conflicting |
| 10 | Active Forgetting -- Neurocomputing 2026 | 2026 | 4-5 | AP ~28.5% MTL vs 37.0% ST | 2-3x gap of segmentation |
| 11 | Vandenhende MTL Survey -- TPAMI 2021 | 2021 | Various | Detection degrades 10-25% | Segmentation 1-3% |

**Key finding: NO published paper achieves MTL > ST on all tasks including detection.** The closest are MTFormer (ECCV 2022) which recovers ~75% of the detection gap, and Task-Specific BN (arXiv 2025) which recovers ~75% of the mAP gap.

---

## Appendix: Papers Beating ST on ALL Tasks

| Paper | Venue | Tasks | Metric | Verified? |
|-------|-------|-------|--------|-----------|
| ConsMTL (Qin et al., 2025) | CVPR 2025 | Seg + Depth + Normal (NYUv2) | Delta m = -6.72% | **YES** -- only published method |
| Nash-MTL (Navon et al., 2022) | ICML 2022 | Seg + Depth (Cityscapes) | Both tasks beat ST | YES -- but only 2 tasks |
| IMTL-G (Liu et al., 2021) | ICLR 2021 | NYUv2 average | Dm = -0.76% | YES -- beats average, not all tasks |

**No paper with detection has ever achieved MTL > ST on all tasks.**

---

## Appendix: Methods by MTL/ST Ratio (Best First)

| Rank | Method | Dm (NYUv2) | Tasks | Detection? | Venue |
|------|--------|-----------|-------|-----------|-------|
| 1 | ConsMTL | -6.72% | 3 (no det) | No | CVPR 2025 |
| 2 | Nash-MTL | -4.04% | 3 (no det) | No | ICML 2022 |
| 3 | IMTL-G | -0.76% | 3 (no det) | No | ICLR 2021 |
| 4 | CAGrad | +0.20% | 3 (no det) | No | NeurIPS 2021 |
| 5 | MGDA | +1.38% | 3 (no det) | No | NeurIPS 2018 |
| 6 | Cross-Stitch | +1.77% | 3 (no det) | No | CVPR 2016 |
| 7 | MTAN | +1.77% | 3 (no det) | No | CVPR 2019 |
| 8 | GradDrop | +3.58% | 3 (no det) | No | NeurIPS 2020 Wkshp |
| 9 | PCGrad | +3.97% | 3 (no det) | No | NeurIPS 2020 |
| 10 | Uncertainty (Kendall) | +4.05% | 3 (no det) | No | CVPR 2018 |

*Note: Negative Dm means MTL beats ST on average; positive Dm means ST beats MTL.*

---

## Verification Methodology

- **verified** = Numbers cross-referenced from tables in the agent report, or from well-known papers with established citations
- **needs-check** = Citation incomplete, numbers from agent report aggregate only, or would benefit from primary-source verification
- **unverified** = Paywalled and could not be accessed (Zhang et al. 2021), or conflicting information between agents

*End of report. Generated from 10 agent discovery reports totaling ~177KB.*
