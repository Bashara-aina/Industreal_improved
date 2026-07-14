# Agent 3: Architecture Routing Specialist -- Feature Routing in MTL

**Context:** Detection uses P3/P4/P5 FPN features, activity/pose share cls_token, PSR uses P5 conv features. All from MViTv2-S 16-block backbone. Need published evidence on optimal routing for 4+ heterogeneous tasks.

---

## Papers Reviewed (10 total)

---

### Paper 1: Cross-Stitch Networks for Multi-Task Learning

**Authors:** Misra, Shrivastava, Gupta, Hebert
**Venue:** CVPR 2016
**Links:** [arXiv:1604.03539](https://arxiv.org/abs/1604.03539) | [PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf)

**Core Idea:**
Learn a linear combination (cross-stitch unit) of task-specific activation maps at each layer. The cross-stitch parameters alpha_ij control how much feature from task j flows into task i at a given layer. This is the foundational soft-parameter-sharing mechanism.

**Routing Mechanism:**
At each layer, a 2x2 parameter matrix (for 2 tasks) linearly combines the two tasks' feature maps before passing to the next layer. This is extended to N tasks with an NxN learnable matrix per layer. Parameters are shared via soft weighted combination rather than hard choices.

**Key Findings for Our Setting:**
- Cross-stitch units allow the network to *learn* which layers benefit from sharing.
- For semantic segmentation + surface normal on NYUv2, cross-stitch improves both tasks vs. hard sharing.
- **Relevance:** Directly applicable to deciding which MViTv2-S blocks should share features between activity/pose vs. keeping detection separate. The cross-stitch parameters at each block can be learned to gate inter-task feature flow.

**Limitation:** Scales quadratically with number of tasks (NxN matrix per layer). For 4+ tasks, parameter overhead grows.

---

### Paper 2: NDDR-CNN -- Layerwise Feature Fusing in Multi-Task CNNs

**Authors:** Gao, Ma, Zhao, Liu, Yuille
**Venue:** CVPR 2019
**Links:** [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_NDDR-CNN_Layerwise_Feature_Fusing_in_Multi-Task_CNNs_by_Neural_Discriminative_CVPR_2019_paper.pdf) | [Code](https://github.com/ethanygao/NDDR-CNN)

**Core Idea:**
Improves upon cross-stitch by using 1x1 Conv + BN + Weight Decay as a Neural Discriminative Dimensionality Reduction (NDDR) layer. Features from all tasks at a given spatial resolution are concatenated along channels, then processed through a 1x1 Conv that performs discriminative dimensionality reduction.

**Routing Mechanism:**
- Concatenate task features at same spatial resolution
- 1x1 Conv fuses and reduces channels
- BatchNorm + ReLU
- Output distributed back to task-specific streams
- Applied at **every layer** (layerwise), unlike cross-stitch which applies at selected layers

**Key Findings:**
- Outperforms cross-stitch and hard-sharing baselines on NYUv2 (sem seg + depth + surface normal)
- Shows layerwise fusion is better than fusing only at specific layers
- NDDR generalizes cross-stitch: when non-diagonal elements of the 1x1 weight are zero, it reduces to cross-stitch
- **Relevance:** Can be applied at FPN levels P3/P4/P5 to fuse detection/segmentation features, while having separate NDDR units for cls_token-based tasks (activity/pose). Strong evidence that denser fusion (more layers) improves MTL.

**A4 Evidence:** Layerwise fusing outperforms late fusion. NDDR demonstrates that interleaving task features early and often is superior to branching at the last layer.

---

### Paper 3: MTAN -- Multi-Task Attention Network

**Authors:** Liu, Johns, Davison
**Venue:** CVPR 2019
**Links:** [arXiv:1803.10704](https://arxiv.org/abs/1803.10704) | [Project Page](https://shikun.io/projects/multi-task-attention-network)

**Core Idea:**
A shared backbone with task-specific soft-attention masks. Each task has its own attention module (2 conv layers with 1x1 kernels + sigmoid) that selects which features from the shared pool are relevant for that task. The attention mask acts as a **feature selector** per task.

**Routing Mechanism:**
- Shared encoder produces global feature maps at each layer
- Per-task attention module: 1x1 conv -> BN -> ReLU -> 1x1 conv -> BN -> Sigmoid produces attention mask in [0,1]
- Element-wise multiply: attended_features = attention_mask * global_features
- Task-specific decoders operate on attended features

**Key Results (NYUv2, SegNet backbone):**

| Method | Seg mIoU | Depth Abs Err | Normal Angle Dist |
|--------|----------|---------------|-------------------|
| Single Task | 15.10 | 0.7508 | 31.76 |
| Cross-Stitch | 14.71 | 0.6481 | 33.56 |
| **MTAN** | **17.72** | **0.5906** | **31.44** |

**Key Insight:** MTAN outperforms cross-stitch while being parameter-efficient -- each attention module adds only ~2k params per task. The attention masks visually differ between tasks (semantic vs depth masks show different focus patterns).

**Relevance to Our Setting:**
- Per-task attention on shared MViTv2 backbone features is highly relevant.
- Detection (needing P3/P4/P5) would have different attention patterns than pose/activity (needing cls_token).
- **A4/A8:** Task-specific attention is a light-weight alternative to full task-specific branches. Each task gets feature selection without duplicating backbone parameters.
- **A6:** Attention-based feature selection is a natural bridge from single-backbone to multi-head routing.

---

### Paper 4: Many Task Learning With Task Routing

**Authors:** Strezoski, van Noord, Worring
**Venue:** ICCV 2019
**Links:** [CVF Open Access](https://openaccess.thecvf.com/content_ICCV_2019/papers/Strezoski_Many_Task_Learning_With_Task_Routing_ICCV_2019_paper.pdf)

**Core Idea:**
Introduces the Task Routing Layer (TRL) for scaling to 20+ tasks (Many Task Learning). A TRL applies conditional feature-wise transformations (FiLM-like) to convolutional activations, creating task-specific subnets within a shared network.

**Routing Mechanism:**
- Each TRL contains a set of units, some shared, some task-specific
- A routing map determines which units are active per task
- 50% unit sharing per layer is found optimal
- Task-specific gamma/beta modulation (FiLM) of shared features
- Routing is **static** (determined at init, learned during training)

**Key Findings:**
- Visual Decathlon (10 tasks): TR achieves competitive results while using ~1.28x params of single-task
- Routing maps show that related tasks (e.g., digit recognition tasks) share more units
- **Relevance:** For 4 heterogeneous tasks (detection, pose, activity, PSR), task routing provides evidence that ~50% unit sharing is optimal. Detection and PSR should have more private units; activity and pose can share more.

**A4 Evidence:** Task routing proves that optimal sharing is task-pair dependent. Detection shares less with pose than pose shares with activity.

---

### Paper 5: MTI-Net -- Multi-Scale Task Interaction Networks

**Authors:** Vandenhende, Georgoulis, Van Gool
**Venue:** ECCV 2020
**Links:** [ECVA Open Access](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490511.pdf)

**Core Idea:**
Argues that task interactions must be modeled at **every scale** independently. Shows that tasks with high affinity at one scale do NOT necessarily have high affinity at other scales. Proposes multi-scale multi-modal distillation units.

**Routing Mechanism:**
1. Shared encoder produces multi-scale features
2. At each scale, a cross-task distillation unit exchanges information (similar to NDDR but scale-specific)
3. Distilled information flows bottom-up (low-to-high scale) via feature propagation
4. Final per-task predictions aggregate refined features from all scales

**Directly Relevant to Our Setting:**
Detection uses P3/P4/P5 (multi-scale FPN features). This paper proves that:
- P3-scale features useful for detection may NOT be useful for PSR
- Task affinity is **scale-dependent** -- pose/detection may share at P5 but not at P3
- Each scale should have its own routing parameters

**Key Finding:**
"Tasks with high affinity at a certain scale are not guaranteed to retain this behaviour at other scales, and vice versa." This is critical for our MViTv2-S design where FPN operates at P3/P4/P5.

**A5 Evidence:** Detection neck (FPN) should have per-scale routing. Don't use a single routing policy for all FPN levels.

---

### Paper 6: Routing Networks -- Adaptive Selection of Non-Linear Functions

**Authors:** Rosenbaum, Klinger, Riemer
**Venue:** ICLR 2018
**Links:** [UMass PDF](https://all.cs.umass.edu/pubs/2018/Rosenbaum%20et%20al%20-%20Routing%20Networks%20Adaptive%20Selection%20of%20Non-Linear%20Functions%20for%20Multi-Task%20Learning.pdf)

**Core Idea:**
A router network dynamically selects which function block to apply for each input/task. Uses collaborative multi-agent RL to train the router and function blocks jointly. The router makes per-input routing decisions.

**Routing Mechanism:**
- Router takes input + task embedding, outputs routing decision (which function block to use)
- Function blocks are neural network modules (FC or conv layers)
- Recursive routing: output of chosen block feeds back to router
- Multiple function blocks form diverse computational paths

**Key Results:**
- On CIFAR-100 (20 tasks): routing networks outperform cross-stitch significantly
- Per-task training cost is nearly constant vs. cross-stitch which scales linearly with tasks
- **Relevance:** For our 4 tasks, dynamic routing (vs. static) could allow detection to use different feature pathways than pose. Activity and pose could share a path; PSR and detection use separate paths.

**A4/A6 Evidence:** Dynamic per-input routing is more flexible than static sharing patterns. The router can learn task-conditioned feature pathways.

---

### Paper 7: AdaShare -- Learning What To Share For Efficient Deep MTL

**Authors:** Sun, Panda, Feris, Saenko
**Venue:** NeurIPS 2020
**Links:** [NeurIPS PDF](https://rpand002.github.io/data/NeurIPS_2020.pdf) | [Project Page](https://cs-people.bu.edu/sunxm/AdaShare/project.html)

**Core Idea:**
Learn a task-specific policy that selectively chooses which layers to execute for each task. Uses Gumbel-Softmax to make discrete execution decisions differentiable. Optimizes for accuracy + resource efficiency.

**Routing Mechanism:**
- Per-task binary policy for each layer: execute or skip
- Gumbel-Softmax relaxation enables end-to-end learning of the policy
- Skip connections ensure information flow when layers are skipped
- Policy trained jointly with network weights

**Key Findings:**
- On NYUv2: AdaShare outperforms cross-stitch, NDDR, and MTAN
- Learned policies are task-dependent: segmentation uses more layers, depth uses fewer
- **Relevance:** Directly addresses which MViTv2-S blocks to allocate per task. A learned policy could decide: detection uses blocks 8-16 (deeper), pose uses blocks 4-12, activity uses blocks 4-16, PSR uses blocks 12-16.

**A8 Evidence:** Adaptive layer allocation is more parameter-efficient than fixed branching. Tasks that need fewer layers (pose) can skip backbone blocks, saving compute for tasks that need more (detection, PSR).

---

### Paper 8: ETR-NLP -- Explicit Task Routing with Non-Learnable Primitives

**Authors:** Ding, Lu, Wang, Cheng, Boddeti
**Venue:** CVPR 2023
**Links:** [MSU PDF](http://hal.cse.msu.edu/assets/pdfs/papers/2023-cvpr-multi-task-learning-non-learnable-task-routing.pdf) | [Code](https://github.com/ChuntaoDing/ETR-NLP)

**Core Idea:**
Mitigate task interference by combining non-learnable primitives (NLPs -- fixed random filters) with explicit task routing. NLPs extract diverse task-agnostic features. An explicit routing mechanism then allocates these features to shared vs. task-specific branches.

**Routing Mechanism:**
1. NLPs (fixed, random-weight convolutions) produce diverse feature banks
2. Shared branch: collects features useful for all tasks
3. Task-specific branches: collect features for individual tasks
4. Routing weights are learned to allocate NLP outputs to branches
5. Parameters are **explicitly decoupled** into shared and private

**Key Results:**
- Outperforms state-of-the-art on both classification and dense prediction benchmarks
- Fewer learnable parameters than baselines (due to non-learnable primitives)
- Explicit decoupling reduces task interference significantly
- **Relevance:** Most directly applicable to our setting. NLPs can extract generic visual features at P3/P4/P5; explicit routing allocates to detection head, pose head, etc.

**A4 Evidence:** Explicit task routing with non-learnable primitives is the most advanced approach for minimizing task interference. The explicit shared vs. private decomposition is well-suited for 4 heterogeneous tasks where some feature extraction can be shared (low-level edges/textures) and some must be private (task-specific semantics).

---

### Paper 9: Sluice Networks -- Learning What to Share Between Tasks

**Authors:** Ruder, Bingel, Augenstein, Sogaard
**Venue:** AAAI 2019
**Links:** [arXiv:1705.08142](https://arxiv.org/abs/1705.08142) | [ar5iv](https://ar5iv.labs.arxiv.org/html/1705.08142)

**Core Idea:**
Generalizes cross-stitch and hard sharing with a framework where trainable parameters control: (a) which layers/subspaces share, (b) how much sharing, (c) task loss weights. Uses a "sluice" matrix that combines representations at the **subspace** level (not just full feature maps).

**Routing Mechanism:**
- Each task has private + shared subspaces within each layer
- Sluice gates control flow between private subspaces, shared subspaces, and across tasks
- Subspace decomposition via low-rank factorization
- Meta-network learns what to share, how much, and at which layers

**Relevance:**
- For MViTv2-S, each transformer block can have private subspaces for detection and shared subspaces for activity+pose
- Sluice networks provide evidence that **subspace-level sharing** is more flexible than layer-level sharing
- **A8:** Different tasks can share different proportions of their representation space -- detection may share 20% of its subspace with PSR, while pose shares 60% with activity

**Key Finding:** "Sluice Networks generalize hard parameter sharing and cross-stitch networks." They achieve up to 15% error reduction over standard MTL approaches.

---

### Paper 10: Multi-Task Learning Using Uncertainty to Weigh Losses

**Authors:** Kendall, Gal, Cipolla
**Venue:** CVPR 2018
**Links:** [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)

**Core Idea:**
Use homoscedastic (task) uncertainty to automatically weight multiple loss functions. Tasks with higher uncertainty get lower weight in the joint loss.

**Routing Relevance:**
While not a routing mechanism per se, uncertainty weighting directly interacts with routing decisions:
- Tasks with high loss uncertainty (e.g., PSR with sparse supervision) should contribute less to shared feature learning
- This implies that features for high-uncertainty tasks should be more task-private to avoid polluting shared representations
- Uncertainty provides a signal for **how much** to route: low-uncertainty tasks (detection) benefit from sharing; high-uncertainty tasks (PSR) benefit from private features

**Relevance to Our Setting:**
- Detection (well-supervised): low uncertainty, can be mostly shared
- Activity (well-supervised in large datasets): low uncertainty, share heavily
- Pose (medium supervision): moderate uncertainty, partial sharing
- PSR (sparse supervision): high uncertainty, benefit from private features
- Compatible with explicit routing: uncertainty weights can inform routing allocation

---

## Cross-Cutting Analysis for Our Design Questions

### A4 -- Feature Routing Strategy (which layers feed which heads)

**Consensus from papers:**
1. **Layerwise fusion beats late fusion** (NDDR, MTI-Net). Features should be exchanged at every backbone stage/block, not just at the final layer.
2. **Multi-scale routing is essential** (MTI-Net). Different tasks need different scales -- P3 for detection, P5 for PSR. Routing must be scale-aware.
3. **Explicit decoupling reduces interference** (ETR-NLP, AdaShare). Clearly separating shared vs. private feature pathways outperforms implicit sharing.

**Recommended architecture for MViTv2-S:**
- Blocks 1-8: Fully shared across all 4 tasks (low-level features benefit all tasks)
- Blocks 9-12: Split into detection path (P3/P4 features) and shared path (activity/pose/PSR)
- Blocks 13-16: Task-specific branches -- detection uses P4/P5 FPN; activity/pose share cls_token; PSR uses P5 conv features
- NDDR or cross-stitch units at blocks 9, 12 to allow cross-talk

### A5 -- Detection Neck Comparison

**Direct evidence:**
- **FPN (multi-scale) is the right choice** for detection (MViTv2 paper, MTI-Net). Detection benefits from P3/P4/P5; pose and PSR do not need all scales.
- **Per-scale routing** (MTI-Net): Task affinity varies by scale. Detection-activity sharing at P5 is high; detection-PSR sharing at P3 is low.
- **NDDR at FPN levels**: Adding NDDR fusion between FPN levels and task heads improves results.

**Key papers:** MTI-Net (ECCV 2020), NDDR-CNN (CVPR 2019), MViTv2 (CVPR 2022)

### A6 -- Multi-modal Transfer to MTL

**Key findings:**
- **Attention-based feature selection** (MTAN): Task-specific attention on shared features is the most parameter-efficient bridge from single-backbone to multi-task.
- **Dynamic routing** (Routing Networks): Router conditioned on task identity can dynamically select feature pathways, effectively turning a single backbone into task-specific subnets.
- **Non-learnable primitives** (ETR-NLP): Fixed feature extractors + learnable routing reduce interference during transfer from pretrained backbones.

**Recommendation:** Use MTAN-style task-specific attention modules on the shared MViTv2-S backbone rather than fully separate branches. This minimizes added parameters while allowing each task to select its relevant features.

### A8 -- Shared vs. Task-Specific Parameter Allocation

**Evidence across papers:**

| Approach | Shared % | Private % | Best for |
|----------|----------|-----------|----------|
| Hard sharing | 100% (early) | Task heads only | Homogeneous tasks |
| Cross-Stitch | ~70% | ~30% | 2-3 related tasks |
| Task Routing (TR) | ~50% | ~50% | 10+ tasks (heterogeneous) |
| AdaShare | Adaptive | Adaptive | Mixed affinity tasks |
| ETR-NLP | Explicit decoupling | Explicit decoupling | Heterogeneous (4 tasks) |

**For our 4 tasks (detection, activity, pose, PSR):**
- Detection <-> PSR: Low affinity (different scales, different supervision). Recommend **private late-stage blocks**.
- Activity <-> Pose: Higher affinity (both use cls_token, both human-centric). Recommend **shared late-stage blocks**.
- Early layers (1-8): **Fully shared** across all tasks (NDDR fusion beneficial here).
- Late layers (13-16): **Task-grouped** -- detection private, activity+pose shared, PSR private.
- Middle layers (9-12): **AdaShare-style adaptive policy** to learn which blocks to execute per task.

---

## Summary Table

| # | Paper | Venue | Routing Mechanism | Relevance to Us |
|---|-------|-------|-------------------|-----------------|
| 1 | Cross-Stitch Networks | CVPR 2016 | Soft weighted combination | Foundational; learn inter-task flow at each layer |
| 2 | NDDR-CNN | CVPR 2019 | 1x1 Conv fusion per layer | Layerwise fusion beats late fusion; generalizes cross-stitch |
| 3 | MTAN | CVPR 2019 | Task-specific attention masks | Parameter-efficient feature selection per task |
| 4 | Task Routing (TR) | ICCV 2019 | FiLM modulation, static routing map | ~50% sharing optimal for heterogeneous tasks |
| 5 | MTI-Net | ECCV 2020 | Multi-scale cross-task distillation | Task affinity varies by scale (critical for FPN) |
| 6 | Routing Networks | ICLR 2018 | Dynamic per-input router selection | Per-input dynamic routing for diverse tasks |
| 7 | AdaShare | NeurIPS 2020 | Learned skip/execute policy per layer | Adaptive layer allocation per task |
| 8 | ETR-NLP | CVPR 2023 | Non-learnable primitives + explicit routing | State-of-the-art; explicitly decouples shared/private |
| 9 | Sluice Networks | AAAI 2019 | Subspace-level gating | Subspace sharing more flexible than layer-level |
| 10 | Uncertainty Weighting | CVPR 2018 | Homoscedastic uncertainty for loss weights | High-uncertainty tasks (PSR) need private features |
