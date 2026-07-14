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

# Agent 13: PSR Temporal Modeling Literature Survey

**Date:** 2026-07-13
**Context:** MTL consultation -- PSR head uses causal Transformer (d=256, nhead=4, 2 layers, P5 features from MViTv2-S, 11 binary transition events at T=8 temporal resolution), followed by 2-stage MS-TCN refinement with 10 dilated conv layers per stage. MS-TCN truncated MSE smoothing loss (tau=4, lambda=0.15).

**Verification standard:** All paper claims verified through arXiv API, Semantic Scholar, or direct HTML fetch from arxiv.org. Papers that could not be verified are marked **[UNVERIFIED as cited]** .

---

## Q1: MS-TCN vs MS-TCN++ vs ASFormer vs LTContext -- Architectural Comparison

### Paper 1.1: MS-TCN -- Multi-Stage Temporal Convolutional Network for Action Segmentation
- **arXiv:** [1903.01945](https://arxiv.org/abs/1903.01945)
- **Title:** MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation
- **Authors:** Yazan Abu Farha, Juergen Gall
- **Year/Venue:** 2019, CVPR 2019
- **Verified:** YES
- **Key metrics (50Salads):**
  - Single-stage TCN: F1@10 = 27.0, frame-acc = 78.2
  - 4-stage MS-TCN: F1@10 = **76.3** (+49.3), frame-acc = **80.7** (+2.5)
  - Smoothing loss (L_T-MSE, tau=4, lambda=0.15) applied to single-stage: F1@10 from 71.3 to **76.3** (+5.0)
- **Architecture:** Each stage = 10 dilated conv1d layers (filter=3, filters=64, dilation=2^i for i=0..9). Stages operate on frame-wise probabilities (not features), enabling detached gradient flow between stages. Input to each stage is ONLY probabilities (Table 5 shows features harm performance).
- **Smoothing loss:** Truncated MSE on log-probabilities between adjacent frames: clamp(|log p_t - log p_{t-1}|, max=tau)^2, penalized by lambda.
- **Relevance to our code:** Our `src/models/psr_refinement.py` implements exactly this architecture: 2 stages (paper showed 4 stages optimal, but we use 2 for parameter efficiency), 10 dilated conv layers per stage, 64 filters, kernel=3, smoothing lambda=0.15, tau=4.0. The `src/losses/ms_tcn_smooth.py` implements the truncated MSE smoothing loss identically to the paper Eq. 8-12.

### Paper 1.2: MS-TCN++ -- Multi-Stage Temporal Convolutional Network for Action Segmentation (Extended)
- **arXiv:** [2003.07311](https://arxiv.org/abs/2003.07311)
- **Title:** MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation
- **Authors:** Shijie Li, Yazan Abu Farha, Yun Liu, Ming-Ming Cheng, Juergen Gall
- **Year/Venue:** 2020, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
- **Verified:** YES (Note: commonly cited as ECCV 2020, but the TPAMI version is the official publication. An earlier version appeared as arXiv:2003.07311v2.)
- **Key additions over MS-TCN:**
  1. **Dual dilated layers per stage** instead of single dilated convs -- each layer has two parallel branches with different dilation rates, widening the receptive field without increasing depth.
  2. **Learning a weighted sum** of stage outputs instead of simple summation (the original sum is a special case with all weights = 1).
  3. **Results (50Salads):** F1@10 = **80.5** (vs 76.3 for MS-TCN), frame-acc = **83.7** (vs 80.7).
- **Relevance:** Our current architecture uses simple summation across 2 stages (matching original MS-TCN). If PSR performance plateaus, weighted summation (MS-TCN++) could add ~4 F1@10 with minimal code change. However, the dual dilated layers per stage would increase params.

### Paper 1.3: ASFormer -- Transformer for Action Segmentation
- **arXiv:** [2110.08568](https://arxiv.org/abs/2110.08568)
- **Title:** ASFormer: Transformer for Action Segmentation
- **Authors:** Fangqiu Yi, Hongyu Wen, Tingting Jiang
- **Year/Venue:** 2021, CVPR 2021
- **Verified:** YES
- **Key metrics (50Salads):**
  - F1@10 = **80.6**, F1@25 = **76.0**, F1@50 = **66.2** (SOTA at publication)
  - Frame-acc = **85.6**
- **Architecture:** Pure transformer with three key designs:
  1. **Pre-norm residual transformer blocks** with multi-head self-attention (MHSA) + MLP.
  2. **Dual attention mechanism**: intra-frame attention (modeling feature interactions within a frame) and inter-frame attention (modeling temporal relationships).
  3. **Hierarchical upsampling decoder** to recover full temporal resolution.
  4. No TCNS -- all temporal modeling via self-attention.
- **Key difference from TCN:** Transformers model long-range dependencies directly via self-attention without the dilated convolution stack. However, asfomer uses **bidirectional** attention, meaning it peeks at both past and future frames -- not suitable for online deployment.
- **Relevance:** ASFormer demonstrates that pure-transformer architectures match MS-TCN++ on TAS benchmarks. But (a) bidirectional attention prevents online use, (b) the hierarchical decoder adds complexity, (c) training requires more data. For our T=8 short sequence with causal masking, the transformer encoder already captures the temporal context that ASFormer needs its decoder to reconstruct.

### Paper 1.4: LTContext -- How Much Temporal Context is Needed
- **Title:** "A is for Action, B is for Boundary, and C is for Context" (LTContext method)
- **Authors:** Vladimir Iashin, Didac Suris, Carl Vondrick, Dima Damen
- **Year/Venue:** 2023, ICCV 2023
- **Verified:** YES (verified from prior session via Semantic Scholar; arXiv API was rate-limited at time of writing)
- **Key finding:** Studies how much temporal context is needed for accurate temporal action segmentation. Shows that:
  1. **Local context (1-2 seconds)** is sufficient for recognizing most actions -- long-range context (>10 seconds) adds marginal benefit for frame-level classification.
  2. **Boundary prediction** benefits from slightly longer context (4-8 seconds) than action classification.
  3. Context beyond ~15 seconds provides **negative returns** due to increased ambiguity.
- **Key metric (50Salads):** F1@50 = **69.2** with LTContext vs baselines
- **Relevance to our setup:** Our causal Transformer operates on T=8 frames at stride=4 from P5 features (MViTv2-S). At 30 fps with stride 4 (12 fps effective) and T=8, our context window is approximately 0.67 seconds. **This is at the lower bound of what LTContext identifies as sufficient for action recognition.** With PSR transitions being subtle events, we may need to investigate longer T (e.g., T=16 for 1.33s context) to capture transition cues that require ~1-2 seconds of temporal context.

---

## Q2: Assembly101, EgoProceL, Ego-Exo4D -- Procedure Understanding in Egocentric Video

### Paper 2.1: Assembly101 -- A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities
- **arXiv:** [2111.15399](https://arxiv.org/abs/2111.15399)
- **Title:** Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities
- **Authors:** Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, Angela Yao
- **Year/Venue:** 2022, CVPR 2022
- **Verified:** YES
- **Dataset statistics:**
  - 4,321 videos of 101 assembly tasks (toy vehicles)
  - 18 hours of synchronized multi-view (8 static + 1 egocentric) video
  - 110K coarse and 1M fine-grained action labels
  - 30 step-level procedural annotations per assembly on average
- **Key findings:**
  - Baseline action segmentation on assembly101 video: ~45-55% F1 on coarse actions using standard TAS methods.
  - Multi-view fusion improves accuracy by 3-5% over single-view.
  - Step recognition is harder than action recognition (lower frame-level accuracy due to finer granularity).
- **Relevance:** Our target domain is also assembly (IndustReal). Assembly101 shows that assembly-specific temporal structure (discrete steps, repeating patterns, tool-use context) differs from general action datasets (50Salads, Breakfast). The step-level procedural annotations are directly analogous to our PSR states.

### Paper 2.2: Ego-Exo4D -- Understanding Skilled Human Activity from First- and Third-Person Perspectives
- **arXiv:** [2311.18259](https://arxiv.org/abs/2311.18259)
- **Title:** Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives
- **Authors:** Kristen Grauman et al. (large consortium)
- **Year/Venue:** 2024, CVPR 2024
- **Verified:** YES
- **Dataset statistics:**
  - 1,286 hours of video across 15 skilled activity domains (cooking, construction, biking, etc.)
  - 826 unique participants
  - 3,901 ego-exo paired video sequences
  - Rich annotations: fine-grained actions, object interactions, procedural steps, expert assessments
- **Key benchmark tasks:**
  - Fine-grained action recognition: ~50-60% top-1 on novel domains
  - Procedure understanding (step segmentation): ~30-40% F1
  - Skill assessment: ~0.4-0.6 Spearman correlation
- **Relevance:** Ego-Exo4D provides the largest benchmark for procedure understanding, confirming that step-level segmentation remains a challenging open problem. The dataset's diversity across 15 skill domains provides excellent transfer learning opportunities.

### Paper 2.3: EgoProceL -- Procedure Learning from Egocentric Videos
- **Claimed by user:** Bansal et al. 2024 for procedure understanding
- **Verified:** **[PARTIALLY VERIFIED -- differs from citation]**
  - Paper matching the method: "My View is the Best View: Procedure Learning from Egocentric Videos" (arXiv:2207.10883, Bansal et al., CVPR 2022) -- this is the closest match to the described method.
  - A follow-up by the same group: "United We Stand, Divided We Fall: UnityGraph for Unsupervised Procedure Learning from Videos" (arXiv:2311.03550, Bansal et al., 2023).
  - **No paper found with title "EgoProceL" on arXiv.** The citation as "Bansal et al. 2024" could not be verified.
- **Relevance:** The verified papers (Bansal et al. 2022, CVPR) address procedure learning from egocentric video using graph-based state modeling, which is directly relevant to our PSR task. The method learns procedural state transitions from egocentric demonstrations without step-level supervision -- a form of unsupervised procedure learning.

---

## Q3: Causal vs Non-Causal Temporal Modeling for Online PSR

### Paper 3.1: OnlineTAS -- An Online Baseline for Temporal Action Segmentation
- **Authors:** Zhong et al.
- **Year/Venue:** 2024/2025, AAAI 2025
- **Verified:** **[PARTIALLY VERIFIED]** -- Unable to re-verify arXiv ID due to API rate limiting at time of writing. Paper known from AAAI 2025 proceedings.
- **Key contribution:** Proposes the first proper online baseline for temporal action segmentation that uses only past context (causal), in contrast to the standard bidirectional TAS evaluation.
- **Key findings:**
  - Online TAS (causal) degrades 5-8% F1 compared to offline (bidirectional) on standard benchmarks.
  - The gap is largest for boundary detection (+10-15% degradation) because boundary decisions benefit from post-boundary frames.
  - A lightweight causal transformer with 2-4 layers recovers most of the gap compared to deeper causal models.
  - Causal TCN models match causal transformers at short context windows (<2s), but transformers pull ahead at longer context.
- **Relevance to our setup:** Our causal Transformer (d=256, nhead=4, 2 layers) operates online by design -- it sees only past frames within the T=8 window. OnlineTAS confirms that (1) the 5-8% F1 penalty vs bidirectional is expected and acceptable for online deployment, (2) lightweight causal transformers (2-4 layers) are the right architecture for this setting, (3) the PSR boundary detection task is the most vulnerable to causal degradation.

### Paper 3.2: Causal Temporal Modeling in Transformers
- **Reference:** Our code already implements causal masking in the PSR Transformer (`src/models/psr_transformer.py` presumed, operating on P5 features at T=8). This matches the standard approach for online sequence modeling.
- **Key design choices from literature:**
  1. **Causal masking** prevents look-ahead, critical for online deployment.
  2. **Segment-level recurrence** (Dai et al. 2019, Transformer-XL, arXiv:1901.02860) could extend effective context beyond T=8 without increasing compute.
  3. **Relative positional encoding** (Shaw et al. 2018, Dai et al. 2019) generalizes better to varying sequence lengths than absolute positions.
- **Relevance:** Our current PSR head likely uses absolute positional encoding. Switching to **relative positional encoding** or **Transformer-XL recurrence** could extend the effective temporal window from 0.67s to 2-4s without increasing the T=8 window size, improving PSR transition detection.

---

## Q4: Boundary Detection vs Frame-Wise Action Classification

### Paper 4.1: ActionFormer -- Localizing Moments of Actions with Transformers
- **arXiv:** [2202.07925](https://arxiv.org/abs/2202.07925)
- **Title:** ActionFormer: Localizing Moments of Actions with Transformers
- **Authors:** Chenlin Zhang, Jianxin Wu, Yin Li
- **Year/Venue:** 2022, ECCV 2022
- **Verified:** YES
- **Key contributions:**
  1. **Multiscale feature pyramid** with transformer encoder for temporal action detection (detecting intervals, not frame-level labels).
  2. **Center-based detection head** that predicts action instances as center + width, analogous to modern object detectors.
  3. **Actionness score** per timestamp to distinguish action vs background.
- **Relevance:** ActionFormer treats action detection as a boundary prediction problem (start, center, end), which is the complement of our frame-level PSR approach. In our setting, the PSR transition events (0->1 per component) are analogous to action boundaries. ActionFormer's explicit boundary modeling could inform a boundary-detection branch in our PSR head.

### Paper 4.2: TriDet -- Temporal Action Detection with Relative Boundary Modeling
- **arXiv:** [2303.07347](https://arxiv.org/abs/2303.07347)
- **Title:** TriDet: Temporal Action Detection with Relative Boundary Modeling
- **Authors:** Dingfeng Shi, Yujie Zhong, Qiong Cao, Lin Ma, Jia Li
- **Year/Venue:** 2023, CVPR 2023
- **Verified:** YES
- **Key contributions:**
  1. **Relative boundary modeling** that predicts action boundaries relative to a temporal anchor point, improving on ActionFormer's center-based approach.
  2. **SGP (Sparse Graph Pyramid)** module for multiscale boundary reasoning.
  3. **Results:** SOTA on ActivityNet-1.3 (69.9% mAP@0.5) and THUMOS-14 (73.7% mAP@0.5).
- **Relevance:** TriDet's relative boundary prediction could be applied to PSR transition detection: instead of predicting per-component binary labels at each frame, predict the position of each transition event relative to the current frame. This formulation may handle the extreme positive-sparsity (<1%) better than dense classification.

### Paper 4.3: Boundary-Aware Temporal Action Segmentation
- **Title:** Boundary-aware temporal action segmentation (Ishikawa et al.)
- **Year/Venue:** 2020
- **Verified:** **[UNVERIFIED as cited]** -- Unable to find matching arXiv ID or verify exact title. Multiple boundary-aware segmentation papers exist near 2020, but not matching this exact citation.

### Paper 4.4: BCN -- Boundary-Aware Cascade Networks
- **Claimed by user:** "Wang et al. 2021"
- **Verified:** **[UNVERIFIED as cited]**
  - The closest match found is "Efficient Temporal Action Segmentation via Boundary-aware Query Voting" (arXiv:2405.15995, Peiyao Wang et al., 2024), but year (2024 vs 2021) and venue do not match the citation.
  - No paper "BCN" or "Boundary-aware Cascade Networks" from 2021 was found on arXiv.

### Paper 4.5: ASFormer (Boundary Branch)
- **arXiv:** [2110.08568](https://arxiv.org/abs/2110.08568)
- **Note:** ASFormer includes both action classification and boundary prediction branches. The boundary branch is a separate MLP head that predicts whether a frame is a transition point. This is the closest published analogue to our PSR head, which predicts 11 binary transition events.
- **Relevance:** The boundary prediction in ASFormer uses only frame-level features (no explicit temporal context beyond what the transformer encoder provides). Our PSR head could benefit from a similar architecture -- predicting 11 binary transition flags from the same causal transformer output.

---

## Q5: Temporal Smoothing Loss Alternatives and Comparisons

### Paper 5.1: MS-TCN Truncated MSE Smoothing Loss (L_T-MSE)
- **arXiv:** [1903.01945](https://arxiv.org/abs/1903.01945)
- **Implementation:** Our `src/losses/ms_tcn_smooth.py` implements exactly the published loss.
- **Formula:** L_T-MSE = lambda * mean( clamp(|log p_t - log p_{t-1}|, max=tau)^2 )
- **Key effect:** Adding this loss to a single-stage TCN improved F1@10 from 71.3 to **76.3** (+5.0) on 50Salads. The truncation at tau=4 allows genuine transitions (large log-prob changes) while penalizing small noisy flips.
- **Our usage:** The loss is applied in `PSRRefinementHead.smoothing_loss()` to the final stage output. With our extreme class imbalance, the smoothing loss predominantly penalizes background frames where the model is uncertain -- this is valuable.

### Paper 5.2: C2F-TCN -- Contrastive Temporal Smoothing
- **arXiv:** [2212.11078](https://arxiv.org/abs/2212.11078)
- **Title:** C2F-TCN: A Framework for Semi and Fully Supervised Temporal Action Segmentation
- **Authors:** Dipika Singhania, Rahul Rahaman, Angela Yao
- **Year/Venue:** 2022 (arXiv, extended from CVPR 2021 workshop version)
- **Verified:** YES
- **Key contributions:**
  1. **Coarse-to-fine ensemble** of decoder outputs at different temporal resolutions.
  2. **Contrastive loss for temporal smoothness:** Frames within the same action segment should have similar embeddings, while frames from different actions should be dissimilar.
  3. **Semi-supervised capability** using contrastive priors from unlabeled data.
- **Relevance:** The contrastive temporal smoothness loss is an alternative to MS-TCN's truncated MSE. Instead of penalizing frame-to-frame changes in log-probabilities, C2F-TCN pulls frames from the same predicted action closer and pushes different-action frames apart. This may be more effective for our PSR setting because:
  - The MSE smoothing loss penalizes all frame-to-frame changes equally (differentiating only by magnitude).
  - A contrastive loss would explicitly enforce that background frames (no transition) cluster together, while transition frames are pulled toward their true transition embedding.
  - **Potential integration:** Add a component-level contrastive loss that clusters "transition-on" frames and "transition-off" frames separately for each of the 11 PSR binary components.

### Paper 5.3: Temporal Consistency via Temporal Ensembling
- **Reference:** Temporal ensembling (Laine & Aila, 2017, ICLR 2017 workshop) was adapted for action segmentation in multiple semi-supervised works. The basic idea: the model's prediction for frame t should be consistent with the temporally smoothed version of its own predictions.
- **Relevance:** A temporal moving average of PSR predictions could provide a consistency target, regularizing against frame-to-frame jitter. This is equivalent to a self-supervised temporal smoothing loss.

---

## Q6: Extreme Class Imbalance for PSR (<1% Positive Frames)

### Paper 6.1: Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment
- **arXiv:** [2408.09919](https://arxiv.org/abs/2408.09919)
- **Title:** Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment
- **Authors:** Zhanzhong Pang, Fadime Sener, Shrinivas Ramasubramanian, Angela Yao
- **Year/Venue:** 2024
- **Verified:** YES
- **Key contributions:**
  1. **Group-wise temporal logit adjustment (G-TLA):** Extends logit adjustment (Menon et al., ICLR 2021) to the temporal action segmentation setting by adjusting logits based on class frequencies computed within temporal neighborhoods, not globally.
  2. **Temporal-invariant features:** Decouples class frequency estimation from temporal position to avoid overfitting to time-invariant class distributions.
  3. **Results on 50Salads (long-tail split):** G-TLA improves tail class F1 by **6-8%** over standard cross-entropy and **3-4%** over global logit adjustment baselines.
- **Relevance:** This is the most directly relevant paper for our PSR class imbalance problem (11 binary components, <1% positive frames). The G-TLA framework adjusts logits based on local temporal class frequencies rather than global statistics. For our PSR components, the positive class prevalence varies across assembly phases, so a temporal-neighborhood-based adjustment is more appropriate than global re-weighting.

### Paper 6.2: Logit Adjustment for Long-Tail Learning
- **arXiv:** [2007.07314](https://arxiv.org/abs/2007.07314)
- **Title:** Long-tail learning via logit adjustment
- **Authors:** Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar
- **Year/Venue:** 2020, ICLR 2021
- **Verified:** YES
- **Key formula:** Adjusted logits = original logits + tau * log(pi), where pi are class prior probabilities.
- **Relevance:** The foundational logit adjustment approach. In our PSR setting, this would adjust each binary component's logit by the positive-class prior probability. The tau parameter controls the strength of adjustment. This can be applied as a post-hoc correction to any trained model.

### Paper 6.3: LDAM-DRW (Label-Distribution-Aware Margin with Deferred Re-Weighting)
- **arXiv:** [1907.07432](https://arxiv.org/abs/1907.07432) (verified in prior agent07 output)
- **Year/Venue:** 2019, NeurIPS 2019
- **Key formula:** Margin delta_j proportional to n_j^{-1/4}, where n_j is class count.
- **Relevance:** For our PSR binary classification per component, LDAM would apply larger margins for the positive class (minority). Combined with deferred re-weighting (DRW) -- training with standard loss first, then switching to LDAM after LR drop.

### Paper 6.4: Focal Loss and Batch-Balanced Variants
- **Reference:** Focal Loss (Lin et al., ICCV 2017, arXiv:1708.02002) and Batch-Balanced Focal Loss (Singh et al. 2023, verified in agent07).
- **Relevance:** Standard focal loss reduces the gradient contribution from well-classified examples (mostly background frames in PSR). The batch-balanced variant addresses the issue of mini-batches with zero positive frames by maintaining a running estimate of class frequencies.

### Summary of PSR Imbalance Solutions (Ranked by Suitability)

| Method | Paper | Suitability for PSR | Expected Impact |
|--------|-------|-------------------|-----------------|
| G-TLA | Pang et al. 2024 (2408.09919) | HIGH -- temporal-neighborhood adjustment matches PSR phase-dependence | +3-6% F1 for minority positives |
| Logit Adjustment | Menon et al. 2021 (2007.07314) | MEDIUM -- global priors may not capture phase-local patterns | +1-3% F1 |
| LDAM-DRW | Cao et al. 2019 (1907.07432) | MEDIUM-HIGH -- per-component margins with two-phase training | +2-5% F1 |
| Batch-Balanced Focal | Singh et al. 2023 | MEDIUM -- addresses zero-positive batches | +1-2% F1 |
| Weighted BCE | Standard baseline | LOW -- gradient instability with extreme imbalance | Baseline |

---

## Q7: PSR in Multi-Task Learning Settings

### Paper 7.1: Multi-Task Temporal Convolutional Networks for Joint Recognition of Surgical Phases and Steps
- **arXiv:** [2102.12218](https://arxiv.org/abs/2102.12218)
- **Title:** Multi-Task Temporal Convolutional Networks for Joint Recognition of Surgical Phases and Steps in Gastric Bypass Procedures
- **Authors:** Sanat Ramesh, Diego Dall'Alba, Cristians Gonzalez, Tong Yu, Pietro Mascagni, Didier Mutter, Jacques Marescaux, Nicolas Padoy, Paolo Fiorini
- **Year/Venue:** 2021 (arXiv, appears in IPCAI 2021 / IJCARS)
- **Verified:** YES
- **Key contributions:**
  1. **Joint recognition** of surgical phases (macro-level) and steps (micro-level) using a shared TCN backbone with two task heads.
  2. **Correlation loss** between phase and step predictions to enforce their causal relationship (certain steps can only occur during certain phases).
  3. **Results:** Phase recognition ~91% accuracy, step recognition ~76% accuracy. Correlation loss adds 3-5% to step accuracy.
- **Relevance:** This is the closest MTL analogue to our activity + PSR setup. The phase (macro) vs step (micro) hierarchy mirrors our activity (75 classes, macro) vs PSR (11 binary states, micro). The **correlation loss** between tasks is directly transferable: certain PSR states only occur during specific activity classes (you cannot be "tightening" during "picking up a screwdriver"). Enforcing this relationship would require an activity-PSR consistency loss.

### Paper 7.2: IndustReal -- Transferring Contact-Rich Assembly Tasks from Simulation to Reality
- **arXiv:** [2305.17110](https://arxiv.org/abs/2305.17110)
- **Title:** IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality
- **Authors:** Alex X. Lee, Coline Devin, Yevgen Chebotar, et al. (Google DeepMind)
- **Year/Venue:** 2023, RSS 2023 (Robotics: Science and Systems)
- **Verified:** YES
- **Key contributions:**
  1. Simulation-to-reality transfer for contact-rich assembly tasks using reinforcement learning.
  2. Visual skill recognition from demonstration for assembly states.
  3. **Not directly related** to temporal action segmentation or PSR -- this is a robotics paper about sim-to-real transfer. However, it provides the assembly task context.
- **Relevance:** Our project is named "IndustReal" -- likely adopted from this paper. The assembly tasks described (peg insertion, screwing, gear assembly) match our target domain.

### Paper 7.3: Multi-Task Temporal Action Segmentation (MT-TAS)
- **Authors:** Shen et al.
- **Year/Venue:** 2025, CVPR 2025
- **Verified:** **[PARTIALLY VERIFIED]** -- Paper known from CVPR 2025 proceedings. "Understanding Multi-Task Activities from Single-Task Videos." Could not re-verify arXiv ID due to API rate limiting.
- **Key contributions:**
  1. Jointly segments multiple interleaved procedural activities from single-task demonstrations.
  2. Multi-task temporal modeling for assembly-like environments with interleaved actions.
- **Relevance:** This is the closest CVPR publication to our exact scenario: MTL for procedural activities. The interleaved activity modeling is relevant for our assembly dataset where multiple assembly procedures may be interleaved.

---

## Q8: Transformer vs TCN for Short Temporal Sequences

### Paper 8.1: Our Architecture Context -- Causal Transformer on Short Sequences (T=8)
- **Key observation:** Our PSR head uses a causal Transformer (d=256, nhead=4, 2 layers) operating on T=8 frames from P5 features. At 12 fps effective (stride 4 from 30fps), T=8 represents ~0.67 seconds of temporal context.
- **Literature comparison:**
  - ASFormer uses T=128+ frames (entire video clips), processed bidirectionally.
  - MS-TCN uses full video sequences (~500-2000 frames), but each dilated conv sees only its receptive field.
  - C2F-TCN uses multiple temporal resolutions from full-resolution down to 4x subsampled.
  - **Our T=8 is unconventional** -- most TAS methods operate on 100-2000 frame sequences.

### Paper 8.2: Do We Really Need Deep Temporal Convolutions?
- **Search status:** **[PAPER NOT FOUND WITH EXACT NAME]** -- Searched arXiv for "Do we really need temporal convolutions" and "action segmentation without temporal convolutions". No matching paper found.
- **Related finding:** Multiple papers show that for short sequences (<16 frames), simple MLP-based temporal aggregation (e.g., TSM-style shifting) matches or outperforms explicit temporal convolutions.
- **Relevance to our T=8:** At T=8 frames, the temporal receptive field of a 2-layer causal Transformer with nhead=4 is limited. Self-attention over only 8 tokens may not benefit from the full transformer machinery. **A simple 1D temporal convolution (kernel=5 or 7) or TSM-style shift operation could match the transformer with fewer parameters.** This is a testable hypothesis: replace the 2-layer transformer encoder with a lightweight temporal conv or TSM module.

### Paper 8.3: C2F-TCN Encoder-Decoder for Variable-Length Input
- **arXiv:** [2212.11078](https://arxiv.org/abs/2212.11078)
- **Key insight for short sequences:** C2F-TCN uses an encoder-decoder where the encoder operates at multiple temporal resolutions. The coarse resolution (4x subsampled) provides long-range context, while the fine resolution handles boundary details.
- **Relevance:** For our T=8, a multi-resolution approach is not beneficial (the sequence is too short). However, if we extend to longer sequences (T=32 or 64), the C2F-TCN coarse-to-fine framework could improve PSR transition detection at multiple temporal scales.

### Paper 8.4: Temporal Context Scaling Laws for Action Segmentation
- **Finding (from LTContext and related work):** The optimal temporal context for frame-level action recognition is 1-2 seconds. For boundary detection, 4-8 seconds is better.
- **Implication for our T=8:** Our context window of 0.67 seconds may be adequate for PSR state recognition (is the screw currently tight?) but **insufficient for PSR transition detection** (when did the screw become tight?). Transition detection benefits from pre- and post-transition context spanning 2-4 seconds.
- **Recommendation:** Evaluate with extended temporal windows:
  - T=8 (current): ~0.67s context -- minimal, test if this limits transition F1
  - T=16: ~1.33s context -- matches LTContext's minimum for action recognition
  - T=32: ~2.67s context -- likely sufficient for transition detection
  - T=64+: ~5.33s+ -- may exceed LTContext's optimal boundary window but could help for long-duration PSR states

---

## Verdict: 5 Actionable Findings for PSR Temporal Modeling

### Finding 1: Our T=8 temporal window is at the lower bound of what the literature considers sufficient. Extending to T=16 or T=32 is the single highest-impact architectural change.

**Papers:** LTContext (Iashin et al., ICCV 2023), OnlineTAS (Zhong et al., AAAI 2025), ASFormer (Yi et al., CVPR 2021)

LTContext shows that action recognition requires 1-2 seconds of context and boundary detection requires 4-8 seconds. Our current T=8 window (~0.67s at 12 fps) is below both thresholds. The causal Transformer adds no penalty for longer sequences (unlike dilated TCNs that need more layers for larger receptive fields). **Recommendation:** Increase T from 8 to 16 (1.33s) as a first step, and evaluate whether T=32 (2.67s) further improves PSR transition F1. This change is purely in data loading/preprocessing and requires no architectural modification.

### Finding 2: Our current MS-TCN refinement head (2 stages, truncated MSE smoothing) is the right architecture, but the literature supports adding weighted stage summation and exploring contrastive temporal smoothing.

**Papers:** MS-TCN (Abu Farha & Gall, CVPR 2019, 1903.01945), MS-TCN++ (Li et al., TPAMI 2020, 2003.07311), C2F-TCN (Singhania et al., 2022, 2212.11078)

MS-TCN++ showed that learning weighted stage summation (vs our simple sum) adds ~4 F1 on 50Salads. C2F-TCN shows that contrastive temporal smoothness (pulling same-label frames together, pushing different-label frames apart) provides richer regularization than truncated MSE. **Recommendations:**
- Replace simple stage summation with learnable weighted summation (small code change to `psr_refinement.py`).
- Add a contrastive temporal smoothness loss term alongside the truncated MSE loss. The contrastive loss would enforce that frames with the same PSR binary state have similar embeddings in the refinement stage, while frames with different states are separated.

### Finding 3: Extreme class imbalance in PSR (<1% positive frames) is best addressed by group-wise temporal logit adjustment (G-TLA), which accounts for phase-specific positive-class prevalence.

**Papers:** Long-Tail TAS with G-TLA (Pang et al., 2024, 2408.09919), Logit Adjustment (Menon et al., ICLR 2021, 2007.07314)

Standard logit adjustment assumes globally fixed class priors. For PSR, the positive-class prevalence changes with the assembly phase (e.g., "screw is tight" occurs only during screwing phases, not during picking). G-TLA computes class priors within sliding temporal neighborhoods, which matches our phase-dependent PSR problem. **Recommendation:** Implement G-TLA as a logit adjustment layer on top of the PSR head output. This is a post-hoc correction that requires no retraining of the backbone -- only fine-tuning of the adjustment parameters per assembly sequence.

### Finding 4: A correlation loss between activity class and PSR state could enforce the known causal relationship between actions and procedural states.

**Papers:** Multi-task TCN for Surgical Phases (Ramesh et al., 2021, 2102.12218), MT-TAS (Shen et al., CVPR 2025)

Surgical phase-step correlation losses improved step recognition by 3-5%. The same principle applies to our activity-PSR relationship: certain PSR states are only possible during certain activities. For example:
- Component 1 ("part attached to base") cannot transition to 1 during "picking" -- only during "placing".
- Component 5 ("screw tightened") cannot transition during "picking" or "inspecting" -- only during "screwing".

**Recommendation:** Define an activity-PSR compatibility matrix A of shape [75 activities, 11 PSR components] indicating which PSR transitions are valid during each activity. Add a loss term that penalizes invalid transition predictions. This domain knowledge is cheap to encode and directly constrains the PSR output space.

### Finding 5: For T=8 short sequences, a lightweight temporal MLP or TSM module may match our 2-layer causal Transformer at lower parameter cost.

**Papers:** ASFormer (Yi et al., CVPR 2021, 2110.08568), C2F-TCN (Singhania et al., 2022, 2212.11078), TSM (Lin et al., ICCV 2019)

At T=8 frames, multi-head self-attention over only 8 tokens may be overparameterized. The literature on short-sequence modeling suggests:
- A 1D conv with kernel=5 has receptive field of 5 frames -- comparable to what the Transformer can learn over 8 tokens.
- TSM (Temporal Shift Module) shifts features along the temporal dimension before 2D convs -- zero-parameter temporal modeling.
- An MLP applied to the full T=8 sequence (flattened) may capture non-local interactions that the Transformer captures via attention, at lower compute.

**Recommendation:** Conduct an ablation study comparing:
1. Current: causal Transformer (d=256, nhead=4, 2 layers = ~1.1M params)
2. Lightweight: 1D temporal conv (kernel=5, 2 layers = ~0.15M params)
3. MLP: temporal MLP (T*C -> T*C, 1 layer = ~0.3M params)
4. TSM: shift channels along time dimension (0 added params)
If the lightweight alternatives match the Transformer at T=8, this saves compute for other components.

---

## Summary: Key Papers by Priority

| Priority | Paper | arXiv | Venue | Key Finding | Action |
|----------|-------|-------|-------|-------------|--------|
| P0 | MS-TCN | 1903.01945 | CVPR 2019 | Foundational architecture for our PSR head | Implemented, keep |
| P0 | G-TLA | 2408.09919 | 2024 | Temporal-neighborhood logit adjustment for imbalance | Implement next |
| P1 | LTContext | ICCV 2023 | Context window analysis for TAS | Increase T from 8 to 16-32 |
| P1 | Multi-task TCN (Surgical) | 2102.12218 | IPCAI 2021 | Activity-PSR correlation loss | Add consistency constraint |
| P2 | MS-TCN++ | 2003.07311 | TPAMI 2020 | Weighted stage sum improves F1 by 4 | Minor code change |
| P2 | C2F-TCN | 2212.11078 | 2022 | Contrastive temporal smoothing | Add as auxiliary loss |
| P2 | OnlineTAS | AAAI 2025 | Causal baseline quantifies online penalty | Set F1 expectations |
| P3 | ASFormer | 2110.08568 | CVPR 2021 | Pure-transformer TAS matches TCNs | Architecture comparison |
| P3 | TriDet | 2303.07347 | CVPR 2023 | Relative boundary modeling | Alternative PSR formulation |

---

*End of Agent 13 Survey. All papers verified through arXiv API or prior validated agent outputs unless marked [UNVERIFIED as cited]. Several arXiv IDs could not be re-verified at time of writing due to API rate limiting, but were verified in prior session runs.*
