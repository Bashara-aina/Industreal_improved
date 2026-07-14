# FINAL CLAUDE SCIENCE QUERIES — ULTIMATE Consultation V2

**Phase:** ULTIMATE Consultation V2 — Phase 3 Final Synthesis (Synthesizer S5)
**Date:** 2026-07-14
**Author:** Synthesizer S5
**Inputs:** All R/D outputs + S1-S4 + 20 V2 agent outputs

---

## How to Use This Document

20 ready-to-paste queries for the final Claude Science session. Each query includes:
- Context (our setup, current baseline)
- Specific question
- What we already know (cite V2 findings)
- Expected answer format

**Categories:** Data (4), Architecture (5), Training (4), Literature (4), Strategy (3).

---

## Category 1 — Data (4 queries)

### Query 1.1 — MediaPipe Pose Comparison
**Context:** Our pose MAE 8.7° on IndustReal (HL2 sensor GT). Pose is our novel contribution (no WACV 2024 baseline). Concern: MediaPipe may beat us on the same data.

**Question:** What is the published accuracy (forward/up angular MAE in degrees) of MediaPipe Face Mesh on egocentric video with HoloLens 2 sensor ground truth? Are there published papers or technical reports benchmarking MediaPipe vs custom pose estimation models in industrial assembly contexts?

**What we know:** MediaPipe achieves 5° MAE on controlled data (Zhu et al. 2023). Our 8.7° may be worse.

**Expected answer:** Specific numbers, comparison methodology, any industrial-context benchmarks.

---

### Query 1.2 — Activity Class 0 Interpretation
**Context:** Class 0 = take_short_brace (797 frames). D1 raised concern that HL2 idle periods might be mislabeled.

**Question:** In the IndustReal dataset, what is the labeling protocol for ID 0 and adjacent ID 1-3? Are take_short_brace and similar low-action classes used as "idle" labels by annotators? Are there published annotations on per-frame label noise in IndustReal?

**What we know:** V2 agent01 confirmed class 0 = take_short_brace via CSV inspection. V1 doc 209 mentions 75 classes.

**Expected answer:** Documentation of labeling protocol, statistics on class boundaries.

---

### Query 1.3 — PSR Transition Frame Precision
**Context:** Our PSR is per-frame binary, not transition detection. Need to know the annotation precision (1 frame, 5 frames, etc.).

**Question:** What is the temporal precision of the original WACV 2024 PSR annotations? Are transitions marked at the exact transition frame, or within a tolerance? Are there published inter-annotator agreement statistics on PSR_labels.csv?

**What we know:** V1 doc 218 reports 301 transition events across 36 recordings. V2 agent04 confirms positive rate <0.5%.

**Expected answer:** Annotation methodology, inter-annotator agreement, tolerance.

---

### Query 1.4 — Body Pose Annotation Status
**Context:** Our config.py states body pose (17 COCO keypoints) has "no real annotations" (pseudo-keypoints from detection).

**Question:** Is there an official body pose annotation in IndustReal? Has any paper used body pose on this dataset? What is the source of the pseudo-keypoints in our `hands.csv`?

**What we know:** HoloLens 2 has native hand tracking (52-D joints). 17 COCO KP derived from detection boxes.

**Expected answer:** Documentation of body pose annotation source, any published usage.

---

## Category 2 — Architecture (5 queries)

### Query 2.1 — ConvNeXt-Tiny for Video MTL
**Context:** Our backbone is ConvNeXt-Tiny (28.59M, ImageNet-1K). Concern: no published evidence for ConvNeXt-Tiny on video tasks. MViTv2-S would likely give +10-15% activity.

**Question:** Are there published papers using ConvNeXt (any size) as backbone for video understanding, especially with task heads attached (detection, classification)? Compare published accuracy to MViTv2-S and VideoMAE-S on the same video benchmarks.

**What we know:** Liu et al. CVPR 2022 introduced ConvNeXt for image classification. No direct video MTL paper found.

**Expected answer:** Papers, accuracy comparisons, justification for ConvNeXt-Tiny choice.

---

### Query 2.2 — BiFPN vs Standard FPN for 24-Class Detection
**Context:** Our FPN is standard P3-P7 (4.48M). BiFPN would add ~3-5M for ~+0.4-0.7 mAP per Tan et al.

**Question:** For 24-class detection with sparse labels (17.9% positive frames) at 224×224 input, is BiFPN expected to outperform standard FPN? Are there ablation studies on small-class detection with FPN variants?

**What we know:** Tan et al. CVPR 2020 reports +0.4-0.7 mAP on COCO with BiFPN. Our setup is much smaller scale.

**Expected answer:** Specific ablation results, recommendation for our setup.

---

### Query 2.3 — Anchor-Free Detection at 224px
**Context:** Our detection head is RetinaNet-style. Anchor-free (YOLOX-style) gives +4.3 mAP on COCO.

**Question:** Are there published 224×224 anchor-free detection benchmarks? Specifically for assembly or industrial detection? What's the ceiling mAP at 224×224 for 24-class detection with 17.9% labeled frames?

**What we know:** YOLOX arxiv 2107.08430 reports +4.3 mAP. Our setup is small-scale.

**Expected answer:** Published ceilings, recommendation for our setup.

---

### Query 2.4 — Activity Head: FeatureBank + TCN + 2×ViT Necessity
**Context:** Our activity head has FeatureBank (T=16) + TCN + 2×ViT (0.69M total). Concern: may be over-engineered.

**Question:** For 75-class activity recognition on T=16 egocentric clips, is the FeatureBank+TCN+ViT combination published? Are there simpler architectures (TCN only, ViT only, MLP only) that match? What's the published accuracy gap?

**What we know:** V2 agent08 describes the activity head. Our backbone is convnext_tiny (no native temporal modeling).

**Expected answer:** Simpler architecture comparisons, accuracy/efficiency trade-offs.

---

### Query 2.5 — 6D Rotation for Head Pose MTL
**Context:** Our `GeometryAwareHeadPose` is implemented but DISABLED (`USE_GEO_HEAD_POSE=False`).

**Question:** Is the Zhou et al. CVPR 2019 6D rotation representation published for MTL? Are there specific MTL papers that report 30-50% MAE reduction from MSE → 6D rotation? Is the geodesic loss implementation standard?

**What we know:** Zhou et al. introduced 6D rotation for general pose. We use it for head pose specifically.

**Expected answer:** MTL-specific evidence, expected MAE improvement.

---

## Category 3 — Training (4 queries)

### Query 3.1 — PCGrad Failure Modes at 4 Tasks
**Context:** We use PCGrad with Kendall + per-task caps. Concern: PCGrad may have known failure modes at 4+ tasks.

**Question:** What are the documented failure modes of PCGrad (Yu et al. NeurIPS 2020) when applied to 4+ task MTL with heterogeneous losses (CE + Focal + MSE)? Are there improvements (CAGrad, Nash-MTL) that consistently beat PCGrad in this regime?

**What we know:** D8 raised this concern. CAGrad (NeurIPS 2021) and Nash-MTL (ICML 2022) are documented improvements.

**Expected answer:** Failure cases, recommended alternative.

---

### Query 3.2 — Per-Task Kendall Caps: Novelty
**Context:** Our per-task caps are det=[-4,2], act=[-0.5,2], psr=[-4,0], pose=[-4,3]. KENDALL_HP_PREC_CAP=True.

**Question:** Are these specific per-task cap configurations (or similar) published? Is KENDALL_HP_PREC_CAP a known technique? Are there alternative cap strategies (e.g., based on gradient norms)?

**What we know:** Kendall CVPR 2018 original paper has no caps. Caps added per R2.

**Expected answer:** Published alternatives, novelty assessment.

---

### Query 3.3 — Multi-Task Distillation from ST Teachers
**Context:** Our distillation module exists (`src/training/distillation.py`). Task #261 implemented.

**Question:** For multi-task MTL, is distilling from single-task teachers a published technique? Does it work for tasks with different loss types (CE + Focal + MSE)? Are there specific distillation losses for MTL?

**What we know:** Knowledge distillation (Hinton et al. 2015) is standard. MTL-specific distillation less common.

**Expected answer:** MTL distillation papers, expected gains.

---

### Query 3.4 — Long-Tail MTL for 75-Class Activity
**Context:** Our activity has 75 classes, 16 with <10 frames. Logit-adjustment (Menon et al. 2020) active. LDAM-DRW module exists.

**Question:** For 75-class long-tail classification in MTL setting, does LDAM-DRW (Liu et al. NeurIPS 2019) outperform logit-adjustment? Are there other published long-tail MTL techniques? What's the expected top-1 improvement on a 16-tail-class setup?

**What we know:** 16 classes with <10 frames. OLTR 2019, BBN 2019, LDAM 2019 are reference methods.

**Expected answer:** Best method for our setup, expected top-1 improvement.

---

## Category 4 — Literature (4 queries)

### Query 4.1 — 2025-2026 MTL on Industrial Video
**Context:** D8 raised that we may have missed 2025-2026 papers.

**Question:** Find MTL papers on industrial assembly, egocentric video, or IndustReal-like datasets published or arXiv-preprinted between January 2025 and July 2026. For each: backbone, task set (must include 2+ of detection/activity/PSR/pose), MTL or ST, metrics.

**What we know:** R3 found no 2025-2026 papers but search may be incomplete.

**Expected answer:** Comprehensive list with arXiv IDs and venues.

---

### Query 4.2 — Head Pose MTL Benchmarks
**Context:** Pose is our novel contribution. No published head pose MTL benchmark exists for IndustReal.

**Question:** Are there MTL benchmarks that include head pose estimation as one of the tasks? Specifically, are there egocentric video MTL benchmarks (not first-person activity but including gaze/head pose)? What's the typical MTL/ST retention for pose?

**What we know:** Pose MAE 8.7° on our setup. No SOTA exists on IndustReal.

**Expected answer:** Benchmark datasets, typical MTL/ST retention ratios.

---

### Query 4.3 — PSR at <1% Positive Rate
**Context:** Our PSR has <0.5% positive rate. Most temporal action detection papers assume 5-50%.

**Question:** Are there published papers that successfully classify or detect temporal events at <1% positive rate? What loss functions work? What's the typical F1 ceiling?

**What we know:** Most published methods test on balanced or moderately imbalanced data. Our setup is extreme.

**Expected answer:** Successful published approaches, expected F1.

---

### Query 4.4 — MTL Negative Transfer Quantification
**Context:** V1 doc 208 says we'll likely lose on some heads. Need to quantify expected negative transfer.

**Question:** For 4-task MTL with 100x+ loss scale differences, what is the expected MTL/ST retention ratio per task? Are there published benchmarks showing negative transfer magnitudes for each task type (detection vs classification vs regression)?

**What we know:** R4 estimates MTL/ST ratios of 0.6-1.0 per task. D8 raised that PCGrad has limitations.

**Expected answer:** Quantitative expectations per task type.

---

## Category 5 — Strategy (3 queries)

### Query 5.1 — AAIML 2027 Topic Alignment
**Context:** D4 raised AAIML scope unverified. AAIML deadline Oct 10, 2026.

**Question:** What is the exact scope of AAIML 2027 (call for papers)? What topics are in scope? Is "industrial assembly perception" or "MTL for industrial video" a primary topic? What recent (2024-2026) AAIML papers cover similar topics?

**What we know:** AAIML = "Advances in AI for Manufacturing, Logistics, and Industrial Systems" (assumption). Deadline Oct 10, 2026.

**Expected answer:** Exact scope, in-scope/out-of-scope topics, related prior papers.

---

### Query 5.2 — Concurrent MTL Submissions Risk
**Context:** D4 raised concern about TU Delft / Stanford / CMU concurrent submissions.

**Question:** Search arXiv and Google Scholar for any submission scheduled for AAIML 2027 (or similar 2026-2027 venues) that covers multi-task learning on industrial assembly. Look for Schoonbeek follow-ups, Stanford AI Lab work, etc.

**What we know:** D4 raised threat from Schoonbeek's group (TU Delft), Stanford, CMU.

**Expected answer:** Concurrent submission probability, threat assessment.

---

### Query 5.3 — Paper Acceptance Probability Estimation
**Context:** D9 estimated 75% acceptance for best case, 40-50% for likely, 10-20% for worst case.

**Question:** For a paper combining MTL methodology (Kendall+PCGrad+caps) with novel task (IndustReal + pose), submitted to AAIML 2027, what is the typical acceptance rate at similar venues (AAAI, IJCAI, NeurIPS workshops)? What reviewer concerns are most likely?

**What we know:** 8-page limit, MTL methodology with novel dataset, single consumer GPU deployment.

**Expected answer:** Acceptance probability, top reviewer concerns.

---

## Quick-Reference: 20 Queries at a Glance

| # | Category | Query |
|---|---|---|
| 1.1 | Data | MediaPipe pose comparison |
| 1.2 | Data | Activity class 0 interpretation |
| 1.3 | Data | PSR transition frame precision |
| 1.4 | Data | Body pose annotation status |
| 2.1 | Architecture | ConvNeXt-Tiny for video MTL |
| 2.2 | Architecture | BiFPN vs standard FPN |
| 2.3 | Architecture | Anchor-free detection at 224px |
| 2.4 | Architecture | Activity head complexity |
| 2.5 | Architecture | 6D rotation for head pose MTL |
| 3.1 | Training | PCGrad failure modes at 4 tasks |
| 3.2 | Training | Per-task Kendall caps novelty |
| 3.3 | Training | Multi-task distillation from ST |
| 3.4 | Training | Long-tail MTL for 75-class activity |
| 4.1 | Literature | 2025-2026 industrial MTL |
| 4.2 | Literature | Head pose MTL benchmarks |
| 4.3 | Literature | PSR at <1% positive rate |
| 4.4 | Literature | MTL negative transfer quantification |
| 5.1 | Strategy | AAIML 2027 topic alignment |
| 5.2 | Strategy | Concurrent MTL submissions risk |
| 5.3 | Strategy | Paper acceptance probability |

---

## Output

This file is the Claude Science query pack. Use these as the basis for the final session, targeting the highest-priority Tier 1 questions first (1.1, 2.1, 2.5, 3.1, 4.1, 5.1).
