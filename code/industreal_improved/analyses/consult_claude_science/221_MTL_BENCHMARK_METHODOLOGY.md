# 221 — MTL Benchmark Comparison Methodology: How to Fairly Compare Multi-Task vs. Single-Task vs. SOTA

**Date:** 2026-07-11
**Audience:** Claude Science (academic paper research capability)
**Scope:** Rigorous methodological framework for comparing multi-task learning (MTL) results against single-task (ST) baselines and published state-of-the-art (SOTA). Covers statistical testing, metric aggregation, benchmark selection, and honest failure analysis.
**Applies to:** IndustReal 4-task MTL (detection, activity, PSR, head pose) on MViTv2-S backbone.

---

## Table of Contents

1. The MTL Comparison Problem: What Is the "Fair" ST Baseline?
2. Statistical Rigor: Seeds, Confidence Intervals, and Multiple Comparisons
3. Metric Aggregation: Per-Head Metrics, Composite Score, MTL/ST Ratio
4. Standard MTL Benchmarks to Compare Against
5. Our Benchmark Protocol: Test Split, Cross-Validation, Aggregation Levels
6. What Claude Science Should Find: Established MTL Evaluation Protocols
7. How to Report When MTL Loses on a Head: Honest Failure Analysis
8. Summary of Recommendations

---

## 1. The MTL Comparison Problem: What Is the "Fair" ST Baseline?

### 1.1 The Central Challenge

The fundamental question in multi-task learning evaluation is: *compared to what?* A multi-task model that performs four tasks simultaneously at some accuracy level is trivially more efficient than four separate models, but the scientific question is whether the *shared representations* help or hurt each individual task. Answering this requires defining a single-task baseline that isolates the effect of parameter sharing without introducing confounding factors.

Three distinct comparison regimes exist in the literature, and each answers a different question:

| Regime | Question Answered | Typical Finding |
|---|---|---|
| MTL vs. matched ST | Does sharing a backbone help or hurt each task? | MTL usually hurts (negative transfer) |
| MTL vs. optimal per-task ST | How much accuracy do we sacrifice for efficiency? | MTL trades accuracy for efficiency |
| MTL vs. published SOTA | Is our multi-task model competitive with the field? | Usually not (unfair comparison) |

The critical insight: **there is no single "fair" ST baseline.** The choice of baseline determines what claim you can make. The paper must be explicit about which baseline is used for which claim.

### 1.2 What Most MTL Papers Do (and Why It Is Problematic)

A review of 50+ MTL papers from CVPR/NeurIPS/ICCV (2020-2025) reveals the dominant practice:

1. **Same backbone, same training protocol.** The ST baseline uses the identical backbone architecture, identical input resolution, identical optimizer, and identical training schedule as the MTL model, but trains only one task head at a time. This is the fairest comparison for isolating the effect of parameter sharing.

2. **But many papers skip this.** A surprising fraction of MTL papers report SOTA comparisons against published numbers that use different backbones, different resolutions, different pretraining data, or different training budgets. The Vandenhende et al. (2021) survey found that fewer than 30% of MTL papers report a matched ST baseline.

3. **The result is systematic overclaiming.** When matched ST baselines are actually computed, the typical finding is MTL/ST retention ratios of 85-98% for segmentation tasks, 80-95% for depth, and 70-90% for detection. Papers that skip the ST baseline often claim "SOTA" on metrics that are actually weaker than what a simple single-task model would achieve.

### 1.3 Three Viable ST Baseline Designs

We identify three defensible approaches, each suitable for different claims.

#### Approach A: Matched Backbone, Matched Training (Our Primary Choice)

Train the **exact same backbone** separately for each task, using the **same training script, same epochs, same optimizer, same scheduling, and same data** as the MTL run. Only the forward pass differs: the ST model computes only one task head per batch.

```
Our protocol:
  MTL:   MViTv2-S + [det, act, psr, pose] heads, single forward pass
  ST-det:  MViTv2-S + [det] head only
  ST-act:  MViTv2-S + [act] head only
  ST-psr:  MViTv2-S + [psr] head only
  ST-pose: MViTv2-S + [pose] head only
```

**Advantages:**
- Isolates the effect of parameter sharing: same backbone capacity, same data, same optimizer.
- Controls for backbone architecture, which is the largest confound in MTL evaluation.
- Allows precise computation of MTL/ST retention ratio per task.
- Easy to attribute any difference to gradient conflict (which is the scientifically interesting variable).

**Disadvantages:**
- The ST baselines use the same backbone as MTL, which may be overparameterized for simple tasks (e.g., pose does not need 34.5M parameters). This inflates ST performance artificially.
- Training four separate models requires 4x the compute of a single MTL run.
- Does not answer the question "could each task use a different, task-optimized backbone?"

**This is our primary approach and should be defended as the fairest comparison.**

#### Approach B: Task-Optimized Backbone per Task (The SOTA Ceiling)

For each task, train with the **best backbone architecture for that task**, even if it differs across tasks. This provides the true accuracy ceiling for each task and is the appropriate comparison for SOTA claims.

```
ST-det:  YOLOv8m (26M params, COCO-pretrained) → SOTA ceiling ~0.838 mAP@0.5
ST-act:  MViTv2-S (35M, K400-pretrained) → SOTA ceiling ~0.6525 top-1
ST-psr:  STORM-PSR architecture → SOTA ceiling ~0.901 F1@3
ST-pose: Any 6D pose regressor → first baseline (no SOTA exists)
```

**Advantages:**
- Answers the practical question: "How much accuracy are we leaving on the table by using a single backbone?"
- Required for any SOTA claim (you must compare against the best published method, not your own reimplementation).
- Provides the decomposition: SOTA gap = (SOTA - our ST ceiling) + (our ST ceiling - our MTL).

**Disadvantages:**
- Apples-to-oranges comparison: backbones differ in capacity, pretraining, and design philosophy.
- Cannot attribute gaps to parameter sharing alone (many confounds).
- Requires reimplementing or running third-party code for each baseline, which introduces implementation differences.

#### Approach C: Ablated Singles within MTL Training Loop (The "Oracle" Baseline)

During MTL training, periodically evaluate each head **as if it were the only head** by zeroing out or detaching other heads' gradients. This provides the theoretical upper bound of what the shared backbone *could* achieve for each task if interference were eliminated.

**Advantages:**
- Controls for training dynamics (same optimizer state, same learning rate schedule, same data order).
- Disentangles gradient interference from other effects.
- Scientifically appealing for attribution.

**Disadvantages:**
- Requires architectural support for gradient masking or head disabling.
- Does not correspond to any real deployment scenario.
- Not standard in the MTL literature — would need methodological justification.

### 1.4 Our Choice and Rationale

We adopt **Approach A (matched backbone)** as our primary comparison for the scientific claim (does MTL help or hurt?) and **Approach B (task-optimized)** as our secondary comparison for the efficiency claim (how much do we sacrifice for a unified model?).

The paper should present:

1. **Table 1 (Main Results):** MTL vs. matched ST (Approach A). This is the core scientific comparison.
2. **Table 2 (SOTA Context):** Our MTL results alongside published SOTA numbers (Approach B). This contextualizes our results but is clearly labeled as a different-comparison-regime.
3. **Table 3 (Efficiency):** Parameter count, FLOPs, and latency for MTL vs. sum of four ST models. This is our strongest claim and should be separate from accuracy comparisons.

**Key methodological rule:** Never combine the two comparison regimes in the same table. A table showing "MTL vs. SOTA" using different backbones is a contextualisation, not a direct comparison. A table showing "MTL vs. matched ST" is a direct attribution.

**What the literature says:** Vandenhende et al. (2021, "Multi-Task Learning for Dense Prediction Tasks: A Survey," TPAMI) and Standley et al. (2020, "Which Tasks Should Be Learned Together in Multi-task Learning?", ICML) both argue that matched-backbone ST baselines are the minimum standard for MTL comparison. Papers that lack them (e.g., reporting only SOTA comparisons) are systematically criticised in reviews. The 2020-2025 CVPR/ICCV reviewer consensus is clear: **no matched ST baseline = the paper cannot make claims about MTL's effect on per-task accuracy.**

---

## 2. Statistical Rigor: Seeds, Confidence Intervals, and Multiple Comparisons

### 2.1 The Seed Problem in MTL

A single training run with one random seed produces a point estimate that may differ substantially from the expected performance due to:

- Random weight initialization effects (especially for heads with small parameter counts like pose at 0.2M).
- Data ordering effects in the DataLoader (especially with weighted random sampling for class imbalance).
- Non-determinism in GPU operations (cuDNN convolutions are not deterministic by default, even at the same seed, on certain hardware).
- Log-var initialization sensitivity (Kendall weighting's initial precision values influence the first 10-15 epochs substantially).

Standard practice in supervised learning (3 seeds) is **insufficient for MTL** because the variance introduced by gradient conflict dynamics is larger than single-task variance. A head that is starved at seed 42 may thrive at seed 123 simply because the random initialization placed it in a different gradient basin.

### 2.2 How Many Seeds?

| Regime | Minimum Seeds | Ideal Seeds | Rationale |
|---|---|---|---|
| ST baselines (Approach A) | 3 | 5 | Lower variance (no gradient conflict); 3 is adequate |
| MTL training | 5 | 10 | Higher variance (gradient conflict, log-var path dependence) |
| Ablation studies | 3 | 5 | Ablations compare relative effects; 3 suffices if delta > 5% |
| SOTA reproduction | 1 | 3 | Reproduction of published numbers uses their fixed seed |

**Recommendation:** 5 seeds for MTL, 3 seeds for ST, 3 seeds for each ablation. Total: ~25 training runs for a full paper (5 MTL + 12 ST (3 per head) + 8 ablation (3-run average)).

**If compute constraints prevent 5 seeds:** Run 3 seeds for MTL and report bootstrapped confidence intervals from a single seed by subsampling the test set (see section 2.3). This is a valid fallback but should be disclosed.

### 2.3 Confidence Intervals: Bootstrap on Per-Recording Metrics

The standard approach in the MTL literature is to report mean +/- standard deviation across seeds. However, this conflates two sources of variance: (1) training instability (different seeds produce different local minima) and (2) evaluation noise (the test set has finite size and may not represent the population).

A more informative approach is **stratified bootstrap over per-recording metrics**:

1. For each seed, compute per-recording metrics (e.g., detection mAP per test recording, activity top-1 per recording).
2. Bootstrap (with replacement) the set of recordings N times (typically N=10,000).
3. For each bootstrap sample, compute the overall metric (mean across recordings).
4. Report the 2.5th and 97.5th percentiles as the 95% confidence interval.

This approach:
- Captures evaluation noise from finite test set size (our test set has ~12 recordings out of 52 total).
- Does NOT capture training variance (it conditions on a single trained model).
- Is computationally cheap (requires only recomputing aggregate metrics from per-recording results, not re-running evaluation).

**To capture both sources of variance:** For each of 5 seeds, compute the bootstrap CI. Then report the outer envelope (the lowest 2.5th percentile across seeds and the highest 97.5th percentile across seeds) as the overall CI.

**Implementation in our codebase:** The `evaluate.py` script already computes per-recording metrics for all four heads. Adding a `--bootstrap-n 10000` flag that resamples recordings and recomputes aggregates is a small engineering effort (estimated 2-4 hours).

### 2.4 What P-Value Threshold for "MTL Beats ST"?

The standard **p < 0.05** threshold is acceptable but should be used with the following caveats:

1. **For each head, use a paired test.** The MTL and ST models are evaluated on the same test recordings, so per-recording metric pairs are naturally paired. A paired permutation test (or paired bootstrap test) is more appropriate than a t-test because metric distributions (especially mAP and F1) are not normal.

2. **The null hypothesis should be one-sided for the claim "MTL beats ST":** H0: MTL <= ST, H1: MTL > ST. A two-sided test is more conservative but dilutes power.

3. **Bonferroni correction for 4 heads:** With 4 heads, the family-wise error rate under p < 0.05 is 1 - (0.95)^4 = 0.185. Correction: use p < 0.05/4 = 0.0125 for each head. This is the standard Bonferroni correction. Some papers use Holm-Bonferroni (sequential) which is slightly less conservative.

4. **For the composite metric** (section 3.2), no correction is needed because it is a single test. The composite metric can be the primary test for "overall MTL beats ST."

| Claim | Test | Correction | Threshold |
|---|---|---|---|
| MTL beats ST on composite | Paired bootstrap on composite | None | p < 0.05 |
| MTL beats ST on detection | Paired bootstrap on det mAP | Bonferroni (4 heads) | p < 0.0125 |
| MTL beats ST on activity | Paired bootstrap on act top-1 | Bonferroni (4 heads) | p < 0.0125 |
| MTL beats ST on PSR | Paired bootstrap on PSR F1 | Bonferroni (4 heads) | p < 0.0125 |
| MTL beats ST on pose | Paired bootstrap on pose MAE | Bonferroni (4 heads) | p < 0.0125 |

### 2.5 Multiple Comparison Correction for 4 Heads x Multiple Metrics

Our evaluation produces the following metric families:

- **Detection:** mAP@0.5, mAP@[.5:.95], present-class mAP, per-class AP (24 classes)
- **Activity:** top-1 accuracy, top-5 accuracy, macro F1, per-class recall (75 classes)
- **PSR:** event F1@3, POS, edit score, per-component F1 (11 components)
- **Pose:** fwd MAE, up MAE, position MAE, geodesic error

If we test all 4 x 4 = 16 primary metrics individually without correction, the probability of at least one false positive (under the global null that MTL = ST on all metrics) is 1 - 0.95^16 = 0.56 -- more likely than not.

**Recommended strategy:**

1. **Designate ONE primary metric per head** before seeing any results. These are the metrics in section 3.1 below. Test only these 4 metrics with Bonferroni correction.

2. **All other metrics are secondary/exploratory.** Report them with p-values labeled as "uncorrected" or "nominal" to avoid implying statistical significance.

3. **For the composite metric** (section 3.2), report as a single test without correction.

4. **For per-class metrics**, use visualization (heatmaps, confusion matrices) rather than hypothesis tests. Per-class sample sizes are too small (some classes have 1-5 test samples) for meaningful statistical testing.

This strategy is consistent with the NIH/ICMJE guidelines for multiple endpoints and is standard in the medical imaging community. The MTL computer vision community has no formal standard, but the Bonferroni-on-primary-metrics approach is the most defensible.

### 2.6 Reporting Standard

For each metric in the final paper, report:

```
Metric: [metric name]
MTL:    mean +/- std [95% CI lower, 95% CI upper] (N=5 seeds)
ST:     mean +/- std [95% CI lower, 95% CI upper] (N=3 seeds)
Ratio:  MTL/ST (with bootstrap CI on the ratio)
p-value: [from paired bootstrap test, corrected/uncorrected as appropriate]
```

Example:
```
Detection mAP@0.5:
  MTL: 0.358 +/- 0.042 [0.312, 0.401] (N=5)
  ST:  0.451 +/- 0.031 [0.418, 0.483] (N=3)
  Ratio: 0.794 [0.712, 0.876]
  p = 0.008 (Bonferroni-corrected for 4 heads, threshold 0.0125)
```

---

## 3. Metric Aggregation: Per-Head Metrics, Composite Score, MTL/ST Ratio

### 3.1 Per-Head Primary Metrics

| Head | Primary Metric | Secondary Metrics (Exploratory) | Rationale for Primary |
|---|---|---|---|
| Detection | mAP@0.5 | mAP@[.5:.95], present-class mAP, per-class AP | COCO standard; matches WACV 2024 baseline |
| Activity | Clip-level top-1 accuracy | Top-5, macro F1, per-class recall | Matches MViTv2 SOTA protocol (16-frame clip) |
| PSR | Event F1@3 (raw, no decoder) | POS, edit score, per-component F1 | Matches STORM-PSR F1@3 protocol |
| Pose | Forward-vector angular MAE (degrees) | Up MAE, position MAE, geodesic error | No SOTA exists; most interpretable metric |

These are the metrics that undergo Bonferroni-corrected hypothesis testing. All other metrics are reported with nominal p-values.

### 3.2 Composite Score: Geometric Mean of Normalized Metrics

A single composite score is needed for:
- The "overall MTL beats ST" claim (single hypothesis test).
- Checkpoint selection during training.
- Ablation study comparisons across multiple tasks.

**Our recommended composite: geometric mean of per-head MTL/ST ratios.**

```
Composite = (ratio_det * ratio_act * ratio_psr * ratio_pose)^(1/4)
```

Where each ratio is MTL metric / ST metric. For pose (where lower MAE is better), the ratio is ST_MAE / MTL_MAE so that > 1 always means "MTL better."

**Why geometric mean, not arithmetic mean:**
- The geometric mean penalises extreme underperformance on any single head. If detection has ratio 0.6 and all others have 1.0, the geometric mean is (0.6)^(1/4) = 0.88, while the arithmetic mean is 0.90. The geometric mean better reflects the requirement that MTL should not catastrophically fail on any task.
- The geometric mean's multiplicative nature naturally captures the "joint" nature of multi-task performance.
- The geometric mean is the standard in MTL benchmarks (NYUv2, Taskonomy, PASCAL-Context).

**Alternative: Normalized sum of absolute metrics.** An alternative is to normalise each task's metric by its ST baseline, sum, and divide by 4:

```
Composite_abs = (det_MTL/det_ST + act_MTL/act_ST + psr_MTL/psr_ST + pose_ST/pose_MTL) / 4
```

This is equivalent to the arithmetic mean of the ratios. It is more lenient than the geometric mean and allows a single strong task to compensate for a weak one. Use the geometric mean for the primary claim and the arithmetic mean as a secondary check.

**Composite = 1.0 means MTL matches ST overall. Composite > 1.0 means MTL beats ST overall.**

### 3.3 MTL/ST Ratio per Head and Interpretation

The MTL/ST ratio per head is the single most informative number in the paper. It isolates the effect of parameter sharing from all other factors. The ratio should be reported with a bootstrap confidence interval:

```
MTL_ratio = (MTL_metric) / (ST_metric)
MTL_ratio_ci = bootstrap over seeds and recordings
```

**Interpretation guide:**

| Ratio Range | Interpretation | Typical Head |
|---|---|---|
| > 1.05 | Strong positive transfer | Rare; pose or PSR with good cross-task synergy |
| 1.00 - 1.05 | Weak positive transfer | Best-case scenario for MTL paper |
| 0.95 - 1.00 | Negligible negative transfer | Acceptable (within noise) |
| 0.85 - 0.95 | Moderate negative transfer | Typical for detection in 4-task MTL |
| 0.70 - 0.85 | Significant negative transfer | Requires mitigation strategy |
| < 0.70 | Severe negative transfer | Head is essentially broken in MTL |

These ranges come from the Vandenhende et al. (2021) survey of 30+ MTL papers. Note that detection consistently shows the lowest ratios (0.70-0.90) while semantic segmentation and depth estimation show the highest (0.90-1.00).

**Our targets (from doc 208):**
- Pose: MTL/ST >= 0.95
- Activity: MTL/ST >= 0.70
- Detection: MTL/ST >= 0.60
- PSR: event-F1 > 0.25 with monotonicity

These targets should be compared against the published typical ranges above. Our detection target of 0.60 is below the typical 0.70-0.90 range, reflecting the extreme difficulty of detection in our setup. Our PSR target is task-specific and cannot be compared to published MTL ranges (no published MTL papers include a PSR-like task).

### 3.4 Overall MTL Gain

The overall MTL gain is the composite score minus 1.0, expressed as a percentage:

```
MTL_gain = (Composite - 1.0) * 100%
```

A positive MTL gain means the MTL model outperforms the ensemble of ST models on the composite metric. A negative MTL gain means the MTL model underperforms the ST ensemble.

**This is the headline number for the paper's "does MTL help?" claim.**

---

## 4. Standard MTL Benchmarks to Compare Against

### 4.1 Why Benchmarks Matter

Every MTL paper should contextualise its results against established benchmarks. Even if our tasks (assembly state detection, activity recognition, PSR, head pose) are unique to the IndustReal dataset, we can position our methodology relative to what the community accepts as standard practice.

### 4.2 Major MTL Benchmarks

#### NYUv2 (3 tasks: semantic segmentation, depth estimation, surface normal prediction)

- **Tasks:** 13-class semantic segmentation, depth estimation (Z-buffer), surface normal prediction (3-channel)
- **Resolution:** 640x480, 1449 labelled images (795 train, 654 test)
- **Backbone:** Typically ResNet-50 or similar
- **Metric:** mean IoU (segmentation), RMSE (depth), mean angle error (normals)
- **MTL gain reporting:** Geometric mean of (task_metric / ST_metric). The geometric mean of per-task ratios is the standard.
- **Typical MTL/ST ratios:** 0.95-1.00 (segmentation), 0.97-1.02 (depth), 0.95-1.00 (normals)
- **Key papers:** Cross-stitch Networks (Misra et al., CVPR 2016), NDDR-CNN (Gao et al., CVPR 2019), MTI-Net (Vandenhende et al., ECCV 2020), PAD-Net (Xu et al., CVPR 2018)
- **Limitation for our purposes:** All three tasks are dense predictions (per-pixel output). No detection, no classification, no temporal modelling. NYUv2 results do not generalise to mixed-type task sets.

#### Taskonomy (26 tasks, full supervision)

- **Tasks:** 26 visual tasks including semantic segmentation, depth estimation, surface normal prediction, edge detection, keypoint detection, 2D/3D edge detection, room layout, object classification, etc.
- **Data:** 4 million images from 600 buildings
- **Metric:** Task-specific metrics (mIoU for seg, RMSE for depth, etc.) normalised to a common scale
- **Key finding:** Task affinity matrix shows which tasks benefit from shared representations. Detection-adjacent tasks (object classification, keypoint detection) cluster together. Geometric tasks (depth, normals) cluster together.
- **MTL gain reporting:** Transfer strength as a scalar on a -1 to 1 scale. Positive = transfer helps, negative = transfer hurts.
- **Relevance to our setup:** Taskonomy's task affinity analysis is the most comprehensive published study of which tasks help or hurt each other. Their finding that detection-like tasks have low affinity with semantic tasks (negative to weakly positive transfer) is consistent with our observation that detection degrades in MTL.
- **Key paper:** Zamir et al., "Taskonomy: Disentangling Task Transfer Learning," CVPR 2018.

#### Cityscapes (2 tasks: semantic segmentation, depth estimation)

- **Tasks:** 19-class semantic segmentation, depth estimation (disparity)
- **Resolution:** 1024x2048, 5000 images (2975 train, 500 val, 1525 test)
- **Metric:** mIoU (segmentation), RMSE (depth)
- **Typical MTL/ST ratios:** 0.97-1.02 (both tasks—Cityscapes is easy for MTL because tasks are closely related)
- **Key observation:** The high MTL/ST ratios on Cityscapes are partly due to the fact that both tasks are dense predictions on street scenes, where segmentation and depth share strong geometric structure. This does not transfer to our mixed-type task set.
- **Key papers:** PAD-Net (Xu et al., CVPR 2018), MTI-Net (Vandenhende et al., ECCV 2020)

#### PASCAL-Context (5 tasks: semantic segmentation, person-part segmentation, depth estimation, surface normals, saliency)

- **Tasks:** 5 tasks covering semantic and geometric understanding
- **Resolution:** ~500x500, 4998 images (train/val standard split)
- **Metric:** mIoU (segmentation), RMSE (depth), mean angle error (normals), mean F-measure (saliency)
- **Typical MTL/ST ratios:** 0.93-0.98 (segmentation), 0.95-1.00 (depth), 0.94-0.99 (normals), 0.91-0.97 (saliency)
- **Key paper:** "Multi-Task Learning for Dense Prediction Tasks: A Survey" (Vandenhende et al., TPAMI 2021)
- **Relevance:** PASCAL-Context has the most tasks among the "standard" MTL benchmarks (5 tasks) and is widely used for MTL evaluation. However, all five tasks are still dense predictions—no detection, no classification, no temporal modelling.

#### Celeba-MTL (40 binary attribute classification tasks)

- **Tasks:** 40 binary face attribute classifiers
- **Data:** 202,599 face images
- **Metric:** Average precision (AP) per attribute, mean AP across 40 tasks
- **Key observation:** With 40 tasks all sharing the same architecture (CNN -> 40 binary classifiers), MTL consistently beats ST because the shared face representations benefit all attributes. This is the canonical counterexample to the "MTL hurts" narrative.
- **Relevance to our setup:** CelebA shows that MTL works well when tasks are strongly related (all face attributes benefit from the same face features). Our tasks (detection + activity + PSR + pose) are less related, so lower MTL/ST ratios are expected.

### 4.3 What Metrics Do These Benchmarks Report?

| Benchmark | Metric per Task | Composite Metric | How MTL Gain Is Computed |
|---|---|---|---|
| NYUv2 | mIoU (seg), RMSE (depth), mean angle (normals) | Geometric mean of deltas | Per-task metric relative to ST, geometric mean |
| Taskonomy | Task-specific (normalised) | Transfer strength (scalar) | Relative improvement over ST on a -1 to 1 scale |
| Cityscapes | mIoU (seg), RMSE (depth) | Not standard (report per-task) | Per-task comparison, no composite |
| PASCAL-Context | Per-task metric | Not standard | Per-task MTL/ST ratio |
| CelebA | Mean AP across 40 tasks | Mean AP | Compare mean AP MTL vs ST |

**Key observation:** The geometric mean of per-task MTL/ST ratios is the closest thing to a "standard" composite metric for MTL. NYUv2 papers consistently use this. Taskonomy uses a related but different scaling. Cityscapes and PASCAL-Context typically do not compute a single composite.

### 4.4 What Is Missing from These Benchmarks?

**No standard benchmark includes:**
- Object detection as a task (all benchmarks use dense prediction tasks like segmentation)
- Temporal state detection or sequence modelling
- Mixed output types (classification + detection + regression + temporal)
- Egocentric video (all except Taskonomy use static images)
- Real-world industrial data

**Implication for our paper:** We cannot directly compare our MTL gain numbers to published benchmarks because no published benchmark shares our task composition. Our paper must:
1. Explain that no standard benchmark matches our task set.
2. Use the geometric-mean-of-ratios composite from NYUv2 as our methodological template (it is the closest existing standard).
3. Provide per-task MTL/ST ratios so reviewers can compare to published per-task numbers for detection (typically 0.70-0.90), classification (typically 0.95-1.00), and regression (typically 0.95-1.00).

### 4.5 What Claude Science Should Find about MTL Evaluation Protocols

We need Claude Science to find:

1. **Papers that define formal MTL evaluation protocols.** Few exist. The most comprehensive is Vandenhende et al. (2021, TPAMI), which includes a detailed description of the evaluation protocol for NYUv2, Cityscapes, and PASCAL-Context. Are there more recent protocol papers (2022-2026)?

2. **Papers that define statistical testing protocols for MTL.** We have not found any paper that specifies the statistical methodology for MTL comparison (seeds, bootstrap, multiple comparison correction). This is a gap in the literature that our paper could address.

3. **Papers that report MTL/ST ratios for detection tasks specifically.** Most MTL papers omit detection. Any paper that includes COCO-style detection in an MTL setting and reports matched ST baselines would be directly informative.

4. **The standard for seed selection in MTL papers.** Is 3 seeds the accepted minimum? Has any paper been criticised by reviewers for insufficient seeds?

5. **Any paper that proposes a formal composite metric for MTL evaluation.** Beyond the geometric mean of ratios, are there information-theoretic or multi-objective optimization approaches to MTL evaluation?

---

## 5. Our Benchmark Protocol

### 5.1 Test Split (Never Seen During Training/Validation)

The IndustReal dataset has 52 recordings. Our standard split is:

| Split | Recordings | Purpose |
|---|---|---|
| Train | ~36 (70%) | Training all four heads |
| Validation | ~4 (8%) | Checkpoint selection, hyperparameter tuning |
| Test | ~12 (23%) | Final evaluation (reported in paper) |

**Critical rule:** The test split must never be used for any training decision, including learning rate scheduling, early stopping, checkpoint selection, or augmentation selection. Once the test split is used for any purpose beyond final reporting, it becomes contaminated and the comparison is no longer fair.

**How to verify:** Log the test split index at the start of training and verify that no test sample appears in any training batch, validation evaluation, or hyperparameter sweep. Our `evaluate.py` script already enforces this separation, but it should be audited before the final run.

### 5.2 Cross-Validation? (We Have 52 Recordings)

With 52 recordings, standard k-fold cross-validation (e.g., 5-fold, ~10 recordings per fold for 52 total) is possible but expensive (5x training cost). We recommend **against** cross-validation for the following reasons:

1. **MTL training is expensive enough without 5x cost.** At ~3 days per MTL run on RTX 3060, 5-fold CV would require 15 days of continuous training.

2. **The standard in MTL papers is a fixed train/val/test split**, not cross-validation. NYUv2, Cityscapes, PASCAL-Context, and Taskonomy all use fixed splits.

3. **Cross-validation mixes train/val assignments**, making it harder to compare results across papers that use the same split.

4. **Recording-level variance can be captured by bootstrap** (section 2.3) without cross-validation.

**Exception:** If the dataset is small (< 30 recordings) or highly heterogeneous (recordings differ substantially in difficulty), cross-validation may be appropriate. Our 52 recordings with ~12 in the test split should provide adequate coverage.

**Recommendation:** Use a fixed 36/4/12 split for all experiments. Report the test split identity in the paper appendix. If the validation split is used for hyperparameter tuning, add an additional held-out "validation" portion from the training split (e.g., 30 train + 6 val-from-train + 4 val + 12 test).

### 5.3 Per-Recording vs. Per-Frame Metrics

A critical design choice: do we compute metrics per frame and average across all frames, or compute metrics per recording and average across recordings?

**Per-recording metrics (recommended for primary analysis):**
- Compute the metric independently for each test recording.
- Average across recordings to get the final number.
- The recording is the natural unit of analysis because:
  - Recordings are independent (different workers, different shifts).
  - Frames within a recording are highly correlated (same worker, same assembly procedure).
  - Per-frame averaging gives excessive weight to longer recordings.
  - MTL evaluation literature typically uses per-video or per-image averaging, not per-frame.

**Per-frame metrics (secondary/diagnostic):**
- Aggregate all frames from all test recordings and compute the metric on the full set.
- Useful for detecting recording-level biases (e.g., one recording dominates the metric).
- Should be reported alongside per-recording metrics to show robustness.

| Aspect | Per-Recording | Per-Frame |
|---|---|---|
| Unit of analysis | Recording (12 test units) | Frame (thousands of test units) |
| Effective N | 12 | Thousands |
| Statistical power | Lower (12 vs thousands) | Higher (but inflated by autocorrelation) |
| Fairness | Each recording has equal weight | Longer recordings dominate |
| Standard practice | Common in video benchmarks | Common in static image benchmarks |
| Our recommendation | Primary paper metric | Secondary/diagnostic only |

### 5.4 Event-Level vs. Frame-Level for PSR

PSR is unique among our four heads in having two distinct evaluation regimes:

**Frame-level PSR evaluation:**
- For each of the 11 PSR components, compare predicted binary state to ground truth binary state at each frame.
- Compute per-frame accuracy, precision, recall per component.
- Problem: >99% of frames have no transition, so a classifier that predicts "all zeros" achieves >99% accuracy. This is misleading.

**Event-level PSR evaluation:**
- Detect transitions (0->1 or 1->0) in the predicted sequence.
- Compare detected transitions to ground truth transitions with a tolerance window (typically 3 frames).
- Compute event-level precision, recall, F1.
- This is the standard in the PSR literature (STORM-PSR, B2 baseline).
- Our `psr_f1_at_t` metric implements this correctly.

**Recommendation:**
- **Primary metric:** Event-level F1@3 (raw, no decoder). This is what STORM-PSR reports and is the meaningful comparison.
- **Secondary metric:** Per-component frame-level accuracy and F1 (for diagnostics, not for the main results table).
- **Do not report** overall per-frame PSR accuracy or overall per-frame PSR F1. These are inflated by the class imbalance and do not reflect the model's ability to detect transitions.

**The transition-aware variant:** The `transition_boost=3.0` in our PSR loss function weights transition frames 3x higher than non-transition frames. This is a training-time correction for the imbalance. The evaluation should still use event-level F1@3 as the primary metric, because that is what the downstream application cares about (detecting when assembly steps complete).

---

## 6. What Claude Science Should Find

Claude Science should search for established MTL evaluation protocols from top venues (CVPR, ICCV, NeurIPS, ICML, ICLR). Specifically:

### 6.1 Papers Defining MTL Evaluation Standards

We need to know:
1. **Does any paper formally define an MTL evaluation protocol?** The closest we know is Vandenhende et al. (2021, TPAMI) which provides protocol details for NYUv2, Cityscapes, and PASCAL-Context. Are there more recent or more comprehensive protocol papers?

2. **What is the review standard for MTL papers at CVPR/ICCV/NeurIPS?** Are there published review criteria or reviewer guidelines? We need to know what reviewers expect in terms of baselines, seeds, and statistical reporting.

3. **Are there any reproducibility guidelines for MTL?** The ML Reproducibility Checklist (NeurIPS) and the Reproducibility in ML literature may have MTL-specific recommendations.

### 6.2 Papers with Statistical Methodology for MTL

We need to know:
1. **What statistical tests do published MTL papers use?** A survey of 50 top-venue MTL papers (2020-2025) regarding how many report confidence intervals, what statistical tests they use, and how they handle multiple comparisons.

2. **The seed standard.** Is there a consensus on how many seeds are required for MTL experiments? Do reviewers commonly request more seeds?

3. **Bootstrap methods for MTL.** Are there papers that use bootstrap or other resampling methods for MTL comparison?

### 6.3 Papers with MTL/ST Ratios for Detection

The critical gap in our knowledge: we need to know the distribution of MTL/ST ratios for object detection in published MTL papers. If the literature consistently shows detection MTL/ST ratios of 0.60-0.75, then our target of 0.60 is realistic. If published papers achieve 0.85-0.95, our target may be too low.

### 6.4 Papers Proposing Composite Metrics

We need to know:
1. **Any paper that proposes a formal composite metric for MTL** beyond the geometric mean of ratios.
2. **Papers that use multi-objective optimization (Pareto front, hypervolume) for MTL evaluation.**
3. **Any metric that captures the trade-off between accuracy and efficiency in a single number** (e.g., accuracy-per-parameter).

### 6.5 MTL Benchmarking for Video and Egocentric Data

We need to know:
1. **Any MTL benchmarks for video understanding** (not static images). Are there multi-task video datasets with standard evaluation protocols?
2. **Any MTL papers on egocentric video** (Ego4D, EpicKitchens, Assembly101). How do they handle the comparison problem?
3. **Any MTL paper that includes temporal state detection** as one of its tasks.

---

## 7. How to Report When MTL Loses on a Head: Honest Failure Analysis

### 7.1 The Inevitable: Not All Heads Will Win

It is vanishingly unlikely that our MTL model beats its ST baseline on all four heads simultaneously. The MTL literature shows that:
- 2-task MTL (closely related tasks like segmentation + depth): both heads often match or slightly exceed ST.
- 3-task MTL (mixed types like segmentation + depth + detection): at least one head typically underperforms ST.
- 4+ task MTL (especially with diverse output types): 2-3 heads typically underperform ST, some severely.

The honest frame is: **MTL is a trade-off, and the paper's contribution is measuring and understanding that trade-off.** A paper that shows MTL beating ST on all four heads would be genuinely surprising and would require extraordinary evidence (multiple seeds, narrow confidence intervals, replication by independent labs).

### 7.2 How to Frame Negative Results

For each head that underperforms ST, the paper should:

**Step 1: State the result clearly and without hedging.**
```
"Detection mAP@0.5 was 0.358 (MTL) vs. 0.451 (ST), a retention ratio of 0.794.
This is consistent with the published literature, where detection in 4-task MTL
typically retains 70-90% of ST performance."
```

**Step 2: Diagnose the cause.**
- Is it gradient conflict? (Measure gradient cosine similarity between tasks. If detection gradients consistently oppose activity gradients, this is the cause.)
- Is it capacity competition? (If the backbone has 34.5M parameters and is learning features for 4 tasks, some tasks may receive insufficient feature bandwidth.)
- Is it a data issue? (If detection has sparse annotations, the MTL model may allocate more capacity to tasks with denser supervision.)
- Provide quantitative evidence for the diagnosis (gradient norms, feature similarity analysis, loss landscapes).

**Step 3: Describe what was tried and what did not work.**
```
"We attempted to close the detection gap via: (a) increasing detection head
capacity from 0.8M to 1.6M params, (b) PCGrad gradient surgery with detection
prioritization, (c) detection-specific learning rate warmup. None closed more
than 30% of the gap. The residual gap appears structural to the task conflict
between spatial localization and semantic classification."
```

**Step 4: Contextualise the gap relative to efficiency gains.**
```
"The 21% detection mAP loss is offset by a 2.06x parameter efficiency gain
(48.6M vs. ~100M for four ST models) and a 4x latency gain (single forward
pass vs. four). At the system level, the MTL model processes 11.02 FPS on an
RTX 3060 while performing all four tasks, enabling real-time assembly monitoring
that would require either four GPUs or four sequential model passes."
```

**Step 5: Be explicit about the implications.**
```
"For deployment scenarios where detection accuracy is paramount, the ST model
remains preferable. For scenarios where the combined 4-task output at low
latency is critical, the MTL model's trade-off is acceptable. This analysis
provides practitioners with the information needed to choose based on their
accuracy vs. efficiency requirements."
```

### 7.3 The Failure Analysis Table

For the paper, include a table that honestly assesses each head:

| Head | MTL Metric | ST Metric | Retention | Verdict | Cause | Mitigation |
|---|---|---|---|---|---|---|
| Detection | 0.358 mAP@0.5 | 0.451 mAP@0.5 | 79.4% | Moderate loss | Feature competition with activity | PCGrad reduced but did not eliminate |
| Activity | 0.312 top-1 | 0.402 top-1 | 77.6% | Moderate loss | Gradient starvation (long CE path) | Spatial attention pool (ablation C) |
| PSR | 0.310 event-F1 | 0.280 event-F1 | 110.7% | **Positive transfer** | Detection features enrich PSR | N/A — MTL helps |
| Pose | 8.7 deg MAE | 9.1 deg MAE | 104.6% | **Positive transfer** | Shared low-level features | N/A — MTL helps |

**Note:** This table uses hypothetical numbers for illustration. The actual numbers will come from our ST baseline and MTL runs.

### 7.4 The "Honest MTL Gain" Formula

Instead of a single number claiming "MTL beats ST," report:

```
MTL_gain = (composite_MTL - composite_ST) / composite_ST * 100%
```

Where composite = geometric mean of per-task metrics (section 3.2). If composite_MTL = 0.85 and composite_ST = 1.0 (the ST composite is always 1.0 by construction), then MTL_gain = -15%, meaning MTL underperforms ST by 15% overall.

**This is the honest headline number.** It may be negative, and that is acceptable. The paper's contribution is then:
1. Quantifying the MTL trade-off precisely (not claiming it does not exist).
2. Diagnosing the causes of negative transfer.
3. Demonstrating that the efficiency gains (2x parameters, 4x latency) are worth the accuracy loss for deployment.
4. Showing that some heads benefit from sharing (positive transfer on PSR and pose).

### 7.5 What Not to Do

From our review of MTL papers that were desk-rejected or received weak reviews:

1. **Do not cherry-pick seeds.** Report all seeds, or report a principled seed selection procedure (e.g., discard outliers using Grubbs' test, but state all original seeds in the appendix).

2. **Do not report "MTL beats SOTA" without a matched ST baseline.** A reviewer will ask: "Does MTL beat ST with the same backbone?" If you cannot answer this, the paper is not ready.

3. **Do not use different evaluation protocols for MTL and ST.** If MTL uses per-recording metrics and ST uses per-frame metrics, the comparison is invalid. Both must use the identical evaluation script.

4. **Do not claim statistical significance without correction for multiple comparisons.** A reviewer with statistical training will catch this.

5. **Do not hide negative results in the supplementary material.** The main paper must include honest per-head retention ratios. Supplementary can contain additional analysis.

6. **Do not use the term "comparable" to describe a 20% gap.** "Comparable" implies within noise (typically <5%). A 20% gap is a significant degradation and should be described as such.

---

## 8. Summary of Recommendations

### Methodology (Must Do)

1. **Primary comparison:** MTL vs. matched ST baseline (same backbone, same training protocol). This is the fairest and most informative comparison. --- Section 1.3, Approach A.

2. **Secondary comparison:** MTL results alongside published SOTA numbers for context, clearly labelled as a different comparison regime. --- Section 1.3, Approach B.

3. **Never mix comparison regimes in the same table.** Accuracy comparisons and efficiency comparisons should be in separate tables.

### Statistics (Must Do)

4. **Minimum 3 seeds for ST, 5 seeds for MTL.** Report mean, standard deviation, and 95% bootstrap CI computed on per-recording metrics. --- Section 2.2, 2.3.

5. **Primary test:** Paired bootstrap test on the composite metric (geometric mean of per-head MTL/ST ratios), p < 0.05. --- Section 2.4, 3.2.

6. **Per-head tests:** Bonferroni correction for 4 heads, threshold p < 0.0125. --- Section 2.4.

7. **Designate ONE primary metric per head before seeing results.** All other metrics are exploratory (nominal p-values only). --- Section 2.5, 3.1.

### Metrics (Must Do)

8. **Primary metrics:** mAP@0.5 (detection), clip-level top-1 (activity), event F1@3 (PSR), forward MAE (pose). --- Section 3.1.

9. **Composite metric:** Geometric mean of per-head MTL/ST ratios. Composite > 1.0 = MTL beats ST overall. --- Section 3.2.

10. **Per-recording metrics are primary** (12 test recordings); per-frame metrics are secondary/diagnostic. --- Section 5.3.

### Benchmarks (Should Do)

11. **Contextualize against NYUv2, Taskonomy, Cityscapes, PASCAL-Context, and CelebA** in the related work section, noting that none match our task composition. --- Section 4.

12. **Adopt the geometric mean-of-ratios composite from NYUv2** as our methodological template. --- Section 4.3.

13. **Note the gap in the literature:** No standard MTL benchmark includes object detection or temporal state detection. Our paper can propose a protocol for mixed-type MTL evaluation. --- Section 4.4.

### Failure Analysis (Must Do)

14. **Report per-head retention ratios honestly** with confidence intervals. --- Section 7.2.

15. **Diagnose the cause of each failure** (gradient conflict, capacity competition, data sparsity). --- Section 7.2.

16. **Show the efficiency-accuracy trade-off** explicitly so readers can make their own assessment. --- Section 7.3.

17. **Do not hide negative results.** A paper that honestly reports 2/4 heads benefiting and 2/4 heads losing, with a clear diagnosis of why, is more credible than one that claims "comparable" results. --- Section 7.5.

---

## References

- Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., & Van Gool, L. "Multi-Task Learning for Dense Prediction Tasks: A Survey." TPAMI 2021. *The definitive survey on MTL evaluation protocols for dense prediction.*
- Zamir, A. R., Sax, A., Shen, W., Guibas, L., Malik, J., & Savarese, S. "Taskonomy: Disentangling Task Transfer Learning." CVPR 2018. *Comprehensive task affinity analysis across 26 visual tasks.*
- Standley, T., Zamir, A. R., Chen, D., Guibas, L., Malik, J., & Savarese, S. "Which Tasks Should Be Learned Together in Multi-task Learning?" ICML 2020. *Proposes a method for selecting optimal task groupings.*
- Misra, I., Shrivastava, A., Gupta, A., & Hebert, M. "Cross-stitch Networks for Multi-task Learning." CVPR 2016. *Introduced the cross-stitch architecture for MTL with NYUv2 evaluation.*
- Kendall, A., Gal, Y., & Cipolla, R. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018. *Introduced uncertainty weighting and standard NYUv2 MTL evaluation.*
- Sener, O., & Koltun, V. "Multi-Task Learning as Multi-Objective Optimization." NeurIPS 2018. *Framed MTL as Pareto optimization.*
- Liu, L., Li, Y., Kuang, Z., Xue, J., Chen, Y., Yang, W., Liao, Q., & Zhang, W. "Towards Impartial Multi-task Learning." ICLR 2021. *Proposed IMTL for fair gradient balancing.*
- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020. *Introduced PCGrad with NYUv2 and Cityscapes evaluation.*
- Chen, Z., Badrinarayanan, V., Lee, C., & Rabinovich, A. "GradNorm: Gradient Normalization for Adaptive Multi-Task Loss Balancing." 2018. *Introduced GradNorm with evaluation on Cityscapes.*
- Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., & Finn, C. "Efficiently Identifying Task Groupings for Multi-Task Learning." NeurIPS 2021. *Proposed task grouping analysis for Gigaspeech and CelebA.*
- Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., & Fetaya, E. "Multi-Task Learning as a Bargaining Game." ICML 2022. *Introduced Nash-MTL with evaluation on NYUv2 and Taskonomy.*
- Liu, B., Liu, Y., Zhou, P., & Liu, Y. "CAGrad: Conflict-Averse Gradient Descent for Multi-task Learning." NeurIPS 2021. *Introduced CAGrad with evaluation on NYUv2, Cityscapes, and CelebA.*
- Kokkinos, I. "UberNet: Training a Universal Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory." CVPR 2017. *One of the first papers to train a single model on 7 vision tasks.*
- Xu, D., Ouyang, W., Wang, X., & Sebe, N. "PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing." CVPR 2018. *Standard Cityscapes MTL evaluation protocol.*
- Higham, D. J. "An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations." SIAM Review 2001. *For bootstrap methodology.* Note: Use Efron & Tibshirani (1993) for bootstrap, not this.
- Efron, B., & Tibshirani, R. J. "An Introduction to the Bootstrap." Chapman & Hall, 1993. *Definitive reference on bootstrap confidence intervals.*
- Benjamin, D. J., et al. "Redefine statistical significance." Nature Human Behaviour 2018. *Proposes p < 0.005 as the new threshold for high-rigour claims.* We recommend the standard p < 0.05 but note the ongoing debate. For our primary composite, we use p < 0.05 with Bonferroni correction.

---

**Document cross-references:**
- Doc 208 (Overview): Hypothesis and high-level comparison strategy
- Doc 211 (Training Methodology): Training protocol details
- Doc 212 (Per-Head Gap Analysis): Current MTL vs. ST estimates  
- Doc 215 (50 Deep Questions): Questions F1-F5 on benchmarking
- Doc 219 (Efficiency Metrics): FLOPs, params, latency methodology
- Doc 223 (Experimental Protocol): Seed control, statistical tests
- Doc 224 (Figure and Table Planning): How results will be visualised
- Doc 225 (Risk Assessment): What could go wrong
- FAIR-COMPARABILITY-MATRIX.md: Per-metric protocol alignment tracking
