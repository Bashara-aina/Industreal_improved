# AAIML 2027 -- Honest Weaknesses and Preemptive Rebuttals

**Paper**: POPW: A Multi-Task Deep Learning Framework for Assembly Verification

---

## 1. Detection Performance (Most Likely Rejection Reason)

**Honest statement**: Our present-class mAP50 of 0.34 is 59% lower than YOLOv8m's 0.838 on the same benchmark. At standard threshold 0.5, the false positive rate is 0.12. This means the detector makes a mistake roughly every 8 frames.

**Why this is happening**: The 24 ASD classes are extremely fine-grained (11-bit binary state encodings). Many class pairs differ by a single component state. This is a fundamentally harder problem than detecting 24 distinct assembly states. YOLOv8m uses 260K synthetic training images plus COCO pretraining; we use ImageNet pretraining and real data only.

**What we can say in rebuttal**:
- 70% of errors are 1-bit Hamming-adjacent: the model identifies the coarse state correctly and confuses only single-component transitions
- At threshold 0.7, FPR drops to 0.04 (usable for high-precision applications)
- The bootstrap 95% CI [0.31, 0.37] shows the result is stable, not a lucky run
- No synthetic data augmentation was used -- this is a lower bound, not an upper bound
- Human inter-annotator agreement on 11-bit state encoding is not 100% either

**What to add before submission**:
- Precision-recall curves at multiple thresholds
- Operating point analysis: "at threshold X, FPR=Y, recall=Z"
- Human baseline for fine-grained state discrimination (if measurable)
- Three-seed variance

---

## 2. Single-Dataset Evaluation

**Honest statement**: All vision results are on one dataset (IndustReal). We have not validated on IKEA ASM, Assembly101, MECCANO, or any other assembly benchmark.

**Why this is happening**: No other dataset supports all five tasks simultaneously. IndustReal is the only dataset with annotation for detection, pose, activity, and procedure steps together.

**What we can say in rebuttal**:
- The architecture is dataset-agnostic (ConvNeXt-Tiny + FPN + task heads is standard)
- Transfer learning via ImageNet pretraining provides a degree of cross-domain robustness
- We commit to evaluating on held-out factory data and IKEA ASM in the extended version
- The five-task combination is unique; no prior work demonstrates this capability on any dataset

**What to add before submission**:
- If possible, report PSR-only or activity-only results on a second dataset (even partial)
- Show that the backbone initialization generalizes by reporting detection-only results on COCO-style benchmarks

---

## 3. Activity Recognition at 18.3% Top-1

**Honest statement**: Our activity classifier picks the correct action in fewer than 1 in 5 frames. This seems low for a system that triggers blockchain payments based on activity.

**Why this is happening**: 74 atomic action classes is a large label space for per-frame classification. Many actions are visually similar (e.g., "pick up wrapper" vs "pick up dumpling"). The 16-frame temporal bank is relatively short.

**What we can say in rebuttal**:
- 14x above chance baseline (1/74 = 1.35%)
- Top-5 accuracy of 41.2% means the correct label is in the top 5 nearly half the time
- Per-frame activity recognition is not the downstream task -- PSR uses temporal smoothing over the full step (5-30 seconds, 24-144 frames)
- With temporal smoothing, per-step accuracy is substantially higher (add measurement)
- FiLM conditioning improves activity recognition by 2.2pp (p=0.032), demonstrating the multi-task benefit

**What to add before submission**:
- Per-class accuracy analysis (which activities are confused?)
- Temporal smoothing baseline: majority vote over 5-30 second windows
- Confusion matrix for activities (74x74, grouped by action type)
- Activity recognition evaluated at the step level, not frame level

---

## 4. Small Pilot Size

**Honest statement**: 20 workers for 2 weeks at one factory (dimsum production). This is not generalizable to all manufacturing environments.

**Why this is happening**: Recruiting factory workers for on-site studies is logistically challenging and expensive. 20 workers is standard for HCI pilot studies and provides sufficient statistical power for large-effect comparisons.

**What we can say in rebuttal**:
- The NASA-TLX reduction (p=0.04) has adequate statistical power for a paired t-test with n=20 if effect size is moderate-to-large (Cohen's d > 0.5)
- The effect size (10.4% reduction, from 65.2 to 58.4) is practically meaningful
- Demographics (age 22-58, 6.3 years mean experience, 12F/8M) span a useful range
- Zero opt-outs across 20 workers is compelling evidence of acceptability
- We frame the pilot as "proof-of-concept validation" not "definitive deployment study"

**What to add before submission**:
- Effect sizes (Cohen's d) for all significant results
- Power analysis: "with n=20, we can detect d>0.66 at alpha=0.05 with 80% power"
- Explicit limitations section bounding generalizability claims

---

## 5. Blockchain Motivation

**Honest statement**: The paper does not compare against a simpler non-blockchain alternative (signed audit log, append-only database, third-party notary). The blockchain section may seem like a solution in search of a problem.

**Why we chose blockchain**: Worker and employer have conflicting incentives. The employer controls the database. A blockchain-based payment record cannot be unilaterally modified. This provides transparency without requiring the worker to trust the employer's database integrity.

**What we can say in rebuttal**:
- "Verifiability without trust" is a well-established blockchain use case for multiparty systems with conflicting incentives
- x402 micropayments are the only standardized protocol for HTTP 402 blockchain payments
- The cost analysis ($0.0002-$0.001/tx gas, $799 3-year TCO) demonstrates blockchain is economically feasible even at microtransaction scale
- Add a comparison baseline: "A signed audit log would cost ~$0 but requires the worker to trust the employer's key management"

**What to add before submission**:
- Direct comparison to alternative approaches (signed log, trusted third party, transparent database)
- Quantify the trust tradeoff: "database: worker must trust employer not to modify records; blockchain: records are immutable"
- Add a subsection titled "Why Blockchain?" that addresses this directly

---

## 6. Ethical Framework Implementation Gaps

**Honest statement**: Three of five IEEE 7005-2021 principles are marked as (P) = design principle, not implemented. Informed consent, transparency, and fairness are planned but not yet deployed.

**Why this is happening**: The pilot was scoped to validate technical feasibility and initial worker acceptance. Full compliance requires software features not yet built into the POPW dashboard.

**What we can say in rebuttal**:
- (P) items are actively scoped for V2 deployment
- The two implemented principles (data governance via edge-only processing, accountability via blockchain log) address the most critical IEEE 7005 requirements
- Transparency via real-time earnings dashboard is partially implemented (blockchain payments are visible)
- The paper is transparent about what is and is not implemented

**What to add before submission**:
- Timeline for (P) item implementation
- Clarify that (P) = planned for V2, not "we will never implement this"
- Add a sentence: "The IEEE 7005 mapping shows the current compliance status; items marked (P) do not affect the technical contribution but are required for production deployment"

---

## 7. Lack of Comparison to SOTA on Each Individual Task

**Honest statement**: We compare to YOLOv8m (detection), but do not directly compare body pose, head pose, activity, or PSR to their respective specialist state-of-the-art on the same data and splits.

**Why this is happening**: Running four separate SOTA models on the same dataset requires reproducing their training pipelines, which is weeks of work per model. The paper priority is demonstrating multi-task feasibility, not beating individual SOTA models.

**What we can say in rebuttal**:
- The SOTA comparison table acknowledges this gap and frames POPW as a multi-task alternative, not a specialist replacement
- Head pose error (9.1 deg) is within the typical range of monocular head pose estimators on consumer hardware
- PSR performance is compared to STORM-PSR's published results (different data splits prevent direct comparison)
- We commit to adding direct SOTA comparisons in the camera-ready version for at least two tasks

**What to add before submission**:
- If possible, run YOLOv8m on the same test split and report its per-class performance
- Report pose estimation PCK or OKS against a simple baseline

---

## 8. No Real-Time Deployment Validation

**Honest statement**: The paper reports 4.8 FPS on an RTX 3060 but does not validate end-to-end latency in a production assembly line. The blockchain 537ms latency is on devnet, not mainnet.

**Why this is happening**: The pilot was designed for feasibility, not production engineering. Real-time constraints were not the primary measurement target.

**What we can say in rebuttal**:
- Assembly steps take 5-30 seconds; even 3x devnet latency is negligible
- 4.8 FPS captures 24-144 frames per step, sufficient for verification
- TensorRT optimization could double frame rate without hardware change

**What to add before submission**:
- Add a "Deployment Considerations" subsection in Discussion
- Estimate production latency: "on mainnet, we expect 800-1500ms based on published Solana confirmation times"
- Discuss buffering strategies for real-time deployment

---

## Weakness Severity Matrix

| Weakness | Severity | Can Fix Before Submission? | Impact on Acceptance | 
|----------|----------|---------------------------|---------------------|
| Detection performance low | HIGH | Partially (operating point analysis) | Could cause rejection |
| Single dataset | MEDIUM | Partially (add partial cross-dataset) | Reduces score |
| Activity 18.3% | MEDIUM | Yes (temporal analysis) | Moderate concern |
| Small pilot | LOW | No (pilot complete) | Minor concern |
| Blockchain motivation | MEDIUM | Yes (add comparison section) | Moderate concern |
| Ethics gaps | LOW | Yes (add timeline) | Minor concern |
| No SOTA per-task comparison | MEDIUM | Partially (add what's feasible) | Moderate concern |
| No production validation | LOW | No | Minor concern |
