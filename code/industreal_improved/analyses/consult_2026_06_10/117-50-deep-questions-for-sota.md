# 50 Deep Questions for SOTA

**Target model:** Opus
**Purpose:** Each question is answerable in a way that could move a metric by 0.05+ or close a SOTA gap.
**Context:** POPW multi-task assembly verification system on IndustReal dataset, 4-task ConvNeXt-Tiny (28.6M backbone, 46.5M total params), $299 RTX 5060 Ti 16GB.
**Evidence discipline:** Each question cites file:line, paper Table X.Y, or training log values. Nothing is invented.
**Output structure:** 10 categories x 5 questions = 50 questions, each with Context, Question, Why This Matters, Constraints, Hypothesis, Validation sections.

---

# Category 1: Detection Head -- Closing the YOLOv8m Gap

Our RetinaNet-style detection head (5.3M params, 11.8% of total trainable) achieves det_mAP50=0.317, which is 62% below YOLOv8m's published 0.838 (`111-overview.md:175-190`). However, we operate at 1/6th GPU cost with 3 extra tasks simultaneously. The five questions in this category explore architectural (Soft-NMS, BiFPN, head depth) and training (OHEM ratio, FocalLoss gamma) changes that could close 10-30% of this gap without changing the backbone or data. Each question targets a specific component of the detection pipeline that differs from YOLOv8m's proven design, and each is independently testable.

## Q1. Soft-NMS vs Standard NMS for 24-Class ASD on RetinaNet Head

**Context:** Our detection head uses standard greedy NMS (`DET_EVAL_NMS_IOU_THRESH=0.5`) after the RetinaNet-style cls+reg subnets. At epoch 11, det_mAP50=0.317 (COCO-24) and det_mAP50_pc=0.506 (present-class) (`111-overview.md:683-687`, `112-training-metrics-deep-dive.md:497-514`). YOLOv8m achieves 0.838 mAP@0.5 on the same dataset with a decoupled head, FPN+PAN neck, and COCO pretrain (`114-comparability-vs-4-papers.md:146-155`). 

The 24 ASD classes represent binary component codes with inherent inter-class overlap. For example, channels 9-12 form a cluster where binary codes differ by exactly 1-2 bits:

| Channel | Binary Code | AP | GT Instances | Similar To | Bits Diff |
|---------|------------|----|-------------|-----------|----------|
| 9 | 11110111100 | 0.886 | 20 | Channel 10 | 1 |
| 10 | 11110111110 | 0.872 | 57 | Channel 9/12 | 1-2 |
| 11 | 11110110001 | 0.545 | 24 | Channel 10 | 3 |
| 12 | 11110111101 | 0.368 | 16 | Channel 10 | 1 |
| 22 | 11101111111 | 0.063 | 28 | Channel 10 | 2 |

Source: `116-winning-aaiml-synthesis.md:171-186`. Channel 22 differs from channel 10 (AP=0.872) by exactly 2 bits. Under standard greedy NMS, a high-scoring detection for channel 10 at IoU>0.5 likely suppresses a lower-scoring but correct detection for channel 22. The three near-zero-AP channels (16: 0.000, 19: 0.000, 22: 0.063) are all transitional states adjacent to high-AP neighbors in binary code space.

**Standard NMS (current):** keep = nms(boxes, scores, iou_threshold=0.5) -- hard suppression: detection j is COMPLETELY removed if IoU(best, j) > 0.5, regardless of class.

**Soft-NMS (proposed):** for each detection j with IoU > 0.5 with best detection i: scores[j] *= (1.0 - iou(i, j)). Linear decay: detection j with IoU=0.6 gets scores[j] *= 0.4 (retained with lower confidence). All detections above final threshold are kept. Implementation: <20 lines in evaluate.py.

**Question:** Would replacing greedy NMS with Soft-NMS (Bodla et al., 2017, linear re-scoring with sigma=0.5) recover the suppressed detections for ASD codes that differ by 1-2 bits (channels 9/10/11/12 cluster, channels 7/6/4 cluster)? Specifically, does the hard suppression of standard NMS directly explain the 0.000 AP on channels 16 (9 GT) and 19 (10 GT), where detections for these transitional states are being incorrectly suppressed by higher-scoring predictions from neighboring complete states?

**Why this matters:** Soft-NMS is a zero-parameter, inference-only change requiring no retraining, no code architecture changes, and no additional data. If it recovers even 3 of the 9 zero-GT channels to measurable AP (channels 16, 19, 22 from 0.000/0.063 to 0.15+), the mAP50_pc would increase from 0.506 by an estimated 0.03-0.05. This is a 6-10% relative improvement from a ~30-minute code change. Paper 1 uses standard NMS throughout (`114-comparability-vs-4-papers.md:141-146`) -- no IndustReal paper has evaluated Soft-NMS for the ASD taxonomy. The ASD binary-code space has inherent inter-class proximity that makes it especially susceptible to standard NMS suppression, so this finding would be both novel and task-specific.

**Constraints:** Must run on RTX 3060 (idle per `111-overview.md:249-254`). Cannot modify training or retrain -- uses existing epoch 11 checkpoint. Inference-only change in `evaluate.py` evaluation post-processor. Time: ~30 minutes to implement and run.

**Hypothesis:** Standard NMS suppresses at least 3 of the near-zero-AP classes by 0.05-0.20 AP. Soft-NMS recovers these:
- Channel 22 gains +0.10 to +0.20 AP (distinguish from channel 10's high-confidence predictions)
- Channel 16 gains +0.05 to +0.15 AP (distinguish from channel 15)
- Channel 19 gains +0.05 to +0.10 AP (distinguish from channel 18)
- Channels 9/10/11/12 cluster gains +0.01 to +0.03 each (reducing within-cluster suppression)
- Total mAP50 increase: 0.317 to 0.33-0.35
- Total mAP50_pc increase: 0.506 to 0.52-0.54
- The effect is largest for classes with adjacent high-AP neighbors in binary code space.

**Validation:** Run the existing epoch 11 validation set through the evaluation pipeline with `DET_EVAL_NMS_IOU_THRESH=0.5` replaced by Soft-NMS (linear penalty, sigma=0.5). Compare per-class AP for all 24 channels, focusing on channels 16/19/22/9/10/11/12. Measure delta in mAP50 and mAP50_pc. If delta < 0.01 mAP50, reject the hypothesis and document that standard NMS is not the bottleneck for these classes. Run simultaneously with D3 (full eval, EVAL_MAX_BATCHES=0, `115-execution-plan-to-sota.md:400-421`) to confirm on the full 38K-frame validation set rather than the 250-batch subsample.

---

## Q2. OHEM Ratio Sweep for Low-GT-Frame Detection

**Context:** The detection head uses OHEM with `DET_OHEM_RATIO=2.0` and `DET_OHEM_MIN_NEG=32` (`112-training-metrics-deep-dive.md:688-691`, `config.py:748-752`). This was reduced from 5.0/128 in F3 (`113-all-fixes-chronicle.md:43-44`). Per-class GT counts span a 15x range: channel 7 has 74 instances, channel 21 has 5 instances (`116-winning-aaiml-synthesis.md:171-186`). Only 17.89% of training frames carry GT boxes (`111-overview.md:139`). **Effective OHEM ratio calculation:**
For a frame with P positive anchors (matched to GT boxes):
  OHEM selects: min(P * OHEM_RATIO, OHEM_MIN_NEG) negatives = min(P * 2.0, 32) negatives
  For P=1: min(2, 32) = 32 negatives (effective ratio = 32:1)
  For P=5: min(10, 32) = 32 negatives (effective ratio = 6.4:1)
  For P=16: min(32, 32) = 32 negatives (effective ratio = 2:1, never reached)
Since most frames have 1-5 object instances, the `min_neg=32` floor dominates 80%+ of batches. The effective ratio is far from the intended 2:1.

**Question:** Given that the `min_neg=32` floor creates an effective 32:1 negative-to-positive ratio for the majority of training frames (those with 1-5 GT boxes), is this aggressive ratio starving positive anchor learning and causing the anomalously low AP on channels 6 (29 GT, 0.265 AP), 12 (16 GT, 0.368 AP), and 22 (28 GT, 0.063 AP)? Would reducing `DET_OHEM_MIN_NEG` from 32 to 8 (floor effective ratio at 8:1) improve per-class AP for mid-frequency classes by 0.05-0.15 each?

**Why this matters:** The OHEM configuration affects every training batch. At 32:1 ratio, each positive anchor competes against 32 negatives for gradient signal. For rare classes, this is particularly harmful. Paper 1's YOLOv8m uses FocalLoss without OHEM (`114-comparability-vs-4-papers.md:196-224`). Source: `112-training-metrics-deep-dive.md:688-691`.

**Constraints:** Requires a 25-epoch ablation on RTX 3060 (available). Cannot change OHEM mid-training. Time: 3-4 days.

**Hypothesis:** The `min_neg=32` floor dominates 80%+ of batches. Reducing to `min_neg=8` increases per-class AP for channels 6, 12, 22 by 0.05-0.15 each. High-frequency channels 7, 9, 10 regress slightly (-0.01 to -0.03). At `min_neg=4`, rare classes improve further but high-frequency channels regress more.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run 25-epoch ablation with `DET_OHEM_MIN_NEG=8` on RTX 3060 using `ablation_det_only` preset. Compare per-class AP at epoch 25 against current ablation best (epoch 14: mAP50_pc=0.2763, `112-training-metrics-deep-dive.md:582-588`). Success criterion: mAP50_pc >= 0.30 at any epoch.

---

## Q3. FPN Neck: Does BiFPN Improve Small-Object Detection?

**Context:** Our FPN neck has 4,474,880 params (`112-training-metrics-deep-dive.md:198`). YOLOv8m uses FPN+PAN (Path Aggregation Network) with cross-stage partial connections (`114-comparability-vs-4-papers.md:196-224`). The FPN received effectively zero gradient for ~11 epochs due to the F1 gradient wipe bug (`113-all-fixes-chronicle.md:460-505`). Channel 22 (28 GT) at 0.063 AP is the worst non-zero class -- it represents a transitional state (binary `11101111111`) where component 4 is absent. This transitional state has a small visual footprint.

**Question:** Would replacing simple FPN with BiFPN (EfficientDet-style, weighted bidirectional feature pyramid, adding ~2M params) improve per-class AP for transitional-state channels (22, 16, 19) by 0.05-0.10 each, at the cost of ~15% more inference time and ~2M additional parameters?

**Why this matters:** The FPN was grad-starved for 11 epochs (`113-all-fixes-chronicle.md:460-505`). Even after fix, the simple top-down FPN lacks bottom-up pathways needed for small objects. Paper 1's CSP-PAN addresses this (`114-comparability-vs-4-papers.md:196-224`). PSR also uses FPN features (`111-overview.md:31`), so BiFPN benefits 2 tasks.

**Constraints:** Code change to model.py. Retrain from scratch. Time: 2 days code + 10 days training. Additional VRAM: ~600 MB.

**Hypothesis:** BiFPN improves channels 22, 16, 19 by 0.05-0.10 each. mAP50_pc improves from 0.506 to 0.53-0.55 by epoch 25. PSR F1 improves 0.01-0.03.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement BiFPN with 3 fusion blocks and P3-P7 outputs. Run 50-epoch training. Compare per-class AP, mAP50_pc, PSR F1 at epochs 10, 25, 50. Success criterion: mAP50_pc >= 0.53 at epoch 25.

---

## Q4. Detection Head Depth: 2x256 vs 4x256 vs 6x256 Ablation

**Context:** RetinaNet detection head uses 4 conv layers with 256 channels (5,305,596 params total, `112-training-metrics-deep-dive.md:198`). For 24 ASD classes on 4,710 GT-bearing frames, head capacity may be overparameterized (157K params/class at 4 layers, vs YOLOv8m's ~104K params/class).

**Capacity analysis:** 2x256 head = 1.18M cls params = 79K/class (below YOLOv8m). 4x256 = 2.36M cls = 157K/class (above YOLOv8m). 6x256 = 2.95M cls = 197K/class.

**Question:** Does the 4x256 head overparameterize the 15-class, 4,710-GT-frame detection problem, causing overfitting on rare classes? Would 2x256 improve bottom-7 classes' AP by 0.03-0.08 each through reduced overfitting?

**Why this matters:** If head is overparameterized, 2.6M params can be freed. Paper 1 does not vary detection head depth (`114-comparability-vs-4-papers.md:196-224`). This is a novel ablation.

**Constraints:** Code change to roi_detector.py. Retrain from scratch. Time: 5 days per variant.

**Hypothesis:** 2x256 improves bottom-7 classes by 0.03-0.08 through reduced overfitting. Top-8 classes regress -0.01 to -0.02. Net mAP50_pc improves from 0.506 to 0.52-0.55. 6x256 provides no improvement over 4x256 (within 0.01).


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Train 2x256 and 6x256 variants for 50 epochs each. Compare per-class AP, mAP50, mAP50_pc at epochs 10, 25, 50. Success criterion: 2x256 achieves mAP50_pc >= 0.52 at epoch 25.

---

## Q5. FocalLoss Gamma-Neg Sweep with OHEM Disabled

**Context:** Double-suppression from FocalLoss (gamma=1.5) + OHEM (min_neg=32) may zero out gradient for rare classes. FocalLoss down-weights easy negatives; OHEM then selects top-K negatives. If FocalLoss already suppresses most negatives near zero, OHEM selects from a near-zero pool, providing no gradient.

**FocalLoss + OHEM math:** Single negative anchor with confidence p: FL(p) = -alpha * (1-p)^gamma * log(p). At gamma=1.5: FL(0.9) = negligible. OHEM then selects top K negatives by FL value from this near-zero pool.

**Question:** Is the double-suppression of FocalLoss+OHEM the root cause of 3 near-zero-AP channels (16, 19, 22)? Setting `DET_OHEM_ENABLED=False` and `DET_GAMMA_NEG=2.0` would remove the OHEM gate and increase FocalLoss's hard-example focus, allowing rare classes to receive useful gradient from negative anchors.

**Why this matters:** The OHEM+FocalLoss interaction is not studied for low-annotation detection. Paper 1 uses YOLOv8m no-OHEM. If disabling OHEM recovers the 3 zero-AP channels, this is a novel finding for the paper's "Training Pathologies" section.

**Hypothesis:** Without OHEM and gamma_neg=2.0: channels 16, 19 achieve AP > 0.05. Channel 22 achieves AP > 0.15. Top-5 channels regress slightly. mAP50_pc improves to 0.52-0.54.

**Constraints:** Config change. 25-epoch ablation on RTX 3060. Time: 3-4 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run ablation with `DET_OHEM_ENABLED=False, GAMMA_NEG=2.0`. Compare per-class AP at epochs 10, 15, 20, 25. Success criterion: channels 16, 19, 22 all >= 0.05 AP.

---

# Category 2: Activity Head -- Temporal Modeling

Our per-frame MLP activity head (0.7M params, 1.5% of total trainable) achieves macro-F1=0.110 on 69 verb-grouped classes (`112-training-metrics-deep-dive.md:497-514`). This is NOT comparable to MViTv2's 65.25% Top-1 on 75 classes with 16-frame temporal clips (`114-comparability-vs-4-papers.md:40-42`). The five questions explore the verb-grouping choice, TCN architecture, attention pooling, gradient amplification, and sampling distortion -- all targeting the estimated +0.04 to +0.10 macro-F1 improvement that would make our activity head competitive with the MViTv2 remapped baseline.

## Q6. Verb-Grouping vs Fine-Grained: Does 69-Class Protocol Help or Harm?

**Context:** Activity head predicts 69 verb-grouped classes, reduced from 75 fine-grained by merging rarely-occurring verb-noun pairs (`111-overview.md:68-69`). Macro-F1 = 0.110, pred_distinct = 35/69 (`112-training-metrics-deep-dive.md:497-514`). Verb grouping removes noun disambiguation -- merged classes like "take_short_brace" + "take_acorn_nut" into "take" have higher intra-class variance without more training data.

**Question:** Does reverting to 75 classes increase macro-F1 to 0.12-0.14 because the noun component provides a stronger discriminative signal that outweighs the metric dilution from more classes?

**Why this matters:** If 69-class has a systematically LOWER ceiling, we have been handicapping ourselves. Paper 1 uses 75 classes (`114-comparability-vs-4-papers.md:40-42`). 75 classes enables cleaner comparison.

**Hypothesis:** 75-class achieves macro-F1 of 0.12-0.14 (vs 0.110). pred_distinct increases to 45-50/75. The 6 merged verb classes had distinct visual signatures that the per-frame MLP can learn.

**Constraints:** Config change (`NUM_CLASSES_ACT=75`). Verify class mapping against Paper 1 taxonomy. 50-epoch ablation on RTX 3060. Time: 5-6 days.

**Validation:** Run 50-epoch ablation with 75 classes. Compare epoch-by-epoch macro-F1 against 69-class baseline. Success criterion: macro-F1 >= 0.12 at any epoch.

---

## Q7. TCN Dilations: Does 4-Layer (1,2,4,8) Outperform 2-Layer (1,2)?

**Context:** Temporal activity head uses 2-layer TCN with kernel_size=3, followed by 2 ViT blocks on 16-frame windows (`115-execution-plan-to-sota.md:520-558`). Receptive field: RF_2layers = 1 + (3-1)*1 + (3-1)*2 = 7 frames. Average action duration = 1.9s = 19 frames at 10 fps (`114-comparability-vs-4-papers.md:320-333`). RF_4layers = 1 + 2 + 4 + 8 + 16 = 31 frames -- covers the full action plus context.

**Question:** Would increasing TCN layers to 4 with exponential dilation (1,2,4,8) improve temporal macro-F1 from an estimated 0.15 to 0.17-0.20 by providing a receptive field (31 frames) that captures the average 1.9s action in a single TCN pass?

**Why this matters:** T2 costs 5 days (`115-execution-plan-to-sota.md:560-571`). If 2-layer TCN is architecturally limited (RF=7 vs 19-frame average action), the investment may produce a result well below the MViTv2 remapped baseline (~0.20). Correcting TCN design before T2 could raise expected outcome from 0.15 to 0.18+.

**Constraints:** Code change to TCN module. Adds ~1.5M params. Must be done before T2 starts (~July 15).

**Hypothesis:** 4-layer TCN achieves macro-F1 0.17-0.20 at epoch 50 vs 0.13-0.15 for 2-layer. Improvement concentrated on longer-duration actions (tightening, screwing).


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement both variants before T2. Run T2 with both. Compare macro-F1 at epochs 10, 25, 50. Success criterion: 4-layer >= 0.02 improvement at epoch 50.

---

## Q8. Attention Pooling vs Global Average Pooling for Activity

**Context:** Temporal activity head pools TCN output via global average across 16 timesteps. Attention pooling learns a weighted sum: z = sum_i softmax(q^T h_i / tau) * h_i. This adds 17 parameters (query vector + temperature).

**Question:** Would replacing global average pooling with attention-based temporal pooling improve temporal macro-F1 by 0.01-0.03 by learning to focus on discriminative temporal positions (grasp moment) while down-weighting transitional frames (hand in mid-air)?

**Why this matters:** 17-parameter improvement of 0.01-0.02 macro-F1 is a free lunch. Implementation cost < 10 lines.

**Hypothesis:** Attention pooling achieves 0.01-0.03 higher macro-F1. Effect strongest on temporally structured actions (tighten, loosen, screw). Static actions (check_instruction) see no improvement.

**Constraints:** Code change in activity head. No additional training time. < 100 params added.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement attention pooling alongside standard pooling. Compare epoch-by-epoch macro-F1 during T2. Success criterion: >= 0.015 improvement at epoch 25.

---

## Q9. ACTIVITY_GRAD_BLEND_RATIO: Is 1.00 Optimal or Should It Be 2.0?

**Context:** `ACTIVITY_GRAD_BLEND_RATIO` increased from 0.10 to 1.00 through 5 changes (`113-all-fixes-chronicle.md:146-147`). Activity contributes only 14.8% of total gradient (`112-training-metrics-deep-dive.md:883-888`). The ratio was designed for TCN+ViT (~8.2M params); with per-frame MLP (0.7M), backbone activity gradient is limited.

**Question:** Would increasing to 2.0 improve act_macro_F1 from 0.110 to 0.13-0.15 by forcing the backbone to learn action-relevant features, without significantly harming detection (mAP50 drop < 0.02)?

**Why this matters:** Config-only change via environment override. Can be applied mid-training. If activity improves 0.02-0.04 with no detection degradation, it's a free lunch.

**Hypothesis:** At BLEND_RATIO=2.0, act_macro_F1 improves to 0.13-0.15 over 5 epochs. Det_mAP50 stays within 0.01 of 0.317. Pose MAE stays within 0.3 deg. At 4.0, detection drops to 0.28-0.29 while activity reaches 0.14-0.16. Optimal is 2.0.

**Constraints:** Environment variable override. Applied from epoch 12+. No code change. Time: 5 epochs (~15 hours).


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Set `ACTIVITY_GRAD_BLEND_RATIO=2.0`. Run 5 epochs. Compare act_macro_F1, det_mAP50, pose MAE at epochs 13-16 vs epoch 11 baseline. Success criterion: macro-F1 >= 0.13 with det_mAP50 >= 0.30.

---

## Q10. DET_GT_FRAME_FRACTION: Does 0.40 Distortion Harm Activity More Than It Helps Detection?

**Context:** `DET_GT_FRAME_FRACTION=0.40` means 40% of batches have detection GT (`111-overview.md:139`). Sampler warns: "max/min ratio=7.4x distorting activity balance" (`111-overview.md:157-158`). Certain activity classes that correlate with GT-box frames are over-sampled. Activity macro-F1 is 0.110 with second-lowest Kendall precision (0.68x).

**Question:** Would `DET_GT_FRAME_FRACTION=1.0` (natural sampling, activity ratio ~1.0x) improve act_macro_F1 by 0.02-0.05 (to 0.13-0.16) by removing the 7.4x class imbalance, while detection stays within 0.01 of 0.317?

**Why this matters:** The 7.4x ratio is a known artifact documented since the earliest runs (`111-overview.md:157-158`) but never tested. Activity head is the second-weakest.

**Hypothesis:** At FRACTION=1.0, act_macro_F1 reaches 0.12-0.15 within 5 epochs. pred_distinct increases to 40-45. det_mAP50 changes < 0.01. Combined metric improves to 0.37-0.40.

**Constraints:** Config change at epoch boundary. No code change. Time: 5 epochs (~15 hours).


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Set `DET_GT_FRAME_FRACTION=1.0` at next epoch boundary. Run 5 epochs. Compare act_macro_F1, pred_distinct, det_mAP50, combined metric vs epoch 11 baseline. Success criterion: macro-F1 >= 0.12 by epoch 15.

---

# Category 3: Ego-Pose Head -- Defending the 8.14 Degree MAE

Our ego-pose head (1.6M + 0.8M FiLM + 0.4M headpose_film params) achieves forward MAE=8.14 degrees -- the first published baseline on IndustReal (`111-overview.md:751-757`). This is a publishable contribution as-is. The five questions here optimize it further (geodesic loss, position removal, FiLM ablation, augmentation, multi-seed) to push toward the estimated HoloLens 2 sensor noise floor of 5-7 degrees. Even without improvement, the current numbers are strong. These experiments primarily strengthen the methodological rigor of the paper.

## Q11. Geodesic Loss vs MSE on Unit Vectors for Forward/Up Prediction

**Context:** Ego-pose uses MSE loss on 3D unit vectors (`112-training-metrics-deep-dive.md:721-749`). Forward MAE = 8.14 deg, up MAE = 7.06 deg (`111-overview.md:751-757`). MSE on unit vectors: MSE_loss = ||v_pred - v_gt||^2 = 2 - 2*cos(theta). Gradient: grad(MSE)/d_theta = 2*sin(theta). At theta=10 deg, gradient = 0.174 -- nearly 6x smaller than geodesic's unit gradient. Geodesic loss: arccos(dot(v_pred, v_gt)). Gradient: 1.0 for all errors.

**Question:** Would geodesic loss (direct arccos of cosine similarity) reduce forward MAE to 7.0-7.5 degrees by providing unit gradient at all error levels (vs MSE's decaying gradient at small angles)?

**Why this matters:** Ego-pose is our strongest contribution (`111-overview.md:751-757`). Sub-7.0 deg MAE approaches the HoloLens 2 sensor noise floor (~5-7 deg, `116-winning-aaiml-synthesis.md:573`) and makes the metric more defensible.

**Hypothesis:** Geodesic loss converges to forward MAE 7.0-7.5 within 15 epochs (vs 8.14 at epoch 11). Up MAE improves to 6.0-6.5.

**Constraints:** Code change to losses.py. 25-epoch ablation on RTX 3060. Time: 2-3 days.

**Validation:** Implement geodesic loss. Run 25-epoch ablation. Compare forward/up MAE at epochs 5, 10, 15, 20, 25. Success criterion: forward MAE <= 7.5 deg.

---

## Q12. Position Loss Removal: Does Noisy HEAD_POSE_POS_SCALE Harm Orientation?

**Context:** Position term uses `HEAD_POSE_POS_SCALE=100.0` and contributes ~1/3 of head pose gradient (`112-training-metrics-deep-dive.md:722-729`). Position values flagged as "UNRELIABLE -- DO NOT USE" (`evaluate.py:1918-1926`). The uncalibrated position loss may inject gradient noise.

**Question:** Would removing the position term (training on angular loss only) reduce forward MAE from 8.14 to 7.5-7.8 degrees by eliminating gradient noise from the uncalibrated position signal?

**Why this matters:** Removing position simultaneously improves orientation accuracy AND removes the evaluate.py caveat from the paper.

**Hypothesis:** Position loss adds gradient noise. Removing it reduces forward MAE to 7.5-7.8, up MAE to 6.3-6.8. Position MAE (logged but not reported) does not diverge.

**Constraints:** Config change. 25-epoch ablation. Time: 2-3 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Restructure head pose loss to exclude position. Run 25-epoch ablation. Compare forward/up MAE. Success criterion: forward MAE <= 7.8 deg.

---

## Q13. FiLM Ablation: Is the HeadPose FiLM (400K Params) Modulating Features or Near-Identity?

**Context:** Head pose FiLM has 400,896 params (`112-training-metrics-deep-dive.md:202-203`). HP_PREC_CAP caps pose at 1.25x precision (`112-training-metrics-deep-dive.md:845-870`), limiting FiLM gradient. If gamma ~ 1, beta ~ 0, FiLM is dead weight.

**Question:** After 11 epochs with HP_PREC_CAP, are FiLM gamma parameters within 0.95-1.05 for 90%+ channels (near-identity, removable) or do they deviate significantly (genuine modulation)?

**Why this matters:** 400K params (0.9% of total) for the smallest head. If dead weight, 400K params can be reallocated to detection head.

**Hypothesis:** FiLM gamma is within [0.95, 1.05] for 90%+ channels. Removal causes < 0.3 deg MAE change. Saves 400K params.

**Constraints:** Inspect epoch 11 checkpoint FiLM parameters. If near-identity, remove and retrain 25 epochs. Time: 2-3 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Load epoch 11 checkpoint. Compute FiLM gamma/beta statistics. If gamma ~ 1.0 ± 0.1 for > 90% channels, remove FiLM and retrain. Success criterion: MAE difference < 0.3 deg.

---

## Q14. Image Rotation Augmentation for Ego-Pose Generalization

**Context:** Ego-pose trained on 12 participants' head tracking data (`114-comparability-vs-4-papers.md:348-354`). Random head rotation augmentation (±15 deg yaw, ±10 pitch, ±5 roll) with geometrically consistent image+label transformation is standard in head pose estimation.

**Question:** Would ±15 deg rotation augmentation reduce forward MAE from 8.14 to 7.0-7.5 deg by improving generalization across participants with varying neck kinematics and HL2 mounting positions?

**Hypothesis:** Rotation augmentation reduces forward MAE to 7.0-7.5 (0.6-1.4 deg improvement). Per-participant variance decreases. Complementary to geodesic loss (Q11).

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Code change to dataset.py for geometrically consistent augmentation. 25-epoch ablation. Time: 1 day code + 3 days training.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement image+label aligned rotation augmentation. Run 25-epoch ablation. Compare forward MAE and per-participant variance. Success criterion: forward MAE <= 7.5 deg.

---

## Q15. Multi-Seed Variance: Seeds 7, 42, 123

**Context:** All metrics from SEED=42 (`115-execution-plan-to-sota.md:95-96`). Multi-seed experiments are HIGH risk (`113-all-fixes-chronicle.md:194`). For AAIML, mean+std across 3 seeds required (`116-winning-aaiml-synthesis.md:575`).

**Question:** Is forward MAE std < 0.3 deg (single-seed 8.14 is representative) or > 0.5 deg (multi-seed required)?

**Hypothesis:** Std(forward MAE) = 0.3-0.5 deg. Std(det_mAP50) = 0.02-0.03. Mean forward MAE = 8.0-8.3 deg across seeds.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** 2 x 25-epoch runs on RTX 5060 Ti. Time: 6 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run seeds 7 and 123 for 25 epochs. Compute mean+std across 3 seeds for all metrics. Success criterion: std(forward MAE) < 0.5 deg.

---

# Category 4: PSR -- Beating SOTA POS=0.812

Our MonotonicDecoder (3.1M params, 6.8% of total trainable) achieves POS=0.968 vs SOTA 0.812 (STORM-PSR) -- a 19% beat (`114-comparability-vs-4-papers.md:432-434`). However, PSR F1=0.144 vs SOTA 0.901 is an 84% deficit. The five questions address: (Q16) D4 experiment isolating detection quality as the bottleneck, (Q17) tau measurement to understand structural F1 limits, (Q18-Q20) three decoder-side improvements (per-component thresholds, temporal smoothing, sequence frequency) that could improve F1 independently of detection quality.

## Q16. D4: YOLOv8m -> MonotonicDecoder (2-3 Hour Experiment)

**Context:** PSR F1 = 0.144 on ConvNeXt detection (mAP=0.317) vs B3 F1=0.883 (`114-comparability-vs-4-papers.md:293-295`). 84% gap attributed to detection quality. D4 feeds YOLOv8m ASD outputs (mAP=0.838) through MonotonicDecoder (`115-execution-plan-to-sota.md:425-461`). This is the single most impactful 2-3 hour experiment (`115-execution-plan-to-sota.md:429`).

**Pipeline:** YOLOv8m inference on validation frames. Map 24-class ASD to 11 PSR components. Run MonotonicDecoder. Compute POS, F1, edit.

**Question:** When YOLOv8m detection feeds our MonotonicDecoder, does PSR F1 reach >= 0.50 (detection quality confirmed as bottleneck, decoder viable) or < 0.30 (per-frame paradigm fundamentally limits F1)?

**Why this matters:** If F1 >= 0.50, the paper can claim "PSR decoder achieves F1 = X on YOLOv8m -- detection quality is the primary bottleneck." If < 0.30, the per-frame paradigm is the bottleneck and transition-detection redesign is needed.

**Hypothesis:** F1 reaches 0.55-0.65. Detection quality explains 70-80% of gap. YOLOv8m -> decoder F1 exceeds STORM-PSR's temporal stream alone (F1=0.506, `114-comparability-vs-4-papers.md:432-434`).

**Constraints:** RTX 3060 idle. YOLOv8m weights from GitHub. Class mapping verification. Time: 2-3 hours.

**Validation:** Run D4. Success criterion: F1 >= 0.50. If >= 0.70, decoder is excellent.

---

## Q17. PSR Tau: What Is MonotonicDecoder's Mean Detection Delay?

**Context:** B3 tau = 22.4s, STORM-PSR = 15.5s (`114-comparability-vs-4-papers.md:259, 432-433`). Our tau NOT MEASURED (E2, `111-overview.md:186`). F1 uses +/-3 frame tolerance. If tau > 3 frames, F1 structurally capped.

**Question:** What is mean per-component detection delay in frames? If tau > 3 frames, the F1 is structurally capped regardless of decoder quality, meaning D4 would also produce low F1.

**Hypothesis:** True positive tau = 3-5 frames (0.3-0.5s). Rare components (h4, h10) have tau > 5 frames. ~50% of detections fall outside +/-3 tolerance.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Implement tau in evaluate.py (E2, 1 day). Uses epoch 11 checkpoint. No training.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement tau. Compute per-component on epoch 11. Compare against B3/STORM. Success criterion: establishes tau baseline for paper.

---

## Q18. Per-Component Adaptive Thresholds for MonotonicDecoder

**Context:** Single threshold for all 11 components. Prevalence ranges 19.1-100% (`112-training-metrics-deep-dive.md:323-337`). Rare components (h4, h7-h9) have gradient RMS < 0.002 (`112-training-metrics-deep-dive.md:1288-1300`). Cascading effect: late detection of component 4 delays detection of all subsequent components.

**Question:** Would per-component thresholds (rare components: 0.15-0.25, common: 0.5-0.7) improve PSR F1 from 0.144 to 0.17-0.22 by reducing cascading delay for rare components?

**Why this matters:** Inference-only change (< 50 lines of code). No retraining. Thresholds grid-searched on validation set.

**Hypothesis:** Optimal thresholds: h4=0.15-0.25 (F1 +0.03-0.10), h10=0.15-0.25 (F1 +0.02-0.08), h7-h9=0.30-0.40 (F1 +0.01-0.03), h0-h3=0.60-0.80 (flat). Net F1 improves to 0.17-0.22.

**Constraints:** Code change to psr_transition.py. Inference only. Time: 1 day.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Grid-search per-component thresholds (0.05-0.95, 0.05 increments). Optimize on epoch 11 validation. Compute F1, POS, edit. Success criterion: F1 >= 0.17 with POS >= 0.95.

---

## Q19. Temporal Smoothing Weight Sweep for PSR

**Context:** `PSR_TEMPORAL_SMOOTH_WEIGHT=0.05` (`112-training-metrics-deep-dive.md:754-761`). Smoothness penalizes per-frame state oscillations.

**Question:** Would increasing to 0.20 (4x) improve PSR F1 from 0.144 to 0.16-0.19 by reducing spurious transitions (false positives from jittery predictions)?

**Hypothesis:** At 0.20, F1 improves to 0.16-0.19 within 5 epochs. At 0.50, over-smoothed, F1 drops below 0.10.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change via env override. Applied mid-training. No code change. Time: 5 epochs.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Set `PSR_TEMPORAL_SMOOTH_WEIGHT=0.20`. Run 5 epochs. Compare F1, POS, edit vs epoch 11. Success criterion: F1 >= 0.16 with POS >= 0.95.

---

## Q20. PSR Sequence Frequency: Every-2-Batches vs Every-4-Batches

**Context:** `PSR_SEQ_EVERY_N_BATCHES=4` (`113-all-fixes-chronicle.md:149`), reduced from 2 in F7. F1 gradient wipe fix now protects backbone gradients (`113-all-fixes-chronicle.md:479-501`). Original F7 rationale no longer applies.

**Question:** With DETACH_PSR_FPN=True protecting backbone, does reverting to seq_freq=2 double PSR transition training signal and improve F1 by 0.02-0.05 (to 0.17-0.22) within 5 epochs?

**Hypothesis:** At freq=2, F1 improves to 0.17-0.22. Rare components (h4, h7-h10) benefit most from more frequent transition gradient. No gradient disruption due to DETACH.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change via env override. Applied at next epoch boundary. Time: 5 epochs.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Set `PSR_SEQ_EVERY_N_BATCHES=2`. Run 5 epochs. Compare F1, per-component liveness, POS. Success criterion: F1 >= 0.17 with POS >= 0.95.

---

# Category 5: Multi-Task Balancing -- Kendall vs GradNorm vs PCGrad

Kendall uncertainty weighting with HP_PREC_CAP is our current multi-task balancing mechanism. Current gradient composition: det 27.2%, pose 27.2% (capped), act 14.8%, psr 30.7% (`112-training-metrics-deep-dive.md:883-888`). The five questions explore: (Q21) whether Kendall is at equilibrium, (Q22) GradNorm as alternative, (Q23) PCGrad for gradient conflict resolution, (Q24) HP_PREC_CAP ablation, (Q25) log-var initialization. Each targets a different aspect of the same core problem: how should 4 tasks share one backbone optimally?

## Q21. Kendall vs Fixed Weights at Equilibrium

**Context:** Current log_vars: lv_det=-0.225, lv_pose=-0.998(capped), lv_act=+0.381, lv_psr=-0.345 (`112-training-metrics-deep-dive.md:822-825`). Equilibrium values computed from task losses: lv_i* = ln(L_i). lv_det* ≈ ln(0.5) = -0.693, lv_act* ≈ ln(1.0) = 0.000, lv_psr* ≈ ln(0.7) = -0.357. Log_var gradients decreasing from 0.5 to 0.3 (`112-training-metrics-deep-dive.md:896-903`).

**Question:** Are log_vars approaching equilibrium or not? det and act are 0.3-0.6 away from equilibrium, meaning Kendall is still actively learning.

**Hypothesis:** Det (current -0.225 vs eq -0.693) and act (current 0.381 vs eq 0.000) are not at equilibrium. Kendall is still beneficial. Fixed weights would underperform by 0.01-0.02 within 15 epochs.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Analysis only (1 hour). Compute equilibrium from per-task losses at epochs 10, 11, 12.

**Validation:** Compare current log_vars to equilibrium. If any > 0.1 away, Kendall is still converging. If all within 0.1, schedule B1.

---

## Q22. Kendall vs GradNorm

**Context:** Kendall gradient: det 27.2%, pose 27.2% (capped), act 14.8%, psr 30.7% (`112-training-metrics-deep-dive.md:883-888`). GradNorm equalizes gradient magnitudes. Activity (14.8%) has the most improvement headroom (0.110 macro-F1, ceiling ~0.20, `115-execution-plan-to-sota.md:504-514`).

**Question:** Would GradNorm (alpha=0.12) increase activity gradient to ~25%, improving macro-F1 from 0.110 to 0.13-0.16, at the cost of detection dropping to 0.30-0.32 mAP and PSR dropping to 0.10-0.12 F1?

**Hypothesis:** Activity improves to 0.13-0.16. Detection drops to 0.29-0.31. PSR drops to 0.10-0.12. Combined metric stays within 0.01 of Kendall baseline.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Code change to losses.py. 25-epoch ablation. Time: 2 days code + 5 days training.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement GradNorm. Run 25-epoch ablation. Compare all metrics vs Kendall. Success criterion: combined metric >= 0.36 (within 0.01 of baseline) OR activity improvement >= 0.03 with combined within 0.01.

---

## Q23. PCGrad: Do Detection and Pose Gradients Conflict?

**Context:** F12 `grad_cosine_probe` exists but never run (`113-all-fixes-chronicle.md:197-198`). Detection and pose gradients may conflict (det wants sharp spatial features, pose wants broad orientation features). Cosine similarity < -0.2 indicates conflict.

**Question:** What is detection-pose gradient cosine similarity at epoch 12? If cos < -0.3, would PCGrad improve detection mAP50 from 0.317 to 0.33-0.35 by removing gradient interference?

**Hypothesis:** cos(det, pose) = -0.3 to -0.5, confirming conflict. PCGrad improves detection to 0.33-0.35. Pose MAE stays within 0.3 deg.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** First run F12 tool (1h). If conflict, implement PCGrad (2 days code + 5 days training).


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run F12. If cos < -0.2, implement PCGrad. Run 25-epoch ablation. Compare det_mAP50, pose MAE. Success criterion: det_mAP50 >= 0.33 with pose MAE <= 8.5 deg.

---

## Q24. HP_PREC_CAP Ablation: Should the Cap Be Removed?

**Context:** HP_PREC_CAP caps pose at 1.25x precision (`112-training-metrics-deep-dive.md:845-870`). Without cap, pose would contribute ~40% of gradient. Pose may provide beneficial auxiliary features for detection.

**Question:** Would removing HP_PREC_CAP paradoxically improve detection mAP50 from 0.317 to 0.33-0.35 because the additional pose gradient provides beneficial orientation cues for spatial understanding?

**Hypothesis:** Without cap, pose share rises to 35-45%. Detection improves to 0.33-0.35. Pose MAE improves to 7.5-8.0. Activity regresses slightly (-0.01 to -0.02).

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change via env override. Applied mid-training. Time: 5 epochs.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Set `KENDALL_HP_PREC_CAP=False`. Run 5 epochs. Compare det_mAP50, pose MAE, act_macro_F1 vs epoch 11. Success criterion: det_mAP50 >= 0.33 and pose MAE <= 8.5 deg.

---

## Q25. Kendall Log_Var Symmetric Initialization

**Context:** Log_vars initialize asymmetrically: lv_pose=-1.0 (historical suppression) while others at 0.0 (`112-training-metrics-deep-dive.md:830-837`). Equal initialization (all 0.0) may converge faster.

**Question:** Would all-0.0 initialization reach combined=0.363 at epoch 7-8 instead of 11, saving 3-4 epochs (10 hours)?

**Hypothesis:** Equal initialization reaches combined=0.363 at epoch 7-8. Final convergence at epoch 25 is within 0.01 of baseline.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change. Fresh 25-epoch run. Time: 3-4 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run ablation with all log_vars initialized at 0.0. Compare epoch-by-epoch combined against stage_rf4 trajectory. Success criterion: combined >= 0.35 by epoch 8.

---

# Category 6: Architecture -- Backbone Swaps

All current results use ConvNeXt-Tiny (28.6M params) with random initialization. The five questions explore: (Q26) ImageNet-1K pretraining with proper fine-tuning, (Q27) Swin-T for attention-based ASD, (Q28) scaling to ConvNeXt-Small, (Q29) EfficientNet-B4 for compute efficiency, (Q30) gradient detachment to quantify multi-task cost. These represent the highest-investment but potentially highest-reward experiments, each requiring 7-17 days of GPU time.

## Q26. ImageNet-1K Pretrained Weights Revisited

**Context:** Random init (no pretraining) was chosen because early experiments showed -0.02 mAP regression with ImageNet weights (`115-execution-plan-to-sota.md:102`). If caused by incorrect fine-tuning (LR too high for the first few epochs), proper discriminative LR could yield +0.03-0.05 mAP.

**Standard fine-tuning:** Phase 1 (epochs 0-5): backbone LR=1e-5 (frozen), heads LR=5e-4. Phase 2 (epochs 5-25): backbone LR=5e-5, heads LR=5e-4. Phase 3 (epochs 25+): all at 5e-5.

**Question:** Was the -0.02 mAP regression caused by catastrophic forgetting from 5e-4 backbone LR on pretrained weights, and would a proper discriminative schedule (backbone LR=1e-5 for 5 epochs then 5e-5) turn this into +0.03 to +0.05 mAP50 improvement?

**Hypothesis:** With proper schedule, det_mAP50 reaches 0.34-0.37 (vs 0.317) at epoch 15. Largest gains on low-GT classes (+0.05-0.10). Activity improves 0.01-0.02. Pose improves 0.2-0.5 deg.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change (discriminative LR, `PRETRAINED=True`). No code change. 25-epoch ablation. Time: 3-4 days.

**Validation:** Run ablation with ImageNet pretrained weights and discriminative LR. Compare epoch-by-epoch metrics vs random-init baseline. Success criterion: det_mAP50 >= 0.34 at epoch 15.

---

## Q27. Swin-T vs ConvNeXt-Tiny for ASD

**Context:** ConvNeXt-Tiny (CNN, 28.6M, 82.1% ImageNet) and Swin-T (transformer, ~29M, 81.3% ImageNet) are comparable size (`112-training-metrics-deep-dive.md:197`). For spatial relationship reasoning in ASD (e.g., "is washer ON bolt vs NEXT TO bolt"), self-attention may outperform convolution.

**Question:** Does Swin-T achieve det_mAP50 improvement >= 0.02 over ConvNeXt-Tiny at the same training schedule, despite lower ImageNet accuracy?

**Hypothesis:** Swin-T achieves mAP50_pc = 0.53-0.56 (vs 0.506) and mAP50 = 0.34-0.37 (vs 0.317) at epoch 25. Small components (channels 22, 16, 19) improve > 0.05 AP. PSR F1 improves 0.01-0.03.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Code change for Swin-T backbone. 20% more VRAM. 50-epoch run. Time: 7-10 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement Swin-T backbone. Run 50-epoch training. Compare all metrics at epochs 10, 25, 50. Success criterion: mAP50_pc >= 0.53 at epoch 50.

---

## Q28. ConvNeXt-Scale: Tiny vs Small

**Context:** ConvNeXt-Tiny (28.6M) uses ~10 GB VRAM (`111-overview.md:241-254`). ConvNeXt-S (50.1M) would use ~13 GB at B=2. RTX 5060 Ti has 16 GB.

**Question:** Would ConvNeXt-S at B=2 (effective 8 via accumulation) improve det_mAP50 by 0.03-0.06 to 0.35-0.38, fitting within 16 GB VRAM?

**Hypothesis:** ConvNeXt-S at B=2 fits in 16 GB. Det_mAP50 reaches 0.35-0.38 at epoch 25. Activity reaches 0.13-0.15. Combined reaches 0.40-0.45.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** First test VRAM. If fits, 50-epoch run. Time: 10 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Test VRAM at B=2. If < 15 GB, run 50 epochs. Compare all metrics against ConvNeXt-Tiny at equivalent epochs. Success criterion: det_mAP50 >= 0.35 at epoch 25.

---

## Q29. EfficientNet-B4: Compute-Optimal Backbone

**Context:** ConvNeXt-Tiny uses ~4.5 GMACs. EfficientNet-B4 uses ~4.2 GMACs at 82.6% ImageNet Top-1. 15-30% faster inference.

**Question:** Does EfficientNet-B4 match ConvNeXt-Tiny's detection accuracy (mAP50 within -0.01) while being 15-30% faster (0.7-0.8 batch/s vs 0.6)?

**Hypothesis:** EfficientNet-B4 matches mAP50 (0.31-0.32) at 15-20% faster training. May be 0.01-0.02 lower due to compound scaling not transferring perfectly to ASD.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Code change for EfficientNet-B4. 50-epoch run. Time: 7-10 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement EfficientNet-B4. Run 50-epoch training. Compare mAP50 and training speed. Success criterion: mAP50 within -0.01 of baseline AND speedup >= 15%.

---

## Q30. Gradient Detachment for Activity and Pose

**Context:** All 4 tasks backpropagate to shared backbone. Freezing backbone for activity/pose quantifies multi-task cost.

**Question:** Does freezing activity/pose gradients improve det_mAP50 from 0.317 to 0.34-0.36 (approaching single-task performance), at the cost of act_macro_F1 dropping to 0.06-0.08 and pose MAE rising to 9-10 deg?

**Hypothesis:** Det_mAP50 reaches 0.34-0.36 with activity/pose frozen out. Multi-task cost for detection = 0.03-0.04 mAP. PSR unaffected (keeps backbone gradient).

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Code change in model.py. 25-epoch ablation. Time: 5 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement gradient detachment. Run 25 epochs. Compare all metrics against full multi-task baseline. Success criterion: det_mAP50 >= 0.34 at any epoch.

---

# Category 7: Training Strategy -- Schedule, Augmentation, EMA

Our training uses OneCycleLR (peak_factor=0.5), EMA (decay=0.995), no mixup, and label smoothing (0.1). The five questions explore: (Q31) learning rate peak factor, (Q32) EMA decay, (Q33) mixup revisitation, (Q34) SWA alternative to EMA, (Q35) label smoothing strength. These are primarily config-level changes requiring 3-10 hours each.

## Q31. OneCycleLR Peak Factor: 0.75 vs 0.5 for Faster Convergence

**Context:** OneCycleLR peak_factor=0.5 (`113-all-fixes-chronicle.md:66-68`). Max head LR = 1.25e-4. Higher LR may converge faster.

**Question:** Would peak_factor=0.75 reach combined=0.363 at epoch 8-9 instead of 11, saving 2-3 epochs?

**Hypothesis:** At 0.75, combined=0.363 at epoch 8-9. At 1.0, detection loss spikes NaN at epoch 3-4.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change. 25-epoch ablation. Time: 3-4 days.

**Validation:** Run with peak_factor=0.75. Compare epoch-by-epoch combined. Success criterion: combined >= 0.35 by epoch 8.

---

## Q32. EMA Decay: 0.999 vs 0.995

**Context:** EMA decay=0.995 from epoch 0 (`111-overview.md:42-44`). Half-life = 138 steps. At 0.999: 693 steps.

**Question:** Does smoother EMA (0.999) provide 3-5% lower epoch-to-epoch variance and 0.01-0.02 higher best combined?

**Hypothesis:** At 0.999, best combined improves to 0.37-0.38 (from 0.363). At 0.99, best combined drops to 0.34-0.35 (too noisy).

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change. 25-epoch ablation from scratch. Time: 4-5 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run with EMA_DECAY=0.999. Compare EMA-based metrics vs raw model. Success criterion: EMA combined >= 0.37 at any epoch.

---

## Q33. Mixup Revisited with F1-F22b Fixes Active

**Context:** Mixup "explicitly broken" (`113-all-fixes-chronicle.md:83`). The early confounded experiment was during F1 gradient wipe (80% backbone signal loss). With all fixes, mixup may provide benefits.

**Question:** Does mixup (alpha=0.2) now provide 0.01-0.03 mAP50 improvement, or does it still degrade detection with all F1-F22b fixes applied?

**Hypothesis:** Mixup provides 0.01-0.02 mAP50_pc improvement. The old result was confounded by F1 gradient wipe.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change. 25-epoch ablation. Time: 4-5 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run with USE_MIXUP=True, MIXUP_ALPHA=0.2. Compare mAP50_pc at equivalent epochs. Success criterion: mAP50_pc >= 0.52 at epoch 15.

---

## Q34. SWA vs EMA for Multi-Task Convergence

**Context:** SWA averages weights over last K epochs. Flatter minima generalize better in multi-task settings.

**Question:** Would SWA (epochs 75-100 average) improve combined metric by 0.01-0.03 over EMA by finding a flatter 4-objective minimum?

**Hypothesis:** SWA combined = 0.37-0.39 (vs 0.363). Activity improvement 0.01-0.02. PSR improvement 0.01-0.02.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Implement SWA using torch.optim.swa_utils. No additional training. Time: 1 day code.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement SWA. Track metrics from epoch 12+ and historical checkpoints. Success criterion: SWA combined >= 0.38.

---

## Q35. Label Smoothing: 0.05 vs 0.10 for Activity

**Context:** Activity uses `CB_LABEL_SMOOTHING=0.1` (`112-training-metrics-deep-dive.md:694-698`). For 69 long-tail classes, smoothing may suppress confidence on 35 predicted classes.

**Question:** Would 0.05 smoothing improve act_macro_F1 from 0.110 to 0.12-0.13 by allowing more confident predictions?

**Hypothesis:** At 0.05, macro-F1 reaches 0.12-0.13 within 5 epochs. At 0.0, reaches 0.13-0.14 but pred_distinct drops from 35 to 30.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change via env override. No retraining. Time: 3 epochs.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Set `CB_LABEL_SMOOTHING=0.05`. Run 3 epochs. Compare macro-F1, pred_distinct. Success criterion: macro-F1 >= 0.12.

---

# Category 8: Data Strategy

The IndustReal dataset has 188K labeled frames but only 26K are in our training split, and only 4,710 have detection GT. The five questions explore: (Q36) per-component PSR loss weighting, (Q37) synthetic data generation (50K images), (Q38) YOLOv8m pseudo-labeling, (Q39) active learning on the 188K pool, (Q40) full-set evaluation bias. These data-centric approaches have the highest ceiling for detection improvement (0.04-0.07 mAP estimated) but require significant setup time for synthetic data generation.

## Q36. Per-Component Weighted Loss for PSR Rare Components

**Context:** PSR prevalence: comp0=100%, comp4=19.1% (`112-training-metrics-deep-dive.md:323-337`). Inverse weights: comp4 = 1/0.191 = 5.24x. Near-zero gradient for rare components (`112-training-metrics-deep-dive.md:1288-1300`).

**Question:** Would inverse-frequency weights (5x for comp4, 4.5x for comp10) increase PSR F1 from 0.144 to 0.18-0.22 by providing stronger gradient for rare components?

**Hypothesis:** F1 improves to 0.18-0.22. h4 F1 improves from near-0 to 0.1-0.3. POS drops to 0.94-0.96.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Code change in losses.py. 25-epoch ablation. Time: 3-4 days.

**Validation:** Implement per-component weights. Run 25-epoch ablation. Compare F1, per-component F1, POS. Success criterion: F1 >= 0.18.

---

## Q37. Synthetic Data from Unity Perception (50K Images)

**Context:** Paper 1 uses 100K synthetic images for +0.085 mAP on YOLOv8m (`114-comparability-vs-4-papers.md:242-248`). Our random init should benefit more.

**Question:** Would 50K synthetic images improve mAP50_pc from 0.506 to 0.55-0.58?

**Hypothesis:** +0.04-0.07 mAP50_pc, concentrated on rare classes. Real+synthetic achieves mAP50_pc >= 0.55.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Unity Perception/BlenderProc setup. 5-8 days setup + 5 days training.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Generate 50K synthetic images. Train 50 epochs with real+synthetic. Compare mAP50_pc. Success criterion: mAP50_pc >= 0.55.

---

## Q38. Pseudo-Labeling with YOLOv8m on 82% Non-GT Frames

**Context:** 82% of training frames have no GT detection boxes (`111-overview.md:139`). YOLOv8m (mAP=0.838) can generate pseudo-labels.

**Question:** Would pseudo-labels (>0.5 confidence) from YOLOv8m on non-GT frames improve mAP50 from 0.317 to 0.35-0.38 by providing positive anchor supervision on frames currently contributing only negative classification loss?

**Hypothesis:** High-confidence pseudo-labels (>0.7) provide accurate annotations on 80%+ frames. Effective detection supervision doubles. mAP50 improves to 0.35-0.38.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** YOLOv8m weights from D1. 1 day pseudo-label generation + 5 days retraining.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Generate pseudo-labels. Filter by confidence > 0.5. Train 50-epoch model. Compare mAP50. Success criterion: mAP50 >= 0.35.

---

## Q39. Active Learning: Adding 1000 Most Uncertain Frames

**Context:** 188K labeled frames exist; 26K in training (`111-overview.md:135-137`). Three channels have 0.000 AP -- likely annotation gaps.

**Question:** Would adding 1000 frames with highest prediction entropy to training improve 3 zero-AP channels to AP > 0.05?

**Hypothesis:** The 3 zero-AP classes have < 5 training examples. Adding 1000 uncertain frames brings each to 15+ examples. After retraining: AP > 0.05.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Run inference on 188K frames (2-3h). Select top 1000 by entropy. Verify GT. Retrain 25 epochs.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run active learning pipeline. Add top-1000 uncertain frames to training set. Retrain 25 epochs. Compare per-class AP for target channels. Success criterion: channels 16, 19, 22 >= 0.05 AP.

---

## Q40. Full Eval vs 250-Batch Subsample (D3)

**Context:** Current validation uses 250/38K frames (~2.6%) (`112-training-metrics-deep-dive.md:364-366`). D3 (EVAL_MAX_BATCHES=0, 1h) would reveal subsample bias.

**Question:** Does the 250-batch subsample (2.6% of 38K) consistently underestimate det_mAP50 by 0.02-0.05 by undersampling certain participants or states?

**Hypothesis:** Full-set eval shows det_mAP50 = 0.33-0.36 (vs 0.317). Rare-class AP increases because the full set has more GT instances.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change. No retraining. Time: 1 hour.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run full eval on epoch 11. Compare all metrics vs 250-batch subsample. Success criterion: establishes subsample bias direction.

---

# Category 9: Comparability and Paper Positioning

The five questions here are experiments that directly enable fair comparison with published benchmarks. They range from 30 minutes (Q43: canonical-order baseline) to 1 day (Q45: MViTv2 remap). Without these experiments, the AAIML paper is incomplete -- reviewers will ask for D1 (YOLOv8m eval) and T4 (act_top1) as minimum requirements for a research paper. These are P0 experiments in the execution plan (`115-execution-plan-to-sota.md:196-199`).

## Q41. D1: YOLOv8m on Our Split (Highest-Impact Experiment)

**Context:** D1 (2h) is the highest-impact experiment (`115-execution-plan-to-sota.md:357-396`). Published YOLOv8m mAP = 0.838. Our split may differ in difficulty.

**Question:** Does YOLOv8m achieve 0.75-0.85 (confirming compatible split, gap 55-62%) or < 0.70 (our split is harder, gap overstated)?

**Hypothesis:** YOLOv8m achieves 0.78-0.82 on our split. Gap is 58-62%.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** RTX 3060 idle. 2 hours.

**Validation:** Run D1. Compute YOLOv8m mAP50 on our split. Compare to published 0.838.

---

## Q42. Act_Top1 Metric (T4)

**Context:** MViTv2 65.25% Top-1 (`114-comparability-vs-4-papers.md:40-42`). Our act_clip=0.0625, act_frame=0.177. T4 (1h) adds act_top1.

**Question:** Is per-frame Top-1 closer to act_clip (0.0625) or act_frame (0.177)?

**Hypothesis:** Per-frame Top-1 ≈ 0.15-0.22. act_clip suppressed because 16-frame majority vote finds no majority. Gap to MViTv2 is ~3-4x (18% vs 65.25%).

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** 10-line code change. No retraining. Time: 1 hour.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Add act_top1 to evaluate.py. Run on epoch 11 checkpoint.

---

## Q43. Canonical-Order Baseline for PSR POS

**Context:** PSR POS = 0.968 beats SOTA by 19% (`114-comparability-vs-4-papers.md:280-286`). Paradigm disclosure says fill-forward inflates POS.

**Question:** What POS does a "blind canonical order" achieve (no visual input)? If > 0.90, our learned improvement is only 0.05-0.06, changing the paper's claim.

**Hypothesis:** Canonical baseline POS = 0.85-0.93. Learned improvement = 0.04-0.12.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** CPU computation. 30 minutes.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Create blind canonical-order model. Evaluate. Compare to 0.968. Success criterion: establishes paradigm disclosure quantitatively.

---

## Q44. Per-Frame PSR Tau (E2)

**Context:** Tau not measured (E2, `111-overview.md:186`). B3 tau = 22.4s, STORM = 15.5s (`114-comparability-vs-4-papers.md:259, 432-433`).

**Question:** Is our per-frame tau (seconds) competitive with B3's 22.4s (faster because no evidence accumulation) or slower (fill-forward delay)?

**Hypothesis:** True positive tau = 0.5-1.5s (5-15 frames). Faster than B3's 22.4s.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Implement tau in evaluate.py (1 day). No training.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement tau. Compute on epoch 11. Compare to B3/STORM.

---

## Q45. MViTv2 Remap to 69 Classes (T3)

**Context:** MViTv2 65.25% Top-1 on 75 classes (`114-comparability-vs-4-papers.md:40-42`). T3 remaps to 69 classes (1 day). Estimated remapped ~25% Top-1 / 0.20 macro-F1.

**Question:** Does remapped MViTv2 achieve macro-F1 of 0.25-0.35 (our temporal target 0.15 is far behind) or 0.15-0.20 (our target is close)?

**Hypothesis:** Remapped MViTv2 achieves macro-F1 0.25-0.35 because many errors are verb-correct but noun-wrong. Our temporal target (0.15) is only 43-60% of SOTA, not 75%.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Download MViTv2 weights. CPU remapping. Time: 1 day.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run T3. Compute remapped macro-F1 on our validation split. Compare to our temporal head result.

---

# Category 10: The Unexpected -- High-Leverage Wildcards

The five wildcard questions explore approaches outside the current architecture's assumptions: cross-modal FiLM sharing (Q46), Feature Bank temporal memory (Q47), MAE self-supervised pretraining (Q48), batch composition adjustment (Q49), and test-time augmentation (Q50). These have the highest variance -- any could produce a 0.05+ metric breakthrough or zero effect. Q50 (TTA) requires just 2 hours and could improve detection by 0.03-0.07 without any training.

## Q46. Cross-Modal FiLM Sharing for Activity and Pose

**Context:** Ego-pose FiLM (400K params) encodes pose-relevant features (`112-training-metrics-deep-dive.md:202-203`). Head orientation correlates with gaze, which correlates with action.

**Question:** Would providing FiLM-modulated features to the activity head improve act_macro_F1 by 0.02-0.05 (to 0.13-0.16) because head orientation provides a strong action prior?

**Hypothesis:** Shared FiLM improves act_macro_F1 to 0.13-0.16. Tool manipulation actions (tighten, loosen, screw) show largest improvement. No additional inference cost.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Code change to connect FiLM to activity head. 50-epoch retraining. Time: 5-6 days.

**Validation:** Implement shared FiLM. Run 50 epochs. Compare act_macro_F1 against baseline. Success criterion: improvement >= 0.02.

---

## Q47. FeatureBank GRU Temporal Memory for Per-Frame Activity

**Context:** Feature Bank with GRU (T=16, stride=1) logged as 0 params (`112-training-metrics-deep-dive.md:205, 212-213`). May be disabled. If enabled, provides temporal smoothing.

**Question:** Is the Feature Bank GRU disabled (0 params)? If so, would enabling it (T=16, hidden=256) improve act_macro_F1 from 0.110 to 0.13-0.17 through temporal smoothing?

**Hypothesis:** Feature Bank is disabled. Enabling it improves act_macro_F1 to 0.13-0.17. Detection benefits +0.01 mAP50 from smoother features.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Investigate config.py and model.py. If disabled, enable and retrain 50 epochs. Time: 1 day + 5 days training.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Check Feature Bank configuration. If disabled, enable. Run 50 epochs. Compare act_macro_F1, det_mAP50. Success criterion: act_macro_F1 >= 0.13.

---

## Q48. MAE Self-Supervised Pretraining on 188K Frames

**Context:** MAE pretraining pipeline exists (`src/training/pretrain_mae.py`, 362 lines). 188K labeled frames available. In-distribution pretraining should be more beneficial than ImageNet.

**Question:** Would MAE pretraining (50 epochs) on all 188K frames improve det_mAP50 by 0.03-0.06 and act_macro_F1 by 0.02-0.04 at epoch 11 of fine-tuning?

**Hypothesis:** MAE pretrained backbone achieves det_mAP50 = 0.35-0.38 (vs 0.317) and act_macro_F1 = 0.13-0.15 (vs 0.110) at fine-tuning epoch 11.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** 50 epochs MAE + 100 epochs fine-tuning. Total: 17 days on RTX 5060 Ti.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run MAE pretraining. Initialize multi-task model. Run 100-epoch fine-tuning. Compare against random-init baseline at equivalent fine-tuning epochs. Success criterion: det_mAP50 >= 0.35 at fine-tuning epoch 11.

---

## Q49. Batch Composition: DET_GT_FRAME_FRACTION=0.60

**Context:** `DET_GT_FRAME_FRACTION=0.40` (`111-overview.md:139`). 0.60 provides 50% more detection supervision.

**Question:** Would 0.60 improve det_mAP50 by 0.02-0.04 (to 0.33-0.35) while activity macro-F1 drops < 0.01?

**Hypothesis:** Detection improves to 0.33-0.35. Activity stays at 0.10-0.11. Net combined improves 0.01-0.02.

**Why this matters:** This experiment directly addresses a specific gap identified in the analysis documents. The metric impact is estimated at 0.02-0.05 for the primary metric, with clear validation criteria defined below.
**Constraints:** Config change. Fresh 25-epoch ablation. Time: 4-5 days.


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Run with DET_GT_FRAME_FRACTION=0.60. Compare det_mAP50, act_macro_F1, combined. Success criterion: combined >= 0.37 at epoch 15.

---

## Q50. Test-Time Augmentation for Detection (Highest-Leverage Wildcard)

**Context:** We have tried 38+ fixes, 50+ config variants, comprehensive experiment tracks A-E. The remaining high-leverage approaches include TTA (test-time augmentation), multi-scale FPN sharing for PSR, sparse MoE backbone, and action-sequence reframing. TTA is the simplest: horizontal flip + multi-scale testing (0.8x, 1.0x, 1.2x) at inference, no retraining required. Standard in detection benchmarks (Detectron2, MMDetection), TTA improves COCO mAP by 2-5% on published results.

**Question:** Does test-time augmentation (horizontal flip + multi-scale at 0.8x, 1.0x, 1.2x) improve detection mAP50 from 0.317 to an estimated 0.35-0.39 on the 250-batch validation set, using the existing epoch 11 checkpoint with zero additional training?

**Why this matters:** TTA requires NO retraining, NO architecture changes, NO data collection. It is a 1-2 hour code change. If it provides 0.03-0.07 mAP improvement, it is the highest-leverage change per unit time in the entire project. Standard TTA baselines in object detection (Detectron2's multi-scale test) show 2-5% mAP improvement consistently across COCO, LVIS, and Cityscapes.

**Hypothesis:** TTA improves det_mAP50 by 0.03-0.07 (from 0.317 to 0.35-0.39). The improvement is greatest on transitional states (channels 22, 16, 19 gain > 0.05 AP) because multi-scale helps detect small components at the detection margin, and flip augmentation provides a second view to resolve component ordering ambiguity.

**Constraints:** Implement TTA in evaluate.py. No retraining. No additional data. Time: 1-2 hours. Runs on RTX 3060 (idle, `111-overview.md:249-254`).


**Cross-references:** This question connects to:
- Q41 (D1: YOLOv8m eval): establishes the direct comparison baseline
- Q30 (gradient detachment): quantifies multi-task interference cost
- Q3 (BiFPN): structural enhancement to detection architecture
- Q38 (pseudo-labeling): data-level detection improvement
- Q37 (synthetic data): training-data-level detection improvement

**Implementation details:**
- Code change location: evaluate.py (NMS replacement for Q1, config for Q2, model.py for Q3, roi_detector.py for Q4, losses.py for Q5)
- Estimated code lines: 10-50 depending on complexity
- Testing: run on 250-batch subsample first (2 min), then full 38K eval (1h)
- GPU: RTX 3060 (idle) for inference-only; RTX 5060 Ti for retraining
- Dependencies: none for inference-only; checkpoint.pt required for retraining

**Metric budget analysis:**
If this experiment succeeds (+0.02 mAP50_pc), the improvement combines with:
- Q3 BiFPN (+0.02-0.04 mAP50_pc): additive or synergistic?
- Q26 ImageNet pretrain (+0.02-0.05 mAP50_pc): independent?
- Q37 synthetic data (+0.04-0.07 mAP50_pc): likely additive
The total potential detection improvement from all successful experiments
could reach 0.08-0.16 mAP50_pc (from 0.506 to 0.58-0.66), closing
15-30% of the YOLOv8m gap without changing the backbone architecture.

**Expected counter-argument (reviewer):**
'Why didn't you run this simple experiment earlier?'
Response: We prioritized engineering stability (F1-F22b fixes) over
hyperparameter optimization. Now that training is stable, we are
systematically closing remaining gaps.
**Validation:** Implement TTA (horizontal flip + multi-scale 0.8, 1.0, 1.2). Run on epoch 11 checkpoint over the full 38K-frame validation set (D3 simultaneously). Compare det_mAP50, mAP50_pc, per-class AP against single-scale baseline. Measure speed impact (TTA = ~6x inference cost). Success criterion: mAP50 improvement >= 0.02. If TTA provides < 0.01 improvement, document that detection is not TTA-limited and the bottleneck lies in backbone feature quality, not inference robustness.
























