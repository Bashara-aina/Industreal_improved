# 134 Debate — Adversarial Review of Detection Questions

**Review date:** 2026-07-07
**Source file:** `134_DETECTION_DEEP_QUESTIONS.md` (50 questions, 54 KB)
**Reviewer role:** Adversarial detection agent (Agent 6)

---

## A. Five Strongest Challenges to the Detection Narrative

### Challenge 1: The ceiling is cross-architecture, not same-architecture. The cost ratio lacks a proper control.

The file frames YOLOv8m's 0.995 as the detection "ceiling" and uses 0.358/0.995 = 36% to define the multi-task cost (Q14, Q23). But these are different architectures — YOLOv8m (CSPDarknet, TaskAlignedAssigner) vs ConvNeXt-Tiny (RetinaNet-style FPN head). The 0.995 is a cross-architecture ceiling, not the same-architecture single-task baseline. A ConvNeXt-Tiny trained as a single-task detector might achieve only 0.80-0.90 mAP, making the true multi-task cost:

- At 0.85 ceiling: 0.358/0.85 = 42% of ceiling (58% cost)
- At 0.85 ceiling, present-class corrected: 0.573/0.85 = 67% of ceiling (33% cost)

Every cost sentence changes depending on which ceiling we pick. Opus 133 D-5 correctly calls for "Ablation A (single-task same-backbone baselines)" but the file does not adopt this as a central question. Q14 dances around it ("could D3 theoretically exceed it?") but never proposes training the control. Without a same-architecture single-task run, the "36-58% of ceiling" claim is architecture-confounded and a reviewer will spot this.

### Challenge 2: The zero-GT class count discrepancy is a 50% error that materially alters the central claim.

Section 7 Item 6 acknowledges the discrepancy: the file counts 6 zero-GT classes while Opus 133 counts 9. The arithmetic:

- 6 zero-GT classes: present-class mAP = 0.358 x 24/18 = 0.477
- 9 zero-GT classes: present-class mAP = 0.358 x 24/15 = 0.573

The difference (0.096 mAP) changes the cost ratio from "58% of ceiling (42% cost)" to "48% of ceiling (52% cost)" — a material shift that propagates into every narrative sentence. This is listed as an open decision for Opus, but it should be a closed fact before any narrative decision is made. The file has the raw per_class_gt data in D1 metrics.json. Count the non-zero entries, reconcile against Opus's count, and lock the number. Every question that references the 0.358/0.573 framing is provisional until this count is resolved.

### Challenge 3: The D3 full eval produces no detection metrics — the paper's primary detection number comes from a 2.6% subsample with no full-set verification.

Q21 flags that d3_full_eval/metrics.json has no detection fields. Q22-Q28 then discuss the subsample's class-balanced sampling concerns. The cumulative weight of this gap is severe: the main multi-task model's primary detection result (0.358) rests on a ~1,000-frame subsample with class-balanced weighting, while the full 38,036-frame evaluation pipeline silently produces no detection output at all. The file's proposed verification ("run D3 eval with detection enabled") is necessary but not sufficient. The concerns are:

1. Class-balanced subsample may inflate mAP (Q28 estimates "optimistic").
2. The evaluation pipeline has a known NaN bug on full-set (D-1 from 127).
3. The 0.358 number itself may not be reproducible if the subsample draws different batches each time.

A reviewer will demand: full-set evaluation with detection metrics enabled, or a clear explanation of why only 2.6% of the data produces usable metrics.

### Challenge 4: The D4 narrative over-interprets a fragile post-hoc optimum.

Q32-Q40 analyze the D4 threshold sweep where a 145-combination search lifted F1 from 0.000 to 0.347. The file's conclusion that "the decoder is not the bottleneck" and "backbone detection density is the binding constraint" rests on this result. Problems:

1. The F1=0.347 is a post-hoc optimum from a multi-parameter sweep on the same data used for reporting (no held-out validation). Q39-Q40 already show that per-component thresholds (which reduce degrees of freedom) perform WORSE than global, suggesting the global optimum captures noise, not signal.
2. The "not the bottleneck" claim requires proving the decoder would achieve >0.7 with a better backbone. But the actual experiment (YOLOv8m with D1R weights, Q36) was dismissed as "incremental confirmation, not a new finding." With a 0.0004 -> 0.995 mAP gap between the two detection backbones, this is not incremental — it is the decisive test.
3. The verdict.json itself says "marginal benefit" and "threshold-partial." The file's narrative softens this to "proven to work," but the experiment proves the opposite: the decoder is acutely sensitive to backbone detection density and essentially non-functional with any backbone below SOTA detection performance.

### Challenge 5: The "COCO convention" hinge is treated as a 30-minute check, but the comparison is split-confounded regardless.

Q23 and §7 Item 1 both depend on whether WACV uses COCO convention (excluding zero-GT classes from mAP averaging). Opus 133 calls this "a 30-minute check that could shrink the paper's biggest weakness by a third." But the corrected number (0.573 if convention matches) is then compared against WACV's 0.838 or 0.641, which come from a different validation split (random split, likely, based on Q41's analysis). Even if the mAP convention matches, the split confound remains.

The correct statement is: under COCO convention, our present-class mAP is 0.573. WACV's numbers are 0.838 (annotated-frames, random split) and 0.641 (entire-video, random split). We cannot say whether 0.573 at our split is better or worse than 0.641 at their split without a cross-split experiment (Q42). Until that experiment is run, the present-class correction improves the narrative framing but does not establish comparability.

---

## B. Five Evidence Gaps in the Questions

### Gap 1: The D1R training split is unverified.

Q11-Q13 assume D1R was trained on the same recording-aware split as D3. The evidence is in results.csv on the workstation, which is not committed to the repo. The file's own verification step says "commit the D1R results.csv" but does not address: is the D1R training config code-identical to the D3 multi-task config for dataset construction? A single line difference in the split definition would invalidate the ceiling comparison. The D1R training script path, config, and dataset constructor should be verified against the D3 training config in the repo, not assumed.

### Gap 2: The class-balanced subsample inflation factor is unmeasured.

Q22 raises the class-balanced concern and Q28 labels it "optimistic," but neither question quantifies the inflation. The gap between class-balanced subsample mAP and full-set natural-distribution mAP is the inflation factor. Without it, we cannot assess whether 0.358 is representative or 2-3x inflated. The verification step "compare class-balanced mAP against natural-distribution mAP" should have a priority marker — it determines whether the paper's headline detection number is trustworthy.

### Gap 3: The D3 mAP50-95 is entirely unknown.

Q29 flags this and guesses a wider gap (0.2-0.3) than D1R's (0.134). But this is not just missing data — it is a critical characterization. If D3's mAP50-95 gap is >0.3, the detection head is producing poorly-localized boxes (high IoU sensitivity) which points to a different pathology than simple class confusion. If the gap matches D1R's (~0.134), the box quality is comparable and the bottleneck is purely classification. This diagnostic value means mAP50-95 should be computed before the per-class AP breakdown.

### Gap 4: The WACV protocol reimplementation experiment lacks a feasibility check.

Q47 asks whether to reimplement WACV's protocol but does not specify: what code does WACV use? Is their evaluation script available? If not, what would a reimplementation cost? Without scoping the effort, "reimplement WACV's protocol" remains an unbounded ask that never gets done. The question should state: "check if WACV eval code is open-source (it is not, per SOTA_STATUS) — approximate effort is 2-3 days to reimplement from paper description; alternative is protocol disclosure table."

### Gap 5: The D4 per-recording variance is missing.

Q39 asks for per-recording D4 F1 but has no data. Given the YOLOv8m checkpoint produces detections on <1% of frames, per-recording F1 on 16 recordings would likely show extreme variance (0.0 on most recordings, >0.5 on recording 22-heavy ones). Without this breakdown, the "binding constraint" conclusion is unsupported. If D4 F1>0 comes entirely from one recording, the constraint is not "backbone detection density" generically but "class 22 prevalence in specific recordings."

---

## C. Five Alternative Interpretations

### Alternative 1: The 0.358 mAP may be consistent with present-class mAP = 0.573 under COCO convention, but WACV may not use COCO convention.

The file assumes the 0.358-to-0.573 correction depends on WACV's protocol (Q23, §7 Item 1). But the correction also depends on whether our own evaluation uses COCO convention. The `evaluate.py` file computes both `det_mAP50` (24-class mean, zeros included) and `det_mAP50_pc` (present-classes-only, per Opus 133 D-4). If our reported 0.358 is the 24-class mean, and WACV reports present-class mean, then the gap is not 0.483-0.265 but 0.573-0.000 (WACV's present-class mAP on zero-GT classes is undefined, not 0). The comparison needs four cells: both models x both conventions.

### Alternative 2: The D1 v1-v3 mAP=0.0004 may not be a domain-gap issue but a class-mapping bug in the eval script.

Q1-Q10 exhaustively analyze the mAP=0.0004 as split mismatch or domain gap. But there is a simpler explanation: the binary string mapping in `eval_yolov8m.py` shifted by one bit position between the checkpoint's training config and our config.py. The file notes that only class 22 (binary "11101111111") fires — but if the binary string has a different bit ordering in the checkpoint's training config than in our DET_CLASS_NAMES, the mapping is systematically wrong for all classes. Q2 asks about 0-indexed vs 1-indexed but doesn't check whether the BIT ORDER within each binary string matches. A single-bit misalignment would explain why only one class fires (the one where the misalignment happens to produce a valid match).

### Alternative 3: The D3 detection head's poor performance may not be multi-task gradient conflict but a config error (different learning rate, different NMS parameters, different anchor configuration).

Q27 discusses gradient conflict but never considers a simpler explanation: the detection head training config differs from the well-tuned single-task setup. If the detection head uses the multi-task optimizer's global learning rate (which may be suboptimal for detection), if NMS parameters differ, or if the RetinaNet anchor configuration doesn't match the data distribution, the 0.358 could be a training-config artifact rather than structural interference. The equal-gradient-update ablation (from the paper's ablation suite) partially addresses this, but the file doesn't reference it.

### Alternative 4: The D3-to-WACV gap may be entirely explained by the split difference, not detection quality.

Q46 correctly notes that WACV's "entire-video" 0.641 is the relevant comparison for our all-frame eval. Combined with present-class correction (0.573), the gap is 0.641 - 0.573 = 0.068 — a small gap that could be entirely split-driven. The file's Q42 guesses WACV's model would drop by 0.04-0.09 on our split. If the drop is exactly 0.068, our model is at WACV-parity and the narrative changes from "significant gap" to "protocol-comparable." This should be a single controlled experiment, not a guessing exercise.

### Alternative 5: The D4 sweep results (per-component worse than global) may indicate that the PSR decoder components share a common signal, not that the signal is noise.

Q40 interprets the global-better-than-per-component result as "per-component thresholds overfit to noise." An alternative reading: the global threshold works better because the relevant detection signal (class-22 activations) simultaneously triggers multiple PSR component transitions (components that change at the terminal assembly state). The shared signal benefits from a common threshold; per-component tuning fragments it. This suggests the decoder is not "marginally beneficial" (verdict.json language) but is correctly extracting correlated component transitions — just very rarely.

---

## D. Five New Questions the File Should Have Asked

### New Question 1: What is the single-task ConvNeXt-Tiny detection ceiling?

The file uses YOLOv8m's 0.995 as the detection ceiling, but this cross-architecture comparison confounds architecture and multi-task cost. The correct control is: train ConvNeXt-Tiny with only the detection head (no activity/PSR/pose heads) on the same data, same optimizer, same schedule. If this achieves 0.85 mAP, the multi-task cost is 0.358/0.85 = 42% (or 0.573/0.85 = 67% present-class). If it achieves 0.95, the cost is 38% (or 60% present-class). Without this number, every "X% of ceiling" sentence has an unknown denominator.

### New Question 2: Does the D3 eval pipeline silently suppress detection metrics via a config flag or a crash guard?

Q21 reports that D3 full_eval/metrics.json contains no detection fields. The file hypothesizes "crashed or wasn't configured." The distinction matters deeply: if a config flag disabled detection eval (e.g., `EVAL_DETECTION=False`), then detection was never measured and the 0.358 is the only datapoint. If detection eval ran but crashed (producing NaN which was caught and silently dropped), then there is a systematic bug that affects all detection evaluation. The eval config and log files should be checked for a detection-enable flag and for any NaN-handling that could suppress detection results.

### New Question 3: What is the ConvNeXt detection head's per-frame detection rate and confidence distribution?

Q26 asks about detection rate but the answer is "unknown." This is a one-line diagnostic: run the D3 checkpoint on 1000 frames, count detections at conf>=0.01, 0.05, 0.25, 0.5. The detection rate (detections per frame) directly distinguishes between: (a) the head fires densely but with wrong labels (detection rate high, mAP low), and (b) the head fires rarely (detection rate low, mAP low by construction). (a) points to classification confusion; (b) points to missed detections (possibly low confidence from gradient competition). This diagnostic takes 10 minutes and would inform the entire detection narrative.

### New Question 4: Can the D3 detection head run at 640x640 input resolution like YOLOv8m, and how does mAP change?

The file mentions IMG_WIDTH/IMG_HEIGHT from config but does not state the actual resolution. If ConvNeXt-Tiny runs at a lower resolution (e.g., 384x384), part of the mAP gap to YOLOv8m's 640x640 is resolution-driven. A resolution-ablation (same model, multiple input sizes) would separate resolution effects from architecture and training effects. This is critical for the efficiency claim: if our model runs at half the resolution, the resolution gap explains part of the accuracy gap and weakens the cost narrative.

### New Question 5: How many of the 24 ASD classes have measurable AP in the D3 model, even at low thresholds?

Q24 asks for per-class AP breakdown but the answer is deferred. This should be the highest-priority diagnostic because it directly answers: is detection failing uniformly (all classes low) or selectively (some classes work, some don't)? If classes 0-5 have AP>0.5 and classes 6-23 have AP near 0, the problem is class-specific (possibly a data imbalance or a model collapse pattern). If all classes are uniformly low, the problem is system-wide and likely gradient competition. Per-class AP at multiple confidence thresholds would distinguish these cases with a single eval run.

---

## E. Summary — Top 3 Strongest Challenges

These are the three challenges that most threaten the file's analysis and the paper's detection narrative:

1. **Cross-architecture ceiling is not the right ceiling.** The file uses YOLOv8m's 0.995 as the detection ceiling, but this mixes architecture and task count. A ConvNeXt-Tiny single-task baseline is required. Until it exists, every cost ratio in the paper has a confounded denominator.

2. **The D3 full-set evaluation produces no detection metrics.** The paper's primary detection number comes from a 2.6% subsample with class-balanced weighting. The full 38,036-frame evaluation pipeline silently produces no detection output for the multi-task model. This is a sand-through-the-hourglass problem: the entire detection narrative depends on a number that cannot be reproduced on the full validation set.

3. **The D4 experiment does not prove the decoder works.** F1=0.347 from a 145-combination post-hoc sweep, with per-component tuning performing worse than global, does not support the conclusion that "the decoder is not the bottleneck." The decisive experiment (D4 with D1R weights, producing dense detections) was declined as "incremental." The decoder bottleneck question remains open.
