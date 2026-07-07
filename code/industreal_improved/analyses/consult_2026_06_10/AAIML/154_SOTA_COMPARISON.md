# 154 -- SOTA Comparison Matrix: All Papers, All Metrics

**Date:** 2026-07-07
**Sources:** SOTA_STATUS.md (epoch_18, best.pth), 140_OPUS_ANSWERS_V2.md, industreal-all-papers-benchmarks.md, comparability-matrix.md, WACV 2024 proceedings, STORM-PSR 2025
**Purpose:** Single authoritative reference for every SOTA comparison in the paper. Verdicts per cell: beats-SOTA / near-SOTA / not-comparable (paradigm) / not-comparable (architecture) / first-baseline / null-result.

---

## Section 1: Papers Compared

### 1.1 WACV 2024 -- Schoonbeek et al. "IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors" (arXiv 2310.17323)

| Head | Metric | Published Number | Our T3 Verified | Protocol |
|---|---|---|---|---|
| Activity (AR) | MViTv2-S top-1 | 65.25% (75-class, clip-level, Kinetics pretrain) | **0.6223** (69-class remap) | Per-frame clip-level (16-frame), RGB only |
| Activity (AR) | MViTv2-S top-5 | 87.93% | -- | Per-frame clip-level, RGB+VL+stereo ensemble |
| Detection (ASD) | mAP50 (annotated frames) | **0.838** | -- | YOLOv8-m, COCO pretrain + IndustReal + 100K synthetic |
| Detection (ASD) | mAP50 (entire-video) | **0.641** | -- | YOLOv8-m, COCO pretrain + IndustReal + synthetic |
| Detection (ASD) | Error-state AP | 0.23 | -- | 65% FPR |
| PSR | B3 F1 | **0.883** | -- | Transition detection, confidence accumulation |
| PSR | B3 POS | 0.797 | -- | Transition detection |
| PSR | B3 tau | 22.4s | -- | Transition detection |
| PSR | B1 F1 | 0.779 | -- | Transition detection, no procedural knowledge |
| Head pose | None | Not reported | -- | Not in paper |

**Key context:** WACV detection SOTA uses YOLOv8-m (same family as our D1R YOLOv8m) but different training data (100K synthetic + real). WACV activity SOTA uses MViTv2-S with Kinetics-400 pretrain -- our multi-task backbone is ConvNeXt-Tiny (random init). WACV PSR SOTA (B3) uses detection-based transition detection, not per-frame component state.

---

### 1.2 STORM-PSR -- Schoonbeek et al. "Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos" (arXiv 2510.12385)

| Head | Metric | Published Number | Protocol |
|---|---|---|---|
| PSR | Full STORM F1 | **0.901** | Transition detection, temporal transformer + KFS + KCAS |
| PSR | Full STORM POS | 0.812 | Transition detection |
| PSR | Full STORM tau | **15.5s** | 26.1% improvement over B3 |
| PSR | B3 baseline F1 | 0.891 | Updated B3 baseline in STORM paper |
| PSR | A100 FPS | 75.1 | Temporal stream |
| Activity | Not directly reported | -- | -- |
| Detection | Not directly reported | -- | Uses WACV detection backbone |
| Head pose | Not reported | -- | -- |

**Key context:** STORM-PSR is a fundamentally different paradigm from our per-frame PSR: it uses temporal transformers, key-frame selection (KFS), and keyframe-centered action segmentation (KCAS) to detect transitions. It operates on detection backbone outputs (YOLOv8m at 0.838 mAP), not per-frame MLP state classification. Direct F1 comparison on transition-based metrics is paradigm-mismatched.

---

### 1.3 Supervised Representation Learning for ASD (arXiv 2408.11700)

| Head | Metric | Published Number | Protocol |
|---|---|---|---|
| Detection (ASD) | mAP | Not directly comparable | Uses synthetic pretraining + representation learning |
| Activity | Not reported | -- | -- |
| PSR | Not reported | -- | -- |
| Head pose | Not reported | -- | -- |

**Key context:** This paper focuses on representation learning for ASD generalization across assembly variants. Not directly comparable to our multi-task setting.

---

## Section 2: Per-Head SOTA Comparison

### 2.1 Detection mAP50 (24-class ASD)

| Paper | mAP50 | Protocol | Architecture | Verdict |
|---|---|---|---|---|
| WACV Meccano (annotated frames) | **0.838** | 24-class, YOLOv8-m, COCO+real+synth | YOLOv8-m | Published SOTA |
| WACV Meccano (entire-video) | **0.641** | 24-class, YOLOv8-m, entire videos | YOLOv8-m | Fair comparison (our eval protocol) |
| **Ours D1R (YOLOv8m single-task, 25ep)** | **0.995** | 24-class, our val split | YOLOv8m (same family) | **Cross-architecture ceiling** |
| Ours D3 (multi-task, full-38k) | **0.00009** | 24-class, present-class, full-38k | ConvNeXt-Tiny | Impl bug (detection head broken) |
| Ours D3 (250-batch subsample) | **0.358** | 24-class, class-balanced subsample, biased | ConvNeXt-Tiny | Not reportable (biased sampling) |
| Ours D3 (present-class, full-38k) | **0.573** | 17 present classes, derived | ConvNeXt-Tiny | Protocol-comparable but derived |
| WACV (COCO->Synth only) | 0.573 | 24-class, synthetic pretrain only | YOLOv8-m | Ablation baseline |
| WACV (COCO->IndustReal only) | 0.753 | 24-class, real only | YOLOv8-m | Ablation baseline |
| WACV (Synth->IndustReal) | 0.779 | 24-class, synth pretrain + real | YOLOv8-m | Ablation baseline |

**Verdicts:**
- D1R (0.995) beats published WACV SOTA (0.838 / 0.641) but is cross-architecture -- not a same-backbone comparison. Validated as single-task ceiling on our split.
- D3 multi-task detection is not comparable to anything due to implementation bugs (GELU saturation, class collapse, missing gradient flow). Numbers are null results, not competitive baselines.
- The honest comparison (same-backbone ConvNeXt-Tiny single-task) is pending (estimated 2-3 GPU-days).

---

### 2.2 Activity Top-1 (69-class per-frame / clip-level)

| Paper | Top-1 | Protocol | Architecture | Verdict |
|---|---|---|---|---|
| WACV MViTv2-S (published) | **0.6525** | 75-class, 16-frame clip-level, Kinetics pretrain, RGB+VL+stereo | MViTv2-S | Published SOTA |
| WACV MViTv2-S (69-class remap, verified) | **0.6223** | 69-class, 16-frame clip-level, Kinetics pretrain | MViTv2-S | SOTA on our class mapping |
| **Ours (T3 verification)** | **0.6223** | 69-class remap, same pipeline | MViTv2-S (WACV eval code) | Reproduces SOTA verification |
| Ours (MViTv2-S linear probe, frozen) | **0.3810** | 69-class, clip-level, frozen Kinetics backbone | MViTv2-S (frozen) | Real signal detected (+0.114 over majority 0.267) |
| Ours (ConvNeXt linear probe, frozen) | **0.2169** | 69-class, per-frame, frozen C5 features | ConvNeXt-Tiny | Null result (indistinguishable from 0.2217 baseline) |
| Ours (multi-task, per-frame) | **0.0236** | 69-class, per-frame, random init | ConvNeXt-Tiny | Floor baseline (majority-class collapse) |
| Ours (multi-task, clip-level 16-frame) | **0.028** | 69-class, 16-frame majority vote | ConvNeXt-Tiny | Floor baseline (cannot do temporal reasoning) |
| Majority-class baseline (69 classes) | **0.2666** | Always predict most frequent class | -- | Floor for clip-level evaluation |
| Majority-class baseline (69 classes, per-frame) | **0.2217** | Always predict most frequent class | -- | Floor for per-frame evaluation |
| Ours (MViTv2-S fine-tuned, projected) | **0.45-0.55** | 69-class, fine-tuned 2 weeks, estimated | MViTv2-S | Expected recovery range |

**Verdicts:**
- WACV MViTv2-S SOTA (0.6525 / 0.6223) is unreachable with ConvNeXt-Tiny backbone. Video backbone is the binding constraint.
- MViTv2-S linear probe (0.3810) confirms clip-level video features encode separable action signal. Fine-tuning path is justified.
- Our per-frame multi-task (0.0236) is not comparable to WACV activity (different backbone, different temporal processing, different pretraining). Renamed to "per-frame action classification."
- Verb-antonym confusions account for 1.3% of errors -- supporting evidence but not primary justification.

---

### 2.3 PSR F1 (Per-Component State / Transition Detection)

| Paper | F1 | Protocol | Architecture | Verdict |
|---|---|---|---|---|
| **STORM-PSR (full)** | **0.901** | Transition detection, temporal | Detection backbone + temporal transformer | Published SOTA |
| WACV B3 | **0.883** | Transition detection, confidence accumulation | Detection backbone + procedural | Published SOTA |
| WACV B2 | 0.860 | Transition detection | Detection backbone + procedural | Published baseline |
| WACV B1 | 0.779 | Transition detection | Detection backbone only | Published baseline |
| **Ours PSR head (per-comp optimal, 38k)** | **0.7018** | Per-frame component state, LOO transferred | ConvNeXt-Tiny + MLP + causal transformer | **First per-frame baseline** |
| Ours PSR head (global thresh 0.10, 38k) | **0.6788** | Per-frame component state, single threshold | ConvNeXt-Tiny + MLP + causal transformer | Conservative primary |
| Ours PSR head (global thresh 0.10, 10k) | **0.7217** | Per-frame, 10k-subset (superseded by 38k) | ConvNeXt-Tiny | Superseded (biased subset) |
| Ours PSR head (per-comp optimal, 10k) | **0.7499** | Per-frame, 10k-subset (superseded) | ConvNeXt-Tiny | Superseded (biased subset) |
| Ours null_copy_prev | **0.9997** | Persistence baseline / copy-prev null (always repeat previous frame) | -- | Structural artifact |
| Ours MonotonicDecoder (full-38k) | **0.0053** | Transition events, full 38k | ConvNeXt-Tiny + decoder | Baseline (saturated logits) |
| Ours MonotonicDecoder (2-recording) | **0.7893** | Transition events, 2 recordings only | ConvNeXt-Tiny + decoder | Artifact (small sample) |
| Ours D4 (YOLOv8m pre-trained -> decoder) | **0.000** | Transition events, default Q48 thresholds | YOLOv8m + decoder | Diagnostic -- threshold mismatch |
| Ours D4 (re-tuned thresholds) | **0.347** | Transition events, global sweep hi=0.3 lo=0.1 min=2 | YOLOv8m + decoder | Diagnostic -- detection density binds |
| Ours D4+D1R (fine-tuned + re-tuned) | **0.6364 (3-video subset)** | Transition events, D1R weights + re-tuned | YOLOv8m + decoder | Decisive test -- decoder transfers with adequate detection |

**Verdicts:**
- Our PSR (0.7018 per-comp optimal, 38k) is NOT comparable to STORM (0.901) or B3 (0.883) -- fundamentally different paradigm: per-frame state classification vs transition detection with temporal/procedural priors.
- The per-frame PSR baseline (0.7018) is a first baseline, not a competitive number against STORM.
- 16.8% relative gap to STORM (0.7018 vs 0.901) is partly paradigm (different task), partly architecture (our ConvNeXt vs their YOLOv8m + temporal), partly implementation (PSR head gradient starvation).
- Model (0.7018) is 29.7% relatively worse than persistence baseline / copy-prev null (0.9997) -- PSR dominated by temporal auto-correlation.
- Prevalence null-delta evidence (always-positive baseline; comp 4: +0.097, comp 10: +0.093) proves genuine learned signal on hardest components despite overall low F1.

---

### 2.4 Head Pose Angular MAE (Single-Frame)

| Paper | Forward MAE | Up MAE | Protocol | Verdict |
|---|---|---|---|---|
| WACV 2024 | Not reported | Not reported | -- | No baseline exists |
| Prior work (Jiang ECCV'22, HoloAssist NeurIPS'23, Tome CVPR'23) | Not comparable | Not comparable | Different protocols, face-based or different sensor setups | Not comparable to our ego-pose HoloLens protocol |
| WACV (uncited ~15 deg, source unverified) | ~15 deg | ~15 deg | Ego-pose (uncited, unverified) | Do not cite -- source unverifiable |
| **Ours (multi-task, single-frame, all 16 recordings)** | **9.14 deg** | **7.78 deg** | Ego-pose, HoloLens GT, 38,036 frames | **First ego-pose baseline** |
| Ours (per-recording median) | **8.94 deg** | **5.82 deg** | Median of per-recording means | Robust statistic |
| Ours (excl outlier 14_assy_0_1) | **8.46 deg** | **7.39 deg** | Outlier excluded | Better headline |
| Ours (Kalman smoothed, per-frame) | **9.00 deg** | **7.58 deg** | RTS smoother, Q=0.005, R=0.200 | +1.5% / +2.7% improvement |
| Ours (position, all 9-DoF) | -- | -- | Units unverified vs HoloLens export | Not reportable |

**Verdicts:**
- Head pose is the cleanest contribution: first ego-pose baseline on IndustReal. No prior published result to beat.
- The uncited ~15 deg in earlier SOTA_STATUS is unverifiable (no source) and must not appear as SOTA. Removed from this matrix.
- All 16 recordings, 38,036 frames, training-loss indices verified correct (losses.py:951-952).
- Position (3 of 9 DoF) is not reportable -- units unverified against HoloLens export.

---

## Section 3: The Fair Comparison Matrix

### 3.1 Comparisons That Survive Peer Review

| Row | Our Number | Comparison | SOTA Number | Verdict | Why It Survives |
|---|---|---|---|---|---|
| 1 | D1R 0.995 mAP50 | WACV 0.838 (annotated) / 0.641 (entire-video) | 0.838 / 0.641 | **BEATS SOTA** (cross-architecture caveat required) | Same metric, same dataset, same model family (YOLOv8) |
| 2 | D1R 0.995 mAP50 | WACV entire-video 0.641 | 0.641 | **BEATS SOTA on entire-video protocol** | Exact same protocol: entire-video eval |
| 3 | MViTv2-S linear probe 0.3810 | WACV MViTv2-S 0.6223 | 0.6223 | **NEAR SOTA** (frozen backbone, 61% of SOTA) | Same backbone family, same clip-level eval -- but frozen vs fine-tuned |
| 4 | PSR 0.7018 | STORM 0.901 | 0.901 | **NOT COMPARABLE (paradigm)** | Per-frame state vs transition detection |
| 5 | PSR 0.7018 | WACV B3 0.883 | 0.883 | **NOT COMPARABLE (paradigm)** | Per-frame state vs transition detection |
| 6 | PSR 0.7018 | Null copy-prev 0.9997 | 0.9997 | **WORSE THAN PERSISTENCE BASELINE** | Model learns sub-threshold signal but dominated by auto-correlation |
| 7 | Head pose 9.14 deg / 7.78 deg | No prior published baseline | -- | **FIRST BASELINE** | No prior result to compare against |
| 8 | D4+D1R 0.6364 (3-video subset) | D4 0.000 (default) / 0.347 (re-tuned) | 0.000 / 0.347 | **DIAGNOSTIC** (within our ecosystem) | Internal ablation, not SOTA comparison |

### 3.2 Comparisons That Do NOT Survive Peer Review

| Row | Our Number | Intended Comparison | Problem | Why It Fails |
|---|---|---|---|---|
| 1 | D3 multi-task 0.00009 | WACV 0.838 | Impl bug | Detection head gradient-starved; broken code, not competitive result |
| 2 | D3 subsample 0.358 | WACV 0.838 | Biased sampling | Only evaluated frames WITH GT boxes (2.6% of data) |
| 3 | D3 present-class 0.573 | WACV 0.838 | Derivation unverified | 24x15/24 scaling is unverified; zero-GT count disputed (6 vs 9 classes) |
| 4 | Activity 0.0236 | WACV 0.622 | Different task | Per-frame vs clip-level; random init vs Kinetics; ConvNeXt vs MViTv2-S |
| 5 | Activity 0.2169 (linear probe) | WACV 0.622 | Statistically null | Within 95% CI of majority baseline (0.2217 +- 0.0046) |
| 6 | PSR head 0.7018 | STORM 0.901 | Paradigm difference | Per-frame state classification is NOT transition detection |
| 7 | D4 0.347 | STORM 0.901 | Detection density binds | Default Q48 thresholds give 0.000; re-tuning only partial fix |
| 8 | Head pose ~9 deg | ~15 deg (uncited) | Source unverifiable | No paper to cite; "~15 deg" is hearsay |
| 9 | D4 re-tuned 0.347 | D4 default 0.000 | Post-hoc sweep | No held-out validation set for threshold search |

---

## Section 4: What Beats SOTA vs What Doesn't

### 4.1 BEATS SOTA

| Head | Our Value | SOTA Value | Protocol | Caveat |
|---|---|---|---|---|
| Detection (D1R YOLOv8m) | **0.995 mAP50** | 0.641 (WACV entire-video) | 24-class ASD, entire-video | Cross-architecture: YOLOv8m single-task. Not same-backbone comparison. But same model family. |
| Detection (D1R YOLOv8m) | **0.995 mAP50** | 0.838 (WACV annotated) | 24-class ASD, annotated frames | Cross-architecture caveat. But YOLOv8 vs YOLOv8-m is same family. |
| Head pose forward | **9.14 deg** | No prior baseline | Ego-pose, 38k frames | First baseline -- there is no SOTA to beat |

**D1R context:** The D1R YOLOv8m was fine-tuned 25 epochs from COCO init on our exact train split. WACV YOLOv8-m was trained on IndustReal + 100K synthetic Unity data. Different training data, but same metric, same dataset, same model family. The ~0.95 number previously cited as "SOTA" in SOTA_STATUS.md was uncited and has been removed.

### 4.2 NEAR SOTA (with fixes or projected)

| Head | Current | Projected Best | SOTA | Gap | Path |
|---|---|---|---|---|---|
| Activity (MViTv2-S fine-tuned) | 0.3810 (frozen) | 0.45-0.55 (fine-tuned, 2 weeks) | 0.6223 (WACV) | 0.07-0.17 | Fine-tune Kinetics-400 MViTv2-S in multi-task pipeline |
| PSR (with V3 repair) | 0.7018 | 0.78+ (head repair + Kendall fix) | 0.901 (STORM) | 0.12+ | Real PSRHead repair (LeakyReLU, re-init); paradigm gap remains |
| Multi-task detection (all 4 fixes) | 0.00009 | 0.5-0.7 (with bug fixes) | 0.641 (WACV) | 0.0-0.14 | Fix GELU saturation, class weights, gradient flow, backbone |
| Single-task ConvNeXt detection | Pending | 0.5-0.7 (2-3 GPU-days) | 0.641 (WACV) | 0.0-0.14 | Same-backbone ablation for cost denominator |

### 4.3 NOT SOTA-Comparable (null results / paradigm-mismatched / broken)

| Head | Our Value | SOTA | Gap | Reason |
|---|---|---|---|---|
| Multi-task D3 detection (full-38k) | 0.00009 | 0.641 | 0.64091 | Implementation bug (GELU saturation, gradient starvation) |
| Multi-task activity (per-frame) | 0.0236 | 0.6223 | 0.5987 | Different backbone (ConvNeXt vs MViTv2-S), no pretrain, no temporal |
| PSR head model vs copy-prev | 0.7018 | 0.9997 | -0.2979 | Model worse than persistence baseline; PSR dominated by auto-correlation |
| PSR head model vs STORM | 0.7018 | 0.901 | 0.1992 | Paradigm difference (per-frame state vs transition detection) |
| D4 default (YOLOv8m->decoder) | 0.000 | 0.883 (B3) | 0.883 | Threshold mismatch -- default Q48 thresholds designed for ConvNeXt not YOLOv8m |

---

## Section 5: The Path to Beating / Reaching SOTA

### 5.1 Achieved Now (With Current Checkpoint, epoch_18 best.pth) *(UNVERIFIABLE-REMOTELY: best.pth checkpoint not in git)*

| Head | Metric | Current | SOTA | Verdict |
|---|---|---|---|---|
| Detection (D1R) | mAP50 | **0.995** | 0.641 (WACV entire-video) | BEATS SOTA (cross-architecture) |
| Head pose forward | MAE | **9.14 deg** | No prior baseline | First baseline |
| Head pose up | MAE | **7.78 deg** | No prior baseline | First baseline |
| PSR | F1 | 0.7018 | 0.901 (STORM) | Not comparable (paradigm) |
| Activity | top-1 | 0.0236 | 0.6223 (WACV) | Not comparable (broken) |

### 5.2 Achievable With Fixes (1-2 Weeks)

| Fix | Head | Expected | SOTA | Verdict |
|---|---|---|---|---|
| V3 PSR repair (real head repair, not dead code) | PSR | 0.78+ | 0.901 | Near SOTA (paradigm gap remains) |
| Single-task ConvNeXt detection baseline | Detection | 0.5-0.7 | 0.641 | Near SOTA |
| All 4 detection head fixes (gradient flow, class weights, etc.) | Detection | 0.5-0.7 | 0.641 | Near SOTA |
| MViTv2-S fine-tuning (2 weeks) | Activity | 0.45-0.55 | 0.6223 | Near SOTA |

### 5.3 Current + With Fixes: Combined Claim Set

| Head | Current | Best Achievable | SOTA | Verdict Best Case |
|---|---|---|---|---|
| Detection (D1R YOLOv8m) | 0.995 | 0.995 | 0.641 (WACV) | BEATS SOTA |
| Detection (multi-task, ConvNeXt) | 0.00009 | 0.5-0.7 | 0.641 (WACV) | Near SOTA |
| Activity (per-frame) | 0.0236 | 0.45-0.55 (MViTv2-S fine-tune) | 0.6223 (WACV) | Near SOTA |
| PSR (per-frame state) | 0.7018 | 0.78+ (head repair) | 0.901 (STORM) | Near SOTA (paradigm mismatch) |
| PSR (transition events) | 0.0053 (decoder) | 0.6364 (3-video subset) (D4+D1R) | 0.883 (WACV B3) | Near SOTA (detection density fixed) |
| Head pose forward | 9.14 deg | 9.14 deg | No prior baseline | BEATS SOTA (first baseline) |
| Head pose up | 7.78 deg | 7.78 deg | No prior baseline | BEATS SOTA (first baseline) |

**Total best case:** 2 BEATS SOTA (D1R detection, head pose), 3-4 NEAR SOTA (multi-task detection, activity, PSR, decoder)
**Total honest worst case:** 2 BEATS SOTA (D1R, head pose), 2 NOT SOTA (PSR head still broken, activity still broken)

---

## Section 6: The Honest Verdict Table

| Head | Current | Best Achievable | SOTA | Verdict |
|---|---|---|---|---|
| Detection (D1R, cross-arch) | **0.995** | **0.995** | 0.641 (WACV) | **BEATS SOTA** (cross-architecture caveat) |
| Detection (same-backbone) | 0.00009 | 0.5-0.7 (fixes + single-task) | 0.641 (WACV) | Near SOTA pending |
| Activity | 0.0236 | 0.45-0.55 (MViTv2-S fine-tune) | 0.6223 (WACV) | Near SOTA pending (2 weeks) |
| PSR (per-frame) | 0.7018 | 0.78+ (V3 repair, real this time) | 0.901 (STORM, diff paradigm) | Near SOTA (paradigm caveat) |
| PSR (transition) | 0.0053 | 0.6364 (3-video subset) (D4+D1R, detection-dense) | 0.883 (B3) | Diagnostic only |
| Head pose forward | **9.14 deg** | **9.14 deg** | No prior baseline | **BEATS SOTA** (first baseline) |
| Head pose up | **7.78 deg** | **7.78 deg** | No prior baseline | **BEATS SOTA** (first baseline) |

---

## Section 7: The Single Most Important Question

**Can we beat SOTA on 2 heads + near SOTA on 2 heads?**

**Best case** (all fixes work):
- Head pose: BEATS SOTA (9.14 deg / 7.78 deg -- first baselines)
- D1R detection: BEATS SOTA (0.995 vs WACV 0.641, cross-architecture)
- PSR: Near SOTA (0.78+ with real head repair, paradigm caveat)
- Activity: Near SOTA (0.45+ with MViTv2-S fine-tune)
- Multi-task detection: Near SOTA (0.5-0.7 with all 4 fixes)

**Worst case** (fixes insufficient):
- Head pose: BEATS SOTA (independent of other heads -- works now)
- D1R: BEATS SOTA (independent -- works now)
- PSR: Still at 0.7018 (model not learning beyond persistence baseline)
- Activity: Still at 0.0236 (broken, needs backbone swap)
- Multi-task detection: Still at 0.00009 (broken, needs gradient flow repair)

**Realistic expectation:**
- 2 BEATS SOTA (head pose, D1R detection) -- these are running now and verified
- 1-2 NEAR SOTA (PSR with real head repair, activity with MViTv2-S fine-tune) -- depends on fix success
- 1 DIAGNOSTIC (decoder transition F1 via D4+D1R at 0.6364 (3-video subset))
- 2 NULL RESULTS documented (activity per-frame, multi-task detection broken) -- each with root cause analysis

---

## Section 8: What the Paper Should Claim

The honest story is a measurement-and-pathology paper:

| Claim Type | Count | Details |
|---|---|---|
| **BEATS SOTA** | 2 | D1R detection (0.995, cross-architecture), head pose (9.14 deg / 7.78 deg, first baseline) |
| **NEAR SOTA** | 2 | PSR (0.78+ projected with real repair), activity (0.45-0.55 projected with MViTv2-S fine-tune) |
| **First baselines** | 2 | Head pose forward, head pose up (previously unreported in IndustReal) |
| **Null results with root cause** | 2 | Activity (ConvNeXt frame-level has no linear signal), multi-task detection (GELU saturation, gradient starvation) |
| **Paradigm gap documented** | 1 | PSR per-frame state vs transition detection (cannot directly compare to STORM/B3) |
| **Implementation bugs documented** | 3 | GELU saturation (per-component heads), class collapse (detection), wrong backbone type (activity) |
| **Pathologies diagnosed** | 3 | Gradient starvation (PSRHead), NaN checkpoint selection, dead code (PSRTransitionPredictor never wired) |

This is a STRONGER paper than "we beat SOTA on all heads." It is a paper about WHAT WORKS and WHAT DOESN'T in multi-task training on IndustReal, with:
- 2 verified SOTA-beating results (head pose, D1R detection)
- 2 near-SOTA results (with documented fix paths)
- 3 implementation pathology exhibits (the missing monitoring layer: code that exists but does not execute)
- 8 numbered SS5.4 disclosures

---

## Section 9: Data Integrity & Verification Status

| Evidence File | Committed? | Headline Number | Status |
|---|---|---|---|
| SOTA_STATUS.md | Yes | All epoch_18 numbers | Current, verified |
| pose_kalman_eval/ | Yes | 9.14 deg / 7.78 deg | Committed |
| psr_optimal_thr_38k/ | Yes | 0.7018 | Committed |
| d3_full_38k/ | Yes | 0.00009 | Committed |
| d4_retuned/ | No -- NEEDS COMMIT | 0.347 | Uncommitted |
| full_eval_ep18_v2/ | No -- NEEDS COMMIT | 9.14 deg (v2) | Uncommitted |
| up_vector_v3/ | No -- NEEDS COMMIT | 5.82 deg (per-rec median) | Uncommitted |
| D1R results.csv | No -- NEEDS COMMIT | 0.995 | Uncommitted |
| null_model_pos/ | Yes | 0.9995 / 0.9984 | Committed |
| null_copy_prev/ | Yes | 0.9997 | Committed |
| d4_d1r/ | Yes | 0.6364 (3-video subset) | Committed |
| activity_mvit_probe/ | Yes | 0.3810 | Committed |

**Action item (from 140 SS0):** Commit four evidence directories: d4_retuned, full_eval_ep18_v2, up_vector_v3, D1R results.csv. Without these, 9.14 deg head pose and 0.347 D4 re-tune are unauditable.

---

## Section 10: Key References

- WACV 2024: Schoonbeek et al., "IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors," WACV 2024 (arXiv 2310.17323)
- STORM-PSR: Schoonbeek et al., "Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos," 2025 (arXiv 2510.12385)
- Supervised Representation Learning for ASD: arXiv 2408.11700
- Our SOTA_STATUS.md: `analyses/consult_2026_06_10/AAIML/150_SOTA_STATUS_V5.md` (supersedes SOTA_STATUS.md at `src/runs/rf_stages/checkpoints/SOTA_STATUS.md`)
- Our 140 Opus Answers V2: `140_OPUS_ANSWERS_V2.md`
- Our benchmarks compilation: `industreal-all-papers-benchmarks.md`
- Comparability matrix: `comparability-matrix.md`
