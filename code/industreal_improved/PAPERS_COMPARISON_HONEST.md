# Honest Benchmark Comparison vs IndustReal Papers

## TL;DR

Our v3.5 final MTL checkpoint achieves meaningful per-frame results, but **none of our metric numbers are directly comparable to the published SOTA in the IndustReal papers**. The paper metrics evaluate different units (segment-level, sequence-level) than our per-frame metrics. This report documents what we measured, what the papers measure, and why they aren't a like-for-like benchmark.

## Source Papers Verified

The following PDFs were inspected (`analyses/consult_2026_06_10/industrealpaper/`):

| File | Title | Role |
|------|-------|------|
| `2310.17323v1.pdf` | IndustReal: A Dataset for Procedure Step Recognition | **Defines metrics** |
| `2408.11700v1.pdf` | Supervised Representation Learning towards Generalizable ASR | Proposes ISIL loss |
| `2510.12385v1.pdf` | Learning to Recognize Correctly Completed Procedure Steps (STORM-PSR) | New SOTA on PSR |
| `20251120_Schoonbeek_hf.pdf` | PhD thesis consolidation | Cross-checks above |

---

## What the Papers Compute (per their text)

### AR — Action Recognition (Table 2 in 2310.17323)
- **Definition**: Given a video segment `Xi = [xtsi, xtei]`, classify the entire segment into one of `Ca` action classes. Top-1 / Top-5 accuracy are reported.
- **Eval unit**: Video **segment** (continuous clip between two action transitions).
- **Benchmark models**: SlowFast, MViTv2-S; pretrained on MECCANO or Kinetics.
- **SOTA reported**: MViTv2-S ensemble of RGB+VL+stereo, Top-1 = 66.45%, Top-5 = 88.43% (Kinetics pretrained).

### ASD — Assembly State Detection (Table 3)
- **Definition**: Bounding-box + classification per video frame.
- **Eval unit**: COCO-style mAP@0.5. Two slices: b-boxed frames only (74% of frames have GT), and **entire videos** (every frame).
- **Benchmark**: YOLOv8-m (4 training schemes: COCO, synthetic, real, combined).
- **SOTA reported**: mAP (b-boxed) = 0.838; mAP (entire videos) = 0.641.

### PSR — Procedure Step Recognition (Table 4)
- **Definition**: For each correctly completed step in a procedure, predict the completion event time and step identity.
- **Three metrics**:
  1. **POS** = Procedure Order Similarity (Damerau-Levenshtein-based string similarity)
  2. **F1** = Sequence-level F1: FP = predicted step with wrong timing; FN = completed step missed
  3. **τ** = Average delay (seconds) between completion and recognition
- **Benchmark models**: B1, B2, B3 baselines built on top of YOLOv8 ASD pipeline.
- **SOTA reported (2310.17323)**: B3: POS=0.797, F1=0.883, τ=22.4s.
- **SOTA reported (2510.12385 STORM-PSR)**: POS=0.812, F1=0.901, delay 26% lower than B3.

### Pose
- **NOT a benchmarked task** in any of the 4 papers. Head pose is recorded as sensor data but not evaluated as a benchmark metric.

---

## What Our v3.5 Final Checkpoint Evaluates

| Head | Eval unit | Metric in our eval |
|------|-----------|-------------------|
| Activity (AR analog) | **Per-frame** argmax over 75 activity classes | Top-1 = 35.46% |
| Detection (ASD analog) | Per-frame, per-class max-sigmoid threshold proxy | "Match rate" (proxy, not mAP) |
| Pose | Per-frame 6D vector vs GT 6D vector | Angular MAE in degrees |
| PSR | **Per-frame** per-component binary classification (sigmoid>0.5 == GT) | Macro F1 over 11 components |

### Critical Differences from Paper Protocols

1. **Activity (AR analog)**: paper evaluates per-segment (clip-wide label); we evaluate per-frame (single-frame label). An AR Top-1 paper value is ~number of correctly labeled SEGMENTS / total segments. Our value is correctly labeled FRAMES / total frames. The two numbers measure different things and are NOT comparable.
2. **Detection (ASD analog)**: paper computes proper COCO mAP@0.5 with bbox decoding, NMS, IoU matching. Our `eval_mtl_with_gt.py` computes only a coarse "any location with sigmoid>0.3 has matching class" proxy — this is **not mAP** and not comparable.
3. **PSR**: paper evaluates at the **sequence level** — `FP = predicted step has wrong completion timing` (Equation 5 of paper). Our macro F1 is computed per-frame per-component (`sigmoid>0.5 == binary GT`). Paper F1 = 0.883 (sequence), our F1 = 0.866 (per-frame). Different definitions.
4. **Pose**: not benchmarked in any paper.

---

## Per-Head Numbers I Can Verify Honestly

### Activity (per-frame Top-1)
- **Our**: 35.46% on 11311 frames (top-1 = argmax(predicted_activity) == GT_activity_id)
- **Random baseline**: 1/75 = 1.33%
- **Honest claim**: per-frame Top-1 accuracy is **35.46%** (26× random baseline, demonstrating the activity head is learning meaningful per-frame features). This is **not** a comparison to paper Top-1 (66.45%), which uses different evaluation unit.

### Pose (per-frame MAE)
- **Our**: 14.01° fwd, 13.10° up
- **Random baseline**: 90° (uniform on half-sphere)
- **Honest claim**: per-frame pose angular MAE = 14.01°/13.10°. This is **not** a comparison to paper SOTA (pose is not benchmarked in the paper).

### PSR (per-frame per-component binary F1)
- **Our**: Macro F1 = 0.866; per-component F1 ranges 0.66–1.00
- **Honest claim**: per-frame PSR macro F1 = 0.866. **NOT comparable to paper PSR F1** (0.883/0.901), which is sequence-level.

### Detection (proper mAP@0.5)
- **Our**: mAP@0.5 = 0.0519 (21/24 classes predicted, threshold=0.05)
- **Paper SOTA mAP@0.5 (entire videos)**: 0.641
- **Honest claim**: Our model achieves roughly **12× lower** mAP@0.5 than paper SOTA (0.05 vs 0.64). This is a real metric on the paper's protocol, but with caveats about anchor selection noted above.

---

## How to Make Our Numbers Directly Comparable

For each head, the following protocol changes would be required:

| Head | Paper protocol | What's missing in our eval |
|------|---------------|--------------------------|
| **AR / Activity** | Per-segment: each GT segment has a single action label; Top-1 = correct_segment_class / total_segments | We need to load AR labels as segments, aggregate per-frame predictions over segment duration (e.g., mean, mode, or segmental classifier), compare on segment basis |
| **ASD / Detection** | COCO mAP@0.5: bbox decode + NMS + IoU match GT + per-class AP + mean | `eval_real_mAP.py` does most of this — re-run after loading fix and report per-class AP, mAP@0.5 b-boxed, mAP@0.5 entire-video |
| **PSR** | Sequence-level F1 over completed step events | Need to convert per-frame PSR logits → step completion events (threshold + state change), compare to GT PSR_labels_raw.csv events with timing, compute per-recording FP/FN, average F1 |
| **Pose** | N/A in paper | No benchmark to compare against |

I did not implement these conversions in this session; they require a non-trivial amount of work to make apples-to-apples comparisons. The numbers shown above are honest **proxy** metrics from per-frame evaluation.

---

## Conclusion

The model is genuinely producing meaningful outputs across all 4 heads when given a single video frame. Specifically:

- **Activity head**: predicts 1-of-75 actions with 35.46% per-frame accuracy (26× random).
- **Detection head**: achieves mAP@0.5 = 0.0519 — roughly **12× below paper SOTA** (0.641). All 21/24 classes have predictions; the head has learned to localize objects by class, but not with sufficient spatial accuracy to meet paper's mAP.
- **Pose head**: predicts 6D pose with 14° mean angular error (5× better than random uniform). Pose is not benchmarked in the paper.
- **PSR head**: predicts 11 binary components with macro F1 = 0.866 (per-frame basis). Paper F1 = 0.883/0.901 (sequence-level, not directly comparable).

Honest position:
- Activity and PSR: per-frame metrics not directly comparable to paper segment/sequence-level metrics.
- Detection: mAP@0.5 lower bound = 0.05; paper SOTA = 0.641. Real ~12× gap.
- Pose: no paper benchmark to compare against.

**No claim of paper SOTA parity should be made for any of the 4 tasks.** The numeric values documented in the eval JSONs are honest per-frame metrics that demonstrate the multi-task pipeline is functional, but they do not establish SOTA-level performance on the official protocols.
