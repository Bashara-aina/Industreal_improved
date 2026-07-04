# IndustReal Dataset – Complete Benchmark Metrics Across AR, ASD, and PSR Tasks

## Overview

This report compiles all quantitative evaluation metrics reported for models trained and evaluated on the IndustReal dataset, covering action recognition (AR), assembly state detection (ASD), and procedure step recognition (PSR), including main paper tables and supplementary material. It is structured to be a detailed reference for benchmarking and downstream comparisons on IndustReal.


## Dataset- and Task-Level Statistics

These quantities are not model metrics but describe the benchmarked dataset and tasks; they are included because they define the evaluation regime of all reported models.

- Participants: 27 total; split into 12 train, 5 validation, and 10 test participants (splits are defined by participants, not recordings).
- Recordings: 84 ego-centric videos, assembly and maintenance procedures on a 3D-printed construction-toy car.
- Action recognition:
  - 75 fine-grained action classes (verb–noun combinations).
  - 9,273 annotated action instances.
  - Average action duration: 1.9 ± 1.4 seconds.
  - 24.2% of action instances overlap with at least one other action in time.
- Assembly state detection:
  - 11 high-level components used to encode assembly state sequences, each component labeled as 1 (correct), 0 (not yet installed), or −1 (incorrectly installed).
  - 22 labeled correct assembly states and 27 distinct error states with bounding boxes.
  - 26.9K video frames (≈13% of all frames) annotated for ASD, including 3,569 frames containing error states.
- Procedure step recognition:
  - 724 correct procedure step completions (≈8.6 ± 1.2 correct completions per recording).
  - 38 incorrect procedure step completions.
  - 35 videos (≈42%) contain missing or incorrectly completed procedure steps.
  - 22 distinct correctly executed procedure step orders, plus 26 distinct execution orders containing error states.
  - 38 errors in total, of which 14 are exclusive to validation and test sets, enabling robustness evaluation to unseen mistakes.


## Action Recognition (AR) Metrics

The AR task is defined as classifying short ego-centric video clips into one of 75 action classes, based on RGB or other modalities. The primary metrics reported are Top-1 and Top-5 classification accuracy on the IndustReal test set.

### Architectures and Pretraining

Two architectures are benchmarked:

- SlowFast CNN (SlowFast ) trained via PySlowFast pipeline.
- MViTv2-S transformer (MViTv2 ).

Pretraining regimes:

- Kinetics-pretrained (Kinetics-400/600 action recognition dataset).
- MECCANO-pretrained (industrial-like procedural AR dataset), then fine-tuned on IndustReal.

Modalities:

- RGB (front-facing PV camera).
- Depth (long-throw depth images, 320×288 at 5 FPS).
- Visible light (ambient light sensor images).
- Stereo images (left/right grayscale cameras, 640×480 at 10 FPS).
- Modalities combined in ensembles: RGB + visible light (VL) + stereo.


### Global AR Benchmark (Paper Table 2)

This table reports Top-1 and Top-5 accuracy for six configurations evaluated on IndustReal.

| Model                     | Pretraining / Modalities       | Top-1 acc. (%) | Top-5 acc. (%) |
|---------------------------|---------------------------------|----------------|----------------|
| SlowFast              | MECCANO-pretrained, RGB        | 57.83          | 82.87          |
| SlowFast              | Kinetics-pretrained, RGB       | 60.39          | 85.21          |
| MViTv2                | MECCANO-pretrained, RGB        | 62.43          | 85.62          |
| MViTv2                | Kinetics-pretrained, RGB       | 65.25          | 87.93          |
| SlowFast              | Kinetics-pretrained, RGB+VL+stereo | 62.34      | 85.97          |
| MViTv2                | Kinetics-pretrained, RGB+VL+stereo | 66.45      | 88.43          |

Observations:

- MViTv2 consistently outperforms SlowFast across corresponding settings (same pretraining and modality combination).
- MECCANO pretraining does not improve downstream performance on IndustReal relative to Kinetics pretraining.
- Combining RGB, visible light, and stereo modalities in an ensemble raises Top-1 and Top-5 accuracy for both architectures relative to RGB-only, indicating complementary information across modalities.


### Per-Modality AR Benchmark (Supplementary Table 5)

The supplementary material reports separate AR performance for each modality and architecture, fine-tuned from Kinetics-pretrained models on IndustReal.

| Model          | Modality       | Top-1 acc. (%) | Top-5 acc. (%) |
|----------------|----------------|----------------|----------------|
| SlowFast   | RGB            | 60.39          | 85.21          |
| SlowFast   | Depth          | 43.20          | 73.98          |
| SlowFast   | Visible light  | 53.75          | 81.48          |
| SlowFast   | Stereo         | 57.72          | 83.03          |
| MViTv2     | RGB            | 65.25          | 87.93          |
| MViTv2     | Depth          | 49.08          | 76.51          |
| MViTv2     | Visible light  | 58.59          | 83.50          |
| MViTv2     | Stereo         | 58.86          | 83.55          |

Key points:

- RGB remains the strongest single modality for AR, with depth significantly weaker due to lower resolution and frame rate.
- Stereo images and visible light each provide intermediate performance; neither individually matches RGB but both contribute positively in ensembles.
- Across modalities, the transformer model MViTv2 outperforms the SlowFast CNN by roughly 5 percentage points in Top-1 accuracy.


## Assembly State Detection (ASD) Metrics

ASD is formulated as an object detection-style problem: given a frame, detect bounding boxes and classify assembly states from a set Za of 22 defined states plus error states. The main metrics are mean average precision (mAP), evaluated either on frames with ASD annotations or on all frames in test recordings.

### Architecture and Training Schemes

The benchmark uses YOLOv8-m as the detection backbone.

Training regimes involve combinations of synthetic data generated from CAD models and real IndustReal frames:

- Synthetic-only: model pretrained on COCO, then trained purely on 100K synthetic samples via Unity Perception.
- Real-only: COCO-pretrained YOLOv8-m trained directly on real IndustReal ASD annotations.
- Synthetic → real fine-tuning: YOLOv8-m pretrained on synthetic data, then fine-tuned on IndustReal ASD labels.
- Real + synthetic: training on combined real and synthetic datasets.


### mAP Benchmarks (Paper Table 3)

The following metrics are reported:

- mAP (b-boxed): mAP evaluated only on frames containing ground-truth ASD bounding boxes.
- mAP (entire videos): mAP evaluated over all frames of IndustReal test recordings.

| Training scheme                        | Pre-trained on | Fine-tuned on               | mAP (b-boxed) | mAP (entire videos) |
|----------------------------------------|----------------|-----------------------------|---------------|----------------------|
| Synthetic only                         | COCO          | Synthetic                   | 0.573         | 0.341                |
| Real only                              | COCO          | IndustReal                  | 0.753         | 0.553                |
| Synthetic → real fine-tune            | Synthetic      | IndustReal                  | 0.779         | 0.575                |
| Real + synthetic (combined training)   | COCO          | IndustReal + synthetic      | 0.838         | 0.641                |

Interpretation:

- Adding real IndustReal data consistently improves performance relative to synthetic-only training.
- Combining real and synthetic data yields the highest mAP, both on annotated frames and entire videos, indicating beneficial sim2real transfer.
- There is a performance drop (≈0.20 absolute mAP) when going from annotated frames to entire videos, due to false positives on visually subtle states and error states.


### Error-State Performance and False Positive Behavior

The paper reports error-focused metrics for the best-performing ASD model (Real + synthetic training):

- False positive rate on assembly states containing an error: 65%.
- Average precision (AP) on error states: 0.23.

These figures highlight that although overall mAP is relatively strong, the detector struggles significantly with execution error states, often misclassifying them as correct assemblies.


## Procedure Step Recognition (PSR) Metrics

PSR is defined to estimate, at any time t, the set and order of correctly completed procedure steps using sensory inputs Xt and procedural description P. The evaluation focuses on sequence-level correctness, the balance of false/true predictions, and temporal delay.

### PSR Metrics Definitions

Three metrics are proposed:

1. Procedure Order Similarity (POS):
   - POS is a similarity score in  comparing predicted sequence ˆy and ground-truth sequence y using a weighted Damerau–Levenshtein distance with deletions and insertions only (no substitutions), normalized by the length of y and converted to a similarity.
   - Higher POS indicates closer match in step types and order.

2. F1 score:
   - Defined over procedure steps with:
     - False positives: predicted step ˆsσ(j) before the actual completion time tρ(i), or prediction of a step that never completes.
     - False negatives: completed ground-truth step sρ(j) with action ai not appearing in predicted sequence ˆy.
     - True positives: predicted step ˆsσ(j) at or after completion time tρ(i) for actions actually completed.
   - F1 combines precision and recall of recognized steps, ignoring time lag.

3. Average delay τ:
   - Average temporal difference between recognition times and ground-truth completion times for true positives only.
   - τ is measured in seconds; smaller values indicate more timely recognition.

These metrics are used jointly to characterize both correctness of sequence and timeliness.


### Baseline PSR Implementations

PSR baselines rely on the ASD backbone outputs.

- B1: Directly converts changes in ASD-predicted assembly state into corresponding completed steps, assuming correctness of each change.
- B2: Aggregates confidence over time for candidate step completions until exceeding a threshold T; only then declares step completion.
- B3: Uses the same confidence accumulation as B2 but restricts candidate steps to those expected by the procedure description (procedural knowledge), limiting impossible or unlikely completions.

Each baseline is evaluated with two ASD backbones:

- Real-trained ASD (best-performing YOLOv8-m with real + synthetic training).
- Synthetic-only ASD backbone (trained exclusively on synthetic data).

The synthetic-only variants are denoted B1-S, B2-S, and B3-S.

Additionally, the paper reports that the ASD + PSR pipeline reaches real-time inference at 178 frames per second on an NVIDIA V100 GPU.


### PSR Benchmark – All Recordings (Paper Table 4)

For all recordings (including those without errors), PSR baselines achieve the following metrics:

| Baseline | ASD training         | POS    | F1     | τ (s) |
|----------|----------------------|--------|--------|-------|
| B1       | Real + synthetic     | 0.570  | 0.779  | 14.9  |
| B1-S     | Synthetic-only       | 0.014  | 0.206  | 36.9  |
| B2       | Real + synthetic     | 0.731  | 0.860  | 22.3  |
| B2-S     | Synthetic-only       | 0.240  | 0.573  | 44.4  |
| B3       | Real + synthetic     | 0.797  | 0.883  | 22.4  |
| B3-S     | Synthetic-only       | 0.597  | 0.734  | 49.5  |

Key observations:

- B3 (real-trained ASD) is the strongest overall baseline, achieving POS 0.797 and F1 0.883, with an average delay of 22.4 seconds.
- Synthetic-only training yields much lower POS and F1 scores for B1 and B2, but B3-S shows a strong improvement, highlighting the value of procedural constraints even with imperfect ASD.
- B1 has lower delay but also lower sequence and step-level correctness; B2 and B3 trade increased delay for higher correctness.


### PSR Benchmark – Recordings with Errors Only

For recordings that contain execution or procedural errors, performance drops notably:

| Baseline | ASD training         | POS    | F1     | τ (s) |
|----------|----------------------|--------|--------|-------|
| B1       | Real + synthetic     | 0.480  | 0.698  | 14.4  |
| B1-S     | Synthetic-only       | 0.000  | 0.174  | 48.4  |
| B2       | Real + synthetic     | 0.636  | 0.784  | 20.2  |
| B2-S     | Synthetic-only       | 0.107  | 0.516  | 60.5  |
| B3       | Real + synthetic     | 0.731  | 0.816  | 20.4  |
| B3-S     | Synthetic-only       | 0.571  | 0.731  | 71.4  |

Insights:

- Error-containing recordings are substantially harder: both POS and F1 are lower compared to all recordings for every baseline.
- Delays generally increase for synthetic-only ASD (B1-S, B2-S, B3-S), indicating that recognition is both less timely and less accurate in the presence of errors.
- B3 again achieves the best balance of sequence correctness and F1 on error recordings, demonstrating that incorporating procedural knowledge improves robustness, though performance is still far from perfect.


### Qualitative PSR–ASD Behavior in Error States

The paper provides illustrative examples showing how ASD and PSR baselines misinterpret error states:

- Case (a) – Incorrect front-rear pin orientation:
  - Ground truth: incorrect orientation of the front–rear chassis pin.
  - ASD predicts a correct state with installed base, rear-rear pin, and rear chassis.
  - PSR baseline B3 infers completion of “Installed rear-rear chassis pin” and “Installed rear chassis,” although the pin is actually incorrect.

- Case (b) – Incorrect brace fastening using nut instead of screw:
  - Ground truth: front brace fastened with a nut rather than the expected screw.
  - ASD predicts entire assembly procedure correctly installed.
  - PSR baseline B3 declares completion of steps including “Installed front wheel assembly,” “Installed front bracket screw,” and “Installed front bracket,” failing to detect incorrect fastening.

These examples highlight that even the strongest PSR baseline is tightly coupled to ASD quality and currently under-sensitive to execution errors with subtle visual differences.


## Summary of Best-Performing Configurations

For quick reference, the strongest reported configurations per task on IndustReal are:

- AR (action recognition):
  - MViTv2 transformer, Kinetics-pretrained, fine-tuned on IndustReal, RGB + visible light + stereo ensemble.
  - Metrics: Top-1 accuracy 66.45%, Top-5 accuracy 88.43%.

- ASD (assembly state detection):
  - YOLOv8-m detector, COCO-pretrained, trained on combined real IndustReal ASD data and synthetic Unity-generated images.
  - Metrics: mAP (annotated frames) 0.838, mAP (entire videos) 0.641.

- PSR (procedure step recognition):
  - Baseline B3 (confidence accumulation + procedural constraints), with best ASD model (Real + synthetic training).
  - Metrics (all recordings): POS 0.797, F1 0.883, average delay τ 22.4 seconds.
  - Metrics (recordings with errors): POS 0.731, F1 0.816, average delay τ 20.4 seconds.

Together, these metrics characterize the current state-of-the-art on IndustReal, illustrating strong performance on clean procedural sequences but significant room for improvement in handling execution errors and visually subtle wrong configurations.