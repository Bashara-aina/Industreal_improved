# Nardon et al. arXiv:2506.15285 -- Differentiation Note

## Paper Identity

- **Title:** AI-driven visual monitoring of industrial assembly tasks
- **Authors:** Mattia Nardon, Stefano Messelodi, Antonio Granata, Fabio Poiesi, Alberto Danese, Davide Boscaini (Fondazione Bruno Kessler, Trento, Italy; Meccanica del Sarca s.p.a.)
- **Date:** Submitted 2025-06-18, updated 2025-07-14
- **Keywords:** Visual monitoring, Assembly task, Object detection

## What the Paper Actually Does (verified from full-text reading)

ViMAT is an object-detection + probabilistic state-tracking system for monitoring industrial assembly. It uses:
- **YOLOv8-X** object detector trained on synthetic data from digital twins (BlenderProc rendering)
- **Multi-view fusion** across 3 RGB-D cameras to detect assembly components (LEGO bricks, hydraulic press mold parts)
- **Viterbi-based probabilistic reasoner** that infers which assembly step is being performed by tracking which components are present/absent in predefined tray regions
- Validation on two scenarios: LEGO component replacement and hydraulic press mold reconfiguration

**No head-pose estimation, no body-pose estimation, no 6-DoF anything.** The perception module detects objects (assembly components), not people. The future work section explicitly states: *"As future work, we aim to incorporate hand pose estimation for finer action recognition"* -- confirming they do no pose estimation of any kind.

## Conflict Resolution

- **A19 assessment (LOW threat, detection + state tracking):** CORRECT. Nardon is single-task object detection with probabilistic state inference on custom data (LEGO, hydraulic press molds).
- **A9 assessment (MODERATE threat, hybrid CNN-attention head pose estimator with 6 DoF):** INCORRECT. There is no head-pose estimation, no 6-DoF anything, and no CNN-attention hybrid architecture in this paper. The ViMAT perception module uses YOLOv8-X (standard CNN detector), not a hybrid CNN-attention head. The confusion likely arose from skimming-related-work references to pose estimators (MegaPose [22], SAM-6D [23]) that are cited as *alternatives*, not as ViMAT's own method.

## Task Scope vs. Ours

| Dimension | Nardon et al. (ViMAT) | This work |
|-----------|----------------------|-----------|
| Tasks | Single: object detection | 4-task MTL: detection, activity recognition, head-pose regression, procedure-step recognition |
| Data | Custom (LEGO, hydraulic press) | IndustReal (real factory assembly) |
| Detection target | Assembly components only | Workers + components |
| Pose estimation | None | Head-pose (6-DoF, MediaPipe+PnP baseline, supervised regression) |
| Procedure tracking | State-transition+Viterbi on component states | Multilabel PSR on procedure steps |
| Multi-task | No | Yes (shared backbone, 4 heads) |
| Training data | Synthetic only (digital twin renders) | Real-world video annotations |

## Differentiation Paragraph (Paper-Ready)

> "Nardon et al. [arXiv:2506.15285] recently proposed ViMAT, a visual monitoring system for industrial assembly that combines YOLOv8-X object detection with a Viterbi-based probabilistic reasoner to track assembly state transitions from multi-view video. ViMAT operates on custom LEGO and hydraulic press datasets, training exclusively on synthetic renderings from digital twins, and its perception module targets assembly components rather than human pose or action. In contrast, our work addresses four complementary tasks on the IndustReal benchmark simultaneously: worker detection, activity classification, head-pose regression, and procedure-step recognition, within a single multi-task learning framework. To the best of our knowledge, this constitutes the first head-pose baseline on IndustReal and the first 4-task multi-task formulation on industrial assembly data."

## First-Head-Pose Claim Status: CONFIRMED

Nardon et al. does zero pose estimation of any kind (head, hand, or body). Therefore our claim of being the "first head-pose baseline on IndustReal" **stands unqualified** with respect to this paper. No wording adjustment needed for Nardon.
