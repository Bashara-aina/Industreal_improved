## Plan: POPW & IndustReal Model Improvements
Date: 2026-04-23
Type: FEATURE

## Context Gathered
- Both models use ResNet-50-FPN backbone with frozen BN (pretrained ImageNet)
- POPW (IKEA ASM): DetectionHead, PoseHead (heatmap+Wing Loss), ActivityHead (GAP+C5+P4)
- IndustReal: DetectionHead, HeadPoseHead (9-DoF), ActivityHead, PSRHead (multi-label)
- Target: Beat IKEA ASM benchmarks (Activity Top-1 >64.15%, mcAP csv >84.47%, PCK@10px >64.3%, AP@0.5 >85.3%)
- Target: Beat IndustReal benchmarks (Activity Top-1 >66.45%, ASD mAP@0.5 >83.8%, PSR F1 >0.901)
- Constraint: <49M params, >291 FPS (PTMA baseline)
- GPU: RTX 3060 12GB, Python 3.11+, AMP

## Key Improvements (from roadmap)
HIGH (P1, P2, P3, P9):
- P1: ConvNeXt-Tiny backbone (replaces ResNet-50) — better efficiency/accuracy tradeoff
- P2: GCN on skeleton topology — POPW only (has 17 keypoints)
- P3: Two-level temporal bank T=8 + T=32 — both models (temporal modeling)
- P9: GRU-based TMA cell (PTMA-inspired) — temporal masked attention

## Risk Assessment
- ConvNeXt-Tiny may have different channel counts than ResNet — need to verify FPN input channels
- GCN requires skeleton adjacency matrix — need to verify POPW has valid keypoint topology
- Temporal bank adds sequence processing — need backward compatibility with existing dataset loaders
- TMA cell adds complexity — need careful integration with activity head

## Approach
Batch 1: Create improved folder structure + ConvNeXt-Tiny backbone + OKS Loss (P1, P4)
- Copy existing code to new folders
- Replace ResNet-50 with ConvNeXt-Tiny (fewer params, better efficiency)
- Replace Wing Loss with OKS Loss for pose (better for keypoint detection)
- Create config_improved.py with new settings

Note: Deep analysis file (swarm-2026-04-23-popw-deep-analysis.md) not found — using roadmap items directly.
