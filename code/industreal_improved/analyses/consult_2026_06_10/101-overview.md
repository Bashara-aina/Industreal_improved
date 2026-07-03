# 101 — Master Overview: Files 89–100

**Date:** 2026-07-03
**Purpose:** Navigation guide for the complete RF4 consultation package. Each file is self-contained; this index explains what each contains, why it matters, and who should read it.

---

## Structure of the Package

The 12 files (89–100) form four layers:

**Layer 1 — Orientation (files 89, 101):**
- 89: Master index with TL;DR and verification checklist
- 101: This file — the overview you're reading

**Layer 2 — Current State & Architecture (files 90, 91, 92, 97):**
- 90: Training status & trajectory — loss tables across epochs 0→4, complete RF4 run history
- 91: Architecture deep-dive — backbone, 5 heads, Kendall, gradient flow, parameter counts
- 92: Loss analysis & multi-task balancing — per-head formulas, magnitudes, Critical Gaps section
- 97: Deep status update with ALL 24 fixes documented, probability assessment for RF4→RF10

**Layer 3 — Validation & Strategy (files 93, 94, 98, 99):**
- 93: Validation metrics history — only one successful val ever (epoch 2), step-val data, gate criteria
- 94: Fix history & paper strategy — 24 fixes, AAIML viability, ablation requirements, contingency plans
- 98: Head-by-head deep analysis — each head's trajectory, fix impact, expected convergence timeline
- 99: AAIML benchmarking report — gap analysis to publishable numbers, fallback tiers (A-E)

**Layer 4 — Questions for Opus (files 95, 100):**
- 95: 50 deep questions — answered by Opus in file 96
- 100: 20 new deep questions — GPU stability, detection, PSR/pose, balancing, infrastructure

**Layer 5 — Opus Answer (file 96):**
- 96: The complete consultation answer — 16 fixes (F1-F16), 7 headline questions answered, eval audit, paper sync, confidence statement

---

## Key Facts at a Glance

| File | Lines | Core Content |
|---|---|---|
| 89 | 70 | Master index, TL;DR, known discrepancies |
| 90 | 121 | Loss tables epochs 0→4, run history, critical concerns |
| 91 | 140 | Architecture: 24 det classes, ~69 act groups, gradient isolation analysis |
| 92 | 118 | Loss formulas, Kendall analysis, 8 critical gaps discovered post-agent-review |
| 93 | 143 | Only successful val (epoch 2: combined=0.168), activity ramp context, epoch 3/6 thresholds |
| 94 | 212 | All 24 fixes, AAIML strategy critique, Tier 1/2/3 contingencies, ablation requirements |
| 95 | 272 | 50 technical questions for Opus (answered in 96) |
| 96 | 521 | Opus answer: F1 (seq-batch wipe critical fix), F4 (LR peak), AAIML strategy, truth table |
| 97 | 226 | Current training state: 8h44m alive, ALL fixes verified, probability assessment |
| 98 | 161 | Per-head analysis: det recovering, activity collapsed, pose converged, PSR not started |
| 99 | 126 | AAIML viability: 40-60% main track, head pose is uncontested contribution |
| 100 | 228 | 20 new deep questions organized in 5 sections |

---

## Critical Path for Opus Review

**Read in this order:**
1. **89** (TL;DR — 2 min)
2. **96** (Opus answer — 15 min)
3. **97** (Current status with all fixes verified — 5 min)
4. **100** (20 new questions that arose after implementing F1-F16 — 10 min)

**Reference as needed:**
- 90 for specific loss numbers
- 98 for head-by-head trajectories
- 99 for paper viability thresholds
- 95 for the original 50 questions (already answered in 96)

---

## Breaking Changes Since File 96

File 96 (Opus answer) was written before the cudaErrorLaunchTimeout was fully diagnosed. Since then, we learned:
- The RTX 5060 Ti (Blackwell) + CUDA 13.0 + driver 595.71.05 combination has a severe cuDNN kernel timeout problem
- CUDNN_BENCHMARK=False is insufficient to prevent crashes
- The crash always occurs at epoch 5 batch ~100 when activity loss spikes after ramp completion
- Current mitigation: ALLOW_TF32=False, CUDNN_BENCHMARK=False, V8_API disabled, batch=4
- This is documented in files 97 (GPU section) and 100 (Q1-Q4)

The F1 fix (seq-batch grad wipe) described in file 96 is confirmed correct and active in all runs since.

---

## How to Use These Files

**If you are new to the project:**
Start with 89, then 97, then 96. Skip 90-95 unless you need specific loss numbers.

**If you need to understand the architecture:**
Read 91 for the architecture, 92 for the loss balancing, 98 for per-head analysis.

**If you're evaluating paper viability:**
Read 99 for the gap analysis, 94 for the ablation strategy, 98 for per-head risk.

**If you're diagnosing a problem:**
Check 97 for the fix list, 100 for specific questions to investigate, 93 for validation thresholds.

---

## Current Training State (as of file creation)

- PID 2916896: Resumed from epoch 5 checkpoint
- GPU: RTX 5060 Ti 16GB, batch=4, accum=4
- Last validation: epoch 2 — combined=0.168, det_mAP50_pc=0.133, pose_fwd=11.32°
- Critical risk: cudaErrorLaunchTimeout at epoch 5 batch ~100
- Expected next validation: epoch 5 (VAL_EVERY=3)
- All 24 fixes verified and active
