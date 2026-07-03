# 110 — Master Overview: Complete Consultation Package (Files 89-109)

**Date:** 2026-07-03
**Purpose:** Navigation guide for all 21 files in the RF4 consultation package.

---

## File Inventory (21 files, 89—109)

```
Layer 1: ORIENTATION
89  — Master Index (file map for all 21 files)  
108 — Complete Overview (files 89-108)
110 — THIS FILE: Complete Overview (files 89-109)

Layer 2: PRE-CONSULTATION STATE (what was wrong)
90  — Training History & Loss Tables
91  — Architecture Deep-Dive (with corrections)
92  — Loss Analysis & Multi-Task Balancing
93  — Validation Metrics & Gate Criteria
94  — Fix History & AAIML Strategy
95  — 50 Original Questions for Opus

Layer 3: OPUS ANSWERS (the fixes and answers)
96  — Round 1-2: F1-F16, critical seq-batch wipe fix
102 — Round 5: F17-F21, GPU crisis playbook, combined unit bug
109 — Round 6: F22-F22b, PSR eval bug fixed, 20 answers

Layer 4: POST-FIX STATUS (where we are now)
97  — Status after F1-F16 (epochs 2-4)
98  — Per-Head Analysis (post-F1-F16)
99  — AAIML Viability (pre-epoch 5 val)
100 — 20 Questions (pre-epoch 5 answer)
101 — Overview (files 89-101)
103 — Latest Status: RF4 PASSED, combined=0.241
104 — Per-Head with epoch 5 metrics
105 — PSR Deep Dive & Eval Bug
106 — AAIML Viability Update (65-80%)
107 — 20 New Questions post-epoch 5
```

---

## Reading Order by Role

**New to the project:** 89 → 96 → 110 → 103 → 104 → 109
**Gate decisions:** 103 (metrics) → 104 (per-head) → 106 (AAIML)
**PSR debugging:** 105 (PSR deep dive) → 109 (Q1-Q4, F22 fix)
**Paper writing:** 106 (AAIML) → 94 (ablation plan) → 96 (strategy)

---

## All Fixes (28 total)

**F1-F16** (doc 96): F1 (seq-batch grad wipe), F2 (KENDALL logging), F3/F3b (PSR loss fixes), F4/F4b (LR peak), F5 (grad centralization off), F6 (BF16), F7 (PSR seq 4), F8 (FOCAL_ALPHA 0.50), F9 (ACT_RAMP 3), F10 (clip 5.0), F11 (GATE 250), F12 (cosine probe), F13 (probe parity), F14/F14b (Kendall wd, pose reset), F15 (env override), F16 (ablation presets)

**F17-F21** (doc 102): F17 (data __init__), F18 (double-ramp), F19 (eff pose), F20 (combined_v2), F21 (auto LR)

**F22-F22b** (doc 109): F22 (PSR grouping misalignment), F22b (MonotonicDecoder dim collapse)

**Stability:** heartbeat race fix, VAL_EVERY_N_STEPS=0, EVAL_MAX_BATCHES=250, CUDNN_BENCHMARK=False

## Key Metrics (epoch 5 val)

| Metric | Value | Status |
|---|---|---|
| RF4 Gate | **PASSED** ✅ | combined=0.241 > 0.0 |
| det_mAP50_pc | **0.339** | Near RF10 floor |
| act_macro_f1 | **0.097** | Recovered from 0.006 |
| pred_distinct | **48/69** | Was 5 at epoch 2 |
| pose fwd MAE | **8.92°** | SOTA-competitive |
| psr comp acc | **0.554** | Above chance (was 0.291) |
| psr transition F1 | **Now measurable** | F22 fix — epoch 8 val first read |

## Current Training

PID 2988577 — epoch 6, 0 errors, all 28 fixes in merged main.
