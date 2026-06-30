# AAIML 2027 — Section-by-Section Guide (Post-Review Revision)

---

## Title

**POPW: Multi-Task Deep Learning for Consumer-GPU Assembly Verification with Cross-Task FiLM Conditioning**

No blockchain, no ethics in the title. Lead with the architecture contribution.

---

## Abstract (150-200 words)

**Structure**: Problem (implicit) → architecture → key result (Δ = −0.03) → efficiency (93 GFLOPs, 53M params) → pilot → impact.

**Key changes from v1**: No blockchain in abstract. No ethics framework in abstract. Activity recognition 14× baseline removed (too defensive). Lead with multi-task claim.

---

## 1. Introduction

**Purpose**: State the problem (3-5 separate models, $12K-$55K), the waste (285+ GFLOPs), and the need for MTL. Present POPW as the answer.

**Contributions (4 items)**:
1. Architecture: 5 tasks, 1 backbone, $299 GPU, Δ = −0.03 interference
2. Controlled ablation (equal gradient updates, not confounded by underfitting)
3. FiLM ablation (p = 0.032, intermediate ablations for PoseFiLM vs HeadPoseFiLM)
4. Factory pilot (N=20, zero opt-outs)

**Key changes from v1**: OpenAI "first to demonstrate" claim → "extending prior work from 2 to 5 tasks." Remove blockchain and ethics from contribution list (now in limitations). Add naive joint training collapse baseline.

---

## 2. Related Work

### 2.1 Multi-Task Learning
Cite EgoPack, CAGrad, PCGrad, Kendall 2018. Acknowledge Li et al. (2-task MTL on consumer GPU). Position POPW as extending from 2 to 5 tasks with controlled ablation.

### 2.2 Assembly Understanding
IndustReal, IKEA ASM, Assembly101, STORM-PSR. All prior work: separate models.

### 2.3 Feature Conditioning
FiLM vs cross-attention: O(N²) vs O(C). Ablation B is the key comparison.

### Competitor Table
Keep Table 1. Add "MTL Baseline?" column: Li et al. = Yes, ViMAT/IFAS = No.

---

## 3. System Architecture

### 3.1 Backbone
ConvNeXt-Tiny + FPN, 53M params, 93 GFLOPs, 4.8 FPS, 24-144 frames per assembly step.

### 3.2 Five Task Heads
One paragraph summary. Key numbers only.

### 3.3 Two-Stage FiLM
γ = 1 + tanh(Wγ z_pose) ∈ (0,2). Stop-gradient on HeadPoseFiLM. Active for activity and PSR.

### 3.4 Training Protocol
Kendall weighting with learned σ. Staged RF1-RF4. Naive joint training baseline: collapsed (act_top1 = 2.1%).

---

## 4. Empirical Results

### 4.1 Primary Results
Table 2: detection (0.34 pc [0.31, 0.37]), head pose (9.1°), activity (18.3%/41.2%), efficiency (53M/93/4.8).

Table 3: SOTA comparison — 90.3M params vs 53.0M, 361 GFLOPs vs 93 GFLOPs.

### 4.2 Controlled Ablation A (Section 4.3 in paper)
**Key methodological contribution**: Equal gradient updates. Both multi and single-task get 30 epochs × detection-only updates. Δ = −0.03 is structural interference, not underfitting.

Also report: naive joint training (all tasks from epoch 1) = collapsed (act_top1 = 2.1%).

### 4.3 Ablation B: FiLM
Full: 18.3%, No FiLM: 16.1% (p = 0.032). PoseFiLM only: 17.2%. HeadPoseFiLM only: 16.8%.

### 4.4 Detection Analysis
24×24 confusion matrix. 70% 1-bit-adjacent. At threshold 0.7, FPR = 0.04.

---

## 5. Factory Pilot
N=20, 2 weeks, dimsum. Tables with effect sizes (Cohen's d = 0.51 for NASA-TLX). Acknowledge Bonferroni: p = 0.04 not significant at corrected α = 0.0125.

---

## 6. Blockchain Micropayments
One page max. 537ms devnet latency. Explicitly acknowledge: oracle problem, devnet vs mainnet, no adversarial analysis. Frame as "feasibility demonstration," not "security guarantee."

---

## 7. Discussion
Honest limitations: single dataset, single seed, blockchain gaps, small pilot.

---

## 8. References
18 citations. All IEEE/Springer/ACM venues.
