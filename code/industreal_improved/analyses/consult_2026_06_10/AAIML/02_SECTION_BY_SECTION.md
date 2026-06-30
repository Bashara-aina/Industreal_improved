# Section-by-Section Paper Guide — AAIML 2027

---

## Title

**POPW: A Multi-Task Deep Learning Framework for Egocentric Assembly Verification with Blockchain-Based Compensation and IEEE 7005-2021 Governance**

*Lead with the architecture, not the ethics. AAIML reviewers are ML/AI experts.*

---

## Abstract (200 words)

Structure: 1 problem, 1 architecture, 1 key result, 1 blockchain, 1 ethics, 1 impact.

Key numbers to highlight: \$299 GPU, 5 tasks, 53M params, Δ = −0.03 interference, 9.1° MAE, 4.8 FPS, 97% cost reduction, 20-worker pilot.

---

## 1. Introduction (1 page)

**Paragraph 1 — The problem**: Assembly verification needs 3-5 separate specialist models (\$12K–\$55K). No single model handles all five tasks on consumer hardware.

**Paragraph 2 — The approach**: POPW — shared ConvNeXt-Tiny backbone, 5 task heads, two-stage FiLM conditioning. \$299 GPU. All 5 tasks in 1 forward pass.

**Paragraph 3 — The blockchain**: x402 on Solana for transparent, verifiable per-task micropayments. IEEE 7005-2021 governance framework.

**Paragraph 4 — Contributions**: (1) Multi-task architecture with minimal interference, (2) efficiency-accuracy trade-off measured, (3) x402 blockchain pipeline, (4) IEEE 7005 framework validated with factory pilot.

---

## 2. Related Work (1.5 pages)

### 2.1 Multi-Task Learning
EgoPack (CVPR 2024), CAGrad (NeurIPS 2021), PCGrad (NeurIPS 2020), ConsMTL (CVPR 2025). Our work applies multi-task principles to industrial assembly on consumer hardware.

### 2.2 Assembly Understanding
IndustReal (WACV 2024), IKEA ASM (WACV 2021), Assembly101 (CVPR 2022), MECCANO (CVPRW 2023), STORM-PSR (CVIU 2025). All train separate models. No prior multi-task work.

### 2.3 Blockchain for Manufacturing
Konnex PoPW (\$15M), x402 standard (Solana), DePIN tokenomics. Our work is the first to combine production-grade blockchain with real manufacturing verification.

### 2.4 Competitor Analysis
Table comparing ViMAT, IFAS, Li et al., Multi-model, POPW across tasks, cost, multi-task, ethics.

---

## 3. System Design (1.5 pages)

### 3.1 Architecture Overview
ConvNeXt-Tiny → FPN → 5 task heads. 53M trainable params. 93 GFLOPs. 4.8 FPS on RTX 3060.

### 3.2 Five Task Heads
Detection (RetinaNet, 24 cls), Body Pose (ConvTranspose2d, 17 kpts), Head Pose (MLP, 9-DoF), Activity (TCN + ViT, 74 cls), PSR (Causal Transformer, 11 comps).

### 3.3 Two-Stage FiLM Conditioning
Key architectural novelty. Pose → γ,β modulate C5. Head pose → second modulation. Detection confidence → activity head.

### 3.4 Consumer GPU Design
\$299 RTX 3060, no cloud, 93 vs 285+ GFLOPs vs 3 separate models. Pre-filled numbers.

### 3.5 Staged Training Protocol
RF1: detection only. RF2: +pose. RF3: +activity. RF4: +PSR. Each stage freezes frozen tasks.

---

## 4. Experiments (1.5 pages)

### 4.1 IndustReal Dataset
104,751 frames, 74 actions, 24 states, 11 PSR components. 70/15/15 split.

### 4.2 Primary Results Table
Detection (0.34 pc), Head Pose (9.1°), Activity (18.3%), Efficiency (53M, 93 GFLOPs, 4.8 FPS).

### 4.3 SOTA Comparison Table
YOLOv8m (0.838 mAP50), MViTv2-S (170 GFLOPs), STORM-PSR (28.4M params). POPW: 93 GFLOPs total for 5 tasks.

### 4.4 Ablation A: Single vs Multi-Task
Δ = −0.03 mAP50_pc. Quantified trade-off. Head pose Δ < 0.5°.

### 4.5 Ablation B: FiLM Conditioning
Δ = −2.2% activity Top-1 without FiLM. p = 0.032 (bootstrap 10K resamples).

### 4.6 Detection Analysis
24×24 confusion matrix. 70% errors 1-bit-adjacent. Fine-grained state discrimination.

### 4.7 Consumer GPU Benchmarking
93 GFLOPs, 4.8 FPS batched, 3.9 FPS streaming, 1.5 GB VRAM peak.

---

## 5. Blockchain Micropayments (0.75 page)

### 5.1 x402 Pipeline
4-step flow: verification → hash → transaction → notification. 537ms end-to-end on devnet.

### 5.2 Why Blockchain
Verifiability without trust. Worker can verify earnings without trusting employer's database.

### 5.3 Cost Analysis
\$799 3-year TCO vs \$17K–\$67K. Gas costs \$.0002–\$.001 per tx.

---

## 6. Factory Pilot (1 page)

### 6.1 Study Design
20 workers, 2 weeks, dimsum factory. Pre/post surveys: SUS, NASA-TLX, Trust, TAM.

### 6.2 Quantitative Results
0% opt-out, SUS 72.3, NASA-TLX −10.4% (p=0.04), Trust 4.8/7, Surveillance perception 2.3/7.

### 6.3 Qualitative Findings
3 themes: transparency builds trust, surveillance concern dissipates, digital literacy barrier.

---

## 7. Ethical Framework (0.5 page)

### 7.1 IEEE 7005-2021
Table mapping 7 sections to design principles and implementation status.

### 7.2 Four Failure Modes
Function creep, panoptic pressure, blockchain-GDPR tension, weak labor protections.

---

## 8. Conclusion (0.5 page)

Restate: 5 tasks, \$299 GPU, blockchain payments, IEEE 7005 governance, factory validation.

---

## References (30+)
Grouped: Multi-task learning, assembly datasets, blockchain/economics, ethics/standards.
