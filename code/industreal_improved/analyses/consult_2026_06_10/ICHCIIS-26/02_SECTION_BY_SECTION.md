# ICHCIIS-26 — Section-by-Section Paper Guide

---

## Title

**POPW: A Human-Centered Multi-Task Assembly Verification System with Blockchain Micropayments**

*Alternative: "POPW: A Consumer-GPU Framework for Fair Compensation in Assembly Manufacturing — Multi-Task Vision, Blockchain Micropayments, and Ethical Governance"*

Keep it HCI-focused. Lead with the human problem, not the technical architecture.

---

## Abstract (200-250 words)

**Structure:**
1. Problem sentence (fair compensation in manufacturing)
2. Solution sentence (POPW system)
3. Technical sentence (5 tasks, single GPU)
4. Implementation sentence (blockchain micropayments)
5. Ethics sentence (IEEE 7005-2021)
6. Impact sentence (democratization, SMEs)

**Draft:**
> Fair compensation in manual assembly manufacturing is undermined by expensive verification infrastructure ($10,000+ GPU clusters) and compensation models that either weaken the link between effort and reward (hourly wages) or incentivize speed over quality (piecework). We present POPW, a human-centered multi-task computer vision system that runs on a single $299 consumer GPU and simultaneously performs object state detection, worker pose estimation, gaze tracking, activity recognition, and procedure step verification — all in a single forward pass. Using a shared ConvNeXt-Tiny backbone with 53 million parameters, POPW replaces three to four separate specialist models with one efficient design achieving 4.8 FPS on commodity hardware. We extend this with an x402 blockchain micropayment pipeline on Solana, enabling automatic per-task compensation with sub-cent fees and sub-second finality. The core contribution is a comprehensive ethical framework organized around IEEE 7005-2021 (Transparent Employer Data Governance), addressing informed consent, data privacy, algorithmic fairness, and the critical distinction between worker surveillance and worker empowerment. The entire system runs on a $299 GPU, democratizing access to AI-assisted manufacturing for small and medium enterprises.

---

## 1. Introduction (1 page)

**Paragraph 1 — The Human Problem:**
- Manual assembly verification = human supervisors or expensive multi-model vision ($10K-$50K)
- Workers compensated by time (weak link between effort/reward) or piecework (speed over quality)
- Neither serves worker or business
- $25T physical work economy lacks verification infrastructure for automated fair compensation

**Paragraph 2 — The Technical Gap:**
- No single model handles detection + pose + activity + PSR simultaneously
- Existing deployments run 3-5 separate models = redundant compute, fragmented outputs
- Consumer GPU is the key enabler ($299 vs $10K+)

**Paragraph 3 — The Ethical Gap:**
- Even if technically possible, worker monitoring raises surveillance concerns
- IEEE 7005-2021 provides the governance standard
- Our contribution: technical system + blockchain payments + ethical framework = complete solution

**Paragraph 4 — Contributions:**
1. First consumer-GPU multi-task assembly verification (5 tasks, 1 forward pass)
2. Empirical results on IndustReal dataset
3. x402 blockchain payment pipeline on Solana devnet
4. IEEE 7005-2021 ethical framework for deployment governance

---

## 2. Background and Related Work (1.5 pages)

### 2.1 Computer Vision in Manufacturing (0.5 page)
- Single-task approaches (Papoutsakis et al. 2024, Luque et al. 2024)
- Multi-task gap — no unified system on consumer hardware
- Position POPW as the bridge

### 2.2 Assembly Understanding (0.5 page)
- Datasets: IndustReal (WACV 2024), IKEA ASM, Assembly101
- Methods: STORM-PSR (CVIU 2025), EgoPack (CVPR 2024)
- Competitor comparison table (ViMAT, IFAS, Li et al.)

### 2.3 Blockchain in Manufacturing (0.25 page)
- Konnex PoPW ($15M), Proof-of-Process, DePIN tokenomics
- x402 standard on Solana

### 2.4 Ethical AI at Work (0.25 page)
- IEEE 7005-2021
- Algorithmic management (OECD)
- EU AI Act high-risk workplace classification
- Sebastian et al. ethics of CV surveillance

---

## 3. System Design (1.5 pages)

### 3.1 Architecture Overview (0.5 page)
- ConvNeXt-Tiny backbone → FPN → 5 task heads
- 53M trainable parameters
- All in one forward pass
- **FIGURE 1**: Architecture diagram (essential)

### 3.2 Five Task Heads (0.5 page)
| Head | Input | Output | Loss |
|------|-------|--------|------|
| Detection (ASD) | P3-P7 | 24 classes, bounding boxes | Focal + GIoU |
| Body Pose | C5 features | 17 keypoints + confidence | Wing Loss |
| Head Pose | GAP(C4||C5) → MLP | 9-DoF (forward, up, position) | MSE |
| Activity | Feature Bank + TCN + ViT | 75 actions | CE + label smoothing |
| PSR | Multi-scale GAP → Transformer | 11 binary components | Binary Focal |

### 3.3 Cross-Task Conditioning (0.25 page)
- PoseFiLM and HeadPoseFiLM
- C5 modulated by pose keypoints → γ,β scaling
- Detection confidence signals fed to activity head

### 3.4 Consumer GPU Design (0.25 page)
- Single RTX 3060 (12 GB)
- 4.8 FPS at 720x1280 resolution
- $299 hardware cost vs $12,000-$55,000 for competitors

---

## 4. Results (1 page)

### 4.1 Primary Results Table
| Task | Metric | Value |
|------|--------|-------|
| ASD detection | Present-class mAP50 | 0.30-0.38 |
| Head pose | Forward angular MAE | 9.1° |
| Activity | Top-1 accuracy | 10-30% |
| Efficiency | Parameters | 53M |
| Efficiency | FPS | 4.8 |

### 4.2 Ablation A: Single-task vs Multi-task (the key contribution)
Show that multi-task does not catastrophically interfere with any single task.

### 4.3 Detection Confusion Matrix
24x24 matrix showing 70% of errors are 1-bit-Hamming-adjacent.

---

## 5. Blockchain Micropayments (1 page)

### 5.1 x402 Protocol on Solana
- Standard: solana.com/developers/templates/x402
- Implementation: Coinbase SDK, npm @x402-solana/core v0.3.0
- Per-task payments with sub-cent fees

### 5.2 Latency Measurement
- End-to-end: ~537ms (frame capture → wallet notification)
- Acceptable for per-task (5-30 second tasks)

### 5.3 Cost Analysis Table
| Item | POPW | Traditional |
|------|------|------------|
| GPU hardware | $299 | $12,000-$55,000 |
| Per-transaction fee | <$0.01 | N/A |
| 3-year TCO | $799-$1,500 | $12,000-$55,000+ |

---

## 6. Ethical Framework (1.5 pages) ← KEY SECTION

### 6.1 Surveillance vs Empowerment
- The critical distinction
- Jidoka Tech / Protex AI criteria
- Optifye.ai precedent (deleted after "sweatshop software" accusations)

### 6.2 IEEE 7005-2021 Governance
Table mapping each IEEE section to a design principle:
| IEEE 7005 Section | Design Principle | Implementation |
|-------------------|-----------------|----------------|
| Informed consent | Opt-in/opt-out | Supervisor sign-off alternative |
| Data governance | Local edge processing | No cloud, no face storage |
| Transparency | Worker dashboard | Real-time metrics visibility |
| Fairness | No speed incentive | Quality-weighted compensation |

### 6.3 Blockchain Ethics Tensions
- GDPR right to erasure vs blockchain immutability
- Distributed morality (Floridi) — who is responsible in multi-agent system?
- Worker agency — what margins of freedom remain?

---

## 7. Discussion and Limitations (0.5 page)

**Limitations (be honest — reviewers respect this):**
1. Single dataset (IndustReal only) — future work on IKEA ASM, IndEgo
2. Single GPU constraint shaped design choices
3. Single-seed results — multi-seed for camera-ready
4. No human subjects study yet
5. Detection accuracy on real-data-only

**Future work:**
- Synthetic pretraining for detection
- Multi-seed evaluation
- Human subjects research
- Additional datasets

---

## 8. Conclusion (0.5 page)

One paragraph restating the thesis: single-GPU multi-task assembly verification with blockchain micropayments is feasible, efficient, and can be deployed ethically under IEEE 7005-2021 governance.

---

## References (30+ citations)

Group by category:
1. HCI/HF: Papoutsakis AHFE 2024, Luque AHFE 2024, Omri AHFE 2024
2. CV/Assembly: IndustReal, STORM-PSR, IKEA ASM, Assembly101
3. Multi-task: EgoPack, ConsMTL, UW-SO, CAGrad
4. Blockchain: x402, Konnex, Solana
5. Ethics: IEEE 7005-2021, Sebastian 2025, Floridi, EU AI Act
