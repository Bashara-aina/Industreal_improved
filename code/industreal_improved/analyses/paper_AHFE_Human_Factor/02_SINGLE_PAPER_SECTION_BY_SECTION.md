# Plan 2: Single Paper — Section-by-Section Content Specification

> **Single paper, 10 pages maximum, EIC track**
> **Title:** POPW: A Consumer-GPU Multi-Task Assembly Verification System with Blockchain Micropayments — A Human-Centered Framework for Fair Compensation

---

## Section-by-Section Blueprint

### ABSTRACT (500 words, single paragraph)
See Plan 1 Section 1.2 for full text.

### 1. INTRODUCTION (1 page — 500 words)

**Opening hook (human problem):**
"Manual assembly verification in manufacturing relies on human supervisors or expensive multi-model vision systems. Workers are compensated by time or piecework — models that either weaken the link between effort and reward or incentivize speed over quality. Neither serves the worker or the business well."

**The three gaps:**
1. **Economic:** AI monitoring locked behind $10K-$50K GPU infrastructure
2. **Technical:** No single model handles detection + pose + activity + procedure
3. **Human:** Workers receive no real-time feedback; compensation disputes are common

**Our answer:** POPW — one model, one $299 GPU, five tasks, blockchain micropayments.

**Contributions (enumerated):**
1. First consumer-GPU multi-task assembly verification system (5 tasks, $299)
2. Empirical results: detection (0.30 present-class mAP), head pose (9.1 deg)
3. x402 blockchain payment pipeline with devnet latency measurements
4. Ethical framework per IEEE 7005-2021 (6 design principles)

### 2. BACKGROUND AND RELATED WORK (1.5 pages — 750 words)

**2.1 Computer Vision in Manufacturing**
Cite: Papoutsakis et al. (AHFE 2024), Luque et al. (AHFE 2024), Omri et al. (AHFE 2024). Prior CV work at AHFE focuses on single tasks (pose, PPE, ergonomics). None unifies multiple tasks or addresses fair compensation.

**2.2 Assembly Understanding Approaches**
TABLE 1: Competitor Analysis

| Approach | Hardware Cost | Tasks Covered | Worker Feedback | Setup Complexity |
|---|---|---|---|---|
| ViMAT (2026) | Multi-GPU | Detection only | None | High |
| IFAS (2026) | Single GPU | Detection only | None | Moderate |
| ENIGMA (2023) | Multi-GPU | Det+Act+Antic | None | High |
| STORM-PSR (2025) | Single GPU | PSR only | None | High |
| Multi-model ensemble | $10K-$50K | All (separate) | Fragmented | Very High |
| **POPW (this work)** | **$299 GPU** | **All 5 unified** | **Real-time** | **Low** |

**2.3 Blockchain for Physical Work**
Cite: Konnex PoPW ($15M, docs.konnex.world), PopChain (Proof-of-Process), Materialize (Proof-of-Make), DePIN tokenomics (Frontiers in Blockchain 2025), Blockchain-Embedded SLA for Assembly (MDPI 2026)

**2.4 Ethics of AI in Manufacturing**
Cite: IEEE 7005-2021 (standard for transparent employer data governance), Parker et al. (CHI 2017, piecework), Milanez et al. (OECD 2025, algorithmic management), EU AI Act (high-risk workplace AI), Privacy-preserving CV for Industry (AAAI 2026)

### 3. SYSTEM DESIGN (1.5 pages — 750 words)

**3.1 POPW Architecture (simplified for HF audience)**
ConvNeXt-Tiny backbone (53M params) with 5 output heads: assembly state detection, body pose, head pose, activity recognition, procedure step recognition. Single forward pass. Fits in 12 GB GPU memory. FIGURE 1.

**3.2 Training on Consumer Hardware**
Staged training: detection first (2 days), then pose+head pose (3 days), then activity (3 days). Total ~8 GPU-days on RTX 3060.

**3.3 Blockchain Payment Pipeline**
x402 protocol (HTTP 402 Payment Required) on Solana. Production infrastructure: official Solana Rust template, @x402-solana/core v0.3.0 (npm), Coinbase reference implementation. FIGURE 2: pipeline diagram.

### 4. EXPERIMENTS AND RESULTS (2 pages — 1000 words)

**4.1 Dataset: IndustReal**
207K egocentric frames, 74 actions, 24 assembly states, 11 PSR components.

**4.2 Primary Results**
TABLE 2: Results with Human Factors Meaning

| Task | Metric | Value | Human Factors Meaning |
|---|---|---|---|
| Assembly state detection (present-class) | mAP@0.5 | 0.30 | Coarse assembly verification viable |
| Head pose (forward) | Angular MAE | 9.1 deg | Gaze zone monitoring (replaces $5K-$30K eye tracking) |
| Activity recognition | Top-1 | [RF3 result] | Action identification |
| Parameters | 53.4M vs 75.4M | 29% reduction | Fewer resources, greener AI |
| GPU cost | $299 vs $10K+ | **97% reduction** | Democratized access |
| Blockchain latency | Frame to wallet | ~537 ms | Real-time payment feasible |

**4.3 Detection Confusion Matrix (FIGURE 3)**
24x24 heatmap. 70% of errors on 1-bit-Hamming-adjacent states = fine-grained discrimination, not object detection failure.

**4.4 Ablation: Single-Task vs Multi-Task**
TABLE 3: Detection mAP50_pc — single-task vs multi-task. Delta < 0.05 = no catastrophic interference.

**4.5 3-Year Cost Analysis**
TABLE 4: $799-$1,500 (POPW) vs $12,000-$55,000+ (traditional).

### 5. BLOCKCHAIN IMPLEMENTATION (1.5 pages — 750 words)

**5.1 x402 Protocol Integration**
We deploy the Solana x402 Rust template (Axum server, /verify and /settle endpoints). Payment flow: POPW detects event → formats x402 header → facilitator verifies → facilitator settles on devnet → worker wallet notified.

**5.2 End-to-End Latency**
TABLE 5: POPW inference (31ms) → Verification (1ms) → x402 verify (80ms) → x402 settle (400ms) → Total (~537ms). Measured over 100 devnet transactions.

### 6. ETHICAL ANALYSIS (2 pages — 1000 words) ★ CORE CONTRIBUTION

**6.1 Surveillance vs Empowerment**
The critical distinction: monitoring FOR the worker vs monitoring OF the worker. Our system gives workers transparent, non-repudiable proof of work quality and automated fair compensation. Reference: Konnex's "verified physical work" framing vs traditional "worker surveillance."

**6.2 IEEE 7005-2021 Compliance Framework**
TABLE 6: Six Design Principles

| Principle | Implementation | IEEE 7005 Reference |
|---|---|---|
| Local processing | All inference on edge GPU | Section 5.2 Data Governance |
| No face storage | Pose vectors only | Section 5.3 Privacy |
| Informed consent | Opt-in with explanation | Section 6.1 Transparency |
| Opt-out provision | Supervisor sign-off alternative | Section 6.2 Alternative Means |
| Appeals process | Human review of disputes | Section 7.1 Accountability |
| Fairness monitoring | Per-worker accuracy metrics | Section 8.2 Non-discrimination |

**6.3 Algorithmic Fairness**
False negatives underpay workers; false positives overpay. Both must be quantified. Confidence threshold tuning. Human-in-the-loop for edge cases. Model fairness across diverse workers.

**6.4 Economic Justice and EU AI Act Compliance**
Micropayments supplement (not replace) base wages. The EU AI Act classifies workplace AI as high-risk — requiring risk management, transparency, human oversight. The EU Platform Work Directive (transposed by Dec 2026) requires transparency about automated monitoring. POPW's design aligns with both.

### 7. DISCUSSION AND LIMITATIONS (1 page — 500 words)

**Deployment scenarios:** Worker training support, quality verification, fair compensation tracking.
**Limitations:** Single-dataset validation, single-GPU ceiling, moderate detection under occlusion, fixed activity taxonomy, blockchain devnet only, human subjects research pending.

### 8. CONCLUSION (0.5 page — 250 words)
5-task assembly verification on $299 GPU is feasible. Head pose enables attention monitoring at 9.1 degrees. x402 blockchain payments enable fair automated compensation. IEEE 7005-2021 ethics framework provides deployment guardrails. This technology can democratize AI-assisted manufacturing for small and medium enterprises.

### REFERENCES (45+ citations)

---

## Critical Writing Rules

1. **No CV jargon.** Write "assembly state detection" not "ASD mAP@0.5." Write "shared neural network" not "ConvNeXt-Tiny backbone with FiLM conditioning."
2. **Worker-centered language.** "Worker support" not "surveillance." "Fair compensation" not "automated payment." "Attention monitoring" not "gaze tracking."
3. **Every technical claim has a human factors meaning.** Table 2's third column ensures this.
4. **IEEE 7005-2021 is cited in every major section.** It's the ethical backbone.
5. **Honest limitations disarm reviewers.** Section 7 acknowledges every weakness before they can find it.
