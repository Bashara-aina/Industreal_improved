# AHFE 2026 — Debate Synthesis & Operational Memo to the AI Writing Team

## After 5-layer iterative analysis (8 agents, 2 rounds of debate + cross-verification)

---

## 1. THE CORE PROBLEM

The current AHFE paper has three structural weaknesses that, if unfixed, will result in rejection:

**(A) The ethical framework is the primary contribution** (per EIC track), but the paper reads like a technical system paper with ethics bolted on. The EIC chair, Andreas Wolkenstein (LMU Munich), expects substantive engagement with ethical theory — not just IEEE 7005 checkboxes.

**(B) The technical comparisons are incomplete** — 14+ required baselines are missing, including the most critical ones (STORM-PSR, EgoPack, ConsMTL). Reviewers will flag this as insufficient literature awareness.

**(C) One citation is hallucinated** — "Parker et al. CHI 2017" does not exist. This is a rejection risk if caught during review.

---

## 2. WHAT TO COMPARE AGAINST (THE BENCHMARKS)

After cross-verifying all agents' findings, here is THE definitive list:

### Must-report benchmarks (in priority order)

| Benchmark | Key Metrics | SOTA to beat | Why it matters for AHFE |
|-----------|------------|--------------|-------------------------|
| **IndustReal PSR** | POS, F1, tau (delay) | STORM-PSR: 0.812/0.901/15.5s | Your primary dataset — MUST match protocol |
| **MECCANO** | Top-1, Top-5 action acc | 52.82% (UCF 2023 challenge) | Most directly comparable industrial-like dataset |
| **IKEA ASM** | Frame accuracy, macro-recall | 80.2% (PoseConv3D+Obj 2023) | Most cited assembly benchmark |
| **Assembly101** | Action rec Top-1, seg F1@10 | ~34% Top-1 | Largest assembly dataset (513h) — shows scale |
| **HA-ViD** | Action rec, seg metrics | NeurIPS 2023 baseline | Real industrial GAB |

### Critical — STORM-PSR is your direct PSR competitor

Published CVIU 2025. Uses same datasets (MECCANO, IndustReal). Must compare:
- POS: 0.812 (STORM-PSR) vs your result
- F1: 0.901 (STORM-PSR) vs your result
- tau: 15.5s (STORM-PSR) vs your result

---

## 3. THE 14 MISSING CITATIONS (verified real — all confirmed)

Every paper below was verified against official proceedings/DOI. None are hallucinated.

### Must-add to related work (ordered by impact)

| # | Paper | Venue | Why critical |
|---|-------|-------|-------------|
| 1 | **STORM-PSR** (Schoonbeek et al.) | CVIU 2025 | Direct PSR SOTA on your datasets |
| 2 | **EgoPack** (Peirone et al.) | CVPR 2024 | Closest egocentric MTL work |
| 3 | **ConsMTL** (Qin et al.) | CVPR 2025 | SOTA gradient conflict resolution |
| 4 | **UW-SO** (Kirchdorfer et al.) | IJCV 2025 | Fixes known Kendall UW failure |
| 5 | **CAGrad** (Liu et al.) | NeurIPS 2021 | Standard gradient-based MTL |
| 6 | **Nash-MTL** (Navon et al.) | ICML 2022 | Note: ICML not NeurIPS |
| 7 | **PCGrad** (Yu et al.) | NeurIPS 2020 | Foundational gradient surgery |
| 8 | **MTLoRA** (Agiza et al.) | CVPR 2024 | 3.6x fewer params challenges your approach |
| 9 | **PromptonomyViT** (Herzig et al.) | WACV 2024 | Alternative architectural choice |
| 10 | **InterroGate** (Bejnordi et al.) | BMVC 2024 | Automates your sharing pattern design |
| 11 | **ASDF** (Schieber et al.) | ISMAR 2024 | Shows pose-state co-dependency |
| 12 | **IndEgo dataset** (Chavan et al.) | NeurIPS 2025 | Largest industrial ego dataset |
| 13 | **IMPACT dataset** (Wen et al.) | arXiv 2026 | Most recent assembly benchmark |
| 14 | **Kendall UW** (Kendall et al.) | **CVPR 2018** — which you already use but DO NOT CITE |

---

## 4. THE CONSUMER GPU CLAIM — PRECISE LANGUAGE NEEDED

After cross-verification with benchmark agents:

**Do NOT say**: "first consumer-GPU multi-task assembly verification system"

Li et al. (Machines, 2025) already published a Swin Transformer multi-task defect detection system on GTX 3060 6GB (28.6 FPS, 99.21% accuracy). YOLOv8 on RTX 3060 is also well-established.

**Say instead**: "first single-model multi-task assembly verification system combining assembly state detection, body/head pose estimation, activity recognition, and procedure step recognition on a single consumer-grade GPU ($299 RTX 3060)"

Your unique claim is the **combination of tasks** in a **single model** for **assembly verification specifically**, not consumer GPU multi-task in general.

### Real RTX 3060 baselines to cite

| Source | Model | FPS | Notes |
|--------|-------|-----|-------|
| Nature Scientific Reports 2026 | YOLOv8 | 83.3 | Simple detection |
| YOLOv8 TensorRT public bench | YOLOv8m FP16 | 30.0 | Detection only |
| Li et al. 2025 (Machines) | Swin Transformer MSTUnet | 28.6 | Multi-task defect detection |
| **POPW** | ConvNeXt-T + 5 heads | **4.8** | 720x1280, 5 tasks |

Your 4.8 FPS is low because of high resolution and head count. Acknowledge this — frame as the cost of multi-task comprehensiveness.

---

## 5. ETHICAL FRAMEWORK — WHAT THE EIC TRACK ACTUALLY EXPECTS

After analyzing the EIC track chair's work and published AHFE proceedings:

### The EIC track is NOT for:
- Papers that mention "ethics" in passing
- Compliance checklists without philosophical grounding
- Technical system papers with a "potential ethical concerns" paragraph

### The EIC track IS for:
- Substantive engagement with ethical theory (Floridi, Foucault, virtue ethics)
- Real trade-offs: surveillance vs safety, transparency vs privacy, autonomy vs efficiency
- Actionable governance recommendations grounded in philosophy AND practice
- Worker-centered analysis (autonomy, dignity, agency, power asymmetries)

### Your ethical section needs:

1. **Theorem**: Blockchain-based worker monitoring creates a distributed morality problem (Floridi 2013, 2016) — no single agent is responsible when multiple agents (CV model, verification engine, smart contract, human manager) collectively determine payment.

2. **Antithesis**: Immutability (blockchain feature) vs GDPR right to erasure — how do workers delete their data from an immutable ledger? Address this directly; do not ignore it.

3. **Synthesis**: IEEE 7005-2021 provides the governance scaffolding to close the accountability gap, but only with specific design choices you define.

### Required ethics citations (verified real)

1. Wolkenstein (2024) — "Healthy Mistrust" — Cambridge Quarterly of Healthcare Ethics — YOUR TRACK CHAIR
2. Sebastian, Ehinger & Miller (2025) — "Ethics of CV for workplace surveillance" — AI and Ethics
3. Floridi (2013) — "Distributed morality in an information society" — Science and Engineering Ethics
4. Floridi (2016) — "Faultless responsibility" — Philosophical Transactions of the Royal Society A
5. Li & Wang (2026) — "Digital panopticism at construction sites" — AI and Ethics
6. Callari et al. (2024) — "Ethical framework for HRC manufacturing" — Technology in Society
7. Brintrup et al. (2025) — "Trustworthy AI in manufacturing" — Data-Centric Engineering
8. Milanez et al. (2025) — "Algorithmic management" OECD survey — OECD AI Papers No. 31
9. Sharif & Ghodoosi (2022) — "Ethics of Blockchain in Organizations" — Journal of Business Ethics
10. Alkhatib, Bernstein & Levi (2017) — REPLACES hallucinated "Parker" citation — CHI 2017

---

## 6. THE BLOCKCHAIN SECTION — CITATION VERIFICATION

All blockchain citations were verified:

| Citation | Status | Verdict |
|----------|--------|---------|
| **Konnex PoPW whitepaper** | REAL | konnex.world — live testnet, GitHub org, docs |
| **x402 protocol** (Reppel et al.) | REAL | Full spec at x402.org, 35M+ transactions |
| **Solana devnet x402** | REAL | Solana docs at solana.com/docs/payments |
| **DePIN tokenomics** (Alshater 2026) | REAL | Frontiers in Blockchain, DOI verified |
| **IEEE 7005-2021** | REAL | Active standard, IEEE published |
| **Koii Network / PoPW** | REAL | koii.network — live protocol |
| **WorkChain** | REAL | Solana protocol, hackathon top 10 |

---

## 7. WHAT TO DO DIFFERENTLY FOR THE AHFE PAPER

### Structural rewrite needed

Current structure: Technical system (60%) -> Blockchain (20%) -> Ethics (20%)

**Needed structure for EIC track**: Ethics framing (15%) -> System as enabler (25%) -> Technical results (15%) -> Blockchain ethics analysis (20%) -> IEEE 7005 governance (25%)

### Specific changes

1. **Open with worker problem, not model**: "A worker assembling furniture has no transparent record of their work quality or compensation basis. We address this through..."

2. **Move ethical framework to Section 3** (before results), not Section 6 (after results). Ethics should frame the technical contribution, not conclude it.

3. **Add limitations as ethical commitments**: "We do not evaluate on Ego4D" should become "Generalization across work environments is an active requirement for fairness (IEEE 7005 §8.2)..."

4. **Add worker agency subsection**: Can workers contest blockchain records? Can they access their own data? What is the appeals process? This is what the EIC track cares about.

5. **Cite the Optifye.ai incident** (Feb 2025, reported by OECD.AI): A YC startup that built similar CV worker monitoring was forced to delete its demo after public backlash. Show you have learned from this.

---

## 8. THE 5-LAYER DEBATE OUTPUT

| Layer | Agents | Key Finding |
|-------|--------|-------------|
| L1 (4 agents) | AHFE profile, IEEE review, benchmarks, human factors | 4 independent research directions |
| L2 (4 agents) | PoPW/blockchain, EIC track, consumer GPU, MTL baselines | Cross-verified every citation, found hallucinated one |
| L3 (this synthesis) | Cross-verify all claims | ENIGMA-360 misclassified, first-mover claim too broad |
| L4 | Recursive check of contradictions | All 14 MTL baselines verified real, Kendall UW failure mode confirmed |
| L5 | Actionable plan | What to change, in what order, with what citations |

**Bottom line**: The paper has strong foundations but needs (1) citation cleanup, (2) ethics restructuring, (3) benchmark expansion, and (4) claim narrowing. These are doable before the July 24 deadline.
